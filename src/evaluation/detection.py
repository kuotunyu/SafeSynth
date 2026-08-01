"""Detection metrics core for Phase 2 (EVAL-05..EVAL-14).

Two things pycocotools decides by default that this project must decide itself:

  * The area ranges behind AP_small / AP_medium / AP_large. They are read from
    configs/evaluation.yaml and pushed into COCOeval.params, so a bucketing
    change is a config change and never a library-upgrade side effect.
  * The coordinate system. Boxes are annotated at 416 and trained at 640.
    Evaluating in 640-space inflates every area by (640/416)^2 ~= 2.37 and
    silently moves most `small` objects into `medium`, which turns the headline
    metric into a measurement of something else without raising (EVAL-07).

`src.training.metrics` owns the Trainer-side evaluator and is imported here, not
copied: `build_coco_ground_truth` and `predictions_to_coco` are reused verbatim
by callers, and `evaluate_detections` supplies the all-zero stats dict for the
no-detections case. The one thing it cannot do is accept explicit area ranges or
hand back the COCOeval object, which per-class and per-bucket reporting need, so
`_run_cocoeval` exists alongside it rather than in place of it.

Every numeric threshold in this module comes from configs/evaluation.yaml.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import io
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import yaml

from src.data.paths import load_project_paths
from src.training.metrics import COCO_STAT_NAMES, evaluate_detections

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "evaluation.yaml"

# COCOeval.summarize() looks its area buckets up BY LABEL ('all', 'small',
# 'medium', 'large'). These names are therefore an interface with pycocotools,
# not a naming preference, and the config must use exactly them.
ALL_LABEL = "all"
BUCKET_LABELS = ("small", "medium", "large")

# `head` is the bare-head class. `helmet` is a helmeted head, and the two are
# mutually exclusive per person (Spike H1 / ADR-007), so bare-head recall is
# recall of this single class and needs no spatial reasoning.
BARE_HEAD_CLASS = "head"

DETECTION_METRICS_FILENAME = "detection_metrics.csv"
DETECTION_METRIC_FIELDS = (
    "arm",
    "seed",
    "split",
    "metric",
    "value",
    "n_instances",
    "n_images",
    "ci_low",
    "ci_high",
    "notes",
)

# Which size bucket's instance count belongs next to which COCO stat (EVAL-08).
_STAT_BUCKET = {
    "map_small": "small",
    "map_medium": "medium",
    "map_large": "large",
    "mar_small": "small",
    "mar_medium": "medium",
    "mar_large": "large",
}

# COCO's own "this number is undefined" sentinel, returned by _summarize() when
# an area bucket or a class has no ground truth at all. Kept rather than
# replaced by 0.0, because "no instances" and "found none of them" are different
# statements and the report must not blur them.
UNDEFINED = -1.0


class EvaluationConfigError(RuntimeError):
    """Raised when configs/evaluation.yaml cannot drive COCOeval as written."""


class HardNegativeSubsetError(RuntimeError):
    """Raised when the hard-negative subset is not actually hard negatives."""


class UnknownCategoryError(RuntimeError):
    """Raised when a detection carries a category id the ground truth lacks."""


class BootstrapError(RuntimeError):
    """Raised when a bootstrap resample produces a value that cannot be pooled."""


class SplitLeakageError(RuntimeError):
    """Raised when Test images appear in a training image list (EVAL-14).

    Not named TestLeakageError: pytest collects anything called Test* as a test
    class and warns about the constructor.
    """


class EvaluatorMismatchError(RuntimeError):
    """Raised when two COCO evaluators disagree on the same input (EVAL-13)."""


@dataclass(frozen=True)
class BucketCounts:
    """Instance counts per size bucket, alongside the AP they belong to."""

    total: dict[str, int]
    by_class: dict[str, dict[str, int]]
    class_totals: dict[str, int]
    n_images: int


@dataclass(frozen=True)
class BareHeadRecall:
    """Recall of the bare-head class at the configured IoU (EVAL-05 #2)."""

    recall: float
    n_matched: int
    n_ground_truth: int
    iou_threshold: float
    score_threshold: float


@dataclass(frozen=True)
class HardNegativeReport:
    """False positives per image on images whose GT is empty (EVAL-05 #3)."""

    false_positives_per_image: float
    n_false_positives: int
    n_images: int
    score_threshold: float


@dataclass(frozen=True)
class BootstrapCI:
    """A percentile bootstrap interval resampled over test IMAGES (EVAL-09).

    `n_undefined` is how many resamples the metric did not exist on (a draw that
    happened to contain no instance of the class). Those draws are excluded from
    the percentile rather than pooled as -1.0, so the count has to travel with
    the interval: an interval over 600 of 1,000 resamples is a different
    statement from one over all 1,000, and the CSV must be able to say which.
    """

    point: float
    low: float
    high: float
    resamples: int
    confidence: float
    seed: int
    n_undefined: int


@dataclass(frozen=True)
class DetectionMetrics:
    """Everything EVAL-05..EVAL-08 asks for on one split of one run."""

    stats: dict[str, float]
    per_class_ap: dict[str, float]
    per_class_ap_small: dict[str, float]
    primary_map: float
    primary_map_small: float
    # Which classes the two `primary_*` numbers were averaged over. Carried on
    # the result because the CSV writer has to report the instance count of the
    # SAME classes the metric used, and it never sees the config (EVAL-08).
    primary_classes: tuple[str, ...]
    counts: BucketCounts
    bare_head: BareHeadRecall
    n_images: int
    n_detections: int


@dataclass(frozen=True)
class MetricRow:
    """One row of results/detection_metrics.csv: arm x seed x metric (EVAL-11).

    Long format on purpose (EVAL-12): every table in the report has to be
    re-aggregatable from this file, so nothing may be pre-summarised into wide
    columns that lose the per-arm, per-seed breakdown.
    """

    arm: str
    seed: int
    split: str
    metric: str
    value: float
    n_instances: int | None = None
    n_images: int | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    notes: str = ""


def load_evaluation_config(path: Path = EVALUATION_CONFIG_PATH) -> dict[str, Any]:
    """Load configs/evaluation.yaml with UTF-8 forced (K-10)."""

    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return config


def _resolve_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return load_evaluation_config() if config is None else config


# spec: EVAL-07
def area_ranges_from_config(
    config: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    """Build COCOeval's areaRng/areaRngLbl pair from the config.

    Returns ('all', span) first because COCOeval.summarize() indexes bucket 0 as
    'all'. The 'all' span is derived from the configured buckets rather than
    hard-coded, so widening `large` cannot leave `all` clipping objects out.

    Areas are in ORIGINAL 416x416 coordinates: the config is the only place the
    boundaries exist, and predictions are mapped back to annotation coordinates
    before they ever reach here.
    """

    config = _resolve_config(config)
    ranges = config["metrics"]["coco_area_ranges"]
    missing = sorted(set(BUCKET_LABELS) - set(ranges))
    unexpected = sorted(set(ranges) - set(BUCKET_LABELS))
    if missing or unexpected:
        raise EvaluationConfigError(
            "metrics.coco_area_ranges must define exactly "
            f"{list(BUCKET_LABELS)}; missing={missing}, unexpected={unexpected}. "
            "COCOeval.summarize() looks these buckets up by label, so a renamed "
            "bucket produces a silent NaN instead of an error."
        )

    buckets: list[tuple[str, tuple[float, float]]] = []
    for label in BUCKET_LABELS:
        low, high = (float(value) for value in ranges[label])
        if not low < high:
            raise EvaluationConfigError(
                f"metrics.coco_area_ranges.{label} = [{low}, {high}] is empty or inverted"
            )
        buckets.append((label, (low, high)))

    span = (min(low for _, (low, _) in buckets), max(high for _, (_, high) in buckets))
    return ((ALL_LABEL, span), *buckets)


def _category_name_by_id(ground_truth: Mapping[str, Any]) -> dict[int, str]:
    return {int(c["id"]): str(c["name"]) for c in ground_truth["categories"]}


def _detection_class_name(names: Mapping[int, str], detection: Mapping[str, Any]) -> str:
    """Class name of a detection, refusing ids the ground truth does not define.

    `names.get(...)` returns None for an unknown id, which silently drops the
    detection. A dropped detection is a false positive that never gets counted,
    and the hard-negative number exists precisely to count false positives, so
    the failure mode is a metric that improves when the prediction file is more
    broken.
    """

    category_id = int(detection["category_id"])
    if category_id not in names:
        raise UnknownCategoryError(
            f"Detection on image {int(detection['image_id'])} has category_id "
            f"{category_id}, which the ground truth does not define "
            f"(known ids: {sorted(names)}). Refusing to drop it silently."
        )
    return names[category_id]


def _category_id_for_name(ground_truth: Mapping[str, Any], name: str) -> int:
    for category_id, category_name in _category_name_by_id(ground_truth).items():
        if category_name == name:
            return category_id
    raise EvaluationConfigError(
        f"Class {name!r} is not in the ground truth categories "
        f"{sorted(_category_name_by_id(ground_truth).values())}"
    )


def _iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    """IoU of two COCO xywh boxes."""

    ax, ay, aw, ah = (float(v) for v in first)
    bx, by, bw, bh = (float(v) for v in second)
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    # A positive intersection implies both boxes have positive area, so the
    # union cannot be zero here and needs no guard.
    intersection = (right - left) * (bottom - top)
    return float(intersection / (aw * ah + bw * bh - intersection))


def _run_cocoeval(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[str, tuple[float, float]]],
):
    """COCOeval with EXPLICIT area ranges, returning the evaluator itself.

    Deliberately not `src.training.metrics.evaluate_detections`: that function
    accepts pycocotools' built-in [0, 32^2, 96^2] buckets and throws the
    evaluator away, while per-class AP and per-bucket AP both need the
    accumulated precision array.
    """

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        # deepcopy, not dict(): COCOeval._prepare() writes an `ignore` key into
        # every ANNOTATION dict it is handed, and a shallow copy of the outer
        # dict shares those annotation dicts with the caller. Evaluating would
        # then mutate the caller's ground truth in place - and the bootstrap
        # hands the same annotations back a thousand times.
        coco_gt.dataset = copy.deepcopy(dict(ground_truth))
        coco_gt.createIndex()
        # Detections only need the shallow copy: loadRes writes `area`, `id`,
        # `iscrowd` and `segmentation` as top-level keys and never touches the
        # bbox list, so the caller's dicts are already protected.
        coco_dt = coco_gt.loadRes([dict(d) for d in detections])
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        # Must be set BEFORE evaluate(): evaluateImg reads areaRng to decide
        # which ground truth is ignored in each bucket.
        evaluator.params.areaRng = [[low, high] for _, (low, high) in buckets]
        evaluator.params.areaRngLbl = [label for label, _ in buckets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


def _average_precision(evaluator, *, class_index: int, area_index: int) -> float:
    """AP @ IoU 0.50:0.95 for one class in one area bucket.

    Reads the accumulated precision array [T, R, K, A, M] the same way
    COCOeval.summarize() does, with the same -1 masking, so a per-class AP and
    the summarize() mAP cannot drift apart. maxDets index -1 is the largest
    (100), matching every stat summarize() reports.
    """

    precision = evaluator.eval["precision"][:, :, class_index, area_index, -1]
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else UNDEFINED


def _mean_defined(values: Iterable[float]) -> float:
    defined = [value for value in values if value > UNDEFINED]
    return float(np.mean(defined)) if defined else UNDEFINED


# spec: EVAL-08
def instance_counts_by_bucket(
    ground_truth: Mapping[str, Any], *, config: Mapping[str, Any] | None = None
) -> BucketCounts:
    """Count ground-truth instances per size bucket, and per class per bucket.

    An AP over nine instances and an AP over nine hundred read identically, so
    the count travels with the number everywhere it is reported.

    The bucket test is COCOeval's own inclusive `low <= area <= high`, which
    means an object whose area is exactly on a boundary is counted in BOTH
    neighbouring buckets. That is intentional: the count has to be the
    denominator the AP actually used, not a tidier partition that disagrees with
    it. Crowd annotations are excluded, again because COCOeval ignores them.
    """

    config = _resolve_config(config)
    buckets = area_ranges_from_config(config)
    names = _category_name_by_id(ground_truth)

    total = {label: 0 for label, _ in buckets}
    by_class = {name: {label: 0 for label, _ in buckets} for name in names.values()}
    class_totals = dict.fromkeys(names.values(), 0)

    for annotation in ground_truth["annotations"]:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        name = names[int(annotation["category_id"])]
        bbox = annotation["bbox"]
        area = float(annotation.get("area", float(bbox[2]) * float(bbox[3])))
        class_totals[name] += 1
        for label, (low, high) in buckets:
            if low <= area <= high:
                total[label] += 1
                by_class[name][label] += 1

    return BucketCounts(
        total=total,
        by_class=by_class,
        class_totals=class_totals,
        n_images=len(ground_truth["images"]),
    )


# spec: EVAL-05
def bare_head_recall(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
    score_threshold: float = 0.0,
    class_name: str = BARE_HEAD_CLASS,
) -> BareHeadRecall:
    """Recall of the bare-head class at metrics.bare_head_recall_iou.

    Computed directly by greedy IoU matching rather than read off a COCOeval
    stat slot: `mar_*` is an average over ten IoU thresholds and every class, so
    no slot in that vector means "recall of `head` at IoU 0.50". Detections are
    matched highest-score-first, one ground-truth box each, so a pile of
    duplicate boxes on one head cannot inflate recall.

    `score_threshold` defaults to 0 because recall as a headline number is taken
    over all detections; pass the compliance operating point to read recall at
    the point the demo actually runs at.
    """

    config = _resolve_config(config)
    iou_threshold = float(config["metrics"]["bare_head_recall_iou"])
    category_id = _category_id_for_name(ground_truth, class_name)

    ground_truth_boxes: dict[int, list[Sequence[float]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        if int(annotation["category_id"]) != category_id:
            continue
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        ground_truth_boxes[int(annotation["image_id"])].append(annotation["bbox"])

    candidates = [
        detection
        for detection in detections
        if int(detection["category_id"]) == category_id
        and float(detection["score"]) >= score_threshold
    ]
    # Stable sort: ties keep input order, so the result is reproducible.
    candidates.sort(key=lambda detection: -float(detection["score"]))

    matched: set[tuple[int, int]] = set()
    for detection in candidates:
        image_id = int(detection["image_id"])
        best_index, best_iou = -1, 0.0
        for index, box in enumerate(ground_truth_boxes.get(image_id, [])):
            if (image_id, index) in matched:
                continue
            iou = _iou_xywh(detection["bbox"], box)
            if iou > best_iou:
                best_index, best_iou = index, iou
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add((image_id, best_index))

    n_ground_truth = sum(len(boxes) for boxes in ground_truth_boxes.values())
    recall = len(matched) / n_ground_truth if n_ground_truth else 0.0
    return BareHeadRecall(
        recall=float(recall),
        n_matched=len(matched),
        n_ground_truth=n_ground_truth,
        iou_threshold=iou_threshold,
        score_threshold=float(score_threshold),
    )


# spec: EVAL-05
def hard_negative_false_positives_per_image(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    image_ids: Iterable[int],
    *,
    config: Mapping[str, Any] | None = None,
    score_threshold: float | None = None,
) -> HardNegativeReport:
    """Mean detections above threshold on images with no primary-class GT.

    A hard negative can never contribute recall; it can only move precision, so
    diluting it into mAP over the whole Test set hides exactly the effect it was
    generated to produce. It gets its own number.

    The subset is validated rather than filtered: an image with primary-class
    ground truth in it means the caller built the wrong subset, and silently
    dropping it would report a number over a different set of images than the
    caller thinks. A repeated image id is the same class of mistake and is
    rejected for the same reason.
    """

    config = _resolve_config(config)
    if score_threshold is None:
        # The operating point, not an mAP threshold: this is a count of boxes a
        # deployed detector would actually draw (EVAL-04).
        score_threshold = float(config["compliance"]["score_threshold"])
    primary = set(config["metrics"]["primary_classes"])

    subset = [int(image_id) for image_id in image_ids]
    if not subset:
        raise HardNegativeSubsetError(
            "The hard-negative subset is empty; there is no per-image mean to report"
        )

    repeated = sorted(image_id for image_id, n in Counter(subset).items() if n > 1)
    if repeated:
        raise HardNegativeSubsetError(
            f"Image ids {repeated} appear more than once in the hard-negative subset. "
            "The numerator counts each image's detections once and the denominator "
            "would count the image twice, so a duplicate silently halves the reported "
            "rate. Deduplicate the subset before calling."
        )

    known = {int(image["id"]) for image in ground_truth["images"]}
    unknown = sorted(set(subset) - known)
    if unknown:
        raise HardNegativeSubsetError(
            f"Image ids {unknown} are not in the ground truth"
        )

    names = _category_name_by_id(ground_truth)
    subset_ids = set(subset)
    annotated = sorted(
        {
            int(annotation["image_id"])
            for annotation in ground_truth["annotations"]
            if int(annotation["image_id"]) in subset_ids
            and names[int(annotation["category_id"])] in primary
            and int(annotation.get("iscrowd", 0)) == 0
        }
    )
    if annotated:
        raise HardNegativeSubsetError(
            f"Images {annotated} carry {sorted(primary)} ground truth, so they are not "
            "hard negatives; false positives per image would be meaningless on them"
        )

    # The class lookup is only reached for detections that land on this subset:
    # a detection elsewhere in the file is out of scope for this number and is
    # not this function's to validate.
    n_false_positives = sum(
        1
        for detection in detections
        if int(detection["image_id"]) in subset_ids
        and _detection_class_name(names, detection) in primary
        and float(detection["score"]) >= score_threshold
    )
    return HardNegativeReport(
        false_positives_per_image=n_false_positives / len(subset),
        n_false_positives=n_false_positives,
        n_images=len(subset),
        score_threshold=float(score_threshold),
    )


def _resample_by_image(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    sampled_ids: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rebuild a GT/detection pair from images drawn WITH REPLACEMENT.

    A bootstrap draw repeats images, and COCO requires unique image ids, so each
    draw gets a fresh id and its own copies of the annotations and detections.
    Reusing the original ids would make pycocotools silently keep one copy.
    """

    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    detections_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for detection in detections:
        detections_by_image[int(detection["image_id"])].append(detection)
    image_by_id = {int(image["id"]): image for image in ground_truth["images"]}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    resampled_detections: list[dict[str, Any]] = []
    annotation_id = 1
    for new_id, old_id in enumerate(sampled_ids, start=1):
        image = dict(image_by_id[int(old_id)])
        image["id"] = new_id
        images.append(image)
        for annotation in annotations_by_image[int(old_id)]:
            copy = dict(annotation)
            copy["image_id"] = new_id
            copy["id"] = annotation_id
            annotation_id += 1
            annotations.append(copy)
        for detection in detections_by_image[int(old_id)]:
            copy = dict(detection)
            copy["image_id"] = new_id
            resampled_detections.append(copy)

    return (
        {
            "images": images,
            "annotations": annotations,
            "categories": list(ground_truth["categories"]),
        },
        resampled_detections,
    )


def _reject_unpoolable(value: float, what: str) -> None:
    """Refuse a value that cannot sit inside a percentile interval.

    Two ways a metric fails to be a number here, and only one of them looks
    wrong: NaN/inf are obviously unusable, while UNDEFINED is -1.0, a perfectly
    finite float that sorts below every real AP and would drag the lower bound
    down as though the model had scored -1.
    """

    if not np.isfinite(value):
        raise BootstrapError(
            f"{what} is {value!r}; a non-finite value cannot be pooled into a "
            "percentile interval"
        )
    if value <= UNDEFINED:
        raise BootstrapError(
            f"{what} is the UNDEFINED sentinel ({value!r}): the metric does not exist "
            "on this input at all, and an interval around a number that does not "
            "exist is not a confidence interval"
        )


# spec: EVAL-09
def bootstrap_metric_ci(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    metric: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], float],
    *,
    seed: int,
    config: Mapping[str, Any] | None = None,
    resamples: int | None = None,
    confidence: float | None = None,
) -> BootstrapCI:
    """Percentile bootstrap over test IMAGES, not instances.

    Instances inside one image are not independent - a crowded site photo
    contributes twenty correlated heads - so resampling instances would
    manufacture a confidence interval several times too narrow. The unit of
    resampling is the image.

    Seeded through numpy's Generator and driven only by that generator, so the
    same seed reproduces the same bounds bit for bit. `results/` numbers are
    supposed to be re-derivable, and an unseeded CI is not.

    UNDEFINED resamples (-1.0, "this draw contained no instance of the class")
    are EXCLUDED from the percentile and counted in `n_undefined`, not pooled.
    Pooling would put a fake -1.0 sample below every real one and pull the lower
    bound to -1; refusing outright would make EVAL-09's required interval on
    `person` AP impossible, since `person` appears in only 158 of 5,000 images
    and some draws legitimately contain none. Excluding narrows the interval to
    the draws where the metric exists, which is a defensible statement as long
    as the reader is told how many draws that was - hence the count.

    The POINT estimate is a different matter: if the metric is undefined on the
    full sample there is nothing to put an interval around, and that raises.
    """

    config = _resolve_config(config)
    if resamples is None:
        resamples = int(config["metrics"]["bootstrap_resamples"])
    if confidence is None:
        confidence = float(config["metrics"]["bootstrap_ci"])
    if resamples <= 0:
        raise BootstrapError(f"resamples must be positive, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise BootstrapError(f"bootstrap_ci must lie in (0, 1), got {confidence}")

    image_ids = [int(image["id"]) for image in ground_truth["images"]]
    if not image_ids:
        raise BootstrapError("Cannot bootstrap over an empty image set")

    # Computed FIRST so an undefined metric fails before a thousand wasted
    # evaluations, not after.
    point = float(metric(ground_truth, detections))
    _reject_unpoolable(point, "The point estimate")

    generator = np.random.default_rng(seed)
    values: list[float] = []
    n_undefined = 0
    for index in range(resamples):
        drawn = generator.integers(0, len(image_ids), size=len(image_ids))
        sampled = [image_ids[position] for position in drawn]
        resampled_gt, resampled_dt = _resample_by_image(ground_truth, detections, sampled)
        value = float(metric(resampled_gt, resampled_dt))
        if not np.isfinite(value):
            raise BootstrapError(
                f"Bootstrap resample {index} produced {value!r}; a non-finite value "
                "cannot be pooled into a percentile interval"
            )
        if value <= UNDEFINED:
            n_undefined += 1
            continue
        values.append(value)

    if not values:
        raise BootstrapError(
            f"All {resamples} resamples were undefined ({UNDEFINED}); there is nothing "
            "to take a percentile of"
        )

    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(values, [tail * 100.0, (1.0 - tail) * 100.0])
    return BootstrapCI(
        point=point,
        low=float(low),
        high=float(high),
        resamples=int(resamples),
        confidence=float(confidence),
        seed=int(seed),
        n_undefined=n_undefined,
    )


# spec: EVAL-05, EVAL-06, EVAL-07, EVAL-08
def evaluate_detection_metrics(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
) -> DetectionMetrics:
    """The whole EVAL-05..EVAL-08 bundle for one split of one run.

    Everything is computed in the coordinate system the ground truth is already
    in, which for this project means the original 416x416 annotation space. Put
    predictions through `detections_for_evaluation` BEFORE calling this; passing
    640-space boxes does not raise, it just measures a different thing.
    """

    config = _resolve_config(config)
    buckets = area_ranges_from_config(config)
    counts = instance_counts_by_bucket(ground_truth, config=config)
    bare_head = bare_head_recall(ground_truth, detections, config=config)
    primary_classes = list(config["metrics"]["primary_classes"])
    names = _category_name_by_id(ground_truth)

    if not detections:
        # An arm can legitimately predict nothing above threshold. Reuse the
        # training-side zero dict so both evaluators agree on what "nothing"
        # scores, instead of crashing inside pycocotools.
        stats = evaluate_detections(dict(ground_truth), [])
        # 0.0 is "found none of them" and UNDEFINED is "there were none to find".
        # A class with no ground truth is the SECOND case whether or not the model
        # produced detections, and the `else` branch below reports it as -1.0
        # because COCOeval does. Reporting a hard 0.0 here would make an
        # untrainable class look like a failed one, and would drag any average
        # taken over the column down towards zero.
        per_class_ap = {
            name: 0.0 if counts.class_totals.get(name, 0) else UNDEFINED
            for name in names.values()
        }
        per_class_ap_small = {
            name: 0.0 if counts.by_class.get(name, {}).get("small", 0) else UNDEFINED
            for name in names.values()
        }
    else:
        evaluator = _run_cocoeval(ground_truth, detections, buckets)
        stats = {
            name: float(value)
            for name, value in zip(COCO_STAT_NAMES, evaluator.stats, strict=True)
        }
        small_index = [label for label, _ in buckets].index("small")
        per_class_ap = {}
        per_class_ap_small = {}
        for class_index, category_id in enumerate(evaluator.params.catIds):
            name = names[int(category_id)]
            per_class_ap[name] = _average_precision(
                evaluator, class_index=class_index, area_index=0
            )
            per_class_ap_small[name] = _average_precision(
                evaluator, class_index=class_index, area_index=small_index
            )

    # Shared by both branches on purpose: the aggregate has to skip UNDEFINED
    # classes the same way regardless of whether the arm predicted anything.
    primary_map = _mean_defined(
        per_class_ap[name] for name in primary_classes if name in per_class_ap
    )
    primary_map_small = _mean_defined(
        per_class_ap_small[name] for name in primary_classes if name in per_class_ap_small
    )

    return DetectionMetrics(
        stats=stats,
        per_class_ap=per_class_ap,
        per_class_ap_small=per_class_ap_small,
        primary_map=primary_map,
        primary_map_small=primary_map_small,
        primary_classes=tuple(primary_classes),
        counts=counts,
        bare_head=bare_head,
        n_images=len(ground_truth["images"]),
        n_detections=len(detections),
    )


# spec: EVAL-07
def rescale_detections_to_original(
    detections: Sequence[Mapping[str, Any]],
    *,
    evaluated_size: tuple[float, float],
    original_sizes: Mapping[int, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Map detections from the evaluated resolution back to annotation space.

    Training runs at 640x640 while the annotations are 416x416. Evaluating the
    640-space boxes against 416-space ground truth does not raise - IoU just
    collapses and every area is 2.37x too big - so the mapping is a function
    with its own test rather than a comment in a script.

    `original_sizes` is per image because not every synthetic output is exactly
    416 wide (see `Sample.width`), so one global scale factor would be wrong for
    a handful of images and nothing would say so.
    """

    evaluated_width, evaluated_height = (float(value) for value in evaluated_size)
    if evaluated_width <= 0 or evaluated_height <= 0:
        raise ValueError(f"evaluated_size must be positive, got {evaluated_size}")

    rescaled: list[dict[str, Any]] = []
    for detection in detections:
        image_id = int(detection["image_id"])
        if image_id not in original_sizes:
            raise KeyError(
                f"No original size for image {image_id}; refusing to guess a scale factor"
            )
        original_width, original_height = (float(v) for v in original_sizes[image_id])
        scale_x = original_width / evaluated_width
        scale_y = original_height / evaluated_height
        x, y, width, height = (float(v) for v in detection["bbox"])
        copy = dict(detection)
        copy["bbox"] = [x * scale_x, y * scale_y, width * scale_x, height * scale_y]
        rescaled.append(copy)
    return rescaled


# spec: EVAL-07
def evaluates_in_original_coordinates(config: Mapping[str, Any] | None = None) -> bool:
    """Read metrics.evaluate_in_original_coordinates.

    EVAL-07's whole rule lives in this one key, so it must drive a branch
    somewhere or it is a comment with a colon in it.
    """

    config = _resolve_config(config)
    return bool(config["metrics"]["evaluate_in_original_coordinates"])


# spec: EVAL-07
def detections_for_evaluation(
    detections: Sequence[Mapping[str, Any]],
    *,
    evaluated_size: tuple[float, float],
    original_sizes: Mapping[int, tuple[float, float]],
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Put detections into the coordinate system the CONFIG says to evaluate in.

    This is the entry point `scripts/eval.py` uses rather than calling
    `rescale_detections_to_original` directly, so that flipping
    metrics.evaluate_in_original_coordinates actually changes what gets measured
    instead of changing nothing:

      true  - map predictions back to the 416-space annotations (EVAL-07, and
              the setting this project ships)
      false - evaluate where the model predicted, which reads AP_small over
              areas inflated by (640/416)^2

    The `false` path is not a supported reporting mode; it exists so the rule can
    be demonstrated to have teeth, and so a future caller cannot set the key to
    false and have it quietly mean nothing.
    """

    if evaluates_in_original_coordinates(config):
        return rescale_detections_to_original(
            detections, evaluated_size=evaluated_size, original_sizes=original_sizes
        )
    return [dict(detection) for detection in detections]


def ground_truth_as_detections(
    ground_truth: Mapping[str, Any], *, score: float = 1.0
) -> list[dict[str, Any]]:
    """Turn the ground truth into perfect detections, for self-evaluation.

    Feeding this back through the evaluator must produce mAP 1.000. It is a
    five-line check that simultaneously catches xywh-vs-xyxy confusion, a wrong
    area field, an iscrowd that silently ignores annotations, and image ids that
    do not line up between the two files.

    "1.000" and not "1.0": pycocotools computes precision as
    `tp / (fp + tp + np.spacing(1))`, so a flawless result is
    0.9999999999999998 and bit-exact equality with 1.0 is unreachable by
    construction. Measured, not assumed - see the self-evaluation test.
    """

    return [
        {
            "image_id": int(annotation["image_id"]),
            "category_id": int(annotation["category_id"]),
            "bbox": [float(value) for value in annotation["bbox"]],
            "score": float(score),
        }
        for annotation in ground_truth["annotations"]
        if int(annotation.get("iscrowd", 0)) == 0
    ]


# spec: EVAL-11, EVAL-12
def detection_metric_rows(
    metrics: DetectionMetrics, *, arm: str, seed: int, split: str
) -> list[MetricRow]:
    """Flatten one evaluation into long-format rows.

    Instance counts are emitted as their own rows as well as riding along with
    the AP they qualify, so a reader can re-derive "AP_small over N instances"
    from the CSV alone without reopening the annotations (EVAL-08 + EVAL-12).
    """

    rows: list[MetricRow] = []

    def add(metric: str, value: float, n_instances: int | None, notes: str = "") -> None:
        rows.append(
            MetricRow(
                arm=arm,
                seed=seed,
                split=split,
                metric=metric,
                value=float(value),
                n_instances=n_instances,
                n_images=metrics.n_images,
                notes=notes,
            )
        )

    for name, value in metrics.stats.items():
        bucket = _STAT_BUCKET.get(name, ALL_LABEL)
        add(name, value, metrics.counts.total.get(bucket))
    for name, value in metrics.per_class_ap.items():
        add(f"ap.{name}", value, metrics.counts.class_totals.get(name))
    for name, value in metrics.per_class_ap_small.items():
        add(f"ap_small.{name}", value, metrics.counts.by_class.get(name, {}).get("small"))
    # counts.total counts EVERY class, but the two primary_* metrics average only
    # metrics.primary_classes. Pairing the all-class count with a primary-class
    # metric hands the reader the wrong denominator, which is exactly the mistake
    # EVAL-08 exists to prevent - on this dataset `person` alone is 751 of the
    # instances, so the two counts are not close.
    primary_instances = sum(
        metrics.counts.class_totals.get(name, 0) for name in metrics.primary_classes
    )
    primary_small_instances = sum(
        metrics.counts.by_class.get(name, {}).get("small", 0)
        for name in metrics.primary_classes
    )
    add("primary_map", metrics.primary_map, primary_instances)
    add("primary_map_small", metrics.primary_map_small, primary_small_instances)
    for bucket, count in metrics.counts.total.items():
        add(f"instances.{bucket}", float(count), count)
    add(
        "bare_head_recall",
        metrics.bare_head.recall,
        metrics.bare_head.n_ground_truth,
        notes=f"iou={metrics.bare_head.iou_threshold}",
    )
    add("n_detections", float(metrics.n_detections), None)
    return rows


# EVAL-09 wants uncertainty next to every main-table number. These are the two
# headline metrics; the per-class rows that also need one are derived from the
# config in `bootstrap_ci_metric_names` rather than named here.
HEADLINE_CI_METRICS = ("primary_map_small", "bare_head_recall")


# spec: EVAL-09
def bootstrap_ci_metric_names(
    ground_truth: Mapping[str, Any], *, config: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Which CSV rows EVAL-09 requires a bootstrap interval on.

    The two headline metrics, plus per-class AP for every class OUTSIDE
    metrics.primary_classes. The second group is derived, not spelled out: on
    this dataset it resolves to `person`, the class EXP-03 says must never carry
    a bare point estimate, and deriving it means adding a fourth class to the
    config cannot leave it silently without an interval.
    """

    config = _resolve_config(config)
    primary = set(config["metrics"]["primary_classes"])
    secondary = sorted(
        f"ap.{name}"
        for name in _category_name_by_id(ground_truth).values()
        if name not in primary
    )
    return (*HEADLINE_CI_METRICS, *secondary)


def _row_metric(
    metric: str, config: Mapping[str, Any]
) -> Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], float]:
    """The callable that recomputes one metric NAME from a resampled split.

    Named by the row's own metric string so an interval cannot end up around a
    differently-computed quantity than the value it sits next to.
    """

    if metric == "bare_head_recall":
        return lambda gt, dt: bare_head_recall(gt, dt, config=config).recall
    if metric in {"primary_map", "primary_map_small"}:
        return lambda gt, dt: float(
            getattr(evaluate_detection_metrics(gt, dt, config=config), metric)
        )
    if metric.startswith("ap_small."):
        name = metric.split(".", 1)[1]
        return lambda gt, dt: evaluate_detection_metrics(
            gt, dt, config=config
        ).per_class_ap_small.get(name, UNDEFINED)
    if metric.startswith("ap."):
        name = metric.split(".", 1)[1]
        return lambda gt, dt: evaluate_detection_metrics(
            gt, dt, config=config
        ).per_class_ap.get(name, UNDEFINED)
    raise KeyError(
        f"No bootstrap recipe for metric {metric!r}; add one rather than shipping a "
        "row whose interval came from a different quantity"
    )


def _join_note(existing: str, extra: str) -> str:
    return f"{existing}; {extra}" if existing else extra


# spec: EVAL-09
def attach_bootstrap_cis(
    rows: Sequence[MetricRow],
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    config: Mapping[str, Any] | None = None,
    resamples: int | None = None,
    metric_names: Sequence[str] | None = None,
) -> list[MetricRow]:
    """Fill ci_low/ci_high on the rows EVAL-09 requires an interval on.

    `detection_metric_rows` cannot do this itself: it is handed a finished
    DetectionMetrics, while an interval needs the ground truth and the detections
    back so the metric can be recomputed on each resample. Without this step the
    ci_low/ci_high columns exist and are empty in every row, which a reader takes
    as "no uncertainty" rather than "never measured".

    A row whose value is UNDEFINED gets no interval and says so in `notes`, since
    bootstrapping a metric that does not exist on the full split is meaningless.
    """

    config = _resolve_config(config)
    wanted = tuple(
        bootstrap_ci_metric_names(ground_truth, config=config)
        if metric_names is None
        else metric_names
    )
    present = {row.metric for row in rows}
    missing = sorted(name for name in wanted if name not in present)
    if missing:
        raise KeyError(
            f"EVAL-09 requires an interval on {missing}, but no such row exists. "
            f"Rows present: {sorted(present)}"
        )

    # Resolved before the row loop and keyed by metric name, so a metric that
    # somehow appears on two rows is bootstrapped once rather than twice.
    intervals = {
        name: bootstrap_metric_ci(
            ground_truth,
            detections,
            _row_metric(name, config),
            seed=seed,
            config=config,
            resamples=resamples,
        )
        for name in sorted(
            {row.metric for row in rows if row.metric in wanted and row.value > UNDEFINED}
        )
    }

    updated: list[MetricRow] = []
    for row in rows:
        if row.metric not in wanted:
            updated.append(row)
            continue
        if row.metric not in intervals:
            updated.append(
                replace(row, notes=_join_note(row.notes, "no CI: metric is undefined"))
            )
            continue
        interval = intervals[row.metric]
        note = _join_note(
            row.notes,
            f"bootstrap ci={interval.confidence:g} resamples={interval.resamples} "
            f"seed={interval.seed}",
        )
        if interval.n_undefined:
            note = _join_note(
                note, f"{interval.n_undefined} resample(s) undefined and excluded"
            )
        updated.append(replace(row, ci_low=interval.low, ci_high=interval.high, notes=note))
    return updated


def default_detection_metrics_path() -> Path:
    """results/detection_metrics.csv, rooted through configs/paths.yaml."""

    return load_project_paths().project_root / "results" / DETECTION_METRICS_FILENAME


def _row_to_record(row: MetricRow) -> dict[str, Any]:
    return {
        "arm": row.arm,
        "seed": row.seed,
        "split": row.split,
        "metric": row.metric,
        "value": repr(float(row.value)),
        "n_instances": "" if row.n_instances is None else int(row.n_instances),
        "n_images": "" if row.n_images is None else int(row.n_images),
        "ci_low": "" if row.ci_low is None else repr(float(row.ci_low)),
        "ci_high": "" if row.ci_high is None else repr(float(row.ci_high)),
        "notes": row.notes,
    }


def _record_to_row(record: Mapping[str, str]) -> MetricRow:
    def optional_int(key: str) -> int | None:
        raw = record[key]
        return None if raw == "" else int(raw)

    def optional_float(key: str) -> float | None:
        raw = record[key]
        return None if raw == "" else float(raw)

    return MetricRow(
        arm=record["arm"],
        seed=int(record["seed"]),
        split=record["split"],
        metric=record["metric"],
        value=float(record["value"]),
        n_instances=optional_int("n_instances"),
        n_images=optional_int("n_images"),
        ci_low=optional_float("ci_low"),
        ci_high=optional_float("ci_high"),
        notes=record["notes"],
    )


# spec: EVAL-11
def write_detection_metrics_csv(
    rows: Sequence[MetricRow], path: Path | None = None
) -> Path:
    """Write results/detection_metrics.csv in long format.

    Values go through repr() so a float round-trips exactly; a table rebuilt
    from this file has to equal the one printed from memory (EVAL-12), and
    "%.4f" would break that in the fourth decimal.

    newline="" is the csv module's contract (it does its own line endings), and
    lineterminator="\\n" then keeps the file LF-only on Windows.
    """

    path = default_detection_metrics_path() if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(DETECTION_METRIC_FIELDS), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_to_record(row))
    return path


# spec: EVAL-12
def read_detection_metrics_csv(path: Path) -> list[MetricRow]:
    """Read the long-format CSV back, so every table can be re-aggregated."""

    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return [_record_to_row(record) for record in csv.DictReader(stream)]


def _image_identity(name: str) -> str:
    """Normalised key for "is this the same image", not "is this the same string".

    The two lists reaching the leak check are written by different producers - a
    manifest that used `Path.as_posix()`, a training log that printed Windows
    paths, a CSV that kept only file names - so raw string equality answers a
    question nobody asked. `data\\test\\A.PNG` and `data/test/a.png` are one
    image, and a check that calls them disjoint reports PASS on a real leak,
    which is strictly worse than having no check at all.

    Basename is the identity here because Hard Hat Workers file names are unique
    across the whole 5,000-image set (DATA-*), so two paths sharing a last
    component ARE the same image. casefold() because Windows paths are
    case-insensitive and the manifest's casing is not guaranteed.
    """

    return PurePosixPath(str(name).replace("\\", "/")).name.casefold()


# spec: EVAL-14
def assert_no_test_images_in_training_list(
    train_names: Iterable[str], test_names: Iterable[str]
) -> None:
    """Abort if any Test image appears in a training list.

    Every claim in this project is a relative one on a frozen Test set, so one
    leaked image does not degrade the result, it invalidates it. The offending
    names are in the message because "leakage detected" is not actionable, and
    both spellings are shown because the mismatch that hid the leak is itself
    the thing to fix.
    """

    train_by_key: dict[str, set[str]] = defaultdict(set)
    for name in train_names:
        train_by_key[_image_identity(name)].add(str(name))
    test_by_key: dict[str, set[str]] = defaultdict(set)
    for name in test_names:
        test_by_key[_image_identity(name)].add(str(name))

    overlap = sorted(set(train_by_key) & set(test_by_key))
    if overlap:
        # Joined by hand rather than interpolating the containers: a list repr
        # escapes every backslash, and the point of the message is to show the
        # two spellings exactly as the caller wrote them.
        offenders = " | ".join(
            f"{key} (train: {', '.join(sorted(train_by_key[key]))}"
            f" / test: {', '.join(sorted(test_by_key[key]))})"
            for key in overlap
        )
        raise SplitLeakageError(
            f"{len(overlap)} Test image(s) appear in the training list: {offenders}"
        )


# spec: EVAL-13
def cross_check_with_faster_coco_eval(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, tuple[float, float]]:
    """Compare pycocotools against faster-coco-eval on identical input.

    pycocotools stays the correctness baseline. faster-coco-eval advertises
    itself as a drop-in, which is a claim worth verifying rather than a premise
    worth trusting, so this function exists and its test activates the moment
    the package is installed. It is NOT a dependency today.

    Returns {stat: (pycocotools, faster)} and raises if any pair differs.
    """

    try:
        from faster_coco_eval import COCO as FasterCOCO
        from faster_coco_eval import COCOeval_faster
    except ImportError as error:
        raise ImportError(
            "faster-coco-eval is not installed. EVAL-13 keeps pycocotools as the "
            "correctness baseline and only uses faster-coco-eval once this "
            "cross-check passes on the same input."
        ) from error

    config = _resolve_config(config)
    buckets = area_ranges_from_config(config)
    baseline = _run_cocoeval(ground_truth, detections, buckets)

    with contextlib.redirect_stdout(io.StringIO()):
        # Deep copy for the same reason `_run_cocoeval` does: the evaluator
        # writes `ignore` into the annotation dicts it is given.
        faster_gt = FasterCOCO(copy.deepcopy(dict(ground_truth)))
        faster_dt = faster_gt.loadRes([dict(d) for d in detections])
        evaluator = COCOeval_faster(faster_gt, faster_dt, iouType="bbox")
        evaluator.params.areaRng = [[low, high] for _, (low, high) in buckets]
        evaluator.params.areaRngLbl = [label for label, _ in buckets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    compared = {
        name: (float(left), float(right))
        for name, left, right in zip(
            COCO_STAT_NAMES, baseline.stats, evaluator.stats, strict=True
        )
    }
    differing = {name: pair for name, pair in compared.items() if pair[0] != pair[1]}
    if differing:
        raise EvaluatorMismatchError(
            f"pycocotools and faster-coco-eval disagree on {differing}"
        )
    return compared
