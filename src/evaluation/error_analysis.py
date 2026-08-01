"""Error analysis for the four-arm comparison (EVAL-15..EVAL-18).

Four things live here, and each exists because the alternative is a report that
reads well and says less than it should:

* **EVAL-15** — a four-way classification of every ground-truth box and every
  detection against a baseline arm and a best arm. `new_false_positive` is the
  cost of synthetic data, so it is not an optional panel: every entry point that
  can render a report calls `assert_reportable_categories` first and refuses a
  category list that omits it.
* **EVAL-16** — false positives per image on the hard-negative subset, delegated
  to `detection.hard_negative_false_positives_per_image` rather than
  reimplemented, so the number in this report and the number in the main table
  are the same computation.
* **EVAL-17** — per-slice metrics plus a *decision rule*, not a human's later
  reading of a table. If the scenario that took the largest share of the
  synthetic budget maps to a slice that improved no more than the others, the
  verdict is `more_data_not_targeted` and the generated prose says exactly
  "more data, not targeted data".
* **EVAL-18** — when no synthetic arm beats the baseline, the report walks the
  experiment_protocol §7 checklist instead of trailing off, and states the
  equal-step exposure confound in numbers computed from
  `src.training.arms.equal_step_budget`.

Five outcomes, not four. The spec names four, and the shipped
`configs/evaluation.yaml` lists four, but a ground-truth box the baseline caught
and the best arm missed is none of them: it is not "neither got it right". That
case is `new_false_negative`, the recall-side twin of `new_false_positive`, and
it is counted in every counts table this module renders. Dropping it would leave
a hole exactly where a synthetic-data regression would fall, which is the
failure mode CLAUDE.md's ban on selective reporting is about. Figures are still
rendered only for the configured categories.

Every analysis threshold comes from `configs/evaluation.yaml`. The rendering
constants below are presentation only — figure size, DPI, crop zoom — and follow
the precedent of `scripts/plot_training_curves.py`; none of them can change a
number in the report.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # No display on this machine; must be set before pyplot.
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.paths import PROJECT_ROOT

# Reused rather than reimplemented. The three underscore-prefixed names below are
# imported deliberately: a second IoU, a second category lookup or a second
# unknown-class rule inside this package is how two modules end up disagreeing
# about which box matched which class, silently. isort splits the aliased imports
# into their own statements; they all come from the same module.
from src.evaluation.detection import (
    UNDEFINED,
    DetectionMetrics,
    HardNegativeReport,
    evaluate_detection_metrics,
    hard_negative_false_positives_per_image,
    load_evaluation_config,
)
from src.evaluation.detection import (
    _category_name_by_id as category_name_by_id,
)
from src.evaluation.detection import (
    _detection_class_name as detection_class_name,
)
from src.evaluation.detection import (
    _iou_xywh as iou_xywh,
)
from src.training.arms import ArmComposition, equal_step_budget

# --- EVAL-15 outcome vocabulary ------------------------------------------------
FIXED_FALSE_NEGATIVE = "fixed_false_negative"
FIXED_FALSE_POSITIVE = "fixed_false_positive"
NEW_FALSE_POSITIVE = "new_false_positive"
NEW_FALSE_NEGATIVE = "new_false_negative"
BOTH_WRONG = "both_wrong"

# Order is the reporting order of the counts table.
OUTCOMES = (
    FIXED_FALSE_NEGATIVE,
    FIXED_FALSE_POSITIVE,
    NEW_FALSE_POSITIVE,
    NEW_FALSE_NEGATIVE,
    BOTH_WRONG,
)

# Categories that may never be dropped from a rendered report. Synthetic data
# buys recall by spending precision; a report that shows only what it fixed is
# the selective reporting CLAUDE.md forbids.
MANDATORY_CATEGORIES = (NEW_FALSE_POSITIVE,)

ANCHOR_GROUND_TRUTH = "ground_truth"
ANCHOR_DETECTION = "detection"

# Which EVAL-17 slice each compose.yaml scenario is supposed to move. Scenarios
# with no slice cannot have their targeting tested by this instrument, and the
# verdict says so instead of guessing.
SCENARIO_SLICE = {
    "small_distant": "small_object",
    "crowded": "crowded",
    "low_light_blur": "low_light",
}

SLICE_METRICS = ("primary_map", "primary_map_small", "bare_head_recall")
DEFAULT_SLICE_METRIC = "primary_map"

# --- Targeting verdicts --------------------------------------------------------
TARGETED = "targeted"
MORE_DATA_NOT_TARGETED = "more_data_not_targeted"
TARGET_SLICE_DID_NOT_IMPROVE = "target_slice_did_not_improve"
TARGET_SLICE_UNMEASURABLE = "target_slice_unmeasurable"
BUDGET_LEADER_HAS_NO_SLICE = "budget_leader_has_no_slice"

# The sentence EVAL-17 requires verbatim when the budget leader's slice improved
# no more than the others. Kept as a constant so a test can assert on it and a
# careless reword cannot soften it.
MORE_DATA_SENTENCE = "more data, not targeted data"

# --- Output locations ----------------------------------------------------------
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "error_analysis"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "error_analysis.md"
DEFAULT_PAYLOAD_PATH = PROJECT_ROOT / "reports" / "error_analysis.json"

# --- Presentation constants, never analysis parameters -------------------------
# Figure geometry only. Changing any of these changes how a grid looks and cannot
# change a single number in the report, which is why they live here and not in
# configs/evaluation.yaml. Sized to keep a 12-sample grid comfortably inside the
# few-hundred-kilobyte range CLAUDE.md allows in the project folder.
CROP_ZOOM = 3.0
MIN_CROP_PX = 48.0
# Two ITEMS (four panels) per row, not three. Each cell carries a three-line
# label naming the arm, the image, the class, the score and the category, and at
# three items per row those labels overlapped their neighbours - a figure whose
# captions cannot be read is not evidence of anything.
CELL_INCHES = 1.45
ITEMS_PER_ROW = 2
LABEL_FONT_SIZE = 5.5
FIGURE_DPI = 80
GT_COLOUR = "#0173b2"
DETECTION_COLOUR = "#d55e00"
# A crop is displayed several times larger than its source pixels. Bilinear
# upscaling invents smooth gradients that PNG cannot compress, which pushed a
# 12-sample grid past 700 KiB; nearest keeps the real pixel blocks, roughly
# quarters the file, and is the more honest rendering of a 12x12 px detection.
IMSHOW_INTERPOLATION = "nearest"


class ErrorAnalysisConfigError(RuntimeError):
    """Raised when configs/evaluation.yaml cannot drive the error analysis."""


class PredictionFileError(RuntimeError):
    """Raised when a per-arm prediction file is not COCO detections."""


class FigureInputError(RuntimeError):
    """Raised when a comparison grid cannot be drawn from what was supplied."""


@dataclass(frozen=True)
class ErrorAnalysisConfig:
    """The `error_analysis.*` block, validated (EVAL-15)."""

    baseline_arm: str
    best_arm: str
    samples_per_category: int
    categories: tuple[str, ...]


@dataclass(frozen=True)
class Box:
    """One box on one panel: ground truth carries no score, a detection does."""

    xywh: tuple[float, float, float, float]
    class_name: str
    score: float | None = None


@dataclass(frozen=True)
class GroundTruthRecord:
    index: int
    image_id: int
    class_name: str
    bbox: tuple[float, float, float, float]

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]


@dataclass(frozen=True)
class DetectionRecord:
    index: int
    image_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    score: float

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]


@dataclass(frozen=True)
class ComparisonItem:
    """One cell pair of the EVAL-15 grid."""

    category: str
    image_id: int
    class_name: str
    anchor: str
    anchor_box: tuple[float, float, float, float]
    baseline: Box | None
    best: Box | None

    @property
    def area(self) -> float:
        return self.anchor_box[2] * self.anchor_box[3]

    def score_for(self, side: str) -> float | None:
        box = self.baseline if side == "baseline" else self.best
        return None if box is None else box.score

    def as_dict(self) -> dict[str, Any]:
        def box_payload(box: Box | None) -> dict[str, Any] | None:
            if box is None:
                return None
            return {
                "bbox": list(box.xywh),
                "class_name": box.class_name,
                "score": box.score,
            }

        return {
            "category": self.category,
            "image_id": self.image_id,
            "class_name": self.class_name,
            "anchor": self.anchor,
            "anchor_box": list(self.anchor_box),
            "area": self.area,
            "baseline": box_payload(self.baseline),
            "best": box_payload(self.best),
        }


@dataclass(frozen=True)
class HardNegativeRow:
    """One arm's EVAL-16 number, with the subset it was measured on."""

    arm: str
    false_positives_per_image: float
    n_false_positives: int
    n_images: int
    score_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "false_positives_per_image": self.false_positives_per_image,
            "n_false_positives": self.n_false_positives,
            "n_images": self.n_images,
            "score_threshold": self.score_threshold,
        }


@dataclass(frozen=True)
class SliceMetric:
    """One arm's metric on one EVAL-17 slice, with the slice's own size."""

    slice_name: str
    arm: str
    metric: str
    value: float
    n_images: int
    n_instances: int

    @property
    def is_defined(self) -> bool:
        return self.value > UNDEFINED

    def as_dict(self) -> dict[str, Any]:
        return {
            "slice": self.slice_name,
            "arm": self.arm,
            "metric": self.metric,
            "value": self.value,
            "n_images": self.n_images,
            "n_instances": self.n_instances,
        }


@dataclass(frozen=True)
class TargetingVerdict:
    """Did the largest slice of the synthetic budget actually hit its target?

    `budget_leader` is the biggest scenario overall; `tested_scenario` is the
    biggest one an EVAL-17 slice can actually isolate. They differ on this
    project's delivered mix, and the two fields are separate so the report can
    say which question it answered.
    """

    verdict: str
    metric: str
    budget_leader: str
    budget_leader_share: float
    tested_scenario: str | None
    tested_share: float | None
    target_slice: str | None
    target_delta: float | None
    other_deltas: dict[str, float]
    sentence: str

    @property
    def max_other_delta(self) -> float | None:
        return max(self.other_deltas.values()) if self.other_deltas else None

    @property
    def tested_the_budget_leader(self) -> bool:
        return self.tested_scenario == self.budget_leader

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "metric": self.metric,
            "budget_leader": self.budget_leader,
            "budget_leader_share": self.budget_leader_share,
            "tested_scenario": self.tested_scenario,
            "tested_share": self.tested_share,
            "tested_the_budget_leader": self.tested_the_budget_leader,
            "target_slice": self.target_slice,
            "target_delta": self.target_delta,
            "other_deltas": dict(self.other_deltas),
            "max_other_delta": self.max_other_delta,
            "sentence": self.sentence,
        }


@dataclass(frozen=True)
class ExposureConfound:
    """How often each arm saw each real image under the equal-step budget."""

    exposures: dict[str, float]
    total_steps: int
    reference_arm: str
    reference_epochs: int
    batch_size: int

    def ratio(self, numerator_arm: str, denominator_arm: str) -> float:
        denominator = self.exposures[denominator_arm]
        if denominator <= 0:
            raise ErrorAnalysisConfigError(
                f"Arm {denominator_arm!r} has {denominator} real-image exposures; "
                "an exposure ratio against it is not a number"
            )
        return self.exposures[numerator_arm] / denominator

    def as_dict(self) -> dict[str, Any]:
        return {
            "exposures": dict(self.exposures),
            "total_steps": self.total_steps,
            "reference_arm": self.reference_arm,
            "reference_epochs": self.reference_epochs,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class NegativeResultVerdict:
    """Did any synthetic arm actually beat the baseline (EVAL-18)?"""

    metric: str
    baseline_arm: str
    baseline_value: float
    arm_values: dict[str, float]
    synthetic_arms: tuple[str, ...]
    winners: tuple[str, ...]

    @property
    def is_negative(self) -> bool:
        return not self.winners

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline_arm": self.baseline_arm,
            "baseline_value": self.baseline_value,
            "arm_values": dict(self.arm_values),
            "synthetic_arms": list(self.synthetic_arms),
            "winners": list(self.winners),
            "is_negative": self.is_negative,
        }


@dataclass(frozen=True)
class ChecklistItem:
    """One line of the experiment_protocol §7 walk (EVAL-18)."""

    key: str
    question: str
    evidence: str


@dataclass(frozen=True)
class ErrorAnalysis:
    """Everything reports/error_analysis.md and its JSON sidecar are built from."""

    config: ErrorAnalysisConfig
    iou_threshold: float
    score_threshold: float
    items: tuple[ComparisonItem, ...]
    counts: dict[str, int]
    counts_by_class: dict[str, dict[str, int]]
    samples: dict[str, tuple[ComparisonItem, ...]]
    figures: dict[str, str]
    hard_negatives: tuple[HardNegativeRow, ...]
    slice_metrics: tuple[SliceMetric, ...]
    slice_deltas: dict[str, float]
    targeting: TargetingVerdict | None
    negative_result: NegativeResultVerdict | None
    exposure: ExposureConfound | None
    budget_shares: dict[str, float]

    def as_payload(self) -> dict[str, Any]:
        return {
            "config": {
                "baseline_arm": self.config.baseline_arm,
                "best_arm": self.config.best_arm,
                "samples_per_category": self.config.samples_per_category,
                "categories": list(self.config.categories),
            },
            "iou_threshold": self.iou_threshold,
            "score_threshold": self.score_threshold,
            "counts": dict(self.counts),
            "counts_by_class": {
                outcome: dict(by_class) for outcome, by_class in self.counts_by_class.items()
            },
            "samples": {
                category: [item.as_dict() for item in items]
                for category, items in self.samples.items()
            },
            "figures": dict(self.figures),
            "hard_negatives": [row.as_dict() for row in self.hard_negatives],
            "slice_metrics": [row.as_dict() for row in self.slice_metrics],
            "slice_deltas": dict(self.slice_deltas),
            "budget_shares": dict(self.budget_shares),
            "targeting": None if self.targeting is None else self.targeting.as_dict(),
            "negative_result": (
                None if self.negative_result is None else self.negative_result.as_dict()
            ),
            "exposure": None if self.exposure is None else self.exposure.as_dict(),
        }


# spec: EVAL-15
def assert_reportable_categories(categories: Sequence[str]) -> None:
    """Refuse a category list that cannot honestly be rendered.

    `new_false_positive` is structurally mandatory. A grid that shows only what
    the best arm fixed is an advertisement, and CLAUDE.md forbids selective
    reporting outright, so every path that could produce a report goes through
    this function rather than trusting the caller to have read the rule.
    """

    names = list(categories)
    if not names:
        raise ErrorAnalysisConfigError(
            "error_analysis.categories is empty; there is nothing to report"
        )
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ErrorAnalysisConfigError(
            f"error_analysis.categories repeats {duplicates}; each category is rendered "
            "once and a repeat silently doubles a figure"
        )
    unknown = sorted(set(names) - set(OUTCOMES))
    if unknown:
        raise ErrorAnalysisConfigError(
            f"error_analysis.categories names {unknown}, which this module does not "
            f"classify. Known outcomes: {list(OUTCOMES)}"
        )
    missing = [name for name in MANDATORY_CATEGORIES if name not in names]
    if missing:
        raise ErrorAnalysisConfigError(
            f"error_analysis.categories omits {missing}. Those categories carry the COST "
            "of synthetic data; a report without them shows only the benefit, which is "
            "the selective reporting CLAUDE.md forbids. Refusing to render."
        )


# spec: EVAL-15
def load_error_analysis_config(
    config: Mapping[str, Any] | None = None,
) -> ErrorAnalysisConfig:
    """Read and validate `error_analysis.*` from configs/evaluation.yaml."""

    payload = load_evaluation_config() if config is None else config
    if "error_analysis" not in payload:
        raise ErrorAnalysisConfigError(
            "configs/evaluation.yaml has no `error_analysis` block; EVAL-15 has no "
            "arms, no sample count and no categories to run on"
        )
    block = payload["error_analysis"]

    arms = [str(name) for name in block["compare_arms"]]
    if len(arms) != 2:
        raise ErrorAnalysisConfigError(
            f"error_analysis.compare_arms must name exactly two arms (baseline, best), "
            f"got {arms}"
        )
    if arms[0] == arms[1]:
        raise ErrorAnalysisConfigError(
            f"error_analysis.compare_arms names {arms[0]!r} twice; comparing an arm "
            "against itself classifies every box as agreement and reports nothing"
        )

    samples = int(block["samples_per_category"])
    if samples <= 0:
        raise ErrorAnalysisConfigError(
            f"error_analysis.samples_per_category must be positive, got {samples}"
        )

    categories = tuple(str(name) for name in block["categories"])
    assert_reportable_categories(categories)
    return ErrorAnalysisConfig(
        baseline_arm=arms[0],
        best_arm=arms[1],
        samples_per_category=samples,
        categories=categories,
    )


def default_iou_threshold(config: Mapping[str, Any] | None = None) -> float:
    """`metrics.bare_head_recall_iou` — the same IoU the headline recall uses."""

    payload = load_evaluation_config() if config is None else config
    return float(payload["metrics"]["bare_head_recall_iou"])


def default_score_threshold(config: Mapping[str, Any] | None = None) -> float:
    """`compliance.score_threshold` — the operating point a demo actually runs at.

    An error grid is a picture of boxes a deployed detector would draw, so it is
    taken at the operating point rather than integrated over every confidence
    the way mAP is.
    """

    payload = load_evaluation_config() if config is None else config
    return float(payload["compliance"]["score_threshold"])


def default_predictions_path(arm: str, directory: Path | None = None) -> Path:
    """results/predictions/<arm>_test.json — the documented default layout."""

    root = DEFAULT_PREDICTIONS_DIR if directory is None else Path(directory)
    return root / f"{arm}_test.json"


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Read one arm's COCO detections, refusing anything that is not that.

    The four required keys are checked here rather than in the middle of the
    comparison, because a file missing `score` produces a KeyError three call
    levels down, and a file that is a COCO *dict* instead of a detection *list*
    would otherwise iterate its top-level keys as strings.
    """

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PredictionFileError(
            f"No prediction file at {path}. The default layout is "
            f"results/predictions/<arm>_test.json; pass an explicit path if the glue "
            "that writes them uses another one."
        ) from error
    except json.JSONDecodeError as error:
        raise PredictionFileError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(payload, list):
        raise PredictionFileError(
            f"{path} holds a {type(payload).__name__}, not a list of COCO detections. "
            "EVAL-15 wants the detection list, not a COCO results dict."
        )
    required = ("image_id", "category_id", "bbox", "score")
    detections: list[dict[str, Any]] = []
    for position, entry in enumerate(payload):
        if not isinstance(entry, Mapping):
            raise PredictionFileError(
                f"{path} entry {position} is a {type(entry).__name__}, not a mapping"
            )
        missing = [key for key in required if key not in entry]
        if missing:
            raise PredictionFileError(f"{path} entry {position} is missing {missing}")
        bbox = entry["bbox"]
        if not isinstance(bbox, Sequence) or isinstance(bbox, str) or len(bbox) != 4:
            raise PredictionFileError(
                f"{path} entry {position} has bbox {bbox!r}; COCO detections carry "
                "[x, y, w, h]"
            )
        detections.append(dict(entry))
    return detections


def ground_truth_records(ground_truth: Mapping[str, Any]) -> tuple[GroundTruthRecord, ...]:
    """Every non-crowd ground-truth box, indexed in file order.

    Crowd annotations are skipped for the same reason COCOeval ignores them: a
    detection landing on one is neither a hit nor a false positive, so counting
    it either way would make the grid disagree with the main table.
    """

    names = category_name_by_id(ground_truth)
    records: list[GroundTruthRecord] = []
    for annotation in ground_truth["annotations"]:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        x, y, width, height = (float(value) for value in annotation["bbox"])
        records.append(
            GroundTruthRecord(
                index=len(records),
                image_id=int(annotation["image_id"]),
                class_name=names[int(annotation["category_id"])],
                bbox=(x, y, width, height),
            )
        )
    return tuple(records)


def detection_records(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    *,
    score_threshold: float,
) -> tuple[DetectionRecord, ...]:
    """Detections at or above the operating point, with class ids validated.

    The class lookup runs on EVERY detection, including the ones the threshold
    then drops. An unknown category id is a broken prediction file whether or
    not that particular box would have been shown, and letting the threshold
    hide it means the file gets more acceptable as it gets more broken.
    """

    names = category_name_by_id(ground_truth)
    records: list[DetectionRecord] = []
    for detection in detections:
        class_name = detection_class_name(names, detection)
        score = float(detection["score"])
        if score < score_threshold:
            continue
        x, y, width, height = (float(value) for value in detection["bbox"])
        records.append(
            DetectionRecord(
                index=len(records),
                image_id=int(detection["image_id"]),
                class_name=class_name,
                bbox=(x, y, width, height),
                score=score,
            )
        )
    return tuple(records)


def match_arm(
    gt_records: Sequence[GroundTruthRecord],
    det_records: Sequence[DetectionRecord],
    *,
    iou_threshold: float,
) -> tuple[dict[int, int], dict[int, int]]:
    """Greedy, class-aware, highest-score-first matching for ONE arm.

    Returns (gt index -> det index, det index -> gt index). One ground-truth box
    absorbs at most one detection, so a pile of duplicate boxes on one head is a
    hit plus false positives rather than several hits — which is what makes
    `fixed_false_positive` mean anything.
    """

    by_key: dict[tuple[int, str], list[GroundTruthRecord]] = defaultdict(list)
    for record in gt_records:
        by_key[(record.image_id, record.class_name)].append(record)

    matched_gt: dict[int, int] = {}
    matched_det: dict[int, int] = {}
    # Stable: ties on score fall back to the detection's own index, so the
    # matching does not depend on dict or file ordering.
    for detection in sorted(det_records, key=lambda item: (-item.score, item.index)):
        best_record: GroundTruthRecord | None = None
        best_iou = 0.0
        for record in by_key.get((detection.image_id, detection.class_name), ()):
            if record.index in matched_gt:
                continue
            value = iou_xywh(detection.bbox, record.bbox)
            if value > best_iou:
                best_record, best_iou = record, value
        if best_record is not None and best_iou >= iou_threshold:
            matched_gt[best_record.index] = detection.index
            matched_det[detection.index] = best_record.index
    return matched_gt, matched_det


def pair_false_positives(
    baseline: Sequence[DetectionRecord],
    best: Sequence[DetectionRecord],
    *,
    iou_threshold: float,
) -> tuple[list[tuple[DetectionRecord, DetectionRecord]], list[DetectionRecord],
           list[DetectionRecord]]:
    """Decide which false positives the two arms share.

    A baseline false positive that the best arm also produced in the same place
    is not "fixed" and the best arm's copy is not "new": both arms hallucinated
    the same box, which is `both_wrong`. Pairing is highest-IoU-first so the
    result does not depend on the order the two files happen to be written in.
    """

    candidates: list[tuple[float, int, int]] = []
    for left in baseline:
        for right in best:
            if left.image_id != right.image_id or left.class_name != right.class_name:
                continue
            value = iou_xywh(left.bbox, right.bbox)
            if value >= iou_threshold:
                candidates.append((value, left.index, right.index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    left_by_index = {record.index: record for record in baseline}
    right_by_index = {record.index: record for record in best}
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[tuple[DetectionRecord, DetectionRecord]] = []
    for _, left_index, right_index in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        pairs.append((left_by_index[left_index], right_by_index[right_index]))

    left_only = [record for record in baseline if record.index not in used_left]
    right_only = [record for record in best if record.index not in used_right]
    return pairs, left_only, right_only


# spec: EVAL-15
def four_way_comparison(
    ground_truth: Mapping[str, Any],
    baseline_detections: Sequence[Mapping[str, Any]],
    best_detections: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | None = None,
    iou_threshold: float | None = None,
    score_threshold: float | None = None,
) -> tuple[ComparisonItem, ...]:
    """Classify every ground-truth box and every detection of the two arms.

    Ground-truth boxes both arms found, and images where neither arm fired, do
    not appear: they are agreement, not error. Everything else lands in exactly
    one item, and a shared false positive is emitted once rather than twice.
    """

    if iou_threshold is None:
        iou_threshold = default_iou_threshold(config)
    if score_threshold is None:
        score_threshold = default_score_threshold(config)

    gt_records = ground_truth_records(ground_truth)
    baseline_records = detection_records(
        ground_truth, baseline_detections, score_threshold=score_threshold
    )
    best_records = detection_records(
        ground_truth, best_detections, score_threshold=score_threshold
    )

    baseline_hits, baseline_matched = match_arm(
        gt_records, baseline_records, iou_threshold=iou_threshold
    )
    best_hits, best_matched = match_arm(
        gt_records, best_records, iou_threshold=iou_threshold
    )
    baseline_by_index = {record.index: record for record in baseline_records}
    best_by_index = {record.index: record for record in best_records}

    items: list[ComparisonItem] = []
    for record in gt_records:
        found_by_baseline = record.index in baseline_hits
        found_by_best = record.index in best_hits
        if found_by_baseline and found_by_best:
            continue
        if found_by_best:
            category = FIXED_FALSE_NEGATIVE
        elif found_by_baseline:
            category = NEW_FALSE_NEGATIVE
        else:
            category = BOTH_WRONG
        baseline_box = None
        if found_by_baseline:
            hit = baseline_by_index[baseline_hits[record.index]]
            baseline_box = Box(hit.bbox, hit.class_name, hit.score)
        best_box = None
        if found_by_best:
            hit = best_by_index[best_hits[record.index]]
            best_box = Box(hit.bbox, hit.class_name, hit.score)
        items.append(
            ComparisonItem(
                category=category,
                image_id=record.image_id,
                class_name=record.class_name,
                anchor=ANCHOR_GROUND_TRUTH,
                anchor_box=record.bbox,
                baseline=baseline_box,
                best=best_box,
            )
        )

    baseline_fps = [r for r in baseline_records if r.index not in baseline_matched]
    best_fps = [r for r in best_records if r.index not in best_matched]
    shared, baseline_only, best_only = pair_false_positives(
        baseline_fps, best_fps, iou_threshold=iou_threshold
    )
    for left, right in shared:
        items.append(
            ComparisonItem(
                category=BOTH_WRONG,
                image_id=left.image_id,
                class_name=left.class_name,
                anchor=ANCHOR_DETECTION,
                anchor_box=left.bbox,
                baseline=Box(left.bbox, left.class_name, left.score),
                best=Box(right.bbox, right.class_name, right.score),
            )
        )
    for record in baseline_only:
        items.append(
            ComparisonItem(
                category=FIXED_FALSE_POSITIVE,
                image_id=record.image_id,
                class_name=record.class_name,
                anchor=ANCHOR_DETECTION,
                anchor_box=record.bbox,
                baseline=Box(record.bbox, record.class_name, record.score),
                best=None,
            )
        )
    for record in best_only:
        items.append(
            ComparisonItem(
                category=NEW_FALSE_POSITIVE,
                image_id=record.image_id,
                class_name=record.class_name,
                anchor=ANCHOR_DETECTION,
                anchor_box=record.bbox,
                baseline=None,
                best=Box(record.bbox, record.class_name, record.score),
            )
        )
    return tuple(items)


# spec: EVAL-15
def count_by_category(items: Sequence[ComparisonItem]) -> dict[str, int]:
    """Counts over ALL five outcomes, whatever the config asks to render.

    Iterating `OUTCOMES` rather than the items means an outcome that occurred
    zero times still gets a row, and an outcome the config does not render still
    gets a count. Both matter: a missing row reads as "not measured", and a
    dropped outcome would hide a regression.
    """

    counts = dict.fromkeys(OUTCOMES, 0)
    for item in items:
        if item.category not in counts:
            raise ErrorAnalysisConfigError(
                f"Item carries category {item.category!r}, which is not one of "
                f"{list(OUTCOMES)}"
            )
        counts[item.category] += 1
    return counts


# spec: EVAL-15
def count_by_category_and_class(
    items: Sequence[ComparisonItem], class_names: Sequence[str] | None = None
) -> dict[str, dict[str, int]]:
    """Per-outcome, per-class counts. `person` stays visible as its own column."""

    if class_names is None:
        observed = sorted({item.class_name for item in items})
    else:
        observed = list(class_names)
    table = {outcome: dict.fromkeys(observed, 0) for outcome in OUTCOMES}
    for item in items:
        if item.class_name not in table[item.category]:
            table[item.category][item.class_name] = 0
        table[item.category][item.class_name] += 1
    return table


def sample_sort_key(item: ComparisonItem) -> tuple[float, ...]:
    """Deterministic ordering for which examples reach the figure.

    Ground-truth-anchored items come first and rank by ASCENDING area: AP_small
    and bare-head recall are the headline metrics, so the smallest missed object
    is the most on-topic example. Detection-anchored items rank by DESCENDING
    score: a confident spurious box costs more than a marginal one. Ties break
    on image id and position, so the selection is reproducible with no RNG at
    all — the same predictions always produce the same figure.
    """

    is_detection = item.anchor == ANCHOR_DETECTION
    score = item.score_for("best") if item.best is not None else item.score_for("baseline")
    primary = -float(score or 0.0) if is_detection else item.area
    return (
        float(is_detection),
        primary,
        float(item.image_id),
        float(item.anchor_box[0]),
        float(item.anchor_box[1]),
    )


# spec: EVAL-15
def select_samples(
    items: Sequence[ComparisonItem],
    *,
    categories: Sequence[str],
    limit: int,
) -> dict[str, tuple[ComparisonItem, ...]]:
    """Up to `limit` examples per configured category, in a stable order."""

    assert_reportable_categories(categories)
    if limit <= 0:
        raise ErrorAnalysisConfigError(f"samples_per_category must be positive, got {limit}")
    grouped: dict[str, list[ComparisonItem]] = {name: [] for name in categories}
    for item in items:
        if item.category in grouped:
            grouped[item.category].append(item)
    return {
        name: tuple(sorted(bucket, key=sample_sort_key)[:limit])
        for name, bucket in grouped.items()
    }


def _crop_window(
    bbox: Sequence[float], image_size: tuple[int, int]
) -> tuple[int, int, int]:
    """A square window around a box, clamped inside the image. Returns (x, y, side)."""

    x, y, width, height = (float(value) for value in bbox)
    image_width, image_height = image_size
    side = max(max(width, height) * CROP_ZOOM, MIN_CROP_PX)
    side = min(side, float(min(image_width, image_height)))
    centre_x, centre_y = x + width / 2.0, y + height / 2.0
    # Rounded to whole pixels FIRST, then clamped, so the returned window is
    # always fully inside the array. Clamping before rounding can push the far
    # edge one pixel past the image and silently truncate the crop.
    span = max(round(side), 1)
    left = min(max(round(centre_x - span / 2.0), 0), image_width - span)
    top = min(max(round(centre_y - span / 2.0), 0), image_height - span)
    return left, top, span


def _draw_panel(
    axis,
    image: np.ndarray,
    item: ComparisonItem,
    *,
    side: str,
    arm: str,
    context_boxes: Sequence[Box],
) -> None:
    """One cell: the crop, the boxes, and the label EVAL-15 asks for."""

    height, width = image.shape[0], image.shape[1]
    left, top, span = _crop_window(item.anchor_box, (width, height))
    axis.imshow(
        image[top : top + span, left : left + span],
        interpolation=IMSHOW_INTERPOLATION,
    )
    axis.set_xticks([])
    axis.set_yticks([])

    for box in context_boxes:
        bx, by, bw, bh = box.xywh
        if bx + bw <= left or bx >= left + span or by + bh <= top or by >= top + span:
            continue
        axis.add_patch(
            mpatches.Rectangle(
                (bx - left, by - top),
                bw,
                bh,
                fill=False,
                edgecolor=GT_COLOUR,
                linewidth=1.0,
                linestyle=(0, (3, 2)),
            )
        )

    drawn = item.baseline if side == "baseline" else item.best
    if drawn is not None:
        bx, by, bw, bh = drawn.xywh
        axis.add_patch(
            mpatches.Rectangle(
                (bx - left, by - top),
                bw,
                bh,
                fill=False,
                edgecolor=DETECTION_COLOUR,
                linewidth=1.4,
            )
        )
    score = None if drawn is None else drawn.score
    score_text = "no detection" if score is None else f"score {score:.2f}"
    # Three short lines rather than two long ones: EVAL-15 wants the image id,
    # the class, the score AND the category on every cell, and that does not fit
    # across a thumbnail on one or two lines.
    axis.set_title(
        f"{arm}\nimg {item.image_id} · {item.class_name}\n{score_text} · {item.category}",
        fontsize=LABEL_FONT_SIZE,
        pad=2.0,
    )


# spec: EVAL-15
def render_comparison_grid(
    category: str,
    items: Sequence[ComparisonItem],
    *,
    image_paths: Mapping[int, Path],
    destination: Path,
    baseline_arm: str,
    best_arm: str,
    ground_truth_boxes: Mapping[int, Sequence[Box]] | None = None,
    total_in_category: int | None = None,
) -> Path:
    """One PNG per category: baseline and best side by side, one pair per item."""

    if not items:
        raise FigureInputError(
            f"No items to draw for category {category!r}. Call this only for categories "
            "that actually occurred; an empty grid is indistinguishable from a grid "
            "that failed to render."
        )
    missing = sorted({item.image_id for item in items} - set(image_paths))
    if missing:
        raise FigureInputError(
            f"No image path for image id(s) {missing}. A blank panel would read as "
            "'the model saw nothing there', so this raises instead."
        )

    per_row = min(ITEMS_PER_ROW, len(items))
    rows = -(-len(items) // per_row)
    figure, axes = plt.subplots(
        rows,
        2 * per_row,
        figsize=(CELL_INCHES * 2 * per_row, CELL_INCHES * 1.38 * rows),
        squeeze=False,
    )
    for axis in axes.ravel():
        axis.axis("off")

    cache: dict[int, np.ndarray] = {}
    for position, item in enumerate(items):
        if item.image_id not in cache:
            with Image.open(image_paths[item.image_id]) as handle:
                cache[item.image_id] = np.asarray(handle.convert("RGB"))
        image = cache[item.image_id]
        context = list((ground_truth_boxes or {}).get(item.image_id, ()))
        row, column = divmod(position, per_row)
        for offset, (side, arm) in enumerate(
            (("baseline", baseline_arm), ("best", best_arm))
        ):
            axis = axes[row][2 * column + offset]
            axis.axis("on")
            _draw_panel(axis, image, item, side=side, arm=arm, context_boxes=context)

    shown = len(items)
    total = shown if total_in_category is None else total_in_category
    subtitle = f"{shown} of {total} shown · left: {baseline_arm} · right: {best_arm}"
    if category in MANDATORY_CATEGORIES:
        subtitle += " · this category is the COST of synthetic data and is never optional"
    figure.suptitle(f"{category} — {subtitle}", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return destination


# spec: EVAL-15
def render_all_grids(
    samples: Mapping[str, Sequence[ComparisonItem]],
    *,
    image_paths: Mapping[int, Path],
    output_dir: Path,
    baseline_arm: str,
    best_arm: str,
    ground_truth_boxes: Mapping[int, Sequence[Box]] | None = None,
    counts: Mapping[str, int] | None = None,
) -> dict[str, str]:
    """Render every configured category. Missing mandatory categories raise."""

    assert_reportable_categories(list(samples))
    output_dir = Path(output_dir)
    figures: dict[str, str] = {}
    for category, items in samples.items():
        if not items:
            continue
        path = render_comparison_grid(
            category,
            items,
            image_paths=image_paths,
            destination=output_dir / f"{category}.png",
            baseline_arm=baseline_arm,
            best_arm=best_arm,
            ground_truth_boxes=ground_truth_boxes,
            total_in_category=None if counts is None else counts.get(category),
        )
        figures[category] = _repo_relative(path)
    return figures


def _repo_relative(path: Path) -> str:
    """Repo-relative posix, which is what the JSON payload should carry."""

    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _markdown_link(stored: str, report_path: Path) -> str:
    """Rewrite a repo-relative figure path so the LINK resolves from the report.

    Markdown resolves a relative link against the directory of the file it is
    written in, not against the repo root. `reports/figures/x.png` inside
    `reports/error_analysis.md` therefore points at `reports/reports/figures/x.png`
    and renders as a broken image on GitHub. The stored value stays repo-relative
    for the JSON payload; only the link is rewritten.
    """

    candidate = Path(stored)
    absolute = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    try:
        return Path(os.path.relpath(absolute, Path(report_path).parent)).as_posix()
    except ValueError:
        # Different drives on Windows: no relative path exists at all.
        return absolute.as_posix()


# spec: EVAL-16
def hard_negative_image_ids(
    ground_truth: Mapping[str, Any], *, config: Mapping[str, Any] | None = None
) -> tuple[int, ...]:
    """Test images carrying NO primary-class ground truth.

    Derived from the ground truth rather than from a hand-kept list, so an image
    cannot drift into the subset by being forgotten. `person`-only images DO
    qualify: `metrics.primary_classes` excludes `person` (ADR-003), and a false
    `helmet` on a person-only image is exactly the kind of miss this number is
    for.
    """

    payload = load_evaluation_config() if config is None else config
    primary = set(payload["metrics"]["primary_classes"])
    names = category_name_by_id(ground_truth)
    annotated = {
        int(annotation["image_id"])
        for annotation in ground_truth["annotations"]
        if names[int(annotation["category_id"])] in primary
        and int(annotation.get("iscrowd", 0)) == 0
    }
    return tuple(
        sorted(
            int(image["id"])
            for image in ground_truth["images"]
            if int(image["id"]) not in annotated
        )
    )


# spec: EVAL-16
def hard_negative_rows(
    ground_truth: Mapping[str, Any],
    detections_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    image_ids: Sequence[int] | None = None,
    config: Mapping[str, Any] | None = None,
    score_threshold: float | None = None,
) -> tuple[HardNegativeRow, ...]:
    """Per-arm false positives per image on the hard-negative subset.

    The arithmetic is `detection.hard_negative_false_positives_per_image`, not a
    copy of it: the number here and the number in the main table have to be the
    same computation or the report contradicts itself.

    A DERIVED subset that comes back empty returns no rows. EVAL-16 allows for
    exactly that ("if Test has no natural hard negatives..."), and on this
    project's frozen split it is what happens: every Test image carries at least
    one `helmet` or `head`. The report says so rather than reporting a rate over
    zero images. An EXPLICIT empty `image_ids` is a different thing — a caller
    that built the wrong subset — and is left to raise.
    """

    if image_ids is None:
        subset = hard_negative_image_ids(ground_truth, config=config)
        if not subset:
            return ()
    else:
        subset = tuple(int(value) for value in image_ids)
    rows: list[HardNegativeRow] = []
    for arm in sorted(detections_by_arm):
        report: HardNegativeReport = hard_negative_false_positives_per_image(
            ground_truth,
            detections_by_arm[arm],
            subset,
            config=config,
            score_threshold=score_threshold,
        )
        rows.append(
            HardNegativeRow(
                arm=arm,
                false_positives_per_image=report.false_positives_per_image,
                n_false_positives=report.n_false_positives,
                n_images=report.n_images,
                score_threshold=report.score_threshold,
            )
        )
    return tuple(rows)


def subset_ground_truth(
    ground_truth: Mapping[str, Any], image_ids: Iterable[int]
) -> dict[str, Any]:
    """The COCO dict restricted to a set of image ids. Categories are preserved."""

    wanted = {int(value) for value in image_ids}
    return {
        "images": [
            dict(image) for image in ground_truth["images"] if int(image["id"]) in wanted
        ],
        "annotations": [
            dict(annotation)
            for annotation in ground_truth["annotations"]
            if int(annotation["image_id"]) in wanted
        ],
        # Kept whole on purpose: dropping a class that happens to be absent from
        # this slice would renumber the categories and change what AP averages
        # over, so a slice would silently measure a different metric.
        "categories": [dict(category) for category in ground_truth["categories"]],
    }


def subset_detections(
    detections: Sequence[Mapping[str, Any]], image_ids: Iterable[int]
) -> list[dict[str, Any]]:
    wanted = {int(value) for value in image_ids}
    return [dict(d) for d in detections if int(d["image_id"]) in wanted]


def slice_metric_value(metrics: DetectionMetrics, metric: str) -> float:
    """Pull one named metric off a DetectionMetrics bundle."""

    if metric == "bare_head_recall":
        return float(metrics.bare_head.recall)
    if metric in {"primary_map", "primary_map_small"}:
        return float(getattr(metrics, metric))
    raise ErrorAnalysisConfigError(
        f"No slice recipe for metric {metric!r}; known metrics are {list(SLICE_METRICS)}"
    )


# spec: EVAL-17
def slice_metric_table(
    ground_truth: Mapping[str, Any],
    detections_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    slices: Mapping[str, frozenset[int]],
    *,
    metric: str = DEFAULT_SLICE_METRIC,
    config: Mapping[str, Any] | None = None,
) -> tuple[SliceMetric, ...]:
    """Every arm's metric on every slice, with the slice's own size beside it.

    An empty slice yields UNDEFINED rather than 0.0: "no images in this bucket"
    and "scored zero on this bucket" are different statements and the report
    must not blur them.
    """

    rows: list[SliceMetric] = []
    for slice_name in sorted(slices):
        members = sorted(slices[slice_name])
        sub_gt = subset_ground_truth(ground_truth, members)
        n_instances = len(sub_gt["annotations"])
        for arm in sorted(detections_by_arm):
            if not members:
                value = UNDEFINED
            else:
                sub_dt = subset_detections(detections_by_arm[arm], members)
                metrics = evaluate_detection_metrics(sub_gt, sub_dt, config=config)
                value = slice_metric_value(metrics, metric)
            rows.append(
                SliceMetric(
                    slice_name=slice_name,
                    arm=arm,
                    metric=metric,
                    value=value,
                    n_images=len(members),
                    n_instances=n_instances,
                )
            )
    return tuple(rows)


# spec: EVAL-17
def slice_deltas(
    rows: Sequence[SliceMetric], *, baseline_arm: str, best_arm: str
) -> dict[str, float]:
    """best − baseline per slice, skipping slices where either side is undefined."""

    by_slice: dict[str, dict[str, SliceMetric]] = defaultdict(dict)
    for row in rows:
        by_slice[row.slice_name][row.arm] = row
    deltas: dict[str, float] = {}
    for slice_name, arms in sorted(by_slice.items()):
        baseline = arms.get(baseline_arm)
        best = arms.get(best_arm)
        if baseline is None or best is None:
            continue
        if not baseline.is_defined or not best.is_defined:
            continue
        deltas[slice_name] = best.value - baseline.value
    return deltas


# spec: EVAL-17
def targeting_verdict(
    budget_shares: Mapping[str, float],
    deltas: Mapping[str, float],
    *,
    metric: str = DEFAULT_SLICE_METRIC,
    scenario_slice: Mapping[str, str] = SCENARIO_SLICE,
) -> TargetingVerdict:
    """Did the scenario that took the largest budget share move its own slice?

    The rule, in full, so nobody has to reconstruct it from prose:

    1. the scenario with the largest delivered share is the budget leader;
    2. the scenario TESTED is the largest-share one an EVAL-17 slice actually
       isolates, which is not always the leader — on this project's delivered
       mix `head_no_helmet` leads at 22.1% with no slice, while `small_distant`
       sits 0.4 points behind it with one. Hinging the whole instrument on that
       gap would silence it for no good reason, so the fallback runs and the
       sentence states plainly which scenario was tested;
    3. if NO scenario has a slice at all, targeting cannot be tested here and
       the verdict says so rather than inventing an answer;
    4. if the tested scenario's slice has no measurable delta, likewise;
    5. if the slice did not improve at all (delta <= 0) the verdict is
       `target_slice_did_not_improve`;
    6. if it improved STRICTLY MORE than every other slice, `targeted`;
    7. otherwise `more_data_not_targeted`, and the sentence contains the exact
       words EVAL-17 requires.

    Step 7 is the default, not the exception: an equal improvement across all
    slices is precisely "more data", so the comparison at step 6 is strict.
    """

    if not budget_shares:
        raise ErrorAnalysisConfigError(
            "No synthetic budget shares supplied; there is no 'largest share' to test "
            "the targeting claim against"
        )
    # sorted() first so a tie in share resolves alphabetically rather than on
    # dict insertion order; the verdict has to be reproducible.
    leader, leader_share = max(sorted(budget_shares.items()), key=lambda item: item[1])
    sliceable = {
        name: share for name, share in budget_shares.items() if name in scenario_slice
    }

    if not sliceable:
        sentence = (
            f"The largest share of the synthetic budget went to `{leader}` "
            f"({leader_share:.1%}), and no scenario in the delivered mix is isolated by "
            "an EVAL-17 slice at all. Targeting cannot be confirmed or refuted from "
            "these slices, and this report does not claim it either way."
        )
        return TargetingVerdict(
            verdict=BUDGET_LEADER_HAS_NO_SLICE,
            metric=metric,
            budget_leader=leader,
            budget_leader_share=float(leader_share),
            tested_scenario=None,
            tested_share=None,
            target_slice=None,
            target_delta=None,
            other_deltas={},
            sentence=sentence,
        )

    tested, tested_share = max(sorted(sliceable.items()), key=lambda item: item[1])
    target = scenario_slice[tested]
    if tested == leader:
        preamble = (
            f"`{tested}` took the largest share of the synthetic budget "
            f"({tested_share:.1%}) and its slice is `{target}`"
        )
    else:
        preamble = (
            f"The largest share overall went to `{leader}` ({leader_share:.1%}), which no "
            f"EVAL-17 slice isolates. The largest share that a slice DOES isolate is "
            f"`{tested}` ({tested_share:.1%}), whose slice is `{target}`, and that is what "
            "is tested here"
        )

    others = {name: value for name, value in deltas.items() if name != target}
    if target not in deltas:
        sentence = (
            f"{preamble}. But `{metric}` is not defined on `{target}` — it is empty, or "
            "the metric does not exist on it. The targeting claim is untested, not "
            "supported."
        )
        return TargetingVerdict(
            verdict=TARGET_SLICE_UNMEASURABLE,
            metric=metric,
            budget_leader=leader,
            budget_leader_share=float(leader_share),
            tested_scenario=tested,
            tested_share=float(tested_share),
            target_slice=target,
            target_delta=None,
            other_deltas=others,
            sentence=sentence,
        )

    target_delta = float(deltas[target])
    best_other = max(others.values()) if others else None
    lead = f"{preamble}. On `{metric}`, the best arm moves that slice by {target_delta:+.4f}"
    if best_other is None:
        others_text = "; there is no other slice to compare it against"
    else:
        breakdown = ", ".join(
            f"`{name}` {value:+.4f}" for name, value in sorted(others.items())
        )
        others_text = (
            f"; the best any other slice moved is {best_other:+.4f} ({breakdown})"
        )

    if target_delta <= 0.0:
        verdict = TARGET_SLICE_DID_NOT_IMPROVE
        sentence = (
            f"{lead}{others_text}. The slice the budget was aimed at did not improve at "
            f"all, so the targeting claim fails on its own terms: whatever the synthetic "
            f"images did, they did not move `{target}`."
        )
    elif best_other is None or target_delta > best_other:
        verdict = TARGETED
        sentence = (
            f"{lead}{others_text}. The targeted slice improved strictly more than every "
            "other slice, which is what a targeting claim predicts."
        )
    else:
        verdict = MORE_DATA_NOT_TARGETED
        sentence = (
            f"{lead}{others_text}. The targeted slice improved no more than the others, "
            f"so the honest conclusion is {MORE_DATA_SENTENCE} — the gain, such as it is, "
            "is not evidence that scenario targeting worked."
        )

    return TargetingVerdict(
        verdict=verdict,
        metric=metric,
        budget_leader=leader,
        budget_leader_share=float(leader_share),
        tested_scenario=tested,
        tested_share=float(tested_share),
        target_slice=target,
        target_delta=target_delta,
        other_deltas=others,
        sentence=sentence,
    )


# spec: EVAL-17
def scenario_shares_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    keep_names: Iterable[str] | None = None,
) -> dict[str, float]:
    """Delivered share of each scenario, from the synthesis provenance records.

    `keep_names` restricts the count to the images an arm actually trained on,
    because the GENERATED mix and the DELIVERED mix differ: configs/compose.yaml
    documents per-scenario filter pass rates that are far from equal, so reading
    the generation weights as the budget would attribute the budget to the wrong
    scenario.
    """

    allowed = None if keep_names is None else {Path(str(n)).name for n in keep_names}
    counts: dict[str, int] = defaultdict(int)
    total = 0
    for record in records:
        name = Path(str(record["file_name"])).name
        if allowed is not None and name not in allowed:
            continue
        counts[str(record["scenario"])] += 1
        total += 1
    if not total:
        raise ErrorAnalysisConfigError(
            "No synthesis records matched; the delivered scenario mix cannot be computed "
            "and the EVAL-17 targeting question has no denominator"
        )
    return {scenario: count / total for scenario, count in sorted(counts.items())}


def read_records_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read records.jsonl with UTF-8 forced (K-10)."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def placeholder_compositions(
    counts: Mapping[str, Mapping[str, int]],
) -> dict[str, ArmComposition]:
    """Rebuild ArmComposition objects from image COUNTS alone.

    `equal_step_budget` reads only `n_train_images` and `len(real_train)`, so the
    exposure arithmetic can be recomputed from a run summary without the 3,500
    file names being present. The names produced here are placeholders and must
    never be used to decide what an arm trains on — that is
    `src.training.arms.build_all_arms`'s job, from the frozen manifest.
    """

    compositions: dict[str, ArmComposition] = {}
    for arm, entry in counts.items():
        n_real = int(entry["n_real_train"])
        n_synthetic = int(entry.get("n_synthetic", 0))
        if n_real <= 0:
            raise ErrorAnalysisConfigError(
                f"Arm {arm!r} reports {n_real} real training images; the exposure "
                "arithmetic divides by that count"
            )
        compositions[arm] = ArmComposition(
            arm=arm,
            real_train=tuple(f"placeholder_real_{index:06d}.png" for index in range(n_real)),
            real_val=(),
            synthetic=tuple(
                f"placeholder_{arm}_syn_{index:06d}.png" for index in range(n_synthetic)
            ),
            augmentation_profile="placeholder",
        )
    return compositions


# spec: EVAL-18
def exposure_confound(
    compositions: Mapping[str, ArmComposition],
    *,
    reference_arm: str,
    reference_epochs: int,
    batch_size: int,
) -> ExposureConfound:
    """Real-image exposures per arm, computed by `arms.equal_step_budget`.

    Delegated rather than reproduced. TRAIN-07 holds optimizer steps equal, so
    an arm carrying twice the data sees each real image half as often; that
    number is a property of the budget function and hardcoding it here would let
    the report and the training plan drift apart without either one raising.
    """

    plan = equal_step_budget(
        compositions,
        reference_arm=reference_arm,
        reference_epochs=reference_epochs,
        batch_size=batch_size,
    )
    return ExposureConfound(
        exposures={arm: float(entry["real_image_exposures"]) for arm, entry in plan.items()},
        total_steps=int(plan[reference_arm]["total_steps"]),
        reference_arm=reference_arm,
        reference_epochs=int(reference_epochs),
        batch_size=int(batch_size),
    )


# spec: EVAL-18
def negative_result_verdict(
    arm_values: Mapping[str, float],
    *,
    metric: str,
    baseline_arm: str,
    synthetic_arms: Sequence[str],
) -> NegativeResultVerdict:
    """Did any synthetic arm beat the baseline?

    A tie is not a win. The comparison is strict, because the whole point of the
    synthetic arms is to be better than the baseline, and "as good as" bought
    with a generation pipeline is a negative result.
    """

    if baseline_arm not in arm_values:
        raise ErrorAnalysisConfigError(
            f"Baseline arm {baseline_arm!r} has no {metric} value; "
            f"got {sorted(arm_values)}"
        )
    missing = [arm for arm in synthetic_arms if arm not in arm_values]
    if missing:
        raise ErrorAnalysisConfigError(
            f"Synthetic arm(s) {missing} have no {metric} value. Dropping an arm from "
            "this comparison is how a losing arm disappears from the report."
        )
    baseline_value = float(arm_values[baseline_arm])
    winners = tuple(
        arm for arm in synthetic_arms if float(arm_values[arm]) > baseline_value
    )
    return NegativeResultVerdict(
        metric=metric,
        baseline_arm=baseline_arm,
        baseline_value=baseline_value,
        arm_values={arm: float(value) for arm, value in arm_values.items()},
        synthetic_arms=tuple(synthetic_arms),
        winners=winners,
    )


# spec: EVAL-18
def experiment_protocol_checklist() -> tuple[ChecklistItem, ...]:
    """The docs/experiment_protocol.md §7 walk, plus the exposure confound.

    Evidence is a POINTER, never a copied number: a number pasted here would go
    stale the moment its source is regenerated, and EVAL-12 requires every figure
    in a report to be re-derivable from the file that produced it.
    """

    return (
        ChecklistItem(
            key="paste_artifact_detectability",
            question=(
                "Did the H4 paste-artifact gate actually pass, or is the synthetic data "
                "trivially separable from real data?"
            ),
            evidence="reports/h4_artifact_gate_m13.md and ADR-011 in docs/decisions.md",
        ),
        ChecklistItem(
            key="scenario_coverage",
            question=(
                "Did the delivered synthetic mix land in the target scenarios, or did "
                "per-scenario filter pass rates move it somewhere else?"
            ),
            evidence=(
                "reports/synthetic_stats.md cross-tab, and the delivered shares in the "
                "EVAL-17 section of this report"
            ),
        ),
        ChecklistItem(
            key="filter_threshold_dominance",
            question="Is one filter threshold doing all of the work (FILT-14)?",
            evidence="reports/threshold_sensitivity.md",
        ),
        ChecklistItem(
            key="baseline_augmentation_overlap",
            question=(
                "Does the standard_aug baseline already supply the variation the "
                "synthetic images add (EXP-01)?"
            ),
            evidence=(
                "the standard_aug versus real_only rows of the main table, and "
                "configs/training.yaml `augmentation`"
            ),
        ),
        ChecklistItem(
            key="equal_step_exposure",
            question=(
                "Did the fixed optimizer-step budget cost the synthetic arms real-image "
                "exposure (TRAIN-07)?"
            ),
            evidence=(
                "computed below from src.training.arms.equal_step_budget, not copied "
                "from the run log"
            ),
        ),
    )


def _percent(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def _value_text(value: float) -> str:
    return "n/a" if value <= UNDEFINED else f"{value:.4f}"


def _counts_section(analysis: ErrorAnalysis, report_path: Path) -> list[str]:
    total = sum(analysis.counts.values())
    lines = [
        "## 1. Four-way comparison (EVAL-15)",
        "",
        (
            f"Baseline `{analysis.config.baseline_arm}` against best "
            f"`{analysis.config.best_arm}`, at IoU {analysis.iou_threshold:g} and score "
            f"threshold {analysis.score_threshold:g}. Ground-truth boxes both arms found, "
            "and images where neither arm fired, are agreement and do not appear below."
        ),
        "",
        "| category | count | % of disagreements | rendered |",
        "|---|---:|---:|---|",
    ]
    for outcome in OUTCOMES:
        count = analysis.counts[outcome]
        figure = analysis.figures.get(outcome)
        if figure:
            rendered = f"[figure]({_markdown_link(figure, report_path)})"
        elif outcome not in analysis.config.categories:
            rendered = "counted only (not a configured category)"
        elif not analysis.samples.get(outcome):
            rendered = "no examples occurred"
        else:
            rendered = "not rendered (no images supplied)"
        lines.append(
            f"| `{outcome}` | {count:,} | {_percent(count, total)} | {rendered} |"
        )
    lines += [
        f"| **total** | {total:,} | 100.0% | |",
        "",
        (
            "`new_false_positive` is the cost of synthetic data and cannot be omitted: "
            "`assert_reportable_categories` refuses to render any report whose category "
            "list drops it."
        ),
        "",
        (
            "`new_false_negative` — the baseline found the box and the best arm lost it — "
            "is not one of the four categories in `configs/evaluation.yaml`, because it is "
            "not 'neither got it right'. It is counted here anyway. It is the recall-side "
            "twin of `new_false_positive`, and leaving it out would put a hole exactly "
            "where a synthetic-data regression falls."
        ),
        "",
    ]

    # An outcome with no grid is easy to skim past, and on this project the
    # largest category is one of them. Say how big it is, in the table's own
    # units, immediately under the table.
    unrendered = [
        (outcome, analysis.counts[outcome])
        for outcome in OUTCOMES
        if analysis.counts[outcome] and outcome not in analysis.figures
    ]
    if unrendered:
        biggest = max(unrendered, key=lambda item: item[1])
        listed = ", ".join(
            f"`{outcome}` ({count:,}, {_percent(count, total)})"
            for outcome, count in unrendered
        )
        emphasis = (
            f"That is {_percent(biggest[1], total)} of all disagreements"
            if len(unrendered) == 1
            else (
                f"The largest of those, `{biggest[0]}`, is "
                f"{_percent(biggest[1], total)} of all disagreements"
            )
        )
        lines += [
            (
                f"**No comparison grid is rendered for {listed}.** {emphasis} — which can "
                "exceed categories that do get a figure. Which categories are rendered is "
                "fixed by `error_analysis.categories` in `configs/evaluation.yaml`; adding "
                "one is a config change, not a code change, and the counts above already "
                "hold the numbers either way."
            ),
            "",
        ]

    classes = sorted(
        {name for by_class in analysis.counts_by_class.values() for name in by_class}
    )
    if classes:
        lines += [
            "### By class",
            "",
            "| category | " + " | ".join(f"`{name}`" for name in classes) + " |",
            "|---|" + "---:|" * len(classes),
        ]
        for outcome in OUTCOMES:
            row = analysis.counts_by_class.get(outcome, {})
            cells = " | ".join(f"{row.get(name, 0):,}" for name in classes)
            lines.append(f"| `{outcome}` | {cells} |")
        lines.append("")

    for category in analysis.config.categories:
        figure = analysis.figures.get(category)
        if not figure:
            continue
        shown = len(analysis.samples.get(category, ()))
        lines += [
            f"### `{category}`",
            "",
            (
                f"{shown} of {analysis.counts[category]:,} shown. Left panel "
                f"`{analysis.config.baseline_arm}`, right panel "
                f"`{analysis.config.best_arm}`. Dashed blue is ground truth, solid orange "
                "is that arm's detection."
            ),
            "",
            f"![{category}]({_markdown_link(figure, report_path)})",
            "",
        ]
    return lines


def _hard_negative_section(analysis: ErrorAnalysis) -> list[str]:
    lines = [
        "## 2. Hard-negative subset (EVAL-16)",
        "",
        (
            "Test images with no `helmet` and no `head` ground truth. A hard negative "
            "cannot contribute recall, only precision, so diluting it into mAP over the "
            "whole split hides the effect it exists to produce."
        ),
        "",
    ]
    if not analysis.hard_negatives:
        lines += [
            (
                "**This Test split has no natural hard negatives.** Every frozen Test "
                "image carries at least one primary-class box, so there is no subset on "
                "which a false positive can be counted without a true positive nearby, "
                "and no per-image rate is reported here."
            ),
            "",
            (
                "EVAL-16 anticipates this: the fallback is to mark candidate regions on "
                "Test with the cutout miner **for analysis only, never for training**, "
                "and measure the rate on those. Until that subset exists, the "
                "`hard_negative` synthesis scenario has no direct verification on Test, "
                "and its share of the budget is spend whose effect this report cannot "
                "confirm."
            ),
            "",
        ]
        return lines
    first = analysis.hard_negatives[0]
    lines += [
        (
            f"Subset: **{first.n_images:,} images**, score threshold "
            f"{first.score_threshold:g}."
        ),
        "",
        "| arm | false positives | per image |",
        "|---|---:|---:|",
    ]
    for row in sorted(analysis.hard_negatives, key=lambda r: r.false_positives_per_image):
        lines.append(
            f"| `{row.arm}` | {row.n_false_positives:,} | "
            f"{row.false_positives_per_image:.4f} |"
        )
    lines.append("")
    return lines


def _slice_section(analysis: ErrorAnalysis) -> list[str]:
    lines = [
        "## 3. Scenario slices (EVAL-17)",
        "",
        (
            "Slices come from `src/evaluation/slices.py` and are **not** mutually "
            "exclusive. Instance counts sit beside every metric because an AP over "
            "nine instances and an AP over nine hundred read identically (EVAL-08)."
        ),
        "",
    ]
    if not analysis.slice_metrics:
        lines += ["No slice metrics were computed.", ""]
        return lines

    arms = sorted({row.arm for row in analysis.slice_metrics})
    metric = analysis.slice_metrics[0].metric
    lines += [
        f"Metric: `{metric}`.",
        "",
        "| slice | images | instances | " + " | ".join(f"`{arm}`" for arm in arms) + " |",
        "|---|---:|---:|" + "---:|" * len(arms),
    ]
    by_slice: dict[str, dict[str, SliceMetric]] = defaultdict(dict)
    for row in analysis.slice_metrics:
        by_slice[row.slice_name][row.arm] = row
    for slice_name, rows in sorted(by_slice.items()):
        any_row = next(iter(rows.values()))
        cells = " | ".join(
            _value_text(rows[arm].value) if arm in rows else "n/a" for arm in arms
        )
        lines.append(
            f"| `{slice_name}` | {any_row.n_images:,} | {any_row.n_instances:,} | "
            f"{cells} |"
        )
    lines.append("")

    if analysis.slice_deltas:
        lines += [
            (
                f"Change from `{analysis.config.baseline_arm}` to "
                f"`{analysis.config.best_arm}`:"
            ),
            "",
            "| slice | delta |",
            "|---|---:|",
        ]
        for slice_name, delta in sorted(analysis.slice_deltas.items()):
            lines.append(f"| `{slice_name}` | {delta:+.4f} |")
        lines.append("")

    if analysis.budget_shares:
        lines += [
            "Delivered synthetic budget, by scenario:",
            "",
            "| scenario | share of delivered images | slice it targets |",
            "|---|---:|---|",
        ]
        for scenario, share in sorted(
            analysis.budget_shares.items(), key=lambda item: (-item[1], item[0])
        ):
            target = SCENARIO_SLICE.get(scenario)
            lines.append(
                f"| `{scenario}` | {share:.1%} | "
                f"{'`' + target + '`' if target else 'no slice isolates it'} |"
            )
        lines.append("")

    if analysis.targeting is not None:
        lines += [
            "### Was the targeting real?",
            "",
            f"**Verdict: `{analysis.targeting.verdict}`.** {analysis.targeting.sentence}",
            "",
        ]
    return lines


def _negative_result_section(analysis: ErrorAnalysis) -> list[str]:
    lines = ["## 4. If synthetic did not help (EVAL-18)", ""]
    verdict = analysis.negative_result
    if verdict is None:
        lines += [
            (
                "No headline comparison was supplied, so this section cannot state "
                "whether the synthetic arms won or lost. That is a missing input, not a "
                "positive result."
            ),
            "",
        ]
        return lines

    lines += [
        f"Headline metric: `{verdict.metric}`.",
        "",
        "| arm | value | vs baseline |",
        "|---|---:|---:|",
    ]
    for arm, value in sorted(verdict.arm_values.items(), key=lambda item: -item[1]):
        delta = value - verdict.baseline_value
        against = "baseline" if arm == verdict.baseline_arm else f"{delta:+.4f}"
        lines.append(f"| `{arm}` | {value:.4f} | {against} |")
    lines.append("")

    if not verdict.is_negative:
        lines += [
            (
                f"At least one synthetic arm beats `{verdict.baseline_arm}`: "
                + ", ".join(f"`{arm}`" for arm in verdict.winners)
                + ". The EVAL-18 checklist below is still reproduced, because a win that "
                "survives none of these questions is not a win worth reporting."
            ),
            "",
        ]
    else:
        lines += [
            (
                f"**No synthetic arm beats `{verdict.baseline_arm}` on `{verdict.metric}`.** "
                "This is the negative-result path. Per docs/experiment_protocol.md §7 the "
                "result is reported as it is and the following checklist is walked; "
                "CLAUDE.md forbids hiding it or quietly reporting a different metric."
            ),
            "",
        ]

    lines += ["| # | question | where the evidence is |", "|---:|---|---|"]
    for number, item in enumerate(experiment_protocol_checklist(), start=1):
        lines.append(f"| {number} | {item.question} | {item.evidence} |")
    lines.append("")

    exposure = analysis.exposure
    if exposure is None:
        lines += [
            (
                "The equal-step exposure confound could not be computed: no arm "
                "composition was supplied. It is not absent because it does not apply."
            ),
            "",
        ]
        return lines

    baseline_arm = verdict.baseline_arm
    lines += [
        "### The equal-step exposure confound",
        "",
        (
            f"TRAIN-07 fixes the optimizer-step budget at **{exposure.total_steps:,} "
            f"steps** for every arm (batch size {exposure.batch_size}, "
            f"{exposure.reference_epochs} epochs for `{exposure.reference_arm}`). An arm "
            "carrying synthetic data therefore covers fewer epochs and sees each REAL "
            "image fewer times. Recomputed here by "
            "`src.training.arms.equal_step_budget`, not copied from the run log:"
        ),
        "",
        "| arm | real-image exposures | relative to baseline |",
        "|---|---:|---:|",
    ]
    for arm, value in sorted(exposure.exposures.items(), key=lambda item: -item[1]):
        if baseline_arm not in exposure.exposures:
            relative = "baseline arm absent from the budget plan"
        elif arm == baseline_arm:
            relative = "baseline"
        else:
            relative = f"{exposure.ratio(arm, baseline_arm):.2f}x"
        lines.append(f"| `{arm}` | {value:.2f} | {relative} |")
    lines += [
        "",
        (
            "This is a confound, not a footnote. The synthetic arms are not only "
            "different in what they trained on; they are also different in how often "
            "they saw the real images the Test set resembles. A loss under this budget "
            "does not by itself establish that the synthetic images are harmful, and a "
            "win would have been the stronger claim precisely because it came with less "
            "real-image exposure."
        ),
        "",
    ]
    return lines


def render_markdown(
    analysis: ErrorAnalysis, *, report_path: Path = DEFAULT_REPORT_PATH
) -> str:
    """reports/error_analysis.md. Regenerate, never hand-edit.

    `report_path` is where the markdown will live, and it is needed BEFORE the
    text exists: image links have to be written relative to that directory.
    """

    # Enforced here as well as at load time: a caller can hand-build an
    # ErrorAnalysis, and the rule is about what gets PUBLISHED, not about which
    # function happened to construct the object.
    assert_reportable_categories(analysis.config.categories)
    lines = [
        "# Error analysis",
        "",
        (
            "Generated by `scripts/error_analysis.py`. Implements **EVAL-15** (four-way "
            "FP/FN comparison), **EVAL-16** (hard-negative subset), **EVAL-17** (scenario "
            "slices and the targeting test) and **EVAL-18** (the negative-result path). "
            "Every threshold comes from `configs/evaluation.yaml`; every number here is "
            "also in `reports/error_analysis.json` so the tables can be re-aggregated "
            "(EVAL-12)."
        ),
        "",
        (
            "Standing caveat (ADR-003): roughly two thirds of the real objects in this "
            "dataset are unannotated, so every statement below is a RELATIVE one between "
            "two arms on the same frozen Test split. A box counted as a false positive "
            "here may be a correctly detected object that nobody labelled."
        ),
        "",
    ]
    lines += _counts_section(analysis, Path(report_path))
    lines += _hard_negative_section(analysis)
    lines += _slice_section(analysis)
    lines += _negative_result_section(analysis)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(
    analysis: ErrorAnalysis,
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
    payload_path: Path = DEFAULT_PAYLOAD_PATH,
) -> tuple[Path, Path]:
    """Write the markdown and its JSON spine. Both, always (EVAL-12)."""

    report_path = Path(report_path)
    payload_path = Path(payload_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown(analysis, report_path=report_path), encoding="utf-8", newline="\n"
    )
    payload_path.write_text(
        json.dumps(analysis.as_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report_path, payload_path


def build_analysis(
    ground_truth: Mapping[str, Any],
    detections_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    config: Mapping[str, Any] | None = None,
    analysis_config: ErrorAnalysisConfig | None = None,
    iou_threshold: float | None = None,
    score_threshold: float | None = None,
    slices: Mapping[str, frozenset[int]] | None = None,
    slice_metric: str = DEFAULT_SLICE_METRIC,
    budget_shares: Mapping[str, float] | None = None,
    headline_values: Mapping[str, float] | None = None,
    headline_metric: str = "primary_map",
    exposure: ExposureConfound | None = None,
    image_paths: Mapping[int, Path] | None = None,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> ErrorAnalysis:
    """Assemble EVAL-15..EVAL-18 into one object the renderers consume.

    Every stage is optional except the four-way comparison, and every omission
    is stated in the report rather than skipped silently: a section that says
    "this input was not supplied" is honest, a section that quietly disappears
    is not.
    """

    settings = load_error_analysis_config(config) if analysis_config is None else analysis_config
    assert_reportable_categories(settings.categories)
    for arm in (settings.baseline_arm, settings.best_arm):
        if arm not in detections_by_arm:
            raise ErrorAnalysisConfigError(
                f"error_analysis.compare_arms names {arm!r}, which has no predictions. "
                f"Supplied arms: {sorted(detections_by_arm)}"
            )

    if iou_threshold is None:
        iou_threshold = default_iou_threshold(config)
    if score_threshold is None:
        score_threshold = default_score_threshold(config)

    items = four_way_comparison(
        ground_truth,
        detections_by_arm[settings.baseline_arm],
        detections_by_arm[settings.best_arm],
        config=config,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    counts = count_by_category(items)
    class_names = [str(entry["name"]) for entry in ground_truth["categories"]]
    by_class = count_by_category_and_class(items, class_names)
    samples = select_samples(
        items, categories=settings.categories, limit=settings.samples_per_category
    )

    figures: dict[str, str] = {}
    if image_paths:
        context = defaultdict(list)
        for record in ground_truth_records(ground_truth):
            context[record.image_id].append(Box(record.bbox, record.class_name, None))
        figures = render_all_grids(
            samples,
            image_paths=image_paths,
            output_dir=figure_dir,
            baseline_arm=settings.baseline_arm,
            best_arm=settings.best_arm,
            ground_truth_boxes=context,
            counts=counts,
        )

    hard_negatives = hard_negative_rows(
        ground_truth,
        detections_by_arm,
        config=config,
        score_threshold=score_threshold,
    )

    rows: tuple[SliceMetric, ...] = ()
    deltas: dict[str, float] = {}
    targeting: TargetingVerdict | None = None
    if slices:
        rows = slice_metric_table(
            ground_truth, detections_by_arm, slices, metric=slice_metric, config=config
        )
        deltas = slice_deltas(
            rows, baseline_arm=settings.baseline_arm, best_arm=settings.best_arm
        )
        if budget_shares:
            targeting = targeting_verdict(budget_shares, deltas, metric=slice_metric)

    negative: NegativeResultVerdict | None = None
    if headline_values:
        synthetic = tuple(
            arm
            for arm in sorted(headline_values)
            if arm != settings.baseline_arm and arm.endswith("_syn")
        )
        negative = negative_result_verdict(
            headline_values,
            metric=headline_metric,
            baseline_arm=settings.baseline_arm,
            synthetic_arms=synthetic,
        )

    return ErrorAnalysis(
        config=settings,
        iou_threshold=float(iou_threshold),
        score_threshold=float(score_threshold),
        items=items,
        counts=counts,
        counts_by_class=by_class,
        samples=samples,
        figures=figures,
        hard_negatives=hard_negatives,
        slice_metrics=rows,
        slice_deltas=deltas,
        targeting=targeting,
        negative_result=negative,
        exposure=exposure,
        budget_shares=dict(budget_shares or {}),
    )
