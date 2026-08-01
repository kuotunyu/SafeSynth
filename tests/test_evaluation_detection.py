"""Detection metrics core: every failure mode here is silent, not an exception.

A wrong area, a wrong coordinate system or a wrong iscrowd all produce a
plausible-looking number. So each branch below is executed by a test rather than
reasoned about (K-18), and the two headline metrics are checked against
arithmetic worked out by hand instead of against the code that produces them.
"""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from src.evaluation import detection
from src.evaluation.detection import (
    PROJECT_ROOT,
    BootstrapError,
    EvaluationConfigError,
    HardNegativeSubsetError,
    MetricRow,
    SplitLeakageError,
    UnknownCategoryError,
    area_ranges_from_config,
    assert_no_test_images_in_training_list,
    attach_bootstrap_cis,
    bare_head_recall,
    bootstrap_ci_metric_names,
    bootstrap_metric_ci,
    cross_check_with_faster_coco_eval,
    default_detection_metrics_path,
    detection_metric_rows,
    detections_for_evaluation,
    evaluate_detection_metrics,
    evaluates_in_original_coordinates,
    ground_truth_as_detections,
    hard_negative_false_positives_per_image,
    instance_counts_by_bucket,
    load_evaluation_config,
    read_detection_metrics_csv,
    rescale_detections_to_original,
    write_detection_metrics_csv,
)
from src.training.data import CLASS_NAMES, Sample
from src.training.metrics import COCO_STAT_NAMES, build_coco_ground_truth

CONFIG = load_evaluation_config()
HELMET, HEAD, PERSON = 0, 1, 2

# pycocotools integrates precision over the 101 recall points 0.00, 0.01, ..., 1.00.
# With no false positives the precision is 1.0 at every recall point up to the
# recall actually reached and 0 after it, so AP is just "how many of the 101
# points are covered". Worked out for the three coverages the fixtures below use:
#   1 of 2 found -> recall 0.5000 -> points 0.00..0.50 -> 51 -> AP = 51/101
#   2 of 3 found -> recall 0.6667 -> points 0.00..0.66 -> 67 -> AP = 67/101
#   1 of 3 found -> recall 0.3333 -> points 0.00..0.33 -> 34 -> AP = 34/101
AP_AT_HALF_RECALL = 51 / 101
AP_AT_TWO_THIRDS_RECALL = 67 / 101
AP_AT_ONE_THIRD_RECALL = 34 / 101

_HAS_FASTER_COCO_EVAL = importlib.util.find_spec("faster_coco_eval") is not None

# A flawless COCOeval result is 0.9999999999999998, not 1.0: pycocotools divides
# by `fp + tp + np.spacing(1)`, so the epsilon is baked into every precision
# value. Rounding to three decimals is therefore the strictest assertion that is
# actually true; `== 1.0` fails on a perfectly correct pipeline.
PERFECT = 3


def _is_perfect(value: float) -> bool:
    return round(float(value), PERFECT) == 1.0


def _raw_config() -> dict:
    """Parse configs/evaluation.yaml WITHOUT the module under test.

    Asserting the `config=None` path against `load_evaluation_config()` would be
    the module checking its own output; the claim worth making is that the
    default path uses the values the shipped FILE contains. Rooted off this test
    file rather than off detection.PROJECT_ROOT for the same reason.

    DO NOT DELETE THIS AS REDUNDANT. Several assertions elsewhere in this file
    compare a returned value against `CONFIG[...]`, i.e. against the module's own
    loader: the bare-head IoU threshold, the hard-negative score threshold, the
    two bootstrap defaults, and the `large` bucket ceiling. Each of those is only
    a real claim because this loader independently pins the same key to the
    shipped file somewhere in the two tests below. Remove `_raw_config` and those
    assertions silently degrade into the module agreeing with itself.
    """

    path = Path(__file__).resolve().parents[1] / "configs" / "evaluation.yaml"
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class _ScriptedMetric:
    """A metric that returns a fixed sequence of values, one per call.

    The bootstrap's percentile arithmetic is only checkable by hand if the pooled
    sample is known exactly, and a metric computed from a random resample is not.
    Strict indexing on purpose: if the code calls the metric more or fewer times
    than scripted, the test raises IndexError instead of quietly re-using a value.
    """

    def __init__(self, values) -> None:
        self.values = [float(value) for value in values]
        self.calls = 0

    def __call__(self, _ground_truth, _detections) -> float:
        value = self.values[self.calls]
        self.calls += 1
        return value


def _sample(image_id: int, boxes, classes) -> Sample:
    return Sample(
        image_path=Path(f"{image_id}.png"),
        image_id=image_id,
        boxes_xywh=tuple(tuple(float(v) for v in box) for box in boxes),
        class_indices=tuple(int(c) for c in classes),
        is_synthetic=False,
    )


def _ground_truth(*samples: Sample) -> dict:
    return build_coco_ground_truth(samples, CLASS_NAMES)


def _detection(image_id: int, category: int, box, score: float) -> dict:
    return {
        "image_id": image_id,
        "category_id": category,
        "bbox": [float(v) for v in box],
        "score": float(score),
    }


def test_self_evaluation_of_the_ground_truth_scores_exactly_one() -> None:
    """Five lines that catch xywh-vs-xyxy, wrong area, wrong iscrowd and orphan ids.

    If any of those is wrong the mAP comes back plausible but below 1, which is
    indistinguishable from "the model is imperfect" once real predictions are in
    the file.

    Observed on this case: map = 1.0 but map_small = 0.9999999999999998 and
    primary_map = 0.9999999999999999. The epsilon from pycocotools'
    `tp / (fp + tp + np.spacing(1))` sometimes averages away and sometimes does
    not, depending on how many precision entries get pooled, which is why this
    asserts to three decimals and not to a bit pattern. bare-head recall is
    computed here rather than by COCOeval, so that one IS bit-exact.
    """

    ground_truth = _ground_truth(
        _sample(1, [(10, 20, 30, 40), (100, 100, 25, 40)], [HELMET, HEAD]),
        _sample(2, [(5, 5, 60, 60)], [HEAD]),
    )

    metrics = evaluate_detection_metrics(
        ground_truth, ground_truth_as_detections(ground_truth), config=CONFIG
    )

    assert _is_perfect(metrics.stats["map"])
    assert _is_perfect(metrics.stats["map_50"])
    assert _is_perfect(metrics.primary_map)
    assert _is_perfect(metrics.per_class_ap["helmet"])
    assert metrics.bare_head.recall == 1.0


def test_area_1000_and_area_1100_land_in_different_buckets() -> None:
    """EVAL-07's constructed case, asserted through the metrics pipeline itself.

    25x40 = 1000 px^2 is small, 25x44 = 1100 px^2 is not. Only the small one is
    detected, so map_small must be 1 and map_medium 0. Checking the bucket
    counter alone would not prove COCOeval received the same boundaries.
    """

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 25, 40)], [HELMET]),
        _sample(2, [(0, 0, 25, 44)], [HELMET]),
    )
    detections = [_detection(1, HELMET, (0, 0, 25, 40), 1.0)]

    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)

    assert metrics.counts.total["small"] == 1
    assert metrics.counts.total["medium"] == 1
    assert _is_perfect(metrics.stats["map_small"])
    assert metrics.stats["map_medium"] == 0.0


def test_per_class_ap_small_reads_the_small_bucket_and_not_the_medium_one() -> None:
    """AP_small is the project's #1 narrative metric; nothing else pins its bucket.

    Three `helmet` boxes, split across the two buckets so their APs are different
    known numbers:
      img 1  20x20 =   400 px^2 -> small   detected exactly   -> TP
      img 2  20x20 =   400 px^2 -> small   no detection       -> FN
      img 3  50x50 = 2,500 px^2 -> medium  detected exactly   -> TP
    In the small bucket the medium box and the medium detection are both ignored,
    so it is 1 of 2 found -> AP = 51/101. In the medium bucket the small ones are
    ignored, so it is 1 of 1 -> AP = 1.000. Over all areas it is 2 of 3 -> 67/101.

    Reading the medium bucket instead would report 1.000 as AP_small, i.e. a
    perfect score on the hardest objects in the dataset, with nothing anywhere
    saying otherwise.
    """

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 20, 20)], [HELMET]),
        _sample(2, [(0, 0, 20, 20)], [HELMET]),
        _sample(3, [(0, 0, 50, 50)], [HELMET]),
    )
    detections = [
        _detection(1, HELMET, (0, 0, 20, 20), 0.9),
        _detection(3, HELMET, (0, 0, 50, 50), 0.8),
    ]

    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)

    assert metrics.counts.total["small"] == 2
    assert metrics.counts.total["medium"] == 1
    assert metrics.per_class_ap["helmet"] == pytest.approx(AP_AT_TWO_THIRDS_RECALL)
    assert metrics.per_class_ap_small["helmet"] == pytest.approx(AP_AT_HALF_RECALL)
    assert metrics.primary_map_small == pytest.approx(AP_AT_HALF_RECALL)
    # The medium bucket's AP is a different number, which is what makes the two
    # buckets distinguishable at all.
    assert _is_perfect(metrics.stats["map_medium"])
    assert metrics.stats["map_small"] == pytest.approx(AP_AT_HALF_RECALL)


def test_changing_the_config_actually_changes_the_bucketing() -> None:
    """The area ranges must come from the config, not from pycocotools' defaults."""

    config = copy.deepcopy(CONFIG)
    config["metrics"]["coco_area_ranges"]["small"] = [0, 1200]
    config["metrics"]["coco_area_ranges"]["medium"] = [1200, 9216]

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 25, 40)], [HELMET]),
        _sample(2, [(0, 0, 25, 44)], [HELMET]),
    )
    detections = [_detection(1, HELMET, (0, 0, 25, 40), 1.0)]

    metrics = evaluate_detection_metrics(ground_truth, detections, config=config)

    # Both objects are now small; one of the two is detected.
    assert metrics.counts.total["small"] == 2
    assert metrics.stats["map_small"] == pytest.approx(AP_AT_HALF_RECALL)
    # The medium bucket is empty of ground truth, which COCO reports as -1
    # ("undefined") rather than 0 ("found none of them").
    assert metrics.stats["map_medium"] == -1.0


def test_evaluating_in_640_space_gives_a_different_map_small() -> None:
    """Pins EVAL-07 as behaviour rather than as a comment.

    Annotated at 416, trained at 640. A 31x31 box is 961 px^2 (small) at 416 and
    2274 px^2 (medium) at 640, so the same predictions score differently
    depending only on which coordinate system the evaluator was handed.
    """

    scale = 640 / 416
    small_box = (0.0, 0.0, 20.0, 20.0)      # 400 px^2 at 416, 947 px^2 at 640: small either way
    boundary_box = (0.0, 0.0, 31.0, 31.0)   # 961 px^2 at 416, 2274 px^2 at 640: flips to medium

    def scaled(box):
        return tuple(value * scale for value in box)

    ground_truth_416 = _ground_truth(
        _sample(1, [small_box], [HELMET]), _sample(2, [boundary_box], [HELMET])
    )
    ground_truth_640 = _ground_truth(
        _sample(1, [scaled(small_box)], [HELMET]),
        _sample(2, [scaled(boundary_box)], [HELMET]),
    )
    detections_416 = [_detection(1, HELMET, small_box, 1.0)]
    detections_640 = [_detection(1, HELMET, scaled(small_box), 1.0)]

    metrics_416 = evaluate_detection_metrics(ground_truth_416, detections_416, config=CONFIG)
    metrics_640 = evaluate_detection_metrics(ground_truth_640, detections_640, config=CONFIG)

    assert metrics_416.counts.total["small"] == 2
    assert metrics_416.stats["map_small"] == pytest.approx(AP_AT_HALF_RECALL)
    assert metrics_640.counts.total["small"] == 1
    assert metrics_640.stats["map_small"] == pytest.approx(1.0)
    assert metrics_640.stats["map_small"] != pytest.approx(metrics_416.stats["map_small"])

    restored = rescale_detections_to_original(
        detections_640,
        evaluated_size=(640, 640),
        original_sizes={1: (416, 416), 2: (416, 416)},
    )
    metrics_restored = evaluate_detection_metrics(ground_truth_416, restored, config=CONFIG)
    assert metrics_restored.stats["map_small"] == pytest.approx(AP_AT_HALF_RECALL)


def test_rescaling_refuses_to_guess_a_missing_image_size() -> None:
    with pytest.raises(KeyError, match="No original size"):
        rescale_detections_to_original(
            [_detection(9, HELMET, (0, 0, 10, 10), 1.0)],
            evaluated_size=(640, 640),
            original_sizes={1: (416, 416)},
        )


def test_rescaling_rejects_a_degenerate_evaluated_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        rescale_detections_to_original(
            [], evaluated_size=(0, 640), original_sizes={}
        )


def test_rescaling_scales_each_axis_by_its_own_factor() -> None:
    """A square 640 -> 416 mapping makes scale_x and scale_y interchangeable.

    The frozen Test split is not one resolution: it holds 416x416, 416x415,
    415x416 and 415x415 images, so a swapped axis is live rather than
    hypothetical. Every number below differs between the two axes.

    Image 1: original 415 wide x 416 high, evaluated at 640x640.
        scale_x = 415/640 = 0.6484375        scale_y = 416/640 = 0.65
        (64, 64, 128, 256) -> x 64*0.6484375 = 41.5   y 64*0.65   =  41.6
                              w 128*0.6484375 = 83.0  h 256*0.65  = 166.4
    Image 2: original 416x416, evaluated at a NON-SQUARE 800x640, so the
    evaluated size's own two axes cannot be swapped either.
        scale_x = 416/800 = 0.52             scale_y = 416/640 = 0.65
        (100, 100, 200, 200) -> (52.0, 65.0, 104.0, 130.0)
    """

    first = rescale_detections_to_original(
        [_detection(1, HEAD, (64, 64, 128, 256), 0.9)],
        evaluated_size=(640, 640),
        original_sizes={1: (415, 416)},
    )
    second = rescale_detections_to_original(
        [_detection(2, HEAD, (100, 100, 200, 200), 0.9)],
        evaluated_size=(800, 640),
        original_sizes={2: (416, 416)},
    )

    assert first[0]["bbox"] == pytest.approx([41.5, 41.6, 83.0, 166.4])
    assert second[0]["bbox"] == pytest.approx([52.0, 65.0, 104.0, 130.0])
    # Everything else on the detection rides along untouched.
    assert first[0]["image_id"] == 1
    assert first[0]["score"] == pytest.approx(0.9)


def test_the_coordinate_system_flag_decides_what_gets_measured() -> None:
    """EVAL-07's rule lives in one config key, so the key must drive a branch.

    A 31x31 box is 961 px^2 at 416 (small) and 2,274 px^2 at 640 (medium), so
    with the flag on the detections come back to annotation space and the small
    bucket sees them; with it off they stay in 640-space and land in a different
    bucket entirely. Same predictions, same ground truth, different metric.
    """

    scale = 640 / 416
    box_640 = (0.0, 0.0, 31.0 * scale, 31.0 * scale)
    ground_truth = _ground_truth(_sample(1, [(0.0, 0.0, 31.0, 31.0)], [HELMET]))
    detections = [_detection(1, HELMET, box_640, 0.9)]

    honoured = copy.deepcopy(CONFIG)
    honoured["metrics"]["evaluate_in_original_coordinates"] = True
    ignored = copy.deepcopy(CONFIG)
    ignored["metrics"]["evaluate_in_original_coordinates"] = False

    kwargs = {"evaluated_size": (640, 640), "original_sizes": {1: (416, 416)}}
    mapped = detections_for_evaluation(detections, config=honoured, **kwargs)
    left_alone = detections_for_evaluation(detections, config=ignored, **kwargs)

    assert evaluates_in_original_coordinates(honoured) is True
    assert evaluates_in_original_coordinates(ignored) is False
    assert mapped[0]["bbox"] == pytest.approx([0.0, 0.0, 31.0, 31.0])
    assert left_alone[0]["bbox"] == pytest.approx(list(box_640))

    # The mapped boxes hit the one small ground-truth box; the unmapped ones are
    # 1.54x too large in each dimension, so IoU collapses to 1/1.54^2 = 0.42 and
    # nothing matches at IoU 0.50 or above.
    assert _is_perfect(
        evaluate_detection_metrics(ground_truth, mapped, config=honoured).stats["map_small"]
    )
    assert (
        evaluate_detection_metrics(ground_truth, left_alone, config=ignored).stats["map_small"]
        == 0.0
    )


def test_bare_head_recall_matches_hand_worked_arithmetic() -> None:
    """Four bare heads, three of them recovered at IoU 0.50.

    GT (all `head`): H1 (0,0,10,10)  H2 (20,0,10,10)  H3 (40,0,10,10)  H4 (60,0,10,10)
      D1 head  (0,0,10,10)  s=0.90 -> IoU 1.00 with H1                      -> match
      D2 head  (0,0,10,10)  s=0.85 -> duplicate box on H1, overlaps nothing
                                      else, so it adds no recall            -> no match
      D3 head  (20,0,10,10) s=0.80 -> IoU 1.00 with H2                      -> match
      D4 head  (40,0,10,5)  s=0.60 -> inter 50, union 100 -> IoU 0.50 == thr -> match
      D5 helmet(60,0,10,10) s=0.95 -> wrong class                           -> no match
      D6 head  (60,0,10,2)  s=0.70 -> inter 20, union 100 -> IoU 0.20 < 0.50 -> no match
    matched = {H1, H2, H3} -> recall = 3/4 = 0.75

    D2 does NOT exercise the one-GT-per-detection branch, and an earlier version
    of this docstring claimed it did. `matched` is a set, so re-adding H1 is
    idempotent and the recall is identical whether the branch skips the taken
    box or not. Promoting a detection to its SECOND-best ground truth is the
    branch's only observable effect and it needs a fixture of its own - see
    test_a_detection_whose_best_ground_truth_is_taken_falls_to_its_second_best.
    Likewise D6 sits exactly on the score threshold used below but can never
    match at any threshold, so it does not pin `>=` either; that is
    test_a_matching_detection_scoring_exactly_the_threshold_is_kept.
    """

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 10, 10), (20, 0, 10, 10), (40, 0, 10, 10), (60, 0, 10, 10)],
                [HEAD, HEAD, HEAD, HEAD])
    )
    detections = [
        _detection(1, HEAD, (0, 0, 10, 10), 0.90),
        _detection(1, HEAD, (0, 0, 10, 10), 0.85),
        _detection(1, HEAD, (20, 0, 10, 10), 0.80),
        _detection(1, HEAD, (40, 0, 10, 5), 0.60),
        _detection(1, HELMET, (60, 0, 10, 10), 0.95),
        _detection(1, HEAD, (60, 0, 10, 2), 0.70),
    ]

    result = bare_head_recall(ground_truth, detections, config=CONFIG)

    assert result.n_ground_truth == 4
    assert result.n_matched == 3
    assert result.recall == pytest.approx(0.75)
    assert result.iou_threshold == CONFIG["metrics"]["bare_head_recall_iou"]

    # Raising the operating point above 0.60 drops D4, leaving H1 and H2: 2/4.
    at_operating_point = bare_head_recall(
        ground_truth, detections, config=CONFIG, score_threshold=0.70
    )
    assert at_operating_point.n_matched == 2
    assert at_operating_point.recall == pytest.approx(0.50)


def test_a_detection_whose_best_ground_truth_is_taken_falls_to_its_second_best() -> None:
    """One ground truth per detection is a real branch that a set hides.

    Re-picking an already-matched box adds a duplicate to a SET, so recall does
    not move and the branch looks dead. It becomes observable only when the
    detection also overlaps a second, still-free box above the IoU gate: skipping
    the taken one promotes that second-best match, not skipping it throws the
    detection away. Two heavily overlapping bare heads do exactly that.

    GT: H1 (0,0,20,20)  H2 (4,0,20,20)   -- both 20*20 = 400 px^2
      D1 head (0,0,20,20) s=0.90
           vs H1  inter 20*20 = 400, union 400            -> IoU 1.000  <- best
           vs H2  inter 16*20 = 320, union 800-320 = 480  -> IoU 0.667
      D2 head (1,0,20,20) s=0.80
           vs H1  inter 19*20 = 380, union 800-380 = 420  -> IoU 0.905  <- best,
                                                              already taken
           vs H2  inter 17*20 = 340, union 800-340 = 460  -> IoU 0.739  <- free,
                                                              and over 0.50
    Skipping the taken box: D1->H1, D2->H2 -> 2 of 2 -> recall 1.00.
    Not skipping it:        D1->H1, D2->H1 again       -> 1 of 2 -> recall 0.50.
    """

    ground_truth = _ground_truth(_sample(1, [(0, 0, 20, 20), (4, 0, 20, 20)], [HEAD, HEAD]))
    detections = [
        _detection(1, HEAD, (0, 0, 20, 20), 0.90),
        _detection(1, HEAD, (1, 0, 20, 20), 0.80),
    ]

    result = bare_head_recall(ground_truth, detections, config=CONFIG)

    assert result.n_ground_truth == 2
    assert result.n_matched == 2
    assert result.recall == pytest.approx(1.0)


def test_a_matching_detection_scoring_exactly_the_threshold_is_kept() -> None:
    """`>=` against `>` at the operating point decides real numbers (EVAL-04).

    EVAL-04 selects the compliance operating point by reading bare-head recall AT
    a threshold, so a detection sitting exactly on it has to be inside the
    filter. Both detections here match their own ground truth at IoU 1.00, so the
    comparison is the only thing that can move the answer:
      D1 s=0.80 >  0.60  -> kept either way
      D2 s=0.60 == 0.60  -> kept by `>=`, dropped by `>`
    `>=` gives 2 of 2 -> recall 1.00; `>` gives 1 of 2 -> recall 0.50.
    """

    ground_truth = _ground_truth(_sample(1, [(0, 0, 20, 20), (40, 0, 20, 20)], [HEAD, HEAD]))
    detections = [
        _detection(1, HEAD, (0, 0, 20, 20), 0.80),
        _detection(1, HEAD, (40, 0, 20, 20), 0.60),
    ]

    result = bare_head_recall(ground_truth, detections, config=CONFIG, score_threshold=0.60)

    assert result.score_threshold == pytest.approx(0.60)
    assert result.n_ground_truth == 2
    assert result.n_matched == 2
    assert result.recall == pytest.approx(1.0)


def test_bare_head_recall_is_zero_when_the_split_has_no_bare_heads() -> None:
    ground_truth = _ground_truth(_sample(1, [(0, 0, 10, 10)], [HELMET]))

    result = bare_head_recall(ground_truth, [], config=CONFIG)

    assert result.n_ground_truth == 0
    assert result.recall == 0.0


def test_bare_head_recall_rejects_a_class_the_ground_truth_does_not_have() -> None:
    ground_truth = _ground_truth(_sample(1, [(0, 0, 10, 10)], [HELMET]))

    with pytest.raises(EvaluationConfigError, match="not in the ground truth categories"):
        bare_head_recall(ground_truth, [], config=CONFIG, class_name="forklift")


def test_boundary_area_is_counted_in_both_buckets_like_cocoeval_does() -> None:
    """32x32 = 1024 is inside [0, 1024] AND inside [1024, 9216].

    COCOeval's area filter is inclusive at both ends, so the count reported next
    to AP_small has to be inclusive too: it is the denominator the AP actually
    used, not a tidier partition that disagrees with it.
    """

    ground_truth = _ground_truth(_sample(1, [(0, 0, 32, 32)], [HELMET]))

    counts = instance_counts_by_bucket(ground_truth, config=CONFIG)

    assert counts.total["small"] == 1
    assert counts.total["medium"] == 1
    assert counts.total["all"] == 1
    assert counts.class_totals == {"helmet": 1, "head": 0, "person": 0}
    assert counts.by_class["helmet"]["small"] == 1


def test_crowd_annotations_are_excluded_from_the_counts() -> None:
    """COCOeval ignores crowd ground truth, so counting it would overstate N."""

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 10, 10), (20, 0, 10, 10)], [HELMET, HELMET])
    )
    ground_truth["annotations"][0]["iscrowd"] = 1

    counts = instance_counts_by_bucket(ground_truth, config=CONFIG)

    assert counts.total["all"] == 1
    assert counts.class_totals["helmet"] == 1


def test_per_class_ap_averages_to_the_summarize_map() -> None:
    """Pins the precision-array extraction against COCOeval.summarize().

    summarize() means over every valid entry of [T, R, K, A, M]; per-class AP
    means over one class's slice. If they disagreed, the per-class column in the
    report would be coming from a different array than the headline mAP.

    Agreement alone is two module outputs compared to each other, so the two
    per-class APs are also pinned to hand arithmetic:

      helmet: 2 boxes, (0,0,40,40) detected exactly and (100,0,40,40) missed.
              An exact box matches at all ten IoU thresholds, so at every one of
              them recall is 0.5 -> 51 of the 101 recall points -> AP = 51/101.
      head:   1 box, detected at a (1,1) offset. inter = 29*29 = 841 and
              union = 900 + 900 - 841 = 959, so IoU = 841/959 = 0.8770. That
              clears IoU 0.50..0.85 and fails 0.90 and 0.95, so the box is found
              at 8 of the 10 thresholds and perfectly found at each of those
              -> AP = 8/10 = 0.8.

    Both classes span the same number of precision entries, which is why
    summarize()'s pooled mean equals the mean of the two per-class APs here.
    """

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 40, 40), (100, 0, 40, 40)], [HELMET, HELMET]),
        _sample(2, [(0, 0, 30, 30)], [HEAD]),
    )
    detections = [
        _detection(1, HELMET, (0, 0, 40, 40), 0.9),
        _detection(2, HEAD, (1, 1, 30, 30), 0.8),
    ]

    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)

    # The absolute anchors first: without them everything below is self-consistency.
    assert metrics.per_class_ap["helmet"] == pytest.approx(AP_AT_HALF_RECALL)
    assert metrics.per_class_ap["head"] == pytest.approx(8 / 10)

    defined = [value for value in metrics.per_class_ap.values() if value > -1.0]
    assert len(defined) == 2
    assert metrics.per_class_ap["person"] == -1.0
    assert metrics.stats["map"] == pytest.approx(sum(defined) / len(defined))
    assert metrics.primary_map == pytest.approx(sum(defined) / len(defined))


def test_evaluating_does_not_mutate_the_callers_ground_truth() -> None:
    """COCOeval._prepare() writes `ignore` into every annotation dict it is given.

    With a shallow copy those dicts are the caller's, so evaluating an arm edits
    the ground truth the caller still holds - and the bootstrap hands the same
    annotations back a thousand times. Nothing raises; the GT just quietly grows
    a field, and any code that later hashes, serialises or diffs it sees a
    different object than it built.
    """

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 25, 40), (100, 0, 60, 60)], [HELMET, HEAD]),
        _sample(2, [(5, 5, 30, 30)], [HEAD]),
    )
    detections = [_detection(1, HELMET, (0, 0, 25, 40), 0.9)]
    ground_truth_before = copy.deepcopy(ground_truth)
    detections_before = copy.deepcopy(detections)

    evaluate_detection_metrics(ground_truth, detections, config=CONFIG)

    assert "ignore" not in ground_truth["annotations"][0]
    assert ground_truth == ground_truth_before
    assert detections == detections_before


def test_per_class_ap_uses_the_hundred_detection_budget_not_the_one() -> None:
    """The maxDets axis of the precision array, which every other fixture hides.

    COCOeval accumulates three detection budgets per image (1, 10, 100) on the
    last axis of eval['precision']. A fixture with at most one detection per
    image scores identically under all three, so the axis is unobservable - which
    is why reading the wrong one survived.

    Here ONE image carries three `helmet` boxes and two exact detections:
      budget 100 -> both detections count -> 2 of 3 found -> AP = 67/101
      budget 1   -> only the top-scoring one -> 1 of 3   -> AP = 34/101
    The 1-detection budget is not a defensible number for this project: site
    photos routinely contain twenty helmets, so a per-image cap of one would
    report a recall ceiling of 1/20 as if it were the model's.
    """

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 20, 20), (40, 0, 20, 20), (80, 0, 20, 20)], [HELMET] * 3)
    )
    detections = [
        _detection(1, HELMET, (0, 0, 20, 20), 0.9),
        _detection(1, HELMET, (40, 0, 20, 20), 0.8),
    ]

    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)

    assert metrics.per_class_ap["helmet"] == pytest.approx(AP_AT_TWO_THIRDS_RECALL)
    assert metrics.per_class_ap["helmet"] != pytest.approx(AP_AT_ONE_THIRD_RECALL)
    # All three boxes are 400 px^2, so the small bucket must report the same AP.
    assert metrics.per_class_ap_small["helmet"] == pytest.approx(AP_AT_TWO_THIRDS_RECALL)
    # summarize() reads the same axis independently; the two must not drift.
    assert metrics.stats["map"] == pytest.approx(AP_AT_TWO_THIRDS_RECALL)
    # mar_1 vs mar_100 is the budget difference made visible: 1/3 against 2/3.
    assert metrics.stats["mar_1"] == pytest.approx(1 / 3)
    assert metrics.stats["mar_100"] == pytest.approx(2 / 3)


def test_primary_map_is_undefined_when_no_primary_class_has_ground_truth() -> None:
    """`person` must never carry the headline number on its own."""

    ground_truth = _ground_truth(_sample(1, [(0, 0, 40, 40)], [PERSON]))
    detections = [_detection(1, PERSON, (0, 0, 40, 40), 0.9)]

    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)

    assert _is_perfect(metrics.per_class_ap["person"])
    assert metrics.primary_map == -1.0
    assert metrics.primary_map_small == -1.0


def test_no_detections_scores_zero_only_for_classes_that_had_ground_truth() -> None:
    """The two branches must agree on what -1.0 means (0.0 is a different claim).

    One 25x40 = 1,000 px^2 `head` in the split, and the arm predicted nothing.
      head    - one instance, none found     -> 0.0 ("found none of them")
      helmet  - no instances at all          -> -1.0 ("there were none to find")
      person  - no instances at all          -> -1.0
    The `else` branch reports exactly this, because COCOeval does; reporting a
    hard 0.0 here would make classes absent from a split look like classes the
    model failed on, and would pull `primary_map` down towards zero for a reason
    that has nothing to do with the model.
    """

    ground_truth = _ground_truth(_sample(1, [(0, 0, 25, 40)], [HEAD]))

    metrics = evaluate_detection_metrics(ground_truth, [], config=CONFIG)

    assert set(metrics.stats) == set(COCO_STAT_NAMES)
    assert metrics.stats["map"] == 0.0
    assert metrics.per_class_ap == {"helmet": -1.0, "head": 0.0, "person": -1.0}
    assert metrics.per_class_ap_small == {"helmet": -1.0, "head": 0.0, "person": -1.0}
    # Averaged over the defined primary classes only, i.e. over `head` alone.
    assert metrics.primary_map == 0.0
    assert metrics.primary_map_small == 0.0
    assert metrics.counts.total["small"] == 1
    assert metrics.n_detections == 0


def test_the_two_branches_agree_on_a_class_with_no_ground_truth() -> None:
    """Same split, same absent class, one detection's difference.

    `helmet` has no ground truth in either run, so both branches have to call it
    UNDEFINED. Before this, the no-detections branch called it 0.0 and the normal
    branch called it -1.0 for the identical situation - and any table averaging
    the per-class column would have silently changed meaning depending on whether
    the arm happened to predict anything.
    """

    ground_truth = _ground_truth(_sample(1, [(0, 0, 25, 40)], [HEAD]))

    empty = evaluate_detection_metrics(ground_truth, [], config=CONFIG)
    scored = evaluate_detection_metrics(
        ground_truth, [_detection(1, HEAD, (0, 0, 25, 40), 0.9)], config=CONFIG
    )

    assert empty.per_class_ap["helmet"] == scored.per_class_ap["helmet"] == -1.0
    assert empty.per_class_ap_small["helmet"] == scored.per_class_ap_small["helmet"] == -1.0
    # And the branches still disagree where they should: nothing found vs found.
    assert empty.per_class_ap["head"] == 0.0
    assert _is_perfect(scored.per_class_ap["head"])


def _hard_negative_case() -> tuple[dict, list[dict]]:
    ground_truth = _ground_truth(
        _sample(1, [], []),
        _sample(2, [], []),
        _sample(3, [(0, 0, 20, 20)], [HELMET]),
    )
    detections = [
        _detection(1, HELMET, (5, 5, 10, 10), 0.90),   # above the operating point
        _detection(1, HEAD, (30, 30, 10, 10), 0.10),   # below it
        _detection(2, PERSON, (0, 0, 50, 50), 0.99),   # not a primary class
    ]
    return ground_truth, detections


def test_hard_negative_false_positives_per_image() -> None:
    """One counted false positive spread over two empty images -> 0.5 per image.

    The threshold is passed explicitly. Leaving it to the config made this test
    depend on `compliance.score_threshold`, which is a VALIDATION-SELECTED value
    (EVAL-04) and legitimately moved from 0.50 to 0.07 once the real sweep ran -
    at which point the 0.10 detection crossed it and the expectation broke. A
    fixture built around a number that is meant to change is a fixture that
    fails for the wrong reason.
    """

    ground_truth, detections = _hard_negative_case()

    report = hard_negative_false_positives_per_image(
        ground_truth, detections, [1, 2], config=CONFIG, score_threshold=0.50
    )

    assert report.score_threshold == pytest.approx(0.50)
    assert report.n_false_positives == 1
    assert report.n_images == 2
    assert report.false_positives_per_image == pytest.approx(0.5)


def test_the_hard_negative_threshold_defaults_to_the_configured_operating_point() -> None:
    """The default still comes from config; it just is not what the case above pins."""

    ground_truth, detections = _hard_negative_case()

    report = hard_negative_false_positives_per_image(
        ground_truth, detections, [1, 2], config=CONFIG
    )

    assert report.score_threshold == pytest.approx(
        float(_raw_config()["compliance"]["score_threshold"])
    )


def test_hard_negative_threshold_can_be_overridden() -> None:
    ground_truth, detections = _hard_negative_case()

    report = hard_negative_false_positives_per_image(
        ground_truth, detections, [1, 2], config=CONFIG, score_threshold=0.05
    )

    assert report.n_false_positives == 2
    assert report.false_positives_per_image == pytest.approx(1.0)


def test_a_false_positive_scoring_exactly_the_threshold_is_counted() -> None:
    """The same `>=` against `>` question, on the hard-negative counter.

    The shared fixture's scores are 0.90 / 0.10 / 0.99 and the thresholds it is
    used with are 0.50 and 0.05, so nothing ever lands on the boundary and the
    comparison itself was never observed. 0.40 is chosen here because exactly one
    detection sits on it:
      image 1  helmet s=0.90 >  0.40  -> counted either way
      image 2  head   s=0.40 == 0.40  -> counted by `>=`, dropped by `>`
      image 1  head   s=0.10 <  0.40  -> never counted
      image 2  person s=0.99          -> not a primary class, never counted
    `>=` gives 2 false positives over 2 images -> 1.0 per image; `>` gives 0.5.
    """

    ground_truth, detections = _hard_negative_case()
    detections = [*detections, _detection(2, HEAD, (0, 0, 10, 10), 0.40)]

    report = hard_negative_false_positives_per_image(
        ground_truth, detections, [1, 2], config=CONFIG, score_threshold=0.40
    )

    assert report.score_threshold == pytest.approx(0.40)
    assert report.n_false_positives == 2
    assert report.n_images == 2
    assert report.false_positives_per_image == pytest.approx(1.0)


def test_hard_negative_subset_rejects_an_image_with_primary_ground_truth() -> None:
    ground_truth, detections = _hard_negative_case()

    with pytest.raises(HardNegativeSubsetError, match=r"\[3\]"):
        hard_negative_false_positives_per_image(
            ground_truth, detections, [1, 3], config=CONFIG
        )


def test_hard_negative_subset_rejects_unknown_image_ids() -> None:
    ground_truth, detections = _hard_negative_case()

    with pytest.raises(HardNegativeSubsetError, match="not in the ground truth"):
        hard_negative_false_positives_per_image(
            ground_truth, detections, [1, 99], config=CONFIG
        )


def test_hard_negative_subset_rejects_an_empty_subset() -> None:
    ground_truth, detections = _hard_negative_case()

    with pytest.raises(HardNegativeSubsetError, match="empty"):
        hard_negative_false_positives_per_image(ground_truth, detections, [], config=CONFIG)


def test_hard_negative_subset_rejects_a_repeated_image_id() -> None:
    """A duplicate id deflates the rate: the numerator counts the image once.

    False positives are counted over the SET of subset ids, so image 1's single
    false positive is counted once no matter how often the id appears; passing
    [1, 1, 2] would divide that 1 by three images and report 0.33 instead of the
    true 0.5. Rejecting rather than deduplicating, for the same reason the
    function rejects an image that carries ground truth: a caller that repeated
    an id does not know which images it is reporting over.
    """

    ground_truth, detections = _hard_negative_case()

    with pytest.raises(HardNegativeSubsetError, match=r"\[1\].*more than once"):
        hard_negative_false_positives_per_image(
            ground_truth, detections, [1, 1, 2], config=CONFIG
        )


def test_hard_negative_refuses_a_detection_class_the_ground_truth_lacks() -> None:
    """A dropped false positive understates the one number that counts them.

    Category id 7 is in no ground truth file this project produces, so a
    prediction carrying it means the class mapping is wrong somewhere upstream.
    Skipping it makes the hard-negative rate look BETTER the more broken the
    prediction file is.
    """

    ground_truth, detections = _hard_negative_case()
    detections = [*detections, _detection(2, 7, (0, 0, 10, 10), 0.99)]

    with pytest.raises(UnknownCategoryError, match="category_id 7"):
        hard_negative_false_positives_per_image(
            ground_truth, detections, [1, 2], config=CONFIG
        )


def test_hard_negative_ignores_a_detection_outside_the_subset() -> None:
    """Only detections on the subset are this function's business to validate."""

    ground_truth, detections = _hard_negative_case()
    detections = [*detections, _detection(3, 7, (0, 0, 10, 10), 0.99)]

    # Explicit threshold for the same reason as the case above: the configured
    # one is validation-selected and moves.
    report = hard_negative_false_positives_per_image(
        ground_truth, detections, [1, 2], config=CONFIG, score_threshold=0.50
    )

    assert report.n_false_positives == 1


def _bootstrap_case() -> tuple[dict, list[dict]]:
    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 20, 20)], [HEAD]),
        _sample(2, [(0, 0, 20, 20), (40, 0, 20, 20)], [HEAD, HEAD]),
        _sample(3, [(0, 0, 20, 20)], [HEAD]),
        _sample(4, [(0, 0, 20, 20)], [HELMET]),
    )
    detections = [
        _detection(1, HEAD, (0, 0, 20, 20), 0.9),
        _detection(2, HEAD, (0, 0, 20, 20), 0.8),
        _detection(4, HELMET, (0, 0, 20, 20), 0.7),
    ]
    return ground_truth, detections


def _recall_metric(ground_truth, detections) -> float:
    return bare_head_recall(ground_truth, detections, config=CONFIG).recall


def test_bootstrap_is_bit_identical_under_a_fixed_seed() -> None:
    """Uses the configured 1,000 resamples, so the default path is the tested path."""

    ground_truth, detections = _bootstrap_case()

    first = bootstrap_metric_ci(
        ground_truth, detections, _recall_metric, seed=7, config=CONFIG
    )
    second = bootstrap_metric_ci(
        ground_truth, detections, _recall_metric, seed=7, config=CONFIG
    )

    assert first == second
    assert first.low == second.low and first.high == second.high
    assert first.resamples == CONFIG["metrics"]["bootstrap_resamples"]
    assert first.confidence == CONFIG["metrics"]["bootstrap_ci"]
    assert 0.0 <= first.low <= first.high <= 1.0
    assert first.point == pytest.approx(0.5)  # 2 of 4 bare heads recovered
    # Recall is defined on every resample here (bare_head_recall returns 0.0, not
    # -1.0, for a draw with no bare heads), so nothing may be excluded.
    assert first.n_undefined == 0


def test_bootstrap_honours_explicit_resamples_and_confidence() -> None:
    ground_truth, detections = _bootstrap_case()

    result = bootstrap_metric_ci(
        ground_truth,
        detections,
        _recall_metric,
        seed=11,
        config=CONFIG,
        resamples=25,
        confidence=0.5,
    )

    assert result.resamples == 25
    assert result.confidence == 0.5


def test_bootstrap_resamples_images_with_replacement_and_relabels_them() -> None:
    """COCO requires unique image ids; a repeated draw must not collapse into one."""

    ground_truth, detections = _bootstrap_case()
    observed: list[tuple[int, int]] = []

    def metric(resampled_gt, resampled_dt) -> float:
        image_ids = [image["id"] for image in resampled_gt["images"]]
        observed.append((len(image_ids), len(set(image_ids))))
        return evaluate_detection_metrics(resampled_gt, resampled_dt, config=CONFIG).stats["map"]

    result = bootstrap_metric_ci(
        ground_truth, detections, metric, seed=3, config=CONFIG, resamples=5
    )

    assert len(observed) == 6  # 5 resamples plus the point estimate
    assert all(total == len(ground_truth["images"]) for total, _ in observed)
    assert all(total == unique for total, unique in observed)
    assert 0.0 <= result.low <= result.high <= 1.0


def test_bootstrap_percentiles_are_the_two_tails_of_the_confidence_level() -> None:
    """The percentile arithmetic itself, on a sample chosen so it can be checked.

    The resample values are scripted to 0.00, 0.05, ..., 0.95 (twenty of them).
    numpy's default percentile interpolates linearly between order statistics:
    for n = 20 the value at percentile q sits at position p = q/100 * 19.

      confidence 0.95 -> tail = 0.025 -> percentiles 2.5 and 97.5
          p = 0.025 * 19 =  0.475 -> 0.00 + 0.475 * 0.05 = 0.02375
          p = 0.975 * 19 = 18.525 -> 0.90 + 0.525 * 0.05 = 0.92625
      confidence 0.50 -> tail = 0.250 -> percentiles 25 and 75
          p = 0.25 * 19 =  4.75   -> 0.20 + 0.750 * 0.05 = 0.2375
          p = 0.75 * 19 = 14.25   -> 0.70 + 0.250 * 0.05 = 0.7125

    Splitting (1 - confidence) across TWO tails is the whole content of the line.
    Putting it all in one tail yields [0.0475, 0.9025] instead - a 90% interval
    labelled 95%, i.e. every number in the report quoted with more confidence
    than it earned, and the arithmetic runs either way without complaint.
    """

    ground_truth, detections = _bootstrap_case()
    resample_values = [0.05 * index for index in range(20)]

    wide = bootstrap_metric_ci(
        ground_truth,
        detections,
        _ScriptedMetric([0.5, *resample_values]),
        seed=5,
        config=CONFIG,
        resamples=len(resample_values),
        confidence=0.95,
    )
    narrow = bootstrap_metric_ci(
        ground_truth,
        detections,
        _ScriptedMetric([0.5, *resample_values]),
        seed=5,
        config=CONFIG,
        resamples=len(resample_values),
        confidence=0.50,
    )

    assert wide.point == pytest.approx(0.5)
    assert wide.low == pytest.approx(0.02375)
    assert wide.high == pytest.approx(0.92625)
    assert narrow.low == pytest.approx(0.2375)
    assert narrow.high == pytest.approx(0.7125)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_bootstrap_excludes_undefined_resamples_and_says_how_many() -> None:
    """-1.0 is finite, sorts below every real AP, and is not an AP.

    Five scripted resamples, two of them UNDEFINED. The pooled sample is
    [0.2, 0.6, 1.0] (n = 3, positions p = q/100 * 2):
        p = 0.025 * 2 = 0.05 -> 0.2 + 0.05 * (0.6 - 0.2) = 0.22
        p = 0.975 * 2 = 1.95 -> 0.6 + 0.95 * (1.0 - 0.6) = 0.98
    Pooling the two -1.0s instead would put them at the bottom of the order
    statistics and drag the lower bound to -1.0 - a confidence interval on an
    average precision that starts below zero, printed without complaint.
    """

    ground_truth, detections = _bootstrap_case()

    result = bootstrap_metric_ci(
        ground_truth,
        detections,
        _ScriptedMetric([0.4, 0.2, -1.0, 0.6, -1.0, 1.0]),
        seed=2,
        config=CONFIG,
        resamples=5,
    )

    assert result.point == pytest.approx(0.4)
    assert result.n_undefined == 2
    assert result.resamples == 5
    assert result.low == pytest.approx(0.22)
    assert result.high == pytest.approx(0.98)


def test_bootstrap_refuses_an_undefined_point_estimate() -> None:
    """An interval around a number that does not exist is not an interval.

    The metric is checked on the full split first, so this raises before a single
    resample is drawn - the scripted metric holds exactly one value and would
    raise IndexError if it were called again.
    """

    ground_truth, detections = _bootstrap_case()

    with pytest.raises(BootstrapError, match="UNDEFINED sentinel"):
        bootstrap_metric_ci(
            ground_truth,
            detections,
            _ScriptedMetric([-1.0]),
            seed=1,
            config=CONFIG,
            resamples=3,
        )


def test_bootstrap_refuses_when_every_resample_is_undefined() -> None:
    ground_truth, detections = _bootstrap_case()

    with pytest.raises(BootstrapError, match="nothing to take a percentile of"):
        bootstrap_metric_ci(
            ground_truth,
            detections,
            _ScriptedMetric([0.5, -1.0, -1.0]),
            seed=1,
            config=CONFIG,
            resamples=2,
        )


def test_bootstrap_rejects_a_non_finite_metric_value() -> None:
    ground_truth, detections = _bootstrap_case()

    with pytest.raises(BootstrapError, match="point estimate is nan.*non-finite"):
        bootstrap_metric_ci(
            ground_truth,
            detections,
            lambda _gt, _dt: float("nan"),
            seed=1,
            config=CONFIG,
            resamples=2,
        )
    # A resample that goes non-finite after a finite point estimate is the other
    # half of the branch, and names the offending resample.
    with pytest.raises(BootstrapError, match="resample 1 produced nan"):
        bootstrap_metric_ci(
            ground_truth,
            detections,
            _ScriptedMetric([0.5, 0.5, float("nan")]),
            seed=1,
            config=CONFIG,
            resamples=2,
        )


def test_bootstrap_rejects_impossible_settings() -> None:
    ground_truth, detections = _bootstrap_case()

    with pytest.raises(BootstrapError, match="must be positive"):
        bootstrap_metric_ci(
            ground_truth, detections, _recall_metric, seed=1, config=CONFIG, resamples=0
        )
    with pytest.raises(BootstrapError, match=r"\(0, 1\)"):
        bootstrap_metric_ci(
            ground_truth,
            detections,
            _recall_metric,
            seed=1,
            config=CONFIG,
            resamples=2,
            confidence=1.5,
        )
    with pytest.raises(BootstrapError, match="empty image set"):
        bootstrap_metric_ci(
            {"images": [], "annotations": [], "categories": []},
            [],
            _recall_metric,
            seed=1,
            config=CONFIG,
            resamples=2,
        )


def test_area_ranges_reject_a_renamed_bucket() -> None:
    """COCOeval.summarize() looks buckets up by label; a rename yields silent NaN."""

    config = copy.deepcopy(CONFIG)
    config["metrics"]["coco_area_ranges"]["tiny"] = config["metrics"]["coco_area_ranges"].pop(
        "small"
    )

    with pytest.raises(EvaluationConfigError, match="unexpected"):
        area_ranges_from_config(config)


def test_area_ranges_reject_an_inverted_range() -> None:
    config = copy.deepcopy(CONFIG)
    config["metrics"]["coco_area_ranges"]["small"] = [1024, 0]

    with pytest.raises(EvaluationConfigError, match="empty or inverted"):
        area_ranges_from_config(config)


def test_area_ranges_span_all_buckets() -> None:
    buckets = area_ranges_from_config(CONFIG)

    labels = [label for label, _ in buckets]
    assert labels == ["all", "small", "medium", "large"]
    assert buckets[0][1] == (0.0, float(CONFIG["metrics"]["coco_area_ranges"]["large"][1]))


def test_the_config_defaults_read_configs_evaluation_yaml() -> None:
    """The `config=None` path is what production calls; skipping it is K-18.

    Asserted against the FILE, parsed here independently, and not against
    `load_evaluation_config()`. Comparing `f()` to `f(CONFIG)` only proves the two
    call sites agree with each other, which they would even if both read the
    wrong file or neither read one - and every value below would be free to
    change without a single test noticing.
    """

    raw = _raw_config()
    ranges = raw["metrics"]["coco_area_ranges"]
    lows = [float(ranges[label][0]) for label in ("small", "medium", "large")]
    highs = [float(ranges[label][1]) for label in ("small", "medium", "large")]

    buckets = dict(area_ranges_from_config())

    assert list(buckets) == ["all", "small", "medium", "large"]
    for index, label in enumerate(("small", "medium", "large")):
        assert buckets[label] == (lows[index], highs[index])
    assert buckets["all"] == (min(lows), max(highs))

    # One pixel under the file's small ceiling is small; one over is not. The
    # boxes are built FROM the file's number, so the assertion tracks a config
    # change and fails if the code stops reading the config at all.
    ceiling = int(highs[0])
    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 1, ceiling - 1)], [HELMET]),
        _sample(2, [(0, 0, 1, ceiling + 1)], [HELMET]),
    )
    counts = instance_counts_by_bucket(ground_truth)
    assert (counts.total["small"], counts.total["medium"]) == (1, 1)

    # The whole bundle on its default path too: only the sub-ceiling box is
    # detected, so the small bucket is perfect and the medium one is a zero.
    metrics = evaluate_detection_metrics(
        ground_truth, [_detection(1, HELMET, (0, 0, 1, ceiling - 1), 0.9)]
    )
    assert _is_perfect(metrics.stats["map_small"])
    assert metrics.stats["map_medium"] == 0.0
    # metrics.primary_classes decides which classes the two headline numbers are
    # averaged over, and it is the one key nothing else reads on the default
    # path: every other test hands `config=CONFIG` in explicitly.
    assert metrics.primary_classes == tuple(raw["metrics"]["primary_classes"])

    heads = _ground_truth(_sample(1, [(0, 0, 10, 10)], [HEAD]))
    assert bare_head_recall(heads, []).iou_threshold == float(
        raw["metrics"]["bare_head_recall_iou"]
    )

    empty_image = _ground_truth(_sample(1, [], []))
    assert hard_negative_false_positives_per_image(
        empty_image, [], [1]
    ).score_threshold == float(raw["compliance"]["score_threshold"])

    assert evaluates_in_original_coordinates() is bool(
        raw["metrics"]["evaluate_in_original_coordinates"]
    )


def test_the_bootstrap_defaults_read_configs_evaluation_yaml() -> None:
    """metrics.bootstrap_resamples and metrics.bootstrap_ci, from the file.

    The scripted metric is strict, so `calls` also proves the loop ran exactly
    the configured number of times plus the one point estimate - a default that
    silently resolved to something else would raise IndexError or leave values
    unconsumed.
    """

    raw = _raw_config()
    expected_resamples = int(raw["metrics"]["bootstrap_resamples"])
    ground_truth, detections = _bootstrap_case()
    metric = _ScriptedMetric([0.5] * (expected_resamples + 1))

    interval = bootstrap_metric_ci(ground_truth, detections, metric, seed=1)

    assert interval.resamples == expected_resamples
    assert interval.confidence == float(raw["metrics"]["bootstrap_ci"])
    assert metric.calls == expected_resamples + 1


def test_a_non_mapping_config_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text("just-a-string\n", encoding="utf-8", newline="\n")

    with pytest.raises(TypeError, match="Expected a mapping"):
        load_evaluation_config(path)


def test_area_falls_back_to_width_times_height_when_the_field_is_absent() -> None:
    """Not every COCO file in the wild carries `area`; guessing wrong is silent.

    The box is 40x40 = 1,600 px^2, which is MEDIUM: the small ceiling is 1,024
    (32^2). The bucket therefore reports the fallback rather than merely
    surviving it. A fixture whose w*h landed in the SMALL bucket could not: a
    wrong fallback of 0.0 is small too, so right and wrong would both answer
    (small=1, medium=0) and the assertion would hold either way.
    """

    ground_truth = _ground_truth(_sample(1, [(0, 0, 40, 40)], [HELMET]))
    del ground_truth["annotations"][0]["area"]

    counts = instance_counts_by_bucket(ground_truth, config=CONFIG)

    assert counts.total["small"] == 0
    assert counts.total["medium"] == 1
    assert counts.total["all"] == 1


def test_crowd_bare_heads_are_ignored_by_recall_and_by_self_evaluation() -> None:
    """COCOeval ignores crowd ground truth; the hand-rolled recall must agree."""

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 10, 10), (20, 0, 10, 10)], [HEAD, HEAD])
    )
    ground_truth["annotations"][0]["iscrowd"] = 1
    detections = [_detection(1, HEAD, (20, 0, 10, 10), 0.9)]

    result = bare_head_recall(ground_truth, detections, config=CONFIG)

    assert result.n_ground_truth == 1
    assert result.recall == 1.0
    assert len(ground_truth_as_detections(ground_truth)) == 1


def test_metric_rows_carry_the_instance_count_next_to_the_ap() -> None:
    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 25, 40), (100, 0, 60, 60)], [HELMET, HEAD]),
    )
    detections = [_detection(1, HELMET, (0, 0, 25, 40), 0.9)]
    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)

    rows = detection_metric_rows(metrics, arm="real_only", seed=42, split="test")
    by_metric = {row.metric: row for row in rows}

    assert by_metric["map_small"].n_instances == metrics.counts.total["small"] == 1
    assert by_metric["instances.small"].value == 1.0
    assert by_metric["ap.head"].n_instances == 1
    assert by_metric["bare_head_recall"].n_instances == metrics.bare_head.n_ground_truth
    assert by_metric["bare_head_recall"].value == pytest.approx(0.0)
    assert all(row.arm == "real_only" and row.seed == 42 for row in rows)
    assert all(row.n_images == 1 for row in rows)


def test_the_primary_rows_count_only_the_primary_classes() -> None:
    """`person` is reported but excluded from the headline claim (ADR-003/EXP-03).

    One 20x20 = 400 px^2 box per class, so all three land in the small bucket:
      counts.total['small']  = 3   (helmet + head + person)
      primary_classes        = helmet, head
      primary_map_small over = 2   instances
    Both primary_* metrics average two classes, so pairing them with the
    all-class 3 hands the reader the wrong denominator - which is the exact
    mistake EVAL-08 exists to prevent. On the real Test split the gap is not one
    box: `person` alone is 751 instances.
    """

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 20, 20), (40, 0, 20, 20), (80, 0, 20, 20)], [HELMET, HEAD, PERSON])
    )
    metrics = evaluate_detection_metrics(
        ground_truth, ground_truth_as_detections(ground_truth), config=CONFIG
    )

    rows = detection_metric_rows(metrics, arm="real_only", seed=42, split="test")
    by_metric = {row.metric: row for row in rows}

    assert metrics.primary_classes == ("helmet", "head")
    assert metrics.counts.total["small"] == 3
    assert by_metric["instances.small"].value == 3.0
    assert by_metric["map_small"].n_instances == 3
    assert by_metric["primary_map_small"].n_instances == 2
    assert by_metric["primary_map"].n_instances == 2


def test_detection_metrics_csv_round_trips_exactly(tmp_path: Path) -> None:
    """EVAL-12: every table is re-aggregated from this file, so it must not lose bits."""

    rows = [
        MetricRow(
            arm="real_only",
            seed=42,
            split="test",
            metric="map_small",
            value=0.123456789012345,
            n_instances=9,
            n_images=3,
        ),
        MetricRow(
            arm="filtered_syn",
            seed=1,
            split="test",
            metric="ap.person",
            value=-1.0,
            ci_low=0.0123456789,
            ci_high=0.9876543211,
            notes="single seed",
        ),
    ]

    path = write_detection_metrics_csv(rows, tmp_path / "detection_metrics.csv")

    assert read_detection_metrics_csv(path) == rows
    assert b"\r\n" not in path.read_bytes()


def _ci_case() -> tuple[dict, list[dict]]:
    """Six images, deliberately imperfect, so the intervals have width.

    All boxes are 20x20 = 400 px^2 (small). `head`: 3 instances, 2 found ->
    recall 2/3. `person`: 2 instances, 1 found. `helmet`: 2 instances, both found.
    """

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 20, 20), (40, 0, 20, 20)], [HELMET, HEAD]),
        _sample(2, [(0, 0, 20, 20)], [HEAD]),
        _sample(3, [(0, 0, 20, 20)], [PERSON]),
        _sample(4, [(0, 0, 20, 20)], [PERSON]),
        _sample(5, [(0, 0, 20, 20)], [HELMET]),
        _sample(6, [(0, 0, 20, 20)], [HEAD]),
    )
    detections = [
        _detection(1, HELMET, (0, 0, 20, 20), 0.9),
        _detection(1, HEAD, (40, 0, 20, 20), 0.8),
        _detection(3, PERSON, (0, 0, 20, 20), 0.7),
        _detection(5, HELMET, (0, 0, 20, 20), 0.6),
        _detection(6, HEAD, (0, 0, 20, 20), 0.5),
    ]
    return ground_truth, detections


def test_eval09_names_the_headline_metrics_and_every_non_primary_class() -> None:
    """`person` is derived from metrics.primary_classes, not spelled out in code."""

    ground_truth, _ = _ci_case()

    assert bootstrap_ci_metric_names(ground_truth, config=CONFIG) == (
        "primary_map_small",
        "bare_head_recall",
        "ap.person",
    )


def test_rows_carry_the_bootstrap_interval_eval09_requires(tmp_path: Path) -> None:
    """Without this wiring ci_low/ci_high are empty in every row of the CSV.

    An empty uncertainty column does not read as "not measured", it reads as "no
    uncertainty", which is the opposite of what EVAL-09 asks the table to say.
    """

    ground_truth, detections = _ci_case()
    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)
    rows = detection_metric_rows(metrics, arm="real_only", seed=42, split="test")
    named = bootstrap_ci_metric_names(ground_truth, config=CONFIG)

    assert all(row.ci_low is None and row.ci_high is None for row in rows)

    with_cis = attach_bootstrap_cis(
        rows, ground_truth, detections, seed=4, config=CONFIG, resamples=16
    )
    by_metric = {row.metric: row for row in with_cis}

    # Hand arithmetic on the fixture, so the intervals are known to sit around
    # the right numbers: helmet 2 of 2 found -> AP_small 1.000, head 2 of 3 ->
    # 67/101, so primary_map_small = (1.000 + 67/101) / 2 = 0.83168.
    # person 1 of 2 -> 51/101. head recall 2 of 3.
    assert by_metric["primary_map_small"].value == pytest.approx(
        (1.0 + AP_AT_TWO_THIRDS_RECALL) / 2
    )
    assert by_metric["ap.person"].value == pytest.approx(AP_AT_HALF_RECALL)
    assert by_metric["bare_head_recall"].value == pytest.approx(2 / 3)

    for name in named:
        row = by_metric[name]
        assert row.ci_low is not None and row.ci_high is not None, name
        assert row.ci_low <= row.ci_high, name
        assert "bootstrap" in row.notes
    # `person` is in 2 of the 6 images, so some draws contain none of it and the
    # metric does not exist on them. Those are excluded from the percentile, and
    # the count rides along in the notes instead of vanishing.
    #
    # The COUNT is not pinned. Which particular draws come out undefined is a
    # function of the seeding scheme, and pinning it made this test fail when
    # EVAL-09 moved to independent per-resample seeds - a change that altered
    # nothing about the behaviour under test. What must hold is that the number
    # is real, positive, and disclosed.
    undefined = re.search(
        r"(\d+) resample\(s\) undefined and excluded", by_metric["ap.person"].notes
    )
    assert undefined is not None, by_metric["ap.person"].notes
    assert 0 < int(undefined.group(1)) < 16
    # At least one is a real interval rather than a degenerate point, otherwise
    # "it has a CI" would be satisfied by writing the point estimate twice.
    assert any(by_metric[name].ci_low < by_metric[name].ci_high for name in named)
    # Rows EVAL-09 does not name are untouched, including their notes.
    assert by_metric["map"].ci_low is None
    assert by_metric["ap.helmet"].ci_low is None
    # bare_head_recall already carried a note; the CI note is appended, not lost.
    assert by_metric["bare_head_recall"].notes.startswith("iou=")

    path = write_detection_metrics_csv(with_cis, tmp_path / "detection_metrics.csv")
    assert read_detection_metrics_csv(path) == with_cis


def test_an_undefined_metric_gets_a_reason_instead_of_an_interval() -> None:
    """No `person` in the split, so `ap.person` is -1.0 and cannot have a CI."""

    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 20, 20)], [HELMET]),
        _sample(2, [(0, 0, 20, 20)], [HEAD]),
    )
    detections = ground_truth_as_detections(ground_truth)
    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)
    rows = detection_metric_rows(metrics, arm="real_only", seed=1, split="test")

    with_cis = attach_bootstrap_cis(
        rows, ground_truth, detections, seed=4, config=CONFIG, resamples=4
    )
    person = next(row for row in with_cis if row.metric == "ap.person")

    assert person.value == -1.0
    assert person.ci_low is None and person.ci_high is None
    assert "undefined" in person.notes


def test_attaching_a_ci_to_a_metric_with_no_row_is_an_error() -> None:
    ground_truth, detections = _ci_case()
    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)
    rows = detection_metric_rows(metrics, arm="real_only", seed=1, split="test")

    with pytest.raises(KeyError, match="ap.forklift"):
        attach_bootstrap_cis(
            rows,
            ground_truth,
            detections,
            seed=1,
            config=CONFIG,
            resamples=2,
            metric_names=["ap.forklift"],
        )


def test_every_ci_metric_name_has_a_bootstrap_recipe() -> None:
    """A row name with no recipe must raise rather than ship without a CI."""

    ground_truth, detections = _ci_case()
    metrics = evaluate_detection_metrics(ground_truth, detections, config=CONFIG)
    rows = detection_metric_rows(metrics, arm="real_only", seed=1, split="test")

    attached = attach_bootstrap_cis(
        rows,
        ground_truth,
        detections,
        seed=1,
        config=CONFIG,
        resamples=4,
        metric_names=["primary_map", "ap_small.head"],
    )
    by_metric = {row.metric: row for row in attached}
    assert by_metric["primary_map"].ci_low is not None
    assert by_metric["ap_small.head"].ci_low is not None

    with pytest.raises(KeyError, match="No bootstrap recipe"):
        attach_bootstrap_cis(
            rows,
            ground_truth,
            detections,
            seed=1,
            config=CONFIG,
            resamples=2,
            metric_names=["n_detections"],
        )


def test_default_metrics_path_is_the_repo_results_directory() -> None:
    path = default_detection_metrics_path()

    assert path.name == "detection_metrics.csv"
    assert path.parent.name == "results"
    assert path.parent.parent == PROJECT_ROOT


def test_writing_without_a_path_uses_the_default_and_creates_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EVAL-11 names the output file, so the no-argument call is a real code path."""

    target = tmp_path / "results" / "detection_metrics.csv"
    monkeypatch.setattr(detection, "default_detection_metrics_path", lambda: target)

    written = write_detection_metrics_csv(
        [MetricRow(arm="a", seed=1, split="test", metric="map", value=0.5)]
    )

    assert written == target
    assert read_detection_metrics_csv(written)[0].metric == "map"


@pytest.mark.parametrize(
    ("train_name", "test_name", "why"),
    [
        ("hard_hat_4242.png", "hard_hat_4242.png", "identical strings"),
        (r"data\test\hard_hat_4242.png", "data/test/hard_hat_4242.png", "separator"),
        ("images/hard_hat_4242.png", "hard_hat_4242.png", "directory prefix"),
        ("HARD_HAT_4242.PNG", "hard_hat_4242.png", "case"),
        (
            r"D:\sdg-data\02-safesynth\raw\hard_hat_4242.png",
            "splits/test/hard_hat_4242.png",
            "absolute vs relative",
        ),
    ],
)
def test_leakage_is_caught_however_the_two_lists_spell_the_path(
    train_name: str, test_name: str, why: str
) -> None:
    """EVAL-14 is a leak check, so a false NEGATIVE invalidates the experiment.

    The train list and the Test manifest are produced by different code at
    different times - one uses `Path.as_posix()`, another prints Windows paths,
    another keeps bare file names - and raw string intersection answers "were
    these written the same way". Every case here is one genuinely leaked image
    that a string comparison reports as clean.
    """

    with pytest.raises(SplitLeakageError, match="hard_hat_4242"):
        assert_no_test_images_in_training_list(
            ["a.png", train_name, "b.png"], [test_name, "c.png"]
        )


def test_leakage_message_shows_both_spellings() -> None:
    """"Leakage detected" is not actionable; the mismatch is the thing to fix."""

    with pytest.raises(SplitLeakageError) as error:
        assert_no_test_images_in_training_list(
            [r"train\hard_hat_4242.png"], ["test/hard_hat_4242.PNG"]
        )

    message = str(error.value)
    assert r"train\hard_hat_4242.png" in message
    assert "test/hard_hat_4242.PNG" in message


def test_leakage_check_passes_on_disjoint_sets() -> None:
    assert (
        assert_no_test_images_in_training_list(["a.png", "b.png"], ["c.png", "d.png"]) is None
    )


def test_leakage_check_does_not_fire_on_different_images_in_the_same_folder() -> None:
    """Normalisation must not turn every shared directory into a leak."""

    assert (
        assert_no_test_images_in_training_list(
            [r"data\train\hard_hat_0001.png", r"data\train\hard_hat_0002.png"],
            ["data/test/hard_hat_0003.png"],
        )
        is None
    )


@pytest.mark.skipif(
    _HAS_FASTER_COCO_EVAL, reason="faster-coco-eval is installed; the cross-check runs instead"
)
def test_cross_check_reports_that_faster_coco_eval_is_absent() -> None:
    """EVAL-13 exists as code even though the package is not a dependency."""

    ground_truth = _ground_truth(_sample(1, [(0, 0, 25, 40)], [HELMET]))

    with pytest.raises(ImportError, match="not installed"):
        cross_check_with_faster_coco_eval(
            ground_truth, ground_truth_as_detections(ground_truth), config=CONFIG
        )


@pytest.mark.skipif(
    not _HAS_FASTER_COCO_EVAL, reason="faster-coco-eval is not a dependency (EVAL-13)"
)
def test_pycocotools_and_faster_coco_eval_agree_on_the_same_input() -> None:
    ground_truth = _ground_truth(
        _sample(1, [(0, 0, 25, 40), (100, 0, 60, 60)], [HELMET, HEAD]),
        _sample(2, [(5, 5, 30, 30)], [HEAD]),
    )
    detections = [
        _detection(1, HELMET, (0, 0, 25, 40), 0.9),
        _detection(2, HEAD, (6, 6, 30, 30), 0.7),
    ]

    compared = cross_check_with_faster_coco_eval(ground_truth, detections, config=CONFIG)

    assert set(compared) == set(COCO_STAT_NAMES)
    assert all(left == right for left, right in compared.values())


# --------------------------------------------------------------------------
# EVAL-09: the interval must not depend on how many cores the machine has
# --------------------------------------------------------------------------


def _tiny_split(n_images: int = 6):
    """A split small enough to bootstrap in-process, with a metric that varies."""

    images = [{"id": i, "file_name": f"{i}.png", "width": 100, "height": 100}
              for i in range(1, n_images + 1)]
    annotations = [
        {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20],
         "area": 400.0, "iscrowd": 0}
        for i in range(1, n_images + 1)
    ]
    ground_truth = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "helmet"}],
    }
    # Half the images are detected perfectly and half not at all, so which
    # images a resample draws actually changes the metric.
    detections = [
        {"image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}
        for i in range(1, n_images // 2 + 1)
    ]
    return ground_truth, detections


def _counting_metric(seen):
    """Records WHICH images each draw contained, by file name.

    Not by id: `_resample_by_image` renumbers ids 1..N, so the id tuple is
    identical for every draw and an assertion on it would pass no matter what
    the sampler did. The file name is what survives renumbering.
    """

    def metric(ground_truth, detections):
        drawn = tuple(sorted(image["file_name"] for image in ground_truth["images"]))
        seen.append(drawn)
        return len(detections) / max(len(ground_truth["images"]), 1)

    return metric


def test_the_same_seed_draws_the_same_resamples_every_time() -> None:
    ground_truth, detections = _tiny_split()
    first, second = [], []

    detection.bootstrap_metric_ci(
        ground_truth, detections, _counting_metric(first), seed=42, resamples=8,
        config=detection.load_evaluation_config(),
    )
    detection.bootstrap_metric_ci(
        ground_truth, detections, _counting_metric(second), seed=42, resamples=8,
        config=detection.load_evaluation_config(),
    )

    assert first == second
    assert len(set(first)) > 1, "fixture degenerated: every draw was identical"


def test_a_different_seed_draws_different_resamples() -> None:
    ground_truth, detections = _tiny_split()
    first, second = [], []

    for seed, sink in ((42, first), (43, second)):
        detection.bootstrap_metric_ci(
            ground_truth, detections, _counting_metric(sink), seed=seed, resamples=8,
            config=detection.load_evaluation_config(),
        )

    assert first != second


def test_resample_k_does_not_depend_on_resamples_1_through_k_minus_1() -> None:
    """The property that makes parallelism safe, tested WITHOUT a process pool.

    A generator advanced in a loop gives resample k a state inherited from every
    draw before it. Under that scheme a 4-resample run and an 8-resample run
    share a prefix but nothing more - and splitting the loop across processes
    would silently change the published bounds with the core count. Independent
    SeedSequence children make draw k a function of (seed, k) alone.
    """

    short, long = [], []
    ground_truth, detections = _tiny_split()

    detection.bootstrap_metric_ci(
        ground_truth, detections, _counting_metric(short), seed=42, resamples=4,
        config=detection.load_evaluation_config(),
    )
    detection.bootstrap_metric_ci(
        ground_truth, detections, _counting_metric(long), seed=42, resamples=8,
        config=detection.load_evaluation_config(),
    )

    # `short` is 1 point estimate + 4 draws; the point estimate is recorded too.
    assert len(short) == 5 and len(long) == 9
    assert long[: len(short)] == short
    assert len(set(long)) > 1, "fixture degenerated: every draw was identical"
