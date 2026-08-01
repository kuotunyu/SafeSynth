"""Tests for src/evaluation/error_analysis.py (EVAL-15..EVAL-18).

Written to the K-19 standard: green proves nothing, so every expected value below
is derived BY HAND in a comment and never read back out of the code under test.
Boundary cases are chosen so that the boundary is the thing that decides — an
equal-improvement slice, a tie on the headline metric, an IoU of exactly the
threshold — because a boundary that is never the winner can be flipped without
any test noticing.

The fixture is one small COCO split whose every box has a named role, listed in
the table above `ground_truth()`. Each classification test states which role it
is exercising, so a change in the fixture that would make a test vacuous is
visible.

The EVAL-15 grids are tested through the `GridPlan` the renderer draws from, not
through the PNG. "The file exists and is under 400 KB" is what let six one-token
bugs — the baseline arm in both columns, a crop pinned to the top-left corner,
an empty caption, the other arm's box, no ground truth at all, and an inverted
colour legend — survive a green suite. The plan says which arm, which crop
window, which box, which colour and which words; that is what gets asserted.
Two tests do go to the pixels, and they sample specific coordinates rather than
weighing the file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.error_analysis import (
    DriverError,
    budget_inputs,
    load_test_samples,
    load_yaml,
    parse_args,
    read_arm_predictions,
    read_best_checkpoint_values,
    read_exposure,
    read_headline_values,
)
from scripts.error_analysis import main as run_driver
from src.evaluation import error_analysis as ea
from src.evaluation.detection import (
    UNDEFINED,
    UnknownCategoryError,
    hard_negative_false_positives_per_image,
)

# --- Fixtures ------------------------------------------------------------------

CATEGORIES = [
    {"id": 0, "name": "helmet"},
    {"id": 1, "name": "head"},
    {"id": 2, "name": "person"},
]
HELMET, HEAD, PERSON = 0, 1, 2


def config() -> dict[str, Any]:
    """A stand-in for configs/evaluation.yaml with only the keys used here."""

    return {
        "compliance": {"score_threshold": 0.50},
        "metrics": {
            "coco_area_ranges": {
                "small": [0, 1024],
                "medium": [1024, 9216],
                "large": [9216, 100000000],
            },
            "evaluate_in_original_coordinates": True,
            "bare_head_recall_iou": 0.50,
            "primary_classes": ["helmet", "head"],
            "bootstrap_resamples": 10,
            "bootstrap_ci": 0.95,
        },
        "error_analysis": {
            "compare_arms": ["real_only", "filtered_syn"],
            "samples_per_category": 2,
            "categories": [
                "fixed_false_negative",
                "fixed_false_positive",
                "new_false_positive",
                "both_wrong",
            ],
        },
    }


def annotation(
    annotation_id: int,
    image_id: int,
    category_id: int,
    bbox: tuple[float, float, float, float],
    iscrowd: int = 0,
) -> dict[str, Any]:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "area": float(bbox[2]) * float(bbox[3]),
        "iscrowd": iscrowd,
    }


# The whole fixture, box by box. `gt` is the index `four_way_comparison` assigns,
# which is the position among NON-CROWD annotations in file order.
#
#   gt  image  class   bbox                role
#   0   1      head    (10,10,20,20)       both arms find it        -> no item
#   1   1      helmet  (50,50,30,30)       baseline only            -> new_false_negative
#   2   2      head    (10,10,10,10)       best only                -> fixed_false_negative
#   3   2      helmet  (70,70,20,20)       neither (best's box is
#                                          below the threshold)     -> both_wrong
#   -   1      person  (0,0,40,90) crowd   skipped entirely
#   4   4      person  (5,5,30,60)         neither arm predicts
#                                          `person` at all          -> both_wrong
#
# The `person` at gt 4 is deliberate. It keeps image 4 a hard negative (the
# subset is defined on the PRIMARY classes only) and it proves that a class
# excluded from the headline still reaches the counts table instead of being
# filtered out of the error analysis.
def ground_truth() -> dict[str, Any]:
    return {
        "images": [{"id": index, "width": 100, "height": 100} for index in (1, 2, 3, 4)],
        "annotations": [
            annotation(1, 1, HEAD, (10, 10, 20, 20)),
            annotation(2, 1, HELMET, (50, 50, 30, 30)),
            annotation(3, 2, HEAD, (10, 10, 10, 10)),
            annotation(4, 2, HELMET, (70, 70, 20, 20)),
            annotation(5, 1, PERSON, (0, 0, 40, 90), iscrowd=1),
            annotation(6, 4, PERSON, (5, 5, 30, 60)),
        ],
        "categories": [dict(entry) for entry in CATEGORIES],
    }


def detection(
    image_id: int, category_id: int, bbox: tuple[float, float, float, float], score: float
) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "score": score,
    }


def baseline_detections() -> list[dict[str, Any]]:
    return [
        detection(1, HEAD, (10, 10, 20, 20), 0.90),  # hits gt 0
        detection(1, HELMET, (50, 50, 30, 30), 0.80),  # hits gt 1
        detection(3, HELMET, (5, 5, 20, 20), 0.70),  # baseline-only FP
        detection(3, HEAD, (60, 60, 10, 10), 0.60),  # FP both arms make
        detection(4, HELMET, (5, 5, 10, 10), 0.55),  # baseline-only FP
    ]


def best_detections() -> list[dict[str, Any]]:
    return [
        detection(1, HEAD, (10, 10, 20, 20), 0.95),  # hits gt 0
        detection(2, HEAD, (10, 10, 10, 10), 0.70),  # hits gt 2
        detection(2, HELMET, (70, 70, 20, 20), 0.30),  # below threshold -> not a hit
        detection(3, HEAD, (60, 60, 10, 10), 0.55),  # FP both arms make
        detection(3, HELMET, (80, 10, 15, 15), 0.90),  # best-only FP
    ]


def detections_by_arm() -> dict[str, list[dict[str, Any]]]:
    return {"real_only": baseline_detections(), "filtered_syn": best_detections()}


def compare(**overrides: Any) -> tuple[ea.ComparisonItem, ...]:
    kwargs: dict[str, Any] = {"config": config()}
    kwargs.update(overrides)
    return ea.four_way_comparison(
        ground_truth(), baseline_detections(), best_detections(), **kwargs
    )


def write_images(directory: Path, image_ids: tuple[int, ...] = (1, 2, 3, 4)) -> dict[int, Path]:
    """Small deterministic PNGs, so the grid renderer has something real to crop."""

    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for image_id in image_ids:
        array = np.full((100, 100, 3), fill_value=(image_id * 40) % 256, dtype=np.uint8)
        path = directory / f"image_{image_id}.png"
        Image.fromarray(array).save(path)
        paths[image_id] = path
    return paths


# 120 wide by 80 high, and never square. The frozen Test split holds 416x415,
# 415x416 and 415x415 images (DATA-25), so width and height genuinely differ in
# production; on a square fixture a transposed read of either the image shape or
# the crop window's `image_size` produces exactly the right answer.
WIDE_SIZE = (120, 80)


def detection_item(
    image_id: int,
    bbox: tuple[float, float, float, float],
    *,
    category: str = ea.NEW_FALSE_POSITIVE,
    baseline_score: float | None = None,
    best_score: float | None = 0.9,
) -> ea.ComparisonItem:
    """A detection-anchored item with whichever sides the caller asks for."""

    return ea.ComparisonItem(
        category,
        image_id,
        "helmet",
        ea.ANCHOR_DETECTION,
        bbox,
        None if baseline_score is None else ea.Box(bbox, "helmet", baseline_score),
        None if best_score is None else ea.Box(bbox, "helmet", best_score),
    )


def plan(
    items: list[ea.ComparisonItem],
    *,
    category: str = ea.NEW_FALSE_POSITIVE,
    image_sizes: dict[int, tuple[int, int]] | None = None,
    **overrides: Any,
) -> ea.GridPlan:
    kwargs: dict[str, Any] = {
        "image_sizes": image_sizes or dict.fromkeys(range(1, 13), (100, 100)),
        "baseline_arm": "real_only",
        "best_arm": "filtered_syn",
    }
    kwargs.update(overrides)
    return ea.plan_comparison_grid(category, items, **kwargs)


# --- The mandatory-category guard (EVAL-15) ------------------------------------


def test_assert_reportable_categories_rejects_a_list_without_new_false_positive() -> None:
    """The whole point: a report showing only the wins must be impossible."""

    with pytest.raises(ea.ErrorAnalysisConfigError, match="new_false_positive"):
        ea.assert_reportable_categories(
            ["fixed_false_negative", "fixed_false_positive", "both_wrong"]
        )


def test_assert_reportable_categories_accepts_the_shipped_four() -> None:
    ea.assert_reportable_categories(config()["error_analysis"]["categories"])


def test_assert_reportable_categories_rejects_empty_unknown_and_duplicate() -> None:
    with pytest.raises(ea.ErrorAnalysisConfigError, match="empty"):
        ea.assert_reportable_categories([])
    with pytest.raises(ea.ErrorAnalysisConfigError, match="does not classify"):
        ea.assert_reportable_categories(["new_false_positive", "mystery_category"])
    with pytest.raises(ea.ErrorAnalysisConfigError, match="repeats"):
        ea.assert_reportable_categories(["new_false_positive", "new_false_positive"])


def test_load_error_analysis_config_raises_when_new_false_positive_is_dropped() -> None:
    payload = config()
    payload["error_analysis"]["categories"] = [
        "fixed_false_negative",
        "fixed_false_positive",
        "both_wrong",
    ]
    with pytest.raises(ea.ErrorAnalysisConfigError, match="COST"):
        ea.load_error_analysis_config(payload)


def test_load_error_analysis_config_reads_the_two_arms_in_order() -> None:
    settings = ea.load_error_analysis_config(config())
    # compare_arms is [baseline, best] and the order is load bearing: swapping it
    # turns every "fixed" into a "new".
    assert settings.baseline_arm == "real_only"
    assert settings.best_arm == "filtered_syn"
    assert settings.samples_per_category == 2
    assert settings.categories == (
        "fixed_false_negative",
        "fixed_false_positive",
        "new_false_positive",
        "both_wrong",
    )


def test_load_error_analysis_config_rejects_bad_arm_lists_and_sample_counts() -> None:
    one_arm = config()
    one_arm["error_analysis"]["compare_arms"] = ["real_only"]
    with pytest.raises(ea.ErrorAnalysisConfigError, match="exactly two arms"):
        ea.load_error_analysis_config(one_arm)

    same_arm = config()
    same_arm["error_analysis"]["compare_arms"] = ["real_only", "real_only"]
    with pytest.raises(ea.ErrorAnalysisConfigError, match="twice"):
        ea.load_error_analysis_config(same_arm)

    zero_samples = config()
    zero_samples["error_analysis"]["samples_per_category"] = 0
    with pytest.raises(ea.ErrorAnalysisConfigError, match="must be positive"):
        ea.load_error_analysis_config(zero_samples)

    no_block = {"metrics": {}, "compliance": {}}
    with pytest.raises(ea.ErrorAnalysisConfigError, match="no `error_analysis` block"):
        ea.load_error_analysis_config(no_block)


def test_shipped_config_is_loadable_and_keeps_new_false_positive() -> None:
    """configs/evaluation.yaml itself must satisfy the guard, not just the fixture."""

    settings = ea.load_error_analysis_config()
    assert ea.NEW_FALSE_POSITIVE in settings.categories
    assert settings.samples_per_category > 0
    assert ea.default_iou_threshold() > 0.0
    assert 0.0 < ea.default_score_threshold() < 1.0


def test_the_two_default_thresholds_read_two_different_config_keys() -> None:
    """An operating point and a spatial overlap are different quantities.

    `compliance.score_threshold` is a confidence and `metrics.bare_head_recall_iou`
    is an IoU. The fixture the rest of this file uses happens to set both to 0.50,
    which is precisely the degenerate fixture K-19 warns about: with the two equal,
    each function can read the other's key and every test still passes. Here they
    are set apart so the key each one reads is the thing being observed.
    """

    payload = config()
    payload["compliance"]["score_threshold"] = 0.42
    payload["metrics"]["bare_head_recall_iou"] = 0.77
    assert ea.default_score_threshold(payload) == pytest.approx(0.42)
    assert ea.default_iou_threshold(payload) == pytest.approx(0.77)


# --- The four-way comparison (EVAL-15) -----------------------------------------


def test_four_way_comparison_counts_match_the_fixture_table() -> None:
    """Hand-derived from the role table above `ground_truth()`.

    gt 1 baseline-only -> new_false_negative                       = 1
    gt 2 best-only     -> fixed_false_negative                     = 1
    gt 3 neither       -> both_wrong                               = 1
    gt 4 person, neither arm predicts that class -> both_wrong     = 1  (total 2)
    img3 head FP made by BOTH arms (IoU 1.0) -> both_wrong         = 1  (total 3)
    img3 helmet(5,5) and img4 helmet, baseline only                = 2  fixed_false_positive
    img3 helmet(80,10), best only                                  = 1  new_false_positive
    gt 0 found by both, and the crowd person, produce nothing.
    """

    counts = ea.count_by_category(compare())
    assert counts == {
        "fixed_false_negative": 1,
        "fixed_false_positive": 2,
        "new_false_positive": 1,
        "new_false_negative": 1,
        "both_wrong": 3,
    }
    # Eight items, and gt 0 plus the crowd box account for the rest of the split.
    assert sum(counts.values()) == 8


def test_the_outcome_order_puts_the_cost_categories_where_they_are_read() -> None:
    """`OUTCOMES` is the reporting order of every table in the report, and the
    tables are read top-down. `new_false_positive` sits third, next to the
    `fixed_*` pair it has to be compared against; pushed to the bottom of the
    table it reads as an afterthought, which is the presentational half of the
    selective reporting CLAUDE.md forbids."""

    assert ea.OUTCOMES == (
        "fixed_false_negative",
        "fixed_false_positive",
        "new_false_positive",
        "new_false_negative",
        "both_wrong",
    )


def test_every_outcome_row_exists_even_when_the_outcome_did_not_occur() -> None:
    """A missing row reads as 'not measured'; a zero reads as 'measured, none'."""

    perfect = ea.four_way_comparison(
        ground_truth(), baseline_detections(), baseline_detections(), config=config()
    )
    counts = ea.count_by_category(perfect)
    assert set(counts) == set(ea.OUTCOMES)
    # Two arms with identical predictions disagree about nothing, so only the
    # boxes neither arm found and the FPs both made survive.
    #   gt 2, gt 3 and gt 4 missed by both  -> 3
    #   img3 helmet, img3 head, img4 helmet -> 3
    assert counts["both_wrong"] == 6
    assert counts["fixed_false_negative"] == 0
    assert counts["fixed_false_positive"] == 0
    assert counts["new_false_positive"] == 0
    assert counts["new_false_negative"] == 0


def test_the_new_false_negative_case_is_not_folded_into_both_wrong() -> None:
    """gt 1 was found by the baseline and lost by the best arm. That is a
    regression, and calling it `both_wrong` would hide it in a bucket labelled
    'hard cases'."""

    items = [item for item in compare() if item.category == ea.NEW_FALSE_NEGATIVE]
    assert len(items) == 1
    item = items[0]
    assert (item.image_id, item.class_name) == (1, "helmet")
    assert item.anchor_box == (50.0, 50.0, 30.0, 30.0)
    assert item.best is None
    assert item.baseline is not None and item.baseline.score == pytest.approx(0.80)


def test_fixed_false_negative_carries_the_detection_that_fixed_it() -> None:
    items = [item for item in compare() if item.category == ea.FIXED_FALSE_NEGATIVE]
    assert len(items) == 1
    item = items[0]
    assert (item.image_id, item.class_name) == (2, "head")
    assert item.baseline is None
    assert item.best is not None and item.best.score == pytest.approx(0.70)


def test_a_false_positive_both_arms_make_is_counted_once_as_both_wrong() -> None:
    """Counting it as fixed AND new would double the disagreement total and make
    each arm look worse than the other simultaneously."""

    shared = [
        item
        for item in compare()
        if item.category == ea.BOTH_WRONG and item.anchor == ea.ANCHOR_DETECTION
    ]
    assert len(shared) == 1
    item = shared[0]
    assert (item.image_id, item.class_name) == (3, "head")
    assert item.baseline is not None and item.baseline.score == pytest.approx(0.60)
    assert item.best is not None and item.best.score == pytest.approx(0.55)


def test_score_threshold_changes_the_classification() -> None:
    """The best arm's img2 helmet sits at 0.30.

    At the 0.50 operating point it does not exist, so gt 3 is `both_wrong`.
    Lower the threshold to 0.20 and the same box becomes a hit, moving gt 3 to
    `fixed_false_negative`. A comparison that ignored the threshold would report
    the same counts for both.
    """

    strict = ea.count_by_category(compare(score_threshold=0.50))
    loose = ea.count_by_category(compare(score_threshold=0.20))
    assert strict["both_wrong"] == 3
    assert strict["fixed_false_negative"] == 1
    assert loose["both_wrong"] == 2
    assert loose["fixed_false_negative"] == 2


def test_iou_threshold_boundary_decides_a_hit() -> None:
    """One GT of 10x10 at the origin, one detection of 10x5 at the origin.

    intersection = 10*5 = 50; union = 100 + 50 - 50 = 100; IoU = 0.50 exactly.
    The match test is `>=`, so at 0.50 this is a hit and at 0.51 it is not.
    """

    gt = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [annotation(1, 1, HEAD, (0, 0, 10, 10))],
        "categories": [dict(entry) for entry in CATEGORIES],
    }
    boundary_detection = [detection(1, HEAD, (0, 0, 10, 5), 0.9)]

    at_threshold = ea.four_way_comparison(
        gt, [], boundary_detection, config=config(), iou_threshold=0.50
    )
    assert ea.count_by_category(at_threshold)["fixed_false_negative"] == 1
    assert ea.count_by_category(at_threshold)["new_false_positive"] == 0

    above_threshold = ea.four_way_comparison(
        gt, [], boundary_detection, config=config(), iou_threshold=0.51
    )
    # The GT is now missed by both arms and the box becomes a new false positive.
    assert ea.count_by_category(above_threshold)["fixed_false_negative"] == 0
    assert ea.count_by_category(above_threshold)["both_wrong"] == 1
    assert ea.count_by_category(above_threshold)["new_false_positive"] == 1


def test_the_default_thresholds_reach_the_comparison_from_the_config_block() -> None:
    """Not just that the two loaders read the right keys — that the comparison
    USES them. Hand-derived at a 0.75 operating point, where the baseline keeps
    only its 0.90 and 0.80 boxes and the best arm only its 0.95 and 0.90:

      gt 1 helmet: baseline still hits it, best no longer does -> new_false_negative
      gt 2, gt 3, gt 4: missed by both                         -> both_wrong x3
      img3 helmet(80,10) 0.90, best only                       -> new_false_positive
      every other detection is now below the operating point, so the two
      `fixed_*` categories empty out entirely.

    At the fixture's own 0.50 the counts are 1/2/1/1/3, so a default that read
    `metrics.bare_head_recall_iou` (0.50) instead would produce the other table.
    """

    payload = config()
    payload["compliance"]["score_threshold"] = 0.75
    counts = ea.count_by_category(
        ea.four_way_comparison(
            ground_truth(), baseline_detections(), best_detections(), config=payload
        )
    )
    assert counts == {
        "fixed_false_negative": 0,
        "fixed_false_positive": 0,
        "new_false_positive": 1,
        "new_false_negative": 1,
        "both_wrong": 3,
    }


def test_the_default_iou_reaches_the_comparison_from_the_metrics_block() -> None:
    """The same box at the same score, decided only by the IoU key.

    One 10x10 GT and one 10x5 detection: intersection 50, union 100, IoU 0.50.
    With `metrics.bare_head_recall_iou` at 0.50 that is a hit; at 0.51 the same
    box is a miss plus a false positive. If this default came from
    `compliance.score_threshold` (0.50 in both configs) the two would agree.
    """

    gt = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [annotation(1, 1, HEAD, (0, 0, 10, 10))],
        "categories": [dict(entry) for entry in CATEGORIES],
    }
    half_overlap = [detection(1, HEAD, (0, 0, 10, 5), 0.9)]

    at_threshold = config()
    at_threshold["metrics"]["bare_head_recall_iou"] = 0.50
    counts = ea.count_by_category(
        ea.four_way_comparison(gt, [], half_overlap, config=at_threshold)
    )
    assert counts["fixed_false_negative"] == 1
    assert counts["new_false_positive"] == 0

    above_threshold = config()
    above_threshold["metrics"]["bare_head_recall_iou"] = 0.51
    counts = ea.count_by_category(
        ea.four_way_comparison(gt, [], half_overlap, config=above_threshold)
    )
    assert counts["fixed_false_negative"] == 0
    assert counts["both_wrong"] == 1
    assert counts["new_false_positive"] == 1


def test_a_detection_exactly_at_the_operating_point_is_kept_by_both_modules() -> None:
    """`detection.py` keeps a box when `score >= threshold`.

    This module has to make the same call on the boundary or the error grid and
    the EVAL-16 table disagree about which boxes exist at the operating point —
    silently, and only for the boxes that land exactly on it. So the boundary is
    asserted here against the OTHER module's answer, not against a hand-copied
    rule that could drift with it.
    """

    gt = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [],
        "categories": [dict(entry) for entry in CATEGORIES],
    }
    on_the_point = [detection(1, HELMET, (0, 0, 10, 10), 0.50)]

    kept = ea.detection_records(gt, on_the_point, score_threshold=0.50)
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.50)
    report = hard_negative_false_positives_per_image(
        gt, on_the_point, [1], config=config(), score_threshold=0.50
    )
    assert report.n_false_positives == 1

    below = [detection(1, HELMET, (0, 0, 10, 10), 0.4999)]
    assert ea.detection_records(gt, below, score_threshold=0.50) == ()
    assert (
        hard_negative_false_positives_per_image(
            gt, below, [1], config=config(), score_threshold=0.50
        ).n_false_positives
        == 0
    )


def test_crowd_annotations_produce_no_items() -> None:
    """The crowd `person` on image 1 is never detected by either arm. If it were
    counted it would add a fifth `both_wrong`, and the report would disagree
    with COCOeval, which ignores crowds."""

    crowd_items = [
        item
        for item in compare()
        if item.class_name == "person" and item.image_id == 1
    ]
    assert crowd_items == []


def test_a_detection_with_an_unknown_category_id_raises_even_below_threshold() -> None:
    """Validate first, filter second. The other order makes a broken prediction
    file MORE acceptable the lower its scores are."""

    broken = [*best_detections(), detection(1, 99, (0, 0, 5, 5), 0.01)]
    with pytest.raises(UnknownCategoryError, match="99"):
        ea.four_way_comparison(ground_truth(), baseline_detections(), broken, config=config())


def test_duplicate_detections_on_one_box_are_one_hit_and_one_false_positive() -> None:
    """Otherwise a model that draws the same box twice buys two hits and
    `fixed_false_positive` stops meaning anything."""

    gt = {
        "images": [{"id": 1, "width": 100, "height": 100}],
        "annotations": [annotation(1, 1, HEAD, (10, 10, 20, 20))],
        "categories": [dict(entry) for entry in CATEGORIES],
    }
    doubled = [
        detection(1, HEAD, (10, 10, 20, 20), 0.9),
        detection(1, HEAD, (10, 10, 20, 20), 0.8),
    ]
    counts = ea.count_by_category(
        ea.four_way_comparison(gt, [], doubled, config=config())
    )
    assert counts["fixed_false_negative"] == 1
    assert counts["new_false_positive"] == 1


def test_pair_false_positives_takes_the_highest_iou_first() -> None:
    """Baseline FP A overlaps best FP B by more than best FP C.

    A = (0,0,10,10). B = (0,0,10,10) -> IoU 1.00. C = (0,0,10,8):
    intersection 80, union 100+80-80 = 100, IoU 0.80. Pairing A with C instead
    would leave B as a `new_false_positive` and C as `both_wrong`, which is the
    opposite of the truth; the greedy pass must consider B first regardless of
    the order the lists arrive in.
    """

    a = ea.DetectionRecord(0, 1, "head", (0.0, 0.0, 10.0, 10.0), 0.9)
    c = ea.DetectionRecord(0, 1, "head", (0.0, 0.0, 10.0, 8.0), 0.7)
    b = ea.DetectionRecord(1, 1, "head", (0.0, 0.0, 10.0, 10.0), 0.6)
    pairs, left_only, right_only = ea.pair_false_positives(
        [a], [c, b], iou_threshold=0.50
    )
    assert len(pairs) == 1
    assert pairs[0][1].bbox == (0.0, 0.0, 10.0, 10.0)
    assert left_only == []
    assert [record.bbox for record in right_only] == [(0.0, 0.0, 10.0, 8.0)]


def test_false_positives_of_different_classes_never_pair() -> None:
    """A `helmet` hallucination and a `head` hallucination in the same place are
    two different errors, and pairing them would report both arms as making the
    same mistake."""

    helmet_fp = ea.DetectionRecord(0, 1, "helmet", (0.0, 0.0, 10.0, 10.0), 0.9)
    head_fp = ea.DetectionRecord(0, 1, "head", (0.0, 0.0, 10.0, 10.0), 0.9)
    pairs, left_only, right_only = ea.pair_false_positives(
        [helmet_fp], [head_fp], iou_threshold=0.50
    )
    assert pairs == []
    assert len(left_only) == 1
    assert len(right_only) == 1


def test_two_false_positives_at_exactly_the_iou_threshold_are_one_shared_error() -> None:
    """The pairing boundary, which decides `both_wrong` against `fixed`+`new`.

    A = (0,0,10,10) from the baseline, B = (0,0,10,5) from the best arm:
    intersection 50, union 100, IoU 0.50 exactly. `match_arm` treats 0.50 as a
    match, so the pairing must too — otherwise the same overlap counts as a hit
    on the ground-truth side and as two separate hallucinations on the
    false-positive side, and the arms are reported as making one mistake each
    when they made the same one.
    """

    baseline_fp = ea.DetectionRecord(0, 1, "head", (0.0, 0.0, 10.0, 10.0), 0.9)
    best_fp = ea.DetectionRecord(0, 1, "head", (0.0, 0.0, 10.0, 5.0), 0.8)

    pairs, left_only, right_only = ea.pair_false_positives(
        [baseline_fp], [best_fp], iou_threshold=0.50
    )
    assert len(pairs) == 1
    assert (left_only, right_only) == ([], [])

    # One notch above the boundary they become two separate errors.
    pairs, left_only, right_only = ea.pair_false_positives(
        [baseline_fp], [best_fp], iou_threshold=0.51
    )
    assert pairs == []
    assert len(left_only) == 1 and len(right_only) == 1


def test_count_by_category_and_class_derives_its_columns_when_given_none() -> None:
    """The default branch. `build_analysis` passes the COCO class list, so
    nothing else here exercises the path that derives the columns from the items
    — and a default that produced no zero rows would look identical on any
    category whose classes all occurred."""

    table = ea.count_by_category_and_class(compare())
    assert set(table) == set(ea.OUTCOMES)
    # Every class that appears anywhere becomes a column on EVERY row, including
    # the rows where it did not occur: `{"helmet": 2}` and
    # `{"helmet": 2, "head": 0, "person": 0}` read very differently.
    for outcome in ea.OUTCOMES:
        assert sorted(table[outcome]) == ["head", "helmet", "person"]
    assert table["fixed_false_positive"] == {"helmet": 2, "head": 0, "person": 0}
    assert table["fixed_false_negative"] == {"helmet": 0, "head": 1, "person": 0}
    assert table["both_wrong"] == {"helmet": 1, "head": 1, "person": 1}


def test_count_by_category_and_class_keeps_person_visible() -> None:
    table = ea.count_by_category_and_class(compare(), ["helmet", "head", "person"])
    assert set(table) == set(ea.OUTCOMES)
    # gt 1 is a helmet; gt 2 is a head.
    assert table["new_false_negative"] == {"helmet": 1, "head": 0, "person": 0}
    assert table["fixed_false_negative"] == {"helmet": 0, "head": 1, "person": 0}
    # img3 helmet(5,5) + img4 helmet
    assert table["fixed_false_positive"]["helmet"] == 2
    assert table["fixed_false_positive"]["head"] == 0
    # gt 4 is the `person` neither arm found. It survives into the table rather
    # than being filtered out for not being a primary class.
    assert table["both_wrong"]["person"] == 1


# --- Sampling ------------------------------------------------------------------


def test_ground_truth_items_sort_by_ascending_area_and_detections_by_score() -> None:
    """AP_small is the headline, so the smallest missed box goes in the figure
    first; a confident hallucination outranks a marginal one."""

    small_gt = ea.ComparisonItem(
        ea.BOTH_WRONG, 1, "head", ea.ANCHOR_GROUND_TRUTH, (0, 0, 10, 10), None, None
    )
    big_gt = ea.ComparisonItem(
        ea.BOTH_WRONG, 1, "head", ea.ANCHOR_GROUND_TRUTH, (0, 0, 40, 40), None, None
    )
    weak_fp = ea.ComparisonItem(
        ea.BOTH_WRONG,
        1,
        "head",
        ea.ANCHOR_DETECTION,
        (0, 0, 10, 10),
        None,
        ea.Box((0, 0, 10, 10), "head", 0.55),
    )
    strong_fp = ea.ComparisonItem(
        ea.BOTH_WRONG,
        1,
        "head",
        ea.ANCHOR_DETECTION,
        (0, 0, 10, 10),
        None,
        ea.Box((0, 0, 10, 10), "head", 0.95),
    )
    ordered = sorted([weak_fp, big_gt, strong_fp, small_gt], key=ea.sample_sort_key)
    # Ground truth (100 px^2 then 1600 px^2), then detections (0.95 then 0.55).
    assert ordered == [small_gt, big_gt, strong_fp, weak_fp]


def test_select_samples_caps_each_category_and_fills_every_configured_key() -> None:
    settings = ea.load_error_analysis_config(config())
    samples = ea.select_samples(
        compare(), categories=settings.categories, limit=settings.samples_per_category
    )
    assert set(samples) == set(settings.categories)
    # fixed_false_positive has 2 items and the limit is 2, so both survive; the
    # single-item categories are unaffected.
    assert len(samples["fixed_false_positive"]) == 2
    assert len(samples["new_false_positive"]) == 1

    capped = ea.select_samples(compare(), categories=settings.categories, limit=1)
    assert len(capped["fixed_false_positive"]) == 1
    # Deterministic: the higher-scoring baseline FP (0.70 on image 3) wins over
    # the 0.55 on image 4.
    kept = capped["fixed_false_positive"][0]
    assert (kept.image_id, kept.baseline.score) == (3, pytest.approx(0.70))


def test_select_samples_refuses_a_category_list_without_new_false_positive() -> None:
    with pytest.raises(ea.ErrorAnalysisConfigError, match="new_false_positive"):
        ea.select_samples(compare(), categories=["both_wrong"], limit=2)


def test_select_samples_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ea.ErrorAnalysisConfigError, match="must be positive"):
        ea.select_samples(compare(), categories=["new_false_positive"], limit=0)


# --- Rendering -----------------------------------------------------------------


def test_crop_window_is_square_centred_and_inside_the_image() -> None:
    """A 20x20 box at (10,10) in a 100x100 image.

    side = max(20,20) * CROP_ZOOM(3.0) = 60, which beats MIN_CROP_PX and fits.
    centre = (20,20); left = round(20 - 30) = -10 -> clamped to 0; same for top.
    """

    left, top, span = ea._crop_window((10, 10, 20, 20), (100, 100))
    assert (left, top, span) == (0, 0, 60)

    # Non-square on purpose: the window follows the LONGER side, so a 30x10 box
    # gives 30*3 = 90 and not 10*3 = 30. An all-square fixture cannot tell
    # `max(w, h)` from `min(w, h)`.
    _, _, span = ea._crop_window((35, 45, 30, 10), (100, 100))
    assert span == 90

    # Same box pushed against the far corner: centre (90,90), left = 60, and the
    # clamp keeps left + span == 100 rather than running off the array.
    left, top, span = ea._crop_window((80, 80, 20, 20), (100, 100))
    assert (left, top, span) == (40, 40, 60)
    assert left + span == 100

    # A 2x2 box: 2*3 = 6 is below MIN_CROP_PX(48), so the floor decides.
    _, _, span = ea._crop_window((50, 50, 2, 2), (100, 100))
    assert span == int(ea.MIN_CROP_PX)

    # A box bigger than the image cannot produce a window bigger than the image.
    _, _, span = ea._crop_window((0, 0, 90, 90), (100, 100))
    assert span == 100


def test_crop_window_reads_image_size_as_width_then_height() -> None:
    """The clamp is what distinguishes them, so the box sits near the RIGHT edge
    of a 120x80 image (DATA-25: the real Test split is not square).

    12x12 box at (100,10): side = 12*3 = 36, below MIN_CROP_PX, so span = 48.
    centre = (105,15); left = round(105-24) = 81, clamped to 120-48 = 72.
    top = round(15-24) = -9, clamped up to 0.
    Read transposed, the same call clamps left to 80-48 = 32 instead, and the
    crop shows a stretch of image 40 px away from the error it is illustrating.
    """

    assert ea._crop_window((100, 10, 10, 10), WIDE_SIZE) == (72, 0, 48)
    assert ea._crop_window((100, 10, 10, 10), (80, 120)) == (32, 0, 48)


def test_array_size_is_width_then_height() -> None:
    """numpy is (rows, columns) and every image API here is (width, height)."""

    assert ea._array_size(np.zeros((80, 120, 3), dtype=np.uint8)) == (120, 80)


def test_visible_context_boxes_keeps_only_what_the_crop_shows() -> None:
    """A ground-truth box outside the window would be drawn at negative
    coordinates and pull the panel's axes out to include it, shrinking the crop
    it was supposed to annotate."""

    crop = (52, 0, 48)  # x 52..100, y 0..48
    inside = ea.Box((78.0, 8.0, 12.0, 12.0), "helmet", None)
    outside = ea.Box((0.0, 60.0, 10.0, 10.0), "head", None)
    touching = ea.Box((40.0, 10.0, 12.0, 10.0), "head", None)  # ends exactly at x=52
    assert ea._visible_context_boxes([inside, outside, touching], crop) == (inside,)


# --- What the grid decides to draw (EVAL-15) -----------------------------------


def test_the_two_panels_of_a_cell_are_the_two_DIFFERENT_arms() -> None:
    """"Baseline and best side by side" is the whole point of the figure. Both
    columns showing the baseline is a figure that cannot show a difference, and
    it looks entirely normal."""

    item = detection_item(1, (40.0, 40.0, 12.0, 12.0), baseline_score=None, best_score=0.9)
    grid = plan([item])

    assert [panel.arm for panel in grid.panels] == ["real_only", "filtered_syn"]
    assert [panel.side for panel in grid.panels] == ["baseline", "best"]
    assert [panel.column for panel in grid.panels] == [0, 1]
    assert grid.panels[0].label.startswith("real_only\n")
    assert grid.panels[1].label.startswith("filtered_syn\n")


def test_each_panel_draws_its_OWN_arms_box_and_its_own_score() -> None:
    """The shared false positive from the fixture: the baseline made it at 0.60
    and the best arm at 0.55. A panel that drew the other side's box would show
    the same rectangle twice and caption it with the wrong confidence — the
    reader would conclude the arms agreed exactly, which is the thing the figure
    exists to check."""

    shared = [
        item
        for item in compare()
        if item.category == ea.BOTH_WRONG and item.anchor == ea.ANCHOR_DETECTION
    ]
    left, right = plan(shared, category=ea.BOTH_WRONG).panels

    assert left.detection is not None and left.detection.score == pytest.approx(0.60)
    assert right.detection is not None and right.detection.score == pytest.approx(0.55)
    assert "score 0.60" in left.label
    assert "score 0.55" in right.label


def test_every_cell_label_names_the_arm_image_class_score_and_category() -> None:
    """EVAL-15 asks for all five on every cell. Spelled out in full rather than
    checked for substrings: a caption that lost its category, or lost itself
    entirely, is a grid of anonymous thumbnails."""

    with_detection = detection_item(3, (80.0, 10.0, 15.0, 15.0), best_score=0.9)
    left, right = plan([with_detection]).panels
    assert left.label == "real_only\nimg 3 · helmet\nno detection · new_false_positive"
    assert right.label == "filtered_syn\nimg 3 · helmet\nscore 0.90 · new_false_positive"


def test_the_crop_window_follows_the_anchor_box() -> None:
    """The anchor box IS the error being illustrated. A crop that always showed
    (0,0) would render twelve pictures of the top-left corner of twelve images,
    each captioned with an error that is not in shot.

    15x15 box at (80,10) in a 100x100 image: side = 45 -> MIN_CROP_PX 48;
    centre = (87.5,17.5); left = round(63.5) = 64 clamped to 100-48 = 52;
    top = round(-6.5) = -6 clamped up to 0.
    """

    grid = plan([detection_item(3, (80.0, 10.0, 15.0, 15.0))])
    assert [panel.crop for panel in grid.panels] == [(52, 0, 48), (52, 0, 48)]
    # And the crop follows the image it is cropping, not a square assumption.
    wide = plan(
        [detection_item(3, (100.0, 10.0, 10.0, 10.0))], image_sizes={3: WIDE_SIZE}
    )
    assert wide.panels[0].crop == (72, 0, 48)


def test_the_ground_truth_context_boxes_reach_every_panel() -> None:
    """Without them a cell is a coloured rectangle floating on a crop, and the
    reader cannot see whether the box was near anything real. This is the panel
    of `both_wrong` and `new_false_positive` doing its job."""

    context = {3: [ea.Box((78.0, 8.0, 12.0, 12.0), "helmet", None)]}
    grid = plan(
        [detection_item(3, (80.0, 10.0, 15.0, 15.0))], ground_truth_boxes=context
    )
    for panel in grid.panels:
        assert panel.context_boxes == (ea.Box((78.0, 8.0, 12.0, 12.0), "helmet", None),)
        assert panel.context_colour == ea.GT_COLOUR
        assert panel.detection_colour == ea.DETECTION_COLOUR


def test_the_markdown_legend_names_the_colours_the_panels_actually_use(
    tmp_path: Path,
) -> None:
    """The report prints "Dashed blue is ground truth, solid orange is that
    arm's detection". Swap the two constants and that sentence becomes false —
    every grid in the report is then read backwards, with the model's mistakes
    taken for the annotations and vice versa. So the sentence and the channels
    are asserted together."""

    def channels(colour: str) -> tuple[int, int, int]:
        value = colour.lstrip("#")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))

    ground_truth_r, ground_truth_g, ground_truth_b = channels(ea.GT_COLOUR)
    assert ground_truth_b > ground_truth_g > ground_truth_r, "ground truth must be BLUE"
    detection_r, detection_g, detection_b = channels(ea.DETECTION_COLOUR)
    assert detection_r > detection_g > detection_b, "a detection must be ORANGE"

    markdown = ea.render_markdown(build(tmp_path))
    assert (
        "Dashed blue is ground truth, solid orange is that arm's detection" in markdown
    )


def test_the_subtitle_says_how_many_of_the_whole_category_are_shown() -> None:
    """12 of 931 and 12 of 12 are opposite claims about the same figure. The
    real `new_false_negative` category held 1,304 items and the grid shows
    twelve of them; a subtitle that counted only what it drew would report the
    sample as the population."""

    grid = plan([detection_item(3, (80.0, 10.0, 15.0, 15.0))], total_in_category=931)
    assert grid.subtitle.startswith(
        "1 of 931 shown · left: real_only · right: filtered_syn"
    )
    assert grid.title.startswith("new_false_positive — 1 of 931 shown")


def test_only_the_mandatory_category_carries_the_cost_subtitle() -> None:
    """`new_false_positive` is the COST of synthetic data, and its grid says so.
    Moving that line onto the other three would put the warning everywhere
    except the one panel it is about, which reads as boilerplate and defeats it."""

    cost = " · this category is the COST of synthetic data and is never optional"
    mandatory = plan([detection_item(3, (80.0, 10.0, 15.0, 15.0))])
    assert mandatory.subtitle.endswith(cost)
    for category in (ea.BOTH_WRONG, ea.FIXED_FALSE_NEGATIVE, ea.FIXED_FALSE_POSITIVE):
        other = plan([detection_item(3, (80.0, 10.0, 15.0, 15.0))], category=category)
        assert "COST" not in other.subtitle
        assert other.title.startswith(f"{category} — 1 of 1 shown")


def test_an_odd_number_of_items_still_gets_a_row_to_live_on() -> None:
    """Two items per row, so three items need TWO rows. Floor division gives one,
    and the third item is then written to `axes[1]` of a one-row grid: an
    IndexError in production for every odd count above one, which is 3, 5, 7, 9
    and 11 of the configured 12."""

    grid = plan([detection_item(image_id, (40.0, 40.0, 12.0, 12.0)) for image_id in (1, 2, 3)])
    assert (grid.rows, grid.columns) == (2, 4)
    assert [(panel.row, panel.column) for panel in grid.panels] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
    ]


def test_an_odd_number_of_items_renders_without_running_off_the_grid(
    tmp_path: Path,
) -> None:
    """The same thing end to end, because the failure mode is an exception at
    report time and no plan assertion proves the axes array was big enough."""

    image_paths = write_images(tmp_path / "images", image_ids=(1, 2, 3))
    destination = ea.render_comparison_grid(
        ea.NEW_FALSE_POSITIVE,
        [detection_item(image_id, (40.0, 40.0, 12.0, 12.0)) for image_id in (1, 2, 3)],
        image_paths=image_paths,
        destination=tmp_path / "odd.png",
        baseline_arm="real_only",
        best_arm="filtered_syn",
    )
    assert destination.is_file()


def test_the_rendered_pixels_show_the_crop_around_the_box(tmp_path: Path) -> None:
    """Pixels, sampled by colour rather than weighed by file size.

    A 120x80 image — deliberately not square — that is black except for a red
    marker in the top-left corner and a green marker inside the anchor box at
    the right-hand edge. The crop window is (72,0,48), so the figure must
    contain the green marker and must not contain a single red pixel. A crop
    pinned to (0,0) shows red and no green; a transposed image size clamps to
    x=32 and shows neither.
    """

    array = np.zeros((80, 120, 3), dtype=np.uint8)
    array[0:12, 0:12] = (255, 0, 0)
    array[10:22, 100:112] = (0, 255, 0)
    source = tmp_path / "wide.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(source)

    destination = ea.render_comparison_grid(
        ea.NEW_FALSE_POSITIVE,
        [detection_item(1, (100.0, 10.0, 10.0, 10.0))],
        image_paths={1: source},
        destination=tmp_path / "grid.png",
        baseline_arm="real_only",
        best_arm="filtered_syn",
    )
    rendered = np.asarray(Image.open(destination).convert("RGB"))
    green = int((rendered == (0, 255, 0)).all(axis=-1).sum())
    red = int((rendered == (255, 0, 0)).all(axis=-1).sum())
    assert green > 100, "the crop does not contain the box it is supposed to show"
    assert red == 0, "the crop shows the top-left corner instead of the box"


def test_plan_comparison_grid_refuses_an_item_whose_image_size_is_unknown() -> None:
    with pytest.raises(ea.FigureInputError, match="No image size"):
        plan([detection_item(99, (0.0, 0.0, 10.0, 10.0))], image_sizes={1: (100, 100)})


def test_the_panel_drawn_is_the_panel_that_was_planned() -> None:
    """The renderer between the plan and the PNG is twenty lines of
    transcription, and matplotlib will happily report success on any of them.
    Rather than guess from pixels, this asks the axes what it was given: the
    caption, the cropped array, and both rectangles with their colours and their
    positions relative to the crop origin.

    Crop is (52,0,48). The ground-truth box at (78,8) therefore lands at (26,8)
    on the panel and the detection at (80,10) lands at (28,10).
    """

    # Imported here, not at module scope: `error_analysis` selects the Agg
    # backend at import time and pyplot must not be imported before it.
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolours

    context = {3: [ea.Box((78.0, 8.0, 12.0, 12.0), "helmet", None)]}
    grid = plan(
        [detection_item(3, (80.0, 10.0, 15.0, 15.0))], ground_truth_boxes=context
    )
    panel = grid.panels[1]
    image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)

    figure, axis = plt.subplots()
    try:
        ea._draw_panel(axis, image, panel)

        assert axis.get_title() == panel.label
        assert axis.get_title().endswith("score 0.90 · new_false_positive")

        drawn_crop = np.asarray(axis.images[0].get_array())
        assert np.array_equal(drawn_crop, image[0:48, 52:100])

        ground_truth_patch, detection_patch = axis.patches
        assert ground_truth_patch.get_xy() == (26.0, 8.0)
        assert (ground_truth_patch.get_width(), ground_truth_patch.get_height()) == (
            12.0,
            12.0,
        )
        assert ground_truth_patch.get_edgecolor() == mcolours.to_rgba(ea.GT_COLOUR)
        assert ground_truth_patch.get_linestyle() != "solid", "ground truth is dashed"

        assert detection_patch.get_xy() == (28.0, 10.0)
        assert detection_patch.get_edgecolor() == mcolours.to_rgba(ea.DETECTION_COLOUR)
        assert detection_patch.get_linestyle() == "solid", "a detection is solid"
    finally:
        plt.close(figure)


def test_a_panel_with_no_detection_draws_only_the_ground_truth() -> None:
    """The baseline side of a `new_false_positive`: there is no box to draw, and
    inventing one would show the reader a detection that arm never made."""

    import matplotlib.pyplot as plt

    context = {3: [ea.Box((78.0, 8.0, 12.0, 12.0), "helmet", None)]}
    grid = plan(
        [detection_item(3, (80.0, 10.0, 15.0, 15.0))], ground_truth_boxes=context
    )
    figure, axis = plt.subplots()
    try:
        ea._draw_panel(axis, image=np.zeros((100, 100, 3), dtype=np.uint8), panel=grid.panels[0])
        assert len(axis.patches) == 1
        assert axis.get_title().endswith("no detection · new_false_positive")
    finally:
        plt.close(figure)


def test_render_comparison_grid_writes_a_small_png(tmp_path: Path) -> None:
    image_paths = write_images(tmp_path / "images")
    settings = ea.load_error_analysis_config(config())
    samples = ea.select_samples(
        compare(), categories=settings.categories, limit=settings.samples_per_category
    )
    destination = tmp_path / "figures" / "new_false_positive.png"
    written = ea.render_comparison_grid(
        ea.NEW_FALSE_POSITIVE,
        samples[ea.NEW_FALSE_POSITIVE],
        image_paths=image_paths,
        destination=destination,
        baseline_arm="real_only",
        best_arm="filtered_syn",
    )
    assert written == destination
    assert destination.is_file()
    # CLAUDE.md keeps only small images in the project folder.
    assert destination.stat().st_size < 400_000


def test_a_full_grid_stays_inside_the_repo_size_budget(tmp_path: Path) -> None:
    """CLAUDE.md keeps only small images in the project folder, and four of these
    grids ship together.

    Worst case on purpose: twelve items over per-pixel-noise images, which is
    less compressible than any real site photo (the same grid over the actual
    Test images measures ~137 KiB). Measured on this fixture:

        interpolation="nearest"   293 KiB
        interpolation="bilinear"  438 KiB

    Upscaling a crop bilinearly invents intermediate values that PNG cannot pack,
    so the budget sits between the two. It is a real constraint, not a style
    preference, and it is the reason the module pins the interpolation mode.
    """

    rng = np.random.default_rng(7)
    directory = tmp_path / "noise"
    directory.mkdir()
    image_paths = {}
    for image_id in range(1, 13):
        array = rng.integers(0, 255, size=(416, 416, 3), dtype=np.uint8)
        path = directory / f"{image_id}.png"
        Image.fromarray(array).save(path)
        image_paths[image_id] = path

    items = [
        ea.ComparisonItem(
            ea.NEW_FALSE_POSITIVE,
            image_id,
            "helmet",
            ea.ANCHOR_DETECTION,
            (140.0 + image_id, 160.0, 24.0, 24.0),
            None,
            ea.Box((140.0 + image_id, 160.0, 24.0, 24.0), "helmet", 0.9),
        )
        for image_id in range(1, 13)
    ]
    destination = ea.render_comparison_grid(
        ea.NEW_FALSE_POSITIVE,
        items,
        image_paths=image_paths,
        destination=tmp_path / "grid.png",
        baseline_arm="real_only",
        best_arm="filtered_syn",
    )
    assert destination.stat().st_size < 360_000


def test_render_comparison_grid_raises_rather_than_drawing_a_blank_panel(
    tmp_path: Path,
) -> None:
    """A blank cell reads as 'the model saw nothing there', which is a lie."""

    image_paths = write_images(tmp_path / "images", image_ids=(1, 2))
    settings = ea.load_error_analysis_config(config())
    samples = ea.select_samples(
        compare(), categories=settings.categories, limit=settings.samples_per_category
    )
    with pytest.raises(ea.FigureInputError, match="No image path"):
        ea.render_comparison_grid(
            ea.NEW_FALSE_POSITIVE,
            samples[ea.NEW_FALSE_POSITIVE],  # lives on image 3
            image_paths=image_paths,
            destination=tmp_path / "out.png",
            baseline_arm="real_only",
            best_arm="filtered_syn",
        )


def test_render_comparison_grid_refuses_an_empty_item_list(tmp_path: Path) -> None:
    with pytest.raises(ea.FigureInputError, match="No items"):
        ea.render_comparison_grid(
            ea.NEW_FALSE_POSITIVE,
            [],
            image_paths={},
            destination=tmp_path / "out.png",
            baseline_arm="real_only",
            best_arm="filtered_syn",
        )


def test_render_all_grids_refuses_to_skip_new_false_positive(tmp_path: Path) -> None:
    image_paths = write_images(tmp_path / "images")
    samples = ea.select_samples(
        compare(),
        categories=["fixed_false_negative", "new_false_positive", "both_wrong"],
        limit=2,
    )
    samples.pop(ea.NEW_FALSE_POSITIVE)
    with pytest.raises(ea.ErrorAnalysisConfigError, match="new_false_positive"):
        ea.render_all_grids(
            samples,
            image_paths=image_paths,
            output_dir=tmp_path / "figures",
            baseline_arm="real_only",
            best_arm="filtered_syn",
        )


# --- Hard negatives (EVAL-16) --------------------------------------------------


def test_hard_negative_image_ids_are_derived_not_listed() -> None:
    """Image 3 has no annotations at all; image 4 has only a `person`, which is
    not a primary class (ADR-003), so a `helmet` fired there is exactly the
    false positive this number is for. Images 1 and 2 carry primary GT."""

    assert ea.hard_negative_image_ids(ground_truth(), config=config()) == (3, 4)


def test_a_split_with_no_hard_negatives_reports_none_rather_than_crashing() -> None:
    """The real frozen Test split is exactly this case: all 744 images carry a
    `helmet` or a `head`. EVAL-16 anticipates it, so a derived-empty subset
    returns no rows and the report explains itself."""

    gt = ground_truth()
    gt["images"] = [image for image in gt["images"] if int(image["id"]) in {1, 2}]
    gt["annotations"] = [
        a for a in gt["annotations"] if int(a["image_id"]) in {1, 2}
    ]
    assert ea.hard_negative_image_ids(gt, config=config()) == ()
    assert ea.hard_negative_rows(gt, detections_by_arm(), config=config()) == ()


def test_an_explicitly_empty_subset_is_still_a_caller_error() -> None:
    """Different from the derived-empty case: the caller asked for a rate over a
    subset they built, and that subset is empty."""

    from src.evaluation.detection import HardNegativeSubsetError

    with pytest.raises(HardNegativeSubsetError, match="empty"):
        ea.hard_negative_rows(
            ground_truth(), detections_by_arm(), image_ids=[], config=config()
        )


def test_hard_negative_rows_are_hand_derived() -> None:
    """Score threshold 0.50, subset {3, 4}, primary classes helmet + head.

    baseline: img3 helmet 0.70, img3 head 0.60, img4 helmet 0.55  -> 3 / 2 = 1.5
    best:     img3 head 0.55, img3 helmet 0.90                    -> 2 / 2 = 1.0
    """

    ordered = ea.hard_negative_rows(ground_truth(), detections_by_arm(), config=config())
    # Alphabetical by arm, not dict-insertion order: this tuple goes straight
    # into reports/error_analysis.json, and an order that depended on which arm
    # happened to be read first would make every regeneration a diff.
    assert [row.arm for row in ordered] == ["filtered_syn", "real_only"]
    rows = {row.arm: row for row in ordered}
    assert rows["real_only"].n_false_positives == 3
    assert rows["real_only"].n_images == 2
    assert rows["real_only"].false_positives_per_image == pytest.approx(1.5)
    assert rows["filtered_syn"].n_false_positives == 2
    assert rows["filtered_syn"].false_positives_per_image == pytest.approx(1.0)


# --- Slices and targeting (EVAL-17) --------------------------------------------


def test_subset_ground_truth_keeps_every_category() -> None:
    """Dropping an absent class would renumber the categories and silently change
    what AP averages over.

    Image 2 is the discriminating slice: it carries `helmet` and `head` and no
    `person` at all, so a subset that kept only the classes it observed would
    come back two categories long and nothing downstream would complain.
    """

    subset = ea.subset_ground_truth(ground_truth(), [2])
    assert [entry["name"] for entry in subset["categories"]] == [
        "helmet",
        "head",
        "person",
    ]
    assert {int(image["id"]) for image in subset["images"]} == {2}
    assert {int(a["image_id"]) for a in subset["annotations"]} == {2}
    assert "person" not in {
        entry["name"]
        for entry in CATEGORIES
        if entry["id"] in {int(a["category_id"]) for a in subset["annotations"]}
    }


def test_slice_metric_table_is_hand_derived() -> None:
    """`primary_map` over helmet + head, per slice, with no score threshold
    (mAP integrates over confidence).

    small_object = {image 1}: GT head(10,10,20,20) and helmet(50,50,30,30).
      real_only detects BOTH exactly     -> AP(head)=1, AP(helmet)=1 -> 1.0
      filtered_syn detects only the head -> AP(head)=1, AP(helmet)=0 -> 0.5
    crowded = {image 2}: GT head(10,10,10,10) and helmet(70,70,20,20).
      real_only has no detections at all -> 0.0
      filtered_syn detects both exactly (the 0.30 helmet counts here) -> 1.0
    """

    slices = {
        "small_object": frozenset({1}),
        "crowded": frozenset({2}),
        "low_light": frozenset(),
    }
    rows = ea.slice_metric_table(
        ground_truth(), detections_by_arm(), slices, config=config()
    )
    table = {(row.slice_name, row.arm): row for row in rows}
    assert table[("small_object", "real_only")].value == pytest.approx(1.0, abs=1e-6)
    assert table[("small_object", "filtered_syn")].value == pytest.approx(0.5, abs=1e-6)
    assert table[("crowded", "real_only")].value == pytest.approx(0.0, abs=1e-6)
    assert table[("crowded", "filtered_syn")].value == pytest.approx(1.0, abs=1e-6)
    # An empty slice is undefined, not zero.
    assert table[("low_light", "real_only")].value == UNDEFINED
    assert table[("low_light", "real_only")].n_images == 0
    # Instance counts travel with the metric (EVAL-08): image 1 carries two
    # non-crowd boxes plus the crowd person, all three of which are annotations.
    assert table[("small_object", "real_only")].n_instances == 3


def test_slice_deltas_skip_the_undefined_slice() -> None:
    slices = {
        "small_object": frozenset({1}),
        "crowded": frozenset({2}),
        "low_light": frozenset(),
    }
    rows = ea.slice_metric_table(
        ground_truth(), detections_by_arm(), slices, config=config()
    )
    deltas = ea.slice_deltas(rows, baseline_arm="real_only", best_arm="filtered_syn")
    assert set(deltas) == {"small_object", "crowded"}
    assert deltas["small_object"] == pytest.approx(-0.5, abs=1e-6)
    assert deltas["crowded"] == pytest.approx(1.0, abs=1e-6)


def test_bare_head_recall_is_available_as_a_slice_metric() -> None:
    """A different quantity from `primary_map`, so the dispatch has to be tested
    with a case where the two disagree.

    crowded = {image 2}: one `head` GT at (10,10,10,10).
      real_only detects nothing there  -> recall 0/1 = 0.0, primary_map 0.0
      filtered_syn detects it exactly  -> recall 1/1 = 1.0, primary_map 1.0
    small_object = {image 1}: one `head` GT, found exactly by BOTH arms.
      recall 1.0 for both, while primary_map is 1.0 vs 0.5 because the arms
      differ on the `helmet`. That second slice is the discriminating one: a
      dispatch that returned primary_map for this name would read 0.5, not 1.0.
    """

    slices = {"small_object": frozenset({1}), "crowded": frozenset({2})}
    rows = {
        (row.slice_name, row.arm): row
        for row in ea.slice_metric_table(
            ground_truth(),
            detections_by_arm(),
            slices,
            metric="bare_head_recall",
            config=config(),
        )
    }
    assert rows[("small_object", "real_only")].value == pytest.approx(1.0)
    assert rows[("small_object", "filtered_syn")].value == pytest.approx(1.0)
    assert rows[("crowded", "real_only")].value == pytest.approx(0.0)
    assert rows[("crowded", "filtered_syn")].value == pytest.approx(1.0)


def test_the_default_slice_metric_is_primary_map_and_not_the_small_one() -> None:
    """The two are the same number on any fixture whose primary boxes are all
    small — which is true of every other fixture in this file, so the default
    could be either and nothing would notice.

    This one carries a MEDIUM helmet (40x40 = 1,600 px^2, above the 1,024 px^2
    small bucket) that the arm misses by a mile, and a SMALL head (20x20 = 400)
    that it hits exactly:

        primary_map       = mean(AP head 1.0, AP helmet 0.0)  = 0.5
        primary_map_small = mean(AP head 1.0)                 = 1.0
                            (the helmet has no small instance, so the small
                             bucket has nothing to hold against it)
    """

    gt = {
        "images": [{"id": 1, "width": 300, "height": 300}],
        "annotations": [
            annotation(1, 1, HEAD, (0, 0, 20, 20)),
            annotation(2, 1, HELMET, (0, 50, 40, 40)),
        ],
        "categories": [dict(entry) for entry in CATEGORIES],
    }
    arm = [
        detection(1, HEAD, (0, 0, 20, 20), 0.9),
        detection(1, HELMET, (200, 200, 40, 40), 0.9),
    ]
    slices = {"mixed": frozenset({1})}

    default_rows = ea.slice_metric_table(gt, {"arm": arm}, slices, config=config())
    assert default_rows[0].metric == "primary_map"
    assert default_rows[0].value == pytest.approx(0.5, abs=1e-6)

    small_rows = ea.slice_metric_table(
        gt, {"arm": arm}, slices, metric="primary_map_small", config=config()
    )
    assert small_rows[0].value == pytest.approx(1.0, abs=1e-6)


def test_slice_metric_value_rejects_an_unknown_metric_name() -> None:
    slices = {"small_object": frozenset({1})}
    with pytest.raises(ea.ErrorAnalysisConfigError, match="No slice recipe"):
        ea.slice_metric_table(
            ground_truth(), detections_by_arm(), slices, metric="ap_tiny", config=config()
        )


def test_the_five_verdict_names_are_the_published_vocabulary() -> None:
    """These strings ARE the conclusion of the EVAL-17 targeting test. They are
    written verbatim into reports/error_analysis.json and quoted in the markdown,
    so they are a contract with the reader rather than an internal label.

    Asserting `verdict.verdict == ea.TARGETED` everywhere else compares the code
    against itself: give `TARGETED` and `MORE_DATA_NOT_TARGETED` each other's
    value and every one of those assertions still passes while the report says
    the opposite of what happened. Only the literals can tell them apart.
    """

    assert ea.TARGETED == "targeted"
    assert ea.MORE_DATA_NOT_TARGETED == "more_data_not_targeted"
    assert ea.TARGET_SLICE_DID_NOT_IMPROVE == "target_slice_did_not_improve"
    assert ea.TARGET_SLICE_UNMEASURABLE == "target_slice_unmeasurable"
    assert ea.BUDGET_LEADER_HAS_NO_SLICE == "budget_leader_has_no_slice"
    assert (
        len(
            {
                ea.TARGETED,
                ea.MORE_DATA_NOT_TARGETED,
                ea.TARGET_SLICE_DID_NOT_IMPROVE,
                ea.TARGET_SLICE_UNMEASURABLE,
                ea.BUDGET_LEADER_HAS_NO_SLICE,
            }
        )
        == 5
    )


def test_targeting_verdict_equal_improvement_is_more_data_not_targeted() -> None:
    """The boundary that decides. `small_distant` owns the budget and its slice
    improved by EXACTLY as much as another slice. 'No more than the others' is
    the failing side, so the comparison at step 5 must be strict."""

    verdict = ea.targeting_verdict(
        {"small_distant": 0.40, "crowded": 0.30, "hard_negative": 0.30},
        {"small_object": 0.02, "crowded": 0.02, "low_light": 0.01},
    )
    assert verdict.verdict == "more_data_not_targeted"
    assert verdict.verdict == ea.MORE_DATA_NOT_TARGETED
    assert verdict.budget_leader == "small_distant"
    assert verdict.tested_scenario == "small_distant"
    assert verdict.tested_the_budget_leader is True
    assert verdict.target_slice == "small_object"
    assert ea.MORE_DATA_SENTENCE in verdict.sentence
    assert verdict.max_other_delta == pytest.approx(0.02)


def test_targeting_verdict_strictly_larger_improvement_is_targeted() -> None:
    verdict = ea.targeting_verdict(
        {"small_distant": 0.40, "crowded": 0.30},
        {"small_object": 0.03, "crowded": 0.02, "low_light": 0.01},
    )
    assert verdict.verdict == "targeted"
    assert verdict.verdict == ea.TARGETED
    assert ea.MORE_DATA_SENTENCE not in verdict.sentence


def test_targeting_verdict_zero_improvement_is_not_an_improvement() -> None:
    """The other boundary. A delta of exactly 0.0 must fail, or `<=` could be
    weakened to `<` and a slice that did not move would be called targeted."""

    verdict = ea.targeting_verdict(
        {"small_distant": 0.40, "crowded": 0.30},
        {"small_object": 0.0, "crowded": -0.05},
    )
    assert verdict.verdict == ea.TARGET_SLICE_DID_NOT_IMPROVE
    # It also beat the only other slice (-0.05), so a rule that checked "beats
    # the others" first would wrongly call this `targeted`.
    assert verdict.target_delta == pytest.approx(0.0)


def test_targeting_verdict_picks_the_largest_share_not_the_first_key() -> None:
    verdict = ea.targeting_verdict(
        {"crowded": 0.60, "small_distant": 0.30},
        {"small_object": -0.50, "crowded": 1.00},
    )
    assert verdict.budget_leader == "crowded"
    assert verdict.target_slice == "crowded"
    assert verdict.verdict == ea.TARGETED


def test_targeting_falls_back_to_the_largest_scenario_a_slice_can_isolate() -> None:
    """The live case. `head_no_helmet` leads the delivered mix and no slice
    isolates it, while `small_distant` is barely behind and has one. Silencing
    the instrument over a 0.4-point gap would be worse than testing the runner-up
    and saying so."""

    verdict = ea.targeting_verdict(
        {"head_no_helmet": 0.221, "small_distant": 0.217, "crowded": 0.141},
        {"small_object": 0.10, "crowded": 0.01},
    )
    assert verdict.budget_leader == "head_no_helmet"
    assert verdict.tested_scenario == "small_distant"
    assert verdict.tested_the_budget_leader is False
    assert verdict.target_slice == "small_object"
    assert verdict.verdict == ea.TARGETED
    # Both scenarios have to be named, or the reader cannot tell which question
    # was answered.
    assert "head_no_helmet" in verdict.sentence
    assert "small_distant" in verdict.sentence


def test_targeting_verdict_admits_when_no_scenario_has_a_slice_at_all() -> None:
    verdict = ea.targeting_verdict(
        {"head_no_helmet": 0.50, "partial_occlusion": 0.30, "hard_negative": 0.20},
        {"small_object": 0.10, "crowded": 0.01},
    )
    assert verdict.verdict == ea.BUDGET_LEADER_HAS_NO_SLICE
    assert verdict.budget_leader == "head_no_helmet"
    assert verdict.tested_scenario is None
    assert verdict.target_slice is None
    assert verdict.target_delta is None


def test_targeting_verdict_admits_when_the_target_slice_is_unmeasurable() -> None:
    verdict = ea.targeting_verdict(
        {"small_distant": 0.50}, {"crowded": 0.10}
    )
    assert verdict.verdict == ea.TARGET_SLICE_UNMEASURABLE
    assert verdict.target_slice == "small_object"
    assert "untested, not supported" in verdict.sentence


def test_targeting_verdict_requires_a_budget() -> None:
    with pytest.raises(ea.ErrorAnalysisConfigError, match="largest share"):
        ea.targeting_verdict({}, {"small_object": 0.1})


def test_scenario_shares_use_the_delivered_mix_not_the_generated_one() -> None:
    """Three of four records are `small_distant`, so the generated share is 0.75.
    Restricting to the two images an arm actually trained on flips the leader to
    `crowded` at 0.5/0.5 — which is why `keep_names` exists."""

    records = [
        {"file_name": "images/s42_000001.png", "scenario": "small_distant"},
        {"file_name": "images/s42_000002.png", "scenario": "small_distant"},
        {"file_name": "images/s42_000003.png", "scenario": "small_distant"},
        {"file_name": "images/s42_000004.png", "scenario": "crowded"},
    ]
    generated = ea.scenario_shares_from_records(records)
    assert generated == {"crowded": pytest.approx(0.25), "small_distant": pytest.approx(0.75)}

    delivered = ea.scenario_shares_from_records(
        records, keep_names=["s42_000001.png", "images/s42_000004.png"]
    )
    assert delivered == {"crowded": pytest.approx(0.5), "small_distant": pytest.approx(0.5)}


def test_scenario_shares_raise_when_nothing_matches() -> None:
    with pytest.raises(ea.ErrorAnalysisConfigError, match="denominator"):
        ea.scenario_shares_from_records(
            [{"file_name": "a.png", "scenario": "crowded"}], keep_names=["b.png"]
        )


def test_read_records_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        '{"file_name": "a.png", "scenario": "crowded"}\n\n'
        '{"file_name": "b.png", "scenario": "small_distant"}\n',
        encoding="utf-8",
        newline="\n",
    )
    assert len(ea.read_records_jsonl(path)) == 2


# --- Exposure and the negative result (EVAL-18) --------------------------------


def test_exposure_confound_matches_the_equal_step_arithmetic() -> None:
    """Reproduces the shipped run: 3,500 real train images, batch 16, 50 epochs.

    steps_per_epoch(real_only) = 3500 // 16 = 218
    total_steps                = 218 * 50   = 10,900
    real_only exposures  = 10900 * 16 * 1.0 / 3500 = 49.8286 -> 49.83
    filtered exposures   = 10900 * 16 * 0.5 / 3500 = 24.9143 -> 24.91
    ratio                = 24.91 / 49.83            = 0.4999
    """

    compositions = ea.placeholder_compositions(
        {
            "real_only": {"n_real_train": 3500, "n_synthetic": 0},
            "filtered_syn": {"n_real_train": 3500, "n_synthetic": 3500},
        }
    )
    confound = ea.exposure_confound(
        compositions, reference_arm="real_only", reference_epochs=50, batch_size=16
    )
    assert confound.total_steps == 10_900
    assert confound.exposures["real_only"] == pytest.approx(49.83)
    assert confound.exposures["filtered_syn"] == pytest.approx(24.91)
    assert confound.ratio("filtered_syn", "real_only") == pytest.approx(0.4999, abs=1e-4)


def test_an_exposure_ratio_against_a_zero_denominator_refuses_to_be_a_number() -> None:
    """Exactly zero, not negative. `real_image_exposures` is rounded to two
    decimals by `equal_step_budget`, so an arm carrying enough data to be seen
    less than once per hundredth of an epoch legitimately records 0.00 — and a
    guard that only caught NEGATIVE denominators would let that arm divide by
    zero instead of reporting a refusal."""

    confound = ea.ExposureConfound(
        exposures={"real_only": 49.83, "starved_syn": 0.0},
        total_steps=10_900,
        reference_arm="real_only",
        reference_epochs=50,
        batch_size=16,
    )
    with pytest.raises(ea.ErrorAnalysisConfigError, match="not a number"):
        confound.ratio("real_only", "starved_syn")


def test_placeholder_compositions_reject_an_arm_with_no_real_images() -> None:
    with pytest.raises(ea.ErrorAnalysisConfigError, match="divides by"):
        ea.placeholder_compositions({"real_only": {"n_real_train": 0}})


def test_a_tie_on_the_headline_metric_is_not_a_win() -> None:
    """The boundary. `>` must not become `>=`: an arm that merely matches the
    baseline, bought with a whole generation pipeline, is a negative result."""

    verdict = ea.negative_result_verdict(
        {"real_only": 0.3564, "filtered_syn": 0.3564, "unfiltered_syn": 0.2988},
        metric="eval_map",
        baseline_arm="real_only",
        synthetic_arms=("filtered_syn", "unfiltered_syn"),
    )
    assert verdict.winners == ()
    assert verdict.is_negative is True


def test_a_strictly_better_synthetic_arm_is_a_win() -> None:
    verdict = ea.negative_result_verdict(
        {"real_only": 0.3564, "filtered_syn": 0.3565},
        metric="eval_map",
        baseline_arm="real_only",
        synthetic_arms=("filtered_syn",),
    )
    assert verdict.winners == ("filtered_syn",)
    assert verdict.is_negative is False


def test_the_real_validation_numbers_are_a_negative_result() -> None:
    """The four arms as they actually came back from Colab."""

    verdict = ea.negative_result_verdict(
        {
            "real_only": 0.3564,
            "standard_aug": 0.3281,
            "filtered_syn": 0.3200,
            "unfiltered_syn": 0.2988,
        },
        metric="eval_map",
        baseline_arm="real_only",
        synthetic_arms=("filtered_syn", "unfiltered_syn"),
    )
    assert verdict.is_negative is True


def test_negative_result_verdict_refuses_to_drop_an_arm() -> None:
    """Dropping an arm is how a losing arm disappears from a report."""

    with pytest.raises(ea.ErrorAnalysisConfigError, match="unfiltered_syn"):
        ea.negative_result_verdict(
            {"real_only": 0.35, "filtered_syn": 0.32},
            metric="eval_map",
            baseline_arm="real_only",
            synthetic_arms=("filtered_syn", "unfiltered_syn"),
        )
    with pytest.raises(ea.ErrorAnalysisConfigError, match="Baseline arm"):
        ea.negative_result_verdict(
            {"filtered_syn": 0.32},
            metric="eval_map",
            baseline_arm="real_only",
            synthetic_arms=("filtered_syn",),
        )


def test_the_checklist_covers_experiment_protocol_section_7() -> None:
    """Four questions from §7, plus the equal-step exposure confound."""

    keys = [item.key for item in ea.experiment_protocol_checklist()]
    assert keys == [
        "paste_artifact_detectability",
        "scenario_coverage",
        "filter_threshold_dominance",
        "baseline_augmentation_overlap",
        "equal_step_exposure",
    ]
    # Evidence is a POINTER, never a copied measurement: a pasted figure goes
    # stale the moment its source is regenerated (EVAL-12). Spec ids like
    # `FILT-14` are fine; a decimal like `0.9053` is not.
    for item in ea.experiment_protocol_checklist():
        assert re.search(r"\d+\.\d+", item.question) is None, item.key
        assert re.search(r"\d+\.\d+", item.evidence) is None, item.key


# --- Assembly and rendering ----------------------------------------------------


def build(
    tmp_path: Path,
    *,
    detections: dict[str, list[dict[str, Any]]] | None = None,
    **overrides: Any,
) -> ea.ErrorAnalysis:
    kwargs: dict[str, Any] = {
        "config": config(),
        "slices": {
            "small_object": frozenset({1}),
            "crowded": frozenset({2}),
            "low_light": frozenset(),
        },
        "budget_shares": {"small_distant": 0.6, "crowded": 0.4},
        "headline_values": {
            "real_only": 0.3564,
            "standard_aug": 0.3281,
            "filtered_syn": 0.3200,
            "unfiltered_syn": 0.2988,
        },
        "headline_metric": "eval_map",
        "exposure": ea.exposure_confound(
            ea.placeholder_compositions(
                {
                    "real_only": {"n_real_train": 3500, "n_synthetic": 0},
                    "filtered_syn": {"n_real_train": 3500, "n_synthetic": 3500},
                }
            ),
            reference_arm="real_only",
            reference_epochs=50,
            batch_size=16,
        ),
        "image_paths": write_images(tmp_path / "images"),
        "figure_dir": tmp_path / "figures",
    }
    kwargs.update(overrides)
    return ea.build_analysis(
        ground_truth(),
        detections_by_arm() if detections is None else detections,
        **kwargs,
    )


def test_build_analysis_renders_every_configured_category_that_occurred(
    tmp_path: Path,
) -> None:
    analysis = build(tmp_path)
    assert set(analysis.figures) == {
        "fixed_false_negative",
        "fixed_false_positive",
        "new_false_positive",
        "both_wrong",
    }
    for relative in analysis.figures.values():
        assert (tmp_path / "figures" / Path(relative).name).is_file()


def test_build_analysis_refuses_a_hand_built_config_that_drops_the_cost_category(
    tmp_path: Path,
) -> None:
    """`analysis_config=` goes around `load_error_analysis_config`, which is
    where the guard usually sits — and the driver uses exactly that argument.

    Nothing may be rendered on the way to the refusal, so the figure directory
    must not exist afterwards. And the guard is the FIRST thing this function
    does, which the module docstring states as a rule: a call that ALSO names an
    arm with no predictions must still fail on the category list, because the
    shape of the report is refused before its inputs are inventoried. That
    ordering is the whole difference between this check and the one inside
    `select_samples` further down.
    """

    crippled = ea.ErrorAnalysisConfig(
        baseline_arm="real_only",
        best_arm="filtered_syn",
        samples_per_category=2,
        categories=("fixed_false_negative", "both_wrong"),
    )
    with pytest.raises(ea.ErrorAnalysisConfigError, match="new_false_positive"):
        ea.build_analysis(
            ground_truth(),
            detections_by_arm(),
            config=config(),
            analysis_config=crippled,
            image_paths=write_images(tmp_path / "images"),
            figure_dir=tmp_path / "figures",
        )
    assert not (tmp_path / "figures").exists()

    with pytest.raises(ea.ErrorAnalysisConfigError, match="COST"):
        ea.build_analysis(
            ground_truth(),
            {"real_only": baseline_detections()},
            config=config(),
            analysis_config=crippled,
            figure_dir=tmp_path / "figures",
        )


def test_build_analysis_requires_predictions_for_both_compared_arms(
    tmp_path: Path,
) -> None:
    with pytest.raises(ea.ErrorAnalysisConfigError, match="filtered_syn"):
        ea.build_analysis(
            ground_truth(),
            {"real_only": baseline_detections()},
            config=config(),
            figure_dir=tmp_path,
        )


def test_the_report_names_new_false_positive_and_its_count(tmp_path: Path) -> None:
    markdown = ea.render_markdown(build(tmp_path))
    # One new false positive, two fixed ones — the counts table must show both,
    # not only the flattering one. Asserting on the ROW, not on the word: the
    # explanatory prose mentions every category name, so a substring check would
    # still pass with the table row deleted.
    assert "| `new_false_positive` | 1 |" in markdown
    assert "| `fixed_false_positive` | 2 |" in markdown
    assert "| `new_false_negative` | 1 |" in markdown
    assert "| `fixed_false_negative` | 1 |" in markdown
    assert "| `both_wrong` | 3 |" in markdown


def test_percent_is_a_percentage_and_says_n_a_with_no_denominator() -> None:
    """A share printed as `0.1%` when it is `12.5%` is not a rounding error, it
    is a hundredfold understatement of how much of the disagreement a category
    accounts for — and it still looks like a percentage."""

    assert ea._percent(1, 8) == "12.5%"
    assert ea._percent(3, 8) == "37.5%"
    assert ea._percent(1, 0) == "n/a"


def test_value_text_prints_n_a_exactly_at_the_undefined_sentinel() -> None:
    """UNDEFINED is -1.0 and the guard is `<=`. Weakened to `<`, an empty slice
    renders as the number -1.0000, which reads as a measured catastrophe rather
    than as 'this bucket had nothing in it'."""

    assert ea._value_text(UNDEFINED) == "n/a"
    assert ea._value_text(0.0) == "0.0000"


def test_the_counts_table_prints_each_category_as_a_share_of_the_total(
    tmp_path: Path,
) -> None:
    """Eight disagreements in the fixture: `new_false_positive` is 1 of them and
    `both_wrong` is 3, so 12.5% and 37.5%. Nothing else in this file reads a
    percentage, and every one of them is computed by the same helper."""

    markdown = ea.render_markdown(build(tmp_path))
    assert "| `new_false_positive` | 1 | 12.5% |" in markdown
    assert "| `both_wrong` | 3 | 37.5% |" in markdown
    assert "| `fixed_false_positive` | 2 | 25.0% |" in markdown
    assert "| **total** | 8 | 100.0% | |" in markdown


def test_an_undefined_slice_reads_n_a_in_the_table(tmp_path: Path) -> None:
    """`low_light` is empty in the fixture, and the arms are columns in
    alphabetical order."""

    markdown = ea.render_markdown(build(tmp_path))
    assert "| `low_light` | 0 | 0 | n/a | n/a |" in markdown
    assert "-1.0000" not in markdown


def test_a_counted_but_unconfigured_category_says_why_it_has_no_figure(
    tmp_path: Path,
) -> None:
    """`new_false_negative` is counted but is not in `error_analysis.categories`,
    so it never gets a grid. The cell has to say that. "no examples occurred"
    printed beside a count of 1 is a self-contradiction the reader will resolve
    in favour of the sentence, and conclude the regression never happened."""

    markdown = ea.render_markdown(build(tmp_path))
    assert (
        "| `new_false_negative` | 1 | 12.5% | counted only (not a configured category) |"
        in markdown
    )
    assert "no examples occurred" not in markdown


def test_a_configured_category_that_did_not_occur_says_no_examples_occurred(
    tmp_path: Path,
) -> None:
    """The other side of that cell, and the reason it cannot just be deleted.

    Both arms are given the SAME predictions, so they disagree about nothing:
    every `fixed_*` and `new_*` category is genuinely zero and only the six
    errors they share survive. That is what "measured, none" looks like.
    """

    identical = {"real_only": baseline_detections(), "filtered_syn": baseline_detections()}
    markdown = ea.render_markdown(build(tmp_path, detections=identical))
    assert "| `fixed_false_negative` | 0 | 0.0% | no examples occurred |" in markdown
    assert "| `new_false_positive` | 0 | 0.0% | no examples occurred |" in markdown
    assert "| `both_wrong` | 6 | 100.0% |" in markdown
    # And the paragraph under the table stays silent: the only outcome that
    # occurred DID get a grid. Warning about four categories that never happened
    # would train the reader to skip the warning on the run where it matters.
    assert "No comparison grid is rendered for" not in markdown


def test_the_counts_table_reports_the_outcomes_in_the_documented_order(
    tmp_path: Path,
) -> None:
    """The order of `OUTCOMES`, actually reaching the page."""

    markdown = ea.render_markdown(build(tmp_path))
    positions = [
        markdown.index(f"| `{outcome}` |")
        for outcome in (
            "fixed_false_negative",
            "fixed_false_positive",
            "new_false_positive",
            "new_false_negative",
            "both_wrong",
        )
    ]
    assert positions == sorted(positions)


def test_the_hard_negative_table_puts_the_cleanest_arm_first(tmp_path: Path) -> None:
    """Fewest false positives per image at the top. The column is a cost, so the
    reader's eye should land on the best arm and read downwards; reversed, the
    top row is the worst arm and the table reads as if it were a leaderboard."""

    markdown = ea.render_markdown(build(tmp_path))
    assert markdown.index("| `filtered_syn` | 2 | 1.0000 |") < markdown.index(
        "| `real_only` | 3 | 1.5000 |"
    )


def test_the_report_names_the_LARGEST_category_that_got_no_figure(
    tmp_path: Path,
) -> None:
    """With no images supplied, nothing is rendered and all five outcomes are
    unrendered: 1, 2, 1, 1 and 3. The one worth naming is `both_wrong` at 3
    (37.5%). Naming the smallest instead tells the reader the omission is
    negligible at the exact moment it is the largest — which is the delivered
    run's situation, where the unrendered `new_false_negative` held 1,304 of the
    3,232 disagreements."""

    markdown = ea.render_markdown(build(tmp_path, image_paths=None))
    assert "The largest of those, `both_wrong`, is 37.5% of all disagreements" in markdown


def test_render_markdown_refuses_a_report_that_drops_new_false_positive(
    tmp_path: Path,
) -> None:
    """Even a hand-built ErrorAnalysis cannot be rendered without it: the rule is
    about what gets published, not about which function built the object."""

    analysis = build(tmp_path)
    crippled = ea.ErrorAnalysis(
        config=ea.ErrorAnalysisConfig(
            baseline_arm="real_only",
            best_arm="filtered_syn",
            samples_per_category=2,
            categories=("fixed_false_negative", "both_wrong"),
        ),
        iou_threshold=analysis.iou_threshold,
        score_threshold=analysis.score_threshold,
        items=analysis.items,
        counts=analysis.counts,
        counts_by_class=analysis.counts_by_class,
        samples=analysis.samples,
        figures=analysis.figures,
        hard_negatives=analysis.hard_negatives,
        slice_metrics=analysis.slice_metrics,
        slice_deltas=analysis.slice_deltas,
        targeting=analysis.targeting,
        negative_result=analysis.negative_result,
        exposure=analysis.exposure,
        budget_shares=analysis.budget_shares,
    )
    with pytest.raises(ea.ErrorAnalysisConfigError, match="new_false_positive"):
        ea.render_markdown(crippled)


def test_the_report_explains_an_absent_hard_negative_subset(tmp_path: Path) -> None:
    analysis = build(tmp_path)
    trimmed = ea.ErrorAnalysis(
        **{
            **{f: getattr(analysis, f) for f in analysis.__dataclass_fields__},
            "hard_negatives": (),
        }
    )
    markdown = ea.render_markdown(trimmed)
    assert "no natural hard negatives" in markdown
    assert "for analysis only, never for training" in markdown


def test_the_report_states_the_targeting_verdict_in_words(tmp_path: Path) -> None:
    """`small_distant` owns 60% of the budget; its slice moved -0.5 while
    `crowded` moved +1.0. The slice the budget was aimed at got worse."""

    analysis = build(tmp_path)
    assert analysis.targeting is not None
    assert analysis.targeting.verdict == ea.TARGET_SLICE_DID_NOT_IMPROVE
    markdown = ea.render_markdown(analysis)
    assert "Verdict: `target_slice_did_not_improve`" in markdown
    assert "did not improve at all" in markdown


def test_the_report_says_more_data_not_targeted_when_that_is_the_answer(
    tmp_path: Path,
) -> None:
    """Same machinery, a budget leader whose slice improved exactly as much as
    another one. The required words must reach the markdown, not just the
    dataclass."""

    analysis = build(tmp_path)
    tied = ea.targeting_verdict(
        {"small_distant": 0.6, "crowded": 0.4},
        {"small_object": 0.02, "crowded": 0.02},
    )
    markdown = ea.render_markdown(
        ea.ErrorAnalysis(
            **{
                **{
                    field: getattr(analysis, field)
                    for field in analysis.__dataclass_fields__
                },
                "targeting": tied,
            }
        )
    )
    assert ea.MORE_DATA_SENTENCE in markdown
    # Spelled out as a literal too. Comparing only against the constant would
    # let someone reword both at once and still pass; EVAL-17 asks for these
    # exact words.
    assert "more data, not targeted data" in markdown
    # And the verdict word itself, which is what the JSON consumer keys on.
    assert "Verdict: `more_data_not_targeted`" in markdown


def test_the_report_states_the_exposure_confound_in_numbers(tmp_path: Path) -> None:
    markdown = ea.render_markdown(build(tmp_path))
    assert "10,900" in markdown  # total optimizer steps
    assert "49.83" in markdown  # real_only real-image exposures
    assert "24.91" in markdown  # filtered_syn real-image exposures
    assert "0.50x" in markdown  # the ratio, spelled out
    assert "confound, not a footnote" in markdown


def test_the_report_walks_the_checklist_on_the_negative_path(tmp_path: Path) -> None:
    analysis = build(tmp_path)
    assert analysis.negative_result is not None
    assert analysis.negative_result.is_negative is True
    markdown = ea.render_markdown(analysis)
    assert "No synthetic arm beats" in markdown
    for item in ea.experiment_protocol_checklist():
        assert item.question in markdown


def test_the_report_says_so_when_an_input_was_not_supplied(tmp_path: Path) -> None:
    """A section that vanishes reads as a section that did not apply."""

    analysis = build(tmp_path, headline_values=None, exposure=None, slices=None)
    markdown = ea.render_markdown(analysis)
    assert "No headline comparison was supplied" in markdown
    assert "No slice metrics were computed" in markdown


def test_the_json_spine_reproduces_every_count_in_the_markdown(tmp_path: Path) -> None:
    """EVAL-12: the tables have to be re-aggregatable from a machine-readable file."""

    analysis = build(tmp_path)
    report_path, payload_path = ea.write_report(
        analysis,
        report_path=tmp_path / "error_analysis.md",
        payload_path=tmp_path / "error_analysis.json",
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["counts"] == {
        "fixed_false_negative": 1,
        "fixed_false_positive": 2,
        "new_false_positive": 1,
        "new_false_negative": 1,
        "both_wrong": 3,
    }
    assert payload["config"]["baseline_arm"] == "real_only"
    # Only the two synthetic arms are on trial. `standard_aug` is a baseline
    # variant, and counting it as synthetic would let a synthetic "win" be
    # claimed on an arm that carries no synthetic images at all.
    assert payload["negative_result"]["synthetic_arms"] == [
        "filtered_syn",
        "unfiltered_syn",
    ]
    assert payload["targeting"]["verdict"] == ea.TARGET_SLICE_DID_NOT_IMPROVE
    assert payload["exposure"]["exposures"]["filtered_syn"] == pytest.approx(24.91)
    hard = {row["arm"]: row for row in payload["hard_negatives"]}
    assert hard["real_only"]["false_positives_per_image"] == pytest.approx(1.5)

    # Every count in the JSON must actually appear in the markdown table.
    markdown = report_path.read_text(encoding="utf-8")
    for outcome, count in payload["counts"].items():
        assert f"| `{outcome}` | {count:,} |" in markdown


def test_figure_links_resolve_from_the_report_not_from_the_repo_root() -> None:
    """Markdown resolves a relative link against the directory of the file it is
    written in. A repo-relative `reports/figures/...` inside `reports/x.md`
    points at `reports/reports/figures/...` and renders as a broken image.

    The stored value stays repo-relative for the JSON payload; only the LINK is
    rewritten, so the two are deliberately different strings here.
    """

    stored = "reports/figures/error_analysis/new_false_positive.png"
    link = ea._markdown_link(stored, ea.PROJECT_ROOT / "reports" / "error_analysis.md")
    assert link == "figures/error_analysis/new_false_positive.png"
    assert (ea.PROJECT_ROOT / "reports" / link).resolve() == (
        ea.PROJECT_ROOT / stored
    ).resolve()

    # A report one level deeper has to reach back up.
    deeper = ea._markdown_link(stored, ea.PROJECT_ROOT / "reports" / "sub" / "x.md")
    assert deeper == "../figures/error_analysis/new_false_positive.png"


def test_figure_paths_are_stored_repo_relative_when_they_are_in_the_repo(
    tmp_path: Path,
) -> None:
    """The JSON payload has to carry a path a reader can act on. Every other test
    renders into `tmp_path`, which is outside the repo and therefore only ever
    exercises the absolute-path fallback — so the branch that runs in production
    is the one branch nothing covered."""

    inside = ea.PROJECT_ROOT / "reports" / "figures" / "error_analysis" / "both_wrong.png"
    assert ea._repo_relative(inside) == "reports/figures/error_analysis/both_wrong.png"

    # A path outside the repo has no repo-relative form and stays absolute.
    outside = ea._repo_relative(tmp_path / "grid.png")
    assert Path(outside).is_absolute()
    assert outside == (tmp_path / "grid.png").resolve().as_posix()


def test_the_report_flags_a_category_that_gets_no_figure(tmp_path: Path) -> None:
    """`new_false_negative` is not in the shipped category list, so it never gets
    a grid. On the real split it is also the LARGEST category, which is exactly
    the situation where a silent omission misleads."""

    markdown = ea.render_markdown(build(tmp_path))
    assert "No comparison grid is rendered for" in markdown
    assert "`new_false_negative` (1," in markdown


def test_written_files_are_utf8_with_lf_endings(tmp_path: Path) -> None:
    """K-10 plus the repo-wide newline rule; CRLF would make every regeneration
    a whole-file diff."""

    analysis = build(tmp_path)
    report_path, payload_path = ea.write_report(
        analysis,
        report_path=tmp_path / "error_analysis.md",
        payload_path=tmp_path / "error_analysis.json",
    )
    for path in (report_path, payload_path):
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        raw.decode("utf-8")


# --- Prediction file loading ---------------------------------------------------


def test_load_predictions_reads_a_detection_list(tmp_path: Path) -> None:
    path = tmp_path / "real_only_test.json"
    path.write_text(json.dumps(baseline_detections()), encoding="utf-8", newline="\n")
    loaded = ea.load_predictions(path)
    assert len(loaded) == 5
    assert loaded[0]["score"] == pytest.approx(0.90)


def test_default_predictions_path_is_the_documented_layout() -> None:
    path = ea.default_predictions_path("filtered_syn")
    assert path.name == "filtered_syn_test.json"
    assert path.parent.name == "predictions"


def test_load_predictions_rejects_a_coco_dict(tmp_path: Path) -> None:
    """A COCO results dict iterates as its keys, which would then be classified
    as detections named 'images' and 'annotations'."""

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"annotations": []}), encoding="utf-8", newline="\n")
    with pytest.raises(ea.PredictionFileError, match="not a list"):
        ea.load_predictions(path)


def test_load_predictions_rejects_missing_keys_and_bad_boxes(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    missing.write_text(
        json.dumps([{"image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1]}]),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ea.PredictionFileError, match="score"):
        ea.load_predictions(missing)

    bad_box = tmp_path / "bad_box.json"
    bad_box.write_text(
        json.dumps([{"image_id": 1, "category_id": 0, "bbox": [0, 0, 1], "score": 0.5}]),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ea.PredictionFileError, match="x, y, w, h"):
        ea.load_predictions(bad_box)

    # Both directions. A five-element box is an xyxy-plus-score box, or a
    # polygon, or a box from another convention entirely; taking its first four
    # numbers as [x, y, w, h] reads a real number out of the wrong slot, which
    # is worse than the too-short box because it does not crash later either.
    long_box = tmp_path / "long_box.json"
    long_box.write_text(
        json.dumps(
            [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 1, 1, 1], "score": 0.5}]
        ),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ea.PredictionFileError, match="x, y, w, h"):
        ea.load_predictions(long_box)


def test_load_predictions_names_the_default_layout_when_the_file_is_absent(
    tmp_path: Path,
) -> None:
    with pytest.raises(ea.PredictionFileError, match="results/predictions"):
        ea.load_predictions(tmp_path / "nope.json")


def test_load_predictions_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8", newline="\n")
    with pytest.raises(ea.PredictionFileError, match="not valid JSON"):
        ea.load_predictions(path)


# --- The driver: scripts/error_analysis.py -------------------------------------
#
# The driver is where the report's inputs are CHOSEN: which COCO, which split,
# which checkpoint's number counts as an arm's headline, and whether the exposure
# arithmetic was checked against the run that actually happened. Every one of
# those decisions produces a plausible report when it is wrong.

TRAINING_YAML = """budget_alignment: equal_steps
run:
  num_train_epochs_real_only: 50
  per_device_train_batch_size: 16
"""


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    return path


def write_summary(path: Path, *, plan_block: dict[str, Any] | None) -> Path:
    """A run summary shaped like results/colab/training_summary.json.

    3,500 real train images, batch 16, 50 epochs for the reference arm, so
    `equal_step_budget` gives 3500 // 16 = 218 steps per epoch, 10,900 total
    steps, and 49.83 / 24.91 real-image exposures — the same arithmetic derived
    by hand in `test_exposure_confound_matches_the_equal_step_arithmetic`.
    """

    payload: dict[str, Any] = {
        "arms": {
            "real_only": {"n_real_train": 3500, "n_synthetic": 0},
            "filtered_syn": {"n_real_train": 3500, "n_synthetic": 3500},
        },
        "records": [
            {"arm": "real_only", "eval_metrics": {"eval_map": 0.3105}},
            {"arm": "real_only", "eval_metrics": {"eval_map": 0.3564}},
            {"arm": "filtered_syn", "eval_metrics": {"eval_map": 0.3200}},
        ],
    }
    if plan_block is not None:
        payload["plan"] = plan_block
    return write_json(path, payload)


def driver_inputs(tmp_path: Path) -> dict[str, Path]:
    """Every file the driver reads, in a throwaway tree.

    The COCO carries the same five non-crowd boxes as `ground_truth()`; the
    driver rebuilds its ground truth through `build_coco_ground_truth`, which
    does not keep the crowd flag, so the crowd box is simply absent here.
    """

    images_root = tmp_path / "images"
    write_images(images_root)
    names = {image_id: f"image_{image_id}.png" for image_id in (1, 2, 3, 4)}

    annotations = write_json(
        tmp_path / "coco_all.json",
        {
            "images": [
                {"id": image_id, "file_name": name, "width": 100, "height": 100}
                for image_id, name in names.items()
            ],
            "annotations": [
                annotation(1, 1, HEAD, (10, 10, 20, 20)),
                annotation(2, 1, HELMET, (50, 50, 30, 30)),
                annotation(3, 2, HEAD, (10, 10, 10, 10)),
                annotation(4, 2, HELMET, (70, 70, 20, 20)),
                annotation(5, 4, PERSON, (5, 5, 30, 60)),
            ],
            "categories": [dict(entry) for entry in CATEGORIES],
        },
    )
    manifest = write_json(
        tmp_path / "split_manifest.json",
        {"images": [{"file_name": name, "split": "test"} for name in names.values()]},
    )

    predictions = tmp_path / "predictions"
    write_json(predictions / "real_only_test.json", baseline_detections())
    write_json(predictions / "filtered_syn_test.json", best_detections())
    write_json(predictions / "standard_aug_test.json", baseline_detections())
    # `unfiltered_syn` is deliberately absent: a missing arm must be reported,
    # not faked, and it must not stop the two compared arms being analysed.

    records = tmp_path / "records.jsonl"
    records.write_text(
        '{"file_name": "s42_000001.png", "scenario": "small_distant"}\n'
        '{"file_name": "s42_000002.png", "scenario": "small_distant"}\n'
        '{"file_name": "s42_000003.png", "scenario": "crowded"}\n',
        encoding="utf-8",
        newline="\n",
    )

    training_config = tmp_path / "training.yaml"
    training_config.write_text(TRAINING_YAML, encoding="utf-8", newline="\n")

    return {
        "annotations": annotations,
        "images_root": images_root,
        "manifest": manifest,
        "predictions": predictions,
        "records": records,
        "training_config": training_config,
        "summary": write_summary(
            tmp_path / "summary.json",
            plan_block={
                "real_only": {"real_image_exposures": 49.83},
                "filtered_syn": {"real_image_exposures": 24.91},
            },
        ),
        "headline": write_json(
            tmp_path / "headline.json",
            {
                "real_only": 0.3564,
                "standard_aug": 0.3281,
                "filtered_syn": 0.3200,
                "unfiltered_syn": 0.2988,
            },
        ),
        "report": tmp_path / "out" / "error_analysis.md",
        "payload": tmp_path / "out" / "error_analysis.json",
        "figures": tmp_path / "out" / "figures",
        "runs": tmp_path / "runs",
    }


def driver_argv(paths: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--test-annotations", str(paths["annotations"]),
        "--images-root", str(paths["images_root"]),
        "--split-manifest", str(paths["manifest"]),
        "--predictions-dir", str(paths["predictions"]),
        "--records", str(paths["records"]),
        "--summary", str(paths["summary"]),
        "--training-config", str(paths["training_config"]),
        "--runs-root", str(paths["runs"]),
        "--headline-json", str(paths["headline"]),
        "--figure-dir", str(paths["figures"]),
        "--report", str(paths["report"]),
        "--payload", str(paths["payload"]),
        *extra,
    ]


def test_load_yaml_refuses_anything_that_is_not_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8", newline="\n")
    with pytest.raises(DriverError, match="Expected a mapping"):
        load_yaml(path)


def test_budget_inputs_refuses_a_schedule_that_is_not_equal_steps(
    tmp_path: Path,
) -> None:
    """The EVAL-18 exposure confound is a CONSEQUENCE of holding optimizer steps
    equal. Under an equal-EPOCH schedule every arm sees each real image the same
    number of times and the table would describe a run that did not happen."""

    equal_steps = tmp_path / "steps.yaml"
    equal_steps.write_text(TRAINING_YAML, encoding="utf-8", newline="\n")
    assert budget_inputs(equal_steps) == ("real_only", 50, 16)

    equal_epochs = tmp_path / "epochs.yaml"
    equal_epochs.write_text(
        TRAINING_YAML.replace("equal_steps", "equal_epochs"),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(DriverError, match="equal_steps"):
        budget_inputs(equal_epochs)


def test_read_exposure_recomputes_and_agrees_with_the_recorded_plan(
    tmp_path: Path,
) -> None:
    """Hand-derived: 3500 // 16 = 218 steps per epoch, x 50 = 10,900 steps;
    10900 * 16 * 1.0 / 3500 = 49.83 and 10900 * 16 * 0.5 / 3500 = 24.91."""

    training_config = tmp_path / "training.yaml"
    training_config.write_text(TRAINING_YAML, encoding="utf-8", newline="\n")
    summary = write_summary(
        tmp_path / "summary.json",
        plan_block={
            "real_only": {"real_image_exposures": 49.83},
            "filtered_syn": {"real_image_exposures": 24.91},
        },
    )
    confound, unchecked = read_exposure(summary, training_config)
    assert unchecked is None
    assert confound.total_steps == 10_900
    assert confound.exposures == {
        "real_only": pytest.approx(49.83),
        "filtered_syn": pytest.approx(24.91),
    }


def test_read_exposure_refuses_to_publish_when_the_run_recorded_something_else(
    tmp_path: Path,
) -> None:
    training_config = tmp_path / "training.yaml"
    training_config.write_text(TRAINING_YAML, encoding="utf-8", newline="\n")
    summary = write_summary(
        tmp_path / "summary.json",
        plan_block={
            "real_only": {"real_image_exposures": 49.83},
            # The run says the synthetic arm saw the real images as often as the
            # baseline did. Under an equal-STEP budget carrying twice the data
            # that cannot be true, so one of the two files is describing another
            # run and neither may be published.
            "filtered_syn": {"real_image_exposures": 49.83},
        },
    )
    with pytest.raises(DriverError, match="disagree"):
        read_exposure(summary, training_config)


def test_read_exposure_says_so_when_there_was_nothing_to_check_against(
    tmp_path: Path,
) -> None:
    """A cross-check that silently did nothing is worse than no cross-check: the
    reader of the driver's output believes the recomputed numbers were confirmed
    against the run. With no `plan` block they were not, and with a partial plan
    only some of them were."""

    training_config = tmp_path / "training.yaml"
    training_config.write_text(TRAINING_YAML, encoding="utf-8", newline="\n")

    no_plan = write_summary(tmp_path / "no_plan.json", plan_block=None)
    confound, unchecked = read_exposure(no_plan, training_config)
    assert confound.exposures["filtered_syn"] == pytest.approx(24.91)
    assert unchecked is not None and "no `plan` block" in unchecked

    partial = write_summary(
        tmp_path / "partial.json",
        plan_block={"real_only": {"real_image_exposures": 49.83}},
    )
    _, unchecked = read_exposure(partial, training_config)
    assert unchecked is not None and "filtered_syn" in unchecked


def test_read_best_checkpoint_values_takes_each_arms_PEAK_not_its_last_eval(
    tmp_path: Path,
) -> None:
    """Every arm in this project peaks early and decays for the rest of its
    schedule, and `load_best_model_at_end` selects the peak — so the peak is the
    number the arms were actually compared at. The curve here rises to 0.42 in
    the middle and falls to 0.20, and the peak is interior on purpose: at either
    end, "max", "first" and "last" would all agree.

    Two checkpoints exist, and only the higher-numbered one carries the whole
    history; reading the older one would report 0.10.
    """

    seed_dir = tmp_path / "real_only" / "seed_1337"
    write_json(
        seed_dir / "checkpoint-100" / "trainer_state.json",
        {"log_history": [{"step": 100, "eval_map": 0.10}]},
    )
    write_json(
        seed_dir / "checkpoint-300" / "trainer_state.json",
        {
            "log_history": [
                {"step": 100, "eval_map": 0.10},
                {"step": 200, "eval_map": 0.42},
                {"step": 300, "eval_map": 0.20},
                {"step": 300, "loss": 0.5},
            ]
        },
    )
    # An arm whose checkpoint carries no state file is skipped, not guessed at.
    (tmp_path / "standard_aug" / "seed_1337" / "checkpoint-100").mkdir(parents=True)

    values = read_best_checkpoint_values(tmp_path, 1337, "eval_map")
    assert values == {"real_only": pytest.approx(0.42)}


def test_read_headline_values_falls_back_to_the_summary_records(
    tmp_path: Path,
) -> None:
    """The documented fallback, and it must say the same thing twice for one arm
    only once: two evals for `real_only`, and the better of them wins."""

    summary = write_summary(tmp_path / "summary.json", plan_block=None)
    assert read_headline_values(summary, "eval_map") == {
        "real_only": pytest.approx(0.3564),
        "filtered_syn": pytest.approx(0.3200),
    }
    with pytest.raises(DriverError, match="eval_map_small"):
        read_headline_values(summary, "eval_map_small")


def test_read_arm_predictions_reports_a_missing_arm_instead_of_faking_it(
    tmp_path: Path,
) -> None:
    paths = driver_inputs(tmp_path)
    loaded, missing = read_arm_predictions(
        paths["predictions"], ["real_only", "filtered_syn", "unfiltered_syn"]
    )
    assert sorted(loaded) == ["filtered_syn", "real_only"]
    assert len(loaded["real_only"]) == 5
    assert len(missing) == 1
    assert missing[0].startswith("unfiltered_syn: no ")


def test_load_test_samples_refuses_a_split_the_coco_cannot_supply(
    tmp_path: Path,
) -> None:
    """A manifest naming images the COCO does not carry is a broken split, and
    analysing whatever survived the intersection would silently report on a
    different Test set from the frozen one."""

    paths = driver_inputs(tmp_path)
    samples = load_test_samples(
        paths["annotations"], paths["images_root"], paths["manifest"]
    )
    assert len(samples) == 4
    assert {int(sample.image_id) for sample in samples} == {1, 2, 3, 4}

    short = write_json(
        tmp_path / "extra_manifest.json",
        {
            "images": [
                {"file_name": "image_1.png", "split": "test"},
                {"file_name": "image_404.png", "split": "test"},
            ]
        },
    )
    with pytest.raises(DriverError, match="only 1 are in"):
        load_test_samples(paths["annotations"], paths["images_root"], short)

    no_test = write_json(
        tmp_path / "train_only.json",
        {"images": [{"file_name": "image_1.png", "split": "train"}]},
    )
    with pytest.raises(DriverError, match="no `test` split"):
        load_test_samples(paths["annotations"], paths["images_root"], no_test)


def test_load_test_samples_says_which_argument_is_wrong_when_the_pngs_are_absent(
    tmp_path: Path,
) -> None:
    """`load_coco_samples` strips the directory off `file_name`, so pointing
    `--images-root` at the PARENT of the image folder fails several hundred
    images into the luminance pass rather than immediately."""

    paths = driver_inputs(tmp_path)
    with pytest.raises(DriverError, match="--images-root"):
        load_test_samples(paths["annotations"], tmp_path, paths["manifest"])


def test_parse_args_rejects_a_slice_metric_the_module_cannot_compute() -> None:
    assert parse_args([]).slice_metric == ea.DEFAULT_SLICE_METRIC
    with pytest.raises(SystemExit):
        parse_args(["--slice-metric", "ap_tiny"])


def test_the_driver_writes_the_report_the_json_and_the_grids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end on a four-image split, with every input pointed at tmp_path.

    Hand-derived at the SHIPPED operating point (`compliance.score_threshold`,
    0.07 — every fixture detection clears it, including the 0.30 helmet that the
    0.50 fixture threshold hides) and the shipped IoU of 0.50:

      gt 1 helmet   baseline only                    -> new_false_negative   1
      gt 2 head     best only                        -> fixed_false_negative 1
      gt 3 helmet   best only, at 0.30               -> fixed_false_negative 2
      gt 4 person   neither arm predicts that class  -> both_wrong           1
      img3 head     both arms, IoU 1.0               -> both_wrong           2
      img3+img4 helmet, baseline only                -> fixed_false_positive 2
      img3 helmet(80,10), best only                  -> new_false_positive   1

    `unfiltered_syn` has no prediction file, which is a reported gap and a
    non-zero exit code, not a reason to skip the two arms being compared.
    """

    paths = driver_inputs(tmp_path)
    assert run_driver(driver_argv(paths)) == 1
    printed = capsys.readouterr().out
    assert "MISSING: unfiltered_syn" in printed

    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    assert payload["counts"] == {
        "fixed_false_negative": 2,
        "fixed_false_positive": 2,
        "new_false_positive": 1,
        "new_false_negative": 1,
        "both_wrong": 2,
    }
    assert payload["score_threshold"] == pytest.approx(ea.default_score_threshold())
    assert payload["iou_threshold"] == pytest.approx(ea.default_iou_threshold())

    # EVAL-17: the delivered mix comes from the records file, two of whose three
    # entries are `small_distant`, and that scenario's slice is `small_object`.
    assert payload["budget_shares"] == {
        "crowded": pytest.approx(1 / 3),
        "small_distant": pytest.approx(2 / 3),
    }
    assert payload["targeting"]["budget_leader"] == "small_distant"
    assert payload["targeting"]["target_slice"] == "small_object"

    # EVAL-18: the headline file decides who won, and only the two `_syn` arms
    # are on trial.
    assert payload["negative_result"]["synthetic_arms"] == [
        "filtered_syn",
        "unfiltered_syn",
    ]
    assert payload["negative_result"]["is_negative"] is True
    assert payload["exposure"]["exposures"]["filtered_syn"] == pytest.approx(24.91)

    markdown = paths["report"].read_text(encoding="utf-8")
    assert "| `new_false_positive` | 1 |" in markdown
    assert "10,900" in markdown
    for category in ea.load_error_analysis_config().categories:
        assert (paths["figures"] / f"{category}.png").is_file()


def test_the_driver_still_writes_the_counts_when_the_figures_are_skipped(
    tmp_path: Path,
) -> None:
    """`--no-figures` is for a machine with no Test images on it. The report is
    smaller, not quieter: the counts table and the JSON spine are unaffected."""

    paths = driver_inputs(tmp_path)
    assert run_driver(driver_argv(paths, "--no-figures", "--no-slices")) == 1
    payload = json.loads(paths["payload"].read_text(encoding="utf-8"))
    assert payload["figures"] == {}
    assert payload["counts"]["new_false_positive"] == 1
    assert payload["slice_metrics"] == []
    assert not paths["figures"].exists()
    assert "No slice metrics were computed" in paths["report"].read_text(encoding="utf-8")


def test_the_driver_prints_the_unchecked_warning_when_the_run_recorded_no_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = driver_inputs(tmp_path)
    write_summary(paths["summary"], plan_block=None)
    assert run_driver(driver_argv(paths, "--no-figures", "--no-slices")) == 1
    assert "UNCHECKED:" in capsys.readouterr().out


def test_the_driver_refuses_to_run_without_the_two_compared_arms(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2, not a report with one arm in it."""

    paths = driver_inputs(tmp_path)
    (paths["predictions"] / "filtered_syn_test.json").unlink()
    assert run_driver(driver_argv(paths, "--no-figures", "--no-slices")) == 2
    assert "FATAL: error_analysis.compare_arms needs 'filtered_syn'" in capsys.readouterr().out
    assert not paths["report"].exists()


def test_the_driver_refuses_to_run_without_the_frozen_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = driver_inputs(tmp_path)
    paths["manifest"].unlink()
    assert run_driver(driver_argv(paths, "--no-figures", "--no-slices")) == 2
    assert "FATAL: missing" in capsys.readouterr().out


# --- Residual gaps the adversarial pass found after the first hardening -------
#
# All three were TEST gaps, not defects: the source was already correct when
# they were found. They are closed because each one changes a NUMBER that
# reaches a report or a figure caption, and none of them would have been caught.


def test_the_category_total_reaches_the_grid_from_render_all_grids(
    tmp_path: Path, monkeypatch
) -> None:
    """A18: the WIRE, not the leaf.

    plan_comparison_grid was already tested to honour total_in_category. What
    nothing pinned was that render_all_grids passes `counts` down to it, and the
    production path is build_analysis -> render_all_grids -> the grid. With the
    argument dropped, a subtitle reads "12 of 12 shown" for a category where 12
    of 931 were shown, understating the cost of synthetic data by two orders of
    magnitude in the one figure that exists to show it.

    The collaborator is intercepted rather than re-invoked, because asserting on
    a grid this test built itself would test the leaf again and say nothing
    about the wire - which is how this gap survived the first hardening pass.
    """

    image_paths = write_images(tmp_path / "images")
    samples = ea.select_samples(
        compare(), categories=[ea.NEW_FALSE_POSITIVE], limit=1
    )

    seen: dict[str, int | None] = {}
    real_plan = ea.plan_comparison_grid

    def recording_plan(category, items, **kwargs):
        seen[category] = kwargs.get("total_in_category")
        return real_plan(category, items, **kwargs)

    monkeypatch.setattr(ea, "plan_comparison_grid", recording_plan)

    ea.render_all_grids(
        samples,
        image_paths=image_paths,
        output_dir=tmp_path / "figures",
        baseline_arm="real_only",
        best_arm="filtered_syn",
        counts={ea.NEW_FALSE_POSITIVE: 931},
    )

    assert seen[ea.NEW_FALSE_POSITIVE] == 931


def test_render_all_grids_passes_no_total_when_it_was_given_no_counts(
    tmp_path: Path, monkeypatch
) -> None:
    """The other direction, so the test above cannot pass by hard-coding 931."""

    image_paths = write_images(tmp_path / "images")
    samples = ea.select_samples(
        compare(), categories=[ea.NEW_FALSE_POSITIVE], limit=1
    )

    seen: dict[str, int | None] = {}
    real_plan = ea.plan_comparison_grid

    def recording_plan(category, items, **kwargs):
        seen[category] = kwargs.get("total_in_category")
        return real_plan(category, items, **kwargs)

    monkeypatch.setattr(ea, "plan_comparison_grid", recording_plan)

    ea.render_all_grids(
        samples,
        image_paths=image_paths,
        output_dir=tmp_path / "figures",
        baseline_arm="real_only",
        best_arm="filtered_syn",
    )

    assert seen[ea.NEW_FALSE_POSITIVE] is None


def test_the_crop_side_is_clamped_DOWN_to_the_short_edge(tmp_path: Path) -> None:
    """A29: on a non-square frame the clamp has to shrink, never grow.

    The Test split is 416x415 / 415x416 (DATA-25). A wide box on a short frame
    makes the requested square exceed the short edge; taking the max instead of
    the min leaves `span` bigger than the image, both offsets clamp NEGATIVE,
    and the numpy slice silently returns a short, off-target crop. The existing
    fixture box was small enough that min and max agreed.
    """

    # 200x60 frame, a 150-wide box: the requested side is well over the 60 px
    # short edge, so the window must come back at exactly 60 and stay in bounds.
    left, top, span = ea._crop_window((20.0, 10.0, 150.0, 20.0), (200, 60))

    assert span == 60
    assert left >= 0 and top >= 0
    assert left + span <= 200
    assert top + span <= 60


def test_a_detection_is_matched_highest_score_first(tmp_path: Path) -> None:
    """A11: two boxes on one ground truth, and the CONFIDENT one must win.

    match_arm's docstring says highest-score-first. Reversing the sort left the
    suite green because the only fixture used two identical boxes and asserted
    only the counts. When the weak detection takes the hit, the confident one
    becomes the false positive, so both the category and the score printed on
    the panel describe the wrong box.
    """

    gt = [ea.GroundTruthRecord(0, 1, "helmet", (10.0, 10.0, 40.0, 40.0))]
    weak = ea.DetectionRecord(0, 1, "helmet", (10.0, 10.0, 40.0, 40.0), 0.08)
    strong = ea.DetectionRecord(1, 1, "helmet", (11.0, 11.0, 40.0, 40.0), 0.24)

    # Offered weak-first, so a stable sort that does not reverse would pick it.
    matched_gt, matched_det = ea.match_arm([gt[0]], [weak, strong], iou_threshold=0.5)

    assert matched_gt[0] == strong.index
    assert strong.index in matched_det
    assert weak.index not in matched_det
