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
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from src.evaluation import error_analysis as ea
from src.evaluation.detection import UNDEFINED, UnknownCategoryError

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

    rows = {
        row.arm: row
        for row in ea.hard_negative_rows(
            ground_truth(), detections_by_arm(), config=config()
        )
    }
    assert rows["real_only"].n_false_positives == 3
    assert rows["real_only"].n_images == 2
    assert rows["real_only"].false_positives_per_image == pytest.approx(1.5)
    assert rows["filtered_syn"].n_false_positives == 2
    assert rows["filtered_syn"].false_positives_per_image == pytest.approx(1.0)


# --- Slices and targeting (EVAL-17) --------------------------------------------


def test_subset_ground_truth_keeps_every_category() -> None:
    """Dropping an absent class would renumber the categories and silently change
    what AP averages over."""

    subset = ea.subset_ground_truth(ground_truth(), [1])
    assert [entry["name"] for entry in subset["categories"]] == [
        "helmet",
        "head",
        "person",
    ]
    assert {int(image["id"]) for image in subset["images"]} == {1}
    assert {int(a["image_id"]) for a in subset["annotations"]} == {1}


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


def test_slice_metric_value_rejects_an_unknown_metric_name() -> None:
    slices = {"small_object": frozenset({1})}
    with pytest.raises(ea.ErrorAnalysisConfigError, match="No slice recipe"):
        ea.slice_metric_table(
            ground_truth(), detections_by_arm(), slices, metric="ap_tiny", config=config()
        )


def test_targeting_verdict_equal_improvement_is_more_data_not_targeted() -> None:
    """The boundary that decides. `small_distant` owns the budget and its slice
    improved by EXACTLY as much as another slice. 'No more than the others' is
    the failing side, so the comparison at step 5 must be strict."""

    verdict = ea.targeting_verdict(
        {"small_distant": 0.40, "crowded": 0.30, "hard_negative": 0.30},
        {"small_object": 0.02, "crowded": 0.02, "low_light": 0.01},
    )
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


def build(tmp_path: Path, **overrides: Any) -> ea.ErrorAnalysis:
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
    return ea.build_analysis(ground_truth(), detections_by_arm(), **kwargs)


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
