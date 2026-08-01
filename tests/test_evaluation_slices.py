"""Tests for EVAL-08 / EVAL-17 Test-set profiling.

K-18 is the rule that shaped this file: a branch no test executes has zero
coverage however green the suite looks. Every branch of src/evaluation/slices.py
and of scripts/profile_test_set.py is exercised here, including the failure
paths, the empty-bucket verdict and the markdown renderer.

Its sequel is the rule that shaped the EVAL-07 section: a branch a test executes
but cannot fail on has zero coverage too. Guarding `size_bucket` alone let a
`box *= 640/416` inside `bucket_instance_counts` and `small_object_images` --
precisely the EVAL-07 failure the guard exists to prevent -- pass the whole
suite, because every fixture box kept its bucket under the rescale. So the
fixtures below are chosen to STRADDLE a bucket boundary (30x30 = 900 px^2 is
`small` at 416 and `medium` at 640; 70x70 = 4,900 px^2 is `medium` at 416 and
`large` at 640) and they are driven through the aggregation functions that
actually produce reports/test_set_profile.md, not through `size_bucket`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.profile_test_set import load_test_samples, main, parse_args
from src.evaluation.slices import (
    EVALUATION_CONFIG,
    SLICE_NAMES,
    AnnotationSpaceError,
    SliceConfig,
    SliceConfigError,
    annotation_space_sizes,
    assert_areas_in_annotation_space,
    box_area,
    bucket_instance_counts,
    build_profile,
    class_image_counts,
    class_instance_counts,
    crowded_images,
    default_slice_config,
    image_mean_luminances,
    load_evaluation_config,
    load_slice_config,
    low_light_images,
    luma,
    mean_luminance,
    membership_histogram,
    pairwise_overlap,
    primary_instance_count,
    render_profile_markdown,
    scenario_slices,
    size_bucket,
    small_bucket_verdict,
    small_object_images,
)
from src.synthetic.composition import _luma
from src.training.data import CLASS_NAMES, Sample

HELMET, HEAD, PERSON = 0, 1, 2
TRAIN_RESOLUTION, ANNOTATION_RESOLUTION = 640, 416


def make_sample(
    image_id: int,
    boxes: list[tuple[float, float, float, float]],
    classes: list[int],
    *,
    image_path: Path | None = None,
    width: int = ANNOTATION_RESOLUTION,
    height: int = ANNOTATION_RESOLUTION,
) -> Sample:
    return Sample(
        image_path=image_path or Path(f"image_{image_id}.png"),
        image_id=image_id,
        boxes_xywh=tuple(boxes),
        class_indices=tuple(classes),
        is_synthetic=False,
        width=width,
        height=height,
    )


def square(side: float) -> tuple[float, float, float, float]:
    """A box of exactly side*side px, positioned somewhere harmless."""

    return (1.0, 1.0, side, side)


def box_of_area(area: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, 1.0, area)


def rescaled_to_640(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """The same box expressed in 640-space -- the EVAL-07 mistake, made explicit."""

    scale = TRAIN_RESOLUTION / ANNOTATION_RESOLUTION
    return tuple(value * scale for value in box)  # type: ignore[return-value]


def custom_config(
    *,
    area_ranges: tuple[tuple[str, float, float], ...] | None = None,
    small_object_min_count: int = 2,
    crowded_min_instances: int = 8,
    low_light_percentile: float = 25.0,
    primary_classes: tuple[str, ...] | None = None,
    small_bucket_min_instances: int | None = None,
) -> SliceConfig:
    """A config built in the test, so slice tests do not drift with the file."""

    shipped = default_slice_config()
    return SliceConfig(
        area_ranges=shipped.area_ranges if area_ranges is None else area_ranges,
        small_object_min_count=small_object_min_count,
        crowded_min_instances=crowded_min_instances,
        low_light_percentile=low_light_percentile,
        primary_classes=shipped.primary_classes if primary_classes is None else primary_classes,
        bootstrap_ci=shipped.bootstrap_ci,
        small_bucket_min_instances=(
            shipped.small_bucket_min_instances
            if small_bucket_min_instances is None
            else small_bucket_min_instances
        ),
    )


# --------------------------------------------------------------------------
# EVAL-07 — the area definition, and the trap it hides
# --------------------------------------------------------------------------


def test_spec_section_5_constructed_case() -> None:
    """Verbatim from docs/evaluation_spec.md section 5."""

    assert size_bucket(1000.0) == "small"
    assert size_bucket(1100.0) != "small"


def test_a_small_box_at_416_would_be_medium_at_640() -> None:
    """The EVAL-07 regression guard. Fails loudly if anyone evaluates at 640."""

    box = square(30.0)  # 900 px^2 in annotation coordinates
    assert size_bucket(box_area(box)) == "small"

    scale = TRAIN_RESOLUTION / ANNOTATION_RESOLUTION
    scaled = tuple(value * scale for value in box)
    assert size_bucket(box_area(scaled)) == "medium"


def test_bucket_counts_are_computed_in_annotation_space_not_640_space() -> None:
    """EVAL-07, guarded where the report is actually produced.

    Boxes chosen so EVERY one of them changes bucket under a 640/416 rescale:

    * 30x30   =    900 px^2  -> `small`  at 416;  x2.3669 =   2,130 -> `medium`
    * 70x70   =  4,900 px^2  -> `medium` at 416;  x2.3669 =  11,598 -> `large`
    * 200x200 = 40,000 px^2  -> `large`  at 416;  x2.3669 =  94,675 -> `large`

    The first two are the ones with teeth: if `bucket_instance_counts` ever
    rescales, helmet moves out of `small` and head moves out of `medium`, and
    this assertion is the thing that says so.
    """

    samples = [
        make_sample(1, [square(30.0), square(30.0), square(70.0)], [HELMET, HELMET, HEAD]),
        make_sample(2, [square(200.0)], [PERSON]),
    ]

    counts = bucket_instance_counts(samples)
    assert counts["helmet"] == {"small": 2, "medium": 0, "large": 0}
    assert counts["head"] == {"small": 0, "medium": 1, "large": 0}
    assert counts["person"] == {"small": 0, "medium": 0, "large": 1}

    # The same physical objects, handed in already rescaled to 640, must produce
    # a DIFFERENT table. That is what makes the assertion above discriminating
    # rather than decorative.
    in_640 = [
        make_sample(1, [rescaled_to_640(square(30.0))] * 2 + [rescaled_to_640(square(70.0))],
                    [HELMET, HELMET, HEAD]),
        make_sample(2, [rescaled_to_640(square(200.0))], [PERSON]),
    ]
    wrong = bucket_instance_counts(in_640)
    assert wrong["helmet"] == {"small": 0, "medium": 2, "large": 0}
    assert wrong["head"] == {"small": 0, "medium": 0, "large": 1}


def test_small_object_slice_is_computed_in_annotation_space_not_640_space() -> None:
    """EVAL-07 again, this time through the slice that feeds `slice_counts`.

    Image 1 holds two 30x30 boxes: 900 px^2 each, both `small` at 416, both
    `medium` at 640. With a threshold of 2 small boxes it qualifies in
    annotation space and qualifies in no other space. Image 2 holds one of each
    size and can never reach two small boxes.
    """

    config = custom_config(small_object_min_count=2)
    samples = [
        make_sample(1, [square(30.0), square(30.0)], [HELMET, HELMET]),
        make_sample(2, [square(30.0), square(70.0)], [HELMET, HEAD]),
    ]
    assert small_object_images(samples, config=config) == frozenset({1})

    in_640 = [
        make_sample(1, [rescaled_to_640(square(30.0))] * 2, [HELMET, HELMET]),
        make_sample(2, [rescaled_to_640(square(30.0)), rescaled_to_640(square(70.0))],
                    [HELMET, HEAD]),
    ]
    assert small_object_images(in_640, config=config) == frozenset()


def test_build_profile_buckets_and_slices_are_in_annotation_space() -> None:
    """The end the report is written from: totals and slice counts together.

    Same straddling boxes. In annotation space three of the five instances are
    `small`, one is `medium`, one is `large`, and both images clear a
    one-small-box `small_object` threshold. Rescale to 640 and none of those
    five numbers survives.
    """

    config = custom_config(small_object_min_count=1, crowded_min_instances=3)
    samples = [
        make_sample(1, [square(30.0), square(30.0), square(70.0)], [HELMET, HELMET, HEAD]),
        make_sample(2, [square(30.0), square(200.0)], [HEAD, PERSON]),
    ]
    luminances = {1: 10.0, 2: 20.0}

    profile = build_profile(samples, luminance_by_image=luminances, config=config)
    assert profile["bucket_totals"] == {"small": 3, "medium": 1, "large": 1}
    assert profile["slice_counts"]["small_object"] == 2
    # 3 of the 4 primary-class instances (2 helmet + 2 head) are small.
    assert profile["verdict"]["n_small_primary"] == 3
    assert profile["verdict"]["n_primary_total"] == 4

    in_640 = [
        make_sample(1, [rescaled_to_640(square(30.0))] * 2 + [rescaled_to_640(square(70.0))],
                    [HELMET, HELMET, HEAD]),
        make_sample(2, [rescaled_to_640(square(30.0)), rescaled_to_640(square(200.0))],
                    [HEAD, PERSON]),
    ]
    wrong = build_profile(in_640, luminance_by_image=luminances, config=config)
    assert wrong["bucket_totals"] == {"small": 0, "medium": 3, "large": 2}
    assert wrong["slice_counts"]["small_object"] == 0
    assert wrong["verdict"]["n_small_primary"] == 0


def test_boundary_convention_is_half_open() -> None:
    """Today's shipped bounds are 1024 and 9216; both belong to the bucket above."""

    ranges = default_slice_config().area_ranges
    (small, _, small_high), (medium, _, medium_high), (large, _, large_high) = ranges

    assert size_bucket(small_high - 1) == small
    assert size_bucket(small_high) == medium
    assert size_bucket(medium_high - 1) == medium
    assert size_bucket(medium_high) == large
    # The final range is closed at its top so nothing inside the configured
    # domain can fall through.
    assert size_bucket(large_high) == large


def test_zero_area_lands_in_the_smallest_bucket() -> None:
    assert size_bucket(0.0) == default_slice_config().smallest_bucket


def test_area_outside_the_configured_domain_raises() -> None:
    largest_high = default_slice_config().area_ranges[-1][2]
    with pytest.raises(ValueError, match="outside the configured ranges"):
        size_bucket(-1.0)
    with pytest.raises(ValueError, match="outside the configured ranges"):
        size_bucket(largest_high + 1.0)
    with pytest.raises(ValueError, match="finite"):
        size_bucket(float("nan"))


def test_size_bucket_accepts_explicit_ranges() -> None:
    ranges = (("tiny", 0.0, 10.0), ("huge", 10.0, 100.0))
    assert size_bucket(9.99, area_ranges=ranges) == "tiny"
    assert size_bucket(10.0, area_ranges=ranges) == "huge"


def test_box_area_is_width_times_height() -> None:
    assert box_area((17.0, 3.0, 4.0, 5.0)) == pytest.approx(20.0)


def test_annotation_space_sizes_reports_every_shape() -> None:
    samples = [
        make_sample(1, [], []),
        make_sample(2, [], []),
        make_sample(3, [], [], width=415, height=416),
    ]
    assert annotation_space_sizes(samples) == {"416x416": 2, "415x416": 1}


def test_assert_areas_in_annotation_space_accepts_consistent_coco() -> None:
    payload = {"annotations": [{"id": 1, "bbox": [0, 0, 32, 32], "area": 1024.0}]}
    assert_areas_in_annotation_space(payload)


def test_assert_areas_in_annotation_space_catches_rescaled_boxes() -> None:
    """Box scaled to 640 space, area left at 416 space -> exactly the EVAL-07 bug."""

    scale = TRAIN_RESOLUTION / ANNOTATION_RESOLUTION
    payload = {
        "annotations": [
            {"id": 7, "bbox": [0, 0, 32 * scale, 32 * scale], "area": 1024.0},
        ]
    }
    with pytest.raises(AnnotationSpaceError, match="different"):
        assert_areas_in_annotation_space(payload)


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------


def test_shipped_config_loads_and_is_cached() -> None:
    config = load_slice_config()
    assert config.bucket_names == ("small", "medium", "large")
    assert config.smallest_bucket == "small"
    assert config.primary_classes == ("helmet", "head")
    # The EVAL-08 absolute floor has to be a real population, not a token: the
    # 20-instance bucket the share-only rule called adequate must be below it.
    assert config.small_bucket_min_instances > 20
    assert default_slice_config() is default_slice_config()
    assert load_evaluation_config(EVALUATION_CONFIG)["metrics"]


def test_config_without_metrics_block_raises(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text("compliance:\n  mode: class_direct\n", encoding="utf-8", newline="\n")
    with pytest.raises(SliceConfigError, match="metrics"):
        load_evaluation_config(path)


def _write_config(tmp_path: Path, area_ranges: object) -> Path:
    payload = {
        "metrics": {
            "coco_area_ranges": area_ranges,
            "slice_small_object_min_count": 1,
            "slice_crowded_min_instances": 8,
            "slice_low_light_percentile": 25,
            "primary_classes": ["helmet", "head"],
            "bootstrap_ci": 0.95,
            "small_bucket_min_instances": 385,
        }
    }
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8", newline="\n")
    return path


def test_area_ranges_are_sorted_by_lower_bound(tmp_path: Path) -> None:
    path = _write_config(tmp_path, {"big": [10, 20], "little": [0, 10]})
    assert load_slice_config(path).bucket_names == ("little", "big")


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        ({}, "non-empty mapping"),
        ([[0, 10]], "non-empty mapping"),
        ({"broken": [0, 10, 20]}, r"\[low, high\] pair"),
        ({"empty": [10, 10]}, "is empty"),
        ({"a": [0, 10], "b": [11, 20]}, "do not meet"),
    ],
)
def test_bad_area_ranges_raise(tmp_path: Path, ranges: object, message: str) -> None:
    path = _write_config(tmp_path, ranges)
    with pytest.raises(SliceConfigError, match=message):
        load_slice_config(path)


# --------------------------------------------------------------------------
# EVAL-08 — counting
# --------------------------------------------------------------------------


def test_bucket_instance_counts_partitions_every_instance() -> None:
    samples = [
        make_sample(1, [square(20.0), square(40.0)], [HELMET, HEAD]),
        make_sample(2, [square(120.0), square(20.0)], [PERSON, HELMET]),
    ]
    counts = bucket_instance_counts(samples)
    assert counts["helmet"] == {"small": 2, "medium": 0, "large": 0}
    assert counts["head"] == {"small": 0, "medium": 1, "large": 0}
    assert counts["person"] == {"small": 0, "medium": 0, "large": 1}
    assert sum(sum(row.values()) for row in counts.values()) == 4


def test_class_counts_separate_instances_from_images() -> None:
    samples = [
        make_sample(1, [square(20.0), square(20.0)], [HELMET, HELMET]),
        make_sample(2, [square(20.0)], [HEAD]),
        make_sample(3, [], []),
    ]
    assert class_instance_counts(samples) == {"helmet": 2, "head": 1, "person": 0}
    assert class_image_counts(samples) == {"helmet": 1, "head": 1, "person": 0}


def test_unknown_class_index_raises() -> None:
    samples = [make_sample(1, [square(20.0)], [len(CLASS_NAMES)])]
    with pytest.raises(ValueError, match="outside"):
        bucket_instance_counts(samples)


# --------------------------------------------------------------------------
# Luminance — must agree with the synthesis-time definition
# --------------------------------------------------------------------------


def test_luma_matches_composition_exactly() -> None:
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, size=(8, 6, 3), dtype=np.uint8)
    np.testing.assert_array_equal(luma(patch), _luma(patch))
    assert mean_luminance(patch) == pytest.approx(float(_luma(patch).mean()))


def test_mean_luminance_of_a_known_grey() -> None:
    grey = np.full((4, 4, 3), 100, dtype=np.uint8)
    assert mean_luminance(grey) == pytest.approx(100.0, abs=1e-3)


def test_image_mean_luminances_reads_from_disk(tmp_path: Path) -> None:
    samples = []
    for index, value in enumerate((10, 200), start=1):
        path = tmp_path / f"image_{index}.png"
        Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)
        samples.append(make_sample(index, [], [], image_path=path))
    luminances = image_mean_luminances(samples)
    assert luminances[1] == pytest.approx(10.0, abs=1e-3)
    assert luminances[2] == pytest.approx(200.0, abs=1e-3)


# --------------------------------------------------------------------------
# EVAL-17 — scenario slices, each with a qualifier and a near miss
# --------------------------------------------------------------------------


def test_small_object_slice_qualifier_and_near_miss() -> None:
    config = custom_config(small_object_min_count=2)
    qualifies = make_sample(1, [square(20.0), square(20.0)], [HELMET, HEAD])
    just_misses = make_sample(2, [square(20.0), square(60.0)], [HELMET, HEAD])
    assert small_object_images([qualifies, just_misses], config=config) == frozenset({1})


def test_small_object_slice_with_the_shipped_threshold() -> None:
    """The shipped config asks for >= 1 small box, so one is enough."""

    qualifies = make_sample(1, [square(20.0)], [HELMET])
    just_misses = make_sample(2, [square(60.0)], [HELMET])
    assert small_object_images([qualifies, just_misses]) == frozenset({1})


def test_crowded_slice_qualifier_and_near_miss() -> None:
    config = custom_config(crowded_min_instances=8)
    qualifies = make_sample(1, [square(20.0)] * 8, [HELMET] * 8)
    just_misses = make_sample(2, [square(20.0)] * 7, [HELMET] * 7)
    assert crowded_images([qualifies, just_misses], config=config) == frozenset({1})


def test_crowded_slice_does_not_count_person() -> None:
    """CLAUDE.md: `person` must never carry load in a decision. This is a decision.

    On the real frozen Test split 8 of the 160 crowded images reached the
    threshold only because their `person` boxes were counted (image ids 4170,
    168, 1030, 3307, 1344 and 3420 among them), which defined a reported slice
    on the strength of the least complete labels in the dataset (ADR-003).
    """

    config = custom_config(crowded_min_instances=8)
    samples = [
        # 8 primary instances: crowded.
        make_sample(1, [square(20.0)] * 8, [HELMET] * 8),
        # 8 boxes but only 5 primary -- crowded ONLY if `person` is counted.
        make_sample(2, [square(20.0)] * 8, [HELMET] * 5 + [PERSON] * 3),
        # 8 primary across both primary classes: crowded.
        make_sample(3, [square(20.0)] * 9, [HELMET] * 7 + [HEAD] + [PERSON]),
        # 7 primary: the near miss, unchanged by the person rule.
        make_sample(4, [square(20.0)] * 7, [HELMET] * 7),
    ]
    assert crowded_images(samples, config=config) == frozenset({1, 3})
    assert [primary_instance_count(sample, config=config) for sample in samples] == [8, 5, 8, 7]


def test_low_light_slice_qualifier_and_near_miss() -> None:
    """Darkest 25% of 4 images is exactly one image; the second darkest misses."""

    config = custom_config(low_light_percentile=25.0)
    luminances = {1: 12.0, 2: 12.5, 3: 90.0, 4: 200.0}
    assert low_light_images(luminances, config=config) == frozenset({1})


def test_low_light_slice_handles_empty_and_zero_count() -> None:
    config = custom_config(low_light_percentile=25.0)
    assert low_light_images({}, config=config) == frozenset()
    # floor(3 * 25 / 100) == 0, so no image is dark enough to qualify.
    assert low_light_images({1: 5.0, 2: 6.0, 3: 7.0}, config=config) == frozenset()


def test_low_light_percentile_out_of_range_raises() -> None:
    config = custom_config(low_light_percentile=120.0)
    with pytest.raises(SliceConfigError, match="within"):
        low_light_images({1: 1.0}, config=config)


def test_slices_are_not_mutually_exclusive() -> None:
    config = custom_config(small_object_min_count=2, crowded_min_instances=3)
    samples = [
        make_sample(1, [square(20.0)] * 4, [HELMET] * 4),  # small + crowded + dark
        make_sample(2, [square(60.0)] * 4, [HELMET] * 4),  # crowded only
        make_sample(3, [square(20.0)] * 2, [HELMET] * 2),  # small only
        make_sample(4, [square(60.0)], [HELMET]),  # nothing
    ]
    luminances = {1: 1.0, 2: 50.0, 3: 60.0, 4: 70.0}
    slices = scenario_slices(samples, luminance_by_image=luminances, config=config)

    assert set(slices) == set(SLICE_NAMES)
    assert slices["small_object"] == frozenset({1, 3})
    assert slices["crowded"] == frozenset({1, 2})
    assert slices["low_light"] == frozenset({1})
    assert pairwise_overlap(slices) == {
        ("small_object", "crowded"): 1,
        ("small_object", "low_light"): 1,
        ("crowded", "low_light"): 1,
    }
    assert membership_histogram(slices, [1, 2, 3, 4]) == {0: 1, 1: 2, 2: 0, 3: 1}


def test_pairwise_overlap_tolerates_an_extra_slice() -> None:
    slices = {
        "crowded": frozenset({1, 2}),
        "small_object": frozenset({2}),
        "night_shift": frozenset({2, 3}),
    }
    overlap = pairwise_overlap(slices)
    # SLICE_NAMES order first, then any extra slice, sorted.
    assert list(overlap) == [
        ("small_object", "crowded"),
        ("small_object", "night_shift"),
        ("crowded", "night_shift"),
    ]
    assert overlap[("crowded", "night_shift")] == 1


# --------------------------------------------------------------------------
# EVAL-08 — the verdict
# --------------------------------------------------------------------------


def test_verdict_on_a_healthy_small_bucket() -> None:
    counts = {
        "helmet": {"small": 600, "medium": 300, "large": 100},
        "head": {"small": 300, "medium": 100, "large": 0},
        "person": {"small": 0, "medium": 0, "large": 50},
    }
    verdict = small_bucket_verdict(counts)
    assert verdict["n_small_primary"] == 900
    # `person` must not enter the primary-class arithmetic.
    assert verdict["n_primary_total"] == 1400
    assert verdict["is_thin"] is False
    assert verdict["is_empty"] is False
    assert verdict["per_instance_points"] == pytest.approx(100.0 / 900.0)


def test_single_arm_and_two_arm_noise_figures_are_different_quantities() -> None:
    """The half width of ONE arm's interval is not the floor for a GAP.

    N = 900 small primary instances, so sqrt(N) = 30. The worst-case binomial
    standard error of one proportion is 0.5/30; at the shipped 95% level the
    normal quantile is 1.9599640, so in percentage points

        single arm   = 100 * 1.9599640 * 0.5 / 30           = 3.26661
        two-arm gap  = 3.26661 * sqrt(2)                     = 4.61968

    They differ by ~41%, which is the whole reason the old single name was
    wrong: the report presented the 3.27 figure as the threshold a between-arm
    gap had to clear.
    """

    counts = {
        "helmet": {"small": 600, "medium": 300, "large": 100},
        "head": {"small": 300, "medium": 100, "large": 0},
        "person": {"small": 0, "medium": 0, "large": 50},
    }
    verdict = small_bucket_verdict(counts)
    assert verdict["single_arm_half_width_points"] == pytest.approx(3.26661, abs=1e-5)
    assert verdict["two_arm_gap_floor_points"] == pytest.approx(4.61968, abs=1e-5)
    # The misleading name must not come back.
    assert "noise_half_width_points" not in verdict


def test_verdict_on_a_thin_small_bucket() -> None:
    counts = {
        "helmet": {"small": 10, "medium": 500, "large": 400},
        "head": {"small": 5, "medium": 50, "large": 35},
        "person": {"small": 0, "medium": 0, "large": 0},
    }
    verdict = small_bucket_verdict(counts)
    assert verdict["n_small_primary"] == 15
    # 15 of 1,000 primary instances is 1.5%, so BOTH tests fail here.
    assert verdict["is_below_floor"] is True
    assert verdict["is_below_even_share"] is True
    assert verdict["is_thin"] is True
    assert verdict["is_empty"] is False


def test_a_high_share_of_a_tiny_bucket_is_still_thin() -> None:
    """The EVAL-08 case a share-only rule got wrong.

    20 instances that are 100% of the primary-class population is 100% of
    nothing much. Before the absolute floor existed this returned is_thin=False
    and the report emitted "**Verdict: adequate.** ... can carry the headline
    claim" over a 20-instance bucket.
    """

    counts = {
        "helmet": {"small": 20, "medium": 0, "large": 0},
        "head": {"small": 0, "medium": 0, "large": 0},
        "person": {"small": 0, "medium": 0, "large": 0},
    }
    verdict = small_bucket_verdict(counts)
    assert verdict["n_small_primary"] == 20
    assert verdict["small_share"] == pytest.approx(1.0)
    assert verdict["is_below_even_share"] is False  # the relative test passes
    assert verdict["is_below_floor"] is True  # the absolute one does not
    assert verdict["is_thin"] is True

    # Bracket the shipped floor from above with the real frozen Test split:
    # 1,447 + 607 = 2,054 small out of 2,825 + 879 = 3,704 primary instances.
    real = {
        "helmet": {"small": 1447, "medium": 1264, "large": 114},
        "head": {"small": 607, "medium": 269, "large": 3},
        "person": {"small": 22, "medium": 50, "large": 41},
    }
    on_test = small_bucket_verdict(real)
    assert (on_test["n_small_primary"], on_test["n_primary_total"]) == (2054, 3704)
    assert on_test["is_thin"] is False


def test_the_absolute_floor_compares_at_exactly_the_configured_count() -> None:
    """`n < floor` is thin, `n == floor` is not. Pins the comparison itself."""

    config = custom_config(small_bucket_min_instances=100)
    # 100 small of 250 primary is 40%, comfortably above the 1/3 even share, so
    # only the absolute floor can decide these two cases.
    at_floor = {
        "helmet": {"small": 100, "medium": 150, "large": 0},
        "head": {"small": 0, "medium": 0, "large": 0},
        "person": {"small": 0, "medium": 0, "large": 0},
    }
    below_floor = {
        "helmet": {"small": 99, "medium": 151, "large": 0},
        "head": {"small": 0, "medium": 0, "large": 0},
        "person": {"small": 0, "medium": 0, "large": 0},
    }
    assert small_bucket_verdict(at_floor, config=config)["min_instances"] == 100
    assert small_bucket_verdict(at_floor, config=config)["is_below_even_share"] is False
    assert small_bucket_verdict(at_floor, config=config)["is_thin"] is False
    assert small_bucket_verdict(below_floor, config=config)["is_below_even_share"] is False
    assert small_bucket_verdict(below_floor, config=config)["is_thin"] is True


def test_the_even_share_benchmark_counts_buckets_not_classes() -> None:
    """The relative test divides by the number of BUCKETS, not the number of classes.

    Every other fixture in this file has three of each -- three size buckets and
    three annotated classes -- so `1 / len(area_ranges)` and `1 / len(counts)`
    are the same float and a wrong denominator is invisible. This fixture breaks
    the tie: TWO configured buckets against THREE annotated classes.

    An even share of two buckets is 1/2 = 50.0%; an even share of three classes
    would be 1/3 = 33.3%. The primary-class small share here is
    (300 + 100) / (300 + 400 + 100 + 200) = 400 / 1,000 = 40.0%, which sits
    between the two -- below the correct benchmark and above the wrong one, so
    the verdict flips depending on which denominator the code used. The floor is
    set far under 400 so the absolute test cannot decide this case.
    """

    config = custom_config(
        area_ranges=(("small", 0.0, 1024.0), ("big", 1024.0, 1e8)),
        small_bucket_min_instances=10,
    )
    counts = {
        "helmet": {"small": 300, "big": 400},
        "head": {"small": 100, "big": 200},
        "person": {"small": 5, "big": 5},
    }
    verdict = small_bucket_verdict(counts, config=config)
    assert verdict["n_small_primary"] == 400
    assert verdict["n_primary_total"] == 1000
    assert verdict["small_share"] == pytest.approx(0.40)
    assert verdict["even_share"] == pytest.approx(0.50)
    assert verdict["is_below_floor"] is False
    assert verdict["is_below_even_share"] is True
    assert verdict["is_thin"] is True


def test_verdict_raises_when_a_configured_primary_class_is_missing() -> None:
    """A one-character typo in configs/evaluation.yaml must not pass silently.

    `heads` for `head` used to drop the class from the arithmetic: the headline
    went from 2,054 / 3,704 to 1,447 / 2,825 with no exception, no warning and
    no log line.
    """

    counts = {
        "helmet": {"small": 1447, "medium": 1264, "large": 114},
        "head": {"small": 607, "medium": 269, "large": 3},
        "person": {"small": 22, "medium": 50, "large": 41},
    }
    typo = custom_config(primary_classes=("helmet", "heads"))
    with pytest.raises(SliceConfigError, match="heads"):
        small_bucket_verdict(counts, config=typo)

    # Missing from the counts rather than from the config: same failure.
    with pytest.raises(SliceConfigError, match="head"):
        small_bucket_verdict({"helmet": counts["helmet"]})


def test_verdict_on_an_empty_small_bucket() -> None:
    counts = {
        "helmet": {"small": 0, "medium": 10, "large": 0},
        "head": {"small": 0, "medium": 5, "large": 0},
        "person": {"small": 0, "medium": 0, "large": 0},
    }
    verdict = small_bucket_verdict(counts)
    assert verdict["is_empty"] is True
    assert verdict["is_thin"] is True
    assert verdict["per_instance_points"] == float("inf")


def test_verdict_with_no_primary_instances_at_all() -> None:
    counts = {name: {"small": 0, "medium": 0, "large": 0} for name in CLASS_NAMES}
    verdict = small_bucket_verdict(counts)
    assert verdict["small_share"] == 0.0
    assert verdict["is_empty"] is True


# --------------------------------------------------------------------------
# Profile assembly and rendering
# --------------------------------------------------------------------------


def _profile_fixture(**overrides) -> dict:
    # A floor of 2 keeps this 3-instance fixture on the `adequate` side, so the
    # renderer's two verdict branches can each be exercised deliberately rather
    # than by accident.
    config = custom_config(
        small_object_min_count=1, crowded_min_instances=3, small_bucket_min_instances=2
    )
    samples = [
        make_sample(1, [square(20.0), square(20.0), square(20.0)], [HELMET, HELMET, HEAD]),
        make_sample(2, [square(60.0)], [HELMET]),
        make_sample(3, [square(120.0)], [PERSON]),
        make_sample(4, [], []),
    ]
    luminances = {1: 5.0, 2: 50.0, 3: 60.0, 4: 70.0}
    profile = build_profile(samples, luminance_by_image=luminances, config=config)
    profile.update(overrides)
    return profile


def test_build_profile_reports_every_section() -> None:
    profile = _profile_fixture()
    assert profile["n_images"] == 4
    assert profile["n_instances"] == 5
    assert profile["instances_per_class"] == {"helmet": 3, "head": 1, "person": 1}
    assert profile["images_per_class"] == {"helmet": 2, "head": 1, "person": 1}
    assert profile["bucket_totals"] == {"small": 3, "medium": 1, "large": 1}
    assert profile["annotation_space"] == {"416x416": 4}
    assert profile["slice_counts"] == {"small_object": 1, "crowded": 1, "low_light": 1}
    assert profile["slice_pairwise_overlap"]["small_object & crowded"] == 1
    assert profile["membership_histogram"] == {0: 3, 1: 0, 2: 0, 3: 1}
    assert profile["thresholds"]["slice_crowded_min_instances"] == 3


def test_render_markdown_contains_the_numbers_and_the_verdict() -> None:
    markdown = render_profile_markdown(_profile_fixture())
    assert "# Frozen Test Set Profile" in markdown
    # 3 of the 4 primary-class instances are small: above an even share, and
    # above this fixture's floor of 2.
    assert "Verdict: adequate" in markdown
    assert "416x416" in markdown
    assert "small_object & crowded" in markdown
    assert "2.37" in markdown  # the EVAL-07 caveat survives into the report
    # Verdict paragraphs must be separated by a blank line or markdown runs them
    # together into one wall of text.
    assert "\n\n**Verdict: adequate.**" in markdown


def test_render_markdown_writes_the_tables_the_profile_actually_contains() -> None:
    """Every rendered cell, checked by hand against the fixture.

    The renderer IS the deliverable -- reports/test_set_profile.md is what a
    reader sees -- and a swapped column or a percentage taken over the wrong
    denominator is invisible to every assertion made on the profile dict. The
    fixture holds

      image 1 - three 20x20 boxes (400 px^2, `small`): helmet, helmet, head
      image 2 - one 60x60 box (3,600 px^2, `medium`): helmet
      image 3 - one 120x120 box (14,400 px^2, `large`): person
      image 4 - no annotations

    so 4 images and 5 instances, and every number below is arithmetic on those.
    """

    markdown = render_profile_markdown(_profile_fixture())

    # 4 images and 5 instances -- and not the other way round.
    assert "- Test images: **4**" in markdown
    assert "- Annotated instances: **5**" in markdown

    # helmet is 3 instances spread over 2 images = 1.50 per image (not 2/3).
    assert "| `helmet` | 3 | 2 | 1.50 |" in markdown
    assert "| `head` | 1 | 1 | 1.00 |" in markdown
    assert "| `person` | 1 | 1 | 1.00 |" in markdown

    # A bucket percentage is a share of that CLASS's own total, not of the
    # split: helmet is 2 small + 1 medium of 3, so 66.7% / 33.3% / 0.0%.
    assert "| `helmet` | 2 | 66.7% | 1 | 33.3% | 0 | 0.0% | 3 |" in markdown
    assert "| `head` | 1 | 100.0% | 0 | 0.0% | 0 | 0.0% | 1 |" in markdown
    assert "| `person` | 0 | 0.0% | 0 | 0.0% | 1 | 100.0% | 1 |" in markdown
    # The totals row is a share of all 5 instances: 3 small, 1 medium, 1 large.
    assert "| **all classes** | 3 | 60.0% | 1 | 20.0% | 1 | 20.0% | 5 |" in markdown

    # A slice percentage is a share of the 4 IMAGES, so one image is 25.0%.
    assert "| `small_object` | 1 | 25.0% |" in markdown
    assert "| `crowded` | 1 | 25.0% |" in markdown
    assert "| `low_light` | 1 | 25.0% |" in markdown

    # Only image 1 lands in any slice and it lands in all three, so 3 images are
    # in nothing and 1 is in three. The rows must ascend 0, 1, 2, 3.
    assert (
        "| 0 | 3 | 75.0% |\n| 1 | 0 | 0.0% |\n| 2 | 0 | 0.0% |\n| 3 | 1 | 25.0% |"
    ) in markdown

    # The prose must not contradict test_boundary_convention_is_half_open, which
    # pins the CODE to sending a boundary area up rather than down.
    assert "An area of exactly a boundary value belongs to the upper bucket." in markdown


def test_render_markdown_says_person_is_excluded_from_the_crowded_slice() -> None:
    markdown = render_profile_markdown(_profile_fixture())
    assert "`person` is deliberately not counted" in markdown
    # The rule line must name the classes that ARE counted.
    assert "annotated `helmet` + `head` instances" in markdown


def test_render_markdown_states_each_slice_rule_over_the_right_bucket() -> None:
    """The `small_object` rule line must name the SMALLEST bucket by name.

    `bucket_names[0]` and `bucket_names[-1]` are both valid indices into the
    same list, so naming the wrong end yields a report that tells the reader
    "at least 1 box(es) in the `large` bucket" while the code goes on counting
    small ones -- a report that misdescribes its own slice definition, which no
    assertion on the slice COUNTS can detect.

    This fixture sets the threshold to 1 box and the low-light percentile to 25,
    so both rule lines are fully determined by the config handed in.
    """

    markdown = render_profile_markdown(_profile_fixture())
    assert "at least 1 box(es) in the `small` bucket" in markdown
    assert "the darkest 25% of images by mean BT.601 luma" in markdown


def test_render_markdown_separates_the_single_arm_and_two_arm_noise_figures() -> None:
    """The report must not present one arm's own half width as a gap threshold.

    Verdict built on N = 900, so (see the verdict test above) the single-arm
    half width is 3.27 points and the two-arm gap floor is 4.62 points. Both
    have to appear, in their own sentences, or the reader takes the smaller
    number as the bar a between-arm difference must clear.
    """

    profile = _profile_fixture()
    profile["verdict"] = small_bucket_verdict(
        {
            "helmet": {"small": 600, "medium": 300, "large": 100},
            "head": {"small": 300, "medium": 100, "large": 0},
            "person": {"small": 0, "medium": 0, "large": 50},
        }
    )
    markdown = render_profile_markdown(profile)
    assert "single arm" in markdown
    assert "difference between two arms" in markdown
    assert "±3.27 points" in markdown
    assert "±4.62 points" in markdown
    # The claim that survives to the verdict line is the two-arm one.
    assert "±4.62 point between-arm floor" in markdown


def test_render_markdown_warns_about_mixed_resolutions() -> None:
    """The real Test split is 416x415 for most images, 416x416 for the rest."""

    profile = _profile_fixture()
    assert "not one single resolution" not in render_profile_markdown(profile)

    profile["annotation_space"] = {"416x415": 3, "416x416": 1}
    assert "not one single resolution" in render_profile_markdown(profile)


def test_render_markdown_for_a_thin_bucket() -> None:
    profile = _profile_fixture()
    profile["verdict"] = small_bucket_verdict(
        {
            "helmet": {"small": 10, "medium": 500, "large": 400},
            "head": {"small": 5, "medium": 50, "large": 35},
            "person": {"small": 0, "medium": 0, "large": 0},
        }
    )
    markdown = render_profile_markdown(profile)
    assert "Verdict: thin" in markdown
    assert "Verdict: adequate" not in markdown
    # 15 of 1,000 is 1.5%, which fails both tests; the report must say both.
    assert "1.5% of the 1,000 primary-class instances" in markdown
    assert "fewer than the required" in markdown
    # The benchmark quoted in the failure list is the even share, 1/3 of three
    # buckets = 33.3%.
    assert "below an even share (33.3%)" in markdown


def test_render_markdown_names_the_absolute_floor_as_the_only_reason() -> None:
    """A high-share, low-count bucket: the report must not call it adequate."""

    profile = _profile_fixture()
    profile["verdict"] = small_bucket_verdict(
        {
            "helmet": {"small": 20, "medium": 0, "large": 0},
            "head": {"small": 0, "medium": 0, "large": 0},
            "person": {"small": 0, "medium": 0, "large": 0},
        }
    )
    markdown = render_profile_markdown(profile)
    assert "Verdict: thin" in markdown
    assert "can carry the headline claim" not in markdown
    assert "fewer than the required" in markdown
    # 20 of 20 is 100%, so the relative test is NOT one of the reasons.
    assert "below an even share" not in markdown


def test_render_markdown_for_an_empty_bucket() -> None:
    profile = _profile_fixture()
    profile["verdict"] = small_bucket_verdict(
        {name: {"small": 0, "medium": 3, "large": 0} for name in CLASS_NAMES}
    )
    markdown = render_profile_markdown(profile)
    assert "cannot carry the headline claim" in markdown
    assert "undefined" in markdown


def test_render_markdown_survives_an_empty_split() -> None:
    """Zero images must render as `n/a`, not divide by zero."""

    config = custom_config()
    profile = build_profile([], luminance_by_image={}, config=config)
    markdown = render_profile_markdown(profile)
    assert "n/a" in markdown


# --------------------------------------------------------------------------
# The script end to end
# --------------------------------------------------------------------------


def _tiny_dataset(tmp_path: Path, *, splits: dict[str, str]) -> dict[str, Path]:
    images_root = tmp_path / "images"
    images_root.mkdir()
    coco: dict[str, list] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }
    manifest_images = []
    for index, (file_name, split) in enumerate(sorted(splits.items()), start=1):
        Image.fromarray(
            np.full((16, 16, 3), 20 * index, dtype=np.uint8)
        ).save(images_root / file_name)
        coco["images"].append(
            {"id": index, "file_name": f"images/{file_name}", "width": 416, "height": 416}
        )
        coco["annotations"].append(
            {
                "id": index,
                "image_id": index,
                "category_id": 1,
                "bbox": [0.0, 0.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            }
        )
        manifest_images.append({"file_name": f"images/{file_name}", "split": split})

    annotations = tmp_path / "coco_all.json"
    annotations.write_text(json.dumps(coco), encoding="utf-8", newline="\n")
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        json.dumps({"images": manifest_images}), encoding="utf-8", newline="\n"
    )
    return {"annotations": annotations, "manifest": manifest, "images_root": images_root}


def test_script_writes_a_report_for_the_frozen_test_split(tmp_path: Path) -> None:
    fixture = _tiny_dataset(
        tmp_path, splits={"a.png": "train", "b.png": "test", "c.png": "test"}
    )
    output = tmp_path / "reports" / "test_set_profile.md"
    exit_code = main(
        [
            "--manifest",
            str(fixture["manifest"]),
            "--annotations",
            str(fixture["annotations"]),
            "--images-root",
            str(fixture["images_root"]),
            "--output",
            str(output),
            "--config",
            str(EVALUATION_CONFIG),
        ]
    )
    assert exit_code == 0
    report = output.read_text(encoding="utf-8")
    assert "Test images: **2**" in report
    assert "Annotated instances: **2**" in report


def test_script_runs_with_no_config_flag_at_all(tmp_path: Path) -> None:
    """The invocation a human actually types: `uv run python -m scripts.profile_test_set`.

    Omitting `--config` used to take a branch inside main() that no test ever
    executed, because the end-to-end test always passed one.
    """

    fixture = _tiny_dataset(tmp_path, splits={"a.png": "test", "b.png": "test"})
    output = tmp_path / "reports" / "test_set_profile.md"
    exit_code = main(
        [
            "--manifest",
            str(fixture["manifest"]),
            "--annotations",
            str(fixture["annotations"]),
            "--images-root",
            str(fixture["images_root"]),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0

    report = output.read_text(encoding="utf-8")
    # Both fixture images carry exactly one 20x20 = 400 px^2 box, which is
    # `small`. The shipped config asks for >= 1 small box, so both images are in
    # `small_object`; its crowded threshold is far above 1, so neither is
    # crowded. Those two lines are only produced if the shipped thresholds were
    # the ones actually loaded.
    assert "| `small_object` | 2 | 100.0% |" in report
    assert "| `crowded` | 0 | 0.0% |" in report


def test_script_defaults_come_from_the_path_config() -> None:
    args = parse_args([])
    assert args.manifest.name == "split_manifest.json"
    assert args.annotations.name == "coco_all.json"
    assert args.output.name == "test_set_profile.md"
    # `--config` defaults to the shipped file rather than to None, so main() has
    # one config-loading path instead of a tested one and a real one.
    assert args.config.name == "evaluation.yaml"
    assert args.config.is_file()


def test_script_refuses_a_manifest_without_a_test_split(tmp_path: Path) -> None:
    fixture = _tiny_dataset(tmp_path, splits={"a.png": "train", "b.png": "val"})
    with pytest.raises(SystemExit, match="no 'test' split"):
        load_test_samples(
            fixture["manifest"], fixture["annotations"], fixture["images_root"]
        )


def test_script_refuses_when_the_loaded_images_differ_from_the_split(
    tmp_path: Path,
) -> None:
    fixture = _tiny_dataset(tmp_path, splits={"a.png": "test", "b.png": "test"})
    payload = json.loads(fixture["annotations"].read_text(encoding="utf-8"))
    payload["images"] = payload["images"][:1]
    payload["annotations"] = payload["annotations"][:1]
    fixture["annotations"].write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    with pytest.raises(SystemExit, match="frozen test split names"):
        load_test_samples(
            fixture["manifest"], fixture["annotations"], fixture["images_root"]
        )


def test_script_refuses_a_coco_whose_areas_are_in_another_space(tmp_path: Path) -> None:
    fixture = _tiny_dataset(tmp_path, splits={"a.png": "test"})
    payload = json.loads(fixture["annotations"].read_text(encoding="utf-8"))
    payload["annotations"][0]["area"] = 999.0
    fixture["annotations"].write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    with pytest.raises(AnnotationSpaceError):
        load_test_samples(
            fixture["manifest"], fixture["annotations"], fixture["images_root"]
        )


# --------------------------------------------------------------------------
# Bad inputs must name the file. A raw JSONDecodeError / KeyError traceback
# through split_real_images tells the operator nothing about what to fix.
# --------------------------------------------------------------------------


def test_script_names_the_manifest_when_it_is_not_json(tmp_path: Path) -> None:
    """json.JSONDecodeError is a ValueError; it must not escape raw."""

    fixture = _tiny_dataset(tmp_path, splits={"a.png": "test"})
    fixture["manifest"].write_text("{ not json", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit, match="Cannot read the frozen split manifest"):
        load_test_samples(
            fixture["manifest"], fixture["annotations"], fixture["images_root"]
        )


def test_script_names_the_manifest_when_an_entry_has_no_file_name(tmp_path: Path) -> None:
    """The KeyError path: valid JSON, wrong shape."""

    fixture = _tiny_dataset(tmp_path, splits={"a.png": "test"})
    fixture["manifest"].write_text(
        json.dumps({"images": [{"split": "test"}]}), encoding="utf-8", newline="\n"
    )
    with pytest.raises(SystemExit, match="Cannot read the frozen split manifest"):
        load_test_samples(
            fixture["manifest"], fixture["annotations"], fixture["images_root"]
        )


def test_script_names_the_annotations_file_when_it_is_absent(tmp_path: Path) -> None:
    """The OSError path."""

    fixture = _tiny_dataset(tmp_path, splits={"a.png": "test"})
    with pytest.raises(SystemExit, match="Cannot read the COCO annotations"):
        load_test_samples(
            fixture["manifest"], tmp_path / "absent.json", fixture["images_root"]
        )


def test_script_names_the_annotations_file_when_categories_are_missing(
    tmp_path: Path,
) -> None:
    """Consistent areas, but nothing to map category ids with."""

    fixture = _tiny_dataset(tmp_path, splits={"a.png": "test"})
    payload = json.loads(fixture["annotations"].read_text(encoding="utf-8"))
    del payload["categories"]
    fixture["annotations"].write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit, match="Cannot build samples from"):
        load_test_samples(
            fixture["manifest"], fixture["annotations"], fixture["images_root"]
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.pop("annotations"), "no `annotations` list"),
        (lambda payload: payload["annotations"][0].pop("area"), "has no 'area'"),
        (lambda payload: payload["annotations"][0].pop("bbox"), "has no 'bbox'"),
        (
            lambda payload: payload["annotations"][0].update({"bbox": [0.0, 0.0, 20.0]}),
            "unreadable area/bbox",
        ),
        (
            lambda payload: payload["annotations"][0].update({"area": "four hundred"}),
            "unreadable area/bbox",
        ),
    ],
)
def test_unverifiable_coco_raises_the_typed_error(
    tmp_path: Path, mutate, message: str
) -> None:
    """A payload whose areas cannot be CHECKED has not met the EVAL-07 contract."""

    fixture = _tiny_dataset(tmp_path, splits={"a.png": "test"})
    payload = json.loads(fixture["annotations"].read_text(encoding="utf-8"))
    mutate(payload)
    fixture["annotations"].write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    with pytest.raises(AnnotationSpaceError, match=message):
        load_test_samples(
            fixture["manifest"], fixture["annotations"], fixture["images_root"]
        )
