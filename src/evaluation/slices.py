"""Frozen-Test profiling: COCO size buckets (EVAL-08) and scenario slices (EVAL-17).

Nothing here needs a trained model, which is exactly the point. If the frozen
Test split holds very few `small` instances then AP_small -- one of the two
headline metrics -- is measuring noise, and EVAL-08 obliges us to say so in the
report instead of discovering it after the analysis has been written.

Two traps are designed into this module:

* EVAL-07. Areas are ``w * h`` in the ORIGINAL 416x416 annotation coordinates.
  Training happens at 640, where the same object is ``(640/416)^2 ~ 2.37`` times
  larger and most `small` objects silently become `medium`. Nothing raises when
  that is wrong, so `annotation_space_sizes` reports which coordinate space the
  numbers were computed in and `assert_areas_in_annotation_space` checks that a
  source COCO file agrees with its own boxes.
* Luminance. `src/synthetic/composition._luma` already defines BT.601 luma for
  this project. A second, disagreeing definition here would make the low-light
  slice mean something different from the low-light logic used during synthesis,
  so the coefficients below are that function's and tests/test_evaluation_slices
  asserts the two agree exactly.

Every threshold is read from configs/evaluation.yaml; this module holds no
numeric thresholds of its own.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import yaml
from PIL import Image

from src.data.paths import PROJECT_ROOT
from src.training.data import CLASS_NAMES, Sample

EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"

# Order is fixed so the report columns and the overlap pairs are stable.
SLICE_NAMES = ("small_object", "crowded", "low_light")

# BT.601 luma weights, copied from src/synthetic/composition._luma so the
# low-light slice and the synthesis-time legibility logic agree by construction.
BT601_COEFFICIENTS = (0.299, 0.587, 0.114)

# Float comparison epsilon for "area equals w*h". Not a tunable threshold.
_AREA_EPSILON = 1e-9


class SliceConfigError(RuntimeError):
    """Raised when configs/evaluation.yaml cannot drive the slicing."""


class AnnotationSpaceError(RuntimeError):
    """Raised when a COCO file's `area` disagrees with its own boxes (EVAL-07)."""


@dataclass(frozen=True)
class SliceConfig:
    """Every number this module uses, loaded from configs/evaluation.yaml."""

    area_ranges: tuple[tuple[str, float, float], ...]
    small_object_min_count: int
    crowded_min_instances: int
    low_light_percentile: float
    primary_classes: tuple[str, ...]
    bootstrap_ci: float
    small_bucket_min_instances: int

    @property
    def bucket_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.area_ranges)

    @property
    def smallest_bucket(self) -> str:
        return self.area_ranges[0][0]


def load_evaluation_config(path: Path = EVALUATION_CONFIG) -> dict[str, Any]:
    """Read configs/evaluation.yaml with explicit UTF-8 (K-10)."""

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict) or "metrics" not in payload:
        raise SliceConfigError(f"{path} has no `metrics` block")
    return payload


def _ordered_area_ranges(raw: Any) -> tuple[tuple[str, float, float], ...]:
    """Sort the configured ranges by lower bound and prove they tile the axis."""

    if not isinstance(raw, Mapping) or not raw:
        raise SliceConfigError("metrics.coco_area_ranges must be a non-empty mapping")

    ranges: list[tuple[str, float, float]] = []
    for name, bounds in raw.items():
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise SliceConfigError(f"Area range {name!r} must be a [low, high] pair")
        low, high = (float(value) for value in bounds)
        if not low < high:
            raise SliceConfigError(f"Area range {name!r} is empty: [{low}, {high})")
        ranges.append((str(name), low, high))

    ranges.sort(key=lambda item: item[1])
    for (left_name, _, left_high), (right_name, right_low, _) in itertools.pairwise(ranges):
        if not math.isclose(left_high, right_low, rel_tol=_AREA_EPSILON, abs_tol=_AREA_EPSILON):
            raise SliceConfigError(
                f"Area ranges {left_name!r} and {right_name!r} do not meet: "
                f"{left_high} != {right_low}"
            )
    return tuple(ranges)


def load_slice_config(path: Path = EVALUATION_CONFIG) -> SliceConfig:
    """Build the immutable threshold bundle from configs/evaluation.yaml."""

    metrics = load_evaluation_config(path)["metrics"]
    return SliceConfig(
        area_ranges=_ordered_area_ranges(metrics["coco_area_ranges"]),
        small_object_min_count=int(metrics["slice_small_object_min_count"]),
        crowded_min_instances=int(metrics["slice_crowded_min_instances"]),
        low_light_percentile=float(metrics["slice_low_light_percentile"]),
        primary_classes=tuple(str(name) for name in metrics["primary_classes"]),
        bootstrap_ci=float(metrics["bootstrap_ci"]),
        small_bucket_min_instances=int(metrics["small_bucket_min_instances"]),
    )


@lru_cache(maxsize=1)
def default_slice_config() -> SliceConfig:
    """Cached view of the shipped config. Frozen + tuples, so sharing is safe."""

    return load_slice_config()


def _resolve(config: SliceConfig | None) -> SliceConfig:
    return default_slice_config() if config is None else config


# spec: EVAL-07
def box_area(box_xywh: Sequence[float]) -> float:
    """COCO area of one [x, y, w, h] box, in whatever space the box is in.

    Callers are responsible for handing this ORIGINAL 416x416 boxes. Feeding it
    640-space boxes returns a number ~2.37x too large and raises nothing.
    """

    _, _, width, height = (float(value) for value in box_xywh)
    return width * height


# spec: EVAL-07
def size_bucket(
    area: float, *, area_ranges: Sequence[tuple[str, float, float]] | None = None
) -> str:
    """Bucket a COCO area (416-space px^2) into one configured size bucket.

    Boundary convention: half-open ``[low, high)``, so with the shipped config an
    area of exactly 1024 is `medium`, not `small`, and exactly 9216 is `large`,
    not `medium`. The final range is closed at its top end so no area inside the
    configured domain can fall through.

    This differs from pycocotools, whose areaRng comparison is closed at BOTH
    ends -- there an object of area exactly 1024 is evaluated in the small range
    *and* in the medium range. That double-counting is acceptable for AP, but
    EVAL-08 wants a partition of the instances, so exactly one bucket wins here.
    """

    ranges = tuple(area_ranges) if area_ranges is not None else default_slice_config().area_ranges
    value = float(area)
    if not math.isfinite(value):
        raise ValueError(f"Area must be finite, got {area!r}")

    last_index = len(ranges) - 1
    for index, (name, low, high) in enumerate(ranges):
        if low <= value < high:
            return name
        if index == last_index and value == high:
            return name
    raise ValueError(
        f"Area {value} lies outside the configured ranges "
        f"[{ranges[0][1]}, {ranges[-1][2]}]"
    )


def _class_name(index: int) -> str:
    if not 0 <= int(index) < len(CLASS_NAMES):
        raise ValueError(f"Class index {index} is outside {CLASS_NAMES}")
    return CLASS_NAMES[int(index)]


# spec: EVAL-08
def bucket_instance_counts(
    samples: Sequence[Sample], *, config: SliceConfig | None = None
) -> dict[str, dict[str, int]]:
    """Per-class x per-size-bucket instance counts, in annotation coordinates."""

    settings = _resolve(config)
    counts = {
        name: dict.fromkeys(settings.bucket_names, 0) for name in CLASS_NAMES
    }
    for sample in samples:
        for box, class_index in zip(sample.boxes_xywh, sample.class_indices, strict=True):
            bucket = size_bucket(box_area(box), area_ranges=settings.area_ranges)
            counts[_class_name(class_index)][bucket] += 1
    return counts


def class_instance_counts(samples: Sequence[Sample]) -> dict[str, int]:
    """Instances per class."""

    counts = dict.fromkeys(CLASS_NAMES, 0)
    for sample in samples:
        for class_index in sample.class_indices:
            counts[_class_name(class_index)] += 1
    return counts


def class_image_counts(samples: Sequence[Sample]) -> dict[str, int]:
    """Images containing at least one instance of each class."""

    counts = dict.fromkeys(CLASS_NAMES, 0)
    for sample in samples:
        for name in {_class_name(index) for index in sample.class_indices}:
            counts[name] += 1
    return counts


# spec: EVAL-07
def annotation_space_sizes(samples: Sequence[Sample]) -> dict[str, int]:
    """How many images carry each (width, height). Evidence for the 416 claim."""

    sizes: dict[str, int] = {}
    for sample in samples:
        key = f"{int(sample.width)}x{int(sample.height)}"
        sizes[key] = sizes.get(key, 0) + 1
    return dict(sorted(sizes.items(), key=lambda item: (-item[1], item[0])))


# spec: EVAL-07
def assert_areas_in_annotation_space(coco_payload: Mapping[str, Any]) -> None:
    """Every COCO `area` must equal w*h of its own box.

    A rescaled-to-640 box next to an untouched 416 area (or the reverse) is
    exactly the EVAL-07 failure, and it is otherwise invisible.

    A payload this function cannot check is also a failure: a missing
    `annotations` list, a missing `area` or `bbox`, or a bbox that is not four
    numbers all raise `AnnotationSpaceError` rather than escaping as a raw
    KeyError / ValueError, because the caller's contract is "these areas are in
    the annotation coordinate space" and an unverifiable payload has not met it.
    """

    if "annotations" not in coco_payload:
        raise AnnotationSpaceError(
            "COCO payload has no `annotations` list, so its areas cannot be checked "
            "against its boxes (EVAL-07)"
        )
    for annotation in coco_payload["annotations"]:
        for key in ("area", "bbox"):
            if key not in annotation:
                raise AnnotationSpaceError(
                    f"Annotation {annotation.get('id')} has no {key!r}; the areas "
                    "cannot be checked against the boxes (EVAL-07)"
                )
        try:
            area = float(annotation["area"])
            expected = box_area(annotation["bbox"])
        except (TypeError, ValueError) as error:
            raise AnnotationSpaceError(
                f"Annotation {annotation.get('id')} has an unreadable area/bbox "
                f"({annotation['area']!r} / {annotation['bbox']!r}): {error}"
            ) from error
        if not math.isclose(area, expected, rel_tol=1e-6, abs_tol=_AREA_EPSILON):
            raise AnnotationSpaceError(
                f"Annotation {annotation.get('id')} declares area {area} but its box "
                f"measures {expected}; the boxes and the areas are in different "
                "coordinate spaces (EVAL-07)"
            )


def luma(image_rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma, bit-identical to src/synthetic/composition._luma."""

    patch = np.asarray(image_rgb, dtype=np.float32)
    return (
        BT601_COEFFICIENTS[0] * patch[..., 0]
        + BT601_COEFFICIENTS[1] * patch[..., 1]
        + BT601_COEFFICIENTS[2] * patch[..., 2]
    )


def mean_luminance(image_rgb: np.ndarray) -> float:
    """Mean BT.601 luma of one RGB image, in the input's own 0-255 scale."""

    return float(luma(image_rgb).mean())


def image_mean_luminances(samples: Sequence[Sample]) -> dict[int, float]:
    """Mean luma per image id, read from disk once per sample."""

    luminances: dict[int, float] = {}
    for sample in samples:
        with Image.open(sample.image_path) as handle:
            rgb = np.asarray(handle.convert("RGB"))
        luminances[int(sample.image_id)] = mean_luminance(rgb)
    return luminances


# spec: EVAL-17
def small_object_images(
    samples: Sequence[Sample], *, config: SliceConfig | None = None
) -> frozenset[int]:
    """Images with at least `slice_small_object_min_count` smallest-bucket boxes."""

    settings = _resolve(config)
    smallest = settings.smallest_bucket
    selected = set()
    for sample in samples:
        n_small = sum(
            1
            for box in sample.boxes_xywh
            if size_bucket(box_area(box), area_ranges=settings.area_ranges) == smallest
        )
        if n_small >= settings.small_object_min_count:
            selected.add(int(sample.image_id))
    return frozenset(selected)


# spec: EVAL-17
def primary_instance_count(sample: Sample, *, config: SliceConfig | None = None) -> int:
    """Instances of `metrics.primary_classes` on one image.

    `person` is excluded on purpose. CLAUDE.md forbids that class carrying load
    in any decision, and ADR-003 records that its annotations are the least
    complete in the dataset, so letting it push an image over a threshold would
    define a slice on the strength of the labels we trust least.
    """

    settings = _resolve(config)
    primary = set(settings.primary_classes)
    return sum(1 for index in sample.class_indices if _class_name(index) in primary)


# spec: EVAL-17
def crowded_images(
    samples: Sequence[Sample], *, config: SliceConfig | None = None
) -> frozenset[int]:
    """Images with at least `slice_crowded_min_instances` PRIMARY-class instances."""

    settings = _resolve(config)
    return frozenset(
        int(sample.image_id)
        for sample in samples
        if primary_instance_count(sample, config=settings) >= settings.crowded_min_instances
    )


# spec: EVAL-17
def low_light_images(
    luminance_by_image: Mapping[int, float], *, config: SliceConfig | None = None
) -> frozenset[int]:
    """The darkest `slice_low_light_percentile` percent of images by mean luma.

    Rank based rather than threshold based, so the slice size is exactly
    ``floor(n * percentile / 100)`` and does not depend on how the luminance
    histogram happens to clump. Ties break on image id, which is arbitrary but
    deterministic: two images of identical mean luma can land on opposite sides
    of the cut.
    """

    settings = _resolve(config)
    percentile = settings.low_light_percentile
    if not 0.0 <= percentile <= 100.0:
        raise SliceConfigError(
            f"slice_low_light_percentile must be within [0, 100], got {percentile}"
        )
    total = len(luminance_by_image)
    count = math.floor(total * percentile / 100.0)
    if count <= 0:
        return frozenset()
    ordered = sorted(luminance_by_image.items(), key=lambda item: (item[1], item[0]))
    return frozenset(image_id for image_id, _ in ordered[:count])


# spec: EVAL-17
def scenario_slices(
    samples: Sequence[Sample],
    *,
    luminance_by_image: Mapping[int, float],
    config: SliceConfig | None = None,
) -> dict[str, frozenset[int]]:
    """All three EVAL-17 slices. They are NOT mutually exclusive."""

    settings = _resolve(config)
    return {
        "small_object": small_object_images(samples, config=settings),
        "crowded": crowded_images(samples, config=settings),
        "low_light": low_light_images(luminance_by_image, config=settings),
    }


# spec: EVAL-17
def pairwise_overlap(slices: Mapping[str, frozenset[int]]) -> dict[tuple[str, str], int]:
    """Image counts shared by each pair of slices, in SLICE_NAMES order."""

    names = [name for name in SLICE_NAMES if name in slices]
    names += sorted(set(slices) - set(SLICE_NAMES))
    return {
        (left, right): len(slices[left] & slices[right])
        for left, right in itertools.combinations(names, 2)
    }


# spec: EVAL-17
def membership_histogram(
    slices: Mapping[str, frozenset[int]], image_ids: Sequence[int]
) -> dict[int, int]:
    """How many images belong to exactly 0, 1, 2, ... slices."""

    histogram = dict.fromkeys(range(len(slices) + 1), 0)
    for image_id in image_ids:
        member_count = sum(1 for members in slices.values() if int(image_id) in members)
        histogram[member_count] += 1
    return histogram


# spec: EVAL-08
def small_bucket_verdict(
    bucket_counts: Mapping[str, Mapping[str, int]], *, config: SliceConfig | None = None
) -> dict[str, Any]:
    """Can AP_small carry the headline claim on this Test split?

    Two INDEPENDENT tests, and the bucket is `thin` unless it passes both:

    * absolute -- at least `metrics.small_bucket_min_instances` instances. A
      relative test on its own is not enough: a bucket of 20 instances is 100%
      of a 20-instance split and still measures nothing, and EVAL-08 exists to
      make us disclose exactly that situation rather than call it adequate.
    * relative -- the bucket's share of primary-class instances against an even
      share (1 / number of configured buckets).

    Three derived quantities, no invented constants:

    * `per_instance_points` -- one instance is 1/N of the bucket, so any
      recall-like number on it moves in steps of that size.
    * `single_arm_half_width_points` -- the worst-case binomial standard error
      of ONE proportion on N items is ``0.5/sqrt(N)``; times the normal quantile
      for `metrics.bootstrap_ci` this is the half width of a single arm's own
      interval. It is NOT a threshold for a gap between two arms.
    * `two_arm_gap_floor_points` -- the same worst case for the DIFFERENCE of
      two independent proportions, ``sqrt(2)`` times larger. The arms are scored
      on the same Test images, so their errors are positively correlated and the
      true paired floor is smaller; treating them as independent is the
      conservative direction.

    Both width figures are first-order proxies only -- AP is not a proportion --
    and EVAL-09's bootstrap over Test images stays the instrument of record.
    """

    settings = _resolve(config)
    smallest = settings.smallest_bucket
    missing = [name for name in settings.primary_classes if name not in bucket_counts]
    if missing:
        raise SliceConfigError(
            f"metrics.primary_classes names {missing}, absent from the bucket counts "
            f"{sorted(bucket_counts)}. Silently dropping it would change the headline "
            "verdict with no error and no log line, so this raises instead."
        )
    primary = list(settings.primary_classes)

    n_small = sum(int(bucket_counts[name][smallest]) for name in primary)
    n_primary = sum(int(count) for name in primary for count in bucket_counts[name].values())
    share = (n_small / n_primary) if n_primary else 0.0
    even_share = 1.0 / len(settings.area_ranges)
    floor = settings.small_bucket_min_instances

    quantile = NormalDist().inv_cdf(0.5 + settings.bootstrap_ci / 2.0)
    per_instance = (100.0 / n_small) if n_small else float("inf")
    single_arm = (100.0 * quantile * 0.5 / math.sqrt(n_small)) if n_small else float("inf")

    below_floor = n_small < floor
    below_even_share = share < even_share
    return {
        "bucket": smallest,
        "primary_classes": tuple(primary),
        "n_small_primary": n_small,
        "n_primary_total": n_primary,
        "small_share": share,
        "even_share": even_share,
        "min_instances": floor,
        "is_below_floor": below_floor,
        "is_below_even_share": below_even_share,
        "is_thin": below_floor or below_even_share,
        "is_empty": n_small == 0,
        "per_instance_points": per_instance,
        "single_arm_half_width_points": single_arm,
        "two_arm_gap_floor_points": single_arm * math.sqrt(2.0),
        "confidence_level": settings.bootstrap_ci,
    }


def build_profile(
    samples: Sequence[Sample],
    *,
    luminance_by_image: Mapping[int, float],
    config: SliceConfig | None = None,
) -> dict[str, Any]:
    """Everything reports/test_set_profile.md needs, as plain JSON-able data."""

    settings = _resolve(config)
    bucket_counts = bucket_instance_counts(samples, config=settings)
    slices = scenario_slices(samples, luminance_by_image=luminance_by_image, config=settings)
    image_ids = [int(sample.image_id) for sample in samples]

    bucket_totals = {
        bucket: sum(bucket_counts[name][bucket] for name in CLASS_NAMES)
        for bucket in settings.bucket_names
    }
    return {
        "n_images": len(samples),
        "n_instances": sum(len(sample.boxes_xywh) for sample in samples),
        "instances_per_class": class_instance_counts(samples),
        "images_per_class": class_image_counts(samples),
        "bucket_counts": bucket_counts,
        "bucket_totals": bucket_totals,
        "annotation_space": annotation_space_sizes(samples),
        "slice_counts": {name: len(members) for name, members in slices.items()},
        "slice_pairwise_overlap": {
            f"{left} & {right}": count
            for (left, right), count in pairwise_overlap(slices).items()
        },
        "membership_histogram": membership_histogram(slices, image_ids),
        "verdict": small_bucket_verdict(bucket_counts, config=settings),
        "thresholds": {
            "coco_area_ranges": {
                name: [low, high] for name, low, high in settings.area_ranges
            },
            "slice_small_object_min_count": settings.small_object_min_count,
            "slice_crowded_min_instances": settings.crowded_min_instances,
            "slice_low_light_percentile": settings.low_light_percentile,
            "primary_classes": list(settings.primary_classes),
            "bootstrap_ci": settings.bootstrap_ci,
            "small_bucket_min_instances": settings.small_bucket_min_instances,
        },
    }


def _percent(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def _verdict_paragraphs(profile: Mapping[str, Any]) -> list[str]:
    """Plainly worded, generated from the numbers. No softening."""

    verdict = profile["verdict"]
    classes = " + ".join(f"`{name}`" for name in verdict["primary_classes"])
    bucket = verdict["bucket"]
    n_small = verdict["n_small_primary"]
    n_total = verdict["n_primary_total"]

    if verdict["is_empty"]:
        return [
            (
                f"**AP_{bucket} cannot carry the headline claim: the {bucket} bucket is "
                f"empty.** The frozen Test split contains 0 {classes} instances in the "
                f"{bucket} bucket out of {n_total}. AP_{bucket} is undefined here and any "
                "value COCOeval prints for it (-1, or 0 after our own coercion) is an "
                "artefact, not a measurement. The headline must move to another metric."
            ),
        ]

    share = _percent(n_small, n_total)
    even = _percent(verdict["even_share"], 1.0)
    step = verdict["per_instance_points"]
    single = verdict["single_arm_half_width_points"]
    gap = verdict["two_arm_gap_floor_points"]
    level = f"{100.0 * verdict['confidence_level']:.0f}%"
    floor = verdict["min_instances"]

    lead = (
        f"The frozen Test split holds **{n_small:,} {classes} instances in the {bucket} "
        f"bucket**, {share} of the {n_total:,} primary-class instances "
        f"(an even share across the {len(profile['bucket_totals'])} buckets would be {even})."
    )
    implication = (
        f"What that implies: one instance is {step:.3f} percentage points of anything "
        f"recall-like on this bucket. The worst-case binomial half width for a **single "
        f"arm** at the configured {level} level is **±{single:.2f} points**; for the "
        f"**difference between two arms** — which is what every claim in this project "
        f"is actually about — the same worst case is **±{gap:.2f} points**, larger by a "
        "factor of √2. Neither is the instrument of record. Both assume a proportion, "
        "and AP is not one; both treat the two arms as independent when in fact they "
        "are scored on these same images, which makes the between-arm figure "
        "conservative. Use them only to recognise a gap that is obviously too small to "
        "discuss (EVAL-10) — the bootstrap over Test images of EVAL-09 is what decides."
    )

    if verdict["is_thin"]:
        failures = []
        if verdict["is_below_floor"]:
            failures.append(f"it holds fewer than the required {floor:,} instances")
        if verdict["is_below_even_share"]:
            failures.append(f"it is below an even share ({even}) of the primary-class instances")
        call = (
            f"**Verdict: thin.** The {bucket} bucket fails the EVAL-08 test because "
            + " and because ".join(failures)
            + f". AP_{bucket} therefore rests on the smallest part of the Test split "
            "while carrying the largest part of the story. Report it with the instance "
            "count beside it every single time, and do not let a sub-noise difference "
            "become a claim."
        )
    else:
        call = (
            f"**Verdict: adequate.** The {bucket} bucket clears both EVAL-08 tests: at "
            f"least {floor:,} instances in absolute terms, and at least an even share "
            f"({even}) of the primary-class instances. AP_{bucket} is measuring a real "
            "population rather than a handful of boxes, so it can carry the headline "
            f"claim — subject to the ±{gap:.2f} point between-arm floor above and to "
            "EVAL-10."
        )

    caveat = (
        "Two standing caveats apply regardless of the count. Roughly 2/3 of the real "
        "objects in this dataset are unannotated (ADR-003), so every claim must be "
        "relative — arm A versus arm B on this same frozen Test — and never an absolute "
        "AP. And these areas are computed in the original annotation coordinates listed "
        "in section 1; recomputing them at the 640 training resolution would multiply "
        "every area by about 2.37 and move most of this bucket into `medium` without "
        "raising a single error (EVAL-07)."
    )
    return [lead, implication, call, caveat]


def render_profile_markdown(profile: Mapping[str, Any]) -> str:
    """reports/test_set_profile.md. Regenerate, never hand-edit."""

    thresholds = profile["thresholds"]
    bucket_names = list(thresholds["coco_area_ranges"])
    n_images = profile["n_images"]

    lines = [
        "# Frozen Test Set Profile",
        "",
        (
            "Generated by `scripts/profile_test_set.py` from `splits/split_manifest.json` "
            "and the interim COCO. No model is involved, and no prediction is read."
        ),
        "",
        (
            "Implements **EVAL-08** (per-size-bucket instance counts) and **EVAL-17** "
            "(scenario slices). All thresholds come from `configs/evaluation.yaml`."
        ),
        "",
        "## 1. Size",
        "",
        f"- Test images: **{n_images:,}**",
        f"- Annotated instances: **{profile['n_instances']:,}**",
        "- Annotation coordinate space: "
        + ", ".join(
            f"`{size}` ({count:,} images)"
            for size, count in profile["annotation_space"].items()
        )
        + " — areas below are `w * h` in this space (EVAL-07).",
    ]
    if len(profile["annotation_space"]) > 1:
        lines.append(
            "- **This split is not one single resolution.** Predictions must be mapped "
            "back to each image's own size, never with one global scale factor; "
            "`Sample` carries per-image width and height for exactly this reason."
        )
    lines += [
        "",
        "| Class | Instances | Images | Instances / image |",
        "|---|---:|---:|---:|",
    ]
    for name, instances in profile["instances_per_class"].items():
        images = profile["images_per_class"][name]
        per_image = f"{instances / images:.2f}" if images else "n/a"
        lines.append(f"| `{name}` | {instances:,} | {images:,} | {per_image} |")

    ranges = thresholds["coco_area_ranges"]
    lines.extend(
        [
            "",
            "## 2. Size buckets (EVAL-08)",
            "",
            "Half-open `[low, high)` in 416-space px²: "
            + ", ".join(f"`{name}` [{low:g}, {high:g})" for name, (low, high) in ranges.items())
            + ". An area of exactly a boundary value belongs to the upper bucket.",
            "",
            "| Class | " + " | ".join(f"{name} | {name} %" for name in bucket_names) + " | Total |",
            "|---|" + "---:|" * (2 * len(bucket_names) + 1),
        ]
    )
    for class_name, counts in profile["bucket_counts"].items():
        total = sum(counts.values())
        cells = []
        for bucket in bucket_names:
            cells.append(f"{counts[bucket]:,}")
            cells.append(_percent(counts[bucket], total))
        lines.append(f"| `{class_name}` | " + " | ".join(cells) + f" | {total:,} |")

    totals = profile["bucket_totals"]
    grand_total = sum(totals.values())
    total_cells = []
    for bucket in bucket_names:
        total_cells.append(f"{totals[bucket]:,}")
        total_cells.append(_percent(totals[bucket], grand_total))
    lines.append("| **all classes** | " + " | ".join(total_cells) + f" | {grand_total:,} |")

    lines.extend(
        [
            "",
            "## 3. Scenario slices (EVAL-17)",
            "",
            (
                "Slices are **not** mutually exclusive; one image can be in all three. "
                "Rules, all config-driven:"
            ),
            "",
            (
                f"- `small_object` — at least "
                f"{thresholds['slice_small_object_min_count']} box(es) in the "
                f"`{bucket_names[0]}` bucket"
            ),
            (
                f"- `crowded` — at least {thresholds['slice_crowded_min_instances']} annotated "
                + " + ".join(f"`{name}`" for name in thresholds["primary_classes"])
                + " instances. **`person` is deliberately not counted.** CLAUDE.md "
                "forbids that class carrying load in any decision and ADR-003 records "
                "its annotations as the least complete in the dataset, so counting it "
                "would admit images to this slice on the strength of the labels we "
                "trust least."
            ),
            (
                f"- `low_light` — the darkest "
                f"{thresholds['slice_low_light_percentile']:g}% of images by mean BT.601 "
                "luma (rank based, so the count is exactly `floor(n * pct / 100)`)"
            ),
            "",
            "| Slice | Images | % of Test |",
            "|---|---:|---:|",
        ]
    )
    for name, count in profile["slice_counts"].items():
        lines.append(f"| `{name}` | {count:,} | {_percent(count, n_images)} |")

    lines.extend(
        [
            "",
            "### Overlap",
            "",
            "| Pair | Images in both | % of Test |",
            "|---|---:|---:|",
        ]
    )
    for pair, count in profile["slice_pairwise_overlap"].items():
        lines.append(f"| `{pair}` | {count:,} | {_percent(count, n_images)} |")

    lines.extend(
        [
            "",
            "| Slices an image belongs to | Images | % of Test |",
            "|---:|---:|---:|",
        ]
    )
    for member_count, images in sorted(profile["membership_histogram"].items()):
        lines.append(f"| {member_count} | {images:,} | {_percent(images, n_images)} |")

    lines.extend(["", "## 4. Verdict on AP_small (EVAL-08)", ""])
    for paragraph in _verdict_paragraphs(profile):
        lines.extend([paragraph, ""])
    return "\n".join(lines)
