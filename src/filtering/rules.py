"""FILT-01..FILT-12 predicates and crash-on-bug invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RejectReason(StrEnum):
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    BOX_TOO_SMALL = "BOX_TOO_SMALL"
    LOW_VISIBLE_FRACTION = "LOW_VISIBLE_FRACTION"
    EXCESSIVE_OVERLAP = "EXCESSIVE_OVERLAP"
    BAD_MASK_COVERAGE = "BAD_MASK_COVERAGE"
    FLOATING_HELMET = "FLOATING_HELMET"
    HELMET_HEAD_MISALIGNED = "HELMET_HEAD_MISALIGNED"
    HELMET_SWALLOWS_HEAD = "HELMET_SWALLOWS_HEAD"
    BAD_SIZE_RATIO = "BAD_SIZE_RATIO"
    CLIPPING_ARTIFACT = "CLIPPING_ARTIFACT"
    SEAM_ARTIFACT = "SEAM_ARTIFACT"
    HARD_NEGATIVE_OVERLAPS_ANNOTATION = "HARD_NEGATIVE_OVERLAPS_ANNOTATION"
    NEAR_DUPLICATE_SYNTHETIC = "NEAR_DUPLICATE_SYNTHETIC"
    NEAR_DUPLICATE_REAL = "NEAR_DUPLICATE_REAL"
    NO_CHANGE = "NO_CHANGE"
    SAM2_MASK_REJECTED = "SAM2_MASK_REJECTED"
    PLACEMENT_RETRIES_EXHAUSTED = "PLACEMENT_RETRIES_EXHAUSTED"


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    reject_reasons: tuple[str, ...]
    first_reason: str | None


def _area(box: Sequence[float]) -> float:
    return max(float(box[2]), 0) * max(float(box[3]), 0)


def _xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x, y, x + width, y + height


def _intersection(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(first)
    bx1, by1, bx2, by2 = _xyxy(second)
    return max(min(ax2, bx2) - max(ax1, bx1), 0) * max(
        min(ay2, by2) - max(ay1, by1), 0
    )


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = _intersection(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _iomin(first: Sequence[float], second: Sequence[float]) -> float:
    return _intersection(first, second) / max(min(_area(first), _area(second)), 1)


def _inside_ratio(box: Sequence[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = _xyxy(box)
    clipped_width = max(min(x2, width) - max(x1, 0), 0)
    clipped_height = max(min(y2, height) - max(y1, 0), 0)
    return clipped_width * clipped_height / max(_area(box), 1)


def _kept_annotations(sample: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        instance
        for instance in sample.get("instances", [])
        if instance.get("kept", True)
        and instance.get("kind", "pasted") != "hard_negative"
    ]


def assert_composition_invariants(
    sample: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """FILT-12: violations are generator bugs, never filter rejections."""

    assertions = config["assertions"]
    invariants = sample.get("invariants", {})
    if assertions["real_annotation_count_invariant"]:
        expected = int(invariants.get("n_real_ann_in", 0)) - len(
            invariants.get("intentional_removals", [])
        )
        actual = int(invariants.get("n_real_ann_out", expected))
        if actual != expected:
            raise AssertionError(
                f"real annotation count invariant failed: actual={actual}, expected={expected}"
            )
    kept = _kept_annotations(sample)
    if assertions["no_zero_area_annotation"]:
        zero = [instance.get("instance_id") for instance in kept if _area(instance["bbox_xywh"]) <= 0]
        if zero:
            raise AssertionError(f"zero-area annotations: {zero}")
    if assertions["z_order_matches_y_bottom"] and kept:
        actual_order = [
            instance.get("instance_id")
            for instance in sorted(kept, key=lambda item: int(item.get("z_index", 0)))
        ]
        expected_order = [
            instance.get("instance_id")
            for instance in sorted(
                kept,
                key=lambda item: (
                    float(item["bbox_xywh"][1]) + float(item["bbox_xywh"][3]),
                    str(item.get("instance_id")),
                ),
            )
        ]
        if actual_order != expected_order:
            raise AssertionError(
                f"z-order mismatch: actual={actual_order}, expected={expected_order}"
            )
    if assertions["test_blocklist_untouched"] and not invariants.get(
        "test_blocklist_untouched", True
    ):
        raise AssertionError("frozen Test blocklist was touched")


def _filt_01_02_03(
    sample: Mapping[str, Any], config: Mapping[str, Any]
) -> list[RejectReason]:
    rules = config["rules"]
    width = int(sample["width"])
    height = int(sample["height"])
    reasons: list[RejectReason] = []
    for instance in _kept_annotations(sample):
        box = instance["bbox_xywh"]
        preclip = instance.get("bbox_xywh_preclip", box)
        x, y, box_width, box_height = (float(value) for value in box)
        min_side = float(rules["bbox_in_bounds"]["min_box_side_px"])
        if (
            x < 0
            or y < 0
            or x + box_width > width
            or y + box_height > height
            or _inside_ratio(preclip, width, height)
            < float(rules["bbox_in_bounds"]["min_inside_ratio"])
        ):
            reasons.append(RejectReason.OUT_OF_BOUNDS)
        if (
            min(box_width, box_height) < min_side
            or _area(box) < float(rules["min_visible_area"]["min_area_px"])
        ):
            reasons.append(RejectReason.BOX_TOO_SMALL)
        minimum_visible = float(
            rules["visible_fraction"][str(instance["class_name"])]
        )
        if float(instance.get("visible_fraction", 1.0)) < minimum_visible:
            reasons.append(RejectReason.LOW_VISIBLE_FRACTION)
    return reasons


def _filt_04(instances: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bool:
    overlap = config["rules"]["overlap"]
    max_iomin = float(overlap["max_overlap_score_same_class"])
    max_iou = float(overlap["max_overlap_iou_same_class"])
    for index, first in enumerate(instances):
        for second in instances[index + 1 :]:
            if first["class_name"] != second["class_name"]:
                continue
            if (
                _iomin(first["bbox_xywh"], second["bbox_xywh"]) > max_iomin
                or _iou(first["bbox_xywh"], second["bbox_xywh"]) > max_iou
            ):
                return True
    return False


def _filt_05_06(
    instances: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[RejectReason]:
    coverage_config = config["rules"]["mask_to_box_coverage"]
    reasons: list[RejectReason] = []
    for instance in instances:
        if "mask_to_box_coverage" in instance:
            low, high = coverage_config[str(instance["class_name"])]
            coverage = float(instance["mask_to_box_coverage"])
            if not float(low) <= coverage <= float(high):
                reasons.append(RejectReason.BAD_MASK_COVERAGE)
        if instance.get("kind") == "pasted" and not instance.get("sam2_qc_pass", True):
            reasons.append(RejectReason.SAM2_MASK_REJECTED)
        if instance.get("placement_retries_exhausted", False):
            reasons.append(RejectReason.PLACEMENT_RETRIES_EXHAUSTED)
    return reasons


def _filt_07(sample: Mapping[str, Any], config: Mapping[str, Any]) -> list[RejectReason]:
    settings = config["rules"]["helmet_above_head"]
    reasons: list[RejectReason] = []
    for pair in sample.get("pairs", []):
        overlap_y = float(pair["overlap_y"])
        pair_iou = float(pair["iou"])
        if (
            overlap_y < float(settings["overlap_y_min"])
            or pair_iou < float(settings["min_iou_helmet_head"])
        ):
            reasons.append(RejectReason.FLOATING_HELMET)
            continue
        if overlap_y > float(settings["overlap_y_max"]):
            reasons.append(RejectReason.HELMET_SWALLOWS_HEAD)
            continue
        if (
            abs(float(pair["dx"])) > float(settings["k_x"])
            or not float(settings["dy_min"]) <= float(pair["dy"]) <= float(settings["dy_max"])
            or not float(settings["r_w"][0])
            <= float(pair["r_w"])
            <= float(settings["r_w"][1])
            or not float(settings["r_h"][0])
            <= float(pair["r_h"])
            <= float(settings["r_h"][1])
        ):
            reasons.append(RejectReason.HELMET_HEAD_MISALIGNED)
    return reasons


def _filt_08(
    instances: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> bool:
    settings = config["rules"]["size_ratio"]
    containment_threshold = float(settings["containment_threshold"])
    persons = [instance for instance in instances if instance["class_name"] == "person"]
    for child in instances:
        class_name = str(child["class_name"])
        if class_name not in {"head", "helmet"}:
            continue
        child_box = child["bbox_xywh"]
        child_area = _area(child_box)
        for person in persons:
            person_box = person["bbox_xywh"]
            containment = _intersection(child_box, person_box) / max(child_area, 1)
            if containment < containment_threshold:
                continue
            area_ratio = child_area / max(_area(person_box), 1)
            key = f"{class_name}_over_person_area"
            low, high = settings[key]
            if not float(low) <= area_ratio <= float(high):
                return True
            if class_name == "head":
                width_ratio = float(child_box[2]) / max(float(person_box[2]), 1)
                width_low, width_high = settings["head_over_person_width"]
                top_position = (
                    float(child_box[1]) - float(person_box[1])
                ) / max(float(person_box[3]), 1)
                if (
                    not float(width_low) <= width_ratio <= float(width_high)
                    or top_position > float(settings["head_top_within_person"])
                ):
                    return True
            break
    return False


def _filt_09_10(
    instances: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[RejectReason]:
    clipping = config["rules"]["clipping_artifact"]
    poke_low, poke_high = (float(value) for value in clipping["poke_through_range"])
    reasons: list[RejectReason] = []
    for instance in instances:
        poke = float(instance.get("poke_through_fraction", 0))
        if instance.get("is_behind_other", False) and poke_low <= poke <= poke_high:
            reasons.append(RejectReason.CLIPPING_ARTIFACT)
        if float(instance.get("seam_energy_ratio", 0)) > float(
            clipping["max_seam_energy_ratio"]
        ):
            reasons.append(RejectReason.SEAM_ARTIFACT)
    hard_limit = float(
        config["rules"]["hard_negative_no_overlap"]["max_iou_with_annotation"]
    )
    for instance in instances:
        if instance.get("kind") == "hard_negative" and float(
            instance.get("max_iou_with_annotation", 0)
        ) > hard_limit:
            reasons.append(RejectReason.HARD_NEGATIVE_OVERLAPS_ANNOTATION)
    return reasons


def _filt_11(sample: Mapping[str, Any], config: Mapping[str, Any]) -> list[RejectReason]:
    settings = config["rules"]["phash_dedup"]
    dedup = sample.get("dedup", {})
    reasons: list[RejectReason] = []
    if float(dedup.get("changed_pixel_ratio", 1.0)) < float(
        settings["min_changed_pixel_ratio"]
    ):
        reasons.append(RejectReason.NO_CHANGE)
    if int(dedup.get("min_hamming_to_accepted_synthetic", 10**9)) <= int(
        settings["max_hamming_to_accepted_synthetic"]
    ):
        reasons.append(RejectReason.NEAR_DUPLICATE_SYNTHETIC)
    if int(dedup.get("min_hamming_to_any_real_image", 10**9)) <= int(
        settings["max_hamming_to_any_real_image"]
    ):
        reasons.append(RejectReason.NEAR_DUPLICATE_REAL)
    return reasons


def filter_sample(
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    check_invariants: bool = True,
) -> FilterResult:
    """Evaluate every filter and return stable, enum-validated reasons."""

    if check_invariants:
        assert_composition_invariants(sample, config)
    instances = _kept_annotations(sample)
    raw_reasons: list[RejectReason] = []
    raw_reasons.extend(_filt_01_02_03(sample, config))
    if _filt_04(instances, config):
        raw_reasons.append(RejectReason.EXCESSIVE_OVERLAP)
    raw_reasons.extend(_filt_05_06(instances, config))
    raw_reasons.extend(_filt_07(sample, config))
    if _filt_08(instances, config):
        raw_reasons.append(RejectReason.BAD_SIZE_RATIO)
    # FILT-09/10 also needs hard negatives, which carry no annotation.
    raw_reasons.extend(_filt_09_10(sample.get("instances", []), config))
    raw_reasons.extend(_filt_11(sample, config))
    reasons = tuple(dict.fromkeys(reason.value for reason in raw_reasons))

    configured = set(config["reject_reasons"])
    unknown = set(reasons) - configured
    if unknown:
        raise AssertionError(f"Filter emitted reasons outside config enum: {sorted(unknown)}")
    return FilterResult(
        passed=not reasons,
        reject_reasons=reasons,
        first_reason=(reasons[0] if reasons else None),
    )
