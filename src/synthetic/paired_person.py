"""Geometry and eligibility rules for anatomically coupled person units."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PairedPersonUnit:
    """One Train person cutout and its single coupled helmet/head label."""

    person: dict[str, Any]
    headlike: dict[str, Any]
    headlike_class: str


def paired_headlike_annotations(
    person_bbox_xywh: Sequence[float],
    *,
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    upper_fraction: float = 0.55,
) -> list[dict[str, Any]]:
    """Return headlike boxes centred inside the upper portion of a person box."""

    x, y, width, height = (float(value) for value in person_bbox_xywh)
    paired: list[dict[str, Any]] = []
    for source in annotations:
        if categories[int(source["category_id"])] not in {"helmet", "head"}:
            continue
        box_x, box_y, box_width, box_height = (
            float(value) for value in source["bbox"]
        )
        center_x = box_x + box_width / 2
        center_y = box_y + box_height / 2
        if (
            x <= center_x <= x + width
            and y <= center_y <= y + upper_fraction * height
        ):
            paired.append(dict(source))
    paired.sort(key=lambda item: int(item["id"]))
    return paired


def strict_unit_reject_reasons(
    person: Mapping[str, Any],
    headlike: Mapping[str, Any],
    *,
    preferred_tier_required: bool,
    min_person_height_px: float,
    min_person_aspect_height_over_width: float,
    max_head_center_y_fraction: float,
    max_head_width_fraction: float,
    max_edge_touch_top: float,
    max_edge_touch_side: float,
    min_source_person_bottom_fraction: float,
    source_image_height: float,
) -> tuple[str, ...]:
    """Return deterministic v6 donor failures without reading any image pixels."""

    reasons: list[str] = []
    person_x, person_y, person_width, person_height = (
        float(value) for value in person["src_bbox_xywh"]
    )
    head_x, head_y, head_width, head_height = (
        float(value) for value in headlike["bbox"]
    )
    del person_x
    sam2 = person["sam2"]
    voc_flags = person["voc_flags"]
    if str(person["class_name"]) != "person":
        reasons.append("NOT_PERSON")
    if str(person["src_split"]) != "train":
        reasons.append("NOT_TRAIN")
    if preferred_tier_required and not bool(person["preferred_tier"]):
        reasons.append("NOT_PREFERRED_TIER")
    if int(voc_flags["truncated"]) or int(voc_flags["difficult"]):
        reasons.append("VOC_FLAGGED")
    if person_height < min_person_height_px:
        reasons.append("PERSON_TOO_SMALL")
    if (
        person_y + person_height
    ) / max(source_image_height, 1.0) < min_source_person_bottom_fraction:
        reasons.append("SOURCE_PERSON_TOO_HIGH")
    if person_height / max(person_width, 1.0) < min_person_aspect_height_over_width:
        reasons.append("PERSON_TOO_HORIZONTAL")
    head_center_y_fraction = (
        head_y + head_height / 2 - person_y
    ) / max(person_height, 1.0)
    if head_center_y_fraction > max_head_center_y_fraction:
        reasons.append("HEAD_TOO_LOW")
    if head_width / max(person_width, 1.0) > max_head_width_fraction:
        reasons.append("HEAD_TOO_WIDE")
    if head_x + head_width <= 0:
        reasons.append("INVALID_HEAD_BOX")
    if float(sam2["edge_touch_top"]) > max_edge_touch_top:
        reasons.append("MASK_TOUCHES_TOP")
    if float(sam2["edge_touch_side"]) > max_edge_touch_side:
        reasons.append("MASK_TOUCHES_SIDE")
    return tuple(reasons)


def transform_linked_box(
    source_box_xywh: Sequence[float],
    *,
    source_person_box_xywh: Sequence[float],
    scale: float,
    hflip: bool,
    patch_width: int,
    patch_left: int,
    patch_top: int,
) -> list[float]:
    """Transform a linked label with the exact scale/flip used on its person."""

    source_x, source_y, source_width, source_height = (
        float(value) for value in source_box_xywh
    )
    person_x, person_y, _, _ = (
        float(value) for value in source_person_box_xywh
    )
    relative_x = (source_x - person_x) * scale
    relative_y = (source_y - person_y) * scale
    output_width = source_width * scale
    output_height = source_height * scale
    if hflip:
        relative_x = float(patch_width) - (relative_x + output_width)
    return [
        float(patch_left) + relative_x,
        float(patch_top) + relative_y,
        output_width,
        output_height,
    ]


def intersection_over_smaller_box(
    first_xywh: Sequence[float],
    second_xywh: Sequence[float],
) -> float:
    """Return intersection area divided by the smaller input-box area."""

    first_x, first_y, first_width, first_height = (
        float(value) for value in first_xywh
    )
    second_x, second_y, second_width, second_height = (
        float(value) for value in second_xywh
    )
    intersection_width = max(
        0.0,
        min(first_x + first_width, second_x + second_width)
        - max(first_x, second_x),
    )
    intersection_height = max(
        0.0,
        min(first_y + first_height, second_y + second_height)
        - max(first_y, second_y),
    )
    intersection = intersection_width * intersection_height
    smaller = min(
        max(first_width * first_height, 0.0),
        max(second_width * second_height, 0.0),
    )
    return intersection / smaller if smaller > 0 else 0.0
