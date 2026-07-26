"""Deterministic binary-mask cleanup, metrics, and COCO RLE serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as coco_mask


def box_mask(shape: tuple[int, int], box_xyxy: list[float] | tuple[float, ...]) -> np.ndarray:
    """Return a boolean mask for a half-open XYXY box clipped to the image."""

    height, width = shape
    x1, y1, x2, y2 = box_xyxy
    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(left, min(width, int(np.ceil(x2))))
    bottom = max(top, min(height, int(np.ceil(y2))))
    result = np.zeros((height, width), dtype=bool)
    result[top:bottom, left:right] = True
    return result


def component_metrics(mask: np.ndarray) -> tuple[int, float]:
    """Return non-empty component count and second/largest area ratio."""

    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = sorted((int(area) for area in stats[1:, cv2.CC_STAT_AREA]), reverse=True)
    if not areas:
        return 0, 0.0
    return len(areas), (areas[1] / areas[0] if len(areas) > 1 else 0.0)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected foreground component."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill foreground holes without changing the exterior component."""

    binary = mask.astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    exterior = flood[1:-1, 1:-1].astype(bool)
    return mask | ~exterior


def mask_solidity(mask: np.ndarray) -> float:
    """Return foreground area divided by the convex-hull area."""

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    points = [contour for contour in contours if len(contour) >= 3]
    if not points:
        return 0.0
    all_points = np.concatenate(points, axis=0)
    hull = cv2.convexHull(all_points)
    hull_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    hull_area = int(hull_mask.sum())
    return min(float(mask.sum()) / hull_area, 1.0) if hull_area > 0 else 0.0


def _edge_touch_fractions(
    mask: np.ndarray, box_xyxy: list[float] | tuple[float, ...]
) -> tuple[float, float]:
    height, width = mask.shape
    x1, y1, x2, y2 = box_xyxy
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    top_fraction = float(mask[top, left:right].mean())
    left_fraction = float(mask[top:bottom, left].mean())
    right_fraction = float(mask[top:bottom, right - 1].mean())
    return top_fraction, max(left_fraction, right_fraction)


def clean_and_measure_mask(
    raw_mask: np.ndarray,
    box_xyxy: list[float] | tuple[float, ...],
    *,
    morph_close_kernel: int = 3,
    morph_close_iterations: int = 1,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Clean a SAM mask and compute QC signals, including pre-clip leakage."""

    raw = np.asarray(raw_mask, dtype=bool)
    prompt = box_mask(raw.shape, box_xyxy)
    raw_area = int(raw.sum())
    outside_ratio = float((raw & ~prompt).sum()) / raw_area if raw_area else 1.0
    component_count, second_component_ratio = component_metrics(raw)

    largest = keep_largest_component(raw)
    before_holes = int(largest.sum())
    before_clip = fill_holes(largest) if before_holes else largest
    if morph_close_kernel > 1 and before_clip.any():
        kernel = np.ones((morph_close_kernel, morph_close_kernel), dtype=np.uint8)
        before_clip = cv2.morphologyEx(
            before_clip.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel,
            iterations=morph_close_iterations,
        ).astype(bool)
    cleaned = before_clip & prompt
    cleaned = keep_largest_component(cleaned)
    cleaned = fill_holes(cleaned)

    area = int(cleaned.sum())
    x1, y1, x2, y2 = box_xyxy
    box_area = max(float(x2 - x1) * float(y2 - y1), 1.0)
    edge_top, edge_side = _edge_touch_fractions(cleaned, box_xyxy)
    metrics: dict[str, float | int] = {
        "mask_area_px": area,
        "mask_to_box_coverage": area / box_area,
        "outside_box_ratio": outside_ratio,
        "component_count": component_count,
        "second_component_ratio": second_component_ratio,
        "hole_fill_ratio": area / max(before_holes, 1),
        "solidity": mask_solidity(cleaned),
        "edge_touch_top": edge_top,
        "edge_touch_side": edge_side,
    }
    return cleaned, metrics


def quality_failures(
    metrics: Mapping[str, float | int],
    *,
    class_name: str,
    config: Mapping[str, Any],
    min_iou_score: float | None = None,
    min_object_score_logit: float | None = None,
) -> list[str]:
    """Evaluate the configured SAM mask-quality gates."""

    quality = config["cutout_bank"]["mask_quality"]
    failures: list[str] = []
    scalar_maxima = {
        "outside_box_ratio": "max_outside_box_ratio",
        "second_component_ratio": "max_second_component_ratio",
        "hole_fill_ratio": "max_hole_fill_ratio",
    }
    iou_threshold = (
        float(quality["min_iou_score"]) if min_iou_score is None else min_iou_score
    )
    object_threshold = (
        float(quality["min_object_score_logit"])
        if min_object_score_logit is None
        else min_object_score_logit
    )
    if float(metrics["iou_score"]) < iou_threshold:
        failures.append("iou_score")
    if float(metrics["object_score_logit"]) < object_threshold:
        failures.append("object_score_logit")
    for metric, setting in scalar_maxima.items():
        if float(metrics[metric]) > float(quality[setting]):
            failures.append(metric)
    low, high = quality["mask_to_box_coverage"][class_name]
    coverage = float(metrics["mask_to_box_coverage"])
    if not float(low) <= coverage <= float(high):
        failures.append("mask_to_box_coverage")
    if float(metrics["solidity"]) < float(quality["min_solidity"][class_name]):
        failures.append("solidity")
    if int(metrics["mask_area_px"]) < int(quality["min_mask_area_px"]):
        failures.append("mask_area_px")
    if (
        class_name in quality.get("max_edge_touch_top", {})
        and float(metrics["edge_touch_top"]) > float(quality["max_edge_touch_top"][class_name])
    ):
        failures.append("edge_touch_top")
    if (
        class_name in quality.get("max_edge_touch_side", {})
        and float(metrics["edge_touch_side"]) > float(quality["max_edge_touch_side"][class_name])
    ):
        failures.append("edge_touch_side")
    return failures


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a boolean mask as JSON-safe compressed COCO RLE."""

    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": encoded["counts"].decode("ascii"),
    }


def decode_rle(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode JSON-safe compressed COCO RLE to boolean H×W."""

    native = {"size": list(rle["size"]), "counts": str(rle["counts"]).encode("ascii")}
    return coco_mask.decode(native).astype(bool)
