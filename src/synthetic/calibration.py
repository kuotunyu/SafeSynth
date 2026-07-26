"""M6/H7 empirical calibration from the frozen Train split and Pass 1 masks."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.data.paths import ProjectPaths
from src.synthetic.mask_ops import decode_rle
from src.synthetic.survey import load_json, train_context

PERCENTILES = (1, 5, 50, 95, 99)


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    """Return H7's required quantiles and population size."""

    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"n": 0, **{f"p{percentile}": math.nan for percentile in PERCENTILES}}
    return {
        "n": int(finite.size),
        **{
            f"p{percentile}": float(np.percentile(finite, percentile))
            for percentile in PERCENTILES
        },
    }


def _intersection(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def _area(box: Sequence[float]) -> float:
    return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)


def _xyxy(annotation: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in annotation["bbox"])
    return x, y, x + width, y + height


def _pairwise_geometry(
    annotations: Sequence[Mapping[str, Any]], category_names: Mapping[int, str]
) -> tuple[list[float], list[float]]:
    max_iomin: list[float] = []
    max_iou: list[float] = []
    by_class: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in annotations:
        by_class[category_names[int(annotation["category_id"])]].append(_xyxy(annotation))
    for boxes in by_class.values():
        for index, first in enumerate(boxes):
            first_area = _area(first)
            best_iomin = best_iou = 0.0
            for other_index, second in enumerate(boxes):
                if index == other_index:
                    continue
                intersection = _intersection(first, second)
                second_area = _area(second)
                best_iomin = max(best_iomin, intersection / max(min(first_area, second_area), 1))
                best_iou = max(
                    best_iou, intersection / max(first_area + second_area - intersection, 1)
                )
            max_iomin.append(best_iomin)
            max_iou.append(best_iou)
    return max_iomin, max_iou


def geometry_distributions(paths: ProjectPaths) -> dict[str, Any]:
    """Measure box, overlap, and child-within-person geometry on frozen Train."""

    _, _, annotations_by_image, category_names, _ = train_context(paths)
    per_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overlaps_iomin: list[float] = []
    overlaps_iou: list[float] = []
    containment_threshold = 0.70
    contained: dict[str, dict[str, list[float]]] = {
        "helmet": defaultdict(list),
        "head": defaultdict(list),
    }

    for image_annotations in annotations_by_image.values():
        for annotation in image_annotations:
            class_name = category_names[int(annotation["category_id"])]
            _, _, width, height = (float(value) for value in annotation["bbox"])
            per_class[class_name]["bbox_area_px"].append(width * height)
            per_class[class_name]["min_side_px"].append(min(width, height))
            per_class[class_name]["aspect_ratio"].append(width / height)
        image_iomin, image_iou = _pairwise_geometry(image_annotations, category_names)
        overlaps_iomin.extend(image_iomin)
        overlaps_iou.extend(image_iou)

        persons = [
            annotation
            for annotation in image_annotations
            if category_names[int(annotation["category_id"])] == "person"
        ]
        for annotation in image_annotations:
            class_name = category_names[int(annotation["category_id"])]
            if class_name not in contained:
                continue
            child_box = _xyxy(annotation)
            child_area = _area(child_box)
            candidates: list[tuple[float, Mapping[str, Any]]] = []
            for person in persons:
                containment = _intersection(child_box, _xyxy(person)) / max(child_area, 1)
                if containment >= containment_threshold:
                    candidates.append((containment, person))
            if not candidates:
                continue
            _, person = max(candidates, key=lambda item: item[0])
            person_box = _xyxy(person)
            person_area = _area(person_box)
            person_width = person_box[2] - person_box[0]
            person_height = person_box[3] - person_box[1]
            contained[class_name]["over_person_area"].append(child_area / person_area)
            contained[class_name]["over_person_width"].append(
                (child_box[2] - child_box[0]) / person_width
            )
            contained[class_name]["top_within_person"].append(
                (child_box[1] - person_box[1]) / person_height
            )

    all_areas = [
        value for class_metrics in per_class.values() for value in class_metrics["bbox_area_px"]
    ]
    return {
        "per_class": {
            class_name: {
                metric: distribution(values) for metric, values in sorted(metrics.items())
            }
            for class_name, metrics in sorted(per_class.items())
        },
        "same_class_overlap": {
            "max_iomin": distribution(overlaps_iomin),
            "max_iou": distribution(overlaps_iou),
        },
        "contained_in_person": {
            class_name: {
                metric: distribution(values) for metric, values in sorted(metrics.items())
            }
            for class_name, metrics in sorted(contained.items())
        },
        "all_bbox_area_px": distribution(all_areas),
        "containment_threshold": containment_threshold,
    }


def seam_energy_ratio(
    image: np.ndarray, mask: np.ndarray, *, band_px: int = 2
) -> float | None:
    """Measure real-boundary gradient energy divided by eroded-mask interior energy."""

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    binary = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=band_px).astype(bool)
    band = mask & ~eroded
    interior = cv2.erode(binary, kernel, iterations=band_px + 1).astype(bool)
    if int(band.sum()) < 4 or int(interior.sum()) < 4:
        return None
    interior_energy = float(magnitude[interior].mean())
    return float(magnitude[band].mean()) / max(interior_energy, 1.0)


def mask_distributions(
    paths: ProjectPaths, *, min_iou: float, min_object_logit: float, seam_band_px: int = 2
) -> dict[str, Any]:
    """Measure masks after non-calibrated structural gates to avoid circular thresholds."""

    records_dir = paths.masks_pass1 / "records"
    metrics_by_class: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rejected_base = 0
    accepted_base = 0
    image_count = 0
    for record_path in sorted(records_dir.glob("*.json")):
        record = load_json(record_path)
        image = np.asarray(
            Image.open(paths.hardhat_raw / record["file_name"]).convert("RGB")
        )
        image_count += 1
        for annotation in record["annotations"]:
            metrics = annotation["metrics"]
            base_valid = (
                int(metrics["mask_area_px"]) >= 250
                and float(metrics["iou_score"]) >= min_iou
                and float(metrics["object_score_logit"]) >= min_object_logit
                and float(metrics["outside_box_ratio"]) <= 0.10
                and float(metrics["second_component_ratio"]) <= 0.20
                and float(metrics["hole_fill_ratio"]) <= 1.25
            )
            if not base_valid:
                rejected_base += 1
                continue
            accepted_base += 1
            class_name = str(annotation["class_name"])
            for metric_name in ("mask_to_box_coverage", "solidity"):
                metrics_by_class[class_name][metric_name].append(float(metrics[metric_name]))
            seam = seam_energy_ratio(
                image, decode_rle(annotation["segmentation"]), band_px=seam_band_px
            )
            if seam is not None:
                metrics_by_class[class_name]["seam_energy_ratio"].append(seam)

    return {
        "indexed_images": image_count,
        "base_valid": accepted_base,
        "base_rejected": rejected_base,
        "base_gate": {
            "min_iou_score": min_iou,
            "min_object_score_logit": min_object_logit,
            "min_mask_area_px": 250,
            "max_outside_box_ratio": 0.10,
            "max_second_component_ratio": 0.20,
            "max_hole_fill_ratio": 1.25,
        },
        "per_class": {
            class_name: {
                metric: distribution(values) for metric, values in sorted(metrics.items())
            }
            for class_name, metrics in sorted(metrics_by_class.items())
        },
    }


def calibrated_values(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Derive conservative config values from empirical quantiles."""

    geometry = calibration["geometry"]
    per_class = geometry["per_class"]
    preferred_min_side = min(
        float(per_class[class_name]["min_side_px"]["p50"]) for class_name in per_class
    )
    preferred_area = min(
        float(per_class[class_name]["bbox_area_px"]["p50"]) for class_name in per_class
    )
    values: dict[str, Any] = {
        "preferred_min_side_px": round(preferred_min_side),
        "preferred_min_area_px": round(preferred_area),
        "min_visible_area_px": math.floor(geometry["all_bbox_area_px"]["p1"]),
        "max_overlap_score_same_class": float(
            geometry["same_class_overlap"]["max_iomin"]["p99"]
        ),
        "max_overlap_iou_same_class": float(
            geometry["same_class_overlap"]["max_iou"]["p99"]
        ),
        "aspect_ratio": {
            class_name: [
                float(per_class[class_name]["aspect_ratio"]["p1"]),
                float(per_class[class_name]["aspect_ratio"]["p99"]),
            ]
            for class_name in per_class
        },
        "size_ratio": {
            "head_over_person_area": [
                float(geometry["contained_in_person"]["head"]["over_person_area"]["p1"]),
                float(geometry["contained_in_person"]["head"]["over_person_area"]["p99"]),
            ],
            "helmet_over_person_area": [
                float(geometry["contained_in_person"]["helmet"]["over_person_area"]["p1"]),
                float(geometry["contained_in_person"]["helmet"]["over_person_area"]["p99"]),
            ],
            "head_over_person_width": [
                float(geometry["contained_in_person"]["head"]["over_person_width"]["p1"]),
                float(geometry["contained_in_person"]["head"]["over_person_width"]["p99"]),
            ],
        },
    }
    if "masks" in calibration:
        mask_classes = calibration["masks"]["per_class"]
        values["mask_to_box_coverage"] = {
            class_name: [
                float(metrics["mask_to_box_coverage"]["p1"]),
                float(metrics["mask_to_box_coverage"]["p99"]),
            ]
            for class_name, metrics in mask_classes.items()
        }
        values["min_solidity"] = {
            class_name: float(metrics["solidity"]["p1"])
            for class_name, metrics in mask_classes.items()
        }
        all_seams = [
            float(metrics["seam_energy_ratio"]["p95"])
            for metrics in mask_classes.values()
            if metrics["seam_energy_ratio"]["n"]
        ]
        values["max_seam_energy_ratio"] = max(all_seams)
    return values


def remaining_guess_lines(project_root: Path) -> list[str]:
    """List every YAML config line that still declares a guessed parameter."""

    lines: list[str] = []
    for path in sorted((project_root / "configs").glob("*.yaml")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"source:\s*guess\b", line):
                lines.append(f"{path.relative_to(project_root).as_posix()}:{line_number}: {line.strip()}")
    return lines


def write_calibration_report(
    *,
    calibration: Mapping[str, Any],
    values: Mapping[str, Any],
    output_path: Path,
    guess_lines: Sequence[str],
) -> None:
    """Write a compact, auditable Markdown summary of H7."""

    geometry = calibration["geometry"]
    lines = [
        "# M6 / Spike H7 calibration",
        "",
        "Input: frozen Train split only. Quantiles are p1, p5, p50, p95, p99.",
        "",
        "## Box geometry",
        "",
        "| class | n | area p1 / p50 / p99 | min-side p1 / p50 / p99 | aspect p1 / p50 / p99 |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name, metrics in geometry["per_class"].items():
        area = metrics["bbox_area_px"]
        side = metrics["min_side_px"]
        aspect = metrics["aspect_ratio"]
        lines.append(
            f"| {class_name} | {area['n']} | {area['p1']:.2f} / {area['p50']:.2f} / "
            f"{area['p99']:.2f} | {side['p1']:.2f} / {side['p50']:.2f} / "
            f"{side['p99']:.2f} | {aspect['p1']:.4f} / {aspect['p50']:.4f} / "
            f"{aspect['p99']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Child contained in person",
            "",
            f"Containment threshold: {geometry['containment_threshold']:.2f}.",
            "",
            (
                "| class | contained n | area ratio p1 / p50 / p99 | "
                "width ratio p1 / p50 / p99 | top position p1 / p50 / p99 |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name, metrics in geometry["contained_in_person"].items():
        area = metrics["over_person_area"]
        width = metrics["over_person_width"]
        top = metrics["top_within_person"]
        lines.append(
            f"| {class_name} | {area['n']} | {area['p1']:.4f} / {area['p50']:.4f} / "
            f"{area['p99']:.4f} | {width['p1']:.4f} / {width['p50']:.4f} / "
            f"{width['p99']:.4f} | {top['p1']:.4f} / {top['p50']:.4f} / "
            f"{top['p99']:.4f} |"
        )
    overlap = geometry["same_class_overlap"]
    lines.extend(
        [
            "",
            "## Same-class overlap",
            "",
            f"- Per-instance maximum IoMin p99: {overlap['max_iomin']['p99']:.6f}",
            f"- Per-instance maximum IoU p99: {overlap['max_iou']['p99']:.6f}",
            "",
        ]
    )
    if "masks" in calibration:
        masks = calibration["masks"]
        lines.extend(
            [
                "## SAM2 mask distributions",
                "",
                (
                    f"Base-valid masks: {masks['base_valid']}; "
                    f"base-rejected: {masks['base_rejected']}; "
                    f"images: {masks['indexed_images']}. Calibrated coverage/solidity "
                    "were excluded from the base gate to avoid circular calibration."
                ),
                "",
                (
                    "| class | n | coverage p1 / p50 / p99 | "
                    "solidity p1 / p50 / p99 | "
                    "real-boundary seam p1 / p50 / p95 / p99 |"
                ),
                "|---|---:|---:|---:|---:|",
            ]
        )
        for class_name, metrics in masks["per_class"].items():
            coverage = metrics["mask_to_box_coverage"]
            solidity = metrics["solidity"]
            seam = metrics["seam_energy_ratio"]
            lines.append(
                f"| {class_name} | {coverage['n']} | {coverage['p1']:.4f} / "
                f"{coverage['p50']:.4f} / {coverage['p99']:.4f} | "
                f"{solidity['p1']:.4f} / {solidity['p50']:.4f} / "
                f"{solidity['p99']:.4f} | {seam['p1']:.4f} / {seam['p50']:.4f} / "
                f"{seam['p95']:.4f} / {seam['p99']:.4f} |"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "## SAM2 mask distributions",
                "",
                "Pending: Pass 1 is not complete, so no mask-dependent config is rewritten.",
                "",
            ]
        )
    lines.extend(
        [
            "## Derived calibrated values",
            "",
            "```json",
            json.dumps(values, indent=2, sort_keys=True),
            "```",
            "",
            (
                "The scalar preferred tier uses the minimum per-class p50 so scarce, "
                "smaller `head` sources are not excluded; it is a preference, "
                "not the H2 hard floor."
            ),
            "",
            f"## Remaining `source: guess` lines ({len(guess_lines)})",
            "",
            "These remain explicit priors and must be listed again in the M13 filter report.",
            "",
            "```text",
            *guess_lines,
            "```",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
