"""Audit Train-only empty regions for prompt-only adjacent-worker generation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import (
    _decode_rle,
    _load_configs,
    _load_context,
    _load_pass1,
    normalize_reflected_padding,
    reflected_padding_guard,
)
from src.synthetic.paired_person import paired_headlike_annotations
from src.synthetic.region_inpaint import (
    mask_edge_margin,
    maximum_other_mask_overlap_fraction,
    rounded_box_mask,
)

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "regional_worker_generation_diagnostic.yaml"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "regional_worker_candidate_audit.json"


def _intersection_over_smaller_box(
    first_xywh: Sequence[float],
    second_xywh: Sequence[float],
) -> float:
    first_x, first_y, first_width, first_height = (
        float(value) for value in first_xywh
    )
    second_x, second_y, second_width, second_height = (
        float(value) for value in second_xywh
    )
    width = max(
        0.0,
        min(first_x + first_width, second_x + second_width)
        - max(first_x, second_x),
    )
    height = max(
        0.0,
        min(first_y + first_height, second_y + second_height)
        - max(first_y, second_y),
    )
    smaller = min(first_width * first_height, second_width * second_height)
    return width * height / smaller if smaller > 0 else 0.0


def _candidate_box(
    anchor_xywh: Sequence[float],
    *,
    scale: float,
    side: str,
    gap_px: int,
) -> tuple[int, int, int, int]:
    anchor_x, anchor_y, anchor_width, anchor_height = (
        float(value) for value in anchor_xywh
    )
    target_width = max(1, round(anchor_width * scale))
    target_height = max(1, round(anchor_height * scale))
    target_y = round(anchor_y + anchor_height - target_height)
    if side == "left":
        target_x = round(anchor_x - gap_px - target_width)
    elif side == "right":
        target_x = round(anchor_x + anchor_width + gap_px)
    else:
        raise ValueError(f"Unknown side: {side}")
    return target_x, target_y, target_width, target_height


def _inside(
    bbox_xywh: Sequence[int],
    *,
    image_shape: tuple[int, int],
) -> bool:
    x, y, width, height = (int(value) for value in bbox_xywh)
    image_height, image_width = image_shape
    return (
        x >= 0
        and y >= 0
        and width > 0
        and height > 0
        and x + width <= image_width
        and y + height <= image_height
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    paths = load_project_paths()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    selection = config["selection"]
    compose_config, _ = _load_configs()
    reflection_config = compose_config["compose"]["context_replacement"][
        "input_guard"
    ]["reflected_padding"]
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco["categories"]
    }
    rejected: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    anchor_count = 0
    person_category_ids = {
        category_id
        for category_id, name in categories.items()
        if name == str(selection["anchor_class"])
    }

    for image_id in sorted(train_images):
        if not any(
            int(annotation["category_id"]) in person_category_ids
            for annotation in annotations[image_id]
        ):
            continue
        image_record = train_images[image_id]
        raw = np.asarray(
            Image.open(
                paths.hardhat_raw / str(image_record["file_name"])
            ).convert("RGB")
        )
        reflection = reflected_padding_guard(
            raw,
            guard_config=reflection_config,
        )
        normalized, transformed_annotations, transformed_pass1, normalization = (
            normalize_reflected_padding(
                raw,
                annotations=annotations[image_id],
                pass1=_load_pass1(paths, image_id),
                reflection=reflection,
                output_shape=raw.shape[:2],
                transform_masks=True,
            )
        )
        qc_masks = {
            annotation_id: _decode_rle(record["segmentation"])
            for annotation_id, record in transformed_pass1.items()
            if bool(record["qc_pass"])
        }
        for anchor in transformed_annotations:
            anchor_id = int(anchor["id"])
            if categories[int(anchor["category_id"])] != str(
                selection["anchor_class"]
            ):
                continue
            if (
                bool(selection["require_pass1_qc"])
                and anchor_id not in qc_masks
            ):
                rejected["ANCHOR_PASS1_QC"] += 1
                continue
            _, y, width, height = (
                float(value) for value in anchor["bbox"]
            )
            if height < float(selection["min_anchor_height_px"]):
                rejected["ANCHOR_TOO_SMALL"] += 1
                continue
            if height / max(width, 1.0) < float(
                selection["min_anchor_height_over_width"]
            ):
                rejected["ANCHOR_TOO_HORIZONTAL"] += 1
                continue
            if (y + height) / normalized.shape[0] < float(
                selection["min_anchor_bottom_fraction"]
            ):
                rejected["ANCHOR_TOO_HIGH"] += 1
                continue
            pairs = paired_headlike_annotations(
                anchor["bbox"],
                annotations=transformed_annotations,
                categories=categories,
            )
            if len(pairs) != int(selection["paired_headlike_count"]):
                rejected["PAIRED_HEADLIKE_COUNT"] += 1
                continue
            if categories[int(pairs[0]["category_id"])] != str(
                selection["paired_headlike_class"]
            ):
                rejected["PAIRED_HEADLIKE_CLASS"] += 1
                continue
            anchor_count += 1

            for scale in selection["target_scale_candidates"]:
                for side in selection["sides"]:
                    for gap_px in selection["horizontal_gap_px"]:
                        target_box = _candidate_box(
                            anchor["bbox"],
                            scale=float(scale),
                            side=str(side),
                            gap_px=int(gap_px),
                        )
                        if not _inside(
                            target_box,
                            image_shape=normalized.shape[:2],
                        ):
                            rejected["REGION_OUTSIDE_FRAME"] += 1
                            continue
                        edit_mask = rounded_box_mask(
                            normalized.shape[:2],
                            target_box,
                            corner_fraction=float(
                                selection["rounded_corner_fraction"]
                            ),
                        )
                        edge_margin = mask_edge_margin(edit_mask)
                        if edge_margin < int(
                            selection["min_region_edge_margin_px"]
                        ):
                            rejected["REGION_NEAR_FRAME_EDGE"] += 1
                            continue
                        maximum_box_overlap = max(
                            (
                                _intersection_over_smaller_box(
                                    target_box,
                                    other["bbox"],
                                )
                                for other in transformed_annotations
                            ),
                            default=0.0,
                        )
                        if maximum_box_overlap > float(
                            selection[
                                "max_other_annotation_overlap_fraction"
                            ]
                        ):
                            rejected["ANNOTATION_OVERLAP"] += 1
                            continue
                        maximum_mask_overlap = (
                            maximum_other_mask_overlap_fraction(
                                edit_mask,
                                qc_masks.values(),
                            )
                        )
                        if maximum_mask_overlap > float(
                            selection["max_other_mask_overlap_fraction"]
                        ):
                            rejected["MASK_OVERLAP"] += 1
                            continue
                        candidates.append(
                            {
                                "image_id": image_id,
                                "group_id": int(
                                    frozen[image_id]["group_id"]
                                ),
                                "anchor_annotation_id": anchor_id,
                                "anchor_bbox_xywh": [
                                    float(value)
                                    for value in anchor["bbox"]
                                ],
                                "anchor_height_px": height,
                                "paired_helmet_annotation_id": int(
                                    pairs[0]["id"]
                                ),
                                "target_bbox_xywh": list(target_box),
                                "target_height_px": int(target_box[3]),
                                "side": str(side),
                                "gap_px": int(gap_px),
                                "scale": float(scale),
                                "edit_edge_margin_px": edge_margin,
                                "maximum_annotation_overlap_fraction": (
                                    maximum_box_overlap
                                ),
                                "maximum_mask_overlap_fraction": (
                                    maximum_mask_overlap
                                ),
                                "normalization_detected_sides": list(
                                    normalization.detected_sides
                                ),
                            }
                        )

    candidates.sort(
        key=lambda item: (
            -int(item["target_height_px"]),
            -int(item["edit_edge_margin_px"]),
            int(item["image_id"]),
            int(item["anchor_annotation_id"]),
            str(item["side"]),
            int(item["gap_px"]),
            -float(item["scale"]),
        )
    )
    candidate_images = {int(item["image_id"]) for item in candidates}
    candidate_groups = {int(item["group_id"]) for item in candidates}
    if test_ids & candidate_images:
        raise AssertionError("Test leakage entered the v9 candidate audit")
    payload = {
        "schema_version": 1,
        "status": "candidate_audit_complete_no_model_inference",
        "architecture": config["architecture"],
        "scope": "frozen Train pixels, labels, and Pass-1 masks only",
        "validation_images_read": 0,
        "test_images_read": 0,
        "model_inference_run": False,
        "h4_auc_computed": False,
        "eligible_anchor_count": anchor_count,
        "candidate_count": len(candidates),
        "candidate_image_count": len(candidate_images),
        "candidate_group_count": len(candidate_groups),
        "rejection_counts": dict(sorted(rejected.items())),
        "candidates": candidates,
    }
    _write_json(REPORT_PATH, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "candidates"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
