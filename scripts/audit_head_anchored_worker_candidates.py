"""Audit helmet-anchored empty regions for the v9b worker generator."""

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
    _load_configs,
    _load_context,
    _load_pass1,
    reflected_padding_guard,
)
from src.synthetic.paired_person import paired_headlike_annotations
from src.synthetic.region_inpaint import (
    adjacent_worker_box,
    infer_person_box_from_headlike,
    mask_edge_margin,
    rounded_box_mask,
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "head_anchored_worker_generation_diagnostic.yaml"
)
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "head_anchored_worker_candidate_audit.json"
)


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


def _calibrate_geometry(
    *,
    annotations: dict[int, list[dict[str, Any]]],
    categories: dict[int, str],
    train_image_ids: set[int],
    calibration: dict[str, Any],
) -> dict[str, float | int]:
    rows: list[tuple[float, float, float, float]] = []
    for image_id in sorted(train_image_ids):
        image_annotations = annotations[image_id]
        for person in image_annotations:
            if categories[int(person["category_id"])] != "person":
                continue
            person_x, person_y, person_width, person_height = (
                float(value) for value in person["bbox"]
            )
            if person_height < float(calibration["min_person_height_px"]):
                continue
            if person_height / max(person_width, 1.0) < float(
                calibration["min_person_height_over_width"]
            ):
                continue
            pairs = paired_headlike_annotations(
                person["bbox"],
                annotations=image_annotations,
                categories=categories,
            )
            helmet_pairs = [
                pair
                for pair in pairs
                if categories[int(pair["category_id"])] == "helmet"
            ]
            if len(helmet_pairs) != 1:
                continue
            helmet_x, helmet_y, helmet_width, helmet_height = (
                float(value) for value in helmet_pairs[0]["bbox"]
            )
            rows.append(
                (
                    person_width / person_height,
                    (
                        helmet_x + helmet_width / 2 - person_x
                    )
                    / person_width,
                    (
                        helmet_y + helmet_height / 2 - person_y
                    )
                    / person_height,
                    helmet_height / person_height,
                )
            )
    if not rows:
        raise RuntimeError("No Train person/helmet geometry rows survived")
    matrix = np.asarray(rows, dtype=np.float64)
    medians = np.median(matrix, axis=0)
    return {
        "pair_count": len(rows),
        "person_width_over_height": float(medians[0]),
        "head_center_x_fraction": float(medians[1]),
        "head_center_y_fraction": float(medians[2]),
        "head_height_fraction": float(medians[3]),
    }


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
    calibration = _calibrate_geometry(
        annotations=annotations,
        categories=categories,
        train_image_ids=set(train_images),
        calibration=config["geometry_calibration"],
    )
    rejected: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    candidate_image_count_before_reflection = 0

    for image_id in sorted(train_images):
        image_annotations = annotations[image_id]
        preliminary: list[dict[str, Any]] = []
        for anchor in image_annotations:
            if categories[int(anchor["category_id"])] != str(
                selection["anchor_class"]
            ):
                continue
            inferred = infer_person_box_from_headlike(
                tuple(float(value) for value in anchor["bbox"]),
                person_width_over_height=float(
                    calibration["person_width_over_height"]
                ),
                head_center_x_fraction=float(
                    calibration["head_center_x_fraction"]
                ),
                head_center_y_fraction=float(
                    calibration["head_center_y_fraction"]
                ),
                head_height_fraction=float(
                    calibration["head_height_fraction"]
                ),
            )
            for scale in selection["target_scale_candidates"]:
                for side in selection["sides"]:
                    for gap_px in selection["horizontal_gap_px"]:
                        target_box = adjacent_worker_box(
                            inferred,
                            scale=float(scale),
                            side=str(side),
                            gap_px=int(gap_px),
                        )
                        target_height = int(target_box[3])
                        if target_height < int(
                            selection["min_target_height_px"]
                        ):
                            rejected["TARGET_TOO_SMALL"] += 1
                            continue
                        if target_height > int(
                            selection["max_target_height_px"]
                        ):
                            rejected["TARGET_TOO_LARGE"] += 1
                            continue
                        if not _inside(
                            target_box,
                            image_shape=(
                                int(train_images[image_id]["height"]),
                                int(train_images[image_id]["width"]),
                            ),
                        ):
                            rejected["REGION_OUTSIDE_FRAME"] += 1
                            continue
                        edit_mask = rounded_box_mask(
                            (
                                int(train_images[image_id]["height"]),
                                int(train_images[image_id]["width"]),
                            ),
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
                        maximum_overlap = max(
                            (
                                _intersection_over_smaller_box(
                                    target_box,
                                    other["bbox"],
                                )
                                for other in image_annotations
                            ),
                            default=0.0,
                        )
                        if maximum_overlap > float(
                            selection[
                                "max_other_annotation_overlap_fraction"
                            ]
                        ):
                            rejected["ANNOTATION_OVERLAP"] += 1
                            continue
                        preliminary.append(
                            {
                                "image_id": image_id,
                                "group_id": int(
                                    frozen[image_id]["group_id"]
                                ),
                                "anchor_annotation_id": int(anchor["id"]),
                                "anchor_bbox_xywh": [
                                    float(value)
                                    for value in anchor["bbox"]
                                ],
                                "inferred_person_bbox_xywh": list(inferred),
                                "target_bbox_xywh": list(target_box),
                                "target_height_px": target_height,
                                "side": str(side),
                                "gap_px": int(gap_px),
                                "scale": float(scale),
                                "edit_edge_margin_px": edge_margin,
                                "maximum_annotation_overlap_fraction": (
                                    maximum_overlap
                                ),
                            }
                        )
        if not preliminary:
            continue
        candidate_image_count_before_reflection += 1
        image_record = train_images[image_id]
        image_rgb = np.asarray(
            Image.open(
                paths.hardhat_raw / str(image_record["file_name"])
            ).convert("RGB")
        )
        reflection = reflected_padding_guard(
            image_rgb,
            guard_config=reflection_config,
        )
        if bool(selection["reject_reflected_padding"]) and reflection.detected:
            rejected["REFLECTED_PADDING_IMAGE"] += len(preliminary)
            continue
        pass1 = _load_pass1(paths, image_id)
        for candidate in preliminary:
            anchor_id = int(candidate["anchor_annotation_id"])
            if (
                bool(selection["require_anchor_pass1_qc"])
                and not bool(pass1[anchor_id]["qc_pass"])
            ):
                rejected["ANCHOR_PASS1_QC"] += 1
                continue
            candidates.append(candidate)

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
        raise AssertionError("Test leakage entered the v9b candidate audit")
    payload = {
        "schema_version": 1,
        "status": "candidate_audit_complete_no_model_inference",
        "architecture": config["architecture"],
        "scope": "frozen Train pixels, labels, and Pass-1 QC only",
        "validation_images_read": 0,
        "test_images_read": 0,
        "model_inference_run": False,
        "h4_auc_computed": False,
        "calibration": calibration,
        "candidate_image_count_before_reflection": (
            candidate_image_count_before_reflection
        ),
        "candidate_count": len(candidates),
        "candidate_image_count": len(candidate_images),
        "candidate_group_count": len(candidate_groups),
        "minimum_group_count": int(
            config["data_scope"]["minimum_group_count_for_four_case_path"]
        ),
        "capacity_pass": len(candidate_groups)
        >= int(config["data_scope"]["minimum_group_count_for_four_case_path"]),
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
