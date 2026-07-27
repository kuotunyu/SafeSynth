"""Audit fixed Train-only candidates for the v8 whole-person edit diagnostic."""

from __future__ import annotations

import json
from collections import Counter
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
    whole_person_edit_mask,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "whole_person_edit_diagnostic.yaml"


def _annotation_id(cutout_id: str) -> int:
    marker = "_ann"
    if marker not in cutout_id:
        raise ValueError(f"Cutout ID lacks {marker}: {cutout_id}")
    return int(cutout_id.rsplit(marker, maxsplit=1)[1])


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    paths = load_project_paths()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    compose_config, _ = _load_configs()
    reflection_config = compose_config["compose"]["context_replacement"][
        "input_guard"
    ]["reflected_padding"]
    coco, bank, train_images, annotations, frozen, test_ids = _load_context(paths)
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco["categories"]
    }
    selection = config["selection"]
    rejected: Counter[str] = Counter()
    accepted: list[dict[str, Any]] = []
    normalized_cache: dict[
        int,
        tuple[
            tuple[dict[str, Any], ...],
            dict[int, dict[str, Any]],
            tuple[str, ...],
        ],
    ] = {}

    for person in bank:
        if str(person["class_name"]) != "person":
            continue
        if str(person["src_split"]) != "train":
            rejected["NOT_TRAIN"] += 1
            continue
        if (
            bool(selection["preferred_person_required"])
            and not bool(person["preferred_tier"])
        ):
            rejected["NOT_PREFERRED"] += 1
            continue
        image_id = int(person["src_image_id"])
        person_id = _annotation_id(str(person["cutout_id"]))
        raw_pairs = paired_headlike_annotations(
            person["src_bbox_xywh"],
            annotations=annotations[image_id],
            categories=categories,
        )
        if len(raw_pairs) != int(selection["paired_headlike_count"]):
            rejected["PAIRED_HEADLIKE_COUNT"] += 1
            continue
        raw_headlike = raw_pairs[0]
        if categories[int(raw_headlike["category_id"])] != str(
            selection["target_class"]
        ):
            rejected["PAIRED_CLASS"] += 1
            continue

        if image_id not in normalized_cache:
            image_record = train_images[image_id]
            raw = np.asarray(
                Image.open(
                    paths.hardhat_raw / str(image_record["file_name"])
                ).convert("RGB")
            )
            raw_pass1 = _load_pass1(paths, image_id)
            reflection = reflected_padding_guard(
                raw,
                guard_config=reflection_config,
            )
            _, transformed_annotations, transformed_pass1, normalization = (
                normalize_reflected_padding(
                    raw,
                    annotations=annotations[image_id],
                    pass1=raw_pass1,
                    reflection=reflection,
                    output_shape=raw.shape[:2],
                    transform_masks=True,
                )
            )
            normalized_cache[image_id] = (
                transformed_annotations,
                transformed_pass1,
                normalization.detected_sides,
            )

        transformed_annotations, transformed_pass1, detected_sides = (
            normalized_cache[image_id]
        )
        annotation_by_id = {
            int(annotation["id"]): annotation
            for annotation in transformed_annotations
        }
        headlike_id = int(raw_headlike["id"])
        if person_id not in annotation_by_id or headlike_id not in annotation_by_id:
            rejected["CROPPED_BY_NORMALIZATION"] += 1
            continue
        person_annotation = annotation_by_id[person_id]
        if float(person_annotation["bbox"][3]) < float(
            selection["min_person_height_px"]
        ):
            rejected["PERSON_TOO_SMALL"] += 1
            continue
        if (
            bool(selection["require_pass1_qc"])
            and (
                not bool(transformed_pass1[person_id]["qc_pass"])
                or not bool(transformed_pass1[headlike_id]["qc_pass"])
            )
        ):
            rejected["PASS1_QC"] += 1
            continue

        person_mask = _decode_rle(
            transformed_pass1[person_id]["segmentation"]
        )
        headlike_mask = _decode_rle(
            transformed_pass1[headlike_id]["segmentation"]
        )
        edit_mask = whole_person_edit_mask(
            person_mask,
            headlike_mask,
            outer_dilate_px=int(selection["outer_dilate_px"]),
        )
        edge_margin = mask_edge_margin(edit_mask)
        if edge_margin < int(selection["min_edit_edge_margin_px"]):
            rejected["EDIT_NEAR_FRAME_EDGE"] += 1
            continue

        other_masks = []
        for annotation in transformed_annotations:
            annotation_id = int(annotation["id"])
            if annotation_id in {person_id, headlike_id}:
                continue
            mask_record = transformed_pass1[annotation_id]
            if bool(mask_record["qc_pass"]):
                other_masks.append(_decode_rle(mask_record["segmentation"]))
        maximum_overlap = maximum_other_mask_overlap_fraction(
            edit_mask,
            other_masks,
        )
        if maximum_overlap > float(
            selection["max_other_mask_overlap_fraction"]
        ):
            rejected["OTHER_INSTANCE_OVERLAP"] += 1
            continue

        accepted.append(
            {
                "cutout_id": str(person["cutout_id"]),
                "person_annotation_id": person_id,
                "headlike_annotation_id": headlike_id,
                "image_id": image_id,
                "group_id": int(frozen[image_id]["group_id"]),
                "person_height_px": float(person_annotation["bbox"][3]),
                "person_mask_area_px": int(person_mask.sum()),
                "edit_mask_area_px": int(edit_mask.sum()),
                "edit_edge_margin_px": edge_margin,
                "maximum_other_mask_overlap_fraction": maximum_overlap,
                "normalization_detected_sides": list(detected_sides),
            }
        )

    accepted.sort(
        key=lambda item: (
            -float(item["person_height_px"]),
            -int(item["edit_edge_margin_px"]),
            str(item["cutout_id"]),
        )
    )
    accepted_groups = {int(item["group_id"]) for item in accepted}
    if test_ids & {int(item["image_id"]) for item in accepted}:
        raise AssertionError("Test leakage entered the v8 candidate audit")
    payload = {
        "schema_version": 1,
        "status": "candidate_audit_complete_no_model_inference",
        "architecture": config["architecture"],
        "scope": "frozen Train pixels, annotations, cutouts, and Pass-1 masks only",
        "validation_images_read": 0,
        "test_images_read": 0,
        "model_inference_run": False,
        "h4_auc_computed": False,
        "accepted_candidate_count": len(accepted),
        "accepted_group_count": len(accepted_groups),
        "rejection_counts": dict(sorted(rejected.items())),
        "candidates": accepted,
    }
    report_path = (
        PROJECT_ROOT / "reports" / "whole_person_edit_candidate_audit.json"
    )
    _write_json(report_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
