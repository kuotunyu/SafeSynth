"""Select the smallest v11 geometry repair on revealed Train-only evidence."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.train_supervised_labeler import (
    _aggregate,
    _build_datasets,
    _evaluation_collate,
    _predict,
    _sha256,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.supervised_labeler import filter_prediction_geometry

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v10.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v10_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v10_training.json"
DIAGNOSIS_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v10_review_diagnosis.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v11_geometry_diagnosis.json"
)
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v11_geometry_diagnosis.md"
)
SCORE_FLOOR = 0.001
QUARANTINED_GT_DEFECT_IMAGE_IDS = {3060, 4155, 4364}
GEOMETRY_CANDIDATES = (
    {
        "name": "v10_frozen",
        "max_relative_area": 0.14,
        "max_relative_height": 0.40,
        "min_aspect_ratio": 0.25,
        "max_aspect_ratio": 4.0,
    },
    {
        "name": "height_045",
        "max_relative_area": 0.14,
        "max_relative_height": 0.45,
        "min_aspect_ratio": 0.25,
        "max_aspect_ratio": 4.0,
    },
    {
        "name": "height_050",
        "max_relative_area": 0.14,
        "max_relative_height": 0.50,
        "min_aspect_ratio": 0.25,
        "max_aspect_ratio": 4.0,
    },
    {
        "name": "edge_large_060",
        "max_relative_area": 0.15,
        "max_relative_height": 0.60,
        "min_aspect_ratio": 0.20,
        "max_aspect_ratio": 4.0,
    },
    {
        "name": "edge_large_070",
        "max_relative_area": 0.18,
        "max_relative_height": 0.70,
        "min_aspect_ratio": 0.18,
        "max_aspect_ratio": 4.0,
    },
)


def _filtered_ids(dataset: Any) -> Subset[Any]:
    indices = [
        index
        for index, image_id in enumerate(dataset.image_ids)
        if int(image_id) not in QUARANTINED_GT_DEFECT_IMAGE_IDS
    ]
    return Subset(dataset, indices)


def _candidate_predictions(
    *,
    raw_predictions: dict[int, list[tuple[float, list[float]]]],
    image_ids: list[int],
    train_images: dict[int, dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[int, list[tuple[float, list[float]]]]:
    return {
        image_id: filter_prediction_geometry(
            raw_predictions[image_id],
            image_width=int(train_images[image_id]["width"]),
            image_height=int(train_images[image_id]["height"]),
            max_relative_area=float(candidate["max_relative_area"]),
            max_relative_height=float(candidate["max_relative_height"]),
            min_aspect_ratio=float(candidate["min_aspect_ratio"]),
            max_aspect_ratio=float(candidate["max_aspect_ratio"]),
        )
        for image_id in image_ids
    }


def main() -> None:
    if OUTPUT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("v11 geometry diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("v11 geometry diagnosis requires CUDA")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    diagnosis = json.loads(DIAGNOSIS_PATH.read_text(encoding="utf-8"))
    (
        _,
        split,
        _,
        train_images,
        _,
        _,
        calibration,
        consumed_audit,
    ) = _build_datasets(config_path=CONFIG_PATH, split_path=SPLIT_PATH)
    if set(diagnosis["owner_confirmed_gt_defect_cells"]) != {31, 41, 42}:
        raise RuntimeError("Expected owner-confirmed v10 GT defects")
    checkpoint = Path(report["checkpoint_path"])
    if _sha256(checkpoint / "model.safetensors") != report["checkpoint_sha256"]:
        raise RuntimeError("Passed v10 checkpoint changed")

    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to("cuda")
    consumed = ConcatDataset(
        [_filtered_ids(calibration), _filtered_ids(consumed_audit)]
    )
    loader = DataLoader(
        consumed,
        batch_size=int(config["optimization"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: _evaluation_collate(processor, batch),
    )
    image_ids, truth, raw_predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=SCORE_FLOOR,
        geometry_filter=None,
    )
    if QUARANTINED_GT_DEFECT_IMAGE_IDS & set(image_ids):
        raise RuntimeError("Known GT defects entered geometry calibration")

    thresholds = [float(value) for value in config["calibration"]["score_thresholds"]]
    match_iou = float(config["calibration"]["match_iou"])
    precision_floor = float(config["calibration"]["min_precision"])
    owner_geometry_misses = []
    for case in diagnosis["problem_cases"]:
        for miss in case["misses_against_dataset_gt"]:
            if miss["reason"] == "removed_by_frozen_geometry_filter":
                owner_geometry_misses.append(
                    {
                        "cell": int(case["cell"]),
                        "image_id": int(case["image_id"]),
                        "score": float(miss["best_raw_candidate"]["score"]),
                        "box": [
                            float(value)
                            for value in miss["best_raw_candidate"]["box"]
                        ],
                    }
                )
    if [row["cell"] for row in owner_geometry_misses] != [6, 7, 10, 40]:
        raise RuntimeError("Unexpected v10 geometry-miss set")

    candidate_rows = []
    for candidate in GEOMETRY_CANDIDATES:
        predictions = _candidate_predictions(
            raw_predictions=raw_predictions,
            image_ids=image_ids,
            train_images=train_images,
            candidate=candidate,
        )
        threshold_rows = [
            {
                "threshold": threshold,
                **_aggregate(
                    image_ids=image_ids,
                    truth=truth,
                    predictions=predictions,
                    threshold=threshold,
                    match_iou=match_iou,
                ),
            }
            for threshold in thresholds
        ]
        eligible = [
            row
            for row in threshold_rows
            if float(row["precision"]) >= precision_floor
        ]
        best = max(
            eligible,
            key=lambda row: (
                float(row["f1"]),
                float(row["precision"]),
                float(row["threshold"]),
            ),
            default=None,
        )
        recovery = []
        for miss in owner_geometry_misses:
            image = train_images[miss["image_id"]]
            kept = filter_prediction_geometry(
                [(miss["score"], miss["box"])],
                image_width=int(image["width"]),
                image_height=int(image["height"]),
                max_relative_area=float(candidate["max_relative_area"]),
                max_relative_height=float(candidate["max_relative_height"]),
                min_aspect_ratio=float(candidate["min_aspect_ratio"]),
                max_aspect_ratio=float(candidate["max_aspect_ratio"]),
            )
            recovery.append(
                {
                    "cell": miss["cell"],
                    "geometry_kept": bool(kept),
                    "passes_selected_threshold": bool(
                        kept
                        and best is not None
                        and miss["score"] >= float(best["threshold"])
                    ),
                }
            )
        candidate_rows.append(
            {
                **candidate,
                "best_registered_threshold": best,
                "owner_geometry_recovery": recovery,
                "owner_geometry_recovered": sum(
                    row["passes_selected_threshold"] for row in recovery
                ),
                "threshold_grid": threshold_rows,
            }
        )

    eligible_candidates = [
        row
        for row in candidate_rows
        if row["owner_geometry_recovered"] == len(owner_geometry_misses)
        and row["best_registered_threshold"] is not None
    ]
    if not eligible_candidates:
        raise RuntimeError("No registered geometry candidate repairs v10 misses")
    best_eligible_f1 = max(
        float(row["best_registered_threshold"]["f1"])
        for row in eligible_candidates
    )
    near_best_candidates = [
        row
        for row in eligible_candidates
        if best_eligible_f1
        - float(row["best_registered_threshold"]["f1"])
        <= 0.003
    ]
    recommended = min(
        near_best_candidates,
        key=lambda row: (
            float(row["max_relative_area"]),
            float(row["max_relative_height"]),
            -float(row["min_aspect_ratio"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "status": "v11_geometry_candidate_selected_on_revealed_train_only_history",
        "eligible_for_generation_gate": False,
        "source_experiment": "supervised_labeler_v10",
        "source_checkpoint_sha256": report["checkpoint_sha256"],
        "source_split_manifest_sha256": split["manifest_sha256"],
        "score_floor": SCORE_FLOOR,
        "revealed_images_read": len(image_ids),
        "quarantined_gt_defect_image_ids": sorted(
            QUARANTINED_GT_DEFECT_IMAGE_IDS
        ),
        "quarantined_images_read": 0,
        "owner_geometry_miss_cells": [
            row["cell"] for row in owner_geometry_misses
        ],
        "candidates": candidate_rows,
        "recommended_candidate": recommended,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler v11 geometry diagnosis",
        "",
        "- Evidence: **revealed Train-only history; not gate-eligible**",
        f"- Revealed images read: **{len(image_ids)}**",
        "- Quarantined GT-defect images read: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image generations: **0**",
        "",
        "| Candidate | Area | Height | Min aspect | Recovered | Threshold | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in candidate_rows:
        best = row["best_registered_threshold"]
        lines.append(
            f"| {row['name']} | {row['max_relative_area']:.2f} | "
            f"{row['max_relative_height']:.2f} | "
            f"{row['min_aspect_ratio']:.2f} | "
            f"{row['owner_geometry_recovered']}/4 | "
            f"{float(best['threshold']):.3f} | "
            f"{float(best['precision']):.4f} | "
            f"{float(best['recall']):.4f} | "
            f"{float(best['f1']):.4f} |"
        )
    lines.extend(
        [
            "",
            f"Recommended: **{recommended['name']}**.",
            "",
        ]
    )
    MARKDOWN_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
