"""Evaluate geometry relaxations on fully revealed v7 Train-only evidence."""

from __future__ import annotations

import gc
import json
from itertools import product
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.diagnose_supervised_labeler_v7_review import REVIEW_PATH
from scripts.train_supervised_labeler import (
    _aggregate,
    _build_datasets,
    _calibration_rows,
    _evaluation_collate,
    _predict,
    _sha256,
    select_calibration_candidate,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import box_iou_xyxy
from src.synthetic.supervised_labeler import filter_prediction_geometry

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v7.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v7_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v7_training.json"
DIAGNOSIS_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v7_review_diagnosis.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v7_geometry_diagnosis.json"
)
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v7_geometry_diagnosis.md"
)
MAX_RELATIVE_AREAS = (0.08, 0.10, 0.12, 0.14, 0.16)
MAX_RELATIVE_HEIGHTS = (0.35, 0.40, 0.45, 0.50)
RECOMMENDED_MAX_RELATIVE_AREA = 0.14
RECOMMENDED_MAX_RELATIVE_HEIGHT = 0.40


def _filter_all(
    *,
    predictions: dict[int, list[tuple[float, list[float]]]],
    train_images: dict[int, Any],
    max_relative_area: float,
    max_relative_height: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> dict[int, list[tuple[float, list[float]]]]:
    filtered = {}
    for image_id, rows in predictions.items():
        image = train_images[int(image_id)]
        filtered[int(image_id)] = filter_prediction_geometry(
            rows,
            image_width=int(image["width"]),
            image_height=int(image["height"]),
            max_relative_area=max_relative_area,
            max_relative_height=max_relative_height,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
    return filtered


def _owner_miss_recovery(
    *,
    diagnosis: dict[str, Any],
    predictions: dict[int, list[tuple[float, list[float]]]],
    threshold: float,
    match_iou: float,
) -> dict[str, int]:
    geometry_total = 0
    geometry_recovered = 0
    all_total = 0
    all_recovered = 0
    for case in diagnosis["problem_cases"]:
        image_id = int(case["image_id"])
        for miss in case["misses"]:
            all_total += 1
            is_geometry = (
                miss["reason"] == "removed_by_frozen_geometry_filter"
            )
            geometry_total += int(is_geometry)
            recovered = any(
                float(score) >= threshold
                and box_iou_xyxy(miss["truth_box"], box) >= match_iou
                for score, box in predictions[image_id]
            )
            all_recovered += int(recovered)
            geometry_recovered += int(recovered and is_geometry)
    return {
        "owner_misses_total": all_total,
        "owner_misses_recovered": all_recovered,
        "geometry_misses_total": geometry_total,
        "geometry_misses_recovered": geometry_recovered,
    }


def main() -> None:
    if OUTPUT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("v7 geometry diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("v7 geometry diagnosis requires CUDA")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    diagnosis = json.loads(DIAGNOSIS_PATH.read_text(encoding="utf-8"))
    if (
        report["status"] != "supervised_labeler_audit_passed"
        or review["status"] != "rejected_by_kuotunyu"
        or diagnosis["status"]
        != "diagnostic_on_revealed_v7_train_only_audit"
    ):
        raise RuntimeError("Expected the frozen v7 owner-review rejection")

    (
        config,
        split,
        _,
        train_images,
        _,
        _,
        calibration,
        consumed_audit,
    ) = _build_datasets(config_path=CONFIG_PATH, split_path=SPLIT_PATH)
    checkpoint = Path(report["checkpoint_path"])
    if _sha256(checkpoint / "model.safetensors") != report["checkpoint_sha256"]:
        raise RuntimeError("Passed v7 checkpoint changed")
    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to("cuda")
    revealed = ConcatDataset([calibration, consumed_audit])
    loader = DataLoader(
        revealed,
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
        score_floor=min(config["calibration"]["score_thresholds"]),
        geometry_filter=None,
    )
    expected_ids = [
        int(value)
        for value in (
            split["calibration_image_ids"] + split["untouched_audit_image_ids"]
        )
    ]
    if image_ids != expected_ids:
        raise RuntimeError("Revealed v7 image order changed")

    calibration_ids = [
        int(value) for value in split["calibration_image_ids"]
    ]
    audit_ids = [
        int(value) for value in split["untouched_audit_image_ids"]
    ]
    match_iou = float(config["calibration"]["match_iou"])
    min_aspect_ratio = float(config["postprocessing"]["min_aspect_ratio"])
    max_aspect_ratio = float(config["postprocessing"]["max_aspect_ratio"])
    rows = []
    for max_area, max_height in product(
        MAX_RELATIVE_AREAS,
        MAX_RELATIVE_HEIGHTS,
    ):
        filtered = _filter_all(
            predictions=raw_predictions,
            train_images=train_images,
            max_relative_area=max_area,
            max_relative_height=max_height,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
        )
        calibration_rows = _calibration_rows(
            epoch=int(report["best_calibration"]["epoch"]),
            image_ids=calibration_ids,
            truth=truth,
            predictions=filtered,
            config=config,
        )
        selected = select_calibration_candidate(
            calibration_rows,
            precision_floor=float(config["calibration"]["min_precision"]),
        )
        if selected is None:
            raise RuntimeError("Geometry candidate has no calibration threshold")
        threshold = float(selected["threshold"])
        audit_metrics = _aggregate(
            image_ids=audit_ids,
            truth=truth,
            predictions=filtered,
            threshold=threshold,
            match_iou=match_iou,
        )
        recovery = _owner_miss_recovery(
            diagnosis=diagnosis,
            predictions=filtered,
            threshold=threshold,
            match_iou=match_iou,
        )
        rows.append(
            {
                "max_relative_area": max_area,
                "max_relative_height": max_height,
                "selected_threshold": threshold,
                "calibration_metrics": {
                    key: selected[key]
                    for key in (
                        "precision",
                        "recall",
                        "f1",
                        "median_matched_iou",
                        "true_positives",
                        "false_positives",
                        "false_negatives",
                    )
                },
                "revealed_audit_metrics": audit_metrics,
                **recovery,
            }
        )

    recommended = next(
        row
        for row in rows
        if row["max_relative_area"] == RECOMMENDED_MAX_RELATIVE_AREA
        and row["max_relative_height"] == RECOMMENDED_MAX_RELATIVE_HEIGHT
    )
    if int(recommended["geometry_misses_recovered"]) != int(
        recommended["geometry_misses_total"]
    ):
        raise RuntimeError("Minimal outward geometry rounding failed")
    payload = {
        "schema_version": 1,
        "status": "geometry_diagnostic_on_revealed_v7_train_only_history",
        "eligible_for_generation_gate": False,
        "reason": (
            "All calibration and v7 audit images are revealed Train-only "
            "history. Results may preregister v8 but cannot pass a gate."
        ),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_split_manifest_sha256": split["manifest_sha256"],
        "review_evidence_sha256": review["evidence_sha256"],
        "revealed_images_read": len(image_ids),
        "revealed_calibration_images_read": len(calibration_ids),
        "revealed_audit_images_read": len(audit_ids),
        "candidate_grid": rows,
        "recommendation_basis": (
            "Smallest round-number outward bounds covering every "
            "owner-reported miss removed by the v7 geometry filter."
        ),
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
        "# Supervised labeler v7 geometry diagnosis",
        "",
        "- Evidence class: **revealed Train-only history; not gate-eligible**",
        f"- Revealed images read: **{len(image_ids)}**",
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image generations: **0**",
        (
            "- Recommended geometry: max relative area "
            f"**{RECOMMENDED_MAX_RELATIVE_AREA:.2f}**, max relative height "
            f"**{RECOMMENDED_MAX_RELATIVE_HEIGHT:.2f}**"
        ),
        "",
        "| Max area | Max height | Threshold | Audit precision | Audit recall | Audit F1 | Geometry misses recovered |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["revealed_audit_metrics"]
        lines.append(
            f"| {float(row['max_relative_area']):.2f} | "
            f"{float(row['max_relative_height']):.2f} | "
            f"{float(row['selected_threshold']):.3f} | "
            f"{float(metrics['precision']):.4f} | "
            f"{float(metrics['recall']):.4f} | "
            f"{float(metrics['f1']):.4f} | "
            f"{row['geometry_misses_recovered']}/"
            f"{row['geometry_misses_total']} |"
        )
    MARKDOWN_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
