"""Test fixed geometry filters on already-consumed Train-only labeler sets."""

from __future__ import annotations

import gc
import json
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.diagnose_supervised_labeler_failure import _experiment_paths
from scripts.train_supervised_labeler import (
    _aggregate,
    _build_datasets,
    _evaluation_collate,
    _predict,
    _sha256,
)
from src.data.paths import PROJECT_ROOT

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_postprocessing_diagnosis.json"
)
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_postprocessing_diagnosis.md"
)
EXPERIMENTS = ("v1", "v3", "v4")
SCORE_THRESHOLDS = (0.015, 0.020, 0.025, 0.030, 0.035)
MAX_RELATIVE_AREAS = (0.025, 0.040, 0.060, 0.080, 0.120, 0.200, 1.000)
MAX_RELATIVE_HEIGHTS = (0.35, 0.45, 0.60, 1.00)


def filter_prediction_geometry(
    predictions: Sequence[tuple[float, Sequence[float]]],
    *,
    image_width: int,
    image_height: int,
    max_relative_area: float,
    max_relative_height: float,
) -> list[tuple[float, list[float]]]:
    """Drop predictions exceeding fixed normalized size limits."""

    image_area = max(float(image_width) * float(image_height), 1.0)
    kept = []
    for score, box in predictions:
        x1, y1, x2, y2 = (float(value) for value in box)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if (
            width * height / image_area <= float(max_relative_area)
            and height / max(float(image_height), 1.0)
            <= float(max_relative_height)
        ):
            kept.append((float(score), [x1, y1, x2, y2]))
    return kept


def _filtered_predictions(
    *,
    predictions: Mapping[int, Sequence[tuple[float, Sequence[float]]]],
    image_records: Mapping[int, Mapping[str, Any]],
    max_relative_area: float,
    max_relative_height: float,
) -> dict[int, list[tuple[float, list[float]]]]:
    return {
        image_id: filter_prediction_geometry(
            rows,
            image_width=int(image_records[image_id]["width"]),
            image_height=int(image_records[image_id]["height"]),
            max_relative_area=max_relative_area,
            max_relative_height=max_relative_height,
        )
        for image_id, rows in predictions.items()
    }


def _grid(
    *,
    image_ids: Sequence[int],
    truth: Mapping[int, Sequence[Sequence[float]]],
    predictions: Mapping[int, Sequence[tuple[float, Sequence[float]]]],
    image_records: Mapping[int, Mapping[str, Any]],
    match_iou: float,
) -> list[dict[str, float | int]]:
    rows = []
    for max_relative_area in MAX_RELATIVE_AREAS:
        for max_relative_height in MAX_RELATIVE_HEIGHTS:
            filtered = _filtered_predictions(
                predictions=predictions,
                image_records=image_records,
                max_relative_area=max_relative_area,
                max_relative_height=max_relative_height,
            )
            for threshold in SCORE_THRESHOLDS:
                rows.append(
                    {
                        "threshold": threshold,
                        "max_relative_area": max_relative_area,
                        "max_relative_height": max_relative_height,
                        **_aggregate(
                            image_ids=image_ids,
                            truth=truth,
                            predictions=filtered,
                            threshold=threshold,
                            match_iou=match_iou,
                        ),
                    }
                )
    return rows


def select_geometry_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    precision_floor: float,
) -> dict[str, Any] | None:
    """Select the strongest deterministic calibration candidate."""

    eligible = [
        dict(row)
        for row in rows
        if float(row["precision"]) >= float(precision_floor)
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            -float(row["f1"]),
            -float(row["recall"]),
            -float(row["median_matched_iou"]),
            float(row["max_relative_area"]),
            float(row["max_relative_height"]),
            -float(row["threshold"]),
        )
    )
    return eligible[0]


def _metrics_for_candidate(
    *,
    candidate: Mapping[str, Any],
    image_ids: Sequence[int],
    truth: Mapping[int, Sequence[Sequence[float]]],
    predictions: Mapping[int, Sequence[tuple[float, Sequence[float]]]],
    image_records: Mapping[int, Mapping[str, Any]],
    match_iou: float,
) -> dict[str, float | int]:
    filtered = _filtered_predictions(
        predictions=predictions,
        image_records=image_records,
        max_relative_area=float(candidate["max_relative_area"]),
        max_relative_height=float(candidate["max_relative_height"]),
    )
    return {
        "threshold": float(candidate["threshold"]),
        "max_relative_area": float(candidate["max_relative_area"]),
        "max_relative_height": float(candidate["max_relative_height"]),
        **_aggregate(
            image_ids=image_ids,
            truth=truth,
            predictions=filtered,
            threshold=float(candidate["threshold"]),
            match_iou=match_iou,
        ),
    }


def main() -> None:
    if REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("Postprocessing diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Postprocessing diagnosis requires CUDA")

    experiment_data: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        paths = _experiment_paths(experiment)
        report = json.loads(paths["training"].read_text(encoding="utf-8"))
        if (
            report["status"] != "supervised_labeler_audit_failed"
            or int(report["untouched_audit_images_read"]) != 48
            or int(report["validation_images_read"]) != 0
            or int(report["test_images_read"]) != 0
        ):
            raise RuntimeError(f"{experiment} is not a completed failed audit")
        (
            config,
            split,
            _,
            image_records,
            _,
            _,
            calibration,
            consumed_audit,
        ) = _build_datasets(
            config_path=paths["config"],
            split_path=paths["split"],
        )
        checkpoint = Path(report["checkpoint_path"])
        if _sha256(checkpoint / "model.safetensors") != report["checkpoint_sha256"]:
            raise RuntimeError(f"{experiment} checkpoint changed")
        processor = AutoImageProcessor.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        model = AutoModelForObjectDetection.from_pretrained(
            checkpoint,
            local_files_only=True,
        ).to("cuda")
        known = ConcatDataset([calibration, consumed_audit])
        loader = DataLoader(
            known,
            batch_size=int(config["optimization"]["eval_batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=partial(_evaluation_collate, processor),
        )
        image_ids, truth, predictions = _predict(
            model=model,
            processor=processor,
            loader=loader,
            device="cuda",
            score_floor=min(SCORE_THRESHOLDS),
        )
        experiment_data[experiment] = {
            "config": config,
            "split": split,
            "report": report,
            "image_ids": image_ids,
            "truth": truth,
            "predictions": predictions,
            "image_records": image_records,
        }
        del model
        gc.collect()
        torch.cuda.empty_cache()

    reference = experiment_data["v4"]
    v4_split = reference["split"]
    v4_calibration_grid = _grid(
        image_ids=[int(value) for value in v4_split["calibration_image_ids"]],
        truth=reference["truth"],
        predictions=reference["predictions"],
        image_records=reference["image_records"],
        match_iou=float(reference["config"]["calibration"]["match_iou"]),
    )
    candidates = {
        str(floor): select_geometry_candidate(
            v4_calibration_grid,
            precision_floor=floor,
        )
        for floor in (0.80, 0.85, 0.87, 0.90)
    }
    selected = candidates["0.8"]
    if selected is None:
        raise RuntimeError("No v4 calibration candidate met precision 0.80")

    audit_results = {}
    local_candidates = {}
    for experiment, data in experiment_data.items():
        split = data["split"]
        calibration_ids = [
            int(value) for value in split["calibration_image_ids"]
        ]
        audit_ids = [
            int(value) for value in split["untouched_audit_image_ids"]
        ]
        local_grid = _grid(
            image_ids=calibration_ids,
            truth=data["truth"],
            predictions=data["predictions"],
            image_records=data["image_records"],
            match_iou=float(data["config"]["calibration"]["match_iou"]),
        )
        local_candidates[experiment] = select_geometry_candidate(
            local_grid,
            precision_floor=0.80,
        )
        audit_results[experiment] = _metrics_for_candidate(
            candidate=selected,
            image_ids=audit_ids,
            truth=data["truth"],
            predictions=data["predictions"],
            image_records=data["image_records"],
            match_iou=float(data["config"]["calibration"]["match_iou"]),
        )

    payload = {
        "schema_version": 1,
        "status": "diagnostic_on_consumed_train_only_sets",
        "eligible_for_generation_gate": False,
        "reason": (
            "All images were already used for calibration or failed audits. "
            "This can preregister a new experiment but cannot pass a gate."
        ),
        "selection_source": "v4 prior calibration only",
        "experiments": list(EXPERIMENTS),
        "score_thresholds": list(SCORE_THRESHOLDS),
        "max_relative_areas": list(MAX_RELATIVE_AREAS),
        "max_relative_heights": list(MAX_RELATIVE_HEIGHTS),
        "selected_at_v4_calibration_precision_0_80": selected,
        "v4_candidates_by_precision_floor": candidates,
        "selected_candidate_failed_audit_results": audit_results,
        "experiment_local_calibration_candidates_at_precision_0_80": (
            local_candidates
        ),
        "checkpoint_sha256": {
            experiment: data["report"]["checkpoint_sha256"]
            for experiment, data in experiment_data.items()
        },
        "source_split_manifest_sha256": {
            experiment: data["split"]["manifest_sha256"]
            for experiment, data in experiment_data.items()
        },
        "images_read": {
            experiment: len(data["image_ids"])
            for experiment, data in experiment_data.items()
        },
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler postprocessing diagnosis",
        "",
        "- Evidence class: **consumed Train-only sets; not gate-eligible**",
        "- Selection source: **v4 prior calibration only**",
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image FLUX generations: **0**",
        "",
        "## Selected v4 calibration candidate",
        "",
        (
            f"- Score threshold: **{float(selected['threshold']):.4f}**; "
            f"maximum relative area: "
            f"**{float(selected['max_relative_area']):.3f}**; "
            f"maximum relative height: "
            f"**{float(selected['max_relative_height']):.2f}**"
        ),
        (
            f"- Calibration precision / recall / F1: "
            f"**{float(selected['precision']):.4f} / "
            f"{float(selected['recall']):.4f} / "
            f"{float(selected['f1']):.4f}**"
        ),
        "",
        "## Same candidate on consumed failed audits",
        "",
        "| Experiment | Precision | Recall | F1 | Median IoU |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {experiment} | {float(row['precision']):.4f} | "
            f"{float(row['recall']):.4f} | {float(row['f1']):.4f} | "
            f"{float(row['median_matched_iou']):.4f} |"
        )
        for experiment, row in audit_results.items()
    )
    lines.append("")
    MARKDOWN_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
