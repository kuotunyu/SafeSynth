"""Diagnose the failed labeler on already-consumed Train-only evaluation sets."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.train_supervised_labeler import (
    _aggregate,
    _build_datasets,
    _evaluation_collate,
    _predict,
    _sha256,
    select_calibration_candidate,
)
from src.data.paths import PROJECT_ROOT


def diagnostic_thresholds() -> list[float]:
    """Return the preregistered low-score diagnostic grid."""

    return [
        0.001,
        0.002,
        0.005,
        0.010,
        0.015,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
        0.045,
        0.050,
    ]


def _experiment_paths(experiment: str) -> dict[str, Path]:
    if experiment == "v1":
        stem = "supervised_labeler"
        diagnosis_stem = "supervised_labeler_failure_diagnosis"
    elif experiment in {"v2", "v3"}:
        stem = f"supervised_labeler_{experiment}"
        diagnosis_stem = f"{stem}_failure_diagnosis"
    else:
        raise ValueError(f"Unknown supervised experiment: {experiment}")
    return {
        "config": PROJECT_ROOT / "configs" / f"{stem}.yaml",
        "split": PROJECT_ROOT / "splits" / f"{stem}_split.json",
        "training": PROJECT_ROOT / "reports" / f"{stem}_training.json",
        "diagnosis": PROJECT_ROOT / "reports" / f"{diagnosis_stem}.json",
        "markdown": PROJECT_ROOT / "reports" / f"{diagnosis_stem}.md",
    }


def _candidate(
    rows: list[dict[str, Any]],
    *,
    precision_floor: float,
) -> dict[str, Any] | None:
    return select_calibration_candidate(
        rows,
        precision_floor=precision_floor,
    )


def _threshold_rows(
    *,
    image_ids: list[int],
    truth: dict[int, list[list[float]]],
    predictions: dict[int, list[tuple[float, list[float]]]],
    thresholds: list[float],
    epoch: int,
    match_iou: float,
) -> list[dict[str, Any]]:
    return [
        {
            "epoch": epoch,
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


def main(experiment: str = "v1") -> None:
    paths = _experiment_paths(experiment)
    if paths["diagnosis"].exists() or paths["markdown"].exists():
        raise RuntimeError("Failure diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Failure diagnosis requires CUDA")
    report = json.loads(paths["training"].read_text(encoding="utf-8"))
    if (
        report["status"] != "supervised_labeler_audit_failed"
        or int(report["untouched_audit_images_read"]) != 48
        or int(report["validation_images_read"]) != 0
        or int(report["test_images_read"]) != 0
    ):
        raise RuntimeError("Expected one completed failed Train-only audit")
    (
        config,
        split,
        _,
        _,
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
        raise RuntimeError("Best checkpoint changed after the failed audit")

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
        collate_fn=lambda batch: _evaluation_collate(processor, batch),
    )
    if experiment == "v1":
        thresholds = diagnostic_thresholds()
    else:
        thresholds = [
            float(value) for value in config["calibration"]["score_thresholds"]
        ]
        if experiment == "v3":
            thresholds = sorted(
                set(thresholds)
                | {0.040 + index * 0.0005 for index in range(1, 10)}
            )
    image_ids, truth, predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=min(thresholds),
    )
    epoch = int(report["best_calibration"]["epoch"])
    match_iou = float(config["calibration"]["match_iou"])
    rows = _threshold_rows(
        image_ids=image_ids,
        truth=truth,
        predictions=predictions,
        thresholds=thresholds,
        epoch=epoch,
        match_iou=match_iou,
    )
    prior_calibration_rows = _threshold_rows(
        image_ids=[int(value) for value in split["calibration_image_ids"]],
        truth=truth,
        predictions=predictions,
        thresholds=thresholds,
        epoch=epoch,
        match_iou=match_iou,
    )
    failed_audit_rows = _threshold_rows(
        image_ids=[
            int(value) for value in split["untouched_audit_image_ids"]
        ],
        truth=truth,
        predictions=predictions,
        thresholds=thresholds,
        epoch=epoch,
        match_iou=match_iou,
    )
    payload = {
        "schema_version": 1,
        "status": "diagnostic_on_consumed_train_only_sets",
        "eligible_for_generation_gate": False,
        "reason": (
            f"All {len(image_ids)} images were already used for calibration "
            "or the failed audit; these metrics may guide a new "
            "preregistered experiment but cannot pass a gate."
        ),
        "experiment": experiment,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_split_manifest_sha256": split["manifest_sha256"],
        "images_read": len(image_ids),
        "previous_calibration_images_read": len(
            split["calibration_image_ids"]
        ),
        "previous_failed_audit_images_read": len(
            split["untouched_audit_image_ids"]
        ),
        "validation_images_read": 0,
        "test_images_read": 0,
        "threshold_grid": rows,
        "prior_calibration_threshold_grid": prior_calibration_rows,
        "failed_audit_threshold_grid": failed_audit_rows,
        "best_at_precision_0_80": _candidate(rows, precision_floor=0.80),
        "best_at_precision_0_85": _candidate(rows, precision_floor=0.85),
        "best_at_precision_0_90": _candidate(rows, precision_floor=0.90),
        "whole_image_generation_run": False,
    }
    paths["diagnosis"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler failure diagnosis",
        "",
        "- Evidence class: **consumed Train-only sets; not gate-eligible**",
        f"- Images read: **{len(image_ids)}**",
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image FLUX generations: **0**",
        "",
        "| Threshold | Precision | Recall | F1 | Median IoU |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        (
            f"| {row['threshold']:.3f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['median_matched_iou']:.4f} |"
        )
        for row in rows
    )
    lines.append("")
    paths["markdown"].write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("v1", "v2", "v3"), default="v1")
    arguments = parser.parse_args()
    main(arguments.experiment)
