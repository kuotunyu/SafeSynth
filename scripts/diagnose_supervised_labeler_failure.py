"""Diagnose the failed labeler on already-consumed Train-only evaluation sets."""

from __future__ import annotations

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

DIAGNOSIS_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_failure_diagnosis.json"
)
V1_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler.yaml"
V1_SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_split.json"
V1_REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_training.json"
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_failure_diagnosis.md"
)


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


def _candidate(
    rows: list[dict[str, Any]],
    *,
    precision_floor: float,
) -> dict[str, Any] | None:
    return select_calibration_candidate(
        rows,
        precision_floor=precision_floor,
    )


def main() -> None:
    if DIAGNOSIS_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("Failure diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Failure diagnosis requires CUDA")
    report = json.loads(V1_REPORT_PATH.read_text(encoding="utf-8"))
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
        config_path=V1_CONFIG_PATH,
        split_path=V1_SPLIT_PATH,
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
    image_ids, truth, predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=min(diagnostic_thresholds()),
    )
    rows = [
        {
            "epoch": int(report["best_calibration"]["epoch"]),
            "threshold": threshold,
            **_aggregate(
                image_ids=image_ids,
                truth=truth,
                predictions=predictions,
                threshold=threshold,
                match_iou=float(config["calibration"]["match_iou"]),
            ),
        }
        for threshold in diagnostic_thresholds()
    ]
    payload = {
        "schema_version": 1,
        "status": "diagnostic_on_consumed_train_only_sets",
        "eligible_for_generation_gate": False,
        "reason": (
            "All 144 images were already used for calibration or the failed "
            "audit; these metrics may guide a new preregistered experiment "
            "but cannot pass a gate."
        ),
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
        "best_at_precision_0_80": _candidate(rows, precision_floor=0.80),
        "best_at_precision_0_85": _candidate(rows, precision_floor=0.85),
        "whole_image_generation_run": False,
    }
    DIAGNOSIS_PATH.write_text(
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
