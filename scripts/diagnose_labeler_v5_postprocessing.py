"""Test fixed geometry filters on the consumed v5 Train-only sets."""

from __future__ import annotations

import gc
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.diagnose_labeler_postprocessing import (
    MAX_RELATIVE_AREAS,
    MAX_RELATIVE_HEIGHTS,
    _grid,
    _metrics_for_candidate,
    select_geometry_candidate,
)
from scripts.diagnose_supervised_labeler_failure import _experiment_paths
from scripts.train_supervised_labeler import (
    _build_datasets,
    _evaluation_collate,
    _predict,
    _sha256,
)
from src.data.paths import PROJECT_ROOT

REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v5_postprocessing_diagnosis.json"
)
MARKDOWN_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v5_postprocessing_diagnosis.md"
)
SCORE_THRESHOLDS = tuple(
    [0.015, 0.020]
    + [round(0.020 + index * 0.001, 3) for index in range(1, 11)]
    + [0.035]
)


def main() -> None:
    if REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("v5 postprocessing diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("v5 postprocessing diagnosis requires CUDA")

    paths = _experiment_paths("v5")
    training = json.loads(paths["training"].read_text(encoding="utf-8"))
    if (
        training["status"] != "supervised_labeler_audit_failed"
        or int(training["untouched_audit_images_read"]) != 48
        or int(training["validation_images_read"]) != 0
        or int(training["test_images_read"]) != 0
    ):
        raise RuntimeError("v5 is not a completed failed Train-only audit")
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
    checkpoint = Path(training["checkpoint_path"])
    if _sha256(checkpoint / "model.safetensors") != training["checkpoint_sha256"]:
        raise RuntimeError("v5 checkpoint changed")

    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to("cuda")
    loader = DataLoader(
        ConcatDataset([calibration, consumed_audit]),
        batch_size=int(config["optimization"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=partial(_evaluation_collate, processor),
    )
    _, truth, predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=min(SCORE_THRESHOLDS),
    )
    calibration_ids = [
        int(value) for value in split["calibration_image_ids"]
    ]
    audit_ids = [
        int(value) for value in split["untouched_audit_image_ids"]
    ]
    match_iou = float(config["calibration"]["match_iou"])
    calibration_grid = _grid(
        image_ids=calibration_ids,
        truth=truth,
        predictions=predictions,
        image_records=image_records,
        match_iou=match_iou,
        score_thresholds=SCORE_THRESHOLDS,
    )
    candidates = {
        str(floor): select_geometry_candidate(
            calibration_grid,
            precision_floor=floor,
        )
        for floor in (0.80, 0.82, 0.85)
    }
    selected = candidates["0.8"]
    if selected is None:
        raise RuntimeError("No v5 geometry candidate met precision 0.80")
    audit_metrics_by_floor = {
        floor: (
            _metrics_for_candidate(
                candidate=candidate,
                image_ids=audit_ids,
                truth=truth,
                predictions=predictions,
                image_records=image_records,
                match_iou=match_iou,
            )
            if candidate is not None
            else None
        )
        for floor, candidate in candidates.items()
    }
    audit_metrics = audit_metrics_by_floor["0.8"]
    payload = {
        "schema_version": 1,
        "status": "diagnostic_on_consumed_train_only_sets",
        "eligible_for_generation_gate": False,
        "reason": (
            "Both calibration and audit were already consumed. This evidence "
            "may guide a new preregistration but cannot pass a gate."
        ),
        "selection_source": "v5 prior calibration only",
        "score_thresholds": list(SCORE_THRESHOLDS),
        "max_relative_areas": list(MAX_RELATIVE_AREAS),
        "max_relative_heights": list(MAX_RELATIVE_HEIGHTS),
        "calibration_candidates_by_precision_floor": candidates,
        "failed_audit_metrics_by_calibration_precision_floor": (
            audit_metrics_by_floor
        ),
        "selected_at_calibration_precision_0_80": selected,
        "selected_candidate_failed_audit_metrics": audit_metrics,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "source_split_manifest_sha256": split["manifest_sha256"],
        "calibration_images_read": len(calibration_ids),
        "failed_audit_images_read": len(audit_ids),
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
        "# Supervised labeler v5 postprocessing diagnosis",
        "",
        "- Evidence class: **consumed Train-only sets; not gate-eligible**",
        "- Selection source: **v5 prior calibration only**",
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image FLUX generations: **0**",
        "",
        "## Selected calibration candidate",
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
        "## Same candidate on the consumed failed audit",
        "",
        (
            f"- Precision / recall / F1: "
            f"**{float(audit_metrics['precision']):.4f} / "
            f"{float(audit_metrics['recall']):.4f} / "
            f"{float(audit_metrics['f1']):.4f}**"
        ),
        (
            f"- Median matched IoU: "
            f"**{float(audit_metrics['median_matched_iou']):.4f}**"
        ),
        "",
    ]
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
