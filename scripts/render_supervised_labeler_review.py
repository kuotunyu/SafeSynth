"""Render every consumed v6 audit image for the required human review."""

from __future__ import annotations

import gc
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.train_supervised_labeler import (
    FIGURE_PATH,
    REPORT_PATH,
    _aggregate,
    _build_datasets,
    _evaluation_collate,
    _predict,
    _render_audit,
    _sha256,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Human-review rendering requires CUDA")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if (
        report["status"] != "supervised_labeler_audit_passed"
        or int(report["untouched_audit_images_read"]) != 48
        or int(report["validation_images_read"]) != 0
        or int(report["test_images_read"]) != 0
    ):
        raise RuntimeError("Expected one passed v6 Train-only audit")
    (
        config,
        split,
        _,
        train_images,
        image_root,
        _,
        _,
        audit,
    ) = _build_datasets()
    if len(split["untouched_audit_image_ids"]) != 48:
        raise RuntimeError("Expected exactly 48 frozen audit images")
    checkpoint = Path(report["checkpoint_path"])
    if _sha256(checkpoint / "model.safetensors") != report["checkpoint_sha256"]:
        raise RuntimeError("Passed v6 checkpoint changed")
    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to("cuda")
    loader = DataLoader(
        audit,
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
        score_floor=min(config["calibration"]["score_thresholds"]),
        geometry_filter=config["postprocessing"],
    )
    best = report["best_calibration"]
    metrics = _aggregate(
        image_ids=image_ids,
        truth=truth,
        predictions=predictions,
        threshold=float(best["threshold"]),
        match_iou=float(config["calibration"]["match_iou"]),
    )
    for key in ("precision", "recall", "median_matched_iou"):
        if abs(float(metrics[key]) - float(report["audit_metrics"][key])) > 1e-12:
            raise RuntimeError(f"Review rerun changed the passed {key}")
    _render_audit(
        rows=image_ids,
        train_images=train_images,
        image_root=image_root,
        truth=truth,
        predictions=predictions,
        threshold=float(best["threshold"]),
    )
    print(
        json.dumps(
            {
                "status": "human_review_sheet_rendered",
                "figure": str(FIGURE_PATH),
                "images": len(image_ids),
                "threshold": best["threshold"],
                "metrics_reproduced": metrics,
                "validation_images_read": 0,
                "test_images_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
