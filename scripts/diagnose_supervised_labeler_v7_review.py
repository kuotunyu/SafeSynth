"""Diagnose owner-reported v7 misses on already revealed Train-only evidence."""

from __future__ import annotations

import gc
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.train_supervised_labeler import (
    _aggregate,
    _build_datasets,
    _evaluation_collate,
    _predict,
    _sha256,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import box_iou_xyxy
from src.synthetic.supervised_labeler import filter_prediction_geometry

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v7.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v7_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v7_training.json"
REVIEW_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v7_human_review.json"
AUDIT_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v7_audit_evidence.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v7_review_diagnosis.json"
)
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v7_review_diagnosis.md"
)
SCORE_FLOOR = 0.001
THRESHOLDS = (
    0.001,
    0.002,
    0.005,
    0.010,
    0.015,
    0.020,
    0.023,
    0.025,
    0.030,
    0.040,
    0.050,
)


def _geometry_kept(
    prediction: tuple[float, Sequence[float]],
    *,
    width: int,
    height: int,
    settings: dict[str, Any],
) -> bool:
    return bool(
        filter_prediction_geometry(
            [prediction],
            image_width=width,
            image_height=height,
            max_relative_area=float(settings["max_relative_area"]),
            max_relative_height=float(settings["max_relative_height"]),
            min_aspect_ratio=float(settings["min_aspect_ratio"]),
            max_aspect_ratio=float(settings["max_aspect_ratio"]),
        )
    )


def _unmatched_truth_indices(
    truth: Sequence[Sequence[float]],
    predictions: Sequence[tuple[float, Sequence[float]]],
    *,
    threshold: float,
    match_iou: float,
) -> list[int]:
    unmatched = set(range(len(truth)))
    for score, box in sorted(predictions, key=lambda row: -float(row[0])):
        if float(score) < threshold or not unmatched:
            continue
        best_index = max(
            unmatched,
            key=lambda index: box_iou_xyxy(truth[index], box),
        )
        if box_iou_xyxy(truth[best_index], box) >= match_iou:
            unmatched.remove(best_index)
    return sorted(unmatched)


def _miss_reason(
    *,
    candidates: Sequence[tuple[float, float, list[float], bool]],
    threshold: float,
    match_iou: float,
) -> str:
    matching = [row for row in candidates if row[0] >= match_iou]
    if not matching:
        return "no_matching_localization"
    qualifying = [row for row in matching if row[1] >= threshold]
    if qualifying and not any(row[3] for row in qualifying):
        return "removed_by_frozen_geometry_filter"
    if not qualifying:
        return "matching_box_below_frozen_score_threshold"
    return "matching_box_consumed_by_another_truth"


def main() -> None:
    if OUTPUT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("v7 owner-review diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("v7 owner-review diagnosis requires CUDA")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    sidecar = json.loads(AUDIT_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        report["status"] != "supervised_labeler_audit_passed"
        or review["status"] != "rejected_by_kuotunyu"
        or not review["problem_cells"]
        or int(report["validation_images_read"]) != 0
        or int(report["test_images_read"]) != 0
    ):
        raise RuntimeError("Expected a numeric pass rejected by owner review")

    (
        config,
        split,
        _,
        train_images,
        _,
        _,
        _,
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
    loader = DataLoader(
        consumed_audit,
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
    geometry = config["postprocessing"]
    filtered_predictions = {}
    for image_id in image_ids:
        image = train_images[int(image_id)]
        filtered_predictions[int(image_id)] = filter_prediction_geometry(
            raw_predictions[int(image_id)],
            image_width=int(image["width"]),
            image_height=int(image["height"]),
            max_relative_area=float(geometry["max_relative_area"]),
            max_relative_height=float(geometry["max_relative_height"]),
            min_aspect_ratio=float(geometry["min_aspect_ratio"]),
            max_aspect_ratio=float(geometry["max_aspect_ratio"]),
        )

    frozen_threshold = float(report["best_calibration"]["threshold"])
    match_iou = float(config["calibration"]["match_iou"])
    reproduced = _aggregate(
        image_ids=image_ids,
        truth=truth,
        predictions=filtered_predictions,
        threshold=frozen_threshold,
        match_iou=match_iou,
    )
    for metric in ("precision", "recall", "f1", "median_matched_iou"):
        if abs(
            float(reproduced[metric]) - float(report["audit_metrics"][metric])
        ) > 1e-12:
            raise RuntimeError(f"v7 audit reproduction changed: {metric}")

    cases_by_cell = {int(case["cell"]): case for case in sidecar["cases"]}
    problem_cases = []
    reason_counts: dict[str, int] = {}
    for cell in review["problem_cells"]:
        case = cases_by_cell[int(cell)]
        image_id = int(case["image_id"])
        image = train_images[image_id]
        unmatched = _unmatched_truth_indices(
            truth[image_id],
            filtered_predictions[image_id],
            threshold=frozen_threshold,
            match_iou=match_iou,
        )
        misses = []
        for truth_index in unmatched:
            truth_box = truth[image_id][truth_index]
            candidates = [
                (
                    box_iou_xyxy(truth_box, box),
                    float(score),
                    [float(value) for value in box],
                    _geometry_kept(
                        (score, box),
                        width=int(image["width"]),
                        height=int(image["height"]),
                        settings=geometry,
                    ),
                )
                for score, box in raw_predictions[image_id]
            ]
            best_iou, best_score, best_box, geometry_kept = max(
                candidates,
                key=lambda row: (row[0], row[1]),
                default=(0.0, 0.0, [], False),
            )
            reason = _miss_reason(
                candidates=candidates,
                threshold=frozen_threshold,
                match_iou=match_iou,
            )
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            misses.append(
                {
                    "truth_index": truth_index,
                    "truth_box": [float(value) for value in truth_box],
                    "best_raw_candidate_box": best_box,
                    "best_raw_candidate_iou": best_iou,
                    "best_raw_candidate_score": best_score,
                    "best_raw_candidate_geometry_kept": geometry_kept,
                    "reason": reason,
                }
            )
        problem_cases.append(
            {
                "cell": int(cell),
                "image_id": image_id,
                "owner_category": "missed_helmet",
                "unmatched_truth_count": len(unmatched),
                "misses": misses,
            }
        )

    threshold_grid = [
        {
            "threshold": threshold,
            **_aggregate(
                image_ids=image_ids,
                truth=truth,
                predictions=filtered_predictions,
                threshold=threshold,
                match_iou=match_iou,
            ),
        }
        for threshold in THRESHOLDS
    ]
    payload = {
        "schema_version": 1,
        "status": "diagnostic_on_revealed_v7_train_only_audit",
        "eligible_for_generation_gate": False,
        "reason": (
            "The 48 audit images were revealed to kuotunyu. This diagnosis "
            "may motivate a new preregistered experiment but cannot pass a gate."
        ),
        "experiment_id": "supervised_labeler_v7",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_split_manifest_sha256": split["manifest_sha256"],
        "review_evidence_sha256": review["evidence_sha256"],
        "problem_cells": review["problem_cells"],
        "problem_instances": sum(
            int(case["unmatched_truth_count"]) for case in problem_cases
        ),
        "reason_counts": reason_counts,
        "problem_cases": problem_cases,
        "frozen_threshold": frozen_threshold,
        "score_floor": SCORE_FLOOR,
        "threshold_grid": threshold_grid,
        "reproduced_frozen_audit_metrics": reproduced,
        "revealed_audit_images_read": len(image_ids),
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
        "# Supervised labeler v7 owner-review diagnosis",
        "",
        "- Evidence class: **revealed Train-only audit; not gate-eligible**",
        f"- Owner problem cells: **{', '.join(str(v) for v in review['problem_cells'])}**",
        f"- Unmatched helmet instances: **{payload['problem_instances']}**",
        f"- Reason counts: **{json.dumps(reason_counts, sort_keys=True)}**",
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image generations: **0**",
        "",
        "| Cell | Image | Misses | Best raw IoU / score / reason |",
        "|---:|---:|---:|---|",
    ]
    for case in problem_cases:
        details = "; ".join(
            (
                f"{float(miss['best_raw_candidate_iou']):.3f} / "
                f"{float(miss['best_raw_candidate_score']):.4f} / "
                f"{miss['reason']}"
            )
            for miss in case["misses"]
        )
        lines.append(
            f"| {case['cell']} | {case['image_id']} | "
            f"{case['unmatched_truth_count']} | {details} |"
        )
    lines.extend(
        [
            "",
            "## Revealed audit threshold grid",
            "",
            "| Threshold | Precision | Recall | F1 | Median IoU |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_grid:
        lines.append(
            f"| {float(row['threshold']):.4f} | "
            f"{float(row['precision']):.4f} | "
            f"{float(row['recall']):.4f} | "
            f"{float(row['f1']):.4f} | "
            f"{float(row['median_matched_iou']):.4f} |"
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
