"""Diagnose the failed v16 one-shot audit without new inference or pixels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import box_iou_xyxy
from src.synthetic.whole_image import canonical_mapping_sha256

TRAINING_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v16_training.json"
)
EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v16_audit_evidence.json"
)
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v16_split.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v16_audit_diagnosis.json"
)
MATCH_IOU = 0.50
DIAGNOSTIC_THRESHOLDS = (0.03, 0.035, 0.04, 0.05)


def _sha256(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _match_case(
    *,
    truth: Sequence[Sequence[float]],
    predictions: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    remaining = set(range(len(truth)))
    matched = []
    false_positives = []
    filtered = sorted(
        (
            {
                "score": float(row["score"]),
                "box": [float(value) for value in row["box"]],
            }
            for row in predictions
            if float(row["score"]) >= threshold
        ),
        key=lambda row: -float(row["score"]),
    )
    for prediction in filtered:
        if not remaining:
            false_positives.append(prediction)
            continue
        index, iou = max(
            (
                (
                    truth_index,
                    box_iou_xyxy(prediction["box"], truth[truth_index]),
                )
                for truth_index in remaining
            ),
            key=lambda row: row[1],
        )
        if iou >= MATCH_IOU:
            remaining.remove(index)
            matched.append(
                {
                    "truth_index": int(index),
                    "score": float(prediction["score"]),
                    "iou": float(iou),
                }
            )
        else:
            false_positives.append(prediction)
    return {
        "true_positives": len(matched),
        "false_positives": len(false_positives),
        "false_negatives": len(remaining),
        "matched": matched,
        "false_positive_predictions": false_positives,
        "missed_truth_indices": sorted(remaining),
    }


def _metrics(cases: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    matched_ious = []
    for case in cases:
        result = _match_case(
            truth=case["truth_boxes"],
            predictions=case["model_predictions"],
            threshold=threshold,
        )
        for key in ("true_positives", "false_positives", "false_negatives"):
            totals[key] += int(result[key])
        matched_ious.extend(float(row["iou"]) for row in result["matched"])
    true_positives = totals["true_positives"]
    false_positives = totals["false_positives"]
    false_negatives = totals["false_negatives"]
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    ordered_ious = sorted(matched_ious)
    midpoint = len(ordered_ious) // 2
    median_iou = (
        (
            ordered_ious[midpoint]
            if len(ordered_ious) % 2
            else (ordered_ious[midpoint - 1] + ordered_ious[midpoint]) / 2
        )
        if ordered_ious
        else 0.0
    )
    return {
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / max(precision + recall, 1e-12)
        ),
        "median_matched_iou": median_iou,
    }


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("v16 audit diagnosis already exists")
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if (
        training["status"] != "supervised_labeler_audit_failed"
        or evidence["status"] != "frozen_one_shot_audit_review_evidence"
        or training["audit_evidence_sha256"] != _sha256(EVIDENCE_PATH)
        or evidence["split_manifest_sha256"] != split["manifest_sha256"]
        or len(evidence["cases"]) != 48
        or int(evidence["validation_images_read"]) != 0
        or int(evidence["test_images_read"]) != 0
    ):
        raise RuntimeError("Frozen v16 audit evidence changed")

    threshold = float(evidence["score_threshold"])
    case_rows = []
    for case in evidence["cases"]:
        result = _match_case(
            truth=case["truth_boxes"],
            predictions=case["model_predictions"],
            threshold=threshold,
        )
        case_rows.append(
            {
                "cell": int(case["cell"]),
                "image_id": int(case["image_id"]),
                "truth_boxes": len(case["truth_boxes"]),
                "predictions": sum(
                    float(row["score"]) >= threshold
                    for row in case["model_predictions"]
                ),
                **result,
            }
        )
    false_positive_rows = sorted(
        (
            {
                "cell": int(row["cell"]),
                "image_id": int(row["image_id"]),
                "truth_boxes": int(row["truth_boxes"]),
                "false_positives": int(row["false_positives"]),
                "false_negatives": int(row["false_negatives"]),
                "scores": [
                    float(item["score"])
                    for item in row["false_positive_predictions"]
                ],
            }
            for row in case_rows
            if int(row["false_positives"]) > 0
        ),
        key=lambda row: (
            -int(row["false_positives"]),
            int(row["cell"]),
        ),
    )
    miss_rows = sorted(
        (
            {
                "cell": int(row["cell"]),
                "image_id": int(row["image_id"]),
                "truth_boxes": int(row["truth_boxes"]),
                "false_positives": int(row["false_positives"]),
                "false_negatives": int(row["false_negatives"]),
            }
            for row in case_rows
            if int(row["false_negatives"]) > 0
        ),
        key=lambda row: (
            -int(row["false_negatives"]),
            int(row["cell"]),
        ),
    )
    threshold_diagnosis = [
        _metrics(evidence["cases"], candidate)
        for candidate in DIAGNOSTIC_THRESHOLDS
    ]
    report = {
        "schema_version": 1,
        "status": "v16_failed_audit_diagnosed_without_new_inference",
        "experiment_id": "supervised_labeler_v16",
        "training_report_path": TRAINING_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "training_report_file_sha256": _sha256(TRAINING_PATH),
        "audit_evidence_path": EVIDENCE_PATH.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "audit_evidence_file_sha256": _sha256(EVIDENCE_PATH),
        "split_manifest_sha256": str(split["manifest_sha256"]),
        "checkpoint_sha256": str(training["checkpoint_sha256"]),
        "frozen_score_threshold": threshold,
        "audit_metrics": training["audit_metrics"],
        "failed_checks": [
            key for key, value in training["checks"].items() if not value
        ],
        "error_case_summary": {
            "cells_with_false_positives": len(false_positive_rows),
            "cells_with_false_negatives": len(miss_rows),
            "false_positives_on_empty_gt": sum(
                int(row["false_positives"])
                for row in false_positive_rows
                if int(row["truth_boxes"]) == 0
            ),
            "false_positives_on_positive_gt": sum(
                int(row["false_positives"])
                for row in false_positive_rows
                if int(row["truth_boxes"]) > 0
            ),
        },
        "false_positive_cases": false_positive_rows,
        "miss_cases": miss_rows,
        "diagnostic_thresholds": threshold_diagnosis,
        "threshold_policy": (
            "Diagnostic only. The one-shot audit cannot select a new threshold "
            "after outcomes are visible."
        ),
        "scope": {
            "model_inference_run": False,
            "source_image_pixels_read": 0,
            "audit_evidence_rows_read": 48,
            "validation_images_read": 0,
            "test_images_read": 0,
            "whole_image_generation_run": False,
        },
        "generation_allowed": False,
    }
    report["report_sha256"] = canonical_mapping_sha256(report)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed_checks": report["failed_checks"],
                "error_case_summary": report["error_case_summary"],
                "top_false_positive_cases": false_positive_rows[:10],
                "miss_cases": miss_rows,
                "diagnostic_thresholds": threshold_diagnosis,
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
