"""Diagnose the frozen v20 numeric-audit failure without new inference."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import box_iou_xyxy
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v20.yaml"
TRAINING_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v20_training.json"
)
AUDIT_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v20_audit_evidence.json"
)
AUDIT_MANIFEST_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v20_adjudicated_audit.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v20_numeric_failure_diagnosis.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_diagnosis(
    *,
    truth_boxes: Sequence[Sequence[float]],
    predictions: Sequence[dict[str, Any]],
    score_threshold: float,
    match_iou: float,
) -> dict[str, Any]:
    remaining = set(range(len(truth_boxes)))
    matched: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    filtered = sorted(
        (
            {
                "box": [float(value) for value in row["box"]],
                "score": float(row["score"]),
            }
            for row in predictions
            if float(row["score"]) >= score_threshold
        ),
        key=lambda row: -row["score"],
    )
    for prediction in filtered:
        all_ious = [
            box_iou_xyxy(prediction["box"], truth_box)
            for truth_box in truth_boxes
        ]
        if remaining:
            best_index = max(
                remaining,
                key=lambda index: all_ious[index],
            )
            best_remaining_iou = float(all_ious[best_index])
        else:
            best_index = -1
            best_remaining_iou = 0.0
        if remaining and best_remaining_iou >= match_iou:
            remaining.remove(best_index)
            matched.append(
                {
                    "ground_truth_index": best_index,
                    "iou": best_remaining_iou,
                    "score": prediction["score"],
                }
            )
            continue

        best_all_iou = float(max(all_ious, default=0.0))
        if not truth_boxes:
            category = "empty_gt_false_positive"
        elif best_all_iou >= match_iou:
            category = "duplicate_detection"
        elif best_all_iou >= 0.10:
            category = "localization_false_positive"
        else:
            category = "semantic_false_positive"
        false_positives.append(
            {
                "best_iou_to_any_truth": best_all_iou,
                "category": category,
                "score": prediction["score"],
            }
        )

    return {
        "false_negatives": len(remaining),
        "false_positive_details": false_positives,
        "false_positives": len(false_positives),
        "matched": matched,
        "true_positives": len(matched),
        "unmatched_truth_indices": sorted(remaining),
    }


def _score_summary(scores: Sequence[float]) -> dict[str, float | int]:
    if not scores:
        return {"count": 0}
    return {
        "count": len(scores),
        "maximum": max(scores),
        "median": statistics.median(scores),
        "minimum": min(scores),
    }


def build_diagnosis(
    *,
    config: dict[str, Any],
    training_report: dict[str, Any],
    evidence: dict[str, Any],
    audit_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return deterministic diagnostics from the already-frozen box evidence."""

    selected_by_cell = {
        int(row["audit_cell"]): row
        for row in audit_manifest["selected_cases"]
    }
    if set(selected_by_cell) != set(range(1, 49)):
        raise RuntimeError("v20 audit manifest cell mapping changed")

    score_threshold = float(evidence["score_threshold"])
    match_iou = float(config["calibration"]["match_iou"])
    per_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    false_positive_types: Counter[str] = Counter()
    false_positive_scores: list[float] = []
    matched_scores: list[float] = []
    cell_rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()

    for case in evidence["cases"]:
        cell = int(case["cell"])
        selected = selected_by_cell[cell]
        if int(case["image_id"]) != int(selected["image_id"]):
            raise RuntimeError(f"v20 audit identity changed at cell {cell}")
        metrics = _prediction_diagnosis(
            truth_boxes=case["truth_boxes"],
            predictions=case["model_predictions"],
            score_threshold=score_threshold,
            match_iou=match_iou,
        )
        stratum = str(selected["stratum"])
        row = {
            "cell": cell,
            "false_negatives": metrics["false_negatives"],
            "false_positive_details": metrics["false_positive_details"],
            "false_positives": metrics["false_positives"],
            "image_id": int(case["image_id"]),
            "prediction_count": len(case["model_predictions"]),
            "stratum": stratum,
            "true_positives": metrics["true_positives"],
            "truth_box_count": len(case["truth_boxes"]),
            "unmatched_truth_indices": metrics["unmatched_truth_indices"],
        }
        cell_rows.append(row)
        per_stratum[stratum]["images"] += 1
        per_stratum[stratum]["truth_boxes"] += len(case["truth_boxes"])
        per_stratum[stratum]["predictions"] += len(case["model_predictions"])
        for key in ("true_positives", "false_positives", "false_negatives"):
            value = int(metrics[key])
            totals[key] += value
            per_stratum[stratum][key] += value
        for detail in metrics["false_positive_details"]:
            false_positive_types[str(detail["category"])] += 1
            false_positive_scores.append(float(detail["score"]))
        matched_scores.extend(
            float(match["score"]) for match in metrics["matched"]
        )

    expected = training_report["audit_metrics"]
    for key in ("true_positives", "false_positives", "false_negatives"):
        if totals[key] != int(expected[key]):
            raise RuntimeError(f"v20 aggregate {key} changed")

    fp_cells = [
        row for row in cell_rows if int(row["false_positives"]) > 0
    ]
    fn_cells = [
        row for row in cell_rows if int(row["false_negatives"]) > 0
    ]
    diagnosis = {
        "schema_version": 1,
        "status": "v20_numeric_failure_diagnosed_without_new_inference",
        "experiment_id": "supervised_labeler_v20",
        "scope": "frozen_one_shot_audit_box_evidence_only",
        "source_files": {
            "audit_evidence_file_sha256": _sha256(AUDIT_EVIDENCE_PATH),
            "audit_manifest_file_sha256": _sha256(AUDIT_MANIFEST_PATH),
            "training_report_file_sha256": _sha256(TRAINING_REPORT_PATH),
        },
        "score_threshold": score_threshold,
        "match_iou": match_iou,
        "aggregate": {
            "false_negatives": totals["false_negatives"],
            "false_positives": totals["false_positives"],
            "true_positives": totals["true_positives"],
        },
        "false_positive_type_counts": dict(
            sorted(false_positive_types.items())
        ),
        "false_positive_score_summary": _score_summary(
            false_positive_scores
        ),
        "matched_prediction_score_summary": _score_summary(matched_scores),
        "per_stratum": {
            stratum: dict(sorted(counts.items()))
            for stratum, counts in sorted(per_stratum.items())
        },
        "false_positive_cells": sorted(
            fp_cells,
            key=lambda row: (-int(row["false_positives"]), int(row["cell"])),
        ),
        "false_negative_cells": sorted(
            fn_cells,
            key=lambda row: (-int(row["false_negatives"]), int(row["cell"])),
        ),
        "root_cause_summary": {
            "precision_gate_failure": (
                "The frozen checkpoint produced 29 false positives against "
                "91 true positives, so the preregistered 0.85 precision gate "
                "failed at 0.7583."
            ),
            "rendering_bug": False,
            "threshold_retuning_allowed": False,
        },
        "fresh_round_constraint": (
            "Do not alter the v20 threshold, checkpoint, or audit decision. "
            "A v21 intervention may use the now-revealed v20 audit examples "
            "for explicit positive and hard-negative replay only after a "
            "new independent v21 audit is frozen."
        ),
        "diagnosis_model_inference_run": False,
        "source_image_pixels_read": 0,
        "sealed_reserve_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    diagnosis["report_sha256"] = canonical_mapping_sha256(diagnosis)
    return diagnosis


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("v20 numeric-failure diagnosis already exists")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    training_report = json.loads(
        TRAINING_REPORT_PATH.read_text(encoding="utf-8")
    )
    evidence = json.loads(AUDIT_EVIDENCE_PATH.read_text(encoding="utf-8"))
    audit_manifest = json.loads(
        AUDIT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if (
        config["status"] != "numeric_audit_failed_precision"
        or training_report["status"] != "supervised_labeler_audit_failed"
        or evidence["status"] != "frozen_one_shot_audit_review_evidence"
        or training_report["audit_evidence_sha256"]
        != _sha256(AUDIT_EVIDENCE_PATH)
        or config["numeric_audit_outcome"]["report_file_sha256"]
        != _sha256(TRAINING_REPORT_PATH)
    ):
        raise RuntimeError("Frozen v20 numeric-audit inputs changed")

    diagnosis = build_diagnosis(
        config=config,
        training_report=training_report,
        evidence=evidence,
        audit_manifest=audit_manifest,
    )
    OUTPUT_PATH.write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "aggregate": diagnosis["aggregate"],
                "false_negative_cells": [
                    row["cell"]
                    for row in diagnosis["false_negative_cells"]
                ],
                "false_positive_cells": [
                    row["cell"]
                    for row in diagnosis["false_positive_cells"]
                ],
                "false_positive_type_counts": diagnosis[
                    "false_positive_type_counts"
                ],
                "report_sha256": diagnosis["report_sha256"],
                "status": diagnosis["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
