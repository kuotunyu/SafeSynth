"""Diagnose the frozen v21 numeric-audit failure without new inference."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import yaml

from scripts.diagnose_supervised_labeler_v20_numeric_failure import (
    _prediction_diagnosis,
    _score_summary,
    _sha256,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v21.yaml"
TRAINING_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v21_training.json"
)
AUDIT_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v21_audit_evidence.json"
)
AUDIT_MANIFEST_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v21_adjudicated_audit.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v21_numeric_failure_diagnosis.json"
)


def build_diagnosis(
    *,
    config: dict[str, Any],
    training_report: dict[str, Any],
    evidence: dict[str, Any],
    audit_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return deterministic diagnostics from the frozen v21 box evidence."""

    selected_by_cell = {
        int(row["audit_cell"]): row
        for row in audit_manifest["selected_cases"]
    }
    if set(selected_by_cell) != set(range(1, 49)):
        raise RuntimeError("v21 audit manifest cell mapping changed")

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
            raise RuntimeError(f"v21 audit identity changed at cell {cell}")
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
            raise RuntimeError(f"v21 aggregate {key} changed")

    fp_cells = [
        row for row in cell_rows if int(row["false_positives"]) > 0
    ]
    fn_cells = [
        row for row in cell_rows if int(row["false_negatives"]) > 0
    ]
    diagnosis = {
        "schema_version": 1,
        "status": "v21_numeric_failure_diagnosed_without_new_inference",
        "experiment_id": "supervised_labeler_v21",
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
            "recall_gate_failure": (
                "The frozen checkpoint missed 34 of 104 worn helmeted heads, "
                "so the preregistered 0.70 recall gate failed at 0.6731."
            ),
            "precision_improved_from_v20": (
                "False positives fell from 29 in v20 to 11 in v21 and "
                "precision rose from 0.7583 to 0.8642."
            ),
            "rendering_bug": False,
            "threshold_retuning_allowed": False,
        },
        "fresh_round_constraint": (
            "Do not alter the v21 threshold, checkpoint, or audit decision. "
            "A v22 intervention may use the now-revealed v21 audit examples "
            "for explicit positive and hard-negative replay only after a "
            "new independent v22 audit is frozen."
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
        raise RuntimeError("v21 numeric-failure diagnosis already exists")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    training_report = json.loads(
        TRAINING_REPORT_PATH.read_text(encoding="utf-8")
    )
    evidence = json.loads(AUDIT_EVIDENCE_PATH.read_text(encoding="utf-8"))
    audit_manifest = json.loads(
        AUDIT_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if (
        config["status"] != "numeric_audit_failed_recall"
        or training_report["status"] != "supervised_labeler_audit_failed"
        or evidence["status"] != "frozen_one_shot_audit_review_evidence"
        or training_report["audit_evidence_sha256"]
        != _sha256(AUDIT_EVIDENCE_PATH)
        or config["numeric_audit_outcome"]["report_file_sha256"]
        != _sha256(TRAINING_REPORT_PATH)
    ):
        raise RuntimeError("Frozen v21 numeric-audit inputs changed")

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
                "per_stratum": diagnosis["per_stratum"],
                "report_sha256": diagnosis["report_sha256"],
                "status": diagnosis["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
