"""Record kuotunyu's final review of the frozen v19 model audit pages."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import greedy_detection_metrics
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v19.yaml"
MODEL_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v19_training.json"
)
MODEL_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v19_audit_evidence.json"
)
MODEL_REVIEW_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v19_model_review_manifest.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v19_model_human_review.json"
)
DIAGNOSIS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v19_review_diagnosis.json"
)
MISSED_CELLS = [10, 14, 23, 28, 48]
FALSE_POSITIVE_CELLS = [22, 25]
EXPECTED_IMAGE_IDS = {
    10: 3117,
    14: 4924,
    22: 2910,
    23: 3651,
    25: 1241,
    28: 118,
    48: 4452,
}
MATCH_IOU = 0.50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_manifest() -> dict[str, Any]:
    manifest = json.loads(
        MODEL_REVIEW_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(manifest)
    embedded_sha = str(canonical.pop("manifest_sha256", ""))
    if (
        canonical_mapping_sha256(canonical) != embedded_sha
        or manifest.get("status")
        != "supervised_labeler_v19_model_review_pages_frozen"
        or int(manifest.get("reviewed_images", -1)) != 48
        or manifest.get("render_model_inference_run") is not False
    ):
        raise RuntimeError("Frozen v19 model-review manifest changed")
    return manifest


def _evidence_by_cell(evidence: dict[str, Any]) -> dict[int, dict[str, Any]]:
    if len(evidence.get("cases", [])) != 48:
        raise RuntimeError("Frozen v19 model evidence changed")
    by_cell = {int(row["cell"]): row for row in evidence["cases"]}
    if {
        cell: int(by_cell[cell]["image_id"]) for cell in EXPECTED_IMAGE_IDS
    } != EXPECTED_IMAGE_IDS:
        raise RuntimeError("Frozen v19 reviewed cells changed")
    return by_cell


def build_review(
    *,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    reviewed_on: str,
) -> dict[str, Any]:
    """Build the exact seven-cell owner rejection."""

    date.fromisoformat(reviewed_on)
    if manifest["source_audit_evidence_sha256"] != _sha256(
        MODEL_EVIDENCE_PATH
    ):
        raise RuntimeError("Frozen v19 model evidence changed")
    _evidence_by_cell(evidence)
    problem_cells = sorted(MISSED_CELLS + FALSE_POSITIVE_CELLS)
    problem_cases = [
        {
            "cell": cell,
            "image_id": EXPECTED_IMAGE_IDS[cell],
            "category": (
                "model_missed_helmeted_head"
                if cell in MISSED_CELLS
                else "model_false_positive"
            ),
        }
        for cell in problem_cells
    ]
    review = {
        "schema_version": 1,
        "status": "rejected_by_kuotunyu",
        "experiment_id": "supervised_labeler_v19",
        "reviewed_by": "kuotunyu",
        "reviewed_on": reviewed_on,
        "decision": "reject",
        "label_semantics": str(manifest["label_semantics"]),
        "problem_count": len(problem_cells),
        "problem_cells": problem_cells,
        "categories": {
            "model_false_positive_cells": FALSE_POSITIVE_CELLS,
            "model_missed_helmeted_head_cells": MISSED_CELLS,
            "ambiguous_dataset_gt_quarantine_cells": [],
        },
        "problem_cases": problem_cases,
        "accepted_exceptions": [],
        "review_note": (
            "kuotunyu identified missed worn helmeted heads in cells 10, 14, "
            "23, 28, and 48, plus magenta predictions on background, faces, "
            "logos, or unworn helmets in cells 22 and 25. All other reviewed "
            "cells were accepted. The required zero-problem visual gate "
            "therefore failed."
        ),
        "numeric_audit_status": "passed_but_owner_visual_gate_failed",
        "numeric_audit_report_path": str(
            MODEL_REPORT_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "numeric_audit_report_file_sha256": _sha256(MODEL_REPORT_PATH),
        "model_evidence_path": str(
            MODEL_EVIDENCE_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "model_evidence_file_sha256": _sha256(MODEL_EVIDENCE_PATH),
        "model_review_manifest_path": str(
            MODEL_REVIEW_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "model_review_manifest_file_sha256": _sha256(
            MODEL_REVIEW_MANIFEST_PATH
        ),
        "model_review_manifest_sha256": str(manifest["manifest_sha256"]),
        "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
        "score_threshold": float(manifest["score_threshold"]),
        "pages": manifest["pages"],
        "generation_allowed": False,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    review["review_sha256"] = canonical_mapping_sha256(review)
    return review


def build_diagnosis(
    *,
    evidence: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Diagnose all reported cells from frozen boxes without new inference."""

    by_cell = _evidence_by_cell(evidence)
    threshold = float(evidence["score_threshold"])
    cell_diagnoses: list[dict[str, Any]] = []
    aggregate = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    for cell_number in sorted(MISSED_CELLS + FALSE_POSITIVE_CELLS):
        cell = by_cell[cell_number]
        predictions = [
            (float(row["score"]), row["box"])
            for row in cell["model_predictions"]
        ]
        metrics = greedy_detection_metrics(
            cell["truth_boxes"],
            predictions,
            score_threshold=threshold,
            match_iou=MATCH_IOU,
        )
        for key in aggregate:
            aggregate[key] += int(metrics[key])
        cell_diagnoses.append(
            {
                "cell": cell_number,
                "image_id": int(cell["image_id"]),
                "owner_category": (
                    "model_missed_helmeted_head"
                    if cell_number in MISSED_CELLS
                    else "model_false_positive"
                ),
                "truth_box_count": len(cell["truth_boxes"]),
                "model_prediction_count": len(cell["model_predictions"]),
                "true_positives": int(metrics["true_positives"]),
                "false_positives": int(metrics["false_positives"]),
                "false_negatives": int(metrics["false_negatives"]),
                "prediction_scores": [
                    float(row["score"])
                    for row in cell["model_predictions"]
                ],
                "rendering_bug": False,
            }
        )
    if aggregate != {
        "true_positives": 10,
        "false_positives": 5,
        "false_negatives": 7,
    }:
        raise RuntimeError("v19 reviewed-cell diagnosis changed")

    diagnosis = {
        "schema_version": 1,
        "status": "v19_owner_review_diagnosed_without_new_inference",
        "experiment_id": "supervised_labeler_v19",
        "source_owner_review_sha256": str(review["review_sha256"]),
        "scope": "frozen_audit_evidence_only",
        "score_threshold": threshold,
        "match_iou": MATCH_IOU,
        "reported_cell_diagnoses": cell_diagnoses,
        "reported_cells_aggregate": aggregate,
        "root_cause_summary": {
            "misses": (
                "The five owner-reported cells contain seven unmatched "
                "ground-truth worn helmeted heads at the frozen threshold."
            ),
            "false_positives": (
                "The two owner-reported empty-GT cells contain five model "
                "predictions, confirming semantic false positives rather than "
                "a rendering defect."
            ),
            "rendering_bug": False,
        },
        "future_intervention_constraint": (
            "Do not change the v19 threshold or geometry after audit. Replay "
            "the five miss image IDs as positives and the two false-positive "
            "image IDs as hard negatives only in a preregistered fresh model "
            "with a new independent audit."
        ),
        "diagnosis_model_inference_run": False,
        "source_image_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    diagnosis["report_sha256"] = canonical_mapping_sha256(diagnosis)
    return diagnosis


def main() -> None:
    if OUTPUT_PATH.exists() or DIAGNOSIS_PATH.exists():
        raise RuntimeError("v19 owner review evidence already exists")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registration = config["model_review_registration"]
    manifest = _verified_manifest()
    report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(MODEL_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        config["status"] != "owner_model_review_pending"
        or registration["owner_review_status"] != "pending_kuotunyu"
        or _sha256(MODEL_REVIEW_MANIFEST_PATH)
        != registration["manifest_file_sha256"]
        or manifest["manifest_sha256"]
        != registration["manifest_sha256"]
        or report.get("status") != "supervised_labeler_audit_passed"
        or not all(report["checks"].values())
        or report["audit_evidence_sha256"]
        != _sha256(MODEL_EVIDENCE_PATH)
    ):
        raise RuntimeError("Configured v19 model outcome changed")

    review = build_review(
        manifest=manifest,
        evidence=evidence,
        reviewed_on="2026-07-30",
    )
    diagnosis = build_diagnosis(evidence=evidence, review=review)
    OUTPUT_PATH.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    DIAGNOSIS_PATH.write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": review["status"],
                "problem_cells": review["problem_cells"],
                "review_sha256": review["review_sha256"],
                "diagnosis_sha256": diagnosis["report_sha256"],
                "reported_cells_aggregate": diagnosis[
                    "reported_cells_aggregate"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
