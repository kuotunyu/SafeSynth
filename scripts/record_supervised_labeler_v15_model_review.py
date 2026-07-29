"""Record kuotunyu's final rejection of the frozen v15 model audit pages."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v15.yaml"
MODEL_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_training.json"
)
MODEL_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_audit_evidence.json"
)
MODEL_REVIEW_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v15_model_review_manifest.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v15_model_human_review.json"
)
FALSE_POSITIVE_CELLS = [11]
AMBIGUOUS_GT_QUARANTINE_CELLS = [29, 38]
PROBLEM_CELLS = sorted(
    FALSE_POSITIVE_CELLS + AMBIGUOUS_GT_QUARANTINE_CELLS
)


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
        != "supervised_labeler_v15_model_review_pages_frozen"
        or int(manifest.get("reviewed_images", -1)) != 48
        or manifest.get("render_model_inference_run") is not False
    ):
        raise RuntimeError("Frozen v15 model-review manifest changed")
    return manifest


def build_review(
    *,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    reviewed_on: str,
) -> dict[str, Any]:
    """Build the exact one-model-FP, two-ambiguous-GT owner rejection."""

    date.fromisoformat(reviewed_on)
    if (
        len(evidence.get("cases", [])) != 48
        or manifest["source_audit_evidence_sha256"]
        != _sha256(MODEL_EVIDENCE_PATH)
    ):
        raise RuntimeError("Frozen v15 model evidence changed")
    evidence_by_cell = {
        int(row["cell"]): row for row in evidence["cases"]
    }
    expected_image_ids = {11: 4100, 29: 787, 38: 243}
    if {
        cell: int(evidence_by_cell[cell]["image_id"])
        for cell in PROBLEM_CELLS
    } != expected_image_ids:
        raise RuntimeError("Frozen v15 problem cells changed")

    problem_cases = []
    for cell in PROBLEM_CELLS:
        problem_cases.append(
            {
                "cell": cell,
                "image_id": expected_image_ids[cell],
                "category": (
                    "model_false_positive"
                    if cell in FALSE_POSITIVE_CELLS
                    else "ambiguous_dataset_gt_quarantine"
                ),
            }
        )
    review = {
        "schema_version": 1,
        "status": "rejected_by_kuotunyu",
        "experiment_id": "supervised_labeler_v15",
        "reviewed_by": "kuotunyu",
        "reviewed_on": reviewed_on,
        "decision": "reject",
        "label_semantics": str(manifest["label_semantics"]),
        "problem_count": len(PROBLEM_CELLS),
        "problem_cells": PROBLEM_CELLS,
        "categories": {
            "model_false_positive_cells": FALSE_POSITIVE_CELLS,
            "model_missed_helmeted_head_cells": [],
            "ambiguous_dataset_gt_quarantine_cells": (
                AMBIGUOUS_GT_QUARANTINE_CELLS
            ),
        },
        "problem_cases": problem_cases,
        "review_note": (
            "Cell 11 contains confirmed model false positives on non-target "
            "content. Cells 29 and 38 are semantically ambiguous: occlusion "
            "or viewing angle makes either boxing or not boxing defensible. "
            "Those complete images are quarantined and are not counted as "
            "model failures."
        ),
        "ambiguity_policy": (
            "A sample without one reliable class-direct answer cannot reward "
            "or penalize the model; quarantine the complete image."
        ),
        "numeric_audit_status": "invalidated_for_final_acceptance_by_two_ambiguous_gt_images",
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


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Review evidence already exists: {OUTPUT_PATH}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registration = config["model_review_registration"]
    manifest = _verified_manifest()
    report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(MODEL_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        config["status"] != "numeric_audit_passed_owner_model_review_pending"
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
        raise RuntimeError("Configured v15 model outcome changed")
    review = build_review(
        manifest=manifest,
        evidence=evidence,
        reviewed_on="2026-07-29",
    )
    OUTPUT_PATH.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
