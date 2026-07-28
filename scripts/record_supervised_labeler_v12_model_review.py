"""Record kuotunyu's review of the exact frozen v12 model audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_supervised_labeler_v12_gt_review import CONFIG_PATH
from scripts.record_supervised_labeler_v6_review import parse_problem_cells
from scripts.run_supervised_labeler_v12_model_audit import (
    EVIDENCE_PATH as MODEL_EVIDENCE_PATH,
)
from scripts.run_supervised_labeler_v12_model_audit import (
    REPORT_PATH as MODEL_REPORT_PATH,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v12_model_human_review.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_json(
    path: Path,
    *,
    hash_field: str,
    expected_status: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = dict(payload)
    embedded_sha = str(canonical.pop(hash_field, ""))
    if (
        canonical_mapping_sha256(canonical) != embedded_sha
        or payload.get("status") != expected_status
    ):
        raise RuntimeError(f"Frozen v12 model evidence changed: {path}")
    return payload


def build_review(
    *,
    report: dict[str, Any],
    evidence: dict[str, Any],
    reviewed_on: str,
    false_positive_cells: list[int],
    missed_cells: list[int],
) -> dict[str, Any]:
    """Build the exact owner rejection with disjoint failure categories."""

    date.fromisoformat(reviewed_on)
    false_positives = set(false_positive_cells)
    misses = set(missed_cells)
    if (
        false_positives != {3, 10, 13, 20, 25, 27, 29, 45}
        or misses != {26, 43}
        or false_positives & misses
        or len(evidence["cases"]) != 48
        or evidence["evidence_sha256"] != report["evidence_sha256"]
        or evidence["audit_manifest_sha256"]
        != report["audit_manifest_sha256"]
        or not all(report["checks"].values())
    ):
        raise RuntimeError("Owner decision does not match exact v12 evidence")
    problem_cells = sorted(false_positives | misses)
    review = {
        "schema_version": 1,
        "status": "rejected_by_kuotunyu",
        "experiment_id": "supervised_labeler_v12",
        "reviewed_by": "kuotunyu",
        "reviewed_on": str(reviewed_on),
        "decision": "reject",
        "label_semantics": "class_direct_helmeted_head_region",
        "problem_count": len(problem_cells),
        "problem_cells": problem_cells,
        "categories": {
            "model_false_positive_cells": sorted(false_positives),
            "model_missed_helmeted_head_cells": sorted(misses),
        },
        "review_note": (
            "Owner reported magenta boxes on background, faces, unworn "
            "helmets, or other objects in cells 03, 10, 13, 20, 25, 27, "
            "29, and 45; real helmeted heads were missed in cells 26 and 43."
        ),
        "numeric_audit_status": str(report["status"]),
        "numeric_audit_report_path": str(
            MODEL_REPORT_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "numeric_audit_report_file_sha256": _sha256(MODEL_REPORT_PATH),
        "numeric_audit_report_sha256": str(report["report_sha256"]),
        "model_evidence_path": str(
            MODEL_EVIDENCE_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "model_evidence_file_sha256": _sha256(MODEL_EVIDENCE_PATH),
        "model_evidence_sha256": str(evidence["evidence_sha256"]),
        "audit_manifest_sha256": str(report["audit_manifest_sha256"]),
        "checkpoint_sha256": str(report["checkpoint_sha256"]),
        "score_threshold": float(report["score_threshold"]),
        "pages": report["pages"],
        "generation_allowed": False,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    review["review_sha256"] = canonical_mapping_sha256(review)
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-on", required=True)
    parser.add_argument(
        "--false-positive-cells",
        required=True,
        type=parse_problem_cells,
    )
    parser.add_argument(
        "--missed-cells",
        required=True,
        type=parse_problem_cells,
    )
    args = parser.parse_args()
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Review evidence already exists: {OUTPUT_PATH}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    outcome = config["model_audit_outcome"]
    report = _verified_json(
        MODEL_REPORT_PATH,
        hash_field="report_sha256",
        expected_status="v12_numeric_audit_passed_owner_review_pending",
    )
    evidence = _verified_json(
        MODEL_EVIDENCE_PATH,
        hash_field="evidence_sha256",
        expected_status="v12_frozen_model_audit_evidence",
    )
    if (
        _sha256(MODEL_REPORT_PATH) != outcome["report_file_sha256"]
        or _sha256(MODEL_EVIDENCE_PATH)
        != outcome["evidence_file_sha256"]
        or report["report_sha256"] != outcome["report_sha256"]
        or evidence["evidence_sha256"] != outcome["evidence_sha256"]
    ):
        raise RuntimeError("Configured v12 model outcome changed")
    review = build_review(
        report=report,
        evidence=evidence,
        reviewed_on=args.reviewed_on,
        false_positive_cells=args.false_positive_cells,
        missed_cells=args.missed_cells,
    )
    OUTPUT_PATH.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(review, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
