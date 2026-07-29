"""Record kuotunyu's rejection of the frozen v14 model audit pages."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from scripts.record_supervised_labeler_v6_review import parse_problem_cells
from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v14.yaml"
MODEL_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_training.json"
)
MODEL_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_audit_evidence.json"
)
MODEL_REVIEW_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v14_model_review_manifest.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v14_model_human_review.json"
)
EXPECTED_FALSE_POSITIVE_CELLS = {10}
EXPECTED_MISSED_CELLS = {7, 40, 43}


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
        != "supervised_labeler_v14_model_review_pages_frozen"
        or int(manifest.get("reviewed_images", -1)) != 48
        or manifest.get("render_model_inference_run") is not False
    ):
        raise RuntimeError("Frozen v14 model-review manifest changed")
    return manifest


def build_review(
    *,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    reviewed_on: str,
    false_positive_cells: list[int],
    missed_cells: list[int],
) -> dict[str, Any]:
    """Build the exact one-false-positive, three-miss owner rejection."""

    date.fromisoformat(reviewed_on)
    false_positives = set(false_positive_cells)
    misses = set(missed_cells)
    if (
        false_positives != EXPECTED_FALSE_POSITIVE_CELLS
        or misses != EXPECTED_MISSED_CELLS
        or false_positives & misses
        or len(evidence.get("cases", [])) != 48
        or manifest["source_audit_evidence_sha256"]
        != _sha256(MODEL_EVIDENCE_PATH)
    ):
        raise RuntimeError("Owner decision does not match exact v14 evidence")
    evidence_by_cell = {
        int(row["cell"]): row for row in evidence["cases"]
    }
    problem_cells = sorted(false_positives | misses)
    problem_cases = []
    for cell in problem_cells:
        category = (
            "model_false_positive"
            if cell in false_positives
            else "model_missed_helmeted_head"
        )
        problem_cases.append(
            {
                "cell": cell,
                "image_id": int(evidence_by_cell[cell]["image_id"]),
                "category": category,
            }
        )
    review = {
        "schema_version": 1,
        "status": "rejected_by_kuotunyu",
        "experiment_id": "supervised_labeler_v14",
        "reviewed_by": "kuotunyu",
        "reviewed_on": str(reviewed_on),
        "decision": "reject",
        "label_semantics": str(manifest["label_semantics"]),
        "problem_count": len(problem_cells),
        "problem_cells": problem_cells,
        "categories": {
            "model_false_positive_cells": sorted(false_positives),
            "model_missed_helmeted_head_cells": sorted(misses),
        },
        "problem_cases": problem_cases,
        "review_note": (
            "Most v14 boxes were good. Cell 10 has a model box on a face "
            "without a worn hard hat. Real worn-helmeted heads were missed "
            "in cells 07, 40, and 43."
        ),
        "numeric_audit_status": "supervised_labeler_audit_passed",
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
    registration = config["model_review_registration"]
    manifest = _verified_manifest()
    report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(MODEL_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        config["status"] != "numeric_audit_passed_human_review_pending"
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
        raise RuntimeError("Configured v14 model outcome changed")
    review = build_review(
        manifest=manifest,
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
