"""Record the zero-problem v13 GT review and freeze its final audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_supervised_labeler_v13_gt_review import (
    CONFIG_PATH,
    EVIDENCE_PATH,
    POOL_PATH,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

OWNER_REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_gt_owner_review.json"
)
AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v13_adjudicated_audit.json"
)


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
        raise RuntimeError(f"Frozen evidence changed: {path}")
    return payload


def build_owner_review(
    *,
    pool: dict[str, Any],
    evidence: dict[str, Any],
    reviewed_on: str,
) -> dict[str, Any]:
    """Build the exact owner's zero-problem decision over all 64 cases."""

    date.fromisoformat(reviewed_on)
    if (
        len(evidence["cases"]) != 64
        or evidence["pool_manifest_sha256"] != pool["manifest_sha256"]
        or evidence["model_boxes_present"] is not False
        or evidence["model_inference_run"] is not False
        or evidence["v13_training_started"] is not False
        or int(evidence["sealed_reserve_pixels_read"]) != 0
        or int(evidence["validation_images_read"]) != 0
        or int(evidence["test_images_read"]) != 0
    ):
        raise RuntimeError("v13 GT-only evidence boundary changed")

    decisions = [
        {
            "cell": int(case["cell"]),
            "group_id": int(case["group_id"]),
            "image_id": int(case["image_id"]),
            "stratum": str(case["stratum"]),
            "decision": "PASS",
            "reason": None,
        }
        for case in evidence["cases"]
    ]
    review = {
        "schema_version": 1,
        "status": "v13_gt_only_primary_adjudicated_zero_problems",
        "experiment_id": str(pool["experiment_id"]),
        "reviewed_by": "kuotunyu",
        "reviewed_on": str(reviewed_on),
        "review_stage": "gt_only",
        "label_semantics": str(evidence["label_semantics"]),
        "pool_manifest_sha256": str(pool["manifest_sha256"]),
        "gt_review_evidence_sha256": str(evidence["evidence_sha256"]),
        "reviewed_images": len(decisions),
        "pass_images": len(decisions),
        "problem_images": 0,
        "quarantined_images": 0,
        "categories": {
            "dataset_gt_false_positive_cells": [],
            "dataset_gt_miss_cells": [],
            "dataset_gt_localization_cells": [],
            "uncertain_cells": [],
        },
        "decisions": decisions,
        "owner_note": "All 64 frozen v13 GT-only primary cells are approved.",
        "model_boxes_present": False,
        "model_inference_run": False,
        "v13_training_started": False,
        "sealed_reserve_images": int(evidence["sealed_reserve_images"]),
        "sealed_reserve_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    review["review_sha256"] = canonical_mapping_sha256(review)
    return review


def build_adjudicated_audit(
    *,
    config: dict[str, Any],
    pool: dict[str, Any],
    evidence: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Select the preregistered first valid cases in every frozen stratum."""

    decisions = {
        int(row["cell"]): str(row["decision"])
        for row in review["decisions"]
    }
    if set(decisions.values()) != {"PASS"}:
        raise RuntimeError("v13 zero-problem approval changed")
    evidence_by_cell = {
        int(row["cell"]): row for row in evidence["cases"]
    }
    quotas = {
        str(key): int(value)
        for key, value in config["final_audit_target"]["quotas"].items()
    }
    selected_cells: set[int] = set()
    selected_per_stratum: Counter[str] = Counter()
    for row in pool["primary_cases"]:
        cell = int(row["cell"])
        stratum = str(row["stratum"])
        if selected_per_stratum[stratum] < quotas[stratum]:
            selected_cells.add(cell)
            selected_per_stratum[stratum] += 1
    if dict(selected_per_stratum) != quotas:
        raise RuntimeError("Approved primary cases do not satisfy quotas")

    selected = []
    surplus = []
    for registration in pool["primary_cases"]:
        cell = int(registration["cell"])
        case = evidence_by_cell[cell]
        common = {
            "primary_cell": cell,
            "image_id": int(registration["image_id"]),
            "group_id": int(registration["group_id"]),
            "stratum": str(registration["stratum"]),
            "file_name": str(registration["file_name"]),
            "source_image_sha256": str(
                registration["source_image_sha256"]
            ),
        }
        if cell in selected_cells:
            selected.append(
                {
                    **common,
                    "audit_cell": len(selected) + 1,
                    "truth_boxes": case["truth_boxes"],
                    "input_normalization": case["input_normalization"],
                }
            )
        else:
            surplus.append(common)

    all_pool_groups = {
        int(row["group_id"])
        for row in [
            *pool["primary_cases"],
            *pool["sealed_reserve_cases"],
        ]
    }
    manifest = {
        "schema_version": 1,
        "status": "v13_adjudicated_audit_frozen_before_training",
        "experiment_id": str(pool["experiment_id"]),
        "source_split": "Train",
        "label_semantics": str(pool["label_semantics"]),
        "selection_policy": str(
            config["final_audit_target"]["selection_policy"]
        ),
        "pool_manifest_sha256": str(pool["manifest_sha256"]),
        "gt_review_evidence_sha256": str(evidence["evidence_sha256"]),
        "owner_review_sha256": str(review["review_sha256"]),
        "quotas": quotas,
        "selected_cases": selected,
        "selected_images": len(selected),
        "selected_stratum_counts": dict(
            sorted(Counter(row["stratum"] for row in selected).items())
        ),
        "valid_primary_surplus_cases": surplus,
        "valid_primary_surplus_images": len(surplus),
        "quarantined_primary_cases": [],
        "quarantined_primary_images": 0,
        "sealed_reserve_images": len(pool["sealed_reserve_cases"]),
        "sealed_reserve_pixels_read": 0,
        "source_group_ids_reserved_from_training": sorted(all_pool_groups),
        "v13_training_started": False,
        "model_boxes_present": False,
        "model_inference_run": False,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    if (
        len(selected) != int(config["final_audit_target"]["images"])
        or len({row["group_id"] for row in selected}) != len(selected)
        or len(all_pool_groups) != 96
        or len(surplus) != 16
    ):
        raise RuntimeError("v13 adjudicated audit violates frozen boundaries")
    manifest["manifest_sha256"] = canonical_mapping_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-on", required=True)
    parser.add_argument(
        "--approve-zero-problems",
        action="store_true",
        help="Record the owner's explicit approval of all 64 GT-only cells.",
    )
    args = parser.parse_args()
    if not args.approve_zero_problems:
        raise RuntimeError("Explicit --approve-zero-problems is required")
    if OWNER_REVIEW_PATH.exists() or AUDIT_PATH.exists():
        raise RuntimeError("v13 GT-only owner evidence already exists")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    pool = _verified_json(
        POOL_PATH,
        hash_field="manifest_sha256",
        expected_status="v13_gt_only_pool_frozen_before_pixel_review_or_training",
    )
    evidence = _verified_json(
        EVIDENCE_PATH,
        hash_field="evidence_sha256",
        expected_status=(
            "v13_gt_only_primary_review_rendered_before_training"
        ),
    )
    review = build_owner_review(
        pool=pool,
        evidence=evidence,
        reviewed_on=args.reviewed_on,
    )
    audit = build_adjudicated_audit(
        config=config,
        pool=pool,
        evidence=evidence,
        review=review,
    )
    OWNER_REVIEW_PATH.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "selected_images": audit["selected_images"],
                "selected_stratum_counts": audit[
                    "selected_stratum_counts"
                ],
                "valid_primary_surplus_images": audit[
                    "valid_primary_surplus_images"
                ],
                "sealed_reserve_pixels_read": audit[
                    "sealed_reserve_pixels_read"
                ],
                "review_sha256": review["review_sha256"],
                "manifest_sha256": audit["manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
