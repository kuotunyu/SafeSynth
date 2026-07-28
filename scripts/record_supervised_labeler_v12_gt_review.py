"""Record the v12 GT-only review and freeze its corrected final audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_supervised_labeler_v12_gt_review import (
    CONFIG_PATH,
    EVIDENCE_PATH,
    POOL_PATH,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

OWNER_REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v12_gt_owner_review.json"
)
AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v12_adjudicated_audit.json"
)


def parse_primary_cells(value: str) -> list[int]:
    """Parse unique canonical cells in the v12 primary range 01-64."""

    try:
        cells = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Cells must be comma-separated integers"
        ) from error
    if not cells or any(cell < 1 or cell > 64 for cell in cells):
        raise argparse.ArgumentTypeError("Cells must be between 1 and 64")
    return cells


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
    uncertain_cells: list[int],
) -> dict[str, Any]:
    """Build the exact owner decision over all 64 GT-only primary cases."""

    date.fromisoformat(reviewed_on)
    if (
        len(evidence["cases"]) != 64
        or evidence["pool_manifest_sha256"] != pool["manifest_sha256"]
        or evidence["model_boxes_present"] is not False
        or evidence["model_inference_run"] is not False
        or int(evidence["sealed_reserve_pixels_read"]) != 0
    ):
        raise RuntimeError("v12 GT-only evidence boundary changed")
    uncertain = set(uncertain_cells)
    if uncertain != {53, 64}:
        raise RuntimeError("Owner decision must record exact cells 53 and 64")

    decisions = []
    for case in evidence["cases"]:
        cell = int(case["cell"])
        decisions.append(
            {
                "cell": cell,
                "group_id": int(case["group_id"]),
                "image_id": int(case["image_id"]),
                "stratum": str(case["stratum"]),
                "decision": "UNCERTAIN" if cell in uncertain else "PASS",
                "reason": (
                    "edge_clipped_target_not_reliably_adjudicable"
                    if cell in uncertain
                    else None
                ),
            }
        )
    review = {
        "schema_version": 1,
        "status": "v12_gt_only_primary_adjudicated",
        "experiment_id": str(pool["experiment_id"]),
        "reviewed_by": "kuotunyu",
        "reviewed_on": str(reviewed_on),
        "review_stage": "gt_only",
        "label_semantics": str(evidence["label_semantics"]),
        "pool_manifest_sha256": str(pool["manifest_sha256"]),
        "gt_review_evidence_sha256": str(evidence["evidence_sha256"]),
        "reviewed_images": len(decisions),
        "pass_images": sum(row["decision"] == "PASS" for row in decisions),
        "quarantined_images": sum(
            row["decision"] == "UNCERTAIN" for row in decisions
        ),
        "categories": {
            "dataset_gt_false_positive_cells": [],
            "dataset_gt_miss_cells": [],
            "dataset_gt_localization_cells": [],
            "uncertain_edge_clipped_cells": sorted(uncertain),
        },
        "decisions": decisions,
        "quarantine_policy": (
            "Quarantine the complete image; do not delete only the edge box."
        ),
        "owner_note": (
            "Cells 53 and 64 contain hard-to-see, edge-clipped boxes that may "
            "be omitted. All other primary cells are approved."
        ),
        "model_boxes_present": False,
        "model_inference_run": False,
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
    """Select the first valid primary cases within every frozen stratum."""

    decisions = {
        int(row["cell"]): str(row["decision"])
        for row in review["decisions"]
    }
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
        if (
            decisions[cell] == "PASS"
            and selected_per_stratum[stratum] < quotas[stratum]
        ):
            selected_cells.add(cell)
            selected_per_stratum[stratum] += 1
    if dict(selected_per_stratum) != quotas:
        raise RuntimeError("Valid primary cases do not satisfy frozen quotas")

    selected = []
    surplus = []
    quarantined = []
    pool_by_cell = {
        int(row["cell"]): row for row in pool["primary_cases"]
    }
    for cell in sorted(pool_by_cell):
        registration = pool_by_cell[cell]
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
        elif decisions[cell] == "PASS":
            surplus.append(common)
        else:
            quarantined.append(
                {
                    **common,
                    "decision": decisions[cell],
                    "reason": (
                        "edge_clipped_target_not_reliably_adjudicable"
                    ),
                }
            )

    reserve_groups = {
        int(row["group_id"]) for row in pool["sealed_reserve_cases"]
    }
    all_pool_groups = {
        int(row["group_id"])
        for row in [
            *pool["primary_cases"],
            *pool["sealed_reserve_cases"],
        ]
    }
    manifest = {
        "schema_version": 1,
        "status": "v12_adjudicated_audit_frozen_before_model_inference",
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
        "quarantined_primary_cases": quarantined,
        "quarantined_primary_images": len(quarantined),
        "sealed_reserve_images": len(pool["sealed_reserve_cases"]),
        "sealed_reserve_group_count": len(reserve_groups),
        "sealed_reserve_pixels_read": 0,
        "source_group_ids_reserved_from_training": sorted(all_pool_groups),
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
        or len(quarantined) != 2
        or len(surplus) != 14
    ):
        raise RuntimeError("v12 adjudicated audit violates a frozen boundary")
    manifest["manifest_sha256"] = canonical_mapping_sha256(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-on", required=True)
    parser.add_argument(
        "--uncertain-cells",
        required=True,
        type=parse_primary_cells,
    )
    args = parser.parse_args()
    if OWNER_REVIEW_PATH.exists() or AUDIT_PATH.exists():
        raise RuntimeError("v12 GT-only owner evidence already exists")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    pool = _verified_json(
        POOL_PATH,
        hash_field="manifest_sha256",
        expected_status="v12_gt_only_pool_frozen_before_pixel_review",
    )
    evidence = _verified_json(
        EVIDENCE_PATH,
        hash_field="evidence_sha256",
        expected_status="v12_gt_only_primary_review_rendered",
    )
    review = build_owner_review(
        pool=pool,
        evidence=evidence,
        reviewed_on=args.reviewed_on,
        uncertain_cells=args.uncertain_cells,
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
                "quarantined_primary_cells": [
                    row["primary_cell"]
                    for row in audit["quarantined_primary_cases"]
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
