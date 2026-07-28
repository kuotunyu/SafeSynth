"""Audit whether the frozen v12 model review was independent of v11 training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.prepare_supervised_labeler_v12_gt_review import POOL_PATH
from scripts.record_supervised_labeler_v12_gt_review import AUDIT_PATH
from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

V11_SPLIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v11_split.json"
)
SOURCE_SPLIT_PATH = PROJECT_ROOT / "splits" / "split_manifest.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v12_audit_independence_erratum.json"
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
        raise RuntimeError(f"Frozen split evidence changed: {path}")
    return payload


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"v12 independence erratum exists: {OUTPUT_PATH}")
    v11 = _verified_json(
        V11_SPLIT_PATH,
        hash_field="manifest_sha256",
        expected_status="frozen_before_supervised_training",
    )
    pool = _verified_json(
        POOL_PATH,
        hash_field="manifest_sha256",
        expected_status="v12_gt_only_pool_frozen_before_pixel_review",
    )
    audit = _verified_json(
        AUDIT_PATH,
        hash_field="manifest_sha256",
        expected_status="v12_adjudicated_audit_frozen_before_model_inference",
    )
    source = json.loads(SOURCE_SPLIT_PATH.read_text(encoding="utf-8"))
    train_groups = {
        int(row["group_id"])
        for row in source["images"]
        if str(row["split"]).lower() == "train"
    }
    v11_training = {int(value) for value in v11["training_group_ids"]}
    v11_calibration = {
        int(value) for value in v11["calibration_group_ids"]
    }
    v11_audit = {
        int(value) for value in v11["untouched_audit_group_ids"]
    }
    partition = v11_training | v11_calibration | v11_audit
    pool_primary = {
        int(row["group_id"]) for row in pool["primary_cases"]
    }
    pool_reserve = {
        int(row["group_id"]) for row in pool["sealed_reserve_cases"]
    }
    selected = {
        int(row["group_id"]) for row in audit["selected_cases"]
    }
    overlaps = {
        "primary_with_v11_training": sorted(pool_primary & v11_training),
        "reserve_with_v11_training": sorted(pool_reserve & v11_training),
        "selected_audit_with_v11_training": sorted(selected & v11_training),
    }
    if (
        partition != train_groups
        or len(overlaps["primary_with_v11_training"]) != 64
        or len(overlaps["reserve_with_v11_training"]) != 32
        or len(overlaps["selected_audit_with_v11_training"]) != 48
    ):
        raise RuntimeError("Unexpected v11/v12 group-boundary result")
    erratum = {
        "schema_version": 1,
        "status": "v12_model_audit_invalidated_by_training_group_overlap",
        "experiment_id": "supervised_labeler_v12",
        "v11_split_manifest_sha256": str(v11["manifest_sha256"]),
        "v12_pool_manifest_sha256": str(pool["manifest_sha256"]),
        "v12_audit_manifest_sha256": str(audit["manifest_sha256"]),
        "train_source_group_count": len(train_groups),
        "v11_partition": {
            "training_group_count": len(v11_training),
            "calibration_group_count": len(v11_calibration),
            "audit_group_count": len(v11_audit),
            "union_group_count": len(partition),
            "covers_every_train_group": partition == train_groups,
        },
        "overlap_counts": {
            key: len(value) for key, value in overlaps.items()
        },
        "overlap_group_ids": overlaps,
        "numeric_audit_independent": False,
        "numeric_audit_claim_valid": False,
        "owner_visual_rejection_valid": True,
        "owner_failure_diagnosis_valid": True,
        "original_evidence_mutated": False,
        "required_next_boundary": (
            "freeze_a_new_group_disjoint_v13_split_before_training_and_train_"
            "v13_from_the_pinned_base_checkpoint_not_the_v11_checkpoint"
        ),
        "new_audit_selection_rule": (
            "exclude every group revealed through v12; select and freeze the "
            "v13 audit before v13 training; keep those groups out of both "
            "training and calibration"
        ),
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    erratum["erratum_sha256"] = canonical_mapping_sha256(erratum)
    OUTPUT_PATH.write_text(
        json.dumps(erratum, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(erratum, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
