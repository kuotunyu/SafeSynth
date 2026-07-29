"""Freeze the leakage-free v15 Train-only model partitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import load_supervised_labeler_config
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v15.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v15_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v15_split.md"
V11_SPLIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v11_split.json"
)
V12_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v12_gt_pool.json"
)
V13_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v13_gt_pool.json"
)
V13_REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_gt_owner_review.json"
)
V14_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v14_gt_pool.json"
)
V14_REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_gt_owner_review.json"
)
V15_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v15_gt_pool.json"
)
V15_AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v15_adjudicated_audit.json"
)


def _file_sha256(path: Path) -> str:
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
        raise RuntimeError(f"Frozen evidence changed: {path}")
    return payload


def _pool_groups(pool: dict[str, Any]) -> set[int]:
    return {
        int(row["group_id"])
        for row in [
            *pool["primary_cases"],
            *pool["sealed_reserve_cases"],
        ]
    }


def _approved_primary_groups(
    pool: dict[str, Any],
    review: dict[str, Any],
) -> set[int]:
    registration = {
        int(row["cell"]): int(row["group_id"])
        for row in pool["primary_cases"]
    }
    return {
        registration[int(row["cell"])]
        for row in review["decisions"]
        if str(row["decision"]) == "PASS"
    }


def main() -> None:
    if SPLIT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Supervised labeler v15 split is already frozen")
    config = load_supervised_labeler_config(CONFIG_PATH)
    if (
        config["experiment_id"] != "supervised_labeler_v15"
        or config["optimization"]["initialization"]
        != "pinned_base_checkpoint_only"
        or config["split_manifest_sha256"] != "pending_split_freeze"
    ):
        raise RuntimeError("Supervised v15 preregistration changed")

    v11_split = json.loads(V11_SPLIT_PATH.read_text(encoding="utf-8"))
    v12_pool = _verified_json(
        V12_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status="v12_gt_only_pool_frozen_before_pixel_review",
    )
    v13_pool = _verified_json(
        V13_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status=(
            "v13_gt_only_pool_frozen_before_pixel_review_or_training"
        ),
    )
    v13_review = _verified_json(
        V13_REVIEW_PATH,
        hash_field="review_sha256",
        expected_status="v13_gt_only_primary_adjudicated_zero_problems",
    )
    v14_pool = _verified_json(
        V14_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status=(
            "v14_gt_only_pool_frozen_before_pixel_review_or_training"
        ),
    )
    v14_review = _verified_json(
        V14_REVIEW_PATH,
        hash_field="review_sha256",
        expected_status="v14_gt_only_primary_adjudicated_one_quarantine",
    )
    v15_pool = _verified_json(
        V15_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status=(
            "v15_gt_only_pool_frozen_before_pixel_review_or_training"
        ),
    )
    v15_audit = _verified_json(
        V15_AUDIT_PATH,
        hash_field="manifest_sha256",
        expected_status="v15_adjudicated_audit_frozen_before_training",
    )
    if (
        v15_audit["pool_manifest_sha256"] != v15_pool["manifest_sha256"]
        or v15_audit["v15_training_started"] is not False
        or v15_audit["model_inference_run"] is not False
        or {row["decision"] for row in v13_review["decisions"]}
        != {"PASS"}
    ):
        raise RuntimeError("v13-v15 owner-reviewed boundary changed")

    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    helmet_category_id = next(
        int(category["id"])
        for category in coco["categories"]
        if str(category["name"]) == "helmet"
    )
    group_for = {
        int(image_id): int(frozen[int(image_id)]["group_id"])
        for image_id in train_images
    }
    v12_groups = _pool_groups(v12_pool)
    v13_approved_groups = _approved_primary_groups(v13_pool, v13_review)
    v13_reserve_groups = {
        int(row["group_id"]) for row in v13_pool["sealed_reserve_cases"]
    }
    v14_approved_groups = _approved_primary_groups(v14_pool, v14_review)
    v14_reserve_groups = {
        int(row["group_id"]) for row in v14_pool["sealed_reserve_cases"]
    }
    v15_groups = _pool_groups(v15_pool)
    if (
        len(v12_groups) != 96
        or len(v13_approved_groups) != 64
        or len(v13_reserve_groups) != 32
        or len(v14_approved_groups) != 63
        or len(v14_reserve_groups) != 32
        or len(v15_groups) != 96
        or v15_groups
        & (
            v12_groups
            | v13_approved_groups
            | v13_reserve_groups
            | v14_approved_groups
            | v14_reserve_groups
        )
    ):
        raise RuntimeError("A frozen GT pool lost source-group independence")

    quarantined_image_ids = {
        int(value)
        for value in config["data"]["quarantined_gt_defect_image_ids"]
    }
    quarantined_groups = {
        group_for[image_id] for image_id in quarantined_image_ids
    }
    revealed_calibration_candidates = {
        int(value)
        for value in [
            *v11_split["calibration_image_ids"],
            *v11_split["untouched_audit_image_ids"],
        ]
    }
    model_excluded_groups = (
        v12_groups
        | v13_reserve_groups
        | v14_reserve_groups
        | v15_groups
        | quarantined_groups
    )
    calibration_ids = sorted(
        image_id
        for image_id in revealed_calibration_candidates
        if image_id not in quarantined_image_ids
        and group_for[image_id] not in model_excluded_groups
    )
    calibration_groups = {group_for[image_id] for image_id in calibration_ids}
    audit_ids = [
        int(row["image_id"]) for row in v15_audit["selected_cases"]
    ]
    audit_groups = {group_for[image_id] for image_id in audit_ids}
    expected_audit_groups = {
        int(row["group_id"]) for row in v15_audit["selected_cases"]
    }
    if (
        len(audit_ids) != 48
        or len(audit_groups) != 48
        or audit_groups != expected_audit_groups
        or not audit_groups <= v15_groups
    ):
        raise RuntimeError("The approved v15 audit changed")

    excluded_from_training = model_excluded_groups | calibration_groups
    training_ids = sorted(
        image_id
        for image_id in train_images
        if group_for[image_id] not in excluded_from_training
    )
    training_groups = {group_for[image_id] for image_id in training_ids}
    positive_replay_ids = {
        int(value)
        for value in config["sampling"]["positive_error_replay_image_ids"]
    }
    hard_negative_replay_ids = {
        int(value)
        for value in config["sampling"][
            "hard_negative_error_replay_image_ids"
        ]
    }
    if not (positive_replay_ids | hard_negative_replay_ids) <= set(
        training_ids
    ):
        raise RuntimeError("A preregistered v14 error replay image is absent")
    training_helmet_annotations = sum(
        int(annotation["category_id"]) == helmet_category_id
        for image_id in training_ids
        for annotation in annotations[image_id]
    )

    group_sets = [training_groups, calibration_groups, v15_groups]
    if any(
        left & right
        for index, left in enumerate(group_sets)
        for right in group_sets[index + 1 :]
    ):
        raise RuntimeError("v15 training/calibration/audit-pool groups overlap")
    all_selected = set(training_ids) | set(calibration_ids) | set(audit_ids)
    if test_ids & all_selected:
        raise RuntimeError("Validation/Test leakage entered v15")
    if (
        v12_groups | v13_reserve_groups | v14_reserve_groups
    ) & (training_groups | calibration_groups):
        raise RuntimeError("Sealed prior development groups entered v15")
    if not v13_approved_groups <= training_groups:
        raise RuntimeError("Approved v13 primary groups are not all replayable")
    if not v14_approved_groups <= training_groups:
        raise RuntimeError("Approved v14 primary groups are not all replayable")

    payload = {
        "schema_version": 1,
        "status": "frozen_before_supervised_training",
        "experiment_id": "supervised_labeler_v15",
        "source_split": "Train",
        "split_seed": int(config["split_seed"]),
        "training_image_ids": training_ids,
        "training_group_ids": sorted(training_groups),
        "training_images": len(training_ids),
        "training_groups": len(training_groups),
        "training_helmet_annotations": training_helmet_annotations,
        "calibration_image_ids": calibration_ids,
        "calibration_group_ids": sorted(calibration_groups),
        "calibration_images": len(calibration_ids),
        "untouched_audit_image_ids": audit_ids,
        "untouched_audit_group_ids": sorted(audit_groups),
        "untouched_audit_images": len(audit_ids),
        "v15_reserved_group_ids": sorted(v15_groups),
        "v15_reserved_groups": len(v15_groups),
        "v14_approved_primary_group_ids": sorted(v14_approved_groups),
        "v14_approved_primary_groups": len(v14_approved_groups),
        "v14_sealed_reserve_excluded_group_ids": sorted(v14_reserve_groups),
        "v14_sealed_reserve_excluded_groups": len(v14_reserve_groups),
        "v13_approved_primary_group_ids": sorted(v13_approved_groups),
        "v13_approved_primary_groups": len(v13_approved_groups),
        "v13_sealed_reserve_excluded_group_ids": sorted(v13_reserve_groups),
        "v13_sealed_reserve_excluded_groups": len(v13_reserve_groups),
        "v12_development_excluded_group_ids": sorted(v12_groups),
        "v12_development_excluded_groups": len(v12_groups),
        "quarantined_gt_defect_image_ids": sorted(quarantined_image_ids),
        "quarantined_gt_defect_group_ids": sorted(quarantined_groups),
        "quarantined_gt_defect_images": len(quarantined_image_ids),
        "positive_error_replay_image_ids": sorted(positive_replay_ids),
        "hard_negative_error_replay_image_ids": sorted(
            hard_negative_replay_ids
        ),
        "gt_owner_review_sha256": str(
            config["independence_registration"]["gt_owner_review_sha256"]
        ),
        "audit_manifest_sha256": str(v15_audit["manifest_sha256"]),
        "audit_manifest_file_sha256": _file_sha256(V15_AUDIT_PATH),
        "initialization": "pinned_base_checkpoint_only",
        "v15_training_started": False,
        "sealed_reserve_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    payload["manifest_sha256"] = canonical_mapping_sha256(payload)
    SPLIT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler v15 leakage-free Train-only split",
        "",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        f"- Training: **{payload['training_images']} images**",
        f"- Calibration: **{payload['calibration_images']} images**",
        "- Frozen independent audit: **48 owner-adjudicated images**",
        "- v15 primary/reserve groups excluded from model data: **96**",
        "- v12 / v13 reserve / v14 reserve excluded: **96 / 32 / 32**",
        "- Approved v13 / v14 primary groups in training: **64 / 63**",
        "- Training/calibration/v15-pool group overlap: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- v15 training/model audit run before freeze: **no / no**",
        "",
    ]
    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "manifest_sha256": payload["manifest_sha256"],
                "training_images": payload["training_images"],
                "training_groups": payload["training_groups"],
                "calibration_images": payload["calibration_images"],
                "calibration_groups": len(calibration_groups),
                "audit_images": payload["untouched_audit_images"],
                "v15_reserved_groups": payload["v15_reserved_groups"],
                "v14_approved_primary_groups": payload[
                    "v14_approved_primary_groups"
                ],
                "v14_sealed_reserve_excluded_groups": payload[
                    "v14_sealed_reserve_excluded_groups"
                ],
                "validation_images_read": 0,
                "test_images_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
