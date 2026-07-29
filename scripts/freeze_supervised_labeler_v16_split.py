"""Freeze the leakage-free v16 Train-only model partitions."""

from __future__ import annotations

import json

from scripts.freeze_supervised_labeler_v15_split import (
    _approved_primary_groups,
    _file_sha256,
    _pool_groups,
    _verified_json,
)
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import load_supervised_labeler_config
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v16.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v16_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v16_split.md"
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
V15_GT_REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_gt_owner_review.json"
)
V15_MODEL_REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_model_human_review.json"
)
V16_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v16_gt_pool.json"
)
V16_AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v16_adjudicated_audit.json"
)


def main() -> None:
    if SPLIT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Supervised labeler v16 split is already frozen")
    config = load_supervised_labeler_config(CONFIG_PATH)
    if (
        config["experiment_id"] != "supervised_labeler_v16"
        or config["optimization"]["initialization"]
        != "pinned_base_checkpoint_only"
        or config["split_manifest_sha256"] != "pending_split_freeze"
    ):
        raise RuntimeError("Supervised v16 preregistration changed")

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
    v15_gt_review = _verified_json(
        V15_GT_REVIEW_PATH,
        hash_field="review_sha256",
        expected_status="v15_gt_only_primary_adjudicated_three_quarantines",
    )
    v15_model_review = _verified_json(
        V15_MODEL_REVIEW_PATH,
        hash_field="review_sha256",
        expected_status="rejected_by_kuotunyu",
    )
    v16_pool = _verified_json(
        V16_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status=(
            "v16_gt_only_pool_frozen_before_pixel_review_or_training"
        ),
    )
    v16_audit = _verified_json(
        V16_AUDIT_PATH,
        hash_field="manifest_sha256",
        expected_status="v16_adjudicated_audit_frozen_before_training",
    )
    if (
        v16_audit["pool_manifest_sha256"] != v16_pool["manifest_sha256"]
        or v16_audit["v16_training_started"] is not False
        or v16_audit["model_inference_run"] is not False
        or {row["decision"] for row in v13_review["decisions"]}
        != {"PASS"}
        or v15_model_review["categories"][
            "ambiguous_dataset_gt_quarantine_cells"
        ]
        != [29, 38]
        or {
            int(row["image_id"])
            for row in v15_model_review["problem_cases"]
            if row["category"] == "ambiguous_dataset_gt_quarantine"
        }
        != {243, 787}
    ):
        raise RuntimeError("v13-v16 owner-reviewed boundary changed")

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
    v15_gt_approved_groups = _approved_primary_groups(
        v15_pool,
        v15_gt_review,
    )
    v15_reserve_groups = {
        int(row["group_id"]) for row in v15_pool["sealed_reserve_cases"]
    }
    v16_groups = _pool_groups(v16_pool)
    prior_groups = (
        v12_groups
        | v13_approved_groups
        | v13_reserve_groups
        | v14_approved_groups
        | v14_reserve_groups
        | _pool_groups(v15_pool)
    )
    if (
        len(v12_groups) != 96
        or len(v13_approved_groups) != 64
        or len(v13_reserve_groups) != 32
        or len(v14_approved_groups) != 63
        or len(v14_reserve_groups) != 32
        or len(v15_gt_approved_groups) != 61
        or len(v15_reserve_groups) != 32
        or len(v16_groups) != 96
        or v16_groups & prior_groups
    ):
        raise RuntimeError("A frozen GT pool lost source-group independence")

    quarantined_image_ids = {
        int(value)
        for value in config["data"][
            "quarantined_gt_defect_or_ambiguous_image_ids"
        ]
    }
    quarantined_groups = {
        group_for[image_id] for image_id in quarantined_image_ids
    }
    v15_approved_groups = v15_gt_approved_groups - quarantined_groups
    if (
        len(v15_approved_groups) != 59
        or {242, 785} & v15_approved_groups
    ):
        raise RuntimeError("Ambiguous v15 groups entered revealed training")

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
        | v15_reserve_groups
        | v16_groups
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
        int(row["image_id"]) for row in v16_audit["selected_cases"]
    ]
    audit_groups = {group_for[image_id] for image_id in audit_ids}
    expected_audit_groups = {
        int(row["group_id"]) for row in v16_audit["selected_cases"]
    }
    if (
        len(audit_ids) != 48
        or len(audit_groups) != 48
        or audit_groups != expected_audit_groups
        or not audit_groups <= v16_groups
    ):
        raise RuntimeError("The approved v16 audit changed")

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
        raise RuntimeError("A preregistered v16 replay image is absent")
    training_helmet_annotations = sum(
        int(annotation["category_id"]) == helmet_category_id
        for image_id in training_ids
        for annotation in annotations[image_id]
    )

    group_sets = [training_groups, calibration_groups, v16_groups]
    if any(
        left & right
        for index, left in enumerate(group_sets)
        for right in group_sets[index + 1 :]
    ):
        raise RuntimeError("v16 training/calibration/audit-pool groups overlap")
    all_selected = set(training_ids) | set(calibration_ids) | set(audit_ids)
    if test_ids & all_selected:
        raise RuntimeError("Validation/Test leakage entered v16")
    sealed_prior_groups = (
        v12_groups
        | v13_reserve_groups
        | v14_reserve_groups
        | v15_reserve_groups
    )
    if sealed_prior_groups & (training_groups | calibration_groups):
        raise RuntimeError("Sealed prior development groups entered v16")
    if not v13_approved_groups <= training_groups:
        raise RuntimeError("Approved v13 primary groups are not replayable")
    if not v14_approved_groups <= training_groups:
        raise RuntimeError("Approved v14 primary groups are not replayable")
    if not v15_approved_groups <= training_groups:
        raise RuntimeError("Approved v15 primary groups are not replayable")

    payload = {
        "schema_version": 1,
        "status": "frozen_before_supervised_training",
        "experiment_id": "supervised_labeler_v16",
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
        "v16_reserved_group_ids": sorted(v16_groups),
        "v16_reserved_groups": len(v16_groups),
        "v15_approved_primary_group_ids": sorted(v15_approved_groups),
        "v15_approved_primary_groups": len(v15_approved_groups),
        "v15_sealed_reserve_excluded_group_ids": sorted(v15_reserve_groups),
        "v15_sealed_reserve_excluded_groups": len(v15_reserve_groups),
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
        "quarantined_gt_image_ids": sorted(quarantined_image_ids),
        "quarantined_gt_group_ids": sorted(quarantined_groups),
        "quarantined_gt_images": len(quarantined_image_ids),
        "positive_error_replay_image_ids": sorted(positive_replay_ids),
        "hard_negative_error_replay_image_ids": sorted(
            hard_negative_replay_ids
        ),
        "gt_owner_review_sha256": str(
            config["independence_registration"]["gt_owner_review_sha256"]
        ),
        "audit_manifest_sha256": str(v16_audit["manifest_sha256"]),
        "audit_manifest_file_sha256": _file_sha256(V16_AUDIT_PATH),
        "initialization": "pinned_base_checkpoint_only",
        "v16_training_started": False,
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
        "# Supervised labeler v16 leakage-free Train-only split",
        "",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        f"- Training: **{payload['training_images']} images**",
        f"- Calibration: **{payload['calibration_images']} images**",
        "- Frozen independent audit: **48 owner-adjudicated images**",
        "- v16 primary/reserve groups excluded from model data: **96**",
        "- Prior sealed groups in model data: **0**",
        "- Approved v13 / v14 / v15 groups replayable: **64 / 63 / 59**",
        "- Training/calibration/v16-pool group overlap: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- v16 training/model audit run before freeze: **no / no**",
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
                "v16_reserved_groups": payload["v16_reserved_groups"],
                "v15_approved_primary_groups": payload[
                    "v15_approved_primary_groups"
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
