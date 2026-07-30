"""Freeze the leakage-free v20 Train-only model partitions."""

from __future__ import annotations

import json

from scripts.freeze_supervised_labeler_v15_split import (
    _file_sha256,
    _pool_groups,
    _verified_json,
)
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import load_supervised_labeler_config
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v20.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v20_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v20_split.md"
V19_SPLIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v19_split.json"
)
V19_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v19_gt_pool.json"
)
V19_AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v19_adjudicated_audit.json"
)
V20_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v20_gt_pool.json"
)
V20_AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v20_adjudicated_audit.json"
)


def main() -> None:
    if SPLIT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Supervised labeler v20 split is already frozen")
    config = load_supervised_labeler_config(CONFIG_PATH)
    if (
        config["experiment_id"] != "supervised_labeler_v20"
        or config["optimization"]["initialization"]
        != "pinned_base_checkpoint_only"
        or config["split_manifest_sha256"] != "pending_split_freeze"
    ):
        raise RuntimeError("Supervised v20 preregistration changed")

    v19_split = _verified_json(
        V19_SPLIT_PATH,
        hash_field="manifest_sha256",
        expected_status="frozen_before_supervised_training",
    )
    v19_pool = _verified_json(
        V19_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status=(
            "v19_gt_only_pool_frozen_before_pixel_review_or_training"
        ),
    )
    v19_audit = _verified_json(
        V19_AUDIT_PATH,
        hash_field="manifest_sha256",
        expected_status="v19_adjudicated_audit_frozen_before_training",
    )
    v20_pool = _verified_json(
        V20_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status=(
            "v20_gt_only_pool_frozen_before_pixel_review_or_training"
        ),
    )
    v20_audit = _verified_json(
        V20_AUDIT_PATH,
        hash_field="manifest_sha256",
        expected_status="v20_adjudicated_audit_frozen_before_training",
    )
    if (
        v19_audit["pool_manifest_sha256"] != v19_pool["manifest_sha256"]
        or v20_audit["pool_manifest_sha256"] != v20_pool["manifest_sha256"]
        or v20_audit["v20_training_started"] is not False
        or v20_audit["model_inference_run"] is not False
    ):
        raise RuntimeError("v19-v20 adjudicated boundary changed")

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
    v19_groups = _pool_groups(v19_pool)
    v20_groups = _pool_groups(v20_pool)
    v19_revealed_audit_groups = {
        int(row["group_id"]) for row in v19_audit["selected_cases"]
    }
    v19_revealed_audit_image_ids = {
        int(row["image_id"]) for row in v19_audit["selected_cases"]
    }
    v19_nonselected_groups = v19_groups - v19_revealed_audit_groups
    if (
        len(v19_groups) != 96
        or len(v20_groups) != 96
        or len(v19_revealed_audit_groups) != 48
        or len(v19_nonselected_groups) != 48
        or v19_groups & v20_groups
    ):
        raise RuntimeError("A v19/v20 source-group boundary changed")

    prior_training_groups = {
        int(value) for value in v19_split["training_group_ids"]
    }
    calibration_ids = [
        int(value) for value in v19_split["calibration_image_ids"]
    ]
    calibration_groups = {
        int(value) for value in v19_split["calibration_group_ids"]
    }
    if (
        v19_groups & (prior_training_groups | calibration_groups)
        or v20_groups & calibration_groups
        or v19_revealed_audit_groups & calibration_groups
    ):
        raise RuntimeError("A fresh or newly revealed group entered calibration")

    v20_prior_training_groups_removed = prior_training_groups & v20_groups
    if len(v20_prior_training_groups_removed) != 96:
        raise RuntimeError("The preregistered v20 model-data exclusion changed")
    training_groups = (
        prior_training_groups - v20_groups
    ) | v19_revealed_audit_groups
    model_excluded_groups = v19_nonselected_groups | v20_groups
    training_ids = sorted(
        int(image_id)
        for image_id in train_images
        if group_for[int(image_id)] in training_groups
        and group_for[int(image_id)] not in model_excluded_groups
    )
    if not v19_revealed_audit_image_ids <= set(training_ids):
        raise RuntimeError("An owner-reviewed v19 audit image is absent")
    audit_ids = [
        int(row["image_id"]) for row in v20_audit["selected_cases"]
    ]
    audit_groups = {
        int(row["group_id"]) for row in v20_audit["selected_cases"]
    }
    if (
        len(audit_ids) != 48
        or len(audit_groups) != 48
        or not audit_groups <= v20_groups
    ):
        raise RuntimeError("The approved v20 audit changed")

    sampling = config["sampling"]
    owner_miss_replay_ids = {
        int(value) for value in sampling["owner_miss_replay_image_ids"]
    }
    positive_replay_ids = {
        int(value) for value in sampling["positive_error_replay_image_ids"]
    }
    hard_negative_replay_ids = {
        int(value)
        for value in sampling["hard_negative_error_replay_image_ids"]
    }
    replay_ids = (
        owner_miss_replay_ids
        | positive_replay_ids
        | hard_negative_replay_ids
    )
    available_replay_ids = (
        set(v19_split["training_image_ids"])
        | v19_revealed_audit_image_ids
    )
    if (
        not replay_ids <= available_replay_ids
        or not replay_ids <= set(training_ids)
        or not {118, 3117, 3651, 4452, 4924} <= owner_miss_replay_ids
        or not {1241, 2910} <= hard_negative_replay_ids
    ):
        raise RuntimeError("A preregistered v20 replay image is unavailable")

    group_sets = [training_groups, calibration_groups, v20_groups]
    if any(
        left & right
        for index, left in enumerate(group_sets)
        for right in group_sets[index + 1 :]
    ):
        raise RuntimeError("v20 training/calibration/audit-pool groups overlap")
    all_selected = set(training_ids) | set(calibration_ids) | set(audit_ids)
    if test_ids & all_selected:
        raise RuntimeError("Validation/Test leakage entered v20")

    sealed_prior_groups = (
        set(v19_split["v12_development_excluded_group_ids"])
        | set(v19_split["v13_sealed_reserve_excluded_group_ids"])
        | set(v19_split["v14_sealed_reserve_excluded_group_ids"])
        | set(v19_split["v15_sealed_reserve_excluded_group_ids"])
        | set(v19_split["v16_nonselected_excluded_group_ids"])
        | set(v19_split["v17_nonselected_excluded_group_ids"])
        | set(v19_split["v18_nonselected_excluded_group_ids"])
        | v19_nonselected_groups
    )
    if sealed_prior_groups & (training_groups | calibration_groups):
        raise RuntimeError("Sealed prior development groups entered v20")
    training_helmet_annotations = sum(
        int(annotation["category_id"]) == helmet_category_id
        for image_id in training_ids
        for annotation in annotations[image_id]
    )

    payload = {
        "schema_version": 1,
        "status": "frozen_before_supervised_training",
        "experiment_id": "supervised_labeler_v20",
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
        "v20_reserved_group_ids": sorted(v20_groups),
        "v20_reserved_groups": len(v20_groups),
        "v20_prior_training_group_ids_removed": sorted(
            v20_prior_training_groups_removed
        ),
        "v20_prior_training_groups_removed": len(
            v20_prior_training_groups_removed
        ),
        "v19_revealed_audit_group_ids": sorted(v19_revealed_audit_groups),
        "v19_revealed_audit_groups": len(v19_revealed_audit_groups),
        "v19_revealed_audit_image_ids": sorted(
            v19_revealed_audit_image_ids
        ),
        "v19_nonselected_excluded_group_ids": sorted(v19_nonselected_groups),
        "v19_nonselected_excluded_groups": len(v19_nonselected_groups),
        "v18_nonselected_excluded_group_ids": v19_split[
            "v18_nonselected_excluded_group_ids"
        ],
        "v17_nonselected_excluded_group_ids": v19_split[
            "v17_nonselected_excluded_group_ids"
        ],
        "v16_nonselected_excluded_group_ids": v19_split[
            "v16_nonselected_excluded_group_ids"
        ],
        "v15_approved_primary_group_ids": v19_split[
            "v15_approved_primary_group_ids"
        ],
        "v15_sealed_reserve_excluded_group_ids": v19_split[
            "v15_sealed_reserve_excluded_group_ids"
        ],
        "v14_approved_primary_group_ids": v19_split[
            "v14_approved_primary_group_ids"
        ],
        "v14_sealed_reserve_excluded_group_ids": v19_split[
            "v14_sealed_reserve_excluded_group_ids"
        ],
        "v13_approved_primary_group_ids": v19_split[
            "v13_approved_primary_group_ids"
        ],
        "v13_sealed_reserve_excluded_group_ids": v19_split[
            "v13_sealed_reserve_excluded_group_ids"
        ],
        "v12_development_excluded_group_ids": v19_split[
            "v12_development_excluded_group_ids"
        ],
        "quarantined_gt_image_ids": sorted(
            int(value)
            for value in config["data"][
                "quarantined_gt_defect_or_ambiguous_image_ids"
            ]
        ),
        "owner_miss_replay_image_ids": sorted(owner_miss_replay_ids),
        "positive_error_replay_image_ids": sorted(positive_replay_ids),
        "hard_negative_error_replay_image_ids": sorted(
            hard_negative_replay_ids
        ),
        "gt_owner_review_sha256": str(
            config["independence_registration"]["gt_owner_review_sha256"]
        ),
        "audit_manifest_sha256": str(v20_audit["manifest_sha256"]),
        "audit_manifest_file_sha256": _file_sha256(V20_AUDIT_PATH),
        "initialization": "pinned_base_checkpoint_only",
        "v20_training_started": False,
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
        "# Supervised labeler v20 leakage-free Train-only split",
        "",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        f"- Training: **{payload['training_images']} images**",
        f"- Calibration: **{payload['calibration_images']} images**",
        "- Frozen independent audit: **48 owner-adjudicated images**",
        "- v20 primary/reserve groups excluded from model data: **96**",
        "- v20 groups removed from prior training data: **96**",
        "- Owner-reviewed v19 audit groups added to training: **48**",
        "- Nonselected v19 pool groups excluded: **48**",
        "- Prior sealed groups in model data: **0**",
        "- Training/calibration/v20-pool group overlap: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- v20 training/model audit run before freeze: **no / no**",
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
                "v20_reserved_groups": payload["v20_reserved_groups"],
                "v20_prior_training_groups_removed": payload[
                    "v20_prior_training_groups_removed"
                ],
                "v19_revealed_audit_groups": payload[
                    "v19_revealed_audit_groups"
                ],
                "v19_nonselected_excluded_groups": payload[
                    "v19_nonselected_excluded_groups"
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
