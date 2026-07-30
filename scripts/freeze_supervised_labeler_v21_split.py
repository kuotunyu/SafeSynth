"""Freeze the leakage-free v21 Train-only model partitions."""

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

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v21.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v21_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v21_split.md"
V20_SPLIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v20_split.json"
)
V20_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v20_gt_pool.json"
)
V20_AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v20_adjudicated_audit.json"
)
V21_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v21_gt_pool.json"
)
V21_AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v21_adjudicated_audit.json"
)


def main() -> None:
    if SPLIT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Supervised labeler v21 split is already frozen")
    config = load_supervised_labeler_config(CONFIG_PATH)
    if (
        config["experiment_id"] != "supervised_labeler_v21"
        or config["optimization"]["initialization"]
        != "pinned_base_checkpoint_only"
        or config["split_manifest_sha256"] != "pending_split_freeze"
    ):
        raise RuntimeError("Supervised v21 preregistration changed")

    v20_split = _verified_json(
        V20_SPLIT_PATH,
        hash_field="manifest_sha256",
        expected_status="frozen_before_supervised_training",
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
    v21_pool = _verified_json(
        V21_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status=(
            "v21_gt_only_pool_frozen_before_pixel_review_or_training"
        ),
    )
    v21_audit = _verified_json(
        V21_AUDIT_PATH,
        hash_field="manifest_sha256",
        expected_status="v21_adjudicated_audit_frozen_before_training",
    )
    if (
        v20_audit["pool_manifest_sha256"] != v20_pool["manifest_sha256"]
        or v21_audit["pool_manifest_sha256"] != v21_pool["manifest_sha256"]
        or v21_audit["v21_training_started"] is not False
        or v21_audit["model_inference_run"] is not False
    ):
        raise RuntimeError("v20-v21 adjudicated boundary changed")

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
    v20_groups = _pool_groups(v20_pool)
    v21_groups = _pool_groups(v21_pool)
    v20_revealed_audit_groups = {
        int(row["group_id"]) for row in v20_audit["selected_cases"]
    }
    v20_revealed_audit_image_ids = {
        int(row["image_id"]) for row in v20_audit["selected_cases"]
    }
    v20_nonselected_groups = v20_groups - v20_revealed_audit_groups
    if (
        len(v20_groups) != 96
        or len(v21_groups) != 96
        or len(v20_revealed_audit_groups) != 48
        or len(v20_nonselected_groups) != 48
        or v20_groups & v21_groups
    ):
        raise RuntimeError("A v20/v21 source-group boundary changed")

    prior_training_groups = {
        int(value) for value in v20_split["training_group_ids"]
    }
    calibration_ids = [
        int(value) for value in v20_split["calibration_image_ids"]
    ]
    calibration_groups = {
        int(value) for value in v20_split["calibration_group_ids"]
    }
    if (
        v20_groups & (prior_training_groups | calibration_groups)
        or v21_groups & calibration_groups
        or v20_revealed_audit_groups & calibration_groups
    ):
        raise RuntimeError("A fresh or newly revealed group entered calibration")

    v21_prior_training_groups_removed = prior_training_groups & v21_groups
    if len(v21_prior_training_groups_removed) != 96:
        raise RuntimeError("The preregistered v21 model-data exclusion changed")
    training_groups = (
        prior_training_groups - v21_groups
    ) | v20_revealed_audit_groups
    model_excluded_groups = v20_nonselected_groups | v21_groups
    training_ids = sorted(
        int(image_id)
        for image_id in train_images
        if group_for[int(image_id)] in training_groups
        and group_for[int(image_id)] not in model_excluded_groups
    )
    if not v20_revealed_audit_image_ids <= set(training_ids):
        raise RuntimeError("An owner-reviewed v20 audit image is absent")

    audit_ids = [
        int(row["image_id"]) for row in v21_audit["selected_cases"]
    ]
    audit_groups = {
        int(row["group_id"]) for row in v21_audit["selected_cases"]
    }
    if (
        len(audit_ids) != 48
        or len(audit_groups) != 48
        or not audit_groups <= v21_groups
    ):
        raise RuntimeError("The approved v21 audit changed")

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
        set(v20_split["training_image_ids"])
        | v20_revealed_audit_image_ids
    )
    required_v20_misses = {462, 1666, 2892, 3926, 4352, 4582}
    required_v20_false_positives = {
        1332,
        1699,
        1789,
        1841,
        2308,
        3089,
        3507,
        3833,
        4031,
        4352,
        4622,
    }
    if (
        not replay_ids <= available_replay_ids
        or not replay_ids <= set(training_ids)
        or not required_v20_misses <= owner_miss_replay_ids
        or not required_v20_false_positives <= hard_negative_replay_ids
    ):
        raise RuntimeError("A preregistered v21 replay image is unavailable")

    group_sets = [training_groups, calibration_groups, v21_groups]
    if any(
        left & right
        for index, left in enumerate(group_sets)
        for right in group_sets[index + 1 :]
    ):
        raise RuntimeError("v21 training/calibration/audit-pool groups overlap")
    all_selected = set(training_ids) | set(calibration_ids) | set(audit_ids)
    if test_ids & all_selected:
        raise RuntimeError("Validation/Test leakage entered v21")

    sealed_prior_groups = {
        int(value)
        for key, values in v20_split.items()
        if key.endswith("_excluded_group_ids") and isinstance(values, list)
        for value in values
    } | v20_nonselected_groups
    if sealed_prior_groups & (training_groups | calibration_groups):
        raise RuntimeError("Sealed prior development groups entered v21")
    training_helmet_annotations = sum(
        int(annotation["category_id"]) == helmet_category_id
        for image_id in training_ids
        for annotation in annotations[image_id]
    )

    payload = {
        "schema_version": 1,
        "status": "frozen_before_supervised_training",
        "experiment_id": "supervised_labeler_v21",
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
        "v21_reserved_group_ids": sorted(v21_groups),
        "v21_reserved_groups": len(v21_groups),
        "v21_prior_training_group_ids_removed": sorted(
            v21_prior_training_groups_removed
        ),
        "v21_prior_training_groups_removed": len(
            v21_prior_training_groups_removed
        ),
        "v20_revealed_audit_group_ids": sorted(v20_revealed_audit_groups),
        "v20_revealed_audit_groups": len(v20_revealed_audit_groups),
        "v20_revealed_audit_image_ids": sorted(
            v20_revealed_audit_image_ids
        ),
        "v20_nonselected_excluded_group_ids": sorted(v20_nonselected_groups),
        "v20_nonselected_excluded_groups": len(v20_nonselected_groups),
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
        "audit_manifest_sha256": str(v21_audit["manifest_sha256"]),
        "audit_manifest_file_sha256": _file_sha256(V21_AUDIT_PATH),
        "initialization": "pinned_base_checkpoint_only",
        "v21_training_started": False,
        "sealed_reserve_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    for key, values in v20_split.items():
        if key.endswith("_excluded_group_ids") and isinstance(values, list):
            payload.setdefault(key, values)
    payload["manifest_sha256"] = canonical_mapping_sha256(payload)
    SPLIT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        "# Supervised labeler v21 leakage-free Train-only split",
        "",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        f"- Training: **{payload['training_images']} images**",
        f"- Calibration: **{payload['calibration_images']} images**",
        "- Frozen independent audit: **48 owner-adjudicated images**",
        "- v21 primary/reserve groups excluded from model data: **96**",
        "- v21 groups removed from prior training data: **96**",
        "- Owner-reviewed v20 audit groups added to training: **48**",
        "- Nonselected v20 pool groups excluded: **48**",
        "- Prior sealed groups in model data: **0**",
        "- Training/calibration/v21-pool group overlap: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- v21 training/model audit run before freeze: **no / no**",
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
                "v21_reserved_groups": payload["v21_reserved_groups"],
                "v21_prior_training_groups_removed": payload[
                    "v21_prior_training_groups_removed"
                ],
                "v20_revealed_audit_groups": payload[
                    "v20_revealed_audit_groups"
                ],
                "v20_nonselected_excluded_groups": payload[
                    "v20_nonselected_excluded_groups"
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
