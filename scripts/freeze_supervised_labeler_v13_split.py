"""Freeze the leakage-free v13 Train-only model partitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import load_supervised_labeler_config
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v13.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v13_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v13_split.md"
V11_SPLIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v11_split.json"
)
V12_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v12_gt_pool.json"
)
V13_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v13_gt_pool.json"
)
V13_AUDIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v13_adjudicated_audit.json"
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


def main() -> None:
    if SPLIT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Supervised labeler v13 split is already frozen")
    config = load_supervised_labeler_config(CONFIG_PATH)
    if (
        config["experiment_id"] != "supervised_labeler_v13"
        or config["optimization"]["initialization"]
        != "pinned_base_checkpoint_only"
    ):
        raise RuntimeError("Supervised v13 preregistration changed")

    v11_split = json.loads(V11_SPLIT_PATH.read_text(encoding="utf-8"))
    v12_pool = _verified_json(
        V12_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status="v12_gt_only_pool_frozen_before_pixel_review",
    )
    v13_pool = _verified_json(
        V13_POOL_PATH,
        hash_field="manifest_sha256",
        expected_status="v13_gt_only_pool_frozen_before_pixel_review_or_training",
    )
    v13_audit = _verified_json(
        V13_AUDIT_PATH,
        hash_field="manifest_sha256",
        expected_status="v13_adjudicated_audit_frozen_before_training",
    )
    if (
        v13_audit["pool_manifest_sha256"] != v13_pool["manifest_sha256"]
        or v13_audit["v13_training_started"] is not False
        or v13_audit["model_inference_run"] is not False
    ):
        raise RuntimeError("v13 audit boundary changed before split freeze")

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
    v13_groups = _pool_groups(v13_pool)
    if len(v12_groups) != 96 or len(v13_groups) != 96:
        raise RuntimeError("A frozen GT pool lost source-group uniqueness")
    if v12_groups & v13_groups:
        raise RuntimeError("v12 and v13 frozen pools overlap")

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
    model_excluded_groups = v12_groups | v13_groups | quarantined_groups
    calibration_ids = sorted(
        image_id
        for image_id in revealed_calibration_candidates
        if image_id not in quarantined_image_ids
        and group_for[image_id] not in model_excluded_groups
    )
    calibration_groups = {group_for[image_id] for image_id in calibration_ids}
    audit_ids = [
        int(row["image_id"]) for row in v13_audit["selected_cases"]
    ]
    audit_groups = {group_for[image_id] for image_id in audit_ids}
    expected_audit_groups = {
        int(row["group_id"]) for row in v13_audit["selected_cases"]
    }
    if (
        len(audit_ids) != 48
        or len(audit_groups) != 48
        or audit_groups != expected_audit_groups
        or not audit_groups <= v13_groups
    ):
        raise RuntimeError("The approved v13 audit changed")

    excluded_from_training = model_excluded_groups | calibration_groups
    training_ids = sorted(
        image_id
        for image_id in train_images
        if group_for[image_id] not in excluded_from_training
    )
    training_groups = {group_for[image_id] for image_id in training_ids}
    training_helmet_annotations = sum(
        int(annotation["category_id"]) == helmet_category_id
        for image_id in training_ids
        for annotation in annotations[image_id]
    )
    group_sets = [training_groups, calibration_groups, v13_groups]
    if any(
        left & right
        for index, left in enumerate(group_sets)
        for right in group_sets[index + 1 :]
    ):
        raise RuntimeError("v13 training/calibration/audit-pool groups overlap")
    all_selected = set(training_ids) | set(calibration_ids) | set(audit_ids)
    if test_ids & all_selected:
        raise RuntimeError("Validation/Test leakage entered v13")
    if v12_groups & (training_groups | calibration_groups):
        raise RuntimeError("v12 development groups entered v13 model data")

    payload = {
        "schema_version": 1,
        "status": "frozen_before_supervised_training",
        "experiment_id": "supervised_labeler_v13",
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
        "v13_reserved_group_ids": sorted(v13_groups),
        "v13_reserved_groups": len(v13_groups),
        "v12_development_excluded_group_ids": sorted(v12_groups),
        "v12_development_excluded_groups": len(v12_groups),
        "quarantined_gt_defect_image_ids": sorted(quarantined_image_ids),
        "quarantined_gt_defect_group_ids": sorted(quarantined_groups),
        "quarantined_gt_defect_images": len(quarantined_image_ids),
        "gt_owner_review_sha256": str(
            config["independence_registration"]["gt_owner_review_sha256"]
        ),
        "audit_manifest_sha256": str(v13_audit["manifest_sha256"]),
        "audit_manifest_file_sha256": _file_sha256(V13_AUDIT_PATH),
        "initialization": "pinned_base_checkpoint_only",
        "v13_training_started": False,
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
        "# Supervised labeler v13 leakage-free Train-only split",
        "",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        f"- Training: **{payload['training_images']} images**",
        f"- Calibration: **{payload['calibration_images']} images**",
        "- Frozen independent audit: **48 approved images**",
        "- v13 primary/reserve groups excluded from model data: **96**",
        "- v12 development groups excluded from model data: **96**",
        "- Training/calibration/v13-pool group overlap: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- v13 training/model audit run before freeze: **no / no**",
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
                "v13_reserved_groups": payload["v13_reserved_groups"],
                "v12_development_excluded_groups": payload[
                    "v12_development_excluded_groups"
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
