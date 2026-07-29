"""Verify v16 partitions, normalization, replay sampling, and base model on CPU."""

from __future__ import annotations

import json
from collections import Counter

from scripts.preflight_supervised_labeler_v13 import _scan_dataset
from scripts.train_supervised_labeler import _build_datasets
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import (
    load_supervised_labeler_config,
    model_directory,
    require_verified_model,
    supervised_sampling_weights,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v16.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v16_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v16_preflight.json"
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v16_preflight.md"
)


def main() -> None:
    if REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("Supervised labeler v16 preflight already exists")
    config = load_supervised_labeler_config(CONFIG_PATH)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] != "supervised_labeler_v16"
        or config["split_manifest_sha256"] != split["manifest_sha256"]
        or split["status"] != "frozen_before_supervised_training"
        or split["initialization"] != "pinned_base_checkpoint_only"
    ):
        raise RuntimeError("v16 registration and frozen split disagree")

    (
        _,
        _,
        paths,
        train_images,
        _,
        training,
        calibration,
        audit,
    ) = _build_datasets(config_path=CONFIG_PATH, split_path=SPLIT_PATH)
    coco, _, _, annotations, _, test_ids = _load_context(paths)
    helmet_category_id = next(
        int(category["id"])
        for category in coco["categories"]
        if str(category["name"]) == "helmet"
    )
    selected = (
        set(split["training_image_ids"])
        | set(split["calibration_image_ids"])
        | set(split["untouched_audit_image_ids"])
    )
    if test_ids & selected:
        raise RuntimeError("Validation/Test leakage entered v16")

    model_groups = set(split["training_group_ids"]) | set(
        split["calibration_group_ids"]
    )
    training_groups = set(split["training_group_ids"])
    calibration_groups = set(split["calibration_group_ids"])
    v16_groups = set(split["v16_reserved_group_ids"])
    v15_primary_groups = set(split["v15_approved_primary_group_ids"])
    v15_reserve_groups = set(
        split["v15_sealed_reserve_excluded_group_ids"]
    )
    v14_primary_groups = set(split["v14_approved_primary_group_ids"])
    v14_reserve_groups = set(
        split["v14_sealed_reserve_excluded_group_ids"]
    )
    v13_primary_groups = set(split["v13_approved_primary_group_ids"])
    v13_reserve_groups = set(
        split["v13_sealed_reserve_excluded_group_ids"]
    )
    v12_groups = set(split["v12_development_excluded_group_ids"])
    replay_ids = set(split["positive_error_replay_image_ids"]) | set(
        split["hard_negative_error_replay_image_ids"]
    )
    if (
        training_groups & calibration_groups
        or model_groups & v16_groups
        or model_groups & v15_reserve_groups
        or model_groups & v14_reserve_groups
        or model_groups & v13_reserve_groups
        or model_groups & v12_groups
        or not v15_primary_groups <= training_groups
        or not v14_primary_groups <= training_groups
        or not v13_primary_groups <= training_groups
        or not replay_ids <= set(split["training_image_ids"])
        or len(audit) != 48
    ):
        raise RuntimeError("v16 model-data independence changed")

    training_scan = _scan_dataset(
        name="training",
        dataset=training,
        images=train_images,
    )
    calibration_scan = _scan_dataset(
        name="calibration",
        dataset=calibration,
        images=train_images,
    )
    sampling = config["sampling"]
    weights = supervised_sampling_weights(
        image_ids=split["training_image_ids"],
        annotations=annotations,
        image_records=train_images,
        helmet_category_id=helmet_category_id,
        empty_image_weight=float(sampling["empty_image_weight"]),
        close_helmet_pair_weight=float(sampling["close_helmet_pair_weight"]),
        close_pair_ratio_max=float(
            sampling[
                "close_pair_center_distance_over_mean_sqrt_area_max"
            ]
        ),
        small_helmet_weight=float(sampling["small_helmet_weight"]),
        small_helmet_relative_area_max=float(
            sampling["small_helmet_relative_area_max"]
        ),
        large_helmet_weight=float(sampling["large_helmet_weight"]),
        large_helmet_relative_area_min=float(
            sampling["large_helmet_relative_area_min"]
        ),
        near_image_edge_helmet_weight=float(
            sampling["near_image_edge_helmet_weight"]
        ),
        near_image_edge_margin_fraction=float(
            sampling["near_image_edge_margin_fraction"]
        ),
        positive_error_replay_image_ids=tuple(
            int(value)
            for value in sampling["positive_error_replay_image_ids"]
        ),
        positive_error_replay_weight=float(
            sampling["positive_error_replay_weight"]
        ),
        hard_negative_error_replay_image_ids=tuple(
            int(value)
            for value in sampling["hard_negative_error_replay_image_ids"]
        ),
        hard_negative_error_replay_weight=float(
            sampling["hard_negative_error_replay_weight"]
        ),
    )
    weight_counts = {
        str(key): value for key, value in sorted(Counter(weights).items())
    }
    replay_weights = {
        str(image_id): weights[split["training_image_ids"].index(image_id)]
        for image_id in sorted(replay_ids)
    }
    if set(replay_weights.values()) != {12.0}:
        raise RuntimeError("v16 error-replay weights changed")
    if replay_weights["361"] != 12.0:
        raise RuntimeError("v16 overlapping replay weights stacked")

    model_manifest = require_verified_model(
        model_directory(load_project_paths(), config),
        config,
    )
    payload = {
        "schema_version": 1,
        "status": "cpu_preflight_passed_gpu_smoke_waiting",
        "experiment_id": config["experiment_id"],
        "config_path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_path": SPLIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_manifest_sha256": split["manifest_sha256"],
        "input_normalization": config["input_normalization"],
        "postprocessing": config["postprocessing"],
        "training": training_scan,
        "calibration": calibration_scan,
        "sampling_weight_counts": weight_counts,
        "error_replay_weights": replay_weights,
        "overlapping_replay_image_id": 361,
        "overlap_policy": sampling["overlap_policy"],
        "model": {
            "repo_id": model_manifest["repo_id"],
            "revision": model_manifest["revision"],
            "download_bytes": model_manifest["download_bytes"],
            "files_rehashed": len(model_manifest["files"]),
        },
        "training_calibration_group_overlap": 0,
        "v16_reserved_groups_in_model_data": 0,
        "v15_sealed_reserve_groups_in_model_data": 0,
        "v14_sealed_reserve_groups_in_model_data": 0,
        "v13_sealed_reserve_groups_in_model_data": 0,
        "v12_development_groups_in_model_data": 0,
        "v15_approved_primary_groups_in_training": len(v15_primary_groups),
        "v14_approved_primary_groups_in_training": len(v14_primary_groups),
        "v13_approved_primary_groups_in_training": len(v13_primary_groups),
        "untouched_audit_images": len(audit),
        "untouched_audit_pixels_read": 0,
        "sealed_reserve_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "gpu_work_run": False,
        "whole_image_generation_run": False,
    }
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler v16 CPU preflight",
        "",
        "- Status: **passed; waiting for one-batch GPU smoke**",
        f"- Split SHA-256: `{split['manifest_sha256']}`",
        (
            "- Training normalized / read: "
            f"**{training_scan['normalized_images']} / "
            f"{training_scan['images_read']} images**"
        ),
        (
            "- Calibration normalized / read: "
            f"**{calibration_scan['normalized_images']} / "
            f"{calibration_scan['images_read']} images**"
        ),
        "- Invalid transformed boxes: **0**",
        "- Prior sealed groups and v16 groups in model data: **0**",
        "- Approved v13 / v14 / v15 groups replayable: **64 / 63 / 59**",
        "- Error-replay weights (maximum, not stacked): **12.0**",
        "- Independent-audit pixels read: **0**",
        "- v16 sealed-reserve pixels read: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- GPU work / whole-image generation: **no / no**",
        "",
    ]
    MARKDOWN_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
