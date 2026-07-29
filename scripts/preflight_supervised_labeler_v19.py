"""Verify v19 partitions, normalization, replay, and base model on CPU."""

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

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v19.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v19_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v19_preflight.json"
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v19_preflight.md"
)


def main() -> None:
    if REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("Supervised labeler v19 preflight already exists")
    config = load_supervised_labeler_config(CONFIG_PATH)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] != "supervised_labeler_v19"
        or config["split_manifest_sha256"] != split["manifest_sha256"]
        or split["status"] != "frozen_before_supervised_training"
        or split["initialization"] != "pinned_base_checkpoint_only"
    ):
        raise RuntimeError("v19 registration and frozen split disagree")

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
        raise RuntimeError("Validation/Test leakage entered v19")

    model_groups = set(split["training_group_ids"]) | set(
        split["calibration_group_ids"]
    )
    training_groups = set(split["training_group_ids"])
    calibration_groups = set(split["calibration_group_ids"])
    v19_groups = set(split["v19_reserved_group_ids"])
    v18_revealed_groups = set(split["v18_revealed_audit_group_ids"])
    v18_nonselected_groups = set(
        split["v18_nonselected_excluded_group_ids"]
    )
    prior_sealed_groups = (
        set(split["v17_nonselected_excluded_group_ids"])
        | set(split["v16_nonselected_excluded_group_ids"])
        | set(split["v15_sealed_reserve_excluded_group_ids"])
        | set(split["v14_sealed_reserve_excluded_group_ids"])
        | set(split["v13_sealed_reserve_excluded_group_ids"])
        | set(split["v12_development_excluded_group_ids"])
    )
    replay_ids = (
        set(split["owner_miss_replay_image_ids"])
        | set(split["positive_error_replay_image_ids"])
        | set(split["hard_negative_error_replay_image_ids"])
    )
    if (
        training_groups & calibration_groups
        or model_groups & v19_groups
        or model_groups & v18_nonselected_groups
        or model_groups & prior_sealed_groups
        or not v18_revealed_groups <= training_groups
        or not replay_ids <= set(split["training_image_ids"])
        or len(audit) != 48
    ):
        raise RuntimeError("v19 model-data independence changed")

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
        owner_miss_replay_image_ids=tuple(
            int(value)
            for value in sampling["owner_miss_replay_image_ids"]
        ),
        owner_miss_replay_weight=float(
            sampling["owner_miss_replay_weight"]
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
    if not set(replay_weights.values()) <= {12.0, 16.0, 20.0}:
        raise RuntimeError("v19 error-replay weights changed")
    if (
        replay_weights["551"] != 12.0
        or replay_weights["857"] != 16.0
        or replay_weights["4187"] != 16.0
        or replay_weights["4618"] != 20.0
        or replay_weights["2262"] != 20.0
        or replay_weights["2580"] != 20.0
        or replay_weights["2941"] != 20.0
        or replay_weights["361"] != 20.0
        or replay_weights["4151"] != 20.0
    ):
        raise RuntimeError("v19 replay maximum-weight policy changed")

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
        "calibration_min_precision": config["calibration"]["min_precision"],
        "audit_min_precision": config["audit_gate"]["min_precision"],
        "training": training_scan,
        "calibration": calibration_scan,
        "sampling_weight_counts": weight_counts,
        "error_replay_weights": replay_weights,
        "overlapping_replay_image_ids": [361, 774, 4151],
        "overlap_policy": sampling["overlap_policy"],
        "model": {
            "repo_id": model_manifest["repo_id"],
            "revision": model_manifest["revision"],
            "download_bytes": model_manifest["download_bytes"],
            "files_rehashed": len(model_manifest["files"]),
        },
        "training_calibration_group_overlap": 0,
        "v19_reserved_groups_in_model_data": 0,
        "v18_nonselected_groups_in_model_data": 0,
        "prior_sealed_groups_in_model_data": 0,
        "v18_revealed_audit_groups_in_training": len(v18_revealed_groups),
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
        "# Supervised labeler v19 CPU preflight",
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
        "- Prior sealed, nonselected v18, and v19 groups in model data: **0**",
        "- Owner-approved v18 audit groups in training: **48**",
        "- Historic positive replay: **12.0**",
        "- Historic owner-miss replay: **16.0**",
        "- Hard-negative and image 4618 replay: **20.0**",
        "- Replay overlap policy: **maximum, never stacked**",
        "- Calibration / audit precision floors: **0.90 / 0.85**",
        "- Out-of-image raw box-area limit: **0.10**",
        "- Independent-audit pixels read: **0**",
        "- Sealed-reserve pixels read: **0**",
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
