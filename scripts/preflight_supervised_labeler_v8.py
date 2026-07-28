"""Verify the sealed v8 registration without reading images or using GPU."""

from __future__ import annotations

import json
from collections import Counter

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import (
    CONFIG_PATH,
    SPLIT_PATH,
    load_supervised_labeler_config,
    model_directory,
    require_verified_model,
    supervised_sampling_weights,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v8_preflight.json"
MARKDOWN_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v8_preflight.md"


def main() -> None:
    if REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("Supervised labeler v8 preflight already exists")
    config = load_supervised_labeler_config(CONFIG_PATH)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] != "supervised_labeler_v8"
        or config["split_manifest_sha256"] != split["manifest_sha256"]
        or split["status"] != "frozen_before_supervised_training"
    ):
        raise RuntimeError("v8 registration and sealed split disagree")
    paths = load_project_paths()
    coco, _, train_images, annotations, _, test_ids = _load_context(paths)
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
        raise RuntimeError("Validation/Test leakage entered v8")
    group_sets = [
        set(split["training_group_ids"]),
        set(split["calibration_group_ids"]),
        set(split["untouched_audit_group_ids"]),
    ]
    if any(
        left & right
        for index, left in enumerate(group_sets)
        for right in group_sets[index + 1 :]
    ):
        raise RuntimeError("v8 source groups overlap")
    if any(int(image_id) not in train_images for image_id in selected):
        raise RuntimeError("v8 contains a non-Train image")

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
    )
    weight_counts = {
        str(key): value
        for key, value in sorted(Counter(weights).items())
    }
    if len(weights) != int(split["training_images"]):
        raise RuntimeError("v8 sampling weights do not cover training")
    if set(weights) - {1.0, 2.0}:
        raise RuntimeError("v8 sampling produced an unregistered weight")
    model_manifest = require_verified_model(
        model_directory(paths, config),
        config,
    )
    payload = {
        "schema_version": 1,
        "status": "cpu_preflight_passed_gpu_training_waiting",
        "experiment_id": config["experiment_id"],
        "config_path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_path": SPLIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_manifest_sha256": split["manifest_sha256"],
        "training_images": split["training_images"],
        "training_groups": split["training_groups"],
        "training_helmet_annotations": split[
            "training_helmet_annotations"
        ],
        "calibration_images": split["calibration_images"],
        "sealed_audit_images": split["untouched_audit_images"],
        "sampling_weight_counts": weight_counts,
        "model": {
            "repo_id": model_manifest["repo_id"],
            "revision": model_manifest["revision"],
            "download_bytes": model_manifest["download_bytes"],
            "files_rehashed": len(model_manifest["files"]),
        },
        "source_group_overlap": 0,
        "training_pixels_read": 0,
        "calibration_pixels_read": 0,
        "sealed_audit_pixels_read": 0,
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
        "# Supervised labeler v8 CPU preflight",
        "",
        "- Status: **passed; waiting for an unoccupied GPU**",
        f"- Split SHA-256: `{split['manifest_sha256']}`",
        (
            f"- Training / calibration / sealed audit: "
            f"**{split['training_images']} / "
            f"{split['calibration_images']} / "
            f"{split['untouched_audit_images']} images**"
        ),
        f"- Sampling weight counts: **{weight_counts}**",
        (
            f"- Rehashed model: **{model_manifest['repo_id']}** at "
            f"`{model_manifest['revision']}`"
        ),
        "- Source-group overlap: **0**",
        "- Train/calibration/sealed-audit pixels read: **0 / 0 / 0**",
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
