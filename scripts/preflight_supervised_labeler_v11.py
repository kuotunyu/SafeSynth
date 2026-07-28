"""Verify v11 Train-only partitions and normalization without GPU."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from scripts.train_supervised_labeler import HelmetDataset, _build_datasets
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import (
    load_supervised_labeler_config,
    model_directory,
    require_verified_model,
    supervised_sampling_weights,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v11.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v11_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v11_preflight.json"
MARKDOWN_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v11_preflight.md"
V10_MODEL_PROBLEM_IMAGE_IDS = {
    478,
    550,
    708,
    2515,
    2826,
    3222,
    3950,
    3975,
    4821,
}
QUARANTINED_GT_DEFECT_IMAGE_IDS = {3060, 4155, 4364}


def _scan_dataset(
    *,
    name: str,
    dataset: HelmetDataset,
    images: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized_images = 0
    detected_side_counts: Counter[str] = Counter()
    source_helmet_annotations = 0
    transformed_helmet_annotations = 0
    removed_helmet_annotations = 0
    images_becoming_empty = 0
    revealed_problem_images: dict[str, Any] = {}
    for index, image_id in enumerate(dataset.image_ids):
        item = dataset[index]
        source_count = sum(
            int(annotation["category_id"]) == dataset.helmet_category_id
            for annotation in dataset.annotations[image_id]
        )
        transformed_count = len(item["truth"])
        normalization = item["input_normalization"]
        width = int(images[image_id]["width"])
        height = int(images[image_id]["height"])
        if item["image"].size != (width, height):
            raise RuntimeError(f"{name} image {image_id} changed output shape")
        if transformed_count > source_count:
            raise RuntimeError(f"{name} image {image_id} gained helmet boxes")
        if any(
            not (
                0 <= float(box[0]) < float(box[2]) <= width
                and 0 <= float(box[1]) < float(box[3]) <= height
            )
            for box in item["truth"]
        ):
            raise RuntimeError(f"{name} image {image_id} has an invalid box")
        source_helmet_annotations += source_count
        transformed_helmet_annotations += transformed_count
        removed_helmet_annotations += source_count - transformed_count
        if source_count and not transformed_count:
            images_becoming_empty += 1
        if normalization["applied"]:
            normalized_images += 1
            detected_side_counts.update(normalization["detected_sides"])
        if image_id in V10_MODEL_PROBLEM_IMAGE_IDS:
            revealed_problem_images[str(image_id)] = {
                "partition": name,
                "source_helmet_annotations": source_count,
                "transformed_helmet_annotations": transformed_count,
                **normalization,
            }
    return {
        "images_read": len(dataset),
        "normalized_images": normalized_images,
        "detected_side_counts": dict(sorted(detected_side_counts.items())),
        "source_helmet_annotations": source_helmet_annotations,
        "transformed_helmet_annotations": transformed_helmet_annotations,
        "removed_helmet_annotations": removed_helmet_annotations,
        "images_becoming_empty": images_becoming_empty,
        "invalid_boxes": 0,
        "revealed_problem_images": revealed_problem_images,
    }


def main() -> None:
    if REPORT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("Supervised labeler v11 preflight already exists")
    config = load_supervised_labeler_config(CONFIG_PATH)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if (
        config["experiment_id"] != "supervised_labeler_v11"
        or config["split_manifest_sha256"] != split["manifest_sha256"]
        or split["status"] != "frozen_before_supervised_training"
    ):
        raise RuntimeError("v11 registration and sealed split disagree")
    if set(split["quarantined_gt_defect_image_ids"]) != (
        QUARANTINED_GT_DEFECT_IMAGE_IDS
    ):
        raise RuntimeError("v11 GT quarantine changed")

    (
        _,
        _,
        paths,
        train_images,
        _,
        training,
        calibration,
        _,
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
        | set(split["quarantined_gt_defect_image_ids"])
        | set(split["untouched_audit_image_ids"])
    )
    if test_ids & selected:
        raise RuntimeError("Validation/Test leakage entered v11")
    if QUARANTINED_GT_DEFECT_IMAGE_IDS & (
        set(training.image_ids) | set(calibration.image_ids)
    ):
        raise RuntimeError("A quarantined GT defect entered v11 model data")
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
        raise RuntimeError("v11 source groups overlap")

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
    revealed_problem_images = {
        **training_scan.pop("revealed_problem_images"),
        **calibration_scan.pop("revealed_problem_images"),
    }
    if set(revealed_problem_images) != {
        str(value) for value in V10_MODEL_PROBLEM_IMAGE_IDS
    }:
        raise RuntimeError("A revealed v10 model problem left calibration")

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
        str(key): value for key, value in sorted(Counter(weights).items())
    }
    model_manifest = require_verified_model(
        model_directory(load_project_paths(), config),
        config,
    )
    payload = {
        "schema_version": 1,
        "status": "cpu_normalization_preflight_passed_gpu_smoke_waiting",
        "experiment_id": config["experiment_id"],
        "config_path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_path": SPLIT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "split_manifest_sha256": split["manifest_sha256"],
        "input_normalization": config["input_normalization"],
        "postprocessing": config["postprocessing"],
        "training": training_scan,
        "calibration": calibration_scan,
        "revealed_v10_model_problem_images": revealed_problem_images,
        "quarantined_gt_defect_image_ids": sorted(
            QUARANTINED_GT_DEFECT_IMAGE_IDS
        ),
        "quarantined_gt_defect_pixels_read": 0,
        "sampling_weight_counts": weight_counts,
        "model": {
            "repo_id": model_manifest["repo_id"],
            "revision": model_manifest["revision"],
            "download_bytes": model_manifest["download_bytes"],
            "files_rehashed": len(model_manifest["files"]),
        },
        "source_group_overlap": 0,
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
        "# Supervised labeler v11 CPU normalization preflight",
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
        "- Quarantined GT-defect pixels read: **0**",
        "- Sealed-audit pixels read: **0**",
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
