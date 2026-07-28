"""Preregistered Train-only split and model integrity for a supervised labeler."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from PIL import Image

from src.data.paths import PROJECT_ROOT, ProjectPaths

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v11.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v11_split.json"


def load_supervised_labeler_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the frozen supervised-labeler registration."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    model = config["model"]
    supported_models = {
        "PekingU/rtdetr_v2_r18vd": "rtdetr_v2_r18vd_helmet_only",
        "PekingU/rtdetr_v2_r50vd": "rtdetr_v2_r50vd_helmet_only",
        "PekingU/rtdetr_v2_r101vd": "rtdetr_v2_r101vd_helmet_only",
    }
    if (
        model["license"] != "apache-2.0"
        or model["local_files_only"] is not True
        or model["repo_id"] not in supported_models
        or config["architecture"] != supported_models[model["repo_id"]]
        or model["transformers_version"] != "5.14.1"
    ):
        raise RuntimeError("Supervised labeler model registration changed")
    if sum(int(value) for value in model["allow_files"].values()) != int(
        model["required_download_bytes"]
    ):
        raise RuntimeError("Supervised labeler registered sizes disagree")
    if (
        config["data"]["source_split"] != "train_only"
        or config["data"]["category"] != "helmet"
        or int(config["data"]["validation_images_read"]) != 0
        or int(config["data"]["test_images_read"]) != 0
    ):
        raise RuntimeError("Supervised labeler data boundary changed")
    if config["generation_gate"]["allowed"] is not False:
        raise RuntimeError("Generation cannot open before supervised audit")
    input_normalization = config.get("input_normalization")
    if input_normalization is not None:
        expected_guard = {
            "action": "normalize_cover_crop",
            "min_pad_px": 8,
            "max_pad_fraction": 0.31,
            "orthogonal_sample_size": 64,
            "max_pair_mae": 3.0,
            "min_pair_correlation": 0.97,
            "min_texture_std": 5.0,
        }
        if (
            input_normalization.get("method")
            != "reflected_padding_normalize_cover_crop"
            or input_normalization.get("implementation")
            != "src.synthetic.compose.normalize_reflected_padding"
            or input_normalization.get("output_shape") != "source_native"
            or input_normalization.get("transform_helmet_annotations") is not True
            or input_normalization.get("guard") != expected_guard
        ):
            raise RuntimeError("Supervised input normalization changed")
    postprocessing = config.get("postprocessing")
    if postprocessing is not None:
        min_aspect_ratio = float(postprocessing.get("min_aspect_ratio", 0.0))
        max_aspect_ratio = float(postprocessing.get("max_aspect_ratio", math.inf))
        if (
            not 0 < float(postprocessing["max_relative_area"]) <= 1
            or not 0 < float(postprocessing["max_relative_height"]) <= 1
            or min_aspect_ratio < 0
            or max_aspect_ratio <= 0
            or min_aspect_ratio > max_aspect_ratio
        ):
            raise RuntimeError("Supervised geometry filter is invalid")
    sampling = config.get("sampling")
    if sampling is not None and (
        sampling.get("strategy") != "deterministic_weighted_replacement"
        or float(sampling["empty_image_weight"]) < 1
        or float(sampling["close_helmet_pair_weight"]) < 1
        or float(sampling.get("small_helmet_weight", 1.0)) < 1
        or not 0
        <= float(sampling.get("small_helmet_relative_area_max", 0.0))
        <= 1
        or float(
            sampling["close_pair_center_distance_over_mean_sqrt_area_max"]
        )
        <= 0
        or sampling.get("overlap_policy") != "maximum_weight"
    ):
        raise RuntimeError("Supervised sampling registration is invalid")
    return config


def filter_prediction_geometry(
    predictions: Sequence[tuple[float, Sequence[float]]],
    *,
    image_width: int,
    image_height: int,
    max_relative_area: float,
    max_relative_height: float,
    min_aspect_ratio: float = 0.0,
    max_aspect_ratio: float = math.inf,
) -> list[tuple[float, list[float]]]:
    """Drop predictions outside fixed normalized size/aspect limits."""

    image_area = max(float(image_width) * float(image_height), 1.0)
    kept = []
    for score, box in predictions:
        x1, y1, x2, y2 = (float(value) for value in box)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width == 0 or height == 0:
            continue
        aspect_ratio = width / height
        if (
            width * height / image_area <= float(max_relative_area)
            and height / max(float(image_height), 1.0)
            <= float(max_relative_height)
            and aspect_ratio >= float(min_aspect_ratio)
            and aspect_ratio <= float(max_aspect_ratio)
        ):
            kept.append((float(score), [x1, y1, x2, y2]))
    return kept


def supervised_sampling_weights(
    *,
    image_ids: Sequence[int],
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    image_records: Mapping[int, Mapping[str, Any]] | None = None,
    helmet_category_id: int,
    empty_image_weight: float,
    close_helmet_pair_weight: float,
    close_pair_ratio_max: float,
    small_helmet_weight: float = 1.0,
    small_helmet_relative_area_max: float = 0.0,
) -> list[float]:
    """Weight registered hard examples using Train annotations only."""

    weights = []
    for image_id in image_ids:
        boxes = [
            [float(value) for value in annotation["bbox"]]
            for annotation in annotations[int(image_id)]
            if int(annotation["category_id"]) == int(helmet_category_id)
        ]
        weight = float(empty_image_weight) if not boxes else 1.0
        close_pair = False
        for left_index, left in enumerate(boxes):
            for right in boxes[left_index + 1 :]:
                left_scale = math.sqrt(max(left[2] * left[3], 0.0))
                right_scale = math.sqrt(max(right[2] * right[3], 0.0))
                mean_scale = max((left_scale + right_scale) / 2.0, 1e-9)
                center_distance = math.hypot(
                    (left[0] + left[2] / 2.0)
                    - (right[0] + right[2] / 2.0),
                    (left[1] + left[3] / 2.0)
                    - (right[1] + right[3] / 2.0),
                )
                if center_distance / mean_scale <= float(close_pair_ratio_max):
                    close_pair = True
                    break
            if close_pair:
                break
        if close_pair:
            weight = max(weight, float(close_helmet_pair_weight))
        if (
            boxes
            and float(small_helmet_weight) > 1.0
            and float(small_helmet_relative_area_max) > 0.0
        ):
            if image_records is None:
                raise ValueError("Small-helmet weighting requires image records")
            image = image_records[int(image_id)]
            image_area = max(
                float(image["width"]) * float(image["height"]),
                1.0,
            )
            if any(
                box[2] * box[3] / image_area
                <= float(small_helmet_relative_area_max)
                for box in boxes
            ):
                weight = max(weight, float(small_helmet_weight))
        weights.append(weight)
    return weights


def model_directory(paths: ProjectPaths, config: Mapping[str, Any]) -> Path:
    """Return the project-isolated supervised-labeler model directory."""

    model = config["model"]
    slug = str(model["repo_id"]).replace("/", "--")
    return paths.cache / "models" / slug / str(model["revision"])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_verified_model(
    model_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Rehash the exact local supervised-labeler checkpoint before use."""

    manifest_path = model_dir / "SAFESYNTH_MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("Pinned supervised labeler is not downloaded")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config["model"]
    if (
        manifest.get("repo_id") != expected["repo_id"]
        or manifest.get("revision") != expected["revision"]
        or manifest.get("license") != expected["license"]
        or int(manifest.get("download_bytes", -1))
        != int(expected["required_download_bytes"])
    ):
        raise RuntimeError("Supervised labeler manifest does not match registration")
    registered = {
        str(record["path"]): record for record in manifest.get("files", [])
    }
    if set(registered) != set(expected["allow_files"]):
        raise RuntimeError("Supervised labeler file list changed")
    for name, expected_size in expected["allow_files"].items():
        path = model_dir / str(name)
        record = registered[str(name)]
        if (
            not path.is_file()
            or path.stat().st_size != int(expected_size)
            or int(record.get("bytes", -1)) != int(expected_size)
            or _sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(
                f"Supervised labeler file failed integrity check: {name}"
            )
    return manifest


def require_verified_audited_checkpoint(
    *,
    config: Mapping[str, Any],
    registration: Mapping[str, Any],
    report: Mapping[str, Any],
    split: Mapping[str, Any],
) -> Path:
    """Verify the exact passed v6 audit and fine-tuned checkpoint."""

    expected_checks = {
        "audit_precision": True,
        "audit_recall": True,
        "audit_median_matched_iou": True,
    }
    expected_split_sha = str(registration["split_manifest_sha256"])
    expected_checkpoint_sha = str(registration["checkpoint_sha256"])
    best = report.get("best_calibration", {})
    metrics = report.get("audit_metrics", {})
    postprocessing = report.get("postprocessing", {})
    if (
        config.get("experiment_id") != registration["experiment_id"]
        or config.get("architecture") != registration["architecture"]
        or config.get("split_manifest_sha256") != expected_split_sha
        or split.get("manifest_sha256") != expected_split_sha
        or report.get("split_manifest_sha256") != expected_split_sha
        or split.get("status") != "frozen_before_supervised_training"
        or report.get("status") != "supervised_labeler_audit_passed"
        or report.get("checks") != expected_checks
        or int(report.get("untouched_audit_images_read", -1))
        != int(registration["audit_images"])
        or int(report.get("validation_images_read", -1)) != 0
        or int(report.get("test_images_read", -1)) != 0
        or int(split.get("validation_images_read", -1)) != 0
        or int(split.get("test_images_read", -1)) != 0
        or report.get("whole_image_generation_run") is not False
        or str(report.get("checkpoint_sha256")) != expected_checkpoint_sha
        or float(best.get("threshold", -1))
        != float(registration["score_threshold"])
        or float(postprocessing.get("max_relative_area", -1))
        != float(registration["max_relative_area"])
        or float(postprocessing.get("max_relative_height", -1))
        != float(registration["max_relative_height"])
        or float(metrics.get("precision", -1))
        != float(registration["audit_precision"])
        or float(metrics.get("recall", -1))
        != float(registration["audit_recall"])
        or float(metrics.get("median_matched_iou", -1))
        != float(registration["audit_median_matched_iou"])
    ):
        raise RuntimeError("Supervised v6 audit evidence changed or is incomplete")

    checkpoint_dir = Path(str(report["checkpoint_path"]))
    checkpoint_path = checkpoint_dir / "model.safetensors"
    if (
        not checkpoint_path.is_file()
        or _sha256_file(checkpoint_path) != expected_checkpoint_sha
    ):
        raise RuntimeError("Supervised v6 checkpoint failed integrity verification")
    for required_name in (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ):
        if not (checkpoint_dir / required_name).is_file():
            raise RuntimeError(
                f"Supervised v6 checkpoint is incomplete: {required_name}"
            )
    return checkpoint_dir


def load_audited_supervised_labeler(
    *,
    checkpoint_dir: Path,
    config: Mapping[str, Any],
    device: str,
) -> tuple[Any, Any]:
    """Load the already-verified fine-tuned v6 detector without network access."""

    if transformers.__version__ != str(
        config["model"]["transformers_version"]
    ):
        raise RuntimeError(
            "Expected transformers "
            f"{config['model']['transformers_version']}, "
            f"got {transformers.__version__}"
        )
    if config["model"]["local_files_only"] is not True:
        raise RuntimeError("Supervised labeler runtime must remain local-only")
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
    )

    processor = AutoImageProcessor.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    ).to(device)
    model.eval()
    return processor, model


def predict_helmet_boxes(
    *,
    processor: Any,
    model: Any,
    images: Sequence[Image.Image],
    device: str,
    score_floor: float,
    geometry_filter: Mapping[str, Any],
) -> list[list[tuple[float, list[float]]]]:
    """Predict class-0 helmet boxes with the frozen v6 geometry rule."""

    if not images:
        return []
    batch = processor(images=list(images), return_tensors="pt")
    pixel_values = batch["pixel_values"].to(device)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        outputs = model(pixel_values=pixel_values)
    target_sizes = torch.tensor(
        [[image.height, image.width] for image in images],
        dtype=torch.int64,
    )
    results = processor.post_process_object_detection(
        outputs,
        threshold=float(score_floor),
        target_sizes=target_sizes,
    )
    predictions = []
    for image, result in zip(images, results, strict=True):
        rows = [
            (
                float(score.item()),
                [float(value) for value in box.tolist()],
            )
            for score, label, box in zip(
                result["scores"],
                result["labels"],
                result["boxes"],
                strict=True,
            )
            if int(label.item()) == 0
        ]
        rows = filter_prediction_geometry(
            rows,
            image_width=image.width,
            image_height=image.height,
            max_relative_area=float(geometry_filter["max_relative_area"]),
            max_relative_height=float(geometry_filter["max_relative_height"]),
            min_aspect_ratio=float(
                geometry_filter.get("min_aspect_ratio", 0.0)
            ),
            max_aspect_ratio=float(
                geometry_filter.get("max_aspect_ratio", math.inf)
            ),
        )
        predictions.append(sorted(rows, key=lambda item: -item[0]))
    return predictions


def _rank(seed: int, value: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _group_ids(
    image_ids: Sequence[int],
    frozen: Mapping[int, Mapping[str, Any]],
) -> set[int]:
    return {int(frozen[image_id]["group_id"]) for image_id in image_ids}


def freeze_supervised_split(
    *,
    config: Mapping[str, Any],
    train_images: Mapping[int, Mapping[str, Any]],
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    frozen: Mapping[int, Mapping[str, Any]],
    helmet_category_id: int,
    zero_shot_calibration_ids: Sequence[int],
    zero_shot_audit_ids: Sequence[int],
) -> dict[str, Any]:
    """Freeze training, calibration, and a new untouched audit by source group."""

    prior_ids = sorted(
        {int(value) for value in zero_shot_calibration_ids}
        | {int(value) for value in zero_shot_audit_ids}
    )
    if not prior_ids or not set(prior_ids) <= set(train_images):
        raise RuntimeError("Prior zero-shot audit IDs are not Train-only")
    calibration_groups = _group_ids(prior_ids, frozen)
    candidates: dict[int, tuple[int, float]] = {}
    split_seed = int(config.get("split_seed", config.get("root_seed", -1)))
    if split_seed < 0:
        raise RuntimeError("Supervised split seed is missing")
    for image_id in sorted(train_images):
        group_id = int(frozen[image_id]["group_id"])
        if group_id in calibration_groups:
            continue
        helmet_boxes = [
            annotation["bbox"]
            for annotation in annotations[image_id]
            if int(annotation["category_id"]) == helmet_category_id
        ]
        if not helmet_boxes:
            continue
        image_area = int(train_images[image_id]["width"]) * int(
            train_images[image_id]["height"]
        )
        relative_areas = [
            float(box[2]) * float(box[3]) / image_area for box in helmet_boxes
        ]
        candidate = (image_id, float(np.median(relative_areas)))
        previous = candidates.get(group_id)
        if previous is None or _rank(split_seed, image_id) < _rank(
            split_seed,
            previous[0],
        ):
            candidates[group_id] = candidate

    ordered = sorted(
        [
            (group_id, image_id, area)
            for group_id, (image_id, area) in candidates.items()
        ],
        key=lambda item: (item[2], _rank(split_seed, item[0])),
    )
    requested = int(config["data"]["new_untouched_audit_images"])
    if requested % 4 != 0 or len(ordered) < requested:
        raise RuntimeError("Cannot form the registered stratified audit")
    per_quartile = requested // 4
    audit_ids: list[int] = []
    for quartile_index, source in enumerate(
        np.array_split(np.asarray(ordered, dtype=object), 4)
    ):
        ranked = sorted(
            source.tolist(),
            key=lambda item: _rank(
                split_seed + quartile_index,
                int(item[0]),
            ),
        )
        audit_ids.extend(int(row[1]) for row in ranked[:per_quartile])
    audit_ids.sort()
    audit_groups = _group_ids(audit_ids, frozen)
    if calibration_groups & audit_groups:
        raise AssertionError("Supervised calibration and audit groups overlap")
    excluded_groups = calibration_groups | audit_groups
    training_ids = sorted(
        image_id
        for image_id in train_images
        if int(frozen[image_id]["group_id"]) not in excluded_groups
    )
    training_groups = _group_ids(training_ids, frozen)
    if training_groups & excluded_groups:
        raise AssertionError("Supervised training group leakage")
    training_helmet_annotations = sum(
        int(annotation["category_id"]) == helmet_category_id
        for image_id in training_ids
        for annotation in annotations[image_id]
    )
    payload = {
        "schema_version": 1,
        "status": "frozen_before_supervised_training",
        "root_seed": split_seed,
        "source_split": "Train",
        "training_image_ids": training_ids,
        "calibration_image_ids": prior_ids,
        "untouched_audit_image_ids": audit_ids,
        "training_group_ids": sorted(training_groups),
        "calibration_group_ids": sorted(calibration_groups),
        "untouched_audit_group_ids": sorted(audit_groups),
        "training_images": len(training_ids),
        "training_groups": len(training_groups),
        "training_helmet_annotations": training_helmet_annotations,
        "calibration_images": len(prior_ids),
        "untouched_audit_images": len(audit_ids),
        "validation_images_read": 0,
        "test_images_read": 0,
    }
    if "split_seed" in config:
        payload["split_seed"] = split_seed
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return payload
