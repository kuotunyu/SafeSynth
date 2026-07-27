"""Preregistered Train-only split and model integrity for a supervised labeler."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.paths import PROJECT_ROOT, ProjectPaths

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v3.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v3_split.json"


def load_supervised_labeler_config(
    path: Path = CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the frozen supervised-labeler registration."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    model = config["model"]
    if (
        model["license"] != "apache-2.0"
        or model["local_files_only"] is not True
        or model["repo_id"] != "PekingU/rtdetr_v2_r18vd"
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
    return config


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
