from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import pytest

from scripts.train_supervised_labeler import select_calibration_candidate
from src.synthetic.supervised_labeler import (
    freeze_supervised_split,
    load_supervised_labeler_config,
    require_verified_model,
)


def test_supervised_split_is_group_disjoint_and_deterministic() -> None:
    config = load_supervised_labeler_config()
    train_images = {}
    annotations = defaultdict(list)
    frozen = {}
    for image_id in range(1, 161):
        train_images[image_id] = {"width": 100, "height": 100}
        annotations[image_id].append(
            {
                "category_id": 1,
                "bbox": [10, 10, 5 + image_id % 20, 6 + image_id % 15],
            }
        )
        frozen[image_id] = {"group_id": image_id}

    first = freeze_supervised_split(
        config=config,
        train_images=train_images,
        annotations=annotations,
        frozen=frozen,
        helmet_category_id=1,
        zero_shot_calibration_ids=list(range(1, 49)),
        zero_shot_audit_ids=list(range(49, 97)),
    )
    second = freeze_supervised_split(
        config=config,
        train_images=train_images,
        annotations=annotations,
        frozen=frozen,
        helmet_category_id=1,
        zero_shot_calibration_ids=list(range(1, 49)),
        zero_shot_audit_ids=list(range(49, 97)),
    )

    assert first == second
    assert first["calibration_images"] == 96
    assert first["untouched_audit_images"] == 48
    assert set(first["training_group_ids"]).isdisjoint(
        first["calibration_group_ids"]
    )
    assert set(first["training_group_ids"]).isdisjoint(
        first["untouched_audit_group_ids"]
    )
    assert set(first["calibration_group_ids"]).isdisjoint(
        first["untouched_audit_group_ids"]
    )
    assert first["validation_images_read"] == 0
    assert first["test_images_read"] == 0


def test_supervised_model_is_rehashed_before_use(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(b"fixed checkpoint")
    config = {
        "model": {
            "repo_id": "owner/model",
            "revision": "fixed",
            "license": "apache-2.0",
            "required_download_bytes": len(b"fixed checkpoint"),
            "allow_files": {"model.safetensors": len(b"fixed checkpoint")},
        }
    }
    manifest = {
        "repo_id": "owner/model",
        "revision": "fixed",
        "license": "apache-2.0",
        "download_bytes": len(b"fixed checkpoint"),
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(b"fixed checkpoint"),
                "sha256": hashlib.sha256(b"fixed checkpoint").hexdigest(),
            }
        ],
    }
    (model_dir / "SAFESYNTH_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    require_verified_model(model_dir, config)
    weight_path.write_bytes(b"broken checkpoint")

    with pytest.raises(RuntimeError, match="failed integrity"):
        require_verified_model(model_dir, config)


def test_calibration_selection_never_weakens_precision_floor() -> None:
    rows = [
        {
            "epoch": 1,
            "threshold": 0.2,
            "precision": 0.60,
            "recall": 0.95,
            "f1": 0.74,
            "median_matched_iou": 0.8,
        },
        {
            "epoch": 1,
            "threshold": 0.4,
            "precision": 0.90,
            "recall": 0.70,
            "f1": 0.79,
            "median_matched_iou": 0.75,
        },
    ]

    selected = select_calibration_candidate(rows, precision_floor=0.85)

    assert selected is not None
    assert selected["threshold"] == 0.4
    assert select_calibration_candidate(rows[:1], precision_floor=0.85) is None
