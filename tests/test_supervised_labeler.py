from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import pytest
from PIL import Image, ImageDraw

from scripts.diagnose_labeler_postprocessing import select_geometry_candidate
from scripts.diagnose_supervised_labeler_failure import diagnostic_thresholds
from scripts.record_supervised_labeler_v6_review import (
    build_review_evidence,
    parse_problem_cells,
)
from scripts.render_supervised_labeler_review import split_review_sheet
from scripts.train_supervised_labeler import select_calibration_candidate
from src.synthetic.grounded_labeler import load_whole_image_config
from src.synthetic.supervised_labeler import (
    filter_prediction_geometry,
    freeze_supervised_split,
    load_supervised_labeler_config,
    require_verified_audited_checkpoint,
    require_verified_model,
)
from src.synthetic.whole_image import human_review_evidence_sha256


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
    assert first["split_seed"] == 20260814
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


def test_passed_finetuned_checkpoint_is_rehashed_before_use(tmp_path) -> None:
    checkpoint_dir = tmp_path / "best"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model.safetensors"
    checkpoint.write_bytes(b"passed fine-tuned checkpoint")
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "preprocessor_config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    split_sha = "a" * 64
    config = {
        "experiment_id": "supervised_labeler_v6",
        "architecture": "rtdetr_v2_r50vd_helmet_only",
        "split_manifest_sha256": split_sha,
    }
    registration = {
        "experiment_id": config["experiment_id"],
        "architecture": config["architecture"],
        "checkpoint_sha256": checkpoint_sha,
        "split_manifest_sha256": split_sha,
        "score_threshold": 0.023,
        "max_relative_area": 0.08,
        "max_relative_height": 0.35,
        "audit_images": 48,
        "audit_precision": 0.90,
        "audit_recall": 0.86,
        "audit_median_matched_iou": 0.84,
    }
    report = {
        "status": "supervised_labeler_audit_passed",
        "checks": {
            "audit_precision": True,
            "audit_recall": True,
            "audit_median_matched_iou": True,
        },
        "split_manifest_sha256": split_sha,
        "checkpoint_path": str(checkpoint_dir),
        "checkpoint_sha256": checkpoint_sha,
        "best_calibration": {"threshold": 0.023},
        "audit_metrics": {
            "precision": 0.90,
            "recall": 0.86,
            "median_matched_iou": 0.84,
        },
        "postprocessing": {
            "max_relative_area": 0.08,
            "max_relative_height": 0.35,
        },
        "untouched_audit_images_read": 48,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    split = {
        "status": "frozen_before_supervised_training",
        "manifest_sha256": split_sha,
        "validation_images_read": 0,
        "test_images_read": 0,
    }

    assert require_verified_audited_checkpoint(
        config=config,
        registration=registration,
        report=report,
        split=split,
    ) == checkpoint_dir
    checkpoint.write_bytes(b"tampered fine-tuned checkpoint")

    with pytest.raises(RuntimeError, match="integrity"):
        require_verified_audited_checkpoint(
            config=config,
            registration=registration,
            report=report,
            split=split,
        )


def test_v6_human_review_record_is_canonical_and_owner_only() -> None:
    registration = load_whole_image_config()["supervised_labeler"]

    evidence = build_review_evidence(
        registration=registration,
        decision="approve",
        reviewed_on="2026-07-28",
        problem_cells=parse_problem_cells(""),
        note="All 48 cells reviewed.",
    )

    assert evidence["status"] == "approved_by_kuotunyu"
    assert evidence["reviewed_by"] == "kuotunyu"
    assert evidence["problem_count"] == 0
    assert evidence["problem_cells"] == []
    assert evidence["evidence_sha256"] == human_review_evidence_sha256(
        evidence
    )
    assert parse_problem_cells("43, 7,43") == [7, 43]


def test_v6_human_review_rejects_inconsistent_decisions() -> None:
    registration = load_whole_image_config()["supervised_labeler"]

    with pytest.raises(ValueError, match="approved review"):
        build_review_evidence(
            registration=registration,
            decision="approve",
            reviewed_on="2026-07-28",
            problem_cells=[7],
            note="",
        )
    with pytest.raises(ValueError, match="rejected review"):
        build_review_evidence(
            registration=registration,
            decision="reject",
            reviewed_on="2026-07-28",
            problem_cells=[],
            note="",
        )


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


def test_failure_diagnostic_grid_extends_below_frozen_threshold() -> None:
    thresholds = diagnostic_thresholds()

    assert thresholds == sorted(set(thresholds))
    assert thresholds[0] == 0.001
    assert thresholds[-1] == 0.05


def test_geometry_filter_drops_only_oversized_predictions() -> None:
    predictions = [
        (0.9, [10, 10, 30, 30]),
        (0.8, [0, 0, 100, 60]),
        (0.7, [20, 0, 40, 80]),
    ]

    kept = filter_prediction_geometry(
        predictions,
        image_width=100,
        image_height=100,
        max_relative_area=0.10,
        max_relative_height=0.50,
    )

    assert kept == [(0.9, [10.0, 10.0, 30.0, 30.0])]


def test_geometry_candidate_requires_precision_floor() -> None:
    rows = [
        {
            "threshold": 0.02,
            "max_relative_area": 0.08,
            "max_relative_height": 0.45,
            "precision": 0.86,
            "recall": 0.80,
            "f1": 0.83,
            "median_matched_iou": 0.8,
        },
        {
            "threshold": 0.03,
            "max_relative_area": 0.12,
            "max_relative_height": 0.60,
            "precision": 0.88,
            "recall": 0.72,
            "f1": 0.79,
            "median_matched_iou": 0.8,
        },
    ]

    selected = select_geometry_candidate(rows, precision_floor=0.87)

    assert selected is not None
    assert selected["threshold"] == 0.03


def test_human_review_sheet_splits_into_three_zoomable_pages(
    tmp_path,
) -> None:
    source = tmp_path / "review.png"
    sheet = Image.new("RGB", (1040, 3538), "white")
    draw = ImageDraw.Draw(sheet)
    for row in range(12):
        color = (row * 20, 0, 0)
        top = 58 + row * 290
        draw.rectangle((0, top, 1039, top + 289), fill=color)
    sheet.save(source)

    pages = split_review_sheet(source)

    assert len(pages) == 3
    assert [Image.open(path).size for path in pages] == [
        (1040, 1218),
        (1040, 1218),
        (1040, 1218),
    ]
    with Image.open(pages[1]) as second:
        assert second.getpixel((0, 58)) == (80, 0, 0)
