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
from scripts.render_supervised_labeler_review_separated import (
    _draw_model_boxes,
    extract_model_boxes,
)
from scripts.train_supervised_labeler import select_calibration_candidate
from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import load_whole_image_config
from src.synthetic.supervised_labeler import (
    filter_prediction_geometry,
    freeze_supervised_split,
    load_supervised_labeler_config,
    require_verified_audited_checkpoint,
    require_verified_model,
    supervised_sampling_weights,
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
    assert first["split_seed"] == 20260815
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


def test_v7_frozen_split_seals_new_audit_from_v6_history() -> None:
    config = load_supervised_labeler_config()
    v6 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v6_split.json"
        ).read_text(encoding="utf-8")
    )
    v7 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v7_split.json"
        ).read_text(encoding="utf-8")
    )
    revealed = set(v6["calibration_image_ids"]) | set(
        v6["untouched_audit_image_ids"]
    )

    assert config["status"] == "cpu_preflight_passed_gpu_training_waiting"
    assert config["split_manifest_sha256"] == v7["manifest_sha256"]
    assert set(v7["calibration_image_ids"]) == revealed
    assert set(v7["untouched_audit_image_ids"]).isdisjoint(revealed)
    assert set(v7["training_group_ids"]).isdisjoint(
        v7["calibration_group_ids"]
    )
    assert set(v7["training_group_ids"]).isdisjoint(
        v7["untouched_audit_group_ids"]
    )
    assert set(v7["calibration_group_ids"]).isdisjoint(
        v7["untouched_audit_group_ids"]
    )
    assert v7["untouched_audit_images"] == 48
    assert v7["validation_images_read"] == 0
    assert v7["test_images_read"] == 0


def test_v7_cpu_preflight_kept_all_pixels_and_gpu_sealed() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "supervised_labeler_v7_preflight.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "cpu_preflight_passed_gpu_training_waiting"
    assert report["sampling_weight_counts"] == {
        "1.0": 2035,
        "2.0": 1010,
    }
    assert report["source_group_overlap"] == 0
    assert report["training_pixels_read"] == 0
    assert report["calibration_pixels_read"] == 0
    assert report["sealed_audit_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["gpu_work_run"] is False
    assert report["whole_image_generation_run"] is False


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
    assert evidence["separated_pages"] == registration["human_review"][
        "separated_pages"
    ]
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


def test_v6_owner_rejection_evidence_is_frozen_and_gate_stays_closed() -> None:
    config = load_whole_image_config()
    review = config["supervised_labeler"]["human_review"]
    evidence = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "supervised_labeler_v6_human_review.json"
        ).read_text(encoding="utf-8")
    )

    assert config["status"] == "supervised_v6_human_review_rejected"
    assert review["status"] == "rejected_by_kuotunyu"
    assert review["problem_count"] == 9
    assert review["problem_cells"] == [4, 6, 7, 13, 23, 27, 38, 43, 45]
    assert config["generation_gate"]["allowed"] is False
    assert evidence["status"] == review["status"]
    assert evidence["reviewed_by"] == "kuotunyu"
    assert evidence["problem_cells"] == review["problem_cells"]
    assert evidence["problem_count"] == review["problem_count"]
    assert evidence["evidence_sha256"] == review["evidence_sha256"]
    assert evidence["evidence_sha256"] == human_review_evidence_sha256(
        evidence
    )
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False


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


def test_geometry_filter_rejects_extreme_aspect_and_degenerate_boxes() -> None:
    predictions = [
        (0.9, [10, 10, 30, 30]),
        (0.8, [0, 0, 60, 10]),
        (0.7, [0, 0, 10, 60]),
        (0.6, [4, 4, 4, 10]),
    ]

    kept = filter_prediction_geometry(
        predictions,
        image_width=100,
        image_height=100,
        max_relative_area=1.0,
        max_relative_height=1.0,
        min_aspect_ratio=0.25,
        max_aspect_ratio=4.0,
    )

    assert kept == [(0.9, [10.0, 10.0, 30.0, 30.0])]


def test_v7_sampling_weights_empty_and_close_pair_images() -> None:
    annotations = {
        1: [],
        2: [{"category_id": 1, "bbox": [0, 0, 10, 10]}],
        3: [
            {"category_id": 1, "bbox": [0, 0, 10, 10]},
            {"category_id": 1, "bbox": [8, 0, 10, 10]},
        ],
        4: [
            {"category_id": 1, "bbox": [0, 0, 10, 10]},
            {"category_id": 1, "bbox": [30, 0, 10, 10]},
        ],
    }

    weights = supervised_sampling_weights(
        image_ids=[1, 2, 3, 4],
        annotations=annotations,
        helmet_category_id=1,
        empty_image_weight=2.0,
        close_helmet_pair_weight=2.0,
        close_pair_ratio_max=1.0,
    )

    assert weights == [2.0, 1.0, 2.0, 1.0]


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


def test_separated_review_extracts_only_new_cyan_model_boxes() -> None:
    original = Image.new("RGB", (20, 20), (0, 220, 220))
    frozen = original.copy()
    draw = ImageDraw.Draw(frozen)
    draw.rectangle((2, 2, 8, 8), outline=(0, 255, 0), width=2)
    draw.rectangle((11, 11, 18, 18), outline=(0, 255, 255), width=2)
    draw.line((13, 11, 13, 15), fill=(0, 255, 255))
    draw.line((2, 12, 4, 12), fill=(0, 255, 255))
    draw.line((2, 12, 2, 16), fill=(0, 255, 255))

    boxes = extract_model_boxes(
        frozen_panel=frozen,
        original_panel=original,
    )

    assert boxes == [(11, 11, 18, 18)]


def test_separated_review_keeps_narrow_and_clipped_model_boxes() -> None:
    original = Image.new("RGB", (30, 30), "black")
    frozen = original.copy()
    draw = ImageDraw.Draw(frozen)
    draw.rectangle((3, 5, 7, 18), outline=(0, 255, 255), width=2)
    draw.line((20, 29, 29, 29), fill=(0, 255, 255), width=1)

    boxes = extract_model_boxes(
        frozen_panel=frozen,
        original_panel=original,
    )

    assert boxes == [(3, 5, 7, 18), (20, 29, 29, 29)]


def test_separated_review_draws_complete_thin_model_rectangle() -> None:
    image = Image.new("RGB", (20, 20), "black")

    _draw_model_boxes(
        image,
        [(3, 5, 7, 18)],
        source_size=(30, 30),
    )

    for x in range(2, 6):
        assert image.getpixel((x, 3)) == (255, 0, 255)
        assert image.getpixel((x, 12)) == (255, 0, 255)
    for y in range(3, 13):
        assert image.getpixel((2, y)) == (255, 0, 255)
        assert image.getpixel((5, y)) == (255, 0, 255)
    assert image.getpixel((3, 4)) == (0, 0, 0)
