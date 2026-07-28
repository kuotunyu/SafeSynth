from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import pytest
from PIL import Image, ImageDraw

from scripts.diagnose_labeler_postprocessing import select_geometry_candidate
from scripts.diagnose_supervised_labeler_failure import diagnostic_thresholds
from scripts.prepare_supervised_labeler_v12_gt_review import (
    EVIDENCE_PATH as V12_GT_EVIDENCE_PATH,
)
from scripts.prepare_supervised_labeler_v12_gt_review import (
    POOL_PATH as V12_GT_POOL_PATH,
)
from scripts.record_supervised_labeler_v6_review import (
    build_review_evidence,
    parse_problem_cells,
)
from scripts.render_supervised_labeler_review import split_review_sheet
from scripts.render_supervised_labeler_review_separated import (
    _draw_model_boxes,
    extract_model_boxes,
)
from scripts.train_supervised_labeler import (
    build_audit_evidence,
    select_calibration_candidate,
)
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
from src.synthetic.whole_image import (
    canonical_mapping_sha256,
    human_review_evidence_sha256,
)

V7_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v7.yaml"
V8_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v8.yaml"
V9_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v9.yaml"
V10_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v10.yaml"
V11_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v11.yaml"
SEMANTICS_ERRATUM_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_human_review_semantics_erratum.json"
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
    assert first["split_seed"] == 20260912
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
    config = load_supervised_labeler_config(V7_CONFIG_PATH)
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

    assert config["status"] == "human_review_rejected"
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


def test_v8_frozen_split_seals_new_audit_from_v7_history() -> None:
    config = load_supervised_labeler_config(V8_CONFIG_PATH)
    v7 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v7_split.json"
        ).read_text(encoding="utf-8")
    )
    v8 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v8_split.json"
        ).read_text(encoding="utf-8")
    )
    revealed = set(v7["calibration_image_ids"]) | set(
        v7["untouched_audit_image_ids"]
    )

    assert config["status"] == "human_review_rejected"
    assert config["split_manifest_sha256"] == v8["manifest_sha256"]
    assert set(v8["calibration_image_ids"]) == revealed
    assert set(v8["untouched_audit_image_ids"]).isdisjoint(revealed)
    assert set(v8["training_group_ids"]).isdisjoint(
        v8["calibration_group_ids"]
    )
    assert set(v8["training_group_ids"]).isdisjoint(
        v8["untouched_audit_group_ids"]
    )
    assert set(v8["calibration_group_ids"]).isdisjoint(
        v8["untouched_audit_group_ids"]
    )
    assert v8["untouched_audit_images"] == 48
    assert v8["validation_images_read"] == 0
    assert v8["test_images_read"] == 0


def test_v9_frozen_split_seals_new_audit_from_v8_history() -> None:
    config = load_supervised_labeler_config(V9_CONFIG_PATH)
    v8 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v8_split.json"
        ).read_text(encoding="utf-8")
    )
    v9 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v9_split.json"
        ).read_text(encoding="utf-8")
    )
    revealed = set(v8["calibration_image_ids"]) | set(
        v8["untouched_audit_image_ids"]
    )

    assert config["status"] == "human_review_rejected"
    assert config["architecture"] == "rtdetr_v2_r101vd_helmet_only"
    assert config["model"]["repo_id"] == "PekingU/rtdetr_v2_r101vd"
    assert config["split_manifest_sha256"] == v9["manifest_sha256"]
    assert set(v9["calibration_image_ids"]) == revealed
    assert set(v9["untouched_audit_image_ids"]).isdisjoint(revealed)
    assert set(v9["training_group_ids"]).isdisjoint(
        v9["calibration_group_ids"]
    )
    assert set(v9["training_group_ids"]).isdisjoint(
        v9["untouched_audit_group_ids"]
    )
    assert set(v9["calibration_group_ids"]).isdisjoint(
        v9["untouched_audit_group_ids"]
    )
    assert v9["training_images"] == 2946
    assert v9["calibration_images"] == 480
    assert v9["untouched_audit_images"] == 48
    assert v9["validation_images_read"] == 0
    assert v9["test_images_read"] == 0


def test_v10_frozen_split_seals_new_audit_from_v9_history() -> None:
    config = load_supervised_labeler_config(V10_CONFIG_PATH)
    v9 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v9_split.json"
        ).read_text(encoding="utf-8")
    )
    v10 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v10_split.json"
        ).read_text(encoding="utf-8")
    )
    revealed = set(v9["calibration_image_ids"]) | set(
        v9["untouched_audit_image_ids"]
    )

    assert config["status"] == "human_review_rejected"
    assert config["architecture"] == "rtdetr_v2_r101vd_helmet_only"
    assert config["model"]["repo_id"] == "PekingU/rtdetr_v2_r101vd"
    assert config["split_manifest_sha256"] == v10["manifest_sha256"]
    assert set(v10["calibration_image_ids"]) == revealed
    assert set(v10["untouched_audit_image_ids"]).isdisjoint(revealed)
    assert set(v10["training_group_ids"]).isdisjoint(
        v10["calibration_group_ids"]
    )
    assert set(v10["training_group_ids"]).isdisjoint(
        v10["untouched_audit_group_ids"]
    )
    assert set(v10["calibration_group_ids"]).isdisjoint(
        v10["untouched_audit_group_ids"]
    )
    assert v10["training_images"] == 2893
    assert v10["calibration_images"] == 528
    assert v10["untouched_audit_images"] == 48
    assert v10["validation_images_read"] == 0
    assert v10["test_images_read"] == 0


def test_v11_split_quarantines_gt_defects_and_seals_new_audit() -> None:
    config = load_supervised_labeler_config(V11_CONFIG_PATH)
    v10 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v10_split.json"
        ).read_text(encoding="utf-8")
    )
    v11 = json.loads(
        (
            PROJECT_ROOT / "splits" / "supervised_labeler_v11_split.json"
        ).read_text(encoding="utf-8")
    )
    revealed = set(v10["calibration_image_ids"]) | set(
        v10["untouched_audit_image_ids"]
    )
    quarantined = {3060, 4155, 4364}

    canonical = dict(v11)
    embedded_sha = canonical.pop("manifest_sha256")
    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert config["split_manifest_sha256"] == embedded_sha
    assert set(v11["quarantined_gt_defect_image_ids"]) == quarantined
    assert set(v11["calibration_image_ids"]) | quarantined == revealed
    assert set(v11["calibration_image_ids"]).isdisjoint(quarantined)
    assert set(v11["untouched_audit_image_ids"]).isdisjoint(revealed)
    assert set(v11["training_group_ids"]).isdisjoint(
        v11["calibration_group_ids"]
    )
    assert set(v11["training_group_ids"]).isdisjoint(
        v11["untouched_audit_group_ids"]
    )
    assert set(v11["calibration_group_ids"]).isdisjoint(
        v11["untouched_audit_group_ids"]
    )
    assert v11["training_images"] == 2842
    assert v11["calibration_images"] == 573
    assert v11["quarantined_gt_defect_images"] == 3
    assert v11["untouched_audit_images"] == 48
    assert v11["validation_images_read"] == 0
    assert v11["test_images_read"] == 0


def test_v11_geometry_registration_matches_revealed_diagnosis() -> None:
    config = load_supervised_labeler_config(V11_CONFIG_PATH)
    evidence = config["diagnostic_evidence"]["geometry_diagnosis"]
    path = PROJECT_ROOT / evidence["source"]
    diagnosis = json.loads(path.read_text(encoding="utf-8"))
    recommended = diagnosis["recommended_candidate"]

    assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence[
        "file_sha256"
    ]
    assert recommended["name"] == "edge_large_060"
    assert recommended["owner_geometry_recovered"] == 4
    assert config["postprocessing"] == {
        "max_aspect_ratio": 4.0,
        "max_relative_area": 0.15,
        "max_relative_height": 0.60,
        "min_aspect_ratio": 0.20,
        "selection_basis": config["postprocessing"]["selection_basis"],
    }
    assert diagnosis["revealed_images_read"] == 573
    assert diagnosis["quarantined_images_read"] == 0
    assert diagnosis["validation_images_read"] == 0
    assert diagnosis["test_images_read"] == 0
    assert diagnosis["whole_image_generation_run"] is False
    assert config["generation_gate"]["allowed"] is False


def test_v11_cpu_preflight_keeps_quarantine_and_new_audit_sealed() -> None:
    config = load_supervised_labeler_config(V11_CONFIG_PATH)
    outcome = config["cpu_preflight_outcome"]
    path = PROJECT_ROOT / outcome["report_path"]
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == outcome[
        "report_file_sha256"
    ]
    assert report["status"] == (
        "cpu_normalization_preflight_passed_gpu_smoke_waiting"
    )
    assert report["training"]["images_read"] == 2842
    assert report["training"]["normalized_images"] == 2825
    assert report["training"]["source_helmet_annotations"] == 10556
    assert report["training"]["transformed_helmet_annotations"] == 7259
    assert report["training"]["invalid_boxes"] == 0
    assert report["calibration"]["images_read"] == 573
    assert report["calibration"]["normalized_images"] == 570
    assert report["calibration"]["source_helmet_annotations"] == 2328
    assert report["calibration"]["transformed_helmet_annotations"] == 1547
    assert report["calibration"]["invalid_boxes"] == 0
    assert report["sampling_weight_counts"] == {"1.0": 776, "2.0": 2066}
    assert set(report["revealed_v10_model_problem_images"]) == {
        "478",
        "550",
        "708",
        "2515",
        "2826",
        "3222",
        "3950",
        "3975",
        "4821",
    }
    assert report["quarantined_gt_defect_image_ids"] == [3060, 4155, 4364]
    assert report["quarantined_gt_defect_pixels_read"] == 0
    assert report["sealed_audit_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["gpu_work_run"] is False
    assert report["whole_image_generation_run"] is False


def test_v11_gpu_smoke_keeps_calibration_and_new_audit_sealed() -> None:
    config = load_supervised_labeler_config(V11_CONFIG_PATH)
    outcome = config["gpu_smoke_outcome"]
    path = PROJECT_ROOT / outcome["report_path"]
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == outcome[
        "report_file_sha256"
    ]
    assert report["status"] == "smoke_passed"
    assert report["batch_size"] == 8
    assert report["helmet_boxes"] == 32
    assert report["loss"] == pytest.approx(287.3628234863281)
    assert report["peak_vram_gib"] == pytest.approx(9.315727710723877)
    assert report["untouched_audit_images_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert config["generation_gate"]["allowed"] is False


def test_v11_numeric_audit_pass_is_frozen_but_generation_stays_closed() -> None:
    config = load_supervised_labeler_config(V11_CONFIG_PATH)
    outcome = config["numeric_audit_outcome"]
    report_path = PROJECT_ROOT / outcome["report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence_path = PROJECT_ROOT / outcome["audit_evidence_path"]

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == outcome[
        "report_file_sha256"
    ]
    assert config["status"] == "human_review_rejected"
    assert report["status"] == "supervised_labeler_audit_passed"
    assert report["checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert report["best_calibration"]["epoch"] == 3
    assert report["best_calibration"]["threshold"] == 0.03
    assert report["audit_metrics"] == {
        "f1": pytest.approx(0.8527131782945736),
        "false_negatives": 26,
        "false_positives": 12,
        "median_matched_iou": pytest.approx(0.8429845884642673),
        "precision": pytest.approx(0.9016393442622951),
        "recall": pytest.approx(0.8088235294117647),
        "true_positives": 110,
    }
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == outcome[
        "audit_evidence_sha256"
    ]
    assert report["checkpoint_sha256"] == outcome["checkpoint_sha256"]
    assert report["untouched_audit_images_read"] == 48
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["whole_image_generation_run"] is False
    assert config["generation_gate"]["allowed"] is False


def test_v11_owner_review_manifest_freezes_every_presented_file() -> None:
    config = load_supervised_labeler_config(V11_CONFIG_PATH)
    outcome = config["numeric_audit_outcome"]
    path = PROJECT_ROOT / outcome["owner_review_manifest_path"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256")
    registration = manifest["registration"]

    assert hashlib.sha256(path.read_bytes()).hexdigest() == outcome[
        "owner_review_manifest_file_sha256"
    ]
    assert embedded_sha == outcome["owner_review_manifest_sha256"]
    assert canonical_mapping_sha256(manifest) == embedded_sha
    assert manifest["status"] == "v11_owner_review_files_frozen"
    assert len(registration["human_review"]["pages"]) == 3
    assert len(registration["human_review"]["separated_pages"]) == 3
    assert registration["audit_evidence"]["sha256"] == (
        "fd82c50ad485e32447fccfc824926005d987a3debca49ba1e1ed60c90c8b1586"
    )
    assert registration["checkpoint_sha256"] == (
        "9b5ee6d360d3768a52ba9261f444e23b6f1da2c56f2eca682e207160604da5c4"
    )
    assert registration["score_threshold"] == 0.03
    assert registration["max_relative_area"] == 0.15
    assert registration["max_relative_height"] == 0.60
    assert registration["min_aspect_ratio"] == 0.20
    assert registration["max_aspect_ratio"] == 4.0
    assert registration["quarantined_gt_defect_image_ids"] == [
        3060,
        4155,
        4364,
    ]
    assert manifest["validation_images_read"] == 0
    assert manifest["test_images_read"] == 0
    assert manifest["whole_image_generation_run"] is False


def test_v11_owner_rejection_and_label_semantics_are_canonical() -> None:
    config = load_supervised_labeler_config(V11_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    path = PROJECT_ROOT / outcome["evidence_path"]
    evidence = json.loads(path.read_text(encoding="utf-8"))

    canonical = dict(evidence)
    embedded_sha = canonical.pop("evidence_sha256")
    assert embedded_sha == human_review_evidence_sha256(canonical)
    assert embedded_sha == outcome["evidence_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == outcome[
        "evidence_file_sha256"
    ]
    assert evidence["status"] == "rejected_by_kuotunyu"
    assert evidence["reviewed_by"] == "kuotunyu"
    assert evidence["problem_cells"] == [
        3,
        6,
        7,
        8,
        14,
        17,
        18,
        19,
        24,
        44,
    ]
    assert outcome["label_semantics"] == (
        "class_direct_helmeted_head_region"
    )
    assert outcome["categories"] == {
        "dataset_gt_missed_helmeted_head": [3],
        "model_missed_helmeted_head": [6, 7, 8, 14, 17, 18, 19, 24, 44],
    }
    assert config["generation_gate"]["allowed"] is False
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False


def test_human_review_semantics_erratum_keeps_generation_locked() -> None:
    erratum = json.loads(SEMANTICS_ERRATUM_PATH.read_text(encoding="utf-8"))
    v11_config = load_supervised_labeler_config(V11_CONFIG_PATH)
    whole_image_config = load_whole_image_config()

    assert erratum["status"] == "active_superseding_erratum"
    assert erratum["canonical_protocol"]["label_semantics"] == (
        "class_direct_helmeted_head_region"
    )
    assert erratum["confirmed_impact_example"] == {
        "experiment_id": "supervised_labeler_v10",
        "cell": 42,
        "image_id": 4364,
        "old_interpretation": (
            "the dataset GT and model both missed a loose hard hat"
        ),
        "correct_interpretation": (
            "the loose unworn hard hat is a negative and should have neither "
            "a green nor a magenta helmeted-head box"
        ),
    }
    assert [item["experiment_id"] for item in erratum["affected_reviews"]] == [
        f"supervised_labeler_v{version}" for version in range(6, 12)
    ]
    for item in erratum["affected_reviews"]:
        path = PROJECT_ROOT / item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item[
            "file_sha256"
        ]
        assert item["disposition"] == (
            "diagnostic_only_semantics_inconclusive"
        )

    assert erratum["downstream_effect"][
        "historical_human_outcomes_valid_for_generation_gate"
    ] is False
    assert erratum["downstream_effect"]["generation_allowed"] is False
    assert erratum["downstream_effect"]["validation_images_read"] == 0
    assert erratum["downstream_effect"]["test_images_read"] == 0
    assert erratum["downstream_effect"]["whole_image_generation_run"] is False
    assert v11_config["posthoc_semantics_erratum"]["active"] is True
    assert v11_config["posthoc_semantics_erratum"][
        "original_human_outcome_valid_for_generation_gate"
    ] is False
    assert v11_config["generation_gate"]["allowed"] is False
    assert whole_image_config["review_semantics_erratum"]["active"] is True
    assert whole_image_config["generation_gate"]["allowed"] is False


def test_v12_gt_only_pool_is_frozen_before_model_output() -> None:
    if not V12_GT_POOL_PATH.is_file():
        pytest.skip("v12 GT-only pool has not been frozen yet")
    pool = json.loads(V12_GT_POOL_PATH.read_text(encoding="utf-8"))
    canonical = dict(pool)
    embedded_sha = canonical.pop("manifest_sha256")
    split_manifest = json.loads(
        (PROJECT_ROOT / "splits" / "split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = {
        int(row["image_id"]): row for row in split_manifest["images"]
    }
    selected = [
        *pool["primary_cases"],
        *pool["sealed_reserve_cases"],
    ]
    selected_groups = {int(row["group_id"]) for row in selected}
    revealed_groups = {
        int(value)
        for version in range(2, 12)
        for key in (
            "calibration_group_ids",
            "untouched_audit_group_ids",
            "quarantined_gt_defect_group_ids",
        )
        for value in json.loads(
            (
                PROJECT_ROOT
                / "splits"
                / f"supervised_labeler_v{version}_split.json"
            ).read_text(encoding="utf-8")
        ).get(key, [])
    }

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert pool["status"] == (
        "v12_gt_only_pool_frozen_before_pixel_review"
    )
    assert pool["label_semantics"] == "class_direct_helmeted_head_region"
    assert pool["primary_images"] == 64
    assert pool["sealed_reserve_images"] == 32
    assert len(selected_groups) == 96
    assert selected_groups.isdisjoint(revealed_groups)
    assert all(
        frozen[int(row["image_id"])]["split"] == "train"
        and int(frozen[int(row["image_id"])]["group_id"])
        == int(row["group_id"])
        for row in selected
    )
    assert pool["primary_pixels_read"] == 0
    assert pool["sealed_reserve_pixels_read"] == 0
    assert pool["model_inference_run"] is False
    assert pool["validation_images_read"] == 0
    assert pool["test_images_read"] == 0
    assert pool["whole_image_generation_run"] is False


def test_v12_gt_only_primary_review_contains_no_model_output() -> None:
    if not V12_GT_EVIDENCE_PATH.is_file():
        pytest.skip("v12 GT-only primary review has not been rendered yet")
    evidence = json.loads(
        V12_GT_EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    pool = json.loads(V12_GT_POOL_PATH.read_text(encoding="utf-8"))
    canonical = dict(evidence)
    embedded_sha = canonical.pop("evidence_sha256")
    primary_ids = [int(row["image_id"]) for row in pool["primary_cases"]]

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert evidence["status"] == "v12_gt_only_primary_review_rendered"
    assert evidence["review_stage"] == "gt_only"
    assert evidence["label_semantics"] == (
        "class_direct_helmeted_head_region"
    )
    assert evidence["pool_manifest_sha256"] == pool["manifest_sha256"]
    assert [int(row["image_id"]) for row in evidence["cases"]] == primary_ids
    assert len(evidence["cases"]) == 64
    assert all(
        not row["truth_boxes"]
        if row["stratum"] == "dataset_gt_empty"
        else bool(row["truth_boxes"])
        for row in evidence["cases"]
    )
    assert evidence["model_boxes_present"] is False
    assert evidence["model_inference_run"] is False
    assert evidence["primary_images_read"] == 64
    assert evidence["primary_images_normalized"] == 64
    assert evidence["sealed_reserve_images"] == 32
    assert evidence["sealed_reserve_pixels_read"] == 0
    assert len(evidence["pages"]) == 4
    for page in evidence["pages"]:
        path = PROJECT_ROOT / page["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page["sha256"]
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False


def test_v10_cpu_normalization_preflight_keeps_new_audit_sealed() -> None:
    config = load_supervised_labeler_config(V10_CONFIG_PATH)
    outcome = config["cpu_preflight_outcome"]
    path = PROJECT_ROOT / outcome["report_path"]
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == outcome[
        "report_file_sha256"
    ]
    assert report["status"] == (
        "cpu_normalization_preflight_passed_gpu_smoke_waiting"
    )
    assert report["training"]["images_read"] == 2893
    assert report["training"]["normalized_images"] == 2875
    assert report["training"]["source_helmet_annotations"] == 10775
    assert report["training"]["transformed_helmet_annotations"] == 7407
    assert report["training"]["invalid_boxes"] == 0
    assert report["calibration"]["images_read"] == 528
    assert report["calibration"]["normalized_images"] == 525
    assert report["calibration"]["source_helmet_annotations"] == 2157
    assert report["calibration"]["transformed_helmet_annotations"] == 1433
    assert report["calibration"]["invalid_boxes"] == 0
    assert report["sampling_weight_counts"] == {"1.0": 787, "2.0": 2106}
    assert set(report["v9_problem_images"]) == {
        "345",
        "1027",
        "1124",
        "3569",
    }
    assert all(
        record["applied"]
        for record in report["v9_problem_images"].values()
    )
    assert report["sealed_audit_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["gpu_work_run"] is False
    assert report["whole_image_generation_run"] is False


def test_v10_gpu_smoke_keeps_calibration_and_new_audit_sealed() -> None:
    config = load_supervised_labeler_config(V10_CONFIG_PATH)
    outcome = config["gpu_smoke_outcome"]
    path = PROJECT_ROOT / outcome["report_path"]
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == outcome[
        "report_file_sha256"
    ]
    assert report["status"] == "smoke_passed"
    assert report["batch_size"] == 8
    assert report["helmet_boxes"] == 32
    assert report["loss"] == pytest.approx(287.3628234863281)
    assert report["peak_vram_gib"] == pytest.approx(9.315727710723877)
    assert report["untouched_audit_images_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert config["generation_gate"]["allowed"] is False


def test_v10_numeric_audit_pass_is_frozen_but_generation_stays_closed() -> None:
    config = load_supervised_labeler_config(V10_CONFIG_PATH)
    outcome = config["numeric_audit_outcome"]
    report_path = PROJECT_ROOT / outcome["report_path"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence_path = PROJECT_ROOT / outcome["audit_evidence_path"]

    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == outcome[
        "report_file_sha256"
    ]
    assert config["status"] == "human_review_rejected"
    assert report["status"] == "supervised_labeler_audit_passed"
    assert report["checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert report["best_calibration"]["epoch"] == 2
    assert report["best_calibration"]["threshold"] == 0.05
    assert report["audit_metrics"] == {
        "f1": pytest.approx(0.8412698412698412),
        "false_negatives": 16,
        "false_positives": 24,
        "median_matched_iou": pytest.approx(0.8043828728197762),
        "precision": pytest.approx(0.8153846153846154),
        "recall": pytest.approx(0.8688524590163934),
        "true_positives": 106,
    }
    assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == outcome[
        "audit_evidence_sha256"
    ]
    assert report["checkpoint_sha256"] == outcome["checkpoint_sha256"]
    assert report["untouched_audit_images_read"] == 48
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["whole_image_generation_run"] is False
    assert config["generation_gate"]["allowed"] is False


def test_v10_owner_review_manifest_freezes_every_presented_file() -> None:
    config = load_supervised_labeler_config(V10_CONFIG_PATH)
    outcome = config["numeric_audit_outcome"]
    path = PROJECT_ROOT / outcome["owner_review_manifest_path"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256")
    registration = manifest["registration"]

    assert embedded_sha == outcome["owner_review_manifest_sha256"]
    assert canonical_mapping_sha256(manifest) == embedded_sha
    assert manifest["status"] == "v10_owner_review_files_frozen"
    assert len(registration["human_review"]["pages"]) == 3
    assert len(registration["human_review"]["separated_pages"]) == 3
    assert registration["audit_evidence"]["sha256"] == (
        "7d04f00bc880061e6f0007c1853dd69dbbbdedd66f092c19a486b7044b3ed30d"
    )
    assert registration["checkpoint_sha256"] == (
        "e987c97fa72f68a80520afa237c3d7b00ca9d27af10853b95ef154a68a7d35bb"
    )
    assert registration["score_threshold"] == 0.05
    assert registration["input_normalization"] == config[
        "input_normalization"
    ]
    assert manifest["validation_images_read"] == 0
    assert manifest["test_images_read"] == 0
    assert manifest["whole_image_generation_run"] is False


def test_v10_owner_rejection_and_gt_defects_are_canonical() -> None:
    config = load_supervised_labeler_config(V10_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    path = PROJECT_ROOT / outcome["evidence_path"]
    evidence = json.loads(path.read_text(encoding="utf-8"))

    canonical = dict(evidence)
    embedded_sha = canonical.pop("evidence_sha256")
    assert embedded_sha == human_review_evidence_sha256(canonical)
    assert embedded_sha == outcome["evidence_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == outcome[
        "evidence_file_sha256"
    ]
    assert evidence["status"] == "rejected_by_kuotunyu"
    assert evidence["reviewed_by"] == "kuotunyu"
    assert evidence["problem_cells"] == [
        6,
        7,
        10,
        27,
        29,
        31,
        34,
        39,
        40,
        41,
        42,
        47,
    ]
    assert outcome["categories"] == {
        "model_missed_helmet": [6, 7, 10, 27, 39, 40],
        "model_false_positive": [29, 34, 47],
        "dataset_gt_false_positive_label": [31],
        "dataset_gt_missed_helmet": [41],
        "model_and_dataset_gt_missed_helmet": [42],
    }
    assert outcome["owner_confirmed_gt_defect_cells"] == [31, 41, 42]
    assert outcome["numeric_audit_contaminated_by_gt_defects"] is True
    assert config["generation_gate"]["allowed"] is False
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False


def test_v10_revealed_diagnosis_separates_model_and_gt_failures() -> None:
    config = load_supervised_labeler_config(V10_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    path = PROJECT_ROOT / outcome["diagnosis_path"]
    diagnosis = json.loads(path.read_text(encoding="utf-8"))
    cases = {int(case["cell"]): case for case in diagnosis["problem_cases"]}

    assert hashlib.sha256(path.read_bytes()).hexdigest() == outcome[
        "diagnosis_file_sha256"
    ]
    assert diagnosis["eligible_for_generation_gate"] is False
    assert diagnosis["owner_confirmed_gt_defect_cells"] == [31, 41, 42]
    assert diagnosis["audit_numeric_metrics_contaminated_by_gt_defects"] is True
    assert diagnosis["owner_category_counts"] == {
        "dataset_gt_false_positive_label": 1,
        "dataset_gt_missed_helmet": 1,
        "model_and_dataset_gt_missed_helmet": 1,
        "model_false_positive": 3,
        "model_missed_helmet": 6,
    }
    assert diagnosis["automatic_miss_reason_counts_against_dataset_gt"] == {
        "matching_box_below_frozen_score_threshold": 4,
        "removed_by_frozen_geometry_filter": 4,
    }
    assert diagnosis["max_owner_false_positive_score"] == pytest.approx(
        0.09423828125
    )
    assert cases[6]["misses_against_dataset_gt"][0][
        "best_raw_candidate"
    ]["score"] == pytest.approx(0.1953125)
    assert cases[7]["misses_against_dataset_gt"][0][
        "best_raw_candidate"
    ]["score"] == pytest.approx(0.1416015625)
    assert cases[40]["misses_against_dataset_gt"][0][
        "best_raw_candidate"
    ]["score"] == pytest.approx(0.1875)
    assert cases[31]["numeric_metrics_reliable"] is False
    assert cases[41]["numeric_metrics_reliable"] is False
    assert cases[42]["numeric_metrics_reliable"] is False
    assert diagnosis["validation_images_read"] == 0
    assert diagnosis["test_images_read"] == 0
    assert diagnosis["whole_image_generation_run"] is False


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


def test_v8_cpu_preflight_kept_all_pixels_and_gpu_sealed() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "supervised_labeler_v8_preflight.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "cpu_preflight_passed_gpu_training_waiting"
    assert report["sampling_weight_counts"] == {
        "1.0": 821,
        "2.0": 2174,
    }
    assert report["source_group_overlap"] == 0
    assert report["training_pixels_read"] == 0
    assert report["calibration_pixels_read"] == 0
    assert report["sealed_audit_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["gpu_work_run"] is False
    assert report["whole_image_generation_run"] is False


def test_v9_cpu_preflight_kept_all_pixels_and_gpu_sealed() -> None:
    report = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "supervised_labeler_v9_preflight.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "cpu_preflight_passed_gpu_training_waiting"
    assert report["sampling_weight_counts"] == {
        "1.0": 804,
        "2.0": 2142,
    }
    assert report["model"]["repo_id"] == "PekingU/rtdetr_v2_r101vd"
    assert report["model"]["revision"] == (
        "2c5dbbd2d4d8c8814827a3b42737ba1afce3cf2a"
    )
    assert report["source_group_overlap"] == 0
    assert report["training_pixels_read"] == 0
    assert report["calibration_pixels_read"] == 0
    assert report["sealed_audit_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["gpu_work_run"] is False
    assert report["whole_image_generation_run"] is False


def test_v7_gpu_smoke_never_reads_sealed_audit_or_val_test() -> None:
    report = json.loads(
        (
            PROJECT_ROOT / "reports" / "supervised_labeler_v7_smoke.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "smoke_passed"
    assert report["batch_size"] == 8
    assert report["untouched_audit_images_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0


def test_v8_gpu_smoke_never_reads_sealed_audit_or_val_test() -> None:
    report = json.loads(
        (
            PROJECT_ROOT / "reports" / "supervised_labeler_v8_smoke.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "smoke_passed"
    assert report["batch_size"] == 8
    assert report["untouched_audit_images_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0


def test_v9_gpu_smoke_never_reads_sealed_audit_or_val_test() -> None:
    report = json.loads(
        (
            PROJECT_ROOT / "reports" / "supervised_labeler_v9_smoke.json"
        ).read_text(encoding="utf-8")
    )

    assert report["status"] == "smoke_passed"
    assert report["batch_size"] == 8
    assert report["loss"] == pytest.approx(226.901611328125)
    assert report["peak_vram_gib"] == pytest.approx(9.324644088745117)
    assert report["untouched_audit_images_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0


def test_v7_numeric_audit_pass_is_frozen_but_generation_stays_closed() -> None:
    config = load_supervised_labeler_config(V7_CONFIG_PATH)
    report_path = (
        PROJECT_ROOT / "reports" / "supervised_labeler_v7_training.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence_path = (
        PROJECT_ROOT
        / "reports"
        / "supervised_labeler_v7_audit_evidence.json"
    )
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    assert report["status"] == "supervised_labeler_audit_passed"
    assert report["checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert report["best_calibration"]["epoch"] == 7
    assert report["best_calibration"]["threshold"] == 0.023
    assert report["audit_metrics"]["precision"] == pytest.approx(
        0.9578947368421052
    )
    assert report["audit_metrics"]["recall"] == pytest.approx(
        0.9054726368159204
    )
    assert report["audit_metrics"]["median_matched_iou"] == pytest.approx(
        0.8511458832997453
    )
    assert evidence_sha == report["audit_evidence_sha256"]
    assert report["untouched_audit_images_read"] == 48
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["whole_image_generation_run"] is False
    assert config["generation_gate"]["allowed"] is False


def test_v8_numeric_audit_pass_is_frozen_but_generation_stays_closed() -> None:
    config = load_supervised_labeler_config(V8_CONFIG_PATH)
    report_path = (
        PROJECT_ROOT / "reports" / "supervised_labeler_v8_training.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence_path = (
        PROJECT_ROOT
        / "reports"
        / "supervised_labeler_v8_audit_evidence.json"
    )
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    assert config["status"] == "human_review_rejected"
    assert report["status"] == "supervised_labeler_audit_passed"
    assert report["checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert report["best_calibration"]["epoch"] == 6
    assert report["best_calibration"]["threshold"] == 0.035
    assert report["audit_metrics"]["precision"] == pytest.approx(
        0.9213483146067416
    )
    assert report["audit_metrics"]["recall"] == pytest.approx(
        0.8677248677248677
    )
    assert report["audit_metrics"]["median_matched_iou"] == pytest.approx(
        0.8442019578744363
    )
    assert evidence_sha == report["audit_evidence_sha256"]
    assert report["untouched_audit_images_read"] == 48
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["whole_image_generation_run"] is False
    assert config["generation_gate"]["allowed"] is False


def test_v9_numeric_audit_pass_is_frozen_but_generation_stays_closed() -> None:
    config = load_supervised_labeler_config(V9_CONFIG_PATH)
    report_path = (
        PROJECT_ROOT / "reports" / "supervised_labeler_v9_training.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence_path = (
        PROJECT_ROOT
        / "reports"
        / "supervised_labeler_v9_audit_evidence.json"
    )
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    assert config["status"] == "human_review_rejected"
    assert report["status"] == "supervised_labeler_audit_passed"
    assert report["checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert report["best_calibration"]["epoch"] == 2
    assert report["best_calibration"]["threshold"] == 0.05
    assert report["audit_metrics"]["precision"] == pytest.approx(
        0.9312169312169312
    )
    assert report["audit_metrics"]["recall"] == pytest.approx(
        0.9072164948453608
    )
    assert report["audit_metrics"]["median_matched_iou"] == pytest.approx(
        0.8151169809813681
    )
    assert evidence_sha == report["audit_evidence_sha256"]
    assert report["untouched_audit_images_read"] == 48
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["whole_image_generation_run"] is False
    assert config["generation_gate"]["allowed"] is False


def test_v7_owner_review_manifest_freezes_every_presented_file() -> None:
    path = (
        PROJECT_ROOT
        / "reports"
        / "supervised_labeler_v7_review_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256")
    registration = manifest["registration"]

    assert canonical_mapping_sha256(manifest) == embedded_sha
    assert manifest["status"] == "v7_owner_review_files_frozen"
    assert len(registration["human_review"]["pages"]) == 3
    assert len(registration["human_review"]["separated_pages"]) == 3
    assert registration["audit_evidence"]["sha256"] == (
        "3a36ba7ee0a66c7764fddbf4e3cecc92136751b5eb26ae0759727370957b832b"
    )
    assert manifest["validation_images_read"] == 0
    assert manifest["test_images_read"] == 0
    assert manifest["whole_image_generation_run"] is False


def test_v8_owner_review_manifest_freezes_every_presented_file() -> None:
    path = (
        PROJECT_ROOT
        / "reports"
        / "supervised_labeler_v8_review_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256")
    registration = manifest["registration"]

    assert canonical_mapping_sha256(manifest) == embedded_sha
    assert manifest["status"] == "v8_owner_review_files_frozen"
    assert len(registration["human_review"]["pages"]) == 3
    assert len(registration["human_review"]["separated_pages"]) == 3
    assert registration["audit_evidence"]["sha256"] == (
        "99ce9cf898d4ab1da7d90e43c780511464341bd6184d52288ea4a424c3fda41a"
    )
    assert registration["max_relative_area"] == 0.14
    assert registration["max_relative_height"] == 0.40
    assert manifest["validation_images_read"] == 0
    assert manifest["test_images_read"] == 0
    assert manifest["whole_image_generation_run"] is False


def test_v9_owner_review_manifest_freezes_every_presented_file() -> None:
    path = (
        PROJECT_ROOT
        / "reports"
        / "supervised_labeler_v9_review_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    embedded_sha = manifest.pop("manifest_sha256")
    registration = manifest["registration"]

    assert canonical_mapping_sha256(manifest) == embedded_sha
    assert manifest["status"] == "v9_owner_review_files_frozen"
    assert len(registration["human_review"]["pages"]) == 3
    assert len(registration["human_review"]["separated_pages"]) == 3
    assert registration["audit_evidence"]["sha256"] == (
        "5652d57750f3471ee8b2b7a1ef0d9644fbed062ca6d66d083cff51a8be949dd6"
    )
    assert registration["architecture"] == "rtdetr_v2_r101vd_helmet_only"
    assert registration["score_threshold"] == 0.05
    assert manifest["validation_images_read"] == 0
    assert manifest["test_images_read"] == 0
    assert manifest["whole_image_generation_run"] is False


def test_v7_owner_review_rejection_is_canonical_and_generation_stays_closed() -> None:
    config = load_supervised_labeler_config(V7_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    path = PROJECT_ROOT / outcome["evidence_path"]
    evidence = json.loads(path.read_text(encoding="utf-8"))

    canonical = dict(evidence)
    embedded_sha = canonical.pop("evidence_sha256")
    assert embedded_sha == human_review_evidence_sha256(canonical)
    assert embedded_sha == outcome["evidence_sha256"]
    assert evidence["status"] == "rejected_by_kuotunyu"
    assert evidence["reviewed_by"] == "kuotunyu"
    assert evidence["problem_cells"] == [8, 11, 13, 36, 39]
    assert evidence["problem_count"] == 5
    assert config["generation_gate"]["allowed"] is False
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False


def test_v8_owner_review_rejection_is_canonical_and_generation_stays_closed() -> None:
    config = load_supervised_labeler_config(V8_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    path = PROJECT_ROOT / outcome["evidence_path"]
    evidence = json.loads(path.read_text(encoding="utf-8"))

    canonical = dict(evidence)
    embedded_sha = canonical.pop("evidence_sha256")
    assert embedded_sha == human_review_evidence_sha256(canonical)
    assert embedded_sha == outcome["evidence_sha256"]
    assert evidence["status"] == "rejected_by_kuotunyu"
    assert evidence["reviewed_by"] == "kuotunyu"
    assert evidence["problem_cells"] == [1, 6, 10, 16, 41, 42]
    assert evidence["problem_count"] == 6
    assert outcome["categories"] == {
        "background_or_other_false_positive": [1, 6, 10],
        "missed_helmet": [16, 42],
        "severe_localization_failure": [41],
    }
    assert config["generation_gate"]["allowed"] is False
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False


def test_v9_owner_review_rejection_is_canonical_and_generation_stays_closed() -> None:
    config = load_supervised_labeler_config(V9_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    path = PROJECT_ROOT / outcome["evidence_path"]
    evidence = json.loads(path.read_text(encoding="utf-8"))

    canonical = dict(evidence)
    embedded_sha = canonical.pop("evidence_sha256")
    assert embedded_sha == human_review_evidence_sha256(canonical)
    assert embedded_sha == outcome["evidence_sha256"]
    assert evidence["status"] == "rejected_by_kuotunyu"
    assert evidence["reviewed_by"] == "kuotunyu"
    assert evidence["problem_cells"] == [6, 11, 12, 37]
    assert evidence["problem_count"] == 4
    assert outcome["categories"] == {
        "background_or_other_false_positive": [6, 12],
        "missed_helmet": [11, 37],
    }
    assert config["generation_gate"]["allowed"] is False
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False


def test_v9_revealed_diagnoses_freeze_threshold_and_reflection_causes() -> None:
    config = load_supervised_labeler_config(V9_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    diagnosis_path = PROJECT_ROOT / outcome["diagnosis_path"]
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    reflection_path = PROJECT_ROOT / outcome["reflection_diagnosis_path"]
    reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
    cases = {int(case["cell"]): case for case in diagnosis["problem_cases"]}
    reflection_cases = {
        int(case["cell"]): case for case in reflection["problem_images"]
    }
    threshold_grid = {
        float(row["threshold"]): row for row in diagnosis["threshold_grid"]
    }

    assert hashlib.sha256(diagnosis_path.read_bytes()).hexdigest() == outcome[
        "diagnosis_file_sha256"
    ]
    assert hashlib.sha256(reflection_path.read_bytes()).hexdigest() == outcome[
        "reflection_diagnosis_file_sha256"
    ]
    assert diagnosis["eligible_for_generation_gate"] is False
    assert diagnosis["owner_category_counts"] == {
        "background_or_other_false_positive": 2,
        "missed_helmet": 2,
    }
    assert diagnosis["automatic_miss_reason_counts"] == {
        "below_score_threshold_and_removed_by_geometry_filter": 2,
        "matching_box_below_frozen_score_threshold": 2,
    }
    assert diagnosis["owner_false_positive_scores"] == [
        pytest.approx(0.055908203125),
        pytest.approx(0.055908203125),
    ]
    assert len(cases[6]["accepted_false_positives"]) == 1
    assert len(cases[12]["accepted_false_positives"]) == 1
    assert len(cases[11]["misses"]) == 2
    assert len(cases[37]["misses"]) == 2
    assert threshold_grid[0.056]["true_positives"] == 173
    assert threshold_grid[0.056]["false_positives"] < threshold_grid[0.05][
        "false_positives"
    ]
    assert threshold_grid[0.056]["false_negatives"] == 21
    assert threshold_grid[0.01]["false_positives"] == 1675

    assert reflection["all_problem_images_reflection_detected"] is True
    assert reflection["problem_cells"] == [6, 11, 12, 37]
    assert reflection_cases[6]["reflection"]["detected_axes"] == [
        "top_bottom"
    ]
    assert reflection_cases[11]["reflection"]["detected_axes"] == [
        "left_right"
    ]
    assert reflection_cases[12]["reflection"]["detected_axes"] == [
        "top_bottom"
    ]
    assert reflection_cases[37]["reflection"]["detected_axes"] == [
        "top_bottom"
    ]
    assert reflection_cases[6]["false_positive_locations"][0][
        "center_inside_clean_crop"
    ] is False
    assert reflection_cases[12]["false_positive_locations"][0][
        "center_inside_clean_crop"
    ] is False
    assert sum(
        miss["center_inside_clean_crop"]
        for miss in reflection_cases[11]["missed_truth_locations"]
    ) == 1
    assert all(
        miss["center_inside_clean_crop"]
        for miss in reflection_cases[37]["missed_truth_locations"]
    )
    assert diagnosis["validation_images_read"] == 0
    assert diagnosis["test_images_read"] == 0
    assert diagnosis["whole_image_generation_run"] is False
    assert reflection["validation_images_read"] == 0
    assert reflection["test_images_read"] == 0
    assert reflection["whole_image_generation_run"] is False


def test_v8_revealed_diagnosis_freezes_owner_failure_causes() -> None:
    config = load_supervised_labeler_config(V8_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    path = PROJECT_ROOT / outcome["diagnosis_path"]
    diagnosis = json.loads(path.read_text(encoding="utf-8"))
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = {int(case["cell"]): case for case in diagnosis["problem_cases"]}
    threshold_grid = {
        float(row["threshold"]): row for row in diagnosis["threshold_grid"]
    }

    assert file_sha == outcome["diagnosis_file_sha256"]
    assert diagnosis["eligible_for_generation_gate"] is False
    assert diagnosis["problem_cells"] == [1, 6, 10, 16, 41, 42]
    assert diagnosis["owner_category_counts"] == {
        "background_or_other_false_positive": 3,
        "missed_helmet": 2,
        "severe_localization_failure": 1,
    }
    assert diagnosis["automatic_miss_reason_counts"] == {
        "below_score_threshold_and_removed_by_geometry_filter": 1,
        "matching_box_below_frozen_score_threshold": 3,
        "no_matching_localization": 2,
    }
    assert diagnosis["max_owner_false_positive_score"] == pytest.approx(
        0.08251953125
    )
    assert len(cases[1]["accepted_false_positives"]) == 1
    assert len(cases[6]["accepted_false_positives"]) == 1
    assert len(cases[10]["accepted_false_positives"]) == 2
    assert cases[41]["owner_category"] == "severe_localization_failure"
    assert cases[41]["accepted_false_positives"][0][
        "best_truth_iou"
    ] == pytest.approx(0.3333147110294451)
    assert cases[41]["misses"][0]["best_raw_candidate"][
        "iou"
    ] == pytest.approx(0.9195399122723662)
    assert threshold_grid[0.083]["recall"] == pytest.approx(
        0.08465608465608465
    )
    assert diagnosis["validation_images_read"] == 0
    assert diagnosis["test_images_read"] == 0
    assert diagnosis["whole_image_generation_run"] is False


def test_v7_revealed_diagnoses_freeze_geometry_and_low_score_causes() -> None:
    review_diagnosis = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "supervised_labeler_v7_review_diagnosis.json"
        ).read_text(encoding="utf-8")
    )
    geometry_diagnosis = json.loads(
        (
            PROJECT_ROOT
            / "reports"
            / "supervised_labeler_v7_geometry_diagnosis.json"
        ).read_text(encoding="utf-8")
    )
    recommended = geometry_diagnosis["recommended_candidate"]

    assert review_diagnosis["eligible_for_generation_gate"] is False
    assert review_diagnosis["problem_instances"] == 6
    assert review_diagnosis["reason_counts"] == {
        "matching_box_below_frozen_score_threshold": 1,
        "removed_by_frozen_geometry_filter": 5,
    }
    assert geometry_diagnosis["eligible_for_generation_gate"] is False
    assert geometry_diagnosis["revealed_images_read"] == 432
    assert recommended["max_relative_area"] == 0.14
    assert recommended["max_relative_height"] == 0.40
    assert recommended["geometry_misses_recovered"] == 5
    assert recommended["geometry_misses_total"] == 5
    assert recommended["revealed_audit_metrics"]["precision"] == pytest.approx(
        0.9540816326530612
    )
    assert recommended["revealed_audit_metrics"]["recall"] == pytest.approx(
        0.9303482587064676
    )
    assert review_diagnosis["validation_images_read"] == 0
    assert review_diagnosis["test_images_read"] == 0
    assert geometry_diagnosis["validation_images_read"] == 0
    assert geometry_diagnosis["test_images_read"] == 0
    assert review_diagnosis["whole_image_generation_run"] is False
    assert geometry_diagnosis["whole_image_generation_run"] is False


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


def test_v8_sampling_adds_small_helmet_images_without_stacking_weights() -> None:
    annotations = {
        1: [],
        2: [{"category_id": 1, "bbox": [0, 0, 5, 5]}],
        3: [{"category_id": 1, "bbox": [0, 0, 20, 20]}],
        4: [
            {"category_id": 1, "bbox": [0, 0, 5, 5]},
            {"category_id": 1, "bbox": [4, 0, 5, 5]},
        ],
    }
    image_records = {
        image_id: {"width": 100, "height": 100}
        for image_id in annotations
    }

    weights = supervised_sampling_weights(
        image_ids=[1, 2, 3, 4],
        annotations=annotations,
        image_records=image_records,
        helmet_category_id=1,
        empty_image_weight=2.0,
        close_helmet_pair_weight=2.0,
        close_pair_ratio_max=1.0,
        small_helmet_weight=2.0,
        small_helmet_relative_area_max=0.0075,
    )

    assert weights == [2.0, 2.0, 1.0, 2.0]


def test_exact_audit_evidence_preserves_boxes_without_raster_parsing() -> None:
    evidence = build_audit_evidence(
        rows=[101],
        truth={101: [[1, 2, 3, 4]]},
        predictions={
            101: [
                (0.8, [5, 6, 7, 8]),
                (0.1, [9, 10, 11, 12]),
            ]
        },
        threshold=0.5,
        split_manifest_sha256="a" * 64,
    )

    assert evidence["split_manifest_sha256"] == "a" * 64
    assert evidence["cases"] == [
        {
            "cell": 1,
            "image_id": 101,
            "truth_boxes": [[1.0, 2.0, 3.0, 4.0]],
            "model_predictions": [
                {
                    "score": 0.8,
                    "box": [5.0, 6.0, 7.0, 8.0],
                }
            ],
        }
    ]
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0


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
