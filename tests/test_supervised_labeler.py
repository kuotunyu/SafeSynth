from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

from scripts.audit_supervised_labeler_v12_independence import (
    OUTPUT_PATH as V12_INDEPENDENCE_ERRATUM_PATH,
)
from scripts.diagnose_labeler_postprocessing import select_geometry_candidate
from scripts.diagnose_supervised_labeler_failure import diagnostic_thresholds
from scripts.diagnose_supervised_labeler_v12_review import (
    OUTPUT_PATH as V12_REVIEW_DIAGNOSIS_PATH,
)
from scripts.prepare_supervised_labeler_v12_gt_review import (
    EVIDENCE_PATH as V12_GT_EVIDENCE_PATH,
)
from scripts.prepare_supervised_labeler_v12_gt_review import (
    POOL_PATH as V12_GT_POOL_PATH,
)
from scripts.prepare_supervised_labeler_v13_gt_review import (
    CONFIG_PATH as V13_GT_CONFIG_PATH,
)
from scripts.prepare_supervised_labeler_v13_gt_review import (
    EVIDENCE_PATH as V13_GT_EVIDENCE_PATH,
)
from scripts.prepare_supervised_labeler_v13_gt_review import (
    POOL_PATH as V13_GT_POOL_PATH,
)
from scripts.prepare_supervised_labeler_v14_gt_review import (
    CONFIG_PATH as V14_GT_CONFIG_PATH,
)
from scripts.prepare_supervised_labeler_v14_gt_review import (
    EVIDENCE_PATH as V14_GT_EVIDENCE_PATH,
)
from scripts.prepare_supervised_labeler_v14_gt_review import (
    POOL_PATH as V14_GT_POOL_PATH,
)
from scripts.prepare_supervised_labeler_v15_gt_review import (
    CONFIG_PATH as V15_GT_CONFIG_PATH,
)
from scripts.prepare_supervised_labeler_v15_gt_review import (
    EVIDENCE_PATH as V15_GT_EVIDENCE_PATH,
)
from scripts.prepare_supervised_labeler_v15_gt_review import (
    POOL_PATH as V15_GT_POOL_PATH,
)
from scripts.record_supervised_labeler_v6_review import (
    build_review_evidence,
    parse_problem_cells,
)
from scripts.record_supervised_labeler_v12_gt_review import (
    AUDIT_PATH as V12_ADJUDICATED_AUDIT_PATH,
)
from scripts.record_supervised_labeler_v12_gt_review import (
    OWNER_REVIEW_PATH as V12_GT_OWNER_REVIEW_PATH,
)
from scripts.record_supervised_labeler_v12_model_review import (
    OUTPUT_PATH as V12_MODEL_HUMAN_REVIEW_PATH,
)
from scripts.record_supervised_labeler_v13_gt_review import (
    AUDIT_PATH as V13_ADJUDICATED_AUDIT_PATH,
)
from scripts.record_supervised_labeler_v13_gt_review import (
    OWNER_REVIEW_PATH as V13_GT_OWNER_REVIEW_PATH,
)
from scripts.record_supervised_labeler_v13_model_review import (
    OUTPUT_PATH as V13_MODEL_HUMAN_REVIEW_PATH,
)
from scripts.record_supervised_labeler_v14_gt_review import (
    AUDIT_PATH as V14_ADJUDICATED_AUDIT_PATH,
)
from scripts.record_supervised_labeler_v14_gt_review import (
    OWNER_REVIEW_PATH as V14_GT_OWNER_REVIEW_PATH,
)
from scripts.record_supervised_labeler_v14_model_review import (
    OUTPUT_PATH as V14_MODEL_HUMAN_REVIEW_PATH,
)
from scripts.record_supervised_labeler_v15_gt_review import (
    AUDIT_PATH as V15_ADJUDICATED_AUDIT_PATH,
)
from scripts.record_supervised_labeler_v15_gt_review import (
    OWNER_REVIEW_PATH as V15_GT_OWNER_REVIEW_PATH,
)
from scripts.render_supervised_labeler_review import split_review_sheet
from scripts.render_supervised_labeler_review_separated import (
    _draw_model_boxes,
    extract_model_boxes,
)
from scripts.run_supervised_labeler_v12_model_audit import (
    EVIDENCE_PATH as V12_MODEL_AUDIT_EVIDENCE_PATH,
)
from scripts.run_supervised_labeler_v12_model_audit import (
    REPORT_PATH as V12_MODEL_AUDIT_REPORT_PATH,
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
V13_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v13.yaml"
V13_SPLIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v13_split.json"
)
V14_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v14.yaml"
V14_SPLIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v14_split.json"
)
V14_PREFLIGHT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_preflight.json"
)
V14_SMOKE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_smoke.json"
)
V14_TRAINING_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_training.json"
)
V14_AUDIT_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_audit_evidence.json"
)
V14_MODEL_REVIEW_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v14_model_review_manifest.json"
)
V14_REVIEW_DIAGNOSIS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v14_review_diagnosis.json"
)
V15_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v15.yaml"
V15_SPLIT_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v15_split.json"
)
V15_PREFLIGHT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_preflight.json"
)
V15_SMOKE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_smoke.json"
)
V13_PREFLIGHT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_preflight.json"
)
V13_SMOKE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_smoke.json"
)
V13_TRAINING_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_training.json"
)
V13_AUDIT_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_audit_evidence.json"
)
V13_MODEL_REVIEW_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v13_model_review_manifest.json"
)
V13_REVIEW_DIAGNOSIS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v13_review_diagnosis.json"
)
V12_GT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "supervised_labeler_v12_gt_review.yaml"
)
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


def test_v12_gt_owner_review_quarantines_only_ambiguous_edge_cases() -> None:
    if not V12_GT_OWNER_REVIEW_PATH.is_file():
        pytest.skip("v12 GT-only owner review has not been recorded yet")
    review = json.loads(
        V12_GT_OWNER_REVIEW_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(review)
    embedded_sha = canonical.pop("review_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert review["status"] == "v12_gt_only_primary_adjudicated"
    assert review["reviewed_by"] == "kuotunyu"
    assert review["reviewed_images"] == 64
    assert review["pass_images"] == 62
    assert review["quarantined_images"] == 2
    assert review["categories"] == {
        "dataset_gt_false_positive_cells": [],
        "dataset_gt_localization_cells": [],
        "dataset_gt_miss_cells": [],
        "uncertain_edge_clipped_cells": [53, 64],
    }
    assert {
        int(row["cell"])
        for row in review["decisions"]
        if row["decision"] == "UNCERTAIN"
    } == {53, 64}
    assert review["model_boxes_present"] is False
    assert review["model_inference_run"] is False
    assert review["sealed_reserve_pixels_read"] == 0
    assert review["validation_images_read"] == 0
    assert review["test_images_read"] == 0
    assert review["whole_image_generation_run"] is False


def test_v12_adjudicated_audit_is_frozen_before_model_inference() -> None:
    if not V12_ADJUDICATED_AUDIT_PATH.is_file():
        pytest.skip("v12 adjudicated audit has not been frozen yet")
    audit = json.loads(
        V12_ADJUDICATED_AUDIT_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(audit)
    embedded_sha = canonical.pop("manifest_sha256")
    selected = audit["selected_cases"]

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert audit["status"] == (
        "v12_adjudicated_audit_frozen_before_model_inference"
    )
    assert audit["selected_images"] == 48
    assert len(selected) == 48
    assert audit["selected_stratum_counts"] == {
        "dataset_gt_empty": 8,
        "positive_area_q1": 10,
        "positive_area_q2": 10,
        "positive_area_q3": 10,
        "positive_area_q4": 10,
    }
    assert len({int(row["image_id"]) for row in selected}) == 48
    assert len({int(row["group_id"]) for row in selected}) == 48
    assert {int(row["primary_cell"]) for row in selected}.isdisjoint(
        {53, 64}
    )
    assert {
        int(row["primary_cell"])
        for row in audit["quarantined_primary_cases"]
    } == {53, 64}
    assert audit["valid_primary_surplus_images"] == 14
    assert audit["sealed_reserve_images"] == 32
    assert audit["sealed_reserve_pixels_read"] == 0
    assert len(audit["source_group_ids_reserved_from_training"]) == 96
    assert audit["model_boxes_present"] is False
    assert audit["model_inference_run"] is False
    assert audit["validation_images_read"] == 0
    assert audit["test_images_read"] == 0
    assert audit["whole_image_generation_run"] is False


def test_v12_model_audit_registration_is_frozen_before_inference() -> None:
    config = yaml.safe_load(V12_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    registration = config["model_audit_registration"]
    report_path = PROJECT_ROOT / registration["source_training_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_path = (
        Path(registration["checkpoint_path"]) / "model.safetensors"
    )

    assert registration["status"] == "frozen_before_v12_model_inference"
    assert registration["source_experiment"] == "supervised_labeler_v11"
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        registration["source_training_report_sha256"]
    )
    assert report["checkpoint_sha256"] == registration["checkpoint_sha256"]
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == (
        registration["checkpoint_sha256"]
    )
    assert registration["score_threshold"] == 0.03
    assert registration["match_iou"] == 0.50
    assert registration["numeric_gates"] == {
        "min_median_matched_iou": 0.60,
        "min_precision": 0.80,
        "min_recall": 0.70,
    }
    assert registration["sealed_reserve_pixels_read"] == 0
    assert registration["validation_images_read"] == 0
    assert registration["test_images_read"] == 0
    assert registration["whole_image_generation_run"] is False


def test_v12_model_audit_uses_only_the_frozen_adjudicated_cases() -> None:
    if not V12_MODEL_AUDIT_REPORT_PATH.is_file():
        pytest.skip("v12 model audit has not run yet")
    report = json.loads(
        V12_MODEL_AUDIT_REPORT_PATH.read_text(encoding="utf-8")
    )
    evidence = json.loads(
        V12_MODEL_AUDIT_EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    audit = json.loads(
        V12_ADJUDICATED_AUDIT_PATH.read_text(encoding="utf-8")
    )
    config = yaml.safe_load(V12_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    outcome = config["model_audit_outcome"]
    canonical_report = dict(report)
    report_sha = canonical_report.pop("report_sha256")
    canonical_evidence = dict(evidence)
    evidence_sha = canonical_evidence.pop("evidence_sha256")

    assert canonical_mapping_sha256(canonical_report) == report_sha
    assert canonical_mapping_sha256(canonical_evidence) == evidence_sha
    assert report["status"] == "v12_numeric_audit_passed_owner_review_pending"
    assert all(report["checks"].values())
    assert outcome["report_sha256"] == report["report_sha256"]
    assert outcome["evidence_sha256"] == evidence["evidence_sha256"]
    assert hashlib.sha256(
        V12_MODEL_AUDIT_REPORT_PATH.read_bytes()
    ).hexdigest() == outcome["report_file_sha256"]
    assert hashlib.sha256(
        V12_MODEL_AUDIT_EVIDENCE_PATH.read_bytes()
    ).hexdigest() == outcome["evidence_file_sha256"]
    assert report["audit_manifest_sha256"] == audit["manifest_sha256"]
    assert evidence["audit_manifest_sha256"] == audit["manifest_sha256"]
    assert [row["image_id"] for row in evidence["cases"]] == [
        row["image_id"] for row in audit["selected_cases"]
    ]
    assert [row["truth_boxes"] for row in evidence["cases"]] == [
        row["truth_boxes"] for row in audit["selected_cases"]
    ]
    assert report["model_images_read"] == 48
    assert evidence["model_images_read"] == 48
    assert len(evidence["cases"]) == 48
    assert report["sealed_reserve_pixels_read"] == 0
    assert evidence["sealed_reserve_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["whole_image_generation_run"] is False
    assert len(report["pages"]) == 3
    for page in report["pages"]:
        path = PROJECT_ROOT / page["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page["sha256"]


def test_v12_model_human_review_rejects_the_numeric_pass() -> None:
    if not V12_MODEL_HUMAN_REVIEW_PATH.is_file():
        pytest.skip("v12 model human review has not been recorded yet")
    review = json.loads(
        V12_MODEL_HUMAN_REVIEW_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(review)
    embedded_sha = canonical.pop("review_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert review["status"] == "rejected_by_kuotunyu"
    assert review["decision"] == "reject"
    assert review["reviewed_by"] == "kuotunyu"
    assert review["problem_count"] == 10
    assert review["problem_cells"] == [3, 10, 13, 20, 25, 26, 27, 29, 43, 45]
    assert review["categories"] == {
        "model_false_positive_cells": [3, 10, 13, 20, 25, 27, 29, 45],
        "model_missed_helmeted_head_cells": [26, 43],
    }
    assert review["generation_allowed"] is False
    assert review["validation_images_read"] == 0
    assert review["test_images_read"] == 0
    assert review["whole_image_generation_run"] is False


def test_v12_review_diagnosis_reads_only_revealed_problem_images() -> None:
    if not V12_REVIEW_DIAGNOSIS_PATH.is_file():
        pytest.skip("v12 owner-review diagnosis has not run yet")
    diagnosis = json.loads(
        V12_REVIEW_DIAGNOSIS_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(diagnosis)
    embedded_sha = canonical.pop("diagnosis_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert diagnosis["status"] == "v12_owner_review_diagnosed"
    assert [
        row["cell"] for row in diagnosis["false_positive_diagnoses"]
    ] == [3, 10, 13, 20, 25, 27, 29, 45]
    assert [
        row["cell"] for row in diagnosis["missed_helmet_diagnoses"]
    ] == [26, 43]
    assert diagnosis["revealed_problem_images_read_for_gpu_diagnosis"] == 2
    assert diagnosis["new_audit_images_read"] == 0
    assert diagnosis["sealed_reserve_pixels_read"] == 0
    assert diagnosis["validation_images_read"] == 0
    assert diagnosis["test_images_read"] == 0
    assert diagnosis["whole_image_generation_run"] is False


def test_v12_independence_erratum_invalidates_only_the_numeric_claim() -> None:
    if not V12_INDEPENDENCE_ERRATUM_PATH.is_file():
        pytest.skip("v12 independence audit has not run yet")
    erratum = json.loads(
        V12_INDEPENDENCE_ERRATUM_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(erratum)
    embedded_sha = canonical.pop("erratum_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert erratum["status"] == (
        "v12_model_audit_invalidated_by_training_group_overlap"
    )
    assert erratum["v11_partition"]["covers_every_train_group"] is True
    assert erratum["overlap_counts"] == {
        "primary_with_v11_training": 64,
        "reserve_with_v11_training": 32,
        "selected_audit_with_v11_training": 48,
    }
    assert erratum["numeric_audit_independent"] is False
    assert erratum["numeric_audit_claim_valid"] is False
    assert erratum["owner_visual_rejection_valid"] is True
    assert erratum["owner_failure_diagnosis_valid"] is True
    assert erratum["original_evidence_mutated"] is False
    assert erratum["validation_images_read"] == 0
    assert erratum["test_images_read"] == 0
    assert erratum["whole_image_generation_run"] is False


def test_v13_gt_pool_is_frozen_before_pixels_or_training() -> None:
    if not V13_GT_POOL_PATH.is_file():
        pytest.skip("v13 GT pool has not been frozen yet")
    pool = json.loads(V13_GT_POOL_PATH.read_text(encoding="utf-8"))
    canonical = dict(pool)
    embedded_sha = canonical.pop("manifest_sha256")
    v12_pool = json.loads(V12_GT_POOL_PATH.read_text(encoding="utf-8"))
    v12_groups = {
        int(row["group_id"])
        for row in [
            *v12_pool["primary_cases"],
            *v12_pool["sealed_reserve_cases"],
        ]
    }
    selected = [*pool["primary_cases"], *pool["sealed_reserve_cases"]]
    selected_groups = {int(row["group_id"]) for row in selected}

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert pool["status"] == (
        "v13_gt_only_pool_frozen_before_pixel_review_or_training"
    )
    assert pool["primary_images"] == 64
    assert pool["sealed_reserve_images"] == 32
    assert len(selected_groups) == 96
    assert selected_groups.isdisjoint(v12_groups)
    assert set(pool["future_v13_training_exclusion_group_ids"]) == (
        selected_groups
    )
    assert pool["primary_pixels_read"] == 0
    assert pool["sealed_reserve_pixels_read"] == 0
    assert pool["v13_training_started"] is False
    assert pool["model_inference_run"] is False
    assert pool["validation_images_read"] == 0
    assert pool["test_images_read"] == 0
    assert pool["whole_image_generation_run"] is False


def test_v13_gt_review_contains_no_model_output_or_training() -> None:
    if not V13_GT_EVIDENCE_PATH.is_file():
        pytest.skip("v13 GT review has not been rendered yet")
    evidence = json.loads(
        V13_GT_EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(evidence)
    embedded_sha = canonical.pop("evidence_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert evidence["status"] == (
        "v13_gt_only_primary_review_rendered_before_training"
    )
    assert evidence["review_stage"] == "gt_only"
    assert len(evidence["cases"]) == 64
    assert evidence["model_boxes_present"] is False
    assert evidence["model_inference_run"] is False
    assert evidence["v13_training_started"] is False
    assert evidence["sealed_reserve_pixels_read"] == 0
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False
    for page in evidence["pages"]:
        path = PROJECT_ROOT / page["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page["sha256"]


def test_v13_config_pins_rendered_gt_review_evidence() -> None:
    if not V13_GT_EVIDENCE_PATH.is_file():
        pytest.skip("v13 GT review has not been rendered yet")
    config = yaml.safe_load(V13_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    outcome = config["render_outcome"]
    evidence = json.loads(
        V13_GT_EVIDENCE_PATH.read_text(encoding="utf-8")
    )

    assert config["status"] == (
        "gt_only_approved_audit_frozen_training_pending"
    )
    assert outcome["status"] == evidence["status"]
    assert outcome["evidence_sha256"] == evidence["evidence_sha256"]
    assert hashlib.sha256(V13_GT_EVIDENCE_PATH.read_bytes()).hexdigest() == (
        outcome["evidence_file_sha256"]
    )
    assert outcome["v13_training_started"] is False
    assert outcome["model_inference_run"] is False
    assert outcome["sealed_reserve_pixels_read"] == 0
    assert outcome["validation_images_read"] == 0
    assert outcome["test_images_read"] == 0
    for rendered_page, pinned_page in zip(
        evidence["pages"], outcome["pages"], strict=True
    ):
        assert rendered_page == pinned_page


def test_v13_gt_owner_review_approves_all_primary_cases() -> None:
    if not V13_GT_OWNER_REVIEW_PATH.is_file():
        pytest.skip("v13 GT-only owner review has not been recorded yet")
    review = json.loads(
        V13_GT_OWNER_REVIEW_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(review)
    embedded_sha = canonical.pop("review_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert review["status"] == (
        "v13_gt_only_primary_adjudicated_zero_problems"
    )
    assert review["reviewed_by"] == "kuotunyu"
    assert review["reviewed_images"] == 64
    assert review["pass_images"] == 64
    assert review["problem_images"] == 0
    assert review["quarantined_images"] == 0
    assert all(
        row["decision"] == "PASS" for row in review["decisions"]
    )
    assert review["categories"] == {
        "dataset_gt_false_positive_cells": [],
        "dataset_gt_localization_cells": [],
        "dataset_gt_miss_cells": [],
        "uncertain_cells": [],
    }
    assert review["model_boxes_present"] is False
    assert review["model_inference_run"] is False
    assert review["v13_training_started"] is False
    assert review["sealed_reserve_pixels_read"] == 0
    assert review["validation_images_read"] == 0
    assert review["test_images_read"] == 0
    assert review["whole_image_generation_run"] is False
    config = yaml.safe_load(V13_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    outcome = config["gt_owner_review_outcome"]
    assert outcome["status"] == review["status"]
    assert outcome["review_sha256"] == review["review_sha256"]
    assert hashlib.sha256(V13_GT_OWNER_REVIEW_PATH.read_bytes()).hexdigest() == (
        outcome["review_file_sha256"]
    )


def test_v13_adjudicated_audit_is_frozen_before_training() -> None:
    if not V13_ADJUDICATED_AUDIT_PATH.is_file():
        pytest.skip("v13 adjudicated audit has not been frozen yet")
    audit = json.loads(
        V13_ADJUDICATED_AUDIT_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(audit)
    embedded_sha = canonical.pop("manifest_sha256")
    selected = audit["selected_cases"]

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert audit["status"] == "v13_adjudicated_audit_frozen_before_training"
    assert audit["selected_images"] == 48
    assert len(selected) == 48
    assert audit["selected_stratum_counts"] == {
        "dataset_gt_empty": 8,
        "positive_area_q1": 10,
        "positive_area_q2": 10,
        "positive_area_q3": 10,
        "positive_area_q4": 10,
    }
    assert len({int(row["image_id"]) for row in selected}) == 48
    assert len({int(row["group_id"]) for row in selected}) == 48
    assert audit["quarantined_primary_images"] == 0
    assert audit["valid_primary_surplus_images"] == 16
    assert audit["sealed_reserve_images"] == 32
    assert audit["sealed_reserve_pixels_read"] == 0
    assert len(audit["source_group_ids_reserved_from_training"]) == 96
    assert audit["v13_training_started"] is False
    assert audit["model_boxes_present"] is False
    assert audit["model_inference_run"] is False
    assert audit["validation_images_read"] == 0
    assert audit["test_images_read"] == 0
    assert audit["whole_image_generation_run"] is False
    config = yaml.safe_load(V13_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    outcome = config["adjudicated_audit_outcome"]
    assert outcome["status"] == audit["status"]
    assert outcome["manifest_sha256"] == audit["manifest_sha256"]
    assert (
        hashlib.sha256(V13_ADJUDICATED_AUDIT_PATH.read_bytes()).hexdigest()
        == outcome["manifest_file_sha256"]
    )


def test_v13_model_registration_keeps_preregistered_intervention() -> None:
    config = load_supervised_labeler_config(V13_CONFIG_PATH)

    assert config["experiment_id"] == "supervised_labeler_v13"
    assert config["optimization"]["initialization"] == (
        "pinned_base_checkpoint_only"
    )
    assert config["sampling"]["empty_image_weight"] == 4.0
    assert config["sampling"]["large_helmet_weight"] == 3.0
    assert config["sampling"]["large_helmet_relative_area_min"] == 0.15
    assert config["postprocessing"]["max_relative_area"] == 0.30
    assert config["postprocessing"]["max_relative_height"] == 0.75
    assert config["generation_gate"]["allowed"] is False
    assert config["independence_registration"]["v13_reserved_groups"] == 96
    assert config["independence_registration"][
        "v12_pool_excluded_from_model_data"
    ] is True
    assert config["independence_registration"]["v13_training_started"] is False
    assert config["independence_registration"][
        "audit_model_inference_run"
    ] is False
    assert config["data"]["validation_images_read"] == 0
    assert config["data"]["test_images_read"] == 0


def test_v13_model_split_excludes_both_frozen_pools() -> None:
    if not V13_SPLIT_PATH.is_file():
        pytest.skip("v13 model split has not been frozen yet")
    config = load_supervised_labeler_config(V13_CONFIG_PATH)
    split = json.loads(V13_SPLIT_PATH.read_text(encoding="utf-8"))
    canonical = dict(split)
    embedded_sha = canonical.pop("manifest_sha256")
    training_groups = set(split["training_group_ids"])
    calibration_groups = set(split["calibration_group_ids"])
    audit_groups = set(split["untouched_audit_group_ids"])
    v13_groups = set(split["v13_reserved_group_ids"])
    v12_groups = set(split["v12_development_excluded_group_ids"])

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert config["split_manifest_sha256"] == embedded_sha
    assert split["status"] == "frozen_before_supervised_training"
    assert split["initialization"] == "pinned_base_checkpoint_only"
    assert split["untouched_audit_images"] == 48
    assert len(audit_groups) == 48
    assert split["v13_reserved_groups"] == 96
    assert split["v12_development_excluded_groups"] == 96
    assert len(v13_groups) == 96
    assert len(v12_groups) == 96
    assert v13_groups.isdisjoint(v12_groups)
    assert training_groups.isdisjoint(calibration_groups | v13_groups)
    assert calibration_groups.isdisjoint(v13_groups)
    assert (training_groups | calibration_groups).isdisjoint(v12_groups)
    assert audit_groups <= v13_groups
    assert split["v13_training_started"] is False
    assert split["sealed_reserve_pixels_read"] == 0
    assert split["validation_images_read"] == 0
    assert split["test_images_read"] == 0
    assert split["whole_image_generation_run"] is False


def test_v13_cpu_preflight_keeps_independent_audit_unread() -> None:
    if not V13_PREFLIGHT_PATH.is_file():
        pytest.skip("v13 CPU preflight has not run yet")
    config = load_supervised_labeler_config(V13_CONFIG_PATH)
    report = json.loads(V13_PREFLIGHT_PATH.read_text(encoding="utf-8"))
    outcome = config["cpu_preflight_outcome"]

    assert report["status"] == "cpu_preflight_passed_gpu_smoke_waiting"
    assert outcome["status"] == report["status"]
    assert hashlib.sha256(V13_PREFLIGHT_PATH.read_bytes()).hexdigest() == (
        outcome["report_file_sha256"]
    )
    assert report["training"]["images_read"] == 2644
    assert report["training"]["invalid_boxes"] == 0
    assert report["calibration"]["images_read"] == 621
    assert report["calibration"]["invalid_boxes"] == 0
    assert report["sampling_weight_counts"] == {
        "1.0": 701,
        "2.0": 1654,
        "3.0": 42,
        "4.0": 247,
    }
    assert report["training_calibration_group_overlap"] == 0
    assert report["v13_reserved_groups_in_model_data"] == 0
    assert report["v12_development_groups_in_model_data"] == 0
    assert report["untouched_audit_pixels_read"] == 0
    assert report["sealed_reserve_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["gpu_work_run"] is False
    assert report["whole_image_generation_run"] is False


def test_v13_gpu_smoke_uses_base_and_keeps_audit_unread() -> None:
    if not V13_SMOKE_PATH.is_file():
        pytest.skip("v13 GPU smoke has not run yet")
    config = load_supervised_labeler_config(V13_CONFIG_PATH)
    report = json.loads(V13_SMOKE_PATH.read_text(encoding="utf-8"))
    outcome = config["gpu_smoke_outcome"]

    assert report["status"] == "smoke_passed"
    assert outcome["status"] == report["status"]
    assert hashlib.sha256(V13_SMOKE_PATH.read_bytes()).hexdigest() == (
        outcome["report_file_sha256"]
    )
    assert report["batch_size"] == 8
    assert report["helmet_boxes"] > 0
    assert report["loss"] > 0
    assert report["peak_vram_gib"] < 24
    assert outcome["initialization"] == "pinned_base_checkpoint_only"
    assert outcome["v13_training_started"] is False
    assert report["untouched_audit_images_read"] == 0
    assert outcome["sealed_reserve_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert outcome["whole_image_generation_run"] is False


def test_v13_independent_numeric_audit_passes_all_frozen_gates() -> None:
    if not V13_TRAINING_REPORT_PATH.is_file():
        pytest.skip("v13 formal training has not completed yet")
    config = load_supervised_labeler_config(V13_CONFIG_PATH)
    report = json.loads(
        V13_TRAINING_REPORT_PATH.read_text(encoding="utf-8")
    )
    evidence = json.loads(
        V13_AUDIT_EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    outcome = config["numeric_audit_outcome"]
    checkpoint = Path(report["checkpoint_path"]) / "model.safetensors"

    assert config["status"] == "human_review_rejected"
    assert report["status"] == "supervised_labeler_audit_passed"
    assert outcome["status"] == report["status"]
    assert report["split_manifest_sha256"] == (
        config["split_manifest_sha256"]
    )
    assert hashlib.sha256(
        V13_TRAINING_REPORT_PATH.read_bytes()
    ).hexdigest() == outcome["report_file_sha256"]
    assert report["best_calibration"]["epoch"] == 2
    assert report["best_calibration"]["threshold"] == 0.05
    assert report["checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert report["audit_metrics"] == {
        "f1": 0.9300000000000002,
        "false_negatives": 6,
        "false_positives": 8,
        "median_matched_iou": 0.8397744015796275,
        "precision": 0.9207920792079208,
        "recall": 0.9393939393939394,
        "true_positives": 93,
    }
    assert report["untouched_audit_images_read"] == 48
    assert len(evidence["cases"]) == 48
    assert hashlib.sha256(
        V13_AUDIT_EVIDENCE_PATH.read_bytes()
    ).hexdigest() == report["audit_evidence_sha256"]
    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        report["checkpoint_sha256"]
    )
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["whole_image_generation_run"] is False
    assert outcome["sealed_reserve_pixels_read"] == 0


def test_v13_model_review_pages_pin_exact_audit_evidence() -> None:
    if not V13_MODEL_REVIEW_MANIFEST_PATH.is_file():
        pytest.skip("v13 model review pages have not been rendered yet")
    config = load_supervised_labeler_config(V13_CONFIG_PATH)
    manifest = json.loads(
        V13_MODEL_REVIEW_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(manifest)
    embedded_sha = canonical.pop("manifest_sha256")
    registration = config["model_review_registration"]

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert manifest["status"] == (
        "supervised_labeler_v13_model_review_pages_frozen"
    )
    assert registration["manifest_sha256"] == embedded_sha
    assert hashlib.sha256(
        V13_MODEL_REVIEW_MANIFEST_PATH.read_bytes()
    ).hexdigest() == registration["manifest_file_sha256"]
    assert manifest["source_audit_evidence_sha256"] == hashlib.sha256(
        V13_AUDIT_EVIDENCE_PATH.read_bytes()
    ).hexdigest()
    assert manifest["reviewed_images"] == 48
    assert manifest["score_threshold"] == 0.05
    assert manifest["numeric_checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert manifest["render_model_inference_run"] is False
    assert manifest["source_model_inference_images"] == 48
    assert manifest["validation_images_read"] == 0
    assert manifest["test_images_read"] == 0
    assert manifest["whole_image_generation_run"] is False
    assert registration["owner_review_status"] == "pending_kuotunyu"
    assert registration["approve_only_if_problem_count"] == 0
    assert len(manifest["pages"]) == 3
    for page in manifest["pages"]:
        path = PROJECT_ROOT / page["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page["sha256"]


def test_v13_owner_model_review_records_exact_three_misses() -> None:
    if not V13_MODEL_HUMAN_REVIEW_PATH.is_file():
        pytest.skip("v13 owner model review has not been recorded yet")
    review = json.loads(
        V13_MODEL_HUMAN_REVIEW_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(review)
    embedded_sha = canonical.pop("review_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert review["status"] == "rejected_by_kuotunyu"
    assert review["experiment_id"] == "supervised_labeler_v13"
    assert review["reviewed_by"] == "kuotunyu"
    assert review["decision"] == "reject"
    assert review["problem_count"] == 3
    assert review["problem_cells"] == [3, 22, 34]
    assert review["categories"] == {
        "model_false_positive_cells": [],
        "model_missed_helmeted_head_cells": [3, 22, 34],
    }
    assert {
        (row["cell"], row["image_id"]) for row in review["problem_cases"]
    } == {(3, 4507), (22, 3593), (34, 681)}
    assert review["numeric_audit_status"] == (
        "supervised_labeler_audit_passed"
    )
    assert review["generation_allowed"] is False
    assert review["validation_images_read"] == 0
    assert review["test_images_read"] == 0
    assert review["whole_image_generation_run"] is False
    config = load_supervised_labeler_config(V13_CONFIG_PATH)
    outcome = config["human_review_outcome"]
    assert outcome["status"] == review["status"]
    assert outcome["problem_cells"] == review["problem_cells"]
    assert outcome["evidence_sha256"] == review["review_sha256"]
    assert hashlib.sha256(
        V13_MODEL_HUMAN_REVIEW_PATH.read_bytes()
    ).hexdigest() == outcome["evidence_file_sha256"]
    assert config["generation_gate"]["allowed"] is False


def test_v13_review_diagnosis_freezes_geometry_and_score_blockers() -> None:
    config = load_supervised_labeler_config(V13_CONFIG_PATH)
    outcome = config["review_diagnosis_outcome"]
    diagnosis = json.loads(
        V13_REVIEW_DIAGNOSIS_PATH.read_text(encoding="utf-8")
    )
    cases = {int(row["cell"]): row for row in diagnosis["cases"]}

    assert hashlib.sha256(
        V13_REVIEW_DIAGNOSIS_PATH.read_bytes()
    ).hexdigest() == outcome["report_file_sha256"]
    assert diagnosis["status"] == "v13_owner_miss_diagnosis_complete"
    assert diagnosis["scope"] == {
        "already_revealed_audit_cells_only": [3, 22, 34],
        "images_read": 3,
        "test_images_read": 0,
        "validation_images_read": 0,
        "whole_image_generation_run": False,
    }
    assert diagnosis["cause_counts"] == {
        "candidate_below_score_and_rejected_by_geometry": 1,
        "candidate_below_score_threshold": 1,
        "candidate_rejected_by_geometry_filter": 1,
    }
    assert cases[3]["misses"][0]["blocking_stages"] == ["geometry_filter"]
    assert cases[22]["misses"][0]["blocking_stages"] == [
        "score_threshold",
        "geometry_filter",
    ]
    assert cases[34]["misses"][0]["blocking_stages"] == ["score_threshold"]
    assert cases[3]["misses"][0]["best_raw_candidate"]["iou"] == pytest.approx(
        0.9461339922604669
    )
    assert cases[22]["misses"][0]["best_raw_candidate"]["iou"] == pytest.approx(
        0.9430958166370554
    )
    assert cases[34]["misses"][0]["best_raw_candidate"]["score"] == pytest.approx(
        0.048095703125
    )
    assert config["generation_gate"]["allowed"] is False


def test_v14_intervention_is_preregistered_before_new_pool_pixels() -> None:
    config = yaml.safe_load(V14_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    intervention = config["future_v14_intervention"]

    assert config["status"] in {
        "preregistered_before_pool_freeze",
        "pool_frozen_before_pixel_review_or_training",
        "gt_only_primary_review_pending_owner",
        "gt_only_adjudicated_audit_frozen_training_pending",
    }
    assert config["protocol"]["label_semantics"] == (
        "class_direct_helmeted_head_region"
    )
    assert config["review_stage"]["model_boxes_allowed"] is False
    assert intervention["status"] == (
        "preregistered_before_v14_pool_pixels_or_training"
    )
    assert intervention["revealed_development_evidence"]["problem_cells"] == [
        3,
        22,
        34,
    ]
    assert intervention["model_facing_changes"] == {
        "calibration_score_grid_minimum": 0.005,
        "include_owner_approved_v13_primary_gt_as_revealed_training_development": True,
        "large_helmet_relative_area_min": 0.15,
        "large_helmet_weight": 6.0,
        "max_relative_area": 0.70,
        "max_relative_height": 0.90,
        "near_image_edge_helmet_weight": 4.0,
        "near_image_edge_margin_fraction": 0.05,
        "owner_miss_replay_weight": 8.0,
        "replay_v13_owner_miss_image_ids": [4507, 3593, 681],
    }
    assert config["independence_boundary"]["validation_images_read"] == 0
    assert config["independence_boundary"]["test_images_read"] == 0
    assert config["generation_gate"]["allowed"] is False


def test_v14_gt_pool_is_frozen_before_pixels_or_training() -> None:
    config = yaml.safe_load(V14_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    pool = json.loads(V14_GT_POOL_PATH.read_text(encoding="utf-8"))
    canonical = dict(pool)
    embedded_sha = canonical.pop("manifest_sha256")
    selected = [*pool["primary_cases"], *pool["sealed_reserve_cases"]]
    selected_groups = {int(row["group_id"]) for row in selected}

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert pool["status"] == (
        "v14_gt_only_pool_frozen_before_pixel_review_or_training"
    )
    assert pool["excluded_group_count"] == 816
    assert len(pool["primary_cases"]) == 64
    assert len(pool["sealed_reserve_cases"]) == 32
    assert len(selected_groups) == 96
    assert pool["future_v14_training_exclusion_group_ids"] == sorted(
        selected_groups
    )
    assert pool["primary_pixels_read"] == 0
    assert pool["sealed_reserve_pixels_read"] == 0
    assert pool["v14_training_started"] is False
    assert pool["model_inference_run"] is False
    assert pool["validation_images_read"] == 0
    assert pool["test_images_read"] == 0
    assert pool["whole_image_generation_run"] is False
    outcome = config["freeze_outcome"]
    assert hashlib.sha256(V14_GT_POOL_PATH.read_bytes()).hexdigest() == outcome[
        "pool_file_sha256"
    ]
    assert outcome["pool_manifest_sha256"] == embedded_sha


def test_v14_gt_pages_contain_only_green_gt_and_keep_reserve_sealed() -> None:
    config = yaml.safe_load(V14_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(V14_GT_EVIDENCE_PATH.read_text(encoding="utf-8"))
    canonical = dict(evidence)
    embedded_sha = canonical.pop("evidence_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert evidence["status"] == (
        "v14_gt_only_primary_review_rendered_before_training"
    )
    assert evidence["review_stage"] == "gt_only"
    assert evidence["model_boxes_present"] is False
    assert evidence["model_inference_run"] is False
    assert evidence["v14_training_started"] is False
    assert evidence["primary_images_read"] == 64
    assert len(evidence["cases"]) == 64
    assert evidence["sealed_reserve_images"] == 32
    assert evidence["sealed_reserve_pixels_read"] == 0
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False
    assert len(evidence["pages"]) == 4
    for page in evidence["pages"]:
        path = PROJECT_ROOT / page["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page["sha256"]
    outcome = config["render_outcome"]
    assert hashlib.sha256(
        V14_GT_EVIDENCE_PATH.read_bytes()
    ).hexdigest() == outcome["evidence_file_sha256"]
    assert outcome["evidence_sha256"] == embedded_sha


def test_v14_gt_review_quarantines_only_cell_60_and_freezes_clean_audit() -> None:
    config = yaml.safe_load(V14_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    review = json.loads(V14_GT_OWNER_REVIEW_PATH.read_text(encoding="utf-8"))
    audit = json.loads(
        V14_ADJUDICATED_AUDIT_PATH.read_text(encoding="utf-8")
    )
    canonical_review = dict(review)
    review_sha = canonical_review.pop("review_sha256")
    canonical_audit = dict(audit)
    audit_sha = canonical_audit.pop("manifest_sha256")

    assert canonical_mapping_sha256(canonical_review) == review_sha
    assert canonical_mapping_sha256(canonical_audit) == audit_sha
    assert review["status"] == (
        "v14_gt_only_primary_adjudicated_one_quarantine"
    )
    assert review["reviewed_by"] == "kuotunyu"
    assert review["reviewed_images"] == 64
    assert review["pass_images"] == 63
    assert review["problem_images"] == 1
    assert review["quarantined_images"] == 1
    assert review["categories"] == {
        "dataset_gt_false_positive_cells": [60],
        "dataset_gt_localization_cells": [],
        "dataset_gt_miss_cells": [],
        "uncertain_cells": [],
    }
    problem = next(
        row for row in review["decisions"] if int(row["cell"]) == 60
    )
    assert problem["image_id"] == 1171
    assert problem["decision"] == "DATASET_GT_FALSE_POSITIVE"
    assert audit["status"] == (
        "v14_adjudicated_audit_frozen_before_training"
    )
    assert audit["selected_images"] == 48
    assert audit["selected_stratum_counts"] == {
        "dataset_gt_empty": 8,
        "positive_area_q1": 10,
        "positive_area_q2": 10,
        "positive_area_q3": 10,
        "positive_area_q4": 10,
    }
    assert audit["valid_primary_surplus_images"] == 15
    assert audit["quarantined_primary_images"] == 1
    assert audit["quarantined_primary_cases"][0]["primary_cell"] == 60
    assert audit["sealed_reserve_pixels_read"] == 0
    assert audit["v14_training_started"] is False
    assert audit["model_inference_run"] is False
    assert audit["validation_images_read"] == 0
    assert audit["test_images_read"] == 0
    assert audit["whole_image_generation_run"] is False
    owner_outcome = config["gt_owner_review_outcome"]
    audit_outcome = config["adjudicated_audit_outcome"]
    assert hashlib.sha256(
        V14_GT_OWNER_REVIEW_PATH.read_bytes()
    ).hexdigest() == owner_outcome["review_file_sha256"]
    assert hashlib.sha256(
        V14_ADJUDICATED_AUDIT_PATH.read_bytes()
    ).hexdigest() == audit_outcome["manifest_file_sha256"]
    assert config["generation_gate"]["allowed"] is False


def test_v14_model_split_replays_v13_primary_but_keeps_new_audit_independent() -> None:
    config = load_supervised_labeler_config(V14_CONFIG_PATH)
    split = json.loads(V14_SPLIT_PATH.read_text(encoding="utf-8"))
    canonical = dict(split)
    embedded_sha = canonical.pop("manifest_sha256")
    training_groups = set(split["training_group_ids"])
    calibration_groups = set(split["calibration_group_ids"])
    v14_groups = set(split["v14_reserved_group_ids"])
    v13_primary_groups = set(split["v13_approved_primary_group_ids"])
    v13_reserve_groups = set(
        split["v13_sealed_reserve_excluded_group_ids"]
    )
    v12_groups = set(split["v12_development_excluded_group_ids"])

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert embedded_sha == config["split_manifest_sha256"]
    assert split["status"] == "frozen_before_supervised_training"
    assert split["initialization"] == "pinned_base_checkpoint_only"
    assert split["training_images"] == 2609
    assert split["training_groups"] == 2520
    assert split["calibration_images"] == 621
    assert split["untouched_audit_images"] == 48
    assert len(v14_groups) == 96
    assert len(v13_primary_groups) == 64
    assert len(v13_reserve_groups) == 32
    assert len(v12_groups) == 96
    assert v13_primary_groups <= training_groups
    assert not training_groups & calibration_groups
    assert not (training_groups | calibration_groups) & v14_groups
    assert not (training_groups | calibration_groups) & v13_reserve_groups
    assert not (training_groups | calibration_groups) & v12_groups
    assert set(split["owner_miss_replay_image_ids"]) == {681, 3593, 4507}
    assert set(split["owner_miss_replay_image_ids"]) <= set(
        split["training_image_ids"]
    )
    assert split["v14_training_started"] is False
    assert split["sealed_reserve_pixels_read"] == 0
    assert split["validation_images_read"] == 0
    assert split["test_images_read"] == 0
    assert split["whole_image_generation_run"] is False
    outcome = config["split_outcome"]
    assert hashlib.sha256(V14_SPLIT_PATH.read_bytes()).hexdigest() == outcome[
        "split_file_sha256"
    ]
    assert config["generation_gate"]["allowed"] is False


def test_v14_cpu_preflight_verifies_replay_and_keeps_audit_pixels_sealed() -> None:
    config = load_supervised_labeler_config(V14_CONFIG_PATH)
    outcome = config["cpu_preflight_outcome"]
    report = json.loads(V14_PREFLIGHT_PATH.read_text(encoding="utf-8"))

    assert hashlib.sha256(
        V14_PREFLIGHT_PATH.read_bytes()
    ).hexdigest() == outcome["report_file_sha256"]
    assert report["status"] == "cpu_preflight_passed_gpu_smoke_waiting"
    assert report["training"]["images_read"] == 2609
    assert report["training"]["normalized_images"] == 2594
    assert report["training"]["invalid_boxes"] == 0
    assert report["calibration"]["images_read"] == 621
    assert report["calibration"]["normalized_images"] == 617
    assert report["calibration"]["invalid_boxes"] == 0
    assert report["sampling_weight_counts"] == {
        "1.0": 360,
        "2.0": 619,
        "4.0": 1586,
        "6.0": 41,
        "8.0": 3,
    }
    assert report["owner_miss_replay_weights"] == {
        "681": 8.0,
        "3593": 8.0,
        "4507": 8.0,
    }
    assert report["v14_reserved_groups_in_model_data"] == 0
    assert report["v13_sealed_reserve_groups_in_model_data"] == 0
    assert report["v12_development_groups_in_model_data"] == 0
    assert report["v13_approved_primary_groups_in_training"] == 64
    assert report["untouched_audit_images"] == 48
    assert report["untouched_audit_pixels_read"] == 0
    assert report["sealed_reserve_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["gpu_work_run"] is False
    assert report["whole_image_generation_run"] is False
    assert config["generation_gate"]["allowed"] is False


def test_v14_gpu_smoke_uses_base_only_and_keeps_audit_sealed() -> None:
    config = load_supervised_labeler_config(V14_CONFIG_PATH)
    outcome = config["gpu_smoke_outcome"]
    report = json.loads(V14_SMOKE_PATH.read_text(encoding="utf-8"))

    assert hashlib.sha256(V14_SMOKE_PATH.read_bytes()).hexdigest() == outcome[
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
    assert config["optimization"]["initialization"] == (
        "pinned_base_checkpoint_only"
    )
    assert config["generation_gate"]["allowed"] is False


def test_v14_independent_numeric_audit_passes_all_frozen_gates() -> None:
    config = load_supervised_labeler_config(V14_CONFIG_PATH)
    report = json.loads(
        V14_TRAINING_REPORT_PATH.read_text(encoding="utf-8")
    )
    evidence = json.loads(
        V14_AUDIT_EVIDENCE_PATH.read_text(encoding="utf-8")
    )
    outcome = config["numeric_audit_outcome"]
    checkpoint = Path(report["checkpoint_path"]) / "model.safetensors"

    assert config["status"] == "human_review_rejected"
    assert report["status"] == "supervised_labeler_audit_passed"
    assert outcome["status"] == report["status"]
    assert report["split_manifest_sha256"] == (
        config["split_manifest_sha256"]
    )
    assert hashlib.sha256(
        V14_TRAINING_REPORT_PATH.read_bytes()
    ).hexdigest() == outcome["report_file_sha256"]
    assert report["best_calibration"]["epoch"] == 2
    assert report["best_calibration"]["threshold"] == 0.05
    assert report["checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert report["audit_metrics"] == {
        "f1": 0.9152542372881356,
        "false_negatives": 7,
        "false_positives": 8,
        "median_matched_iou": 0.843415371431786,
        "precision": 0.9101123595505618,
        "recall": 0.9204545454545454,
        "true_positives": 81,
    }
    assert report["untouched_audit_images_read"] == 48
    assert len(evidence["cases"]) == 48
    assert hashlib.sha256(
        V14_AUDIT_EVIDENCE_PATH.read_bytes()
    ).hexdigest() == report["audit_evidence_sha256"]
    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        report["checkpoint_sha256"]
    )
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["whole_image_generation_run"] is False
    assert outcome["sealed_reserve_pixels_read"] == 0
    assert config["generation_gate"]["allowed"] is False


def test_v14_model_review_pages_pin_exact_audit_evidence() -> None:
    config = load_supervised_labeler_config(V14_CONFIG_PATH)
    manifest = json.loads(
        V14_MODEL_REVIEW_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(manifest)
    embedded_sha = canonical.pop("manifest_sha256")
    registration = config["model_review_registration"]

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert manifest["status"] == (
        "supervised_labeler_v14_model_review_pages_frozen"
    )
    assert registration["manifest_sha256"] == embedded_sha
    assert hashlib.sha256(
        V14_MODEL_REVIEW_MANIFEST_PATH.read_bytes()
    ).hexdigest() == registration["manifest_file_sha256"]
    assert manifest["source_audit_evidence_sha256"] == hashlib.sha256(
        V14_AUDIT_EVIDENCE_PATH.read_bytes()
    ).hexdigest()
    assert manifest["reviewed_images"] == 48
    assert manifest["score_threshold"] == 0.05
    assert manifest["numeric_checks"] == {
        "audit_median_matched_iou": True,
        "audit_precision": True,
        "audit_recall": True,
    }
    assert manifest["render_model_inference_run"] is False
    assert manifest["source_model_inference_images"] == 48
    assert manifest["validation_images_read"] == 0
    assert manifest["test_images_read"] == 0
    assert manifest["whole_image_generation_run"] is False
    assert registration["owner_review_status"] == "rejected_by_kuotunyu"
    assert registration["approve_only_if_problem_count"] == 0
    assert config["generation_gate"]["allowed"] is False


def test_v14_owner_rejection_and_failure_diagnosis_are_exact() -> None:
    config = load_supervised_labeler_config(V14_CONFIG_PATH)
    review = json.loads(
        V14_MODEL_HUMAN_REVIEW_PATH.read_text(encoding="utf-8")
    )
    diagnosis = json.loads(
        V14_REVIEW_DIAGNOSIS_PATH.read_text(encoding="utf-8")
    )
    review_outcome = config["human_review_outcome"]
    diagnosis_outcome = config["review_diagnosis_outcome"]
    canonical_review = dict(review)
    embedded_review_sha = canonical_review.pop("review_sha256")

    assert canonical_mapping_sha256(canonical_review) == embedded_review_sha
    assert review["status"] == "rejected_by_kuotunyu"
    assert review["reviewed_by"] == "kuotunyu"
    assert review["problem_cells"] == [7, 10, 40, 43]
    assert review["categories"] == {
        "model_false_positive_cells": [10],
        "model_missed_helmeted_head_cells": [7, 40, 43],
    }
    assert [row["image_id"] for row in review["problem_cases"]] == [
        361,
        210,
        2534,
        3605,
    ]
    assert hashlib.sha256(
        V14_MODEL_HUMAN_REVIEW_PATH.read_bytes()
    ).hexdigest() == review_outcome["review_file_sha256"]
    assert review_outcome["review_sha256"] == embedded_review_sha

    assert diagnosis["status"] == "v14_owner_failure_diagnosis_complete"
    assert diagnosis["scope"]["already_revealed_audit_cells_only"] == [
        7,
        10,
        40,
        43,
    ]
    assert diagnosis["scope"]["images_read"] == 4
    assert diagnosis["cause_counts"] == {
        "candidate_below_score_threshold": 3,
        "unmatched_prediction_above_threshold": 2,
    }
    assert hashlib.sha256(
        V14_REVIEW_DIAGNOSIS_PATH.read_bytes()
    ).hexdigest() == diagnosis_outcome["report_file_sha256"]
    cases = {int(row["cell"]): row for row in diagnosis["cases"]}
    assert cases[7]["misses"][0]["best_raw_candidate"]["score"] == (
        pytest.approx(0.03466796875)
    )
    assert cases[40]["misses"][0]["best_raw_candidate"]["score"] == (
        pytest.approx(0.008056640625)
    )
    assert cases[43]["misses"][0]["best_raw_candidate"]["score"] == (
        pytest.approx(0.0140380859375)
    )
    assert cases[10]["false_positives"][0]["score"] == pytest.approx(
        0.0517578125
    )
    assert diagnosis["scope"]["validation_images_read"] == 0
    assert diagnosis["scope"]["test_images_read"] == 0
    assert diagnosis["scope"]["whole_image_generation_run"] is False
    assert config["generation_gate"]["allowed"] is False


def test_figure_cleanup_keeps_current_evidence_and_removes_legacy_duplicates() -> None:
    figure_root = PROJECT_ROOT / "reports" / "figures"
    for name in (
        "supervised_labeler_v14_audit.png",
        "supervised_labeler_v14_model_review_page_01.png",
        "supervised_labeler_v14_model_review_page_02.png",
        "supervised_labeler_v14_model_review_page_03.png",
        "supervised_labeler_v13_audit.png",
        "supervised_labeler_v6_audit.png",
    ):
        assert (figure_root / name).is_file()
    assert not list(figure_root.glob("supervised_labeler_v7_audit*.png"))
    assert not list(figure_root.glob("supervised_labeler_v8_audit*.png"))
    assert not list(figure_root.glob("supervised_labeler_v9_audit*.png"))
    assert not list(figure_root.glob("supervised_labeler_v10_audit*.png"))
    assert not list(figure_root.glob("supervised_labeler_v11_audit*.png"))
    assert not list(figure_root.glob("*_failed.png"))


def test_v15_intervention_and_fresh_gt_pool_are_frozen_before_training() -> None:
    config = yaml.safe_load(V15_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    pool = json.loads(V15_GT_POOL_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(V15_GT_EVIDENCE_PATH.read_text(encoding="utf-8"))
    intervention = config["future_v15_intervention"]
    canonical_pool = dict(pool)
    pool_sha = canonical_pool.pop("manifest_sha256")
    canonical_evidence = dict(evidence)
    evidence_sha = canonical_evidence.pop("evidence_sha256")

    assert config["status"] == (
        "gt_only_primary_adjudicated_three_images_quarantined"
    )
    assert intervention["status"] == (
        "preregistered_before_v15_pool_pixels_or_training"
    )
    assert intervention["initialization"] == "pinned_base_checkpoint_only"
    assert intervention["model_facing_changes"] == {
        "include_owner_approved_v14_audit_gt_as_revealed_training_development": True,
        "positive_error_replay_image_ids": [361, 2534, 3605],
        "positive_error_replay_weight": 12.0,
        "hard_negative_error_replay_image_ids": [210, 361],
        "hard_negative_error_replay_weight": 12.0,
        "overlap_policy": "maximum_weight",
        "epochs": 6,
    }
    assert config["gpu_schedule"][
        "recommended_contiguous_gpu_window_minutes"
    ] == 40
    assert canonical_mapping_sha256(canonical_pool) == pool_sha
    assert pool["status"] == (
        "v15_gt_only_pool_frozen_before_pixel_review_or_training"
    )
    assert pool["excluded_group_count"] == 912
    assert pool["primary_images"] == 64
    assert pool["sealed_reserve_images"] == 32
    assert len(pool["future_v15_training_exclusion_group_ids"]) == 96
    assert len(
        {
            int(row["group_id"])
            for row in [
                *pool["primary_cases"],
                *pool["sealed_reserve_cases"],
            ]
        }
    ) == 96
    assert pool["primary_pixels_read"] == 0
    assert pool["sealed_reserve_pixels_read"] == 0
    assert pool["v15_training_started"] is False
    assert pool["model_inference_run"] is False
    assert pool["validation_images_read"] == 0
    assert pool["test_images_read"] == 0

    assert canonical_mapping_sha256(canonical_evidence) == evidence_sha
    assert evidence["status"] == (
        "v15_gt_only_primary_review_rendered_before_training"
    )
    assert evidence["pool_manifest_sha256"] == pool_sha
    assert evidence["primary_images_read"] == 64
    assert evidence["primary_images_normalized"] == 64
    assert evidence["sealed_reserve_pixels_read"] == 0
    assert evidence["model_boxes_present"] is False
    assert evidence["model_inference_run"] is False
    assert evidence["v15_training_started"] is False
    assert evidence["validation_images_read"] == 0
    assert evidence["test_images_read"] == 0
    assert evidence["whole_image_generation_run"] is False
    assert hashlib.sha256(V15_GT_POOL_PATH.read_bytes()).hexdigest() == (
        config["freeze_outcome"]["pool_file_sha256"]
    )
    assert hashlib.sha256(V15_GT_EVIDENCE_PATH.read_bytes()).hexdigest() == (
        config["render_outcome"]["evidence_file_sha256"]
    )
    for page in evidence["pages"]:
        path = PROJECT_ROOT / page["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == page["sha256"]
    assert config["generation_gate"]["allowed"] is False


def test_v15_gt_adjudication_quarantines_exact_owner_reported_images() -> None:
    config = yaml.safe_load(V15_GT_CONFIG_PATH.read_text(encoding="utf-8"))
    review = json.loads(
        V15_GT_OWNER_REVIEW_PATH.read_text(encoding="utf-8")
    )
    audit = json.loads(
        V15_ADJUDICATED_AUDIT_PATH.read_text(encoding="utf-8")
    )
    canonical_review = dict(review)
    review_sha = canonical_review.pop("review_sha256")
    canonical_audit = dict(audit)
    audit_sha = canonical_audit.pop("manifest_sha256")

    assert canonical_mapping_sha256(canonical_review) == review_sha
    assert canonical_mapping_sha256(canonical_audit) == audit_sha
    assert review["status"] == (
        "v15_gt_only_primary_adjudicated_three_quarantines"
    )
    assert review["categories"] == {
        "dataset_gt_false_positive_cells": [6, 61],
        "dataset_gt_localization_cells": [],
        "dataset_gt_miss_cells": [48],
        "uncertain_cells": [],
    }
    assert review["accepted_ambiguities"] == {
        "cell_06_overlapping_middle_hard_hats": (
            "One or two boxes are both acceptable; this is not a defect."
        )
    }
    problems = {
        int(row["cell"]): int(row["image_id"])
        for row in review["decisions"]
        if row["decision"] != "PASS"
    }
    assert problems == {6: 3272, 48: 837, 61: 2686}
    assert review["pass_images"] == 61
    assert review["sealed_reserve_pixels_read"] == 0
    assert review["v15_training_started"] is False

    assert audit["status"] == (
        "v15_adjudicated_audit_frozen_before_training"
    )
    assert audit["selected_images"] == 48
    assert audit["valid_primary_surplus_images"] == 13
    assert audit["quarantined_primary_images"] == 3
    assert audit["sealed_reserve_images"] == 32
    assert audit["sealed_reserve_pixels_read"] == 0
    assert audit["selected_stratum_counts"] == {
        "dataset_gt_empty": 8,
        "positive_area_q1": 10,
        "positive_area_q2": 10,
        "positive_area_q3": 10,
        "positive_area_q4": 10,
    }
    assert {
        int(row["image_id"]) for row in audit["quarantined_primary_cases"]
    } == {837, 2686, 3272}
    assert not {
        int(row["image_id"]) for row in audit["quarantined_primary_cases"]
    } & {
        int(row["image_id"]) for row in audit["selected_cases"]
    }
    outcome = config["owner_adjudication_outcome"]
    assert hashlib.sha256(
        V15_GT_OWNER_REVIEW_PATH.read_bytes()
    ).hexdigest() == outcome["owner_review_file_sha256"]
    assert hashlib.sha256(
        V15_ADJUDICATED_AUDIT_PATH.read_bytes()
    ).hexdigest() == outcome["adjudicated_audit_file_sha256"]


def test_v15_split_replays_revealed_errors_and_keeps_audit_independent() -> None:
    config = load_supervised_labeler_config(V15_CONFIG_PATH)
    split = json.loads(V15_SPLIT_PATH.read_text(encoding="utf-8"))
    canonical = dict(split)
    embedded_sha = canonical.pop("manifest_sha256")

    assert canonical_mapping_sha256(canonical) == embedded_sha
    assert split["status"] == "frozen_before_supervised_training"
    assert split["initialization"] == "pinned_base_checkpoint_only"
    assert split["training_images"] == 2580
    assert split["calibration_images"] == 621
    assert split["untouched_audit_images"] == 48
    assert len(split["v15_reserved_group_ids"]) == 96
    assert len(split["v14_approved_primary_group_ids"]) == 63
    assert len(split["v13_approved_primary_group_ids"]) == 64
    assert set(split["positive_error_replay_image_ids"]) == {
        361,
        2534,
        3605,
    }
    assert set(split["hard_negative_error_replay_image_ids"]) == {210, 361}
    assert {
        837,
        1171,
        2686,
        3060,
        3272,
        4155,
        4364,
    }.isdisjoint(split["training_image_ids"])
    assert set(split["training_group_ids"]).isdisjoint(
        split["calibration_group_ids"]
    )
    assert (
        set(split["training_group_ids"])
        | set(split["calibration_group_ids"])
    ).isdisjoint(split["v15_reserved_group_ids"])
    outcome = config["split_outcome"]
    assert embedded_sha == outcome["manifest_sha256"]
    assert hashlib.sha256(V15_SPLIT_PATH.read_bytes()).hexdigest() == outcome[
        "file_sha256"
    ]


def test_v15_cpu_preflight_verifies_replay_and_keeps_audit_pixels_sealed() -> None:
    config = load_supervised_labeler_config(V15_CONFIG_PATH)
    outcome = config["cpu_preflight_outcome"]
    report = json.loads(V15_PREFLIGHT_PATH.read_text(encoding="utf-8"))

    assert hashlib.sha256(V15_PREFLIGHT_PATH.read_bytes()).hexdigest() == (
        outcome["report_sha256"]
    )
    assert report["status"] == "cpu_preflight_passed_gpu_smoke_waiting"
    assert report["training"]["images_read"] == 2580
    assert report["calibration"]["images_read"] == 621
    assert report["training"]["invalid_boxes"] == 0
    assert report["calibration"]["invalid_boxes"] == 0
    assert report["error_replay_weights"] == {
        "210": 12.0,
        "361": 12.0,
        "2534": 12.0,
        "3605": 12.0,
    }
    assert report["overlap_policy"] == "maximum_weight"
    assert report["v15_reserved_groups_in_model_data"] == 0
    assert report["v14_sealed_reserve_groups_in_model_data"] == 0
    assert report["v13_sealed_reserve_groups_in_model_data"] == 0
    assert report["v12_development_groups_in_model_data"] == 0
    assert report["v14_approved_primary_groups_in_training"] == 63
    assert report["v13_approved_primary_groups_in_training"] == 64
    assert report["untouched_audit_pixels_read"] == 0
    assert report["sealed_reserve_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert report["gpu_work_run"] is False


def test_v15_gpu_smoke_uses_base_only_and_keeps_audit_sealed() -> None:
    config = load_supervised_labeler_config(V15_CONFIG_PATH)
    outcome = config["gpu_smoke_outcome"]
    report = json.loads(V15_SMOKE_PATH.read_text(encoding="utf-8"))

    assert hashlib.sha256(V15_SMOKE_PATH.read_bytes()).hexdigest() == outcome[
        "report_sha256"
    ]
    assert report["status"] == "smoke_passed"
    assert report["batch_size"] == 8
    assert report["helmet_boxes"] == 32
    assert report["loss"] == pytest.approx(287.3628234863281)
    assert report["peak_vram_gib"] == pytest.approx(9.315727710723877)
    assert outcome["initialization"] == "pinned_base_checkpoint_only"
    assert outcome["v15_training_started"] is False
    assert report["untouched_audit_images_read"] == 0
    assert outcome["sealed_reserve_pixels_read"] == 0
    assert report["validation_images_read"] == 0
    assert report["test_images_read"] == 0
    assert outcome["whole_image_generation_run"] is False


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


def test_v13_sampling_adds_large_helmet_images_without_stacking_weights() -> None:
    annotations = {
        1: [],
        2: [{"category_id": 1, "bbox": [0, 0, 5, 5]}],
        3: [{"category_id": 1, "bbox": [0, 0, 40, 40]}],
        4: [
            {"category_id": 1, "bbox": [0, 0, 40, 40]},
            {"category_id": 1, "bbox": [35, 0, 40, 40]},
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
        empty_image_weight=4.0,
        close_helmet_pair_weight=2.0,
        close_pair_ratio_max=1.0,
        small_helmet_weight=2.0,
        small_helmet_relative_area_max=0.0075,
        large_helmet_weight=3.0,
        large_helmet_relative_area_min=0.15,
    )

    assert weights == [4.0, 2.0, 3.0, 3.0]


def test_v14_sampling_adds_edge_and_owner_miss_replay_without_stacking() -> None:
    annotations = {
        1: [],
        2: [{"category_id": 1, "bbox": [40, 40, 5, 5]}],
        3: [{"category_id": 1, "bbox": [20, 20, 40, 40]}],
        4: [{"category_id": 1, "bbox": [0, 40, 10, 10]}],
        5: [{"category_id": 1, "bbox": [40, 40, 10, 10]}],
        6: [{"category_id": 1, "bbox": [0, 0, 40, 40]}],
    }
    image_records = {
        image_id: {"width": 100, "height": 100}
        for image_id in annotations
    }

    weights = supervised_sampling_weights(
        image_ids=[1, 2, 3, 4, 5, 6],
        annotations=annotations,
        image_records=image_records,
        helmet_category_id=1,
        empty_image_weight=4.0,
        close_helmet_pair_weight=2.0,
        close_pair_ratio_max=1.0,
        small_helmet_weight=2.0,
        small_helmet_relative_area_max=0.0075,
        large_helmet_weight=6.0,
        large_helmet_relative_area_min=0.15,
        near_image_edge_helmet_weight=4.0,
        near_image_edge_margin_fraction=0.05,
        owner_miss_replay_image_ids=[5],
        owner_miss_replay_weight=8.0,
    )

    assert weights == [4.0, 2.0, 6.0, 4.0, 8.0, 6.0]


def test_v15_positive_and_hard_negative_replay_use_maximum_weight() -> None:
    annotations = {
        1: [],
        2: [{"category_id": 1, "bbox": [40, 40, 10, 10]}],
        3: [{"category_id": 1, "bbox": [40, 40, 10, 10]}],
        4: [{"category_id": 1, "bbox": [40, 40, 10, 10]}],
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
        empty_image_weight=4.0,
        close_helmet_pair_weight=2.0,
        close_pair_ratio_max=1.0,
        positive_error_replay_image_ids=[2, 3],
        positive_error_replay_weight=12.0,
        hard_negative_error_replay_image_ids=[3, 4],
        hard_negative_error_replay_weight=12.0,
    )

    assert weights == [4.0, 12.0, 12.0, 12.0]


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
