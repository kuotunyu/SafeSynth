from __future__ import annotations

import numpy as np
import yaml

from scripts.prepare_paired_person_preflight import (
    load_train_only_clip_embeddings,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.paired_person import (
    intersection_over_smaller_box,
    paired_headlike_annotations,
    strict_unit_reject_reasons,
    transform_linked_box,
)


def test_paired_headlike_requires_upper_person_support() -> None:
    annotations = [
        {"id": 1, "category_id": 1, "bbox": [20, 12, 20, 18]},
        {"id": 2, "category_id": 2, "bbox": [21, 78, 18, 16]},
        {"id": 3, "category_id": 1, "bbox": [80, 10, 18, 18]},
        {"id": 4, "category_id": 3, "bbox": [10, 10, 40, 100]},
    ]

    paired = paired_headlike_annotations(
        [10, 10, 40, 100],
        annotations=annotations,
        categories={1: "helmet", 2: "head", 3: "person"},
    )

    assert [annotation["id"] for annotation in paired] == [1]


def test_linked_box_follows_scale_translation_and_flip() -> None:
    ordinary = transform_linked_box(
        [20, 12, 10, 8],
        source_person_box_xywh=[10, 10, 40, 100],
        scale=2,
        hflip=False,
        patch_width=80,
        patch_left=100,
        patch_top=50,
    )
    flipped = transform_linked_box(
        [20, 12, 10, 8],
        source_person_box_xywh=[10, 10, 40, 100],
        scale=2,
        hflip=True,
        patch_width=80,
        patch_left=100,
        patch_top=50,
    )

    assert ordinary == [120, 54, 20, 16]
    assert flipped == [140, 54, 20, 16]


def test_strict_unit_gate_rejects_small_horizontal_or_truncated_people() -> None:
    person = {
        "class_name": "person",
        "src_split": "train",
        "src_bbox_xywh": [10, 10, 100, 50],
        "preferred_tier": True,
        "voc_flags": {"truncated": 1, "difficult": 0},
        "sam2": {"edge_touch_top": 0, "edge_touch_side": 0},
    }
    reasons = strict_unit_reject_reasons(
        person,
        {"bbox": [20, 12, 20, 15]},
        preferred_tier_required=True,
        min_person_height_px=80,
        min_person_aspect_height_over_width=1.3,
        max_head_center_y_fraction=0.35,
        max_head_width_fraction=0.6,
        max_edge_touch_top=0.1,
        max_edge_touch_side=0.1,
        min_source_person_bottom_fraction=0.65,
        source_image_height=100,
    )

    assert reasons == (
        "VOC_FLAGGED",
        "PERSON_TOO_SMALL",
        "SOURCE_PERSON_TOO_HIGH",
        "PERSON_TOO_HORIZONTAL",
    )


def test_intersection_is_normalized_by_smaller_box() -> None:
    assert intersection_over_smaller_box([0, 0, 10, 10], [5, 0, 5, 10]) == 1
    assert intersection_over_smaller_box([0, 0, 10, 10], [20, 0, 5, 10]) == 0


def test_clip_loader_materializes_only_frozen_train_rows(tmp_path) -> None:
    feature_path = tmp_path / "features.npy"
    np.save(
        feature_path,
        np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.float32),
        allow_pickle=False,
    )

    features, row_by_image = load_train_only_clip_embeddings(
        feature_path,
        coco_images=[{"id": 1}, {"id": 2}, {"id": 3}],
        train_image_ids={1, 3},
    )

    assert row_by_image == {1: 0, 3: 1}
    np.testing.assert_array_equal(features, [[10, 11], [30, 31]])


def test_paired_person_v7_capacity_failure_is_frozen() -> None:
    config = yaml.safe_load(
        (
            PROJECT_ROOT / "configs" / "paired_person_preflight.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["status"] == "paired_person_v7_capacity_rejected"
    assert config["architecture"] == "paired_person_scene_position_insert_v7"
    assert config["root_seed"] == 20260803
    assert config["model_gate"]["model_inference_allowed"] is False
    assert config["outcome"]["built_images"] == 63
    assert config["outcome"]["required_images"] == 64
    assert config["outcome"]["attempted_train_backgrounds"] == 3500
    assert config["outcome"]["model_inference_run"] is False
    assert config["outcome"]["h4_auc_computed"] is False
