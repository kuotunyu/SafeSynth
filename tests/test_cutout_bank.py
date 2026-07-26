from __future__ import annotations

import numpy as np

from src.synthetic.cutout_bank import (
    appearance_statistics,
    cheap_gate_record,
    soft_alpha,
)


def sample_config() -> dict:
    return {
        "cutout_bank": {
            "respect_voc_flags": True,
            "hard_floor": {"min_side_px": 16, "min_area_px": 400},
            "preferred_tier": {"min_side_px": 23, "min_area_px": 667},
            "aspect_ratio": {
                "helmet": [0.4, 4.0],
                "head": [0.4, 4.0],
                "person": [0.18, 5.0],
            },
            "min_distance_to_image_edge_px": 2,
            "max_occlusion_by_others": 0.15,
        },
        "compose": {
            "blending": {
                "erode_before_feather_px": 1,
                "feather_sigma_base": 0.6,
                "feather_sigma_per_px": 0.02,
                "feather_sigma_clip": [0.6, 2.0],
            }
        },
    }


def test_cheap_gates_ignore_head_inside_person_for_occlusion() -> None:
    head = {
        "id": 1,
        "image_id": 10,
        "category_id": 2,
        "bbox": [100, 100, 30, 30],
        "truncated": 0,
        "difficult": 0,
    }
    person = {
        "id": 2,
        "image_id": 10,
        "category_id": 3,
        "bbox": [90, 90, 80, 200],
        "truncated": 0,
        "difficult": 0,
    }
    record = cheap_gate_record(
        annotation=head,
        image_record={"id": 10, "file_name": "x.png", "width": 416, "height": 416},
        image_annotations=[head, person],
        category_names={1: "helmet", 2: "head", 3: "person"},
        frozen_record={"sha256": "abc", "group_id": 4, "split": "train"},
        config=sample_config(),
    )

    assert record["cheap_gate_pass"]
    assert record["measurements"]["occlusion_by_others"] == 0


def test_cheap_gates_reject_same_kind_occlusion_and_bad_voc_flag() -> None:
    first = {
        "id": 1,
        "image_id": 10,
        "category_id": 1,
        "bbox": [100, 100, 30, 30],
        "truncated": 1,
        "difficult": 0,
    }
    second = {
        "id": 2,
        "image_id": 10,
        "category_id": 1,
        "bbox": [105, 105, 30, 30],
        "truncated": 0,
        "difficult": 0,
    }
    record = cheap_gate_record(
        annotation=first,
        image_record={"id": 10, "file_name": "x.png", "width": 416, "height": 416},
        image_annotations=[first, second],
        category_names={1: "helmet", 2: "head", 3: "person"},
        frozen_record={"sha256": "abc", "group_id": 4, "split": "train"},
        config=sample_config(),
    )

    assert record["cheap_gate_failures"][0] == "G1_VOC_FLAG"
    assert "G6_OCCLUDED" in record["cheap_gate_failures"]


def test_soft_alpha_is_non_degenerate() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True

    alpha = soft_alpha(mask, config=sample_config())

    assert alpha.dtype == np.uint8
    assert alpha.max() == 255
    assert alpha.min() == 0
    assert np.any((alpha > 0) & (alpha < 255))


def test_appearance_statistics_are_finite() -> None:
    rgb = np.full((20, 20, 3), [255, 200, 0], dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 2:18] = True

    stats = appearance_statistics(rgb, mask)

    assert all(np.isfinite(stats["lab_mean"]))
    assert stats["hf_noise_sigma"] < 1e-4
    assert 0 <= stats["dominant_hue_deg"] < 360
