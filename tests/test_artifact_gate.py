from __future__ import annotations

import numpy as np

from src.filtering.artifact_gate import (
    _context_crop,
    _match_real_annotations,
    has_person_context,
    patch_feature,
    roc_auc,
)


def test_context_crop_clips_at_image_boundary() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)

    crop = _context_crop(image, [0, 0, 20, 10], context_scale=2)

    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
    assert crop.shape[0] <= 40
    assert crop.shape[1] <= 50


def test_patch_feature_is_fixed_width() -> None:
    small = np.zeros((12, 20, 3), dtype=np.uint8)
    large = np.zeros((80, 30, 3), dtype=np.uint8)

    assert patch_feature(small, size=64).shape == patch_feature(large, size=64).shape


def test_roc_auc_handles_ties_and_perfect_ranking() -> None:
    labels = np.asarray([0, 0, 1, 1])

    assert roc_auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1
    assert roc_auc(labels, np.ones(4)) == 0.5


def test_real_controls_are_geometry_matched_without_replacement() -> None:
    candidates = [
        {"id": 1, "bbox": [0, 0, 10, 20]},
        {"id": 2, "bbox": [0, 0, 40, 50]},
        {"id": 3, "bbox": [0, 0, 12, 18]},
    ]

    matched = _match_real_annotations(
        [[0, 0, 11, 19], [0, 0, 38, 52]],
        candidates,
    )

    assert [item["id"] for item in matched] == [3, 2]


def test_person_context_uses_upper_body_and_horizontal_expansion() -> None:
    person_boxes = [[35.0, 20.0, 30.0, 100.0]]

    assert has_person_context([45.0, 15.0, 10.0, 10.0], person_boxes)
    assert not has_person_context([100.0, 15.0, 10.0, 10.0], person_boxes)
    assert not has_person_context([45.0, 100.0, 10.0, 10.0], person_boxes)
