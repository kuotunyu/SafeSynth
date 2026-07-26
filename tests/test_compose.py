from __future__ import annotations

import numpy as np

from src.synthetic.compose import (
    _requested_classes,
    _sample_seed,
    _scenario_sequence,
    _transform_scale,
)


def scenario_config() -> dict:
    return {
        "scenarios": {
            "small_distant": {"weight": 0.25},
            "head_no_helmet": {"weight": 0.25},
            "partial_occlusion": {"weight": 0.15},
            "crowded": {"weight": 0.12},
            "hard_negative": {"weight": 0.13},
            "low_light_blur": {"weight": 0.10},
            "context_replacement": {"weight": 1.00},
        }
    }


def test_sample_seed_is_stable_and_index_isolated() -> None:
    assert _sample_seed(42, 7) == _sample_seed(42, 7)
    assert _sample_seed(42, 7) != _sample_seed(42, 8)
    assert _sample_seed(42, 7) != _sample_seed(43, 7)


def test_default_preview_covers_every_unblocked_scenario() -> None:
    sequence = _scenario_sequence(
        32, scenario_config(), None, np.random.default_rng(42)
    )

    assert set(sequence) == {
        "small_distant",
        "head_no_helmet",
        "partial_occlusion",
        "crowded",
        "low_light_blur",
    }
    assert "hard_negative" not in sequence


def test_partial_occlusion_adds_one_person_occluder() -> None:
    classes = _requested_classes(
        "partial_occlusion",
        3,
        np.random.default_rng(3),
        person_crowded_fallback=True,
    )

    assert len(classes) == 4
    assert classes[-1] == "person"


def test_crowded_fallback_uses_only_diverse_headlike_material() -> None:
    classes = _requested_classes(
        "crowded",
        20,
        np.random.default_rng(4),
        person_crowded_fallback=True,
    )

    assert set(classes) <= {"helmet", "head"}


def test_context_replacement_is_explicit_only() -> None:
    default = _scenario_sequence(
        32, scenario_config(), None, np.random.default_rng(42)
    )
    explicit = _scenario_sequence(
        3,
        scenario_config(),
        ["context_replacement"],
        np.random.default_rng(42),
    )

    assert "context_replacement" not in default
    assert explicit == ["context_replacement"] * 3


def test_context_replacement_scale_matches_target_area() -> None:
    rgba = np.zeros((24, 34, 4), dtype=np.uint8)
    rgba[2:22, 2:32, 3] = 255
    config = {"compose": {"max_paste_scale": {"helmet": 1.15}}}

    scale = _transform_scale(
        scenario="context_replacement",
        settings={},
        class_name="helmet",
        rgba=rgba,
        config=config,
        rng=np.random.default_rng(1),
        target_bbox_xywh=[0, 0, 15, 10],
    )

    assert np.isclose(scale, 0.5)
