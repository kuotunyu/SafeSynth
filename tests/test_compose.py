from __future__ import annotations

import numpy as np

from src.synthetic.compose import _requested_classes, _scenario_sequence


def scenario_config() -> dict:
    return {
        "scenarios": {
            "small_distant": {"weight": 0.25},
            "head_no_helmet": {"weight": 0.25},
            "partial_occlusion": {"weight": 0.15},
            "crowded": {"weight": 0.12},
            "hard_negative": {"weight": 0.13},
            "low_light_blur": {"weight": 0.10},
        }
    }


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
