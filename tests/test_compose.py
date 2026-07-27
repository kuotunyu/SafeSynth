from __future__ import annotations

import numpy as np

from src.synthetic.compose import (
    Paste,
    _generative_seed,
    _requested_classes,
    _sample_seed,
    _scenario_sequence,
    _transform_scale,
    _visible_paste_masks,
    context_replacement_input_guard,
    reflected_padding_guard,
)
from src.synthetic.composition import Layer


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


def test_generative_seed_is_stable_and_instance_isolated() -> None:
    assert _generative_seed(42, 7, "paste:0") == _generative_seed(
        42, 7, "paste:0"
    )
    assert _generative_seed(42, 7, "paste:0") != _generative_seed(
        42, 7, "paste:1"
    )


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


def test_context_replacement_guard_rejects_background_edge_headlike() -> None:
    annotations = [
        {"id": 1, "category_id": 1, "bbox": [20, 20, 20, 20]},
        {"id": 2, "category_id": 1, "bbox": [40, 0, 20, 12]},
    ]

    result = context_replacement_input_guard(
        image_shape=(100, 100),
        annotations=annotations,
        categories={1: "helmet"},
        pass1={1: {"qc_pass": True}, 2: {"qc_pass": True}},
        guard_config={
            "background_headlike_min_edge_margin_px": 4,
            "anchor_min_edge_margin_px": 8,
            "anchor_min_edge_margin_long_side_fraction": 0.10,
        },
    )

    assert not result.accepted
    assert result.reject_reason == "BACKGROUND_HEADLIKE_NEAR_FRAME_EDGE"


def test_context_replacement_guard_keeps_only_safe_qc_anchors() -> None:
    annotations = [
        {"id": 1, "category_id": 1, "bbox": [8, 8, 20, 20]},
        {"id": 2, "category_id": 2, "bbox": [40, 40, 40, 20]},
        {"id": 3, "category_id": 3, "bbox": [0, 0, 100, 100]},
    ]

    result = context_replacement_input_guard(
        image_shape=(100, 100),
        annotations=annotations,
        categories={1: "helmet", 2: "head", 3: "person"},
        pass1={1: {"qc_pass": True}, 2: {"qc_pass": False}},
        guard_config={
            "background_headlike_min_edge_margin_px": 4,
            "anchor_min_edge_margin_px": 8,
            "anchor_min_edge_margin_long_side_fraction": 0.10,
        },
    )

    assert result.accepted
    assert result.eligible_annotation_ids == (1,)
    assert result.anchor_margins == ((1, 8.0, 8),)


def reflection_config() -> dict:
    return {
        "min_pad_px": 16,
        "max_pad_fraction": 0.31,
        "seam_probe_px": 16,
        "orthogonal_sample_size": 64,
        "max_pair_mae": 2.0,
        "min_pair_correlation": 0.995,
        "min_texture_std": 5.0,
    }


def test_reflected_padding_guard_detects_exact_top_bottom_mirror() -> None:
    rng = np.random.default_rng(42)
    center = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    image = np.concatenate(
        [center[:20][::-1], center, center[-20:][::-1]],
        axis=0,
    )

    result = reflected_padding_guard(image, guard_config=reflection_config())

    assert result.detected
    assert result.detected_axes == ("top_bottom",)
    assert result.top_bottom.pad_px == 20
    assert result.top_bottom.max_pair_mae == 0


def test_reflected_padding_guard_keeps_unrelated_borders() -> None:
    image = np.random.default_rng(7).integers(
        0,
        256,
        size=(100, 100, 3),
        dtype=np.uint8,
    )

    result = reflected_padding_guard(image, guard_config=reflection_config())

    assert not result.detected
    assert result.detected_axes == ()


def test_visible_paste_mask_excludes_only_layers_in_front() -> None:
    paste_mask = np.zeros((20, 20), dtype=bool)
    paste_mask[4:16, 4:16] = True
    front_mask = np.zeros_like(paste_mask)
    front_mask[10:18, 10:18] = True
    back_mask = np.zeros_like(paste_mask)
    back_mask[3:8, 3:8] = True
    paste_layer = Layer(
        instance_id="paste:0",
        class_name="helmet",
        kind="pasted",
        mask=paste_mask,
        bbox_xywh_original=[4, 4, 12, 12],
        y_bottom=16,
        z_index=1,
    )
    paste = Paste(
        layer=paste_layer,
        rgba=np.zeros((12, 12, 4), dtype=np.uint8),
        frame_slice=(slice(4, 16), slice(4, 16)),
        patch_slice=(slice(0, 12), slice(0, 12)),
        bank={},
        bbox_preclip=[4, 4, 12, 12],
    )
    back = Layer(
        instance_id="real:back",
        class_name="person",
        kind="existing",
        mask=back_mask,
        bbox_xywh_original=[3, 3, 5, 5],
        y_bottom=8,
        z_index=0,
    )
    front = Layer(
        instance_id="real:front",
        class_name="person",
        kind="existing",
        mask=front_mask,
        bbox_xywh_original=[10, 10, 8, 8],
        y_bottom=18,
        z_index=2,
    )

    visible = _visible_paste_masks(
        existing_layers=[back, front],
        pastes=[paste],
    )["paste:0"]

    assert np.array_equal(visible, paste_mask & ~front_mask)
    assert np.any(visible & back_mask)
