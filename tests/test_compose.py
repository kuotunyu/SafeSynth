from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.synthetic.compose import (
    Paste,
    _decode_rle,
    _drop_illegible_source_material,
    _encode_rle,
    _generative_seed,
    _requested_classes,
    _sample_seed,
    _scenario_sequence,
    _transform_scale,
    _visible_masks,
    context_replacement_input_guard,
    normalize_reflected_padding,
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


def test_default_preview_covers_every_scenario() -> None:
    """M9 closed, so hard_negative now participates like any other scenario.

    kuotunyu signed off the H6 sheet at 0/64 real helmets and the procedural bank
    is materialised, so the compositor no longer excludes it. Its distractors are
    unannotated by construction (ADR-004) and are asserted separately in
    test_hard_negatives_never_produce_annotations.
    """

    sequence = _scenario_sequence(
        32, scenario_config(), None, np.random.default_rng(42)
    )

    assert set(sequence) == {
        "small_distant",
        "head_no_helmet",
        "partial_occlusion",
        "crowded",
        "hard_negative",
        "low_light_blur",
    }


def test_hard_negative_requests_no_annotated_pastes() -> None:
    """The scenario must contribute zero annotated instances of its own.

    Everything _requested_classes returns becomes an annotated paste and then a
    COCO annotation. Hard negatives are composited in a separate unannotated
    pass, so this must stay empty or the distractors would gain labels.
    """

    assert (
        _requested_classes(
            "hard_negative", 3, np.random.default_rng(0), person_crowded_fallback=False
        )
        == []
    )


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
        "action": "normalize_cover_crop",
        "min_pad_px": 8,
        "max_pad_fraction": 0.31,
        "orthogonal_sample_size": 64,
        "max_pair_mae": 3.0,
        "min_pair_correlation": 0.97,
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
    assert result.top_bottom.start.pad_px == 20
    assert result.top_bottom.end.pad_px == 20
    assert result.top_bottom.start.pair_mae == 0
    assert result.top_bottom.end.pair_mae == 0


def test_reflected_padding_guard_detects_low_texture_near_mirror() -> None:
    rng = np.random.default_rng(123)
    center = np.clip(
        128 + rng.normal(0, 10, size=(80, 60, 3)),
        0,
        255,
    ).astype(np.uint8)
    image = np.concatenate(
        [center[:20][::-1], center, center[-20:][::-1]],
        axis=0,
    ).astype(np.int16)
    noise = np.zeros_like(image)
    noise[:20] = np.rint(
        rng.normal(0, 2, size=noise[:20].shape)
    ).astype(np.int16)
    noise[-20:] = np.rint(
        rng.normal(0, 2, size=noise[-20:].shape)
    ).astype(np.int16)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    result = reflected_padding_guard(image, guard_config=reflection_config())

    assert result.detected
    assert 0.97 <= result.top_bottom.start.pair_correlation < 0.995
    assert result.top_bottom.start.pair_mae <= 3.0
    assert result.top_bottom.start.texture_std >= 5.0


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


def test_reflected_padding_normalization_transforms_labels_and_masks() -> None:
    rng = np.random.default_rng(91)
    center = rng.integers(0, 256, size=(60, 100, 3), dtype=np.uint8)
    image = np.concatenate(
        [center[:20][::-1], center, center[-20:][::-1]],
        axis=0,
    )
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:50, 40:60] = True
    annotations = [{"id": 7, "category_id": 1, "bbox": [40, 30, 20, 20]}]
    pass1 = {
        7: {
            "qc_pass": True,
            "segmentation": _encode_rle(mask),
        }
    }
    reflection = reflected_padding_guard(
        image,
        guard_config=reflection_config(),
    )

    normalized, transformed, transformed_pass1, provenance = (
        normalize_reflected_padding(
            image,
            annotations=annotations,
            pass1=pass1,
            reflection=reflection,
            output_shape=(100, 100),
        )
    )

    assert normalized.shape == (100, 100, 3)
    assert provenance.applied
    assert provenance.crop_xyxy == (0, 20, 100, 80)
    assert provenance.detected_sides == ("top", "bottom")
    assert len(transformed) == 1
    assert transformed[0]["bbox"][2] > 30
    transformed_mask = _decode_rle(transformed_pass1[7]["segmentation"])
    assert transformed_mask.shape == (100, 100)
    assert transformed_mask.any()


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

    visible = _visible_masks(
        existing_layers=[back, front],
        pastes=[paste],
    )["paste:0"]

    assert np.array_equal(visible, paste_mask & ~front_mask)
    assert np.any(visible & back_mask)


def _write_cutout(directory, name, *, luma: int, size: int = 24) -> dict:
    """A solid-colour RGBA cutout with a circular alpha, at a chosen brightness."""

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = luma
    yy, xx = np.mgrid[0:size, 0:size]
    rgba[..., 3] = np.where(
        (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size / 2 - 1) ** 2, 255, 0
    )
    path = directory / f"{name}.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    return {"cutout_id": name, "class_name": "head", "file": path.name}


# spec: CUT-14
def test_black_silhouette_source_material_is_refused(tmp_path) -> None:
    """FILT-15 cannot catch these: harmonization lifts the mean before it looks.

    Measured case 001610_ann008186 — source luma 8.5, composite luma 45.4, which
    clears the FILT-15 floor while still being a featureless blob.
    """

    bank = [
        _write_cutout(tmp_path, "silhouette", luma=9),
        _write_cutout(tmp_path, "usable", luma=120),
    ]
    paths = SimpleNamespace(cutouts=tmp_path)

    kept, report = _drop_illegible_source_material(
        bank, paths=paths, floors={"head": 23.19}
    )

    assert [item["cutout_id"] for item in kept] == ["usable"]
    assert report == {"n_before": 2, "n_after": 1, "dropped_by_class": {"head": 1}}


# spec: CUT-14
def test_source_gate_ignores_classes_without_a_floor(tmp_path) -> None:
    bank = [_write_cutout(tmp_path, "dark", luma=3)]
    paths = SimpleNamespace(cutouts=tmp_path)

    kept, report = _drop_illegible_source_material(bank, paths=paths, floors={})

    assert len(kept) == 1
    assert report["dropped_by_class"] == {}


# spec: COMP-18
def test_swapped_head_inherits_the_removed_helmet_size() -> None:
    """The swap sets position AND scale from its anchor.

    Regression: only the centre was inherited, so the head kept the scenario's
    generic scale_range and came out several times too large for the body -
    measured 52x68 replacing a 24x30 helmet.
    """

    rgba = np.zeros((60, 50, 4), dtype=np.uint8)
    rgba[5:55, 5:45, 3] = 255
    anchor = [213.0, 169.0, 24.0, 30.0]

    scale = _transform_scale(
        scenario="head_no_helmet",
        settings={"scale_range": (0.60, 1.00)},
        class_name="head",
        rgba=rgba,
        config={"compose": {"max_paste_scale": {"head": 3.0}}},
        rng=np.random.default_rng(0),
        target_bbox_xywh=anchor,
    )

    assert abs(scale * 40 - anchor[2]) < 1.0
    assert abs(scale * 50 - anchor[3]) < 1.0


# spec: COMP-18
def test_swap_scale_ignores_the_generic_scale_range() -> None:
    """A wide scale_range must not move the result when an anchor is given."""

    rgba = np.zeros((60, 50, 4), dtype=np.uint8)
    rgba[5:55, 5:45, 3] = 255
    anchor = [10.0, 10.0, 24.0, 30.0]
    kwargs = {
        "scenario": "head_no_helmet",
        "class_name": "head",
        "rgba": rgba,
        "config": {"compose": {"max_paste_scale": {"head": 3.0}}},
        "target_bbox_xywh": anchor,
    }

    narrow = _transform_scale(
        settings={"scale_range": (0.60, 0.61)}, rng=np.random.default_rng(1), **kwargs
    )
    wide = _transform_scale(
        settings={"scale_range": (0.20, 2.50)}, rng=np.random.default_rng(2), **kwargs
    )

    assert narrow == wide
