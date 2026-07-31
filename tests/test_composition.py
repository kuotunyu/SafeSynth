from __future__ import annotations

import cv2
import numpy as np

from src.synthetic.composition import (
    Layer,
    LegibilityTarget,
    alpha_composite,
    apply_postfx,
    assign_z_order,
    box_legibility,
    decontaminate_soft_edge,
    draw_contact_shadow,
    feather_alpha,
    inpaint_masked_object,
    match_high_frequency_noise,
    object_legibility,
    placement_slices,
    poisson_composite,
    recompute_visible_annotations,
    seam_energy_ratio,
    tight_bbox,
    warp_rgba,
)


def rectangle(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x, y, width, height = box
    mask[y : y + height, x : x + width] = True
    return mask


def test_tight_bbox_uses_half_open_coordinates() -> None:
    mask = rectangle((30, 40), (7, 5, 11, 13))

    assert tight_bbox(mask) == [7.0, 5.0, 11.0, 13.0]


def test_visible_bbox_matches_closed_form_within_one_pixel() -> None:
    existing = Layer(
        "existing",
        "helmet",
        "existing",
        rectangle((100, 100), (10, 10, 40, 40)),
        [10, 10, 40, 40],
        y_bottom=50,
    )
    pasted = Layer(
        "pasted",
        "person",
        "pasted",
        rectangle((100, 100), (30, 0, 40, 80)),
        [30, 0, 40, 80],
        y_bottom=80,
    )

    result = recompute_visible_annotations(
        [existing, pasted],
        min_visible_fraction_pasted=0.2,
        existing_keep_original_above=0.6,
        existing_recompute_above=0.2,
    )

    assert result.accepted
    existing_output = next(
        item for item in result.annotations if item["instance_id"] == "existing"
    )
    assert existing_output["bbox_xywh"] == [10.0, 10.0, 20.0, 40.0]
    assert existing_output["visible_fraction"] == 0.5


def test_existing_heavily_occluded_rejects_placement_not_label() -> None:
    existing = Layer(
        "e",
        "head",
        "existing",
        rectangle((60, 60), (10, 10, 30, 30)),
        [10, 10, 30, 30],
        y_bottom=40,
    )
    covering = Layer(
        "p",
        "helmet",
        "pasted",
        rectangle((60, 60), (8, 8, 35, 35)),
        [8, 8, 35, 35],
        y_bottom=43,
    )

    result = recompute_visible_annotations(
        [existing, covering],
        min_visible_fraction_pasted=0.2,
        existing_keep_original_above=0.6,
        existing_recompute_above=0.2,
    )

    assert not result.accepted
    assert result.reason == "EXISTING_OBJECT_TOO_OCCLUDED"
    assert result.rejected_instance_ids == ("e",)


def test_y_bottom_controls_z_order() -> None:
    layers = [
        Layer("near", "person", "pasted", rectangle((10, 10), (0, 0, 1, 1)), [0, 0, 1, 1], 9),
        Layer("far", "person", "pasted", rectangle((10, 10), (0, 0, 1, 1)), [0, 0, 1, 1], 2),
    ]

    assert [layer.instance_id for layer in assign_z_order(layers)] == ["far", "near"]


def test_placement_slices_reports_preclip_inside_ratio() -> None:
    frame_slice, patch_slice, inside = placement_slices(
        frame_shape=(100, 100), patch_shape=(40, 40), center_xy=(10, 50)
    )

    assert inside == 0.75
    assert frame_slice == (slice(30, 70), slice(0, 30))
    assert patch_slice == (slice(0, 40), slice(10, 40))


def test_warp_and_alpha_composite_are_geometry_safe() -> None:
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[2:18, 3:27, :3] = [255, 200, 0]
    rgba[2:18, 3:27, 3] = 255
    warped = warp_rgba(rgba, scale=0.5, rotation_deg=10, hflip=True)
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame_slice, patch_slice, inside = placement_slices(
        frame_shape=(50, 50), patch_shape=warped.shape[:2], center_xy=(25, 25)
    )
    output = alpha_composite(
        frame,
        warped[..., :3],
        warped[..., 3],
        frame_slice=frame_slice,
        patch_slice=patch_slice,
    )

    assert inside == 1
    assert output.sum() > 0


def test_poisson_composite_is_deterministic_and_preserves_shape() -> None:
    frame = np.full((64, 64, 3), [40, 70, 100], dtype=np.uint8)
    patch = np.zeros((24, 24, 3), dtype=np.uint8)
    patch[..., 0] = np.arange(24, dtype=np.uint8)[:, None] * 8
    patch[..., 1] = np.arange(24, dtype=np.uint8)[None, :] * 8
    alpha = np.zeros((24, 24), dtype=np.uint8)
    alpha[3:21, 3:21] = 255
    frame_slice, patch_slice, _ = placement_slices(
        frame_shape=frame.shape[:2],
        patch_shape=patch.shape[:2],
        center_xy=(32, 32),
    )

    first = poisson_composite(
        frame,
        patch,
        alpha,
        frame_slice=frame_slice,
        patch_slice=patch_slice,
    )
    second = poisson_composite(
        frame,
        patch,
        alpha,
        frame_slice=frame_slice,
        patch_slice=patch_slice,
    )

    assert first.shape == frame.shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, frame)


def test_inpaint_covers_dilated_helmet_mask() -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    image[20:40, 20:44] = [255, 200, 0]
    mask = rectangle((64, 64), (20, 20, 24, 20))

    output, inpaint_mask = inpaint_masked_object(
        image, mask, dilate_px=3, radius=3
    )

    assert output.shape == image.shape
    assert inpaint_mask.sum() > mask.sum()
    assert not np.array_equal(output[mask], image[mask])


def test_feather_alpha_never_expands_support() -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    alpha[8:24, 8:24] = 255
    config = {
        "erode_before_feather_px": 1,
        "feather_sigma_base": 0.6,
        "feather_sigma_per_px": 0.02,
        "feather_sigma_clip": [0.6, 2.0],
    }

    result = feather_alpha(alpha, config=config)

    assert not np.any(result[alpha == 0])
    assert np.any((result > 0) & (result < 255))
    assert result[8, 16] > 0


def test_soft_edge_decontamination_extends_foreground_colour() -> None:
    rgb = np.full((9, 9, 3), [20, 180, 20], dtype=np.uint8)
    rgb[3:6, 3:6] = [220, 30, 30]
    alpha = np.zeros((9, 9), dtype=np.uint8)
    alpha[2:7, 2:7] = 80
    alpha[3:6, 3:6] = 255

    result = decontaminate_soft_edge(rgb, alpha, core_alpha_min=192)

    assert np.array_equal(result[2, 4], [220, 30, 30])
    assert np.array_equal(result[0, 0], rgb[0, 0])


def test_noise_matching_is_seed_reproducible() -> None:
    patch = np.full((20, 20, 3), 120, dtype=np.uint8)
    alpha = np.full((20, 20), 255, dtype=np.uint8)
    target = np.indices((20, 20)).sum(axis=0) % 2 * 100
    target = np.repeat(target[..., None], 3, axis=2).astype(np.uint8)
    mask = np.ones((20, 20), dtype=bool)

    first = match_high_frequency_noise(
        patch, alpha, target, mask, sigma_cap=8, rng=np.random.default_rng(4)
    )
    second = match_high_frequency_noise(
        patch, alpha, target, mask, sigma_cap=8, rng=np.random.default_rng(4)
    )

    assert np.array_equal(first, second)
    assert not np.array_equal(first, patch)


def test_postfx_preserves_shape_and_is_seed_reproducible() -> None:
    image = np.full((24, 24, 3), 180, dtype=np.uint8)
    config = {
        "low_light": {
            "prob_given_postfx": 1,
            "gamma": [2, 2],
            "gain": [0.5, 0.5],
            "noise_sigma": [3, 3],
            "wb_gain_r": [1, 1],
            "wb_gain_b": [1, 1],
        },
        "motion_blur": {
            "prob_given_postfx": 1,
            "kernel_lengths": [3],
            "angle_deg": [0, 0],
        },
    }

    first, metadata = apply_postfx(
        image, config=config, rng=np.random.default_rng(8)
    )
    second, _ = apply_postfx(image, config=config, rng=np.random.default_rng(8))

    assert first.shape == image.shape
    assert np.array_equal(first, second)
    assert set(metadata) == {"low_light", "motion_blur"}


def _blackout_postfx_config() -> dict:
    """The measured worst case from the m13 1x pool: gamma 3.17 with gain 0.54."""

    return {
        "low_light": {
            "prob_given_postfx": 1,
            "gamma": [3.17, 3.17],
            "gain": [0.54, 0.54],
            "noise_sigma": [0, 0],
            "wb_gain_r": [1, 1],
            "wb_gain_b": [1, 1],
        },
        "motion_blur": {
            "prob_given_postfx": 0,
            "kernel_lengths": [3],
            "angle_deg": [0, 0],
        },
    }


def _textured_scene() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    image[16:48, 16:48] = rng.integers(90, 210, size=(32, 32, 3), dtype=np.uint8)
    return image


def test_postfx_blackout_is_clamped_when_an_object_must_stay_legible() -> None:
    image = _textured_scene()
    target = LegibilityTarget(bbox_xywh=(16, 16, 32, 32), min_mean_luma=45.58)

    unclamped, unclamped_meta = apply_postfx(
        image, config=_blackout_postfx_config(), rng=np.random.default_rng(3)
    )
    clamped, clamped_meta = apply_postfx(
        image,
        config=_blackout_postfx_config(),
        rng=np.random.default_rng(3),
        legibility_targets=[target],
    )

    assert box_legibility(unclamped, target.bbox_xywh)[0] < target.min_mean_luma
    assert box_legibility(clamped, target.bbox_xywh)[0] >= target.min_mean_luma
    assert unclamped_meta["low_light"]["strength_scale"] == 1.0
    assert 0 < clamped_meta["low_light"]["strength_scale"] < 1.0
    # The requested draw is preserved so the clamp is auditable, not invisible.
    assert clamped_meta["low_light"]["requested_gamma"] == 3.17
    assert clamped_meta["low_light"]["gamma"] < 3.17


def test_postfx_clamp_tracks_the_object_not_its_box() -> None:
    """A dark object inside a bright box is what the box statistic misses."""

    image = np.full((64, 64, 3), 240, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 28:36] = True
    image[mask] = 70

    box_only = LegibilityTarget(bbox_xywh=(16, 16, 32, 32), min_mean_luma=45.58)
    object_aware = LegibilityTarget(
        bbox_xywh=(16, 16, 32, 32), min_mean_luma=45.58, mask=mask
    )

    _, box_meta = apply_postfx(
        image,
        config=_blackout_postfx_config(),
        rng=np.random.default_rng(4),
        legibility_targets=[box_only],
    )
    darkened, object_meta = apply_postfx(
        image,
        config=_blackout_postfx_config(),
        rng=np.random.default_rng(4),
        legibility_targets=[object_aware],
    )

    assert box_meta["low_light"]["strength_scale"] == 1.0
    assert object_meta["low_light"]["strength_scale"] < 1.0
    assert object_legibility(darkened, mask, (16, 16, 32, 32))[0] >= 45.58


def test_postfx_clamp_does_not_disturb_rng_determinism() -> None:
    image = _textured_scene()
    target = LegibilityTarget(bbox_xywh=(16, 16, 32, 32), min_mean_luma=45.58)
    config = _blackout_postfx_config()
    config["low_light"]["noise_sigma"] = [6, 6]

    first, _ = apply_postfx(
        image, config=config, rng=np.random.default_rng(11), legibility_targets=[target]
    )
    second, _ = apply_postfx(
        image, config=config, rng=np.random.default_rng(11), legibility_targets=[target]
    )

    assert np.array_equal(first, second)


def test_postfx_leaves_full_strength_when_the_scene_can_absorb_it() -> None:
    image = np.full((64, 64, 3), 250, dtype=np.uint8)
    image[16:48, 16:32] = 210
    target = LegibilityTarget(bbox_xywh=(16, 16, 32, 32), min_mean_luma=10.0)

    _, metadata = apply_postfx(
        image,
        config=_blackout_postfx_config(),
        rng=np.random.default_rng(5),
        legibility_targets=[target],
    )

    assert metadata["low_light"]["strength_scale"] == 1.0


def _shadow_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "opacity": 0.32,
        "width_fraction": 0.95,
        "height_fraction": 0.26,
        "y_offset_fraction": 0.02,
        "blur_sigma_fraction": 0.16,
    }
    config.update(overrides)
    return config


def _dome_alpha(size: int = 40) -> np.ndarray:
    alpha = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(
        alpha, (size // 2, size - 1), (size // 2 - 1, size - 2), 0, 180, 360, 255, -1
    )
    return alpha


# spec: COMP-25
def test_contact_shadow_falls_below_the_footprint() -> None:
    """The shadow has to escape the patch rectangle or the object covers it."""

    image = np.full((120, 120, 3), 200, dtype=np.uint8)
    alpha = np.zeros((120, 120), dtype=np.uint8)
    alpha[40:80, 40:80] = _dome_alpha()

    shadowed = draw_contact_shadow(
        image,
        alpha,
        frame_slice=(slice(0, 120), slice(0, 120)),
        patch_slice=(slice(0, 120), slice(0, 120)),
        config=_shadow_config(),
    )

    below = shadowed[80:92, 45:75]
    assert below.mean() < 200, "no darkening appeared underneath the object"
    far_away = shadowed[0:20, 0:20]
    assert np.array_equal(far_away, image[0:20, 0:20])


# spec: COMP-25
def test_contact_shadow_is_wider_than_it_is_tall() -> None:
    """Both axes scale with footprint WIDTH; a tall shadow reads as an object."""

    image = np.full((160, 160, 3), 220, dtype=np.uint8)
    alpha = np.zeros((160, 160), dtype=np.uint8)
    alpha[60:100, 60:100] = _dome_alpha()

    shadowed = draw_contact_shadow(
        image,
        alpha,
        frame_slice=(slice(0, 160), slice(0, 160)),
        patch_slice=(slice(0, 160), slice(0, 160)),
        config=_shadow_config(),
    )
    darkened = shadowed.mean(axis=2) < 218
    darkened[:100] = False  # ignore rows the object itself occupies
    ys, xs = np.where(darkened)

    assert len(ys) > 0
    assert np.ptp(xs) > np.ptp(ys)


# spec: COMP-25
def test_contact_shadow_disabled_is_a_no_op() -> None:
    image = np.full((80, 80, 3), 180, dtype=np.uint8)
    alpha = np.zeros((80, 80), dtype=np.uint8)
    alpha[20:60, 20:60] = _dome_alpha()

    unchanged = draw_contact_shadow(
        image,
        alpha,
        frame_slice=(slice(0, 80), slice(0, 80)),
        patch_slice=(slice(0, 80), slice(0, 80)),
        config=_shadow_config(enabled=False),
    )

    assert np.array_equal(unchanged, image)


def test_object_legibility_ignores_background_inside_the_box() -> None:
    """The measured owner-review case: box mean 67.9, object mean 32.2."""

    image = np.full((64, 64, 3), 240, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 28:36] = True
    image[mask] = 30

    box_mean, _ = box_legibility(image, (16, 16, 32, 32))
    object_mean, _ = object_legibility(image, mask, (16, 16, 32, 32))

    assert box_mean > 150
    assert 25 < object_mean < 35


def test_object_legibility_falls_back_to_the_box_without_a_mask() -> None:
    image = _textured_scene()

    assert object_legibility(image, None, (16, 16, 32, 32)) == box_legibility(
        image, (16, 16, 32, 32)
    )


def test_seam_energy_detects_hard_boundary() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 255
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True

    assert seam_energy_ratio(image, mask, band_px=2) > 1
