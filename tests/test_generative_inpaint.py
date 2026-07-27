from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.synthetic.generative_inpaint import (
    GenerativeBoundaryInpainter,
    boundary_edit_mask,
    enforce_identity_regions,
    load_generative_config,
    reference_canvas,
)


def rectangle(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x, y, width, height = box
    mask[y : y + height, x : x + width] = True
    return mask


def test_registered_model_and_h4_gate_are_frozen() -> None:
    config = load_generative_config()

    assert config["model"]["revision"] == "a3b4f4849157f664bdbc776fd7453c2783562f4d"
    assert config["model"]["license"] == "apache-2.0"
    assert config["model"]["required_download_bytes"] == 15_980_131_711
    assert config["runtime"]["local_files_only"] is True
    assert config["final_h4"]["max_auc_for_scaleup"] == 0.60
    assert config["status"] == "guarded_v3_preregistered_no_output"
    assert config["pilot"]["architecture"] == "guarded_context_replacement_v3"
    assert config["pilot"]["root_seed"] == 20260729
    assert config["pilot"]["previous_failed_root_seed"] == 20260728
    assert config["pilot"]["original_failed_root_seed"] == 20260727
    assert config["pilot"]["input_preflight_issue_max_count"] == 0


def test_boundary_mask_preserves_minimum_core_and_edits_both_sides() -> None:
    support = rectangle((48, 48), (15, 14, 16, 18))

    edit, core = boundary_edit_mask(
        support,
        outer_dilate_px=5,
        protected_core_erode_px=2,
        minimum_protected_core_fraction=0.35,
    )

    assert not np.any(edit & core)
    assert core.sum() >= np.ceil(0.35 * support.sum())
    assert np.any(edit & support)
    assert np.any(edit & ~support)


def test_identity_enforcement_discards_every_model_change_outside_band() -> None:
    draft = np.full((24, 24, 3), 20, dtype=np.uint8)
    generated = np.full_like(draft, 240)
    edit = rectangle((24, 24), (8, 8, 8, 8))
    core = rectangle((24, 24), (10, 10, 4, 4))
    edit &= ~core

    output = enforce_identity_regions(
        draft,
        generated,
        edit_mask=edit,
        protected_core=core,
    )

    assert np.all(output[edit] == 240)
    assert np.array_equal(output[~edit], draft[~edit])
    assert np.array_equal(output[core], draft[core])


def test_reference_canvas_keeps_rgba_object_on_neutral_background() -> None:
    rgba = np.zeros((12, 8, 4), dtype=np.uint8)
    rgba[2:10, 1:7, :3] = [240, 190, 20]
    rgba[2:10, 1:7, 3] = 255

    canvas = np.asarray(
        reference_canvas(
            rgba,
            canvas_size=64,
            max_fill=0.75,
            background_rgb=(127, 127, 127),
        )
    )

    assert canvas.shape == (64, 64, 3)
    assert np.any(np.all(canvas == [240, 190, 20], axis=2))
    assert np.all(canvas[0, 0] == [127, 127, 127])


class FakePipeline:
    def __call__(self, **kwargs: object) -> SimpleNamespace:
        image = np.asarray(kwargs["image"], dtype=np.uint8)
        generated = np.full_like(image, [220, 30, 30])
        return SimpleNamespace(images=[Image.fromarray(generated)])


def test_engine_is_pixel_exact_outside_registered_edit_mask() -> None:
    config = load_generative_config()
    draft = np.full((64, 64, 3), [40, 70, 100], dtype=np.uint8)
    object_mask = rectangle((64, 64), (24, 22, 16, 20))
    rgba = np.zeros((20, 16, 4), dtype=np.uint8)
    rgba[..., :3] = [220, 180, 30]
    rgba[..., 3] = 255

    result = GenerativeBoundaryInpainter(FakePipeline(), config).generate(
        draft_rgb=draft,
        object_mask=object_mask,
        reference_rgba=rgba,
        class_name="helmet",
        seed=42,
    )

    assert np.array_equal(result.image_rgb[~result.edit_mask], draft[~result.edit_mask])
    assert np.array_equal(
        result.image_rgb[result.protected_core], draft[result.protected_core]
    )
    assert result.provenance["identity_metrics"]["outside_edit_changed_pixels"] == 0
    assert result.provenance["identity_metrics"]["protected_core_changed_pixels"] == 0
