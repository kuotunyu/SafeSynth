from __future__ import annotations

import numpy as np

from src.synthetic.region_inpaint import (
    adjacent_worker_box,
    enforce_outside_edit_exact,
    infer_person_box_from_headlike,
    mask_edge_margin,
    maximum_other_mask_overlap_fraction,
    region_identity_metrics,
    rounded_box_mask,
    whole_person_edit_mask,
)


def rectangle(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x, y, width, height = box
    mask[y : y + height, x : x + width] = True
    return mask


def test_whole_person_mask_includes_both_labels_and_outer_context() -> None:
    person = rectangle((32, 32), (10, 10, 8, 14))
    helmet = rectangle((32, 32), (11, 7, 6, 4))

    edit = whole_person_edit_mask(
        person,
        helmet,
        outer_dilate_px=2,
    )

    assert np.all(edit[person])
    assert np.all(edit[helmet])
    assert edit.sum() > (person | helmet).sum()
    assert mask_edge_margin(edit) == 5


def test_overlap_is_relative_to_smaller_instance() -> None:
    edit = rectangle((32, 32), (8, 8, 16, 16))
    unrelated = rectangle((32, 32), (20, 20, 6, 6))

    overlap = maximum_other_mask_overlap_fraction(edit, [unrelated])

    assert overlap == 16 / 36


def test_model_changes_are_pixel_exact_outside_whole_region() -> None:
    draft = np.full((24, 24, 3), [20, 40, 60], dtype=np.uint8)
    generated = np.full_like(draft, [220, 180, 20])
    edit = rectangle((24, 24), (7, 6, 10, 12))

    output = enforce_outside_edit_exact(
        draft,
        generated,
        edit_mask=edit,
    )

    assert np.array_equal(output[~edit], draft[~edit])
    assert np.array_equal(output[edit], generated[edit])
    metrics = region_identity_metrics(draft, output, edit_mask=edit)
    assert metrics["outside_edit_changed_pixels"] == 0
    assert metrics["edit_mask_changed_fraction"] == 1.0


def test_rounded_box_mask_stays_inside_registered_box() -> None:
    mask = rounded_box_mask(
        (40, 50),
        (10, 8, 20, 24),
        corner_fraction=0.2,
    )

    assert mask.any()
    assert not mask[:8].any()
    assert not mask[32:].any()
    assert not mask[:, :10].any()
    assert not mask[:, 30:].any()
    assert not mask[8, 10]
    assert mask[20, 20]


def test_head_geometry_infers_person_and_adjacent_ground_alignment() -> None:
    inferred = infer_person_box_from_headlike(
        (46.0, 24.0, 12.0, 16.0),
        person_width_over_height=0.4,
        head_center_x_fraction=0.5,
        head_center_y_fraction=0.2,
        head_height_fraction=0.2,
    )
    right = adjacent_worker_box(
        inferred,
        scale=0.9,
        side="right",
        gap_px=8,
    )

    assert inferred == (36.0, 16.0, 32.0, 80.0)
    assert right == (76, 24, 29, 72)
    assert right[1] + right[3] == round(inferred[1] + inferred[3])
