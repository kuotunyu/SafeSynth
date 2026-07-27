"""CPU-safe geometry and identity guards for whole-region inpainting."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def _ellipse_kernel(radius: int) -> np.ndarray:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    size = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def whole_person_edit_mask(
    person_mask: np.ndarray,
    headlike_mask: np.ndarray,
    *,
    outer_dilate_px: int,
) -> np.ndarray:
    """Return the dilated union of an existing person and its paired head label."""

    person = np.asarray(person_mask, dtype=bool)
    headlike = np.asarray(headlike_mask, dtype=bool)
    if person.ndim != 2 or headlike.shape != person.shape:
        raise ValueError("person_mask and headlike_mask must be matching 2D masks")
    support = person | headlike
    if not support.any():
        raise ValueError("The paired person support is empty")
    if outer_dilate_px == 0:
        return support
    return cv2.dilate(
        support.astype(np.uint8),
        _ellipse_kernel(outer_dilate_px),
    ).astype(bool)


def mask_edge_margin(mask: np.ndarray) -> int:
    """Return the minimum pixel margin between a mask and the image frame."""

    support = np.asarray(mask, dtype=bool)
    if support.ndim != 2 or not support.any():
        raise ValueError("mask must be a non-empty 2D mask")
    rows, columns = np.nonzero(support)
    height, width = support.shape
    return int(
        min(
            columns.min(),
            rows.min(),
            width - 1 - columns.max(),
            height - 1 - rows.max(),
        )
    )


def maximum_other_mask_overlap_fraction(
    edit_mask: np.ndarray,
    other_masks: Iterable[np.ndarray],
) -> float:
    """Measure overlap against the smaller mask for every unrelated instance."""

    edit = np.asarray(edit_mask, dtype=bool)
    if edit.ndim != 2 or not edit.any():
        raise ValueError("edit_mask must be a non-empty 2D mask")
    maximum = 0.0
    for source in other_masks:
        other = np.asarray(source, dtype=bool)
        if other.shape != edit.shape:
            raise ValueError("Every other mask must match edit_mask")
        denominator = min(int(edit.sum()), int(other.sum()))
        if denominator == 0:
            continue
        overlap = int((edit & other).sum()) / denominator
        maximum = max(maximum, float(overlap))
    return maximum


def enforce_outside_edit_exact(
    draft_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    *,
    edit_mask: np.ndarray,
) -> np.ndarray:
    """Keep generated pixels only inside the registered whole-region mask."""

    draft = np.asarray(draft_rgb, dtype=np.uint8)
    generated = np.asarray(generated_rgb, dtype=np.uint8)
    editable = np.asarray(edit_mask, dtype=bool)
    if (
        draft.ndim != 3
        or draft.shape != generated.shape
        or draft.shape[:2] != editable.shape
    ):
        raise ValueError("Draft, generated image, and edit mask shapes disagree")
    output = draft.copy()
    output[editable] = generated[editable]
    return output


def region_identity_metrics(
    draft_rgb: np.ndarray,
    output_rgb: np.ndarray,
    *,
    edit_mask: np.ndarray,
) -> dict[str, float | int]:
    """Measure the v8 pixel-exact invariant and whether the model edited the mask."""

    draft = np.asarray(draft_rgb, dtype=np.uint8)
    output = np.asarray(output_rgb, dtype=np.uint8)
    editable = np.asarray(edit_mask, dtype=bool)
    if draft.shape != output.shape or draft.shape[:2] != editable.shape:
        raise ValueError("Draft, output image, and edit mask shapes disagree")
    changed = np.any(draft != output, axis=2)
    return {
        "outside_edit_changed_pixels": int((changed & ~editable).sum()),
        "edit_mask_changed_fraction": float(
            (changed & editable).sum() / max(int(editable.sum()), 1)
        ),
    }
