from __future__ import annotations

import numpy as np

from scripts.analyze_flux2_v2_diagnostic import masked_metrics, pairwise_metrics


def test_masked_metrics_separate_inside_and_outside_changes() -> None:
    draft = np.zeros((4, 4, 3), dtype=np.uint8)
    output = draft.copy()
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    output[1, 1] = [3, 6, 9]
    output[0, 0] = [1, 1, 1]

    metrics = masked_metrics(draft, output, mask)

    assert metrics["edit_mask_pixels"] == 4
    assert metrics["changed_pixels_inside_mask"] == 1
    assert metrics["changed_pixel_fraction_inside_mask"] == 0.25
    assert metrics["outside_mask_changed_pixels"] == 1
    assert metrics["mae_rgb_inside_mask"] == 1.5


def test_pairwise_metrics_measure_only_registered_mask() -> None:
    first = np.zeros((3, 3, 3), dtype=np.uint8)
    second = first.copy()
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    second[1, 1] = [2, 4, 6]
    second[0, 0] = [100, 100, 100]

    metrics = pairwise_metrics(first, second, mask)

    assert metrics["different_pixels_inside_mask"] == 1
    assert metrics["different_pixel_fraction_inside_mask"] == 1.0
    assert metrics["mae_rgb_inside_mask"] == 4.0
