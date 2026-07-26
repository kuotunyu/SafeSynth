from __future__ import annotations

import numpy as np

from src.synthetic.calibration import distribution, seam_energy_ratio


def test_distribution_has_required_h7_percentiles() -> None:
    result = distribution(range(1, 101))

    assert result["n"] == 100
    assert result["p1"] == 1.99
    assert result["p50"] == 50.5
    assert result["p99"] == 99.01


def test_seam_energy_detects_strong_boundary() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 255
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    ratio = seam_energy_ratio(image, mask, band_px=2)

    assert ratio is not None
    assert ratio > 10


def test_seam_energy_is_zero_without_texture() -> None:
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:14, 2:14] = True

    assert seam_energy_ratio(image, mask) == 0
