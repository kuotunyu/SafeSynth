from __future__ import annotations

import numpy as np
from PIL import Image

from src.synthetic.mask_ops import (
    clean_and_measure_mask,
    decode_rle,
    encode_rle,
    fill_holes,
    keep_largest_component,
)
from src.synthetic.sam2_runner import build_crop_canvas, crop_mask_to_global


def test_mask_cleanup_clips_leakage_and_keeps_largest_component() -> None:
    raw = np.zeros((20, 20), dtype=bool)
    raw[5:15, 5:15] = True
    raw[8:10, 8:10] = False
    raw[0:2, 0:2] = True

    cleaned, metrics = clean_and_measure_mask(
        raw, [4, 4, 16, 16], morph_close_kernel=1
    )

    assert cleaned.sum() == 100
    assert not cleaned[0:2, 0:2].any()
    assert metrics["component_count"] == 2
    assert metrics["outside_box_ratio"] > 0
    assert metrics["hole_fill_ratio"] > 1


def test_fill_holes_and_largest_are_deterministic() -> None:
    mask = np.zeros((12, 12), dtype=bool)
    mask[2:10, 2:10] = True
    mask[4:8, 4:8] = False
    mask[0, 0] = True

    largest = keep_largest_component(mask)
    filled = fill_holes(largest)

    assert largest.sum() == 48
    assert filled.sum() == 64


def test_json_safe_rle_round_trip() -> None:
    mask = np.zeros((17, 23), dtype=bool)
    mask[3:13, 4:19] = True
    mask[5:7, 8:10] = False

    assert np.array_equal(decode_rle(encode_rle(mask)), mask)


def test_crop_canvas_mask_round_trip_geometry() -> None:
    image = Image.new("RGB", (416, 416), "gray")
    box = [100.0, 120.0, 140.0, 160.0]
    _, prompt, transform = build_crop_canvas(
        image,
        box,
        context_pad_frac=0.6,
        min_crop_side_px=96,
        target_size=512,
    )
    canvas_mask = np.zeros((1024, 1024), dtype=bool)
    left, top, right, bottom = (round(value) for value in prompt)
    canvas_mask[top:bottom, left:right] = True

    restored = crop_mask_to_global(canvas_mask, transform)

    ys, xs = np.where(restored)
    assert abs(int(xs.min()) - 100) <= 1
    assert abs(int(xs.max()) + 1 - 140) <= 1
    assert abs(int(ys.min()) - 120) <= 1
    assert abs(int(ys.max()) + 1 - 160) <= 1
