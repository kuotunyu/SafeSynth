from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.synthetic.hard_negatives import (
    procedural_hard_negative,
    select_h6_images,
    validate_human_signoff,
)


def test_h6_selection_is_stable_and_order_independent() -> None:
    first = select_h6_images(list(range(100)), seed=42, n=20)
    second = select_h6_images(list(reversed(range(100))), seed=42, n=20)

    assert first == second
    assert len(first) == 20


@pytest.mark.parametrize("shape", ["dome", "ellipse", "rounded_cylinder", "arc"])
def test_procedural_shapes_preserve_real_texture_variation(shape: str) -> None:
    yy, xx = np.mgrid[0:64, 0:64]
    texture = np.dstack(
        ((xx * 4) % 256, (yy * 4) % 256, ((xx + yy) * 2) % 256)
    ).astype(np.uint8)

    rgba = procedural_hard_negative(texture, shape=shape, seed=42)
    inside = rgba[..., 3] >= 128

    assert rgba.shape == (64, 64, 4)
    assert inside.any()
    assert rgba[..., :3][inside].std() > 5
    assert np.any((rgba[..., 3] > 0) & (rgba[..., 3] < 255))


def test_human_signoff_is_bound_to_user_and_grid(tmp_path: Path) -> None:
    path = tmp_path / "signoff.json"
    path.write_text(
        json.dumps(
            {
                "approved": True,
                "approved_by": "kuotunyu",
                "grid_sha256": "abc",
                "real_helmet_count": 2,
            }
        ),
        encoding="utf-8",
    )

    result = validate_human_signoff(
        signoff_path=path, expected_grid_sha256="abc"
    )

    assert result["real_helmet_count"] == 2


def test_missing_human_signoff_hard_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="required and missing"):
        validate_human_signoff(
            signoff_path=tmp_path / "missing.json", expected_grid_sha256="abc"
        )
