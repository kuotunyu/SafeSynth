"""The baseline's photometric augmentation must track the synthetic post-fx.

TRAIN-05 / EXP-01: if `standard_aug` does not receive photometric augmentation
comparable to what synthetic images carry, part of any `filtered_syn` win is just
"it got an augmentation the baseline never saw", and the claim collapses under
the first sharp question.

These two files drifted once already. ADR-013 recalibrated compose.yaml's
low-light range from [1.8, 3.2] to [1.8067, 2.3942] and training.yaml kept the
old numbers while still claiming to match, which is worse than never having
claimed it - the comment asserted a coupling the values no longer had.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "configs" / "compose.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def training() -> dict:
    return yaml.safe_load((ROOT / "configs" / "training.yaml").read_text(encoding="utf-8"))


# spec: TRAIN-05
@pytest.mark.parametrize(
    ("compose_key", "training_key"),
    [
        ("gamma", "gamma_range"),
        ("gain", "gain_range"),
        ("noise_sigma", "gauss_noise_sigma"),
    ],
)
def test_standard_aug_tracks_synthetic_low_light(
    compose: dict, training: dict, compose_key: str, training_key: str
) -> None:
    synthetic = compose["postfx"]["low_light"][compose_key]
    baseline = training["augmentation"]["standard_aug"][training_key]

    assert [float(v) for v in baseline] == [float(v) for v in synthetic], (
        f"configs/training.yaml augmentation.standard_aug.{training_key} = {baseline} "
        f"no longer matches configs/compose.yaml postfx.low_light.{compose_key} = "
        f"{synthetic}. Update both together or the four-arm comparison is unfair."
    )


# spec: TRAIN-05
def test_standard_aug_tracks_synthetic_jpeg_quality(compose: dict, training: dict) -> None:
    synthetic = compose["postfx"]["jpeg"]["quality"]
    baseline = training["augmentation"]["standard_aug"]["jpeg_quality"]

    assert [int(v) for v in baseline] == [int(v) for v in synthetic]


# spec: TRAIN-05
def test_blur_limit_covers_the_synthetic_kernel_lengths(
    compose: dict, training: dict
) -> None:
    """The baseline must be able to blur at least as hard as the synthetic pass."""

    longest_synthetic = max(compose["postfx"]["motion_blur"]["kernel_lengths"])
    baseline_limit = training["augmentation"]["standard_aug"]["blur_limit"]

    assert baseline_limit >= longest_synthetic - 2, (
        f"blur_limit {baseline_limit} is far below the synthetic motion-blur kernel "
        f"length {longest_synthetic}"
    )


# spec: TRAIN-06
def test_synthetic_arms_share_the_baseline_augmentation(training: dict) -> None:
    """Arms 3 and 4 differing in augmentation would confound the whole result."""

    assert training["augmentation"]["synthetic_arms_use_standard_aug"] is True


# spec: TRAIN-03
def test_there_are_exactly_four_arms_and_no_fifth(training: dict) -> None:
    """The generic five-arm protocol's 'full-real ceiling' does not apply here.

    Real-only already consumes the entire real Train split, so there is no higher
    real-data ceiling to measure against.
    """

    augmentation = training["augmentation"]
    assert "real_only" in augmentation
    assert "standard_aug" in augmentation
    assert augmentation["synthetic_arms_use_standard_aug"] is True
