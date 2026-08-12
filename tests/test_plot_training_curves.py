"""Curve re-aggregation: the numbers, not the pixels.

The figure itself is checked by eye. What is worth
pinning here is everything the figure and its summary table are derived from,
because a wrong best-step or a decay computed the wrong way round would produce
a plausible-looking chart that says the opposite of the truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_training_curves import (
    CurveDataError,
    read_arm_curve,
    render_summary,
)

# A curve that rises, peaks in the MIDDLE, then falls. The peak has to be
# interior: if it were the first or last point, "take the max" and "take the
# first" and "take the last" would all agree and none of them would be tested.
POINTS = [
    (200, 0.10, 0.05),
    (400, 0.30, 0.22),
    (600, 0.42, 0.31),   # <- peak
    (800, 0.25, 0.18),
    (1000, 0.20, 0.14),  # <- final
]
PEAK_STEP, PEAK_MAP, FINAL_MAP = 600, 0.42, 0.20


def _write_arm(root: Path, arm: str, points=POINTS, *, seed: int = 1337) -> Path:
    seed_dir = root / arm / f"seed_{seed}"
    # Two checkpoints, and the LOWER-numbered one gets the richer history, so a
    # reader that picks the wrong checkpoint sees different numbers.
    (seed_dir / "checkpoint-600").mkdir(parents=True, exist_ok=True)
    (seed_dir / "checkpoint-600" / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"eval_map": 0.42, "step": 600}]}),
        encoding="utf-8",
    )
    last = seed_dir / "checkpoint-1000"
    last.mkdir(parents=True, exist_ok=True)
    last.joinpath("trainer_state.json").write_text(
        json.dumps(
            {
                "log_history": [
                    {"loss": 9.0, "step": step} for step, _, _ in points
                ]
                + [
                    {"eval_map": m, "eval_map_small": s, "step": step}
                    for step, m, s in points
                ]
            }
        ),
        encoding="utf-8",
    )
    return seed_dir


def test_the_curve_comes_from_the_last_checkpoint_not_the_first(tmp_path: Path) -> None:
    """checkpoint-600 holds one point; only the last checkpoint has the history."""

    _write_arm(tmp_path, "real_only")

    curve = read_arm_curve(tmp_path, "real_only", 1337)

    assert len(curve.steps) == len(POINTS)


def test_the_peak_is_the_maximum_not_the_first_or_last_point(tmp_path: Path) -> None:
    _write_arm(tmp_path, "real_only")

    curve = read_arm_curve(tmp_path, "real_only", 1337)

    assert curve.best_step == PEAK_STEP
    assert curve.best_map == pytest.approx(PEAK_MAP)
    assert curve.final_map == pytest.approx(FINAL_MAP)


def test_decay_is_peak_minus_final_and_is_positive_when_the_arm_fell(
    tmp_path: Path,
) -> None:
    """0.42 - 0.20 = 0.22. Subtracting the other way round would be negative."""

    _write_arm(tmp_path, "real_only")

    curve = read_arm_curve(tmp_path, "real_only", 1337)

    assert curve.decay_points == pytest.approx(0.22)


def test_train_loss_entries_are_not_mistaken_for_eval_points(tmp_path: Path) -> None:
    """log_history interleaves loss rows that carry no eval_map."""

    _write_arm(tmp_path, "real_only")

    curve = read_arm_curve(tmp_path, "real_only", 1337)

    assert len(curve.steps) == 5  # not 10, which is what ignoring the filter gives


def test_points_are_ordered_by_step_even_when_the_log_is_not(tmp_path: Path) -> None:
    shuffled = [POINTS[3], POINTS[0], POINTS[4], POINTS[2], POINTS[1]]
    _write_arm(tmp_path, "real_only", points=shuffled)

    curve = read_arm_curve(tmp_path, "real_only", 1337)

    assert curve.steps == [200, 400, 600, 800, 1000]
    assert curve.series["eval_map"][0] == pytest.approx(0.10)


def test_a_missing_arm_directory_names_the_arm(tmp_path: Path) -> None:
    with pytest.raises(CurveDataError, match="no checkpoint"):
        read_arm_curve(tmp_path, "filtered_syn", 1337)


def test_a_checkpoint_without_trainer_state_is_reported(tmp_path: Path) -> None:
    (tmp_path / "real_only" / "seed_1337" / "checkpoint-100").mkdir(parents=True)

    with pytest.raises(CurveDataError, match="no trainer_state.json"):
        read_arm_curve(tmp_path, "real_only", 1337)


def test_a_state_with_no_eval_entries_is_reported(tmp_path: Path) -> None:
    checkpoint = tmp_path / "real_only" / "seed_1337" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"loss": 1.0, "step": 10}]}), encoding="utf-8"
    )

    with pytest.raises(CurveDataError, match="no eval entries"):
        read_arm_curve(tmp_path, "real_only", 1337)


def test_the_summary_ranks_by_best_and_names_the_worst_decay(tmp_path: Path) -> None:
    _write_arm(tmp_path, "real_only")
    steep = [(200, 0.05, 0.02), (400, 0.50, 0.40), (600, 0.05, 0.02)]
    _write_arm(tmp_path, "filtered_syn", points=steep)

    curves = [
        read_arm_curve(tmp_path, "real_only", 1337),
        read_arm_curve(tmp_path, "filtered_syn", 1337),
    ]
    rendered = render_summary(curves, Path("reports/figures/training_curves.png"))

    body = [line for line in rendered.splitlines() if line.startswith("| `")]
    # filtered_syn peaks at 0.50 against real_only's 0.42, so it ranks first.
    assert body[0].startswith("| `filtered_syn`")
    # and it falls 0.45 against real_only's 0.22, so it is also the worst decay.
    assert "the largest fall is `filtered_syn`" in rendered
    assert "−0.4500" in rendered


def test_the_summary_states_the_equal_step_exposure_confound() -> None:
    """Any comparison of these curves is invalid without it, so it is not optional."""

    rendered = render_summary(
        [
            type(
                "C",
                (),
                {
                    "arm": "real_only",
                    "steps": [1],
                    "series": {},
                    "best_step": 1,
                    "best_map": 0.3,
                    "final_map": 0.2,
                    "decay_points": 0.1,
                },
            )()
        ],
        Path("reports/figures/training_curves.png"),
    )

    assert "TRAIN-07" in rendered
    assert "half as often" in rendered
