"""The slope arithmetic behind every hours figure this project reports.

The reporting contract forbids extrapolated schedules because one was wrong by 3x and cost
the user a night. The replacement is this calculation, so it had better be
right: a factor-of-two error here produces a confident, measured-looking, wrong
number - the exact failure mode the rule exists to prevent.
"""

from __future__ import annotations

import pytest

from scripts import probe_train_speed as speed_probe
from scripts.probe_train_speed import (
    PRODUCTION_STEPS,
    SpeedProbe,
    validate_step_pair,
)


def _probe(short_steps, long_steps, short_seconds, long_seconds) -> SpeedProbe:
    return SpeedProbe(
        label="probe",
        short_steps=short_steps,
        long_steps=long_steps,
        short_seconds=short_seconds,
        long_seconds=long_seconds,
    )


def test_the_slope_recovers_a_known_rate_and_a_known_overhead() -> None:
    """Construct from ground truth: 0.25 s/step plus 60 s of fixed cost."""

    probe = _probe(40, 140, 60 + 0.25 * 40, 60 + 0.25 * 140)

    assert probe.seconds_per_step == pytest.approx(0.25)
    assert probe.steps_per_second == pytest.approx(4.0)
    assert probe.fixed_overhead_seconds == pytest.approx(60.0)


def test_fixed_cost_cancels_which_is_the_entire_point() -> None:
    """Two machines, same per-step rate, wildly different setup cost.

    A single-run measurement would report them as different speeds. The slope
    must not.
    """

    quick_setup = _probe(40, 140, 5 + 0.25 * 40, 5 + 0.25 * 140)
    slow_setup = _probe(40, 140, 600 + 0.25 * 40, 600 + 0.25 * 140)

    assert quick_setup.seconds_per_step == pytest.approx(slow_setup.seconds_per_step)
    assert quick_setup.fixed_overhead_seconds == pytest.approx(5.0)
    assert slow_setup.fixed_overhead_seconds == pytest.approx(600.0)


def test_naive_division_would_have_been_wrong_and_by_how_much() -> None:
    """Why the slope exists, stated as a number rather than as a claim."""

    probe = _probe(40, 140, 60 + 0.25 * 40, 60 + 0.25 * 140)
    naive_rate = probe.short_seconds / probe.short_steps

    assert naive_rate == pytest.approx(1.75)
    assert naive_rate > 6 * probe.seconds_per_step, "the naive figure is 7x too slow"


def test_the_production_estimate_includes_the_fixed_cost_once() -> None:
    """Not zero times (understates), not once per step (absurd)."""

    probe = _probe(40, 140, 60 + 0.25 * 40, 60 + 0.25 * 140)
    expected = (0.25 * PRODUCTION_STEPS + 60.0) / 3600.0

    assert probe.production_hours() == pytest.approx(expected)
    assert probe.production_hours(steps=0) == pytest.approx(60.0 / 3600.0)


def test_a_slower_machine_reports_more_hours_not_fewer() -> None:
    """Sign errors in an inverse relationship are easy and catastrophic."""

    fast = _probe(40, 140, 0.10 * 40, 0.10 * 140)
    slow = _probe(40, 140, 0.80 * 40, 0.80 * 140)

    assert slow.production_hours() > fast.production_hours()
    assert slow.steps_per_second < fast.steps_per_second


def test_a_negative_intercept_is_surfaced_rather_than_swallowed() -> None:
    """It means the two runs disagree about setup cost, so the slope is junk.

    Reporting a confident hours figure from an incoherent measurement is worse
    than reporting nothing, so the rendered line has to say so.
    """

    # The long run came in disproportionately SLOW, so the fitted slope exceeds
    # even the short run's naive average and the line crosses below zero.
    # (My first attempt at this fixture had a POSITIVE intercept and asserted
    # nothing - the test passed while testing the opposite of its name.)
    incoherent = _probe(40, 140, 10.0, 100.0)

    assert incoherent.fixed_overhead_seconds < 0
    assert "NEGATIVE" in incoherent.render()


def test_a_coherent_probe_does_not_cry_wolf() -> None:
    assert "NEGATIVE" not in _probe(40, 140, 60 + 0.25 * 40, 60 + 0.25 * 140).render()


def test_a_probe_with_a_negative_intercept_is_rejected() -> None:
    """Removing the intercept gate would turn incoherent timings into an ETA."""

    with pytest.raises(speed_probe.SpeedProbeError, match="negative fixed overhead"):
        speed_probe.validate_probe(_probe(40, 140, 10.0, 100.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_training_loss_is_rejected(value: float) -> None:
    """A poisoned short run must not feed the slope or start production."""

    with pytest.raises(speed_probe.SpeedProbeError, match="train_loss"):
        speed_probe.validate_finite_run_record(
            {"train_loss": value, "eval_metrics": {}}
        )


def test_a_valid_probe_and_finite_record_pass() -> None:
    probe = _probe(40, 140, 60 + 0.25 * 40, 60 + 0.25 * 140)

    assert speed_probe.validate_probe(probe) is None
    assert speed_probe.validate_finite_run_record(
        {"train_loss": 1.25, "eval_metrics": {"eval_map": 0.1}}
    ) is None


def test_the_rendered_line_carries_the_units_a_reader_needs() -> None:
    rendered = _probe(40, 140, 60 + 0.25 * 40, 60 + 0.25 * 140).render()

    assert "4.00 it/s" in rendered
    assert "250 ms/step" in rendered
    assert str(PRODUCTION_STEPS) in rendered
    assert " h" in rendered


def test_the_l4_reference_point_reproduces_the_published_colab_figure() -> None:
    """1.7-1.9 it/s over 10,900 steps was reported as about 1.6-1.75 hours.

    If this drifts, either the arithmetic or the recorded history is wrong, and
    the reference the report prints alongside every new number is misleading.
    """

    for rate, expected_hours in ((1.7, 1.78), (1.9, 1.59)):
        probe = _probe(40, 140, 40 / rate, 140 / rate)
        assert probe.production_hours() == pytest.approx(expected_hours, abs=0.02)


# --------------------------------------------------------------------------
# the step-pair guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("short", "long"), [(40, 40), (140, 40), (10, 0), (0, 0)])
def test_a_degenerate_step_pair_is_refused(short, long) -> None:
    """Equal counts divide by zero; inverted ones give a NEGATIVE rate, and so
    negative hours, reported with a straight face.

    Asserted against the guard directly, NOT through main(). A version of this
    test that called main() started real training the moment a mutation removed
    the guard.
    """

    with pytest.raises(SystemExit, match="must exceed"):
        validate_step_pair(short, long)


def test_a_valid_pair_passes_silently() -> None:
    assert validate_step_pair(40, 140) is None


def test_probe_checks_gpu_safety_before_both_measured_runs(
    monkeypatch, tmp_path
) -> None:
    observed: list[tuple[str, str, int | None]] = []

    class RecordingPolicy:
        def check(self, *, stage: str, arm: str, step: int | None = None):
            observed.append((stage, arm, step))

    def fake_timed_run(config, arm_name, steps, *, seed, val_images, callbacks):
        assert len(callbacks) == 1
        return 10.0 if steps == 40 else 30.0

    monkeypatch.setattr(speed_probe, "timed_run", fake_timed_run)

    measured = speed_probe.probe(
        "rfdetr",
        tmp_path / "unused.yaml",
        "real_only",
        short=40,
        long=140,
        seed=1337,
        val_images=16,
        config={"schedule": {"warmup_steps": 2_000}, "run": {}},
        safety_policy=RecordingPolicy(),
    )

    assert measured.short_seconds == 10.0
    assert measured.long_seconds == 30.0
    assert observed == [
        ("before_probe", "real_only", 40),
        ("before_probe", "real_only", 140),
    ]
