"""Exposure re-indexing: the arithmetic that decides whether synthetic helped.

Every expected value below is derived by hand in a comment. Importing the
project's own constants and asserting against them would make these tests agree
with the code by construction, which is the failure mode K-19 is about.
"""

from __future__ import annotations

import pytest

from scripts.exposure_analysis import headline_from_data, link_target, shared_grid
from src.evaluation.exposure import (
    ExposureError,
    build_exposure_curve,
    find_crossover,
    real_epochs,
    real_fraction,
)

BATCH = 16
# Chosen so the fixtures land on WHOLE real epochs: at batch 16 a real-only arm
# covers 3200 images in exactly 200 steps, and a half-real arm in exactly 400.
# With the project's actual 3500 the exposures are 0.9966, 1.9931, ... and a
# grid point of 2.0 falls outside the measured range - correct behaviour, but it
# obscures what these tests are checking.
N_REAL = 3200


def _points(pairs, metric="eval_map"):
    return [{"step": step, metric: value} for step, value in pairs]


def _curve(arm, pairs, *, n_synthetic=0, metric="eval_map"):
    return build_exposure_curve(
        arm,
        _points(pairs, metric),
        metric=metric,
        batch_size=BATCH,
        n_real_train=N_REAL,
        n_synthetic=n_synthetic,
    )


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------


def test_a_real_only_pool_is_entirely_real() -> None:
    assert real_fraction(3500, 0) == pytest.approx(1.0)


def test_an_equal_sized_synthetic_pool_halves_the_real_share() -> None:
    """3500 real against 3500 synthetic is 0.5, which is the whole confound."""

    assert real_fraction(3500, 3500) == pytest.approx(0.5)


def test_an_unequal_pool_is_not_assumed_to_be_a_half() -> None:
    """3500 / (3500 + 1750) = 2/3. A hardcoded 0.5 would pass the test above."""

    assert real_fraction(3500, 1750) == pytest.approx(2 / 3)


def test_real_epochs_counts_passes_over_the_real_set_only() -> None:
    # The project's real numbers: 218 steps x 16 = 3488 images, all real, over
    # the 3500-image real training set.
    assert real_epochs(218, batch_size=BATCH, n_real_train=3500, fraction=1.0) == (
        pytest.approx(3488 / 3500)
    )
    # The same step count on a half-real pool is half as many real images.
    assert real_epochs(218, batch_size=BATCH, n_real_train=3500, fraction=0.5) == (
        pytest.approx(1744 / 3500)
    )


def test_a_half_real_arm_needs_twice_the_steps_for_the_same_exposure() -> None:
    """This is the 'same labels, more compute' caveat, pinned as behaviour."""

    solo = real_epochs(1000, batch_size=BATCH, n_real_train=3500, fraction=1.0)
    mixed = real_epochs(2000, batch_size=BATCH, n_real_train=3500, fraction=0.5)

    assert solo == pytest.approx(mixed)


def test_a_zero_batch_size_is_refused() -> None:
    with pytest.raises(ExposureError, match="batch_size"):
        real_epochs(10, batch_size=0, n_real_train=N_REAL, fraction=1.0)


def test_a_pool_with_no_real_images_is_refused() -> None:
    with pytest.raises(ExposureError, match="must be positive"):
        real_fraction(0, 3500)


# --------------------------------------------------------------------------
# re-indexing
# --------------------------------------------------------------------------


def test_the_same_step_lands_at_different_exposures_for_different_pools() -> None:
    """If it did not, the whole re-indexing would be a no-op."""

    solo = _curve("real_only", [(2000, 0.30)])
    mixed = _curve("filtered_syn", [(2000, 0.30)], n_synthetic=N_REAL)

    assert solo.points[0].real_epochs == pytest.approx(2 * mixed.points[0].real_epochs)


def test_points_are_sorted_by_exposure_even_when_the_log_is_not() -> None:
    curve = _curve("real_only", [(600, 0.3), (200, 0.1), (400, 0.2)])

    assert [point.step for point in curve.points] == [200, 400, 600]


def test_a_metric_that_is_absent_from_every_entry_is_refused() -> None:
    with pytest.raises(ExposureError, match="eval_map_small"):
        build_exposure_curve(
            "real_only",
            [{"step": 100, "eval_map": 0.1}],
            metric="eval_map_small",
            batch_size=BATCH,
            n_real_train=N_REAL,
            n_synthetic=0,
        )


def test_interpolation_is_linear_between_measured_points() -> None:
    # Steps 200 and 400 sit at exactly 1 and 2 real epochs, so 1.5 must give the
    # mean of 0.20 and 0.40.
    curve = _curve("real_only", [(200, 0.20), (400, 0.40)])

    assert curve.value_at(1.5) == pytest.approx(0.30)


def test_no_extrapolation_beyond_the_measured_range() -> None:
    """A 50/50 arm stops at half the exposure; inventing a tail invents a result."""

    curve = _curve("filtered_syn", [(400, 0.2), (800, 0.4)], n_synthetic=N_REAL)

    assert curve.value_at(50.0) is None
    assert curve.value_at(0.0) is None


# --------------------------------------------------------------------------
# the crossover, which is the actual finding
# --------------------------------------------------------------------------


def _crossover_pair():
    """Challenger starts high, baseline overtakes it. Crossover between 3 and 4."""

    baseline = _curve(
        "real_only",
        [(200, 0.10), (400, 0.20), (600, 0.30), (800, 0.45), (1000, 0.55)],
    )
    challenger = _curve(
        "filtered_syn",
        [(400, 0.25), (800, 0.32), (1200, 0.35), (1600, 0.36), (2000, 0.37)],
        n_synthetic=N_REAL,
    )
    return baseline, challenger


def test_the_crossover_is_where_the_lead_ends_not_where_it_starts() -> None:
    baseline, challenger = _crossover_pair()

    # Both curves are measured over 1..5 real epochs by construction.
    result = find_crossover(baseline, challenger, grid=[1, 2, 3, 4, 5])

    # baseline 0.10 0.20 0.30 0.45 0.55  vs  challenger 0.25 0.32 0.35 0.36 0.37
    # challenger leads at 1, 2 and 3; loses at 4 and 5.
    assert result.leads_below == 3
    assert "then falls behind" in result.verdict


def test_the_largest_lead_is_reported_with_where_it_occurred() -> None:
    baseline, challenger = _crossover_pair()

    result = find_crossover(baseline, challenger, grid=[1, 2, 3, 4, 5])

    # Largest gap is at 1 real epoch: 0.25 - 0.10 = 0.15.
    assert result.max_lead == pytest.approx(0.15)
    assert result.max_lead_at == 1


def test_a_challenger_that_never_leads_says_so() -> None:
    baseline = _curve("real_only", [(200, 0.50), (400, 0.60)])
    challenger = _curve("filtered_syn", [(400, 0.10), (800, 0.20)], n_synthetic=N_REAL)

    result = find_crossover(baseline, challenger, grid=[1, 2])

    assert result.leads_below is None
    assert result.max_lead < 0
    assert result.max_lead_at is None
    assert "never leads" in result.verdict


def test_a_challenger_still_ahead_at_the_end_is_not_called_a_crossover() -> None:
    baseline = _curve("real_only", [(200, 0.10), (400, 0.20)])
    challenger = _curve("filtered_syn", [(400, 0.30), (800, 0.40)], n_synthetic=N_REAL)

    result = find_crossover(baseline, challenger, grid=[1, 2])

    assert result.leads_below == 2
    assert "no crossover observed" in result.verdict


def test_grid_points_outside_either_measured_range_are_dropped() -> None:
    """Comparing at an exposure only one arm reached would be a fabricated row."""

    baseline = _curve("real_only", [(200, 0.10), (2000, 0.60)])
    challenger = _curve("filtered_syn", [(400, 0.30), (800, 0.40)], n_synthetic=N_REAL)

    # The challenger only spans 1..2 real epochs; 5 and 9 are outside it.
    result = find_crossover(baseline, challenger, grid=[1, 2, 5, 9])

    assert result.leads_below == 2


def test_disjoint_ranges_raise_rather_than_returning_an_empty_verdict() -> None:
    baseline = _curve("real_only", [(2000, 0.6), (4000, 0.7)])
    challenger = _curve("filtered_syn", [(400, 0.3)], n_synthetic=N_REAL)

    with pytest.raises(ExposureError, match="share no measured exposure"):
        find_crossover(baseline, challenger, grid=[1, 2])


# --------------------------------------------------------------------------
# the reporting layer: the grid the table is built on, and the headline
#
# The headline is generated from the numbers rather than written by hand, so a
# result that reverses must reverse the caption too. That property is only real
# if it is tested in BOTH directions.
# --------------------------------------------------------------------------


def test_the_shared_grid_stops_at_the_shortest_arm() -> None:
    """A 50/50 arm reaches half the exposure; rows past that would be half empty."""

    curves = {
        "real_only": _curve("real_only", [(200 * k, 0.1 * k) for k in range(1, 11)]),
        "filtered_syn": _curve(
            "filtered_syn",
            [(400 * k, 0.1 * k) for k in range(1, 6)],
            n_synthetic=N_REAL,
        ),
    }

    # real_only spans 1..10 passes, filtered_syn 1..5. The shared range is 1..5.
    assert shared_grid(curves) == [1, 2, 3, 4, 5]


def test_the_shared_grid_starts_after_the_latest_first_point() -> None:
    """An arm whose first eval is at 3 passes cannot be quoted at 1 or 2."""

    curves = {
        "real_only": _curve("real_only", [(200 * k, 0.1) for k in range(1, 7)]),
        "filtered_syn": _curve(
            "filtered_syn", [(1200, 0.2), (1600, 0.3), (2000, 0.4)], n_synthetic=N_REAL
        ),
    }

    assert shared_grid(curves) == [3, 4, 5]


def test_the_headline_names_the_leader_and_its_crossover() -> None:
    baseline, challenger = _crossover_pair()

    headline = headline_from_data(
        {"real_only": baseline, "filtered_syn": challenger}, [1, 2, 3, 4, 5]
    )

    assert "filtered_syn" in headline
    assert "+0.150" in headline  # 0.25 - 0.10 at one pass
    assert "4 passes" not in headline  # it stops leading after 3, not 4
    assert "3 passes" in headline


def test_the_headline_reverses_when_nothing_beats_the_baseline() -> None:
    """The caption must not survive a result that contradicts it."""

    baseline = _curve("real_only", [(200, 0.50), (400, 0.60)])
    loser = _curve("filtered_syn", [(400, 0.10), (800, 0.20)], n_synthetic=N_REAL)

    headline = headline_from_data({"real_only": baseline, "filtered_syn": loser}, [1, 2])

    assert "no arm beats" in headline
    assert "leads by" not in headline


def test_the_headline_picks_the_largest_lead_not_the_first_arm() -> None:
    baseline = _curve("real_only", [(200, 0.10), (400, 0.20)])
    weak = _curve("standard_aug", [(200, 0.12), (400, 0.21)])
    strong = _curve("filtered_syn", [(400, 0.40), (800, 0.45)], n_synthetic=N_REAL)

    headline = headline_from_data(
        {"real_only": baseline, "standard_aug": weak, "filtered_syn": strong}, [1, 2]
    )

    assert "filtered_syn" in headline
    assert "standard_aug" not in headline


def test_a_figure_outside_the_repo_still_produces_a_usable_link() -> None:
    """--figure takes any path; relative_to would raise instead of falling back."""

    from pathlib import Path

    assert link_target(Path("D:/elsewhere/fig.png")) == "D:/elsewhere/fig.png"
