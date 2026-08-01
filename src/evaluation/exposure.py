"""Compare the arms on real-image exposure rather than on optimizer steps.

TRAIN-07 equalises optimizer steps, which controls compute and answers the
sharpest objection ("it only won because it trained longer"). It does not
equalise how often each arm sees a real image: a batch drawn from a 50/50
real-plus-synthetic pool carries half as many real images as a batch drawn from
real data alone, so at the same step count the synthetic arms have seen each
real photograph half as many times.

Both axes are legitimate and they answer different questions:

    matched steps            "for a fixed compute budget, which arm wins?"
    matched real exposure    "for a fixed ANNOTATION budget, which arm wins?"

The second is the one the whole project is about. Labelling is the expensive
resource; synthetic composites cost compute and no annotation. So a comparison
that only ever fixes steps cannot see the effect the method is claiming.

The catch, which every caller must state: matching real exposure UNMATCHES
compute. At one pass over the real training set a 50/50 arm has taken twice as
many optimizer steps as a real-only arm. "Same labels, more compute" is the
honest description, not "same conditions".
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ExposureError(RuntimeError):
    """Raised when a curve cannot be placed on the exposure axis."""


@dataclass(frozen=True)
class ExposurePoint:
    step: int
    real_epochs: float
    value: float


@dataclass(frozen=True)
class ExposureCurve:
    """One arm's metric re-indexed from optimizer steps to real-image passes."""

    arm: str
    real_fraction: float
    points: tuple[ExposurePoint, ...]

    def value_at(self, real_epochs: float) -> float | None:
        """Linear interpolation; None outside the measured range.

        Refusing to extrapolate matters: the arms stop at different exposures
        (a 50/50 arm reaches half as many passes for the same step budget), and
        an extrapolated tail would invent a comparison that was never measured.
        """

        xs = [point.real_epochs for point in self.points]
        ys = [point.value for point in self.points]
        if not xs or real_epochs < xs[0] or real_epochs > xs[-1]:
            return None
        index = bisect.bisect_left(xs, real_epochs)
        if index == 0:
            return ys[0]
        x0, x1 = xs[index - 1], xs[index]
        y0, y1 = ys[index - 1], ys[index]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (real_epochs - x0) / (x1 - x0)


# spec: TRAIN-07
def real_fraction(n_real_train: int, n_synthetic: int) -> float:
    """Share of each batch that is real, assuming uniform sampling over the pool."""

    total = n_real_train + n_synthetic
    if n_real_train <= 0 or total <= 0:
        raise ExposureError(
            f"n_real_train must be positive and the pool non-empty; got "
            f"{n_real_train} real and {n_synthetic} synthetic"
        )
    return n_real_train / total


# spec: TRAIN-07
def real_epochs(step: int, *, batch_size: int, n_real_train: int, fraction: float) -> float:
    """How many times the real training set has been seen by `step`."""

    if batch_size <= 0:
        raise ExposureError("batch_size must be positive")
    return step * batch_size * fraction / n_real_train


def build_exposure_curve(
    arm: str,
    curve_points: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    batch_size: int,
    n_real_train: int,
    n_synthetic: int,
) -> ExposureCurve:
    """Re-index one arm's eval points onto the real-exposure axis."""

    fraction = real_fraction(n_real_train, n_synthetic)
    points = [
        ExposurePoint(
            step=int(entry["step"]),
            real_epochs=real_epochs(
                int(entry["step"]),
                batch_size=batch_size,
                n_real_train=n_real_train,
                fraction=fraction,
            ),
            value=float(entry[metric]),
        )
        for entry in curve_points
        if metric in entry and "step" in entry
    ]
    if not points:
        raise ExposureError(f"{arm}: no points carrying {metric!r}")
    points.sort(key=lambda point: point.real_epochs)
    return ExposureCurve(arm=arm, real_fraction=fraction, points=tuple(points))


@dataclass(frozen=True)
class Crossover:
    """Where a challenger stops leading the baseline on the exposure axis."""

    challenger: str
    baseline: str
    leads_below: float | None
    max_lead: float
    max_lead_at: float | None
    verdict: str


# spec: EVAL-18
def find_crossover(
    baseline: ExposureCurve,
    challenger: ExposureCurve,
    *,
    grid: Sequence[float],
) -> Crossover:
    """The exposure at which the challenger's lead turns into a deficit.

    Returns the LAST grid point at which the challenger still leads, so a curve
    that crosses back and forth is described by where it finally gives up rather
    than by its first dip. Both readings are defensible; this one is chosen
    because the interesting claim is "synthetic helps until you have about N
    passes of real data", and reporting the first momentary dip would understate
    the range over which the effect held.
    """

    comparable = [
        (point, baseline.value_at(point), challenger.value_at(point))
        for point in grid
    ]
    comparable = [
        (point, base, chal)
        for point, base, chal in comparable
        if base is not None and chal is not None
    ]
    if not comparable:
        raise ExposureError(
            f"{challenger.arm} and {baseline.arm} share no measured exposure range"
        )

    leading = [point for point, base, chal in comparable if chal > base]
    leads_below = max(leading) if leading else None
    best = max(comparable, key=lambda row: row[2] - row[1])
    max_lead = best[2] - best[1]

    if leads_below is None:
        verdict = "never leads on any measured exposure"
    elif leads_below >= comparable[-1][0]:
        verdict = "still leading at the largest shared exposure; no crossover observed"
    else:
        verdict = f"leads up to about {leads_below:g} passes over the real set, then falls behind"

    return Crossover(
        challenger=challenger.arm,
        baseline=baseline.arm,
        leads_below=leads_below,
        max_lead=max_lead,
        max_lead_at=best[0] if max_lead > 0 else None,
        verdict=verdict,
    )
