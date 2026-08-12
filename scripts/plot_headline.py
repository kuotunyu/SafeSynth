"""One figure for the README: what the four arms did, and where synthetic helped.

Two panels, because the project's result needs both to be honest.

LEFT is the outcome on the frozen Test split: synthetic lost. Read on its own it
says the method does not work.

RIGHT is the same experiment indexed on passes over the REAL training set rather
than on optimizer steps. Annotation is the resource this method claims to save,
and on that axis the filtered arm leads until roughly the fourth pass. Read on
its own it says the method works.

Neither panel is the answer; the pair is. Showing only the left would hide where
the effect lives, and showing only the right would be the selective reporting
the reporting contract forbids.

Everything is read from results/detection_metrics.csv and the arms' raw
trainer_state.json (EVAL-12). Nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import pairwise
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")  # No display; must precede pyplot.
import matplotlib.pyplot as plt

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.exposure import build_exposure_curve, find_crossover
from src.training.arms import ARMS
from src.training.ingest import latest_checkpoint, training_curve

METRICS_CSV = PROJECT_ROOT / "results" / "detection_metrics.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "headline.png"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"
BASELINE_ARM = "real_only"
CHALLENGER_ARM = "filtered_syn"
LEFT_METRICS = (("primary_map_small", "AP_small"), ("primary_map", "mAP 50-95"))

ARM_COLOUR = {
    "real_only": "#0173b2",
    "standard_aug": "#de8f05",
    "unfiltered_syn": "#029e73",
    "filtered_syn": "#cc78bc",
}


class HeadlineError(RuntimeError):
    """Raised when the figure's inputs are missing."""


def read_test_metrics(path: Path) -> dict[str, dict[str, float]]:
    """arm -> metric -> value, restricted to the whole-Test rows.

    The CSV also carries per-slice rows under `test/small_object` and friends;
    taking them by metric name alone would silently plot a slice.
    """

    if not path.is_file():
        raise HeadlineError(f"{path} not found - run scripts/eval.py first")
    values: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test" or not row["value"]:
                continue
            values.setdefault(row["arm"], {})[row["metric"]] = float(row["value"])
    missing = [arm for arm in ARMS if arm not in values]
    if missing:
        raise HeadlineError(f"no test rows for {missing}")
    return values


# spec: EVAL-09
def read_test_intervals(path: Path, metric: str) -> dict[str, tuple[float, float]]:
    """arm -> (ci_low, ci_high) for whole-Test rows that carry an interval.

    Returns only the arms that have one. A bar chart sorted by value asserts a
    ranking to anyone who looks at it, and before EVAL-09 ran there was nothing
    on this figure to say which steps of that ranking the data supports. Two of
    them turn out not to be supported at all.
    """

    intervals: dict[str, tuple[float, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test" or row["metric"] != metric:
                continue
            if not row.get("ci_low") or not row.get("ci_high"):
                continue
            intervals[row["arm"]] = (float(row["ci_low"]), float(row["ci_high"]))
    return intervals


# spec: EVAL-09
def separable(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Do two intervals fail to overlap? The only ranking claim this figure may make."""

    return a[1] < b[0] or b[1] < a[0]


def read_exposure_curves(runs_root: Path, seed: int, batch_size: int) -> dict:
    curves = {}
    for arm in (BASELINE_ARM, CHALLENGER_ARM):
        seed_dir = Path(runs_root) / arm / f"seed_{seed}"
        checkpoint = latest_checkpoint(seed_dir)
        if checkpoint is None:
            raise HeadlineError(f"{arm}: no checkpoint under {seed_dir}")
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        composition = json.loads(
            (seed_dir / "run_record.json").read_text(encoding="utf-8")
        )["composition"]
        curves[arm] = build_exposure_curve(
            arm,
            training_curve(state),
            metric="eval_map",
            batch_size=batch_size,
            n_real_train=int(composition["n_real_train"]),
            n_synthetic=int(composition["n_synthetic"]),
        )
    return curves


def draw(
    values: dict, curves: dict, destination: Path, intervals: dict | None = None
) -> tuple[Path, str]:
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

    intervals = {} if intervals is None else intervals
    order = sorted(ARMS, key=lambda arm: -values[arm]["primary_map_small"])
    width = 0.38
    positions = range(len(order))
    for offset, (metric, label) in zip((-width / 2, width / 2), LEFT_METRICS, strict=True):
        left.bar(
            [position + offset for position in positions],
            [values[arm][metric] for arm in order],
            width=width,
            label=label,
            color=["#4a4a4a" if offset < 0 else "#a8a8a8"] * len(order),
            edgecolor="white",
        )
        # EVAL-09 whiskers on AP_small only. Sorting the bars by height asserts
        # a four-way ranking; without these a reader has no way to see that two
        # of its three steps are inside the noise.
        if metric == "primary_map_small" and intervals:
            drawn = [(i, arm) for i, arm in enumerate(order) if arm in intervals]
            left.errorbar(
                [i + offset for i, _ in drawn],
                [values[arm][metric] for _, arm in drawn],
                yerr=[
                    [values[arm][metric] - intervals[arm][0] for _, arm in drawn],
                    [intervals[arm][1] - values[arm][metric] for _, arm in drawn],
                ],
                fmt="none",
                ecolor="#d94801",
                elinewidth=1.6,
                capsize=4,
                capthick=1.6,
                zorder=5,
            )
    for position, arm in zip(positions, order, strict=True):
        # The colour swatch is a cross-reference to the right-hand panel, so it
        # is drawn ONLY for the arms that appear there. A swatch under an arm
        # with no line to point at is a legend entry for nothing.
        if arm in curves:
            left.plot(
                [position], [-0.012], marker="s", markersize=9, color=ARM_COLOUR[arm]
            )
        # Clear of the whisker, not on top of it: the label used to sit at
        # value + 0.008, which is inside the interval now that one is drawn.
        top = values[arm]["primary_map_small"]
        if arm in intervals:
            top = max(top, intervals[arm][1])
        left.text(
            position - width / 2,
            top + 0.012,
            f"{values[arm]['primary_map_small']:.3f}",
            ha="center",
            fontsize=8,
        )
    left.set_xticks(list(positions))
    # 8.5 pt ran "standard_aug" into "unfiltered_syn" at this figure width.
    left.set_xticklabels(order, fontsize=7.6, rotation=12, ha="right")
    left.set_ylabel("score")
    left.set_ylim(-0.03, max(values[arm]["primary_map"] for arm in order) * 1.18)
    left.set_title("Frozen Test, helmet + head — synthetic lost")
    left.legend(fontsize=9, loc="upper right")
    if intervals:
        # Which steps of the sorted order the intervals actually support. Named
        # in the figure rather than left to the caption, because the bars are
        # sorted and a reader takes that as the claim.
        pairs = [
            (a, b)
            for a, b in pairwise(order)
            if a in intervals and b in intervals
        ]
        unsupported = [f"{a} vs {b}" for a, b in pairs if not separable(intervals[a], intervals[b])]
        note = (
            "bars are sorted, but not every step is a result. "
            + (
                "overlapping 95% CI: " + "; ".join(unsupported)
                if unsupported
                else "every adjacent pair separates at 95%"
            )
        )
        left.text(
            0.0,
            -0.185,
            note,
            transform=left.transAxes,
            fontsize=7.6,
            color="#d94801",
            va="top",
        )
    left.grid(axis="y", alpha=0.25, linewidth=0.6)

    grid = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 24]
    for arm, curve in curves.items():
        right.plot(
            [point.real_epochs for point in curve.points],
            [point.value for point in curve.points],
            color=ARM_COLOUR[arm],
            linestyle="-" if arm == BASELINE_ARM else ":",
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            label=arm,
        )
    crossover = find_crossover(curves[BASELINE_ARM], curves[CHALLENGER_ARM], grid=grid)
    if crossover.leads_below is not None:
        right.axvspan(
            curves[CHALLENGER_ARM].points[0].real_epochs,
            crossover.leads_below,
            color="#cc78bc",
            alpha=0.12,
        )
        right.text(
            curves[CHALLENGER_ARM].points[0].real_epochs * 1.05,
            0.02,
            f"synthetic ahead\nby up to {crossover.max_lead:+.3f}",
            fontsize=8.5,
            color="#8a3f7c",
            va="bottom",
        )
    right.set_xscale("log")
    right.set_xticks([1, 2, 3, 5, 10, 20, 50])
    right.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    right.set_xlabel("passes over the REAL training set (log scale)")
    right.set_ylabel("validation mAP 50-95")
    right.set_title("Same runs, indexed on annotation budget — synthetic led early")
    right.legend(fontsize=9, loc="lower right")
    right.grid(alpha=0.25, linewidth=0.6, which="both")

    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return destination, crossover.verdict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = args.runs_root or load_project_paths().runs
    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    batch_size = int(config["run"]["per_device_train_batch_size"])

    try:
        values = read_test_metrics(args.metrics_csv)
        curves = read_exposure_curves(runs_root, args.seed, batch_size)
    except HeadlineError as error:
        print(f"cannot draw: {error}")
        return 2

    intervals = read_test_intervals(args.metrics_csv, "primary_map_small")
    destination, verdict = draw(values, curves, args.figure, intervals)
    print(f"wrote {destination}")
    print(f"  crossover: {verdict}")
    for arm in sorted(ARMS, key=lambda a: -values[a]["primary_map_small"]):
        print(
            f"  {arm:15s} AP_small {values[arm]['primary_map_small']:.4f}  "
            f"mAP {values[arm]['primary_map']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
