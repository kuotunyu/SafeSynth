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
CLAUDE.md forbids.

Everything is read from results/detection_metrics.csv and the arms' raw
trainer_state.json (EVAL-12). Nothing is typed in by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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


def draw(values: dict, curves: dict, destination: Path) -> tuple[Path, str]:
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))

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
    for position, arm in zip(positions, order, strict=True):
        # The colour swatch is a cross-reference to the right-hand panel, so it
        # is drawn ONLY for the arms that appear there. A swatch under an arm
        # with no line to point at is a legend entry for nothing.
        if arm in curves:
            left.plot(
                [position], [-0.012], marker="s", markersize=9, color=ARM_COLOUR[arm]
            )
        left.text(
            position - width / 2,
            values[arm]["primary_map_small"] + 0.008,
            f"{values[arm]['primary_map_small']:.3f}",
            ha="center",
            fontsize=8,
        )
    left.set_xticks(list(positions))
    left.set_xticklabels(order, fontsize=8.5)
    left.set_ylabel("score")
    left.set_ylim(-0.03, max(values[arm]["primary_map"] for arm in order) * 1.18)
    left.set_title("Frozen Test, helmet + head — synthetic lost")
    left.legend(fontsize=9, loc="upper right")
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

    destination, verdict = draw(values, curves, args.figure)
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
