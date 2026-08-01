"""Re-plot the four arms against real-image exposure instead of optimizer steps.

Motivation, in one line: TRAIN-07 fixes compute, but the resource this project
claims to save is ANNOTATION, and those are different axes. See
src/evaluation/exposure.py for the full argument and the caveat that matching
real exposure necessarily unmatches compute.

Everything is re-aggregated from each arm's `trainer_state.json` and
`run_record.json` (EVAL-12). Validation only — the Test numbers belong to the
main table, and validation is where model selection legitimately happened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")  # No display on this machine; must precede pyplot.
import matplotlib.pyplot as plt

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.exposure import ExposureError, build_exposure_curve, find_crossover
from src.training.arms import ARMS
from src.training.ingest import latest_checkpoint, training_curve

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "exposure_curves.png"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "exposure_analysis.md"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"
BASELINE_ARM = "real_only"
METRICS = (("eval_map", "mAP 50-95"), ("eval_map_small", "AP_small"))

ARM_STYLE = {
    "real_only": ("#0173b2", "-"),
    "standard_aug": ("#de8f05", "--"),
    "unfiltered_syn": ("#029e73", "-."),
    "filtered_syn": ("#cc78bc", ":"),
}


def read_arm(runs_root: Path, arm: str, seed: int) -> tuple[list[dict], dict]:
    """Eval points and training-set composition for one arm."""

    seed_dir = Path(runs_root) / arm / f"seed_{seed}"
    checkpoint = latest_checkpoint(seed_dir)
    if checkpoint is None:
        raise ExposureError(f"{arm}: no checkpoint under {seed_dir}")
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    record = json.loads((seed_dir / "run_record.json").read_text(encoding="utf-8"))
    return training_curve(state), record["composition"]


def build_curves(runs_root: Path, seed: int, batch_size: int, metric: str) -> dict:
    curves = {}
    for arm in ARMS:
        points, composition = read_arm(runs_root, arm, seed)
        curves[arm] = build_exposure_curve(
            arm,
            points,
            metric=metric,
            batch_size=batch_size,
            n_real_train=int(composition["n_real_train"]),
            n_synthetic=int(composition["n_synthetic"]),
        )
    return curves


def shared_grid(curves: dict) -> list[float]:
    """Integer real-epochs covered by every arm, so no row is half-empty."""

    lowest = max(curve.points[0].real_epochs for curve in curves.values())
    highest = min(curve.points[-1].real_epochs for curve in curves.values())
    return [value for value in range(1, int(highest) + 1) if value >= lowest]


def plot(curves_by_metric: dict, destination: Path, *, headline: str) -> Path:
    figure, axes = plt.subplots(1, len(METRICS), figsize=(12.5, 4.6))
    for axis, (metric, title) in zip(axes, METRICS, strict=True):
        for arm, curve in curves_by_metric[metric].items():
            colour, style = ARM_STYLE.get(arm, ("#444444", "-"))
            axis.plot(
                [point.real_epochs for point in curve.points],
                [point.value for point in curve.points],
                color=colour,
                linestyle=style,
                linewidth=1.7,
                marker="o",
                markersize=2.5,
                label=arm,
            )
        # Log x on purpose. The effect being reported lives between one and five
        # passes over the real set; on a linear axis running to 50 that is the
        # leftmost tenth of the panel, and the figure fails to show the thing its
        # own caption claims. Ticks are labelled as plain integers so the axis
        # still reads as a count of passes rather than as a decade scale.
        axis.set_xscale("log")
        axis.set_xticks([1, 2, 3, 5, 10, 20, 50])
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_title(f"Validation {title}")
        axis.set_xlabel("passes over the REAL training set (log scale)")
        axis.grid(alpha=0.25, linewidth=0.6, which="both")
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    figure.suptitle(headline, fontsize=11)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return destination


def headline_from_data(curves: dict, grid: list[float]) -> str:
    """State what the numbers show, rather than what the analysis hoped to show."""

    baseline = curves[BASELINE_ARM]
    crossovers = [
        find_crossover(baseline, curve, grid=grid)
        for arm, curve in curves.items()
        if arm != BASELINE_ARM
    ]
    best = max(crossovers, key=lambda crossover: crossover.max_lead)
    if best.max_lead <= 0:
        return (
            f"Matched on annotation budget: no arm beats `{BASELINE_ARM}` at any "
            f"measured exposure"
        )
    if best.leads_below is None:
        return "Matched on annotation budget: leads appear but never at a grid point"
    return (
        f"Matched on annotation budget, not on compute — {best.challenger} leads by "
        f"{best.max_lead:+.3f} mAP up to about {best.leads_below:g} passes over the "
        f"real set"
    )


def link_target(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def render(curves_by_metric: dict, grid: list[float], figure_path: Path) -> str:
    lines = [
        "# The arms compared at equal annotation budget",
        "",
        (
            "[TRAIN-07](../docs/training_spec.md) equalises optimizer STEPS, which "
            "controls compute. It does not equalise how often each arm sees a real "
            "photograph: a batch drawn from a 50/50 pool carries half as many real "
            "images, so at the same step the synthetic arms have seen each real image "
            "half as many times. Annotation is the resource this project claims to "
            "save, so it is the axis the method has to be judged on."
        ),
        "",
        (
            "⚠️ **Matching real exposure unmatches compute.** At one pass over the real "
            "training set a 50/50 arm has taken twice as many optimizer steps as a "
            "real-only arm. The honest description of every row below is *same labels, "
            "more compute* — which is exactly the trade synthetic data offers, but it "
            "is not *same conditions*."
        ),
        "",
        f"![exposure curves]({link_target(figure_path)})",
        "",
        (
            "Re-aggregated from each arm's `trainer_state.json` (EVAL-12). "
            "**Validation only**, single seed."
        ),
        "",
    ]

    for metric, title in METRICS:
        curves = curves_by_metric[metric]
        lines += [
            f"## {title} at matched real-image exposure",
            "",
            "| passes over real set | " + " | ".join(f"`{arm}`" for arm in ARMS) + " |",
            "|---:|" + "---:|" * len(ARMS),
        ]
        for exposure in grid:
            cells = []
            for arm in ARMS:
                value = curves[arm].value_at(exposure)
                cells.append("—" if value is None else f"{value:.4f}")
            lines.append(f"| {exposure:g} | " + " | ".join(cells) + " |")
        lines.append("")

        baseline = curves[BASELINE_ARM]
        lines += ["| challenger | largest lead | at | verdict |", "|---|---:|---:|---|"]
        for arm in ARMS:
            if arm == BASELINE_ARM:
                continue
            crossover = find_crossover(baseline, curves[arm], grid=grid)
            at = "—" if crossover.max_lead_at is None else f"{crossover.max_lead_at:g}"
            lines.append(
                f"| `{arm}` | {crossover.max_lead:+.4f} | {at} | {crossover.verdict} |"
            )
        lines.append("")

    lines += [
        "## What this does and does not establish",
        "",
        (
            "It does not overturn the main table. At the full step budget, and at every "
            "arm's own best checkpoint, `real_only` is ahead — that result stands and is "
            "reported as it is."
        ),
        "",
        (
            "What it adds is where the synthetic data was doing something. If a "
            "challenger's lead is large at one or two passes over the real set and gone "
            "by four or five, then the composites help while real labels are scarce and "
            "stop helping once they are not. This dataset supplies 5,000 labelled "
            "images, which is the regime where synthetic augmentation has least to "
            "offer."
        ),
        "",
        (
            "**Single seed.** [EVAL-10](../docs/evaluation_spec.md) forbids claiming a "
            "win from a small single-seed gap, and early training is the noisiest part "
            "of the curve. Treat the crossover as a direction worth testing, not as a "
            "measured constant."
        ),
        "",
        (
            "**The experiment this points at** is a real-data-fraction ablation: retrain "
            "on 10%, 25% and 50% of the real training set with and without the same "
            "synthetic pool. If the reading above is right, the gap should widen as the "
            "real fraction shrinks. That is a cheaper experiment than the one already "
            "run, because every arm in it trains on less data."
        ),
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = args.runs_root or load_project_paths().runs
    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    batch_size = int(config["run"]["per_device_train_batch_size"])

    try:
        curves_by_metric = {
            metric: build_curves(runs_root, args.seed, batch_size, metric)
            for metric, _ in METRICS
        }
    except ExposureError as error:
        print(f"cannot build exposure curves: {error}")
        return 2

    grid = shared_grid(curves_by_metric["eval_map"])
    if not grid:
        print("the arms share no whole-epoch exposure range")
        return 2

    headline = headline_from_data(curves_by_metric["eval_map"], grid)
    plot(curves_by_metric, args.figure, headline=headline)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        render(curves_by_metric, grid, args.figure), encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.figure}")
    print(f"wrote {args.summary}")
    print(f"shared exposure grid: {grid[0]:g}..{grid[-1]:g} passes over the real set")
    baseline = curves_by_metric["eval_map"][BASELINE_ARM]
    for arm in ARMS:
        if arm == BASELINE_ARM:
            continue
        crossover = find_crossover(baseline, curves_by_metric["eval_map"][arm], grid=grid)
        print(f"  {arm:15s} lead {crossover.max_lead:+.4f} -> {crossover.verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
