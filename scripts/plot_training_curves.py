"""Re-aggregate the four arms' validation curves from the raw trainer state.

EVAL-12 forbids copying numbers off the notebook display, so this reads
`trainer_state.json` out of each arm's last checkpoint and plots `log_history`
directly.

The figure exists because a single best-checkpoint number hides the shape of
what happened: every arm peaked early and then decayed for the rest of its 50
(or 25) epochs. Reporting only the peak would be honest about the value and
silent about the fact that the schedule was far too long, which is the more
useful thing for anyone reading the result.

The x axis is optimizer STEPS, not epochs, and that is not cosmetic. TRAIN-07
holds the step budget equal across arms, so the synthetic arms - carrying twice
the data - cover half as many epochs in the same 10,900 steps. Plotting against
epochs would put four curves of different lengths on one axis and invite the
reader to compare them at points where the arms have had different amounts of
compute.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # No display on this machine; must be set before pyplot.
import matplotlib.pyplot as plt

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.training.arms import ARMS
from src.training.ingest import latest_checkpoint, training_curve

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "training_curves.png"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "training_curves.md"

# Colour-blind-safe; distinguishable in greyscale by the linestyle pairing below.
ARM_STYLE = {
    "real_only": ("#0173b2", "-"),
    "standard_aug": ("#de8f05", "--"),
    "unfiltered_syn": ("#029e73", "-."),
    "filtered_syn": ("#cc78bc", ":"),
}
PANELS = (("eval_map", "mAP 50-95"), ("eval_map_small", "AP_small"))


class CurveDataError(RuntimeError):
    """Raised when an arm's raw training state cannot be read."""


@dataclass(frozen=True)
class ArmCurve:
    arm: str
    steps: list[int]
    series: dict[str, list[float]]
    best_step: int
    best_map: float
    final_map: float

    @property
    def decay_points(self) -> float:
        """How far the arm fell from its own peak by the end of training."""

        return self.best_map - self.final_map


def read_arm_curve(runs_root: Path, arm: str, seed: int) -> ArmCurve:
    seed_dir = Path(runs_root) / arm / f"seed_{seed}"
    checkpoint = latest_checkpoint(seed_dir)
    if checkpoint is None:
        raise CurveDataError(f"{arm}: no checkpoint-* under {seed_dir}")
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise CurveDataError(f"{arm}: no trainer_state.json in {checkpoint.name}")

    state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    points = [point for point in training_curve(state) if "eval_map" in point]
    if not points:
        raise CurveDataError(f"{arm}: trainer_state.json carries no eval entries")

    steps = [int(point["step"]) for point in points]
    series = {
        metric: [float(point.get(metric, float("nan"))) for point in points]
        for metric, _ in PANELS
    }
    best = max(points, key=lambda point: point["eval_map"])
    return ArmCurve(
        arm=arm,
        steps=steps,
        series=series,
        best_step=int(best["step"]),
        best_map=float(best["eval_map"]),
        final_map=float(points[-1]["eval_map"]),
    )


def plot_curves(curves: list[ArmCurve], destination: Path) -> Path:
    figure, axes = plt.subplots(1, len(PANELS), figsize=(12, 4.5), sharex=True)
    for axis, (metric, title) in zip(axes, PANELS, strict=True):
        for curve in curves:
            colour, style = ARM_STYLE.get(curve.arm, ("#444444", "-"))
            axis.plot(
                curve.steps,
                curve.series[metric],
                color=colour,
                linestyle=style,
                linewidth=1.6,
                label=curve.arm,
            )
            if metric == "eval_map":
                axis.plot(
                    [curve.best_step],
                    [curve.best_map],
                    marker="o",
                    color=colour,
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    zorder=5,
                )
        axis.set_title(f"Validation {title}")
        axis.set_xlabel("optimizer steps (equal budget, TRAIN-07)")
        axis.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    figure.suptitle(
        "Every arm peaks early and then decays — each is reported at its own best checkpoint",
        fontsize=11,
    )
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return destination


def _markdown_link_target(figure_path: Path) -> str:
    """Repo-relative POSIX path when possible, otherwise the path as given.

    `--figure` accepts an arbitrary destination, so `relative_to` cannot be
    assumed to succeed; it raises rather than falling back, which would turn a
    perfectly reasonable invocation into a crash after the figure had already
    been written.
    """

    try:
        return figure_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return figure_path.as_posix()


def render_summary(curves: list[ArmCurve], figure_path: Path) -> str:
    lines = [
        "# Validation curves, re-aggregated from trainer_state.json",
        "",
        (
            "Read from each arm's last checkpoint (EVAL-12: never copied from the "
            "notebook display). Validation only — the Test numbers live in the main table."
        ),
        "",
        f"![training curves]({_markdown_link_target(figure_path)})",
        "",
        "| arm | eval points | best mAP | at step | final mAP | decay from peak |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for curve in sorted(curves, key=lambda c: -c.best_map):
        lines.append(
            f"| `{curve.arm}` | {len(curve.steps)} | {curve.best_map:.4f} | "
            f"{curve.best_step} | {curve.final_map:.4f} | −{curve.decay_points:.4f} |"
        )
    worst = max(curves, key=lambda c: c.decay_points)
    lines += [
        "",
        (
            f"Every arm decays after its peak; the largest fall is `{worst.arm}` at "
            f"−{worst.decay_points:.4f} mAP from step {worst.best_step} to the end. "
            "The schedule was far longer than this dataset needs. Because "
            "`load_best_model_at_end` selects on validation mAP, each arm is reported "
            "at its own best checkpoint, so the comparison is between four "
            "early-stopped models rather than four over-trained ones."
        ),
        "",
        (
            "The arms carrying synthetic data record half as many eval points for the "
            "same step budget: TRAIN-07 fixes optimizer steps, so twice the data means "
            "half the epochs, and each real image is seen about half as often. That "
            "confound belongs beside any comparison of these curves."
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

    curves, failures = [], []
    for arm in ARMS:
        try:
            curves.append(read_arm_curve(runs_root, arm, args.seed))
        except CurveDataError as error:
            failures.append(str(error))

    for failure in failures:
        print("MISSING:", failure)
    if not curves:
        print(f"no readable curves under {runs_root}")
        return 2

    plot_curves(curves, args.figure)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        render_summary(curves, args.figure), encoding="utf-8", newline="\n"
    )
    print(f"wrote {args.figure}")
    print(f"wrote {args.summary}")
    for curve in curves:
        print(
            f"  {curve.arm:15s} best={curve.best_map:.4f}@{curve.best_step:<6d} "
            f"final={curve.final_map:.4f}  decay=-{curve.decay_points:.4f}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
