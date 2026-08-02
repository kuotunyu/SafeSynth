"""Measure this machine's training speed, so a schedule can be stated instead of guessed.

CLAUDE.md forbids reporting hours derived from "roughly N times slower than X".
That rule exists because it was broken once: the Colab estimate was extrapolated
at 2-2.5x and came in at 3x, which turned a stated 4-5 hours into an overnight
job nobody was warned about. Every number this script prints is measured here.

METHOD: TWO RUNS AND A SLOPE. A single timed run cannot separate the per-step
cost from the fixed cost - model download, dataset construction, the final
evaluate - and the fixed part is large enough to dominate a short probe. So the
same arm is run at two step counts and the rate comes from the difference:

    seconds_per_step = (t_long - t_short) / (steps_long - steps_short)

Whatever the fixed cost is, it appears in both terms and cancels. The intercept
is reported too, because a fixed cost far from zero means something else is
wrong with the measurement and the reader should see it rather than trust the
slope blindly.

Nothing here is a benchmark of the architectures against each other. It answers
one question: how long would the real run take on this machine.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.training.arms import build_all_arms
from src.training.config import load_training_config
from src.training.health import TrainerHealthCallback, UnattendedSafetyPolicy
from src.training.run import RunPaths, run_arm

# The budget the real four-arm run used, so the extrapolation lands on a number
# that means something rather than on an arbitrary horizon.
PRODUCTION_STEPS = 10_900

# The 1x pool the four production arms trained on. Named here for the same
# reason scripts/smoke_train.py names it: `paths.synthetic` is a parent holding
# many pools, several of them stale, and reaching straight into it silently
# resolves to nothing.
POOL_TAG = "m13_pool_1x"


class SpeedProbeError(RuntimeError):
    """The measured probe cannot support a production-time projection."""


@dataclass(frozen=True)
class SpeedProbe:
    """One (config, arm) measured at two step counts."""

    label: str
    short_steps: int
    long_steps: int
    short_seconds: float
    long_seconds: float

    @property
    def seconds_per_step(self) -> float:
        return (self.long_seconds - self.short_seconds) / (self.long_steps - self.short_steps)

    @property
    def steps_per_second(self) -> float:
        return 1.0 / self.seconds_per_step

    @property
    def fixed_overhead_seconds(self) -> float:
        """The intercept. Large is fine; NEGATIVE means the probe is unreliable."""

        return self.short_seconds - self.seconds_per_step * self.short_steps

    def production_hours(self, steps: int = PRODUCTION_STEPS) -> float:
        return (self.seconds_per_step * steps + self.fixed_overhead_seconds) / 3600.0

    def render(self) -> str:
        warning = ""
        if self.fixed_overhead_seconds < 0:
            warning = "   <-- NEGATIVE intercept: treat the slope as unreliable"
        return (
            f"{self.label:<28} {self.steps_per_second:>6.2f} it/s   "
            f"{self.seconds_per_step * 1000:>7.0f} ms/step   "
            f"fixed {self.fixed_overhead_seconds:>6.1f} s   "
            f"{PRODUCTION_STEPS} steps -> {self.production_hours():>5.2f} h{warning}"
        )


def validate_finite_run_record(record: Mapping[str, Any]) -> None:
    """Reject poisoned training evidence before it reaches a speed estimate."""

    values = {"train_loss": record.get("train_loss")}
    eval_metrics = record.get("eval_metrics", {})
    if isinstance(eval_metrics, Mapping):
        values.update(eval_metrics)
    for name, value in values.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SpeedProbeError(f"non-finite {name}: {value}")


def validate_probe(probe: SpeedProbe) -> None:
    """Require a finite positive slope and coherent non-negative intercept."""

    if probe.long_seconds <= probe.short_seconds:
        raise SpeedProbeError("long probe did not take longer than short probe")
    if not math.isfinite(probe.seconds_per_step) or probe.seconds_per_step <= 0:
        raise SpeedProbeError("seconds per step must be finite and positive")
    if not math.isfinite(probe.fixed_overhead_seconds):
        raise SpeedProbeError("fixed overhead must be finite")
    if probe.fixed_overhead_seconds < 0:
        raise SpeedProbeError("negative fixed overhead makes the slope unreliable")


def timed_run(
    config: dict,
    arm_name: str,
    steps: int,
    *,
    seed: int,
    val_images: int,
    callbacks: tuple[Any, ...] = (),
) -> float:
    """Wall seconds for a complete run_arm at `steps`, from a clean directory."""

    paths = load_project_paths()
    pool = paths.synthetic / POOL_TAG
    arms = build_all_arms(
        manifest_path=paths.splits / "split_manifest.json",
        synthetic_annotations={
            "filtered": pool / "annotations_filtered_1x.json",
            "unfiltered": pool / "annotations_unfiltered_1x.json",
        },
    )
    composition = arms[arm_name]
    composition = replace(composition, real_val=composition.real_val[:val_images])

    output_dir = paths.runs / "speed_probe" / f"{arm_name}_{steps}"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_paths = RunPaths(
        real_images=paths.hardhat_raw / "images",
        real_coco=paths.interim / "coco_all.json",
        synthetic_images=pool / "images",
        synthetic_coco=(
            pool / "annotations_filtered_1x.json"
            if arm_name == "filtered_syn"
            else pool / "annotations_unfiltered_1x.json"
        ),
        output_dir=output_dir,
    )

    started = time.perf_counter()
    record = run_arm(
        composition,
        run_paths,
        config=config,
        total_steps=steps,
        seed=seed,
        resume=False,
        callbacks=callbacks,
    )
    validate_finite_run_record(record)
    elapsed = time.perf_counter() - started
    shutil.rmtree(output_dir, ignore_errors=True)
    return elapsed


def probe(
    label: str,
    config_path: Path,
    arm: str,
    *,
    short: int,
    long: int,
    seed: int,
    val_images: int,
    config: dict[str, Any] | None = None,
    safety_policy: UnattendedSafetyPolicy | None = None,
) -> SpeedProbe:
    config = copy.deepcopy(
        load_training_config(config_path) if config is None else config
    )
    # A 2000-step warmup inside a 120-step probe would measure the warmup only.
    config["schedule"]["warmup_steps"] = 1
    config["run"]["eval_on_n_val_images"] = val_images

    print(f"  {label}: {short} steps...", flush=True)
    callbacks: tuple[Any, ...] = ()
    if safety_policy is not None:
        safety_policy.check(stage="before_probe", arm=arm, step=short)
        callbacks = (TrainerHealthCallback(policy=safety_policy, arm=arm),)
    short_seconds = timed_run(
        config,
        arm,
        short,
        seed=seed,
        val_images=val_images,
        callbacks=callbacks,
    )
    print(f"    {short_seconds:.1f} s", flush=True)

    print(f"  {label}: {long} steps...", flush=True)
    if safety_policy is not None:
        safety_policy.check(stage="before_probe", arm=arm, step=long)
    long_seconds = timed_run(
        config,
        arm,
        long,
        seed=seed,
        val_images=val_images,
        callbacks=callbacks,
    )
    print(f"    {long_seconds:.1f} s", flush=True)

    return SpeedProbe(
        label=label,
        short_steps=short,
        long_steps=long,
        short_seconds=short_seconds,
        long_seconds=long_seconds,
    )


# spec: TRAIN-01
def validate_step_pair(short: int, long: int) -> None:
    """Refuse a degenerate step pair before anything expensive starts.

    Equal counts divide by zero. An inverted pair produces a NEGATIVE rate and
    therefore negative hours, reported with a straight face.

    A free function rather than an inline check in main(), because the test for
    it must not call main(). The first version of that test did, and when a
    mutation removed the guard the test started a REAL multi-hour training run -
    the mutation harness had to be killed at ten minutes, and being killed meant
    its restore never ran and it left the mutation in the working tree (K-21).
    """

    if long <= short:
        raise SystemExit(f"--long must exceed --short, got {long} <= {short}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="measured training speed on this machine")
    parser.add_argument("--arm", default="real_only")
    parser.add_argument("--short", type=int, default=40)
    parser.add_argument("--long", type=int, default=140)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-images", type=int, default=16)
    parser.add_argument("--min-free-gib", type=float, default=50.0)
    parser.add_argument("--max-runtime-hours", type=float, default=4.0)
    parser.add_argument("--max-gpu-temperature-c", type=float, default=85.0)
    parser.add_argument("--health-log", type=Path, default=None)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["configs/training.yaml", "configs/training_rfdetr.yaml"],
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "train_speed.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_step_pair(args.short, args.long)

    paths = load_project_paths()
    safety_policy = UnattendedSafetyPolicy(
        output_root=paths.runs,
        health_log=args.health_log
        or (PROJECT_ROOT / "reports" / "rfdetr_probe_health.jsonl"),
        deadline_utc=datetime.now(UTC)
        + timedelta(hours=float(args.max_runtime_hours)),
        min_free_gib=float(args.min_free_gib),
        max_gpu_temperature_c=float(args.max_gpu_temperature_c),
    )
    safety_policy.check(stage="startup")

    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"device: {device}")
    print(f"slope from {args.short} and {args.long} steps, arm={args.arm}\n")

    probes = []
    for config_path in args.configs:
        path = Path(config_path)
        label = path.stem.replace("training_", "").replace("training", "rtdetrv2")
        measured = probe(
                label,
                path,
                args.arm,
                short=args.short,
                long=args.long,
                seed=args.seed,
                val_images=args.val_images,
                safety_policy=safety_policy,
            )
        validate_probe(measured)
        probes.append(measured)

    print("\n" + "=" * 100)
    for item in probes:
        print(item.render())

    lines = [
        "# Measured training speed on this machine",
        "",
        f"- Device: `{device}`",
        f"- Arm: `{args.arm}`, seed `{args.seed}`",
        (
            f"- Method: two runs ({args.short} and {args.long} steps); the rate is the "
            "SLOPE between them, so fixed setup cost cancels rather than being "
            "amortised into a per-step number."
        ),
        (
            "- Warmup steps forced to 1 for the probe. The production schedule warms up "
            "over 2,000, which would otherwise be the only thing a short probe measures."
        ),
        "",
        f"| Model | it/s | ms/step | fixed cost (s) | {PRODUCTION_STEPS} steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in probes:
        lines.append(
            f"| `{item.label}` | {item.steps_per_second:.2f} | "
            f"{item.seconds_per_step * 1000:.0f} | {item.fixed_overhead_seconds:.1f} | "
            f"**{item.production_hours():.2f} h** |"
        )
    lines += [
        "",
        (
            "Reference point, measured not assumed: the four production arms ran on a "
            "Colab L4 at 1.7-1.9 it/s, about 1.6-1.75 hours each."
        ),
        "",
        (
            "A negative fixed cost means the two runs disagree about setup overhead; "
            "the safety gate rejects that probe before it can write an ETA report."
        ),
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {args.out}")

    payload = {
        "device": device,
        "arm": args.arm,
        "probes": [
            {
                "label": p.label,
                "steps_per_second": p.steps_per_second,
                "seconds_per_step": p.seconds_per_step,
                "fixed_overhead_seconds": p.fixed_overhead_seconds,
                "production_hours": p.production_hours(),
            }
            for p in probes
        ],
    }
    (args.out.with_suffix(".json")).write_text(
        json.dumps(payload, indent=2), encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
