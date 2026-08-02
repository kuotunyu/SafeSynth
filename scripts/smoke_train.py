"""TRAIN-13: prove the training path runs locally before it costs Colab time.

Three things are verified:

1. A tiny number of real optimizer steps completes and writes a checkpoint.
2. The EVALUATION path runs and returns real COCO metrics.
3. Running again with that checkpoint present RESUMES rather than restarting.

Point 2 is here because its absence cost a four-arm Colab run. The first version
of this script set eval_strategy="no" to keep the smoke test fast, so
compute_metrics was never executed locally, and all four arms died in it after
about two minutes of training each. A smoke test that skips a code path does not
cover that code path, however green it looks.

The evaluation runs on a small slice of val rather than all 756 images, which
keeps it quick while still exercising the batching, the shape extraction and
COCOeval end to end.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.training.arms import build_all_arms
from src.training.config import load_training_config
from src.training.run import RunPaths, run_arm
from src.training.trainer import find_resumable_checkpoint

POOL_TAG = "m13_pool_1x"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training.yaml")
    )
    parser.add_argument("--arm", default="filtered_syn")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resume-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-images", type=int, default=16)
    parser.add_argument("--keep", action="store_true", help="do not delete the run dir")
    return parser.parse_args(argv)


def smoke_output_dir(
    runs_root: Path, config_path: Path, arm: str, seed: int
) -> Path:
    """Keep checkpoints from different detector architectures isolated."""

    return Path(runs_root) / "smoke" / Path(config_path).stem / f"{arm}_seed_{seed}"


def main() -> None:
    args = parse_args()
    paths = load_project_paths()
    pool = paths.synthetic / POOL_TAG
    config = load_training_config(args.config)
    # Evaluation MUST run here. Skipping it is exactly what let a broken
    # compute_metrics reach Colab and kill all four arms.
    config["run"]["eval_strategy"] = "steps"
    config["run"]["save_strategy"] = "steps"
    config["run"]["load_best_model_at_end"] = False
    config["schedule"]["warmup_steps"] = 1
    config["run"]["eval_on_n_val_images"] = args.val_images

    arms = build_all_arms(
        manifest_path=paths.splits / "split_manifest.json",
        synthetic_annotations={
            "filtered": pool / "annotations_filtered_1x.json",
            "unfiltered": pool / "annotations_unfiltered_1x.json",
        },
    )
    composition = arms[args.arm]
    # Evaluate on a slice of val so the smoke test stays fast while still
    # running the full eval path: batching, shape extraction, COCOeval.
    composition = replace(
        composition, real_val=composition.real_val[: args.val_images]
    )

    output_dir = smoke_output_dir(paths.runs, args.config, args.arm, args.seed)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_paths = RunPaths(
        real_images=paths.hardhat_raw / "images",
        real_coco=paths.interim / "coco_all.json",
        synthetic_images=pool / "images",
        synthetic_coco=(
            pool / "annotations_filtered_1x.json"
            if args.arm == "filtered_syn"
            else pool / "annotations_unfiltered_1x.json"
            if args.arm == "unfiltered_syn"
            else None
        ),
        output_dir=output_dir,
    )

    print(f"=== cold start: {args.steps} steps, no checkpoint present ===")
    assert find_resumable_checkpoint(output_dir) is None, "run dir was not clean"
    first = run_arm(
        composition,
        run_paths,
        config=config,
        total_steps=args.steps,
        seed=args.seed,
        resume=True,
    )
    checkpoint = find_resumable_checkpoint(output_dir)
    print(f"    steps={first['total_steps']}  resumed_from={first['resumed_from']}")
    print(f"    checkpoint written: {checkpoint}")
    if checkpoint is None:
        raise SystemExit("TRAIN-10 failed: no checkpoint written on the cold run")

    print(f"\n=== warm start: {args.resume_steps} steps, checkpoint present ===")
    second = run_arm(
        composition,
        run_paths,
        config=config,
        total_steps=args.resume_steps,
        seed=args.seed,
        resume=True,
    )
    print(f"    steps={second['total_steps']}  resumed_from={second['resumed_from']}")
    if second["resumed_from"] is None:
        raise SystemExit("TRAIN-10 failed: the warm run ignored the checkpoint")

    report = {
        "arm": args.arm,
        "cold_start": first,
        "warm_start": second,
        "resume_verified": second["resumed_from"] is not None,
        "checkpoint_written_on_cold_start": True,
    }
    (PROJECT_ROOT / "reports" / "train_smoke.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("\nresume branch verified both ways; wrote reports/train_smoke.json")

    if not args.keep:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"cleaned {output_dir}")


if __name__ == "__main__":
    main()
