"""TRAIN-13: prove the training path runs locally before it costs Colab time.

Two things are verified, and the second is the one that matters on Colab:

1. A tiny number of real optimizer steps completes and writes a checkpoint.
2. Running again with that checkpoint present RESUMES rather than restarting.

TRAIN-10 says resume is not an optional feature because Colab disconnects, and
that it is accepted by measurement rather than by inspection. So this script runs
the branch both ways and compares.
"""

from __future__ import annotations

import argparse
import json
import shutil

import yaml

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.training.arms import build_all_arms
from src.training.run import RunPaths, run_arm
from src.training.trainer import find_resumable_checkpoint

POOL_TAG = "m13_pool_1x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="filtered_syn")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--resume-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--keep", action="store_true", help="do not delete the run dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_project_paths()
    pool = paths.synthetic / POOL_TAG
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "training.yaml").read_text(encoding="utf-8")
    )
    # A smoke test must not evaluate the full 756-image val set; that would take
    # longer than the training it is smoke-testing.
    config["run"]["eval_strategy"] = "no"
    config["run"]["save_strategy"] = "steps"
    config["run"]["load_best_model_at_end"] = False
    config["schedule"]["warmup_steps"] = 1

    arms = build_all_arms(
        manifest_path=paths.splits / "split_manifest.json",
        synthetic_annotations={
            "filtered": pool / "annotations_filtered_1x.json",
            "unfiltered": pool / "annotations_unfiltered_1x.json",
        },
    )
    composition = arms[args.arm]

    output_dir = paths.runs / "smoke" / f"{args.arm}_seed_{args.seed}"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    run_paths = RunPaths(
        real_images=paths.hardhat_raw / "images",
        real_coco=paths.interim / "coco_all.json",
        synthetic_images=pool / "images",
        synthetic_coco=pool / "annotations_filtered_1x.json"
        if args.arm == "filtered_syn"
        else pool / "annotations_unfiltered_1x.json",
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
