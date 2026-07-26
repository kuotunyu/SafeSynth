"""Prepare, build, finalize, and verify the frozen-Train SAM2 cutout bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.cutout_bank import (
    finalize_bank,
    prepare_candidates,
    run_pass2,
    validate_bank,
    verify_mask_reproducibility,
)
from src.synthetic.sam2_runner import Sam2BoxSegmenter
from src.synthetic.survey import load_compose_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "build", "finalize", "validate", "repro"))
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument(
        "--compose-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "compose.yaml",
    )
    return parser.parse_args()


def build_segmenter(config: dict[str, Any]) -> Sam2BoxSegmenter:
    cleanup = config["sam2"]["cleanup"]
    return Sam2BoxSegmenter(
        model_id=config["sam2"]["model_id"],
        dtype=config["sam2"]["dtype"],
        morph_close_kernel=int(cleanup["morph_close_kernel"]),
        morph_close_iterations=int(cleanup["morph_close_iterations"]),
    )


def main() -> None:
    args = parse_args()
    paths = load_project_paths()
    config = load_compose_config(args.compose_config)
    if args.phase == "prepare":
        summary = prepare_candidates(paths=paths, config=config)
    elif args.phase == "build":
        summary = run_pass2(
            paths=paths,
            config=config,
            segmenter=build_segmenter(config),
            config_path=args.compose_config,
            max_candidates=args.max_candidates,
            force=args.force,
        )
    elif args.phase == "finalize":
        summary = finalize_bank(
            paths=paths, config=config, allow_incomplete=args.allow_incomplete
        )
    elif args.phase == "validate":
        summary = validate_bank(paths=paths)
    else:
        summary = verify_mask_reproducibility(
            paths=paths,
            config=config,
            segmenter=build_segmenter(config),
            n=args.n,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

