"""Run Spike H2 or the resumable frozen-Train SAM2 Pass 1 survey."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.sam2_runner import Sam2BoxSegmenter
from src.synthetic.survey import (
    load_compose_config,
    rebuild_pass1_index,
    run_h2,
    run_pass1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("h2", "pass1", "reindex"))
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--recheck-qc", action="store_true")
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
    if args.phase == "reindex":
        summary = rebuild_pass1_index(
            paths=paths, config=config, recheck_qc=args.recheck_qc
        )
    elif args.phase == "h2":
        summary = run_h2(
            paths=paths,
            config=config,
            segmenter=build_segmenter(config),
            output_json=paths.cache / "h2" / "h2_results.json",
            figure_dir=paths.figures,
        )["summary"]
    else:
        summary = run_pass1(
            paths=paths,
            config=config,
            segmenter=build_segmenter(config),
            max_images=args.max_images,
            force=args.force,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

