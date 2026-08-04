"""Prepare the two local Hugging Face payloads for owner review and upload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.release.hf_bundle import (
    ReleaseBundleError,
    prepare_dataset_bundle,
    prepare_model_bundle,
)

PUBLISHING_ROOT = PROJECT_ROOT / "publishing" / "huggingface"


def _preflight_inputs(args: argparse.Namespace) -> None:
    """Reject incomplete release inputs before creating either output directory."""

    required = (
        args.dataset_source / "annotations_filtered_1x.json",
        args.dataset_source / "annotations_unfiltered_1x.json",
        args.dataset_source / "records.jsonl",
        args.checkpoint / "config.json",
        args.checkpoint / "model.safetensors",
        args.processor_config,
        args.dataset_card,
        args.model_card,
    )
    for path in required:
        if not path.is_file():
            raise ReleaseBundleError(f"missing release input: {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    paths = load_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-source", type=Path, default=paths.synthetic / "m13_pool_1x"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=paths.runs / "real_only" / "seed_1337" / "checkpoint-1752",
    )
    parser.add_argument(
        "--processor-config",
        type=Path,
        default=PUBLISHING_ROOT / "model" / "preprocessor_config.json",
    )
    parser.add_argument(
        "--dataset-card", type=Path, default=PUBLISHING_ROOT / "dataset" / "README.md"
    )
    parser.add_argument(
        "--model-card", type=Path, default=PUBLISHING_ROOT / "model" / "README.md"
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=paths.data_root / "publish" / "safesynth-hard-hat",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=paths.data_root / "publish" / "safesynth-rtdetrv2-r18",
    )
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _preflight_inputs(args)
        dataset = prepare_dataset_bundle(
            args.dataset_source,
            args.dataset_output,
            args.dataset_card,
            link_mode=args.link_mode,
        )
        model = prepare_model_bundle(
            args.checkpoint,
            args.processor_config,
            args.model_output,
            args.model_card,
        )
    except (OSError, ValueError, ReleaseBundleError) as error:
        print(f"release preparation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset": dataset, "model": model}, indent=2, sort_keys=True))
    print(f"dataset payload: {args.dataset_output}")
    print(f"model payload: {args.model_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
