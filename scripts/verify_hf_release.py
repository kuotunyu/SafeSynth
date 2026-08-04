"""Verify both local Hugging Face payloads before their owner uploads them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.paths import load_project_paths
from src.release.hf_bundle import (
    ReleaseBundleError,
    verify_dataset_bundle,
    verify_model_bundle,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    data_root = load_project_paths().data_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=data_root / "publish" / "safesynth-hard-hat",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=data_root / "publish" / "safesynth-rtdetrv2-r18",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset = verify_dataset_bundle(args.dataset)
        model = verify_model_bundle(args.model)
    except (OSError, ValueError, ReleaseBundleError) as error:
        print(f"Hugging Face release verification failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset": dataset, "model": model}, indent=2, sort_keys=True))
    print("PASS: both Hugging Face owner-upload payloads are complete and internally consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
