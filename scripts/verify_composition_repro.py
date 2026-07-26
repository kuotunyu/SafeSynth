"""Compare two deterministic composition runs and persist the M10 proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.paths import load_project_paths


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-tag", default="m10_seed42")
    parser.add_argument("--second-tag", default="m10_repro")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_project_paths()
    first_path = paths.synthetic / args.first_tag / "summary.json"
    second_path = paths.synthetic / args.second_tag / "summary.json"
    first = _read(first_path)
    second = _read(second_path)
    first_hashes = first["image_hashes"]
    second_hashes = second["image_hashes"]
    all_ids = sorted(set(first_hashes) | set(second_hashes))
    mismatches = [
        sample_id
        for sample_id in all_ids
        if first_hashes.get(sample_id) != second_hashes.get(sample_id)
    ]
    result = {
        "first_summary": str(first_path),
        "second_summary": str(second_path),
        "first_images": len(first_hashes),
        "second_images": len(second_hashes),
        "mismatches": mismatches,
        "passed": not mismatches and len(first_hashes) == len(second_hashes),
    }
    output = paths.reports / "composition_reproducibility.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not result["passed"]:
        raise RuntimeError(f"Composition reproducibility failed: {mismatches[:10]}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
