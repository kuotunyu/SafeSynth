"""Generate M12's one-at-a-time filter threshold sensitivity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.filtering.sensitivity import analyze_threshold_sensitivity


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _render_markdown(result: dict[str, Any], source: Path) -> str:
    lines = [
        "# M12 filter threshold sensitivity",
        "",
        f"- Source records: `{source}`",
        f"- Samples: {result['n_samples']}",
        f"- Baseline acceptance: {result['baseline_acceptance']:.2%}",
        f"- Alarm threshold: {result['alarm_points']:.1f} percentage points",
        f"- Thresholds above alarm: {result['alarm_count']}",
        "",
        "Each numeric rule leaf is changed independently by -20% and +20%.",
        "",
        "| threshold | base | -20% rate | +20% rate | max swing (pp) | alarm |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['path']}` | {row['baseline_value']:.6g} | "
            f"{row['minus_20pct']:.2%} | {row['plus_20pct']:.2%} | "
            f"{row['max_swing_percentage_points']:.2f} | "
            f"{'YES' if row['alarm'] else ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        help="Defaults to <synthetic>/m10_seed42/records.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_project_paths()
    records_path = args.records or paths.synthetic / "m10_seed42" / "records.jsonl"
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "filtering.yaml").read_text(encoding="utf-8")
    )
    result = analyze_threshold_sensitivity(_read_jsonl(records_path), config)
    output_json = paths.reports / "threshold_sensitivity.json"
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (paths.reports / "threshold_sensitivity.md").write_text(
        _render_markdown(result, records_path),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "n_samples",
                    "baseline_acceptance",
                    "alarm_count",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
