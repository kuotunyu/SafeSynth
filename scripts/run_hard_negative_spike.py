"""Run guarded H6 mining or render procedural hard-negative evidence."""

from __future__ import annotations

import argparse
import json

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.hard_negatives import (
    mine_hard_negatives,
    render_procedural_grid,
    validate_human_signoff,
)
from src.synthetic.survey import load_compose_config, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("mine", "procedural", "check-signoff"))
    parser.add_argument("--n-images", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_project_paths()
    compose_config = load_compose_config(PROJECT_ROOT / "configs" / "compose.yaml")
    if args.phase == "mine":
        summary = mine_hard_negatives(
            paths=paths,
            compose_config=compose_config,
            calibration=load_json(paths.reports / "calibration.json"),
            n_images=args.n_images,
        )
    elif args.phase == "procedural":
        summary = render_procedural_grid(paths=paths, compose_config=compose_config)
    else:
        report_text = (paths.reports / "h6_hard_negative_spike.md").read_text(
            encoding="utf-8"
        )
        marker = "Contact-sheet SHA256: `"
        grid_sha = report_text.split(marker, 1)[1].split("`", 1)[0]
        summary = validate_human_signoff(
            signoff_path=paths.reports / "hard_negative_signoff.json",
            expected_grid_sha256=grid_sha,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

