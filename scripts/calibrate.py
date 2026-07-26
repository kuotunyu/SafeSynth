"""Run M6/H7 calibration on the frozen Train split and optional Pass 1 masks."""

from __future__ import annotations

import argparse
import json

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.calibration import (
    calibrated_values,
    geometry_distributions,
    mask_distributions,
    remaining_guess_lines,
    write_calibration_report,
)
from src.synthetic.survey import load_compose_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-iou", type=float)
    parser.add_argument("--min-object-logit", type=float)
    parser.add_argument("--expected-pass1-images", type=int, default=3500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_project_paths()
    compose_config = load_compose_config(PROJECT_ROOT / "configs" / "compose.yaml")
    pass1_config = compose_config["sam2"]["pass1_survey"]
    min_iou = (
        float(args.min_iou)
        if args.min_iou is not None
        else float(pass1_config["min_iou_score"])
    )
    min_object_logit = (
        float(args.min_object_logit)
        if args.min_object_logit is not None
        else float(pass1_config["min_object_score_logit"])
    )
    calibration = {"geometry": geometry_distributions(paths)}
    record_count = len(list((paths.masks_pass1 / "records").glob("*.json")))
    if record_count == args.expected_pass1_images:
        calibration["masks"] = mask_distributions(
            paths,
            min_iou=min_iou,
            min_object_logit=min_object_logit,
        )
    values = calibrated_values(calibration)
    guess_lines = remaining_guess_lines(paths.project_root)
    (paths.reports / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    write_calibration_report(
        calibration=calibration,
        values=values,
        output_path=paths.reports / "calibration.md",
        guess_lines=guess_lines,
    )
    print(
        json.dumps(
            {
                "pass1_record_count": record_count,
                "pass1_complete": record_count == args.expected_pass1_images,
                "calibrated_values": values,
                "remaining_guess_lines": len(guess_lines),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
