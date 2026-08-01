"""Put the operating-point numbers into detection_metrics.csv so they are checkable.

`scripts/eval.py` writes the COCO metrics. Three more numbers belong in the
README and are produced elsewhere:

  bare_head_recall_at_op   Test recall read at the frozen compliance threshold.
                           EVAL-05b: the value at threshold 0 is a ceiling, not
                           a result, so the headline number is this one.
  operating_point          the threshold EVAL-04 selected on Validation, with
  op_bare_head_recall      the recall and precision it was selected for.
  op_compliance_precision

Without them in the CSV, `scripts/verify_readme.py` correctly rejects every
README cell that quotes them - the whole point of PUB-01 is that a number with
no source in `results/` is a failure rather than a warning. Appending them here
keeps one file as the thing every report is checked against (EVAL-12).

Idempotent: rows this script owns are replaced, not duplicated. Reads the stored
detections, so no inference and no GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.detection import bare_head_recall, load_evaluation_config
from src.inference.compliance import (
    detections_from_coco,
    select_operating_point,
    sweep_operating_points,
)
from src.training.arms import ARMS
from src.training.data import load_coco_samples
from src.training.metrics import build_coco_ground_truth

CLASS_NAMES = ("helmet", "head", "person")
METRICS_CSV = PROJECT_ROOT / "results" / "detection_metrics.csv"
INDEX_PATH = PROJECT_ROOT / "results" / "predictions_index.json"
FIELDS = (
    "arm",
    "seed",
    "split",
    "metric",
    "value",
    "n_instances",
    "n_images",
    "ci_low",
    "ci_high",
    "notes",
)
# Rows this script owns. Anything with these metric names is rewritten on each
# run so repeated invocations cannot stack duplicates.
OWNED = (
    "op_usable",
    "bare_head_recall_at_op",
    "operating_point",
    "op_bare_head_recall",
    "op_compliance_precision",
)
SWEEP_GRID = tuple(round(0.005 * step, 3) for step in range(1, 61)) + (0.35, 0.40, 0.50)


class DerivedMetricsError(RuntimeError):
    """Raised when the inputs the derived rows need are missing."""


def split_samples(paths, split: str):
    manifest = json.loads(
        (paths.splits / "split_manifest.json").read_text(encoding="utf-8")
    )
    images = manifest["images"] if isinstance(manifest, dict) else manifest
    names = sorted(
        entry["file_name"].split("/")[-1]
        for entry in images
        if entry["split"] == split
    )
    return load_coco_samples(
        paths.interim / "coco_all.json", paths.hardhat_raw / "images", keep_names=names
    )


def stored(index: dict, arm: str, split: str, seed: int) -> list[dict]:
    entry = index.get(f"{arm}/{split}/seed_{seed}")
    if entry is None:
        raise DerivedMetricsError(f"no stored {split} predictions for {arm}")
    return json.loads(Path(entry["path"]).read_text(encoding="utf-8"))


def derived_rows(paths, index: dict, seed: int, config: dict) -> list[dict]:
    threshold = float(config["compliance"]["score_threshold"])
    test_samples = split_samples(paths, "test")
    test_gt = build_coco_ground_truth(test_samples, CLASS_NAMES)
    val_samples = split_samples(paths, "val")

    rows: list[dict] = []
    for arm in ARMS:
        recall = bare_head_recall(
            test_gt,
            stored(index, arm, "test", seed),
            config=config,
            score_threshold=threshold,
        )
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "split": "test",
                "metric": "bare_head_recall_at_op",
                "value": recall.recall,
                "n_instances": recall.n_ground_truth,
                "n_images": len(test_samples),
                "ci_low": "",
                "ci_high": "",
                "notes": (
                    f"IoU {recall.iou_threshold:.2f} at the frozen compliance "
                    f"threshold {threshold:.2f} (EVAL-05b); the value at threshold 0 "
                    f"is a ceiling, not a result"
                ),
            }
        )

        points = sweep_operating_points(
            detections_from_coco(stored(index, arm, "val", seed)),
            val_samples,
            split="val",
            config=config,
            thresholds=SWEEP_GRID,
        )
        chosen = select_operating_point(points, config=config)
        floor = float(config["compliance"]["min_compliance_precision"])
        note = (
            f"EVAL-04 selected on VALIDATION against a {floor:.2f} precision floor"
            if chosen is not None
            else (
                f"no threshold reaches the {floor:.2f} precision floor while still "
                f"detecting anything; this arm has no usable operating point"
            )
        )
        # `op_usable` carries the fact; the three numeric rows exist only when
        # there is something to put in them. An arm with no usable point gets no
        # threshold row rather than an empty or zero one - "" breaks the CSV
        # round-trip, and 0.0 would read as "threshold zero" and "recall zero",
        # both of which are false statements about a arm that simply has no point.
        measured = (
            ()
            if chosen is None
            else (
                ("operating_point", chosen.score_threshold),
                ("op_bare_head_recall", chosen.bare_head_recall),
                ("op_compliance_precision", chosen.compliance_precision),
            )
        )
        for metric, value in (("op_usable", 0.0 if chosen is None else 1.0), *measured):
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "split": "val",
                    "metric": metric,
                    "value": value,
                    "n_instances": "",
                    "n_images": len(val_samples),
                    "ci_low": "",
                    "ci_high": "",
                    "notes": note,
                }
            )
    return rows


def merge(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    kept = [row for row in existing if row["metric"] not in OWNED]
    return kept + new_rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.metrics_csv.is_file():
        print(f"{args.metrics_csv} not found - run scripts/eval.py first")
        return 2
    if not args.index.is_file():
        print(f"{args.index} not found - run scripts/dump_predictions.py first")
        return 2

    paths = load_project_paths()
    config = load_evaluation_config()
    index = json.loads(args.index.read_text(encoding="utf-8"))

    with args.metrics_csv.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))

    try:
        new_rows = derived_rows(paths, index, args.seed, config)
    except DerivedMetricsError as error:
        print(f"cannot derive: {error}")
        return 2

    merged = merge(existing, new_rows)
    with args.metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)

    print(f"wrote {args.metrics_csv}: {len(merged)} rows ({len(new_rows)} derived)")
    for row in new_rows:
        if row["metric"] == "bare_head_recall_at_op":
            print(f"  {row['arm']:15s} bare_head_recall_at_op {row['value']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
