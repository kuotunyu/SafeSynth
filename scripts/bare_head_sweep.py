"""EVAL-05b: bare-head recall is only a metric once you say at what threshold.

RT-DETRv2 emits a fixed 300 queries per image. Matched one-to-one at IoU 0.50
with no score floor, essentially every ground-truth bare head finds some box, so
recall at threshold 0 sits near the ceiling for every arm and separates none of
them. Measured on this project's four arms it landed between 0.987 and 0.990 -
a spread of three parts in a thousand across models whose mAP differs by 15%.

So this sweeps the threshold and prints the curve. The value that belongs in the
main table is the one at `compliance.score_threshold`; the threshold-0 value is
a RECALL CEILING and is labelled as such.

Reads the detections written by scripts/dump_predictions.py. No inference, no
GPU: the sweep is pure arithmetic over stored boxes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.detection import bare_head_recall, load_evaluation_config
from src.training.arms import ARMS
from src.training.data import load_coco_samples
from src.training.metrics import build_coco_ground_truth

CLASS_NAMES = ("helmet", "head", "person")
INDEX_PATH = PROJECT_ROOT / "results" / "predictions_index.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "bare_head_recall_sweep.md"
# Dense at the bottom: this model's scores are compressed (max 0.2495 over
# 223,200 test detections), so a grid spaced for a calibrated detector would
# put the entire usable range in its first cell. 0.07 is the frozen operating
# point selected by EVAL-04 and must be present so the main table can read it.
THRESHOLDS = (0.0, 0.02, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50)


class SweepError(RuntimeError):
    """Raised when stored predictions are missing or unreadable."""


def load_index(path: Path = INDEX_PATH) -> dict:
    if not path.is_file():
        raise SweepError(
            f"{path} not found - run `uv run python -m scripts.dump_predictions` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def split_ground_truth(paths, split: str):
    manifest = json.loads(
        (paths.splits / "split_manifest.json").read_text(encoding="utf-8")
    )
    images = manifest["images"] if isinstance(manifest, dict) else manifest
    names = sorted(
        entry["file_name"].split("/")[-1]
        for entry in images
        if entry["split"] == split
    )
    samples = load_coco_samples(
        paths.interim / "coco_all.json", paths.hardhat_raw / "images", keep_names=names
    )
    return build_coco_ground_truth(samples, CLASS_NAMES)


def sweep(ground_truth, detections, thresholds, config) -> list[dict]:
    rows = []
    for threshold in thresholds:
        result = bare_head_recall(
            ground_truth, detections, config=config, score_threshold=threshold
        )
        rows.append(
            {
                "score_threshold": threshold,
                "recall": result.recall,
                "n_matched": result.n_matched,
                "n_ground_truth": result.n_ground_truth,
                "iou_threshold": result.iou_threshold,
            }
        )
    return rows


def render(by_arm: dict, *, split: str, operating_point: float, iou: float) -> str:
    arms = list(by_arm)
    lines = [
        "# Bare-head recall against detector confidence (EVAL-05b)",
        "",
        (
            f"Split: **{split}**. IoU {iou:.2f}, one-to-one greedy matching, "
            "highest score first. Re-aggregated from the stored detections "
            "(EVAL-12), no inference."
        ),
        "",
        (
            "⚠️ **The row at threshold 0 is a recall CEILING, not a result.** The "
            "detector emits a fixed 300 queries per image, so with no score floor "
            "almost every bare head finds some box and every arm scores near the "
            "top. Read the main-table value at the frozen compliance operating "
            f"point of **{operating_point:.2f}**."
        ),
        "",
        "| threshold | " + " | ".join(f"`{arm}`" for arm in arms) + " | spread |",
        "|---:|" + "---:|" * (len(arms) + 1),
    ]
    for index, threshold in enumerate(THRESHOLDS):
        values = [by_arm[arm][index]["recall"] for arm in arms]
        spread = max(values) - min(values)
        marker = " **←**" if abs(threshold - operating_point) < 1e-9 else ""
        lines.append(
            f"| {threshold:.2f}{marker} | "
            + " | ".join(f"{value:.4f}" for value in values)
            + f" | {spread:.4f} |"
        )
    lines += [
        "",
        (
            "The `spread` column is the point: where it is near zero the metric is "
            "not measuring anything that distinguishes these models, and quoting a "
            "single number from that region would suggest a tie that the underlying "
            "detectors do not have."
        ),
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = load_project_paths()
    config = load_evaluation_config()
    operating_point = float(config["compliance"]["score_threshold"])
    iou = float(config["metrics"]["bare_head_recall_iou"])

    try:
        index = load_index()
    except SweepError as error:
        print(error)
        return 2

    ground_truth = split_ground_truth(paths, args.split)
    by_arm: dict[str, list[dict]] = {}
    for arm in ARMS:
        entry = index.get(f"{arm}/{args.split}/seed_{args.seed}")
        if entry is None:
            print(f"  {arm}: no stored predictions for {args.split}")
            continue
        detections = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        by_arm[arm] = sweep(ground_truth, detections, THRESHOLDS, config)
        at_point = next(
            row["recall"]
            for row in by_arm[arm]
            if abs(row["score_threshold"] - operating_point) < 1e-9
        )
        ceiling = by_arm[arm][0]["recall"]
        print(f"  {arm:15s} ceiling {ceiling:.4f}   @{operating_point:.2f} {at_point:.4f}")

    if not by_arm:
        print("no stored predictions for any arm")
        return 2

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render(by_arm, split=args.split, operating_point=operating_point, iou=iou),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
