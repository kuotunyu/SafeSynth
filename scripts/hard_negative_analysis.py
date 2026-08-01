"""EVAL-16 on the frozen Test split, using the spec's own fallback.

The metric wants false positives per image on images that contain hard
negatives. Test has no such images: all 744 contain a helmet or a head, so the
subset is empty and M19 could not produce the table. EVAL-16's fallback is to
mine candidate regions on Test FOR ANALYSIS ONLY, which is what runs here.

ANALYSIS ONLY is a property this script has to keep, not just claim:

  * nothing written here is read by training or by generation - the outputs are
    a markdown report and a contact sheet under reports/;
  * the operating point is READ from configs/evaluation.yaml, where EVAL-04 put
    it after selecting on Validation. This script never searches for one, so it
    cannot tune anything on Test.

The regions are also rendered to a contact sheet, because Phase 1's purity
number (0 real helmets in 64 cells, H6) was measured on Train. The guards are
content-based so it should carry, but "should carry" is not a measurement and
the sheet is what lets someone check.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.detection import load_evaluation_config
from src.evaluation.hard_negatives import (
    HardNegativeAnalysisError,
    Region,
    count_false_positives,
    discriminates,
    mine_regions,
    render_table,
)

REPORT_NAME = "hard_negative_false_positives.md"
SHEET_NAME = "hard_negative_test_regions.png"
SPLIT = "test"


def load_split_images(paths, split: str):
    """Image records and per-image annotations for one split of the frozen data."""

    manifest = json.loads((paths.splits / "split_manifest.json").read_text(encoding="utf-8"))
    wanted = {e["file_name"] for e in manifest["images"] if e["split"] == split}
    coco = json.loads((paths.interim / "coco_all.json").read_text(encoding="utf-8"))

    images = {
        int(image["id"]): image
        for image in coco["images"]
        if str(image["file_name"]) in wanted
    }
    annotations: dict[int, list[dict]] = {}
    for record in coco["annotations"]:
        image_id = int(record["image_id"])
        if image_id in images:
            annotations.setdefault(image_id, []).append(record)
    category_names = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    return images, annotations, category_names


def render_region_sheet(
    regions: Sequence[Region], images_root: Path, destination: Path, *, columns: int = 8
) -> Path:
    """Crops of what was mined, so purity can be checked rather than assumed."""

    cell = 96
    rows = max(1, (len(regions) + columns - 1) // columns)
    sheet = np.full((rows * cell, columns * cell, 3), 24, dtype=np.uint8)
    for index, region in enumerate(regions):
        image = cv2.imread(str(Path(images_root) / region.file_name))
        if image is None:
            continue
        x, y, width, height = (round(v) for v in region.bbox)
        pad = max(4, int(0.25 * max(width, height)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(image.shape[1], x + width + pad)
        y1 = min(image.shape[0], y + height + pad)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)
        # Cyan box on the mined region itself, matching the H6 sheet a human
        # already learned to read.
        scale_x, scale_y = cell / max(x1 - x0, 1), cell / max(y1 - y0, 1)
        cv2.rectangle(
            crop,
            (int((x - x0) * scale_x), int((y - y0) * scale_y)),
            (int((x + width - x0) * scale_x), int((y + height - y0) * scale_y)),
            (255, 255, 0),
            1,
        )
        r, c = divmod(index, columns)
        sheet[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = crop
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).save(destination)
    return destination


def load_predictions(index: Mapping[str, Mapping], arm: str, split: str):
    key = f"{arm}/{split}/seed_1337"
    if key not in index:
        raise HardNegativeAnalysisError(f"{key} is not in results/predictions_index.json")
    return json.loads(Path(index[key]["path"]).read_text(encoding="utf-8"))


def _spread(results) -> int:
    """Widest gap in false-positive count across the arms."""

    counts = [r.n_false_positives for r in results]
    return max(counts) - min(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EVAL-16 hard-negative false positives")
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports" / REPORT_NAME)
    parser.add_argument(
        "--sheet", type=Path, default=PROJECT_ROOT / "reports" / "figures" / SHEET_NAME
    )
    parser.add_argument("--max-sheet-cells", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_project_paths()
    evaluation = load_evaluation_config()
    threshold = float(evaluation["compliance"]["score_threshold"])

    compose = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "compose.yaml").read_text(encoding="utf-8")
    )
    calibration = json.loads(
        (paths.reports / "calibration.json").read_text(encoding="utf-8")
    )
    mining = compose["hard_negatives"]["mining"]
    helmet_geometry = calibration["geometry"]["per_class"]["helmet"]

    images, annotations, category_names = load_split_images(paths, args.split)
    print(f"{args.split}: {len(images)} images, {sum(map(len, annotations.values()))} annotations")

    def load_hsv(file_name: str) -> np.ndarray:
        rgb = np.asarray(Image.open(paths.hardhat_raw / file_name).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    regions, rejects = mine_regions(
        images,
        annotations,
        category_names=category_names,
        helmet_geometry=helmet_geometry,
        mining=mining,
        load_hsv=load_hsv,
    )
    print(f"mined {len(regions)} regions; guard rejections {rejects}")
    if not regions:
        print("no regions survived the guards - EVAL-16 has nothing to measure here")
        return 1

    index = json.loads(
        (PROJECT_ROOT / "results" / "predictions_index.json").read_text(encoding="utf-8")
    )
    results = []
    for arm in ("real_only", "standard_aug", "unfiltered_syn", "filtered_syn"):
        detections = load_predictions(index, arm, args.split)
        result = count_false_positives(
            detections, regions, arm=arm, score_threshold=threshold
        )
        results.append(result)
        print(f"  {arm:<15} {result.n_false_positives:>5} FP  {result.per_image:.3f} per image")

    sheet = render_region_sheet(
        regions[: args.max_sheet_cells], paths.hardhat_raw, args.sheet
    )
    shown = min(len(regions), args.max_sheet_cells)

    report = "\n".join(
        [
            "# EVAL-16 — false positives on hard-negative regions (Test)",
            "",
            "> **Analysis only.** The frozen Test split contains no naturally empty",
            "> images - all 744 carry a helmet or a head - so the subset EVAL-16 asks",
            "> for is empty. This is the spec's own fallback: mine candidate regions on",
            "> Test and measure against those. Nothing here feeds training, and the",
            "> operating point is read from `configs/evaluation.yaml`, where EVAL-04",
            "> placed it after selecting on Validation. No threshold is searched here.",
            "",
            "## Result",
            "",
            render_table(results),
            "",
            *(
                []
                if discriminates(results)
                else [
                    "**This metric does not separate the arms, and the numbers above",
                    "should not be read as a ranking.** The spread across four arms is",
                    f"{_spread(results)} detection(s). Comparing 0.005 against 0.000",
                    "here would be reading noise.",
                    "",
                    "The contact sheet below shows why, and it confirms a cost this",
                    "project already disclosed ([K-11](../docs/troubleshooting.md)): the",
                    "miner selects on HUE and roundness, so most of what it finds is",
                    "yellow or orange but nothing like a helmet in shape - planks,",
                    "barriers, machinery panels, bare arms. Those are not hard for the",
                    "detector, so no arm fires on them and there is nothing to compare.",
                    "A genuinely discriminating version of EVAL-16 would need",
                    "distractors that are helmet-SHAPED and not worn, which this",
                    "dataset does not supply in quantity on Test.",
                    "",
                ]
            ),
            "## What was mined",
            "",
            f"Guard rejections: `{rejects}`",
            "",
            f"![mined regions](figures/{SHEET_NAME})",
            "",
            f"{shown} of {len(regions)} regions shown, cyan box on the region itself.",
            "",
            "Phase 1 purity was human-checked on TRAIN: 64 cells, **0 real helmets**,",
            "against a maximum tolerated 10% (`reports/h6_hard_negative_spike.md`). The",
            "guards are content-based rather than split-dependent, so that expectation",
            "should carry to Test - but it is an expectation, and this sheet is what",
            "lets someone confirm it instead of taking it on trust.",
            "",
            "## Why this is defined over mined regions and not over unmatched detections",
            "",
            "Roughly two thirds of real objects in this dataset are unannotated",
            "(SHEL5K re-labelled the same 5,000 images and found 75,570 objects against",
            "the original 25,502). A detection with no matching ground truth is",
            "therefore usually a CORRECT detection of an unannotated object, and",
            "counting those as false positives would measure the annotation gap. A",
            "mined region has passed the worn-helmet guards, so a box on one is an",
            "error rather than a gap.",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(f"wrote {args.report}")
    print(f"wrote {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
