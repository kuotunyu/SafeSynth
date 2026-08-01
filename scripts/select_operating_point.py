"""M17 / EVAL-04: pick the compliance score threshold, on VALIDATION only.

The detector's confidence threshold for compliance is a different question from
mAP. mAP integrates over every confidence; a deployed compliance check has to
commit to one operating point. EVAL-04 says pick the point that maximises
bare-head recall subject to a floor on compliance precision, choose it on
Validation, and then freeze it.

Choosing it on Test would be tuning on the test set. That is not guarded here by
a comment: `sweep_operating_points` takes the split name and raises on "test",
so pointing this script at Test fails rather than quietly producing a number.

Runs on CPU. Inference over the validation split takes a few minutes and needs
no GPU, which matters because the GPU is often busy with another project.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import torch
from PIL import Image

matplotlib.use("Agg")  # No display; must precede pyplot.
import matplotlib.pyplot as plt

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.inference.compliance import (
    detections_from_coco,
    load_evaluation_config,
    select_operating_point,
    sweep_operating_points,
)
from src.training.data import load_coco_samples
from src.training.ingest import latest_checkpoint
from src.training.metrics import predictions_to_coco

CLASS_NAMES = ("helmet", "head", "person")
PROCESSOR_ID = "PekingU/rtdetr_v2_r18vd"

# `sweep_operating_points` defaults its candidate thresholds to every DISTINCT
# detection score, which is the right default for a handful of boxes and
# unusable here: the stored predictions keep all 300 queries per image, so a
# validation split is ~226,800 scores and the sweep becomes quadratic.
#
# The grid below is deliberately dense at the bottom. This model's scores are
# compressed: over 223,200 test detections the maximum is 0.2495 and nothing
# exceeds 0.30, because the best checkpoint arrives at step 1752 of 10,900 and
# the reinitialised 3-class head has not saturated. A grid spaced for a
# well-calibrated detector would put every usable point in its first cell.
SWEEP_GRID = tuple(round(0.005 * step, 3) for step in range(1, 61)) + (
    0.35,
    0.40,
    0.50,
)
REPORT_PATH = PROJECT_ROOT / "reports" / "compliance_operating_point.md"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "compliance_sweep.png"
BASELINE_ARM = "real_only"


class OperatingPointError(RuntimeError):
    """Raised when the sweep cannot be run against real predictions."""


def validation_samples(paths, limit: int | None = None):
    manifest = json.loads(
        (paths.splits / "split_manifest.json").read_text(encoding="utf-8")
    )
    images = manifest["images"] if isinstance(manifest, dict) else manifest
    names = sorted(
        entry["file_name"].split("/")[-1]
        for entry in images
        if entry["split"] == "val"
    )
    if not names:
        raise OperatingPointError("the frozen manifest lists no validation images")
    samples = load_coco_samples(
        paths.interim / "coco_all.json", paths.hardhat_raw / "images", keep_names=names
    )
    return samples[:limit] if limit else samples


def best_checkpoint(paths, arm: str, seed: int) -> Path:
    """The checkpoint validation selected, not the last one (see K-20)."""

    seed_dir = paths.runs / arm / f"seed_{seed}"
    last = latest_checkpoint(seed_dir)
    if last is None:
        raise OperatingPointError(f"{arm}: no checkpoint under {seed_dir}")
    state = json.loads((last / "trainer_state.json").read_text(encoding="utf-8"))
    recorded = state.get("best_model_checkpoint")
    if not recorded:
        return last
    # Recorded on Colab, so it is an absolute /content/... path that does not
    # exist here; only its name is portable.
    resolved = seed_dir / Path(recorded).name
    return resolved if resolved.is_dir() else last


def stored_predictions(arm: str, seed: int, split: str = "val") -> list[dict] | None:
    """Reuse scripts/dump_predictions.py output when it exists.

    Inference over validation is about 200 CPU-seconds per arm and produces
    exactly the same boxes every time, so paying for it again here would be
    waste. Returns None when nothing has been dumped, and the caller falls back
    to running the model.
    """

    index_path = PROJECT_ROOT / "results" / "predictions_index.json"
    if not index_path.is_file():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = index.get(f"{arm}/{split}/seed_{seed}")
    if entry is None:
        return None
    stored = Path(entry["path"])
    if not stored.is_file():
        return None
    print(f"reusing stored predictions: {stored.name} ({entry['n_detections']} boxes)")
    return json.loads(stored.read_text(encoding="utf-8"))


def predict(checkpoint: Path, samples) -> list[dict]:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(PROCESSOR_ID)
    model = AutoModelForObjectDetection.from_pretrained(
        str(checkpoint), dtype=torch.float32
    ).eval()
    records: list[dict] = []
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            encoded = processor(images=image, return_tensors="pt")
            outputs = model(**encoded)
            # Per-image target size: the split is not one resolution (DATA-25).
            target = torch.tensor([[sample.height, sample.width]], dtype=torch.float32)
            processed = processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=target
            )
            records += predictions_to_coco(processed, [int(sample.image_id)])
    return records


def plot_sweep(points, chosen, floor: float, destination: Path) -> Path:
    thresholds = [point.score_threshold for point in points]
    figure, axis = plt.subplots(figsize=(7.5, 4.4))
    axis.plot(
        thresholds,
        [point.bare_head_recall for point in points],
        color="#0173b2",
        label="bare-head recall",
    )
    axis.plot(
        thresholds,
        [point.compliance_precision for point in points],
        color="#de8f05",
        linestyle="--",
        label="compliance precision",
    )
    axis.axhline(
        floor, color="#949494", linestyle=":", linewidth=1.2, label="precision floor"
    )
    if chosen is not None:
        axis.axvline(chosen.score_threshold, color="#029e73", linewidth=1.2)
        axis.annotate(
            f"chosen {chosen.score_threshold:.2f}",
            xy=(chosen.score_threshold, chosen.bare_head_recall),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color="#029e73",
        )
    axis.set_xlabel("detector score threshold")
    axis.set_ylabel("score")
    axis.set_title("Compliance operating point, selected on VALIDATION (EVAL-04)")
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(loc="best", fontsize=9)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return destination


def render(points, chosen, *, arm, checkpoint, n_images, floor, figure_path) -> str:
    lines = [
        "# Compliance operating point (EVAL-04)",
        "",
        (
            f"Swept on the **frozen validation split** ({n_images} images) using "
            f"`{arm}` at `{checkpoint}`. Never swept on Test: "
            "`sweep_operating_points` takes the split name and raises on `\"test\"`, "
            "so tuning on the test set fails rather than producing a number."
        ),
        "",
        (
            "This threshold is deliberately separate from mAP evaluation, which "
            "integrates over every confidence. A deployed compliance check has to "
            "commit to one point."
        ),
        "",
        f"![sweep]({figure_path})",
        "",
        (
            "| threshold | bare-head recall | compliance precision | "
            "pred. compliant | pred. non-compliant |"
        ),
        "|---:|---:|---:|---:|---:|",
    ]
    for point in points:
        marker = " **←**" if chosen is not None and point is chosen else ""
        # Precision is undefined, not zero, at a threshold that predicts nothing:
        # there is no denominator. Printing 0.0000 would read as "it got them all
        # wrong" rather than "it made no call".
        precision = (
            "—"
            if point.compliance_precision is None
            else f"{point.compliance_precision:.4f}"
        )
        recall = (
            "—" if point.bare_head_recall is None else f"{point.bare_head_recall:.4f}"
        )
        lines.append(
            f"| {point.score_threshold:.3f}{marker} | {recall} | {precision} | "
            f"{point.n_predicted_compliant} | {point.n_predicted_non_compliant} |"
        )
    lines.append("")

    if chosen is None:
        lines += [
            "## No point clears the floor",
            "",
            (
                f"No swept threshold reaches a compliance precision of {floor:.2f}. "
                "Reported as it is rather than by lowering the floor to manufacture a "
                "selection: the floor is a statement about what a safety check has to "
                "be worth, and moving it after seeing the curve would make it "
                "meaningless. The remedy is a better detector, not a softer criterion."
            ),
        ]
    else:
        lines += [
            "## Selected",
            "",
            f"- `compliance.score_threshold` = **{chosen.score_threshold:.2f}**",
            f"- bare-head recall {chosen.bare_head_recall:.4f}",
            (
                f"- compliance precision {chosen.compliance_precision:.4f} "
                f"(floor {floor:.2f})"
            ),
            (
                "- ground-truth bare heads in validation: "
                f"{chosen.n_ground_truth_bare_heads}"
            ),
            "",
            (
                "Write this value into `configs/evaluation.yaml` under "
                "`compliance.score_threshold` and change its `source:` tag from "
                "`validation (placeholder)` to `validation`. It is then frozen: "
                "re-selecting it after seeing Test results would be tuning on Test."
            ),
        ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=BASELINE_ARM)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None, help="debug: fewer images")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = load_project_paths()
    config = load_evaluation_config()
    floor = float(config["compliance"]["min_compliance_precision"])

    try:
        samples = validation_samples(paths, args.limit)
        checkpoint = best_checkpoint(paths, args.arm, args.seed)
    except OperatingPointError as error:
        print(f"cannot sweep: {error}")
        return 2

    print(f"{args.arm} @ {checkpoint.name}: {len(samples)} validation images")
    records = None if args.limit else stored_predictions(args.arm, args.seed)
    if records is None:
        started = time.perf_counter()
        records = predict(checkpoint, samples)
        print(f"inference {time.perf_counter() - started:.0f}s, {len(records)} detections")

    points = sweep_operating_points(
        detections_from_coco(records),
        samples,
        split="val",
        config=config,
        thresholds=SWEEP_GRID,
    )
    chosen = select_operating_point(points, config=config)

    plot_sweep(points, chosen, floor, args.figure)
    try:
        link = args.figure.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        link = args.figure.as_posix()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render(
            points,
            chosen,
            arm=args.arm,
            checkpoint=checkpoint.name,
            n_images=len(samples),
            floor=floor,
            figure_path=link,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {args.report}")
    print(f"wrote {args.figure}")
    if chosen is None:
        print(f"NO POINT clears the precision floor of {floor:.2f}")
        return 1
    print(
        f"selected threshold {chosen.score_threshold:.2f}  "
        f"bare-head recall {chosen.bare_head_recall:.4f}  "
        f"precision {chosen.compliance_precision:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
