"""Run each arm once and persist its detections, so nothing downstream re-infers.

Every later analysis - the main table, the threshold sweep for bare-head recall,
the compliance operating point, the four-way error grid - needs the same
predictions. Inference is the expensive step (about 200 CPU-seconds per arm per
split), so it happens here once and the artefacts are reused.

Predictions land on the DATA drive, not in the repository. At 300 queries per
image and 744 test images an arm produces on the order of 200,000 detections;
CLAUDE.md keeps the project folder to code, config, docs and small figures.

Boxes are written in each image's OWN original annotation coordinates (DATA-25:
the split holds 416x416, 416x415, 415x416 and 415x415 images, so scale_x and
scale_y differ and a single global factor would be wrong). Downstream code can
therefore treat these as ground-truth-space boxes with no further conversion,
which is what EVAL-07 requires.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.training.arms import ARMS
from src.training.data import load_coco_samples
from src.training.ingest import latest_checkpoint
from src.training.metrics import predictions_to_coco

PROCESSOR_ID = "PekingU/rtdetr_v2_r18vd"
INDEX_PATH = PROJECT_ROOT / "results" / "predictions_index.json"


class PredictionError(RuntimeError):
    """Raised when an arm cannot be scored."""


def split_samples(paths, split: str, limit: int | None = None):
    manifest = json.loads(
        (paths.splits / "split_manifest.json").read_text(encoding="utf-8")
    )
    images = manifest["images"] if isinstance(manifest, dict) else manifest
    names = sorted(
        entry["file_name"].split("/")[-1]
        for entry in images
        if entry["split"] == split
    )
    if not names:
        raise PredictionError(f"the frozen manifest lists no {split!r} images")
    samples = load_coco_samples(
        paths.interim / "coco_all.json", paths.hardhat_raw / "images", keep_names=names
    )
    return samples[:limit] if limit else samples


def resolve_checkpoint(seed_dir: Path) -> Path:
    """The checkpoint validation selected (K-20: never the in-memory final model)."""

    last = latest_checkpoint(seed_dir)
    if last is None:
        raise PredictionError(f"no checkpoint under {seed_dir}")
    state = json.loads((last / "trainer_state.json").read_text(encoding="utf-8"))
    recorded = state.get("best_model_checkpoint")
    if not recorded:
        return last
    # Recorded on Colab as an absolute /content/... path; only the name is portable.
    resolved = seed_dir / Path(recorded).name
    return resolved if resolved.is_dir() else last


def predict_split(
    checkpoint: Path, samples, *, processor_source: str = PROCESSOR_ID
) -> list[dict]:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(processor_source)
    model = AutoModelForObjectDetection.from_pretrained(
        str(checkpoint), dtype=torch.float32
    ).eval()
    records: list[dict] = []
    with torch.no_grad():
        for sample in samples:
            image = Image.open(sample.image_path).convert("RGB")
            encoded = processor(images=image, return_tensors="pt")
            outputs = model(**encoded)
            target = torch.tensor([[sample.height, sample.width]], dtype=torch.float32)
            processed = processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=target
            )
            records += predictions_to_coco(processed, [int(sample.image_id)])
    return records


def write_predictions(records: list[dict], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Compact separators: this file has hundreds of thousands of entries and the
    # pretty-printed form is several times larger for no benefit.
    destination.write_text(
        json.dumps(records, separators=(",", ":")), encoding="utf-8", newline="\n"
    )
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["test", "val"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, default=None, help="debug: fewer images")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--processor", default=PROCESSOR_ID)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = load_project_paths()
    out_root = args.out_root or (paths.runs / "predictions")
    runs_root = args.runs_root or paths.runs

    index: dict[str, dict] = {}
    if args.index.is_file():
        index = json.loads(args.index.read_text(encoding="utf-8"))

    failures = []
    for split in args.splits:
        samples = split_samples(paths, split, args.limit)
        print(f"\n=== {split}: {len(samples)} images")
        for arm in args.arms:
            seed_dir = runs_root / arm / f"seed_{args.seed}"
            try:
                checkpoint = resolve_checkpoint(seed_dir)
            except PredictionError as error:
                print(f"  {arm}: {error}")
                failures.append(f"{arm}/{split}")
                continue

            started = time.perf_counter()
            records = predict_split(
                checkpoint, samples, processor_source=args.processor
            )
            destination = write_predictions(
                records, out_root / f"{arm}_{split}_seed{args.seed}.json"
            )
            elapsed = time.perf_counter() - started
            index[f"{arm}/{split}/seed_{args.seed}"] = {
                "arm": arm,
                "split": split,
                "seed": args.seed,
                "checkpoint": checkpoint.name,
                "n_images": len(samples),
                "n_detections": len(records),
                "path": destination.as_posix(),
                "coordinates": "original per-image annotation space (DATA-25)",
                "score_threshold": 0.0,
            }
            print(
                f"  {arm:15s} {checkpoint.name:16s} {len(records):>7d} detections  "
                f"-> {destination.name}  ({elapsed:.0f}s)"
            )

    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.index.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nwrote {args.index}")
    if failures:
        print("FAILED:", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
