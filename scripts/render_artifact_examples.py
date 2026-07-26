"""Render the H4 classifier's easiest and hardest real/pasted patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.filtering.artifact_gate import _context_crop

TILE_SIZE = 128
CAPTION_HEIGHT = 34


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tile(patch: np.ndarray, caption: str, *, synthetic: bool) -> np.ndarray:
    resized = cv2.resize(
        patch,
        (TILE_SIZE, TILE_SIZE),
        interpolation=cv2.INTER_NEAREST if min(patch.shape[:2]) < 32 else cv2.INTER_AREA,
    )
    tile = np.full((TILE_SIZE + CAPTION_HEIGHT, TILE_SIZE, 3), 248, dtype=np.uint8)
    tile[:TILE_SIZE] = resized
    color = (220, 60, 30) if synthetic else (30, 120, 40)
    cv2.rectangle(tile, (0, 0), (TILE_SIZE - 1, TILE_SIZE - 1), color, 2)
    cv2.putText(
        tile,
        caption,
        (4, TILE_SIZE + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="m11_h4_seed42")
    parser.add_argument("--per-row", type=int, default=8)
    args = parser.parse_args()

    paths = load_project_paths()
    run_dir = paths.synthetic / args.run_tag
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "filtering.yaml").read_text(encoding="utf-8")
    )
    context_scale = float(config["artifact_gate"]["patch_context_scale"])
    result = _read_json(paths.reports / "h4_artifact_gate.json")
    records = {
        record["sample_id"]: record for record in _read_jsonl(run_dir / "records.jsonl")
    }
    coco = _read_json(paths.interim / "coco_all.json")
    annotations = {int(item["id"]): item for item in coco["annotations"]}
    images = {int(item["id"]): item for item in coco["images"]}

    examples = list(
        zip(
            result["test_example_ids"],
            result["test_labels"],
            result["test_scores"],
            result["test_classes"],
            strict=True,
        )
    )
    groups = [
        ("PASTED high score", 1, True),
        ("PASTED low score", 1, False),
        ("REAL high score", 0, True),
        ("REAL low score", 0, False),
    ]
    rows: list[np.ndarray] = []
    for title, label, descending in groups:
        ranked = sorted(
            (item for item in examples if int(item[1]) == label),
            key=lambda item: float(item[2]),
            reverse=descending,
        )[: args.per_row]
        tiles: list[np.ndarray] = []
        for example_id, _, score, class_name in ranked:
            if label == 1:
                sample_id, instance_id = str(example_id).split(":", 1)
                record = records[sample_id]
                instance = next(
                    item
                    for item in record["instances"]
                    if item["instance_id"] == instance_id
                )
                image = np.asarray(
                    Image.open(run_dir / record["file_name"]).convert("RGB")
                )
                bbox = instance["bbox_xywh"]
            else:
                annotation_id = int(str(example_id).split(":")[1])
                annotation = annotations[annotation_id]
                image_record = images[int(annotation["image_id"])]
                image = np.asarray(
                    Image.open(paths.hardhat_raw / image_record["file_name"]).convert(
                        "RGB"
                    )
                )
                bbox = annotation["bbox"]
            patch = _context_crop(image, bbox, context_scale=context_scale)
            caption = f"{class_name[:3]} p={float(score):.2f}"
            tiles.append(_tile(patch, caption, synthetic=label == 1))
        row = np.concatenate(tiles, axis=1)
        label_strip = np.full((28, row.shape[1], 3), 248, dtype=np.uint8)
        cv2.putText(
            label_strip,
            title,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        rows.append(np.concatenate((label_strip, row), axis=0))
    output = np.concatenate(rows, axis=0)
    output_path = paths.figures / "h4_ranked_patches.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
