"""Render one preview grid per target scenario for human spot-checking (PREV-01..05).

This is the one place in Phase 1 where only a person can judge the result. The
grids exist so the repository owner can answer questions no assertion can:
does a shrunk helmet read as a distant worker or as a sticker, does a swapped
bare head sit on a body, is the occlusion plausible, is the crowd ordered
front-to-back correctly.

PREV-02 is the exception that is easy to mistake for a bug: the hard-negative
grid is drawn WITHOUT boxes on the distractors, because they carry no annotation
by construction (ADR-004). A caption says so on the image itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from src.data.paths import load_project_paths

CLASS_COLORS = {
    "helmet": (255, 196, 0),
    "head": (230, 70, 70),
    "person": (30, 170, 240),
}
HARD_NEGATIVE_COLOR = (255, 0, 255)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_rank(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode()).hexdigest()


def _draw_cell(
    *,
    image: Image.Image,
    record: dict[str, Any],
    draw_boxes: bool,
) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    if draw_boxes:
        for instance in record["instances"]:
            x, y, width, height = (float(value) for value in instance["bbox_xywh"])
            colour = CLASS_COLORS.get(str(instance["class_name"]), (200, 200, 200))
            draw.rectangle((x, y, x + width, y + height), outline=colour, width=2)
            label = str(instance["class_name"])
            if instance.get("kind") == "existing":
                label += "*"  # carried through from the real background annotation
            draw.text((x + 2, max(0, y - 10)), label, fill=colour)
    footer = 26
    out = Image.new("RGB", (canvas.width, canvas.height + footer), (16, 16, 16))
    out.paste(canvas, (0, 0))
    footer_draw = ImageDraw.Draw(out)
    scores = record.get("scores", {})
    detail = f"{record['sample_id']}  n_ann={len(record['instances'])}"
    if record.get("hard_negatives"):
        detail += f"  distractors={len(record['hard_negatives'])}"
    if "changed_pixel_ratio" in record.get("dedup", {}):
        detail += f"  changed={record['dedup']['changed_pixel_ratio']:.3f}"
    if scores:
        detail += f"  {scores}"[:40]
    footer_draw.text((4, canvas.height + 7), detail, fill=(235, 235, 235))
    return out


def render(*, pool_tag: str, rows: int, cols: int) -> dict[str, Any]:
    paths = load_project_paths()
    pool_dir = paths.synthetic / pool_tag
    records = [
        json.loads(line)
        for line in (pool_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted = [record for record in records if record["passed"]]
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for record in accepted:
        by_scenario.setdefault(str(record["scenario"]), []).append(record)

    written: dict[str, Any] = {}
    for scenario, items in sorted(by_scenario.items()):
        items.sort(key=lambda item: _stable_rank(str(item["sample_id"])))
        chosen = items[: rows * cols]
        if not chosen:
            continue
        # PREV-02: distractors are unannotated on purpose, so do not draw them.
        draw_boxes = scenario != "hard_negative"
        sample = Image.open(pool_dir / "images" / Path(chosen[0]["file_name"]).name)
        cell_w, cell_h = sample.width, sample.height + 26
        header = 30
        sheet = Image.new("RGB", (cols * cell_w, header + rows * cell_h), (16, 16, 16))
        head_draw = ImageDraw.Draw(sheet)
        caption = f"SCENARIO: {scenario}   ({len(items)} accepted in pool)"
        if not draw_boxes:
            caption += "   -- NO BOXES DRAWN: distractors carry no annotation by design (ADR-004)"
        head_draw.text((6, 9), caption, fill=(255, 255, 255))
        for index, record in enumerate(chosen):
            image = Image.open(pool_dir / "images" / Path(record["file_name"]).name)
            cell = _draw_cell(image=image, record=record, draw_boxes=draw_boxes)
            sheet.paste(cell, ((index % cols) * cell_w, header + (index // cols) * cell_h))
        output = paths.figures / f"preview_{scenario}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, optimize=True)
        written[scenario] = {
            "accepted_in_pool": len(items),
            "cells": len(chosen),
            "file": output.name,
            "sha256": _sha256_file(output),
        }
    return {"grids": written, "pool_tag": pool_tag}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-tag", default="m13_pool_1x")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            render(pool_tag=args.pool_tag, rows=args.rows, cols=args.cols),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
