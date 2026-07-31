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

from PIL import Image, ImageDraw, ImageFont

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


ZOOM = 2  # 416 px cells are too small to judge paste realism on screen
FONT = ImageFont.load_default(size=28)
SMALL_FONT = ImageFont.load_default(size=17)


def _draw_cell(
    *,
    image: Image.Image,
    record: dict[str, Any],
    draw_boxes: bool,
    index: int,
) -> Image.Image:
    # NEAREST keeps paste edges and resampling artifacts exactly as they are; a
    # smooth filter would hide the very thing the reviewer is looking for.
    canvas = image.convert("RGB").resize(
        (image.width * ZOOM, image.height * ZOOM), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(canvas)
    if draw_boxes:
        for instance in record["instances"]:
            x, y, width, height = (float(value) * ZOOM for value in instance["bbox_xywh"])
            colour = CLASS_COLORS.get(str(instance["class_name"]), (200, 200, 200))
            draw.rectangle((x, y, x + width, y + height), outline=colour, width=3)
            label = str(instance["class_name"])
            if instance.get("kind") == "existing":
                label += "*"  # carried through from the real background annotation
            draw.text((x + 3, max(0, y - 19)), label, fill=colour, font=SMALL_FONT)

    header = 44
    footer = 30
    out = Image.new("RGB", (canvas.width, canvas.height + header + footer), (16, 16, 16))
    out.paste(canvas, (0, header))
    marker = ImageDraw.Draw(out)
    # The cell number is what the reviewer quotes back, so it has to be unmissable.
    marker.rectangle((0, 0, 78, header), fill=(250, 210, 0))
    marker.text((14, 6), f"{index:02d}", fill=(0, 0, 0), font=FONT)
    detail = f"{record['sample_id']}   n_ann={len(record['instances'])}"
    if record.get("hard_negatives"):
        detail += f"   distractors={len(record['hard_negatives'])}"
    # The darkest annotated object, so a "can't see it" report can be checked
    # against what FILT-15 measured instead of argued about (ADR-013).
    lumas = [
        float(instance["object_mean_luma"])
        for instance in record["instances"]
        if "object_mean_luma" in instance
    ]
    if lumas:
        detail += f"   min_obj_luma={min(lumas):.0f}"
    strength = record.get("postfx", {}).get("low_light", {}).get("strength_scale")
    if strength is not None:
        detail += f"   lowlight={strength:.1f}"
    if "changed_pixel_ratio" in record.get("dedup", {}):
        detail += f"   changed={record['dedup']['changed_pixel_ratio']:.3f}"
    marker.text((92, 10), detail, fill=(235, 235, 235), font=SMALL_FONT)
    return out


def render(*, pool_tag: str, rows: int, cols: int, pages: int) -> dict[str, Any]:
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
        per_page = rows * cols
        chosen = items[: per_page * pages]
        if not chosen:
            continue
        # PREV-02: distractors are unannotated on purpose, so do not draw them.
        draw_boxes = scenario != "hard_negative"
        files: list[dict[str, Any]] = []
        for page in range(pages):
            batch = chosen[page * per_page : (page + 1) * per_page]
            if not batch:
                break
            probe = Image.open(pool_dir / "images" / Path(batch[0]["file_name"]).name)
            cell_w = probe.width * ZOOM
            cell_h = probe.height * ZOOM + 44 + 30
            header = 40
            sheet = Image.new("RGB", (cols * cell_w, header + rows * cell_h), (16, 16, 16))
            head_draw = ImageDraw.Draw(sheet)
            caption = (
                f"SCENARIO: {scenario}   page {page + 1}/{pages}   "
                f"cells {page * per_page + 1}-{page * per_page + len(batch)}   "
                f"({len(items)} accepted in pool)"
            )
            if not draw_boxes:
                caption += "   |  NO BOXES: distractors carry no annotation by design (ADR-004)"
            head_draw.text((8, 11), caption, fill=(255, 255, 255), font=SMALL_FONT)
            for offset, record in enumerate(batch):
                image = Image.open(pool_dir / "images" / Path(record["file_name"]).name)
                cell = _draw_cell(
                    image=image,
                    record=record,
                    draw_boxes=draw_boxes,
                    index=page * per_page + offset + 1,
                )
                sheet.paste(
                    cell,
                    ((offset % cols) * cell_w, header + (offset // cols) * cell_h),
                )
            output = paths.figures / f"preview_{scenario}_p{page + 1}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(output, optimize=True)
            files.append({"file": output.name, "sha256": _sha256_file(output)})
        written[scenario] = {
            "accepted_in_pool": len(items),
            "cells": len(chosen),
            "pages": files,
        }
    return {"grids": written, "pool_tag": pool_tag}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-tag", default="m13_pool_1x")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--pages", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            render(pool_tag=args.pool_tag, rows=args.rows, cols=args.cols, pages=args.pages),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
