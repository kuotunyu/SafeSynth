"""Render frozen v6 review evidence with GT and model boxes separated."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from scripts.train_supervised_labeler import FIGURE_PATH, _build_datasets
from src.data.paths import PROJECT_ROOT

OUTPUT_STEM = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "supervised_labeler_v6_audit_separated"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_model_marks(
    *,
    frozen_panel: Image.Image,
    original_panel: Image.Image,
) -> Image.Image:
    """Extract cyan model box lines without the nearby score glyphs."""

    frozen = np.asarray(frozen_panel.convert("RGB"), dtype=np.int16)
    original = np.asarray(original_panel.convert("RGB"), dtype=np.int16)
    if frozen.shape != original.shape:
        raise ValueError("Frozen and original panels must have the same size")
    changed = np.abs(frozen - original).sum(axis=2) >= 24
    cyan_like = (
        (frozen[..., 0] <= 135)
        & (frozen[..., 1] >= 135)
        & (frozen[..., 2] >= 135)
        & ((frozen[..., 1] + frozen[..., 2] - 2 * frozen[..., 0]) >= 160)
    )
    raw = np.where(changed & cyan_like, 1, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw,
        connectivity=8,
    )
    boxes_only = np.zeros_like(raw)
    panel_height, panel_width = raw.shape
    for component_id in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[component_id]
        )
        component = (
            labels[y : y + height, x : x + width] == component_id
        )
        side_coverages = (
            float(component[0, :].mean()),
            float(component[-1, :].mean()),
            float(component[:, 0].mean()),
            float(component[:, -1].mean()),
        )
        touches_panel_edge = (
            x == 0
            or y == 0
            or x + width == panel_width
            or y + height == panel_height
        )
        normal_box = width >= 7 and height >= 7
        compact_rectangle = (
            min(width, height) >= 4
            and max(width, height) >= 5
            and min(side_coverages) >= 0.75
        )
        clipped_box_line = (
            touches_panel_edge
            and max(width, height) >= 7
            and area >= max(width, height)
        )
        if normal_box or compact_rectangle or clipped_box_line:
            cv2.rectangle(
                boxes_only,
                (x, y),
                (x + width - 1, y + height - 1),
                color=255,
                thickness=1,
            )
    return Image.fromarray(boxes_only, mode="L")


def _draw_truth(
    image: Image.Image,
    truth: list[list[float]],
    *,
    source_size: tuple[int, int],
    width: int,
) -> None:
    draw = ImageDraw.Draw(image)
    scale_x = image.width / source_size[0]
    scale_y = image.height / source_size[1]
    for box in truth:
        x1, y1, x2, y2 = (float(value) for value in box)
        draw.rectangle(
            (
                round(x1 * scale_x),
                round(y1 * scale_y),
                round(x2 * scale_x),
                round(y2 * scale_y),
            ),
            outline=(0, 255, 0),
            width=width,
        )


def _paint_model_marks(image: Image.Image, mask: Image.Image) -> None:
    magenta = Image.new("RGB", image.size, (255, 0, 255))
    image.paste(magenta, mask=mask)


def _render_case(
    *,
    original: Image.Image,
    truth: list[list[float]],
    model_mask: Image.Image,
    panel_size: int,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    base = original.resize(
        (panel_size, panel_size),
        Image.Resampling.LANCZOS,
    )
    mask = model_mask.resize(
        (panel_size, panel_size),
        Image.Resampling.NEAREST,
    )
    truth_only = base.copy()
    _draw_truth(
        truth_only,
        truth,
        source_size=original.size,
        width=2,
    )
    model_only = base.copy()
    _paint_model_marks(model_only, mask)
    overlay = base.copy()
    _draw_truth(
        overlay,
        truth,
        source_size=original.size,
        width=2,
    )
    _paint_model_marks(overlay, mask)
    return truth_only, model_only, overlay


def render_separated_pages() -> list[dict[str, Any]]:
    """Render three deterministic 16-case pages without model inference."""

    if not FIGURE_PATH.is_file():
        raise RuntimeError("Frozen v6 review sheet is missing")
    (
        _,
        split,
        _,
        train_images,
        _,
        _,
        _,
        audit,
    ) = _build_datasets()
    image_ids = [int(value) for value in split["untouched_audit_image_ids"]]
    if len(image_ids) != 48 or len(audit) != 48:
        raise RuntimeError("Expected the frozen 48-image v6 audit")

    frozen_panel_size = 260
    frozen_caption = 30
    frozen_legend = 58
    frozen_columns = 4
    with Image.open(FIGURE_PATH) as handle:
        frozen_sheet = handle.convert("RGB").copy()

    panel = 170
    cases_per_page = 16
    cases_per_row = 2
    group_width = panel * 3
    group_height = panel + 48
    page_header = 62
    page_width = group_width * cases_per_row
    page_height = page_header + 8 * group_height
    outputs = []
    for page_index in range(3):
        page = Image.new("RGB", (page_width, page_height), "white")
        page_draw = ImageDraw.Draw(page)
        page_draw.text(
            (8, 7),
            "V6 REVIEW | GREEN ONLY = DATASET GT | "
            "MAGENTA ONLY = MODEL | OVERLAY = BOTH",
            fill="black",
        )
        page_draw.text(
            (8, 30),
            "Judge model boxes only. Green without magenta = possible miss; "
            "magenta on background = false positive.",
            fill="black",
        )
        start = page_index * cases_per_page
        for local_index in range(cases_per_page):
            canonical_index = start + local_index
            item = audit[canonical_index]
            image_id = int(item["image_id"])
            if image_id != image_ids[canonical_index]:
                raise RuntimeError("Frozen audit order changed")
            original = item["image"].convert("RGB")
            frozen_x = (canonical_index % frozen_columns) * frozen_panel_size
            frozen_y = frozen_legend + (
                canonical_index // frozen_columns
            ) * (frozen_panel_size + frozen_caption)
            frozen_panel = frozen_sheet.crop(
                (
                    frozen_x,
                    frozen_y,
                    frozen_x + frozen_panel_size,
                    frozen_y + frozen_panel_size,
                )
            )
            original_frozen_size = original.resize(
                (frozen_panel_size, frozen_panel_size),
                Image.Resampling.LANCZOS,
            )
            model_mask = extract_model_marks(
                frozen_panel=frozen_panel,
                original_panel=original_frozen_size,
            )
            truth_only, model_only, overlay = _render_case(
                original=original,
                truth=item["truth"],
                model_mask=model_mask,
                panel_size=panel,
            )
            group_column = local_index % cases_per_row
            group_row = local_index // cases_per_row
            x0 = group_column * group_width
            y0 = page_header + group_row * group_height
            labels = ("GT GREEN", "MODEL MAGENTA", "OVERLAY")
            panels = (truth_only, model_only, overlay)
            for panel_index, (label, source) in enumerate(
                zip(labels, panels, strict=True)
            ):
                panel_x = x0 + panel_index * panel
                page_draw.text((panel_x + 4, y0 + 2), label, fill="black")
                page.paste(source, (panel_x, y0 + 20))
            page_draw.text(
                (x0 + 4, y0 + panel + 25),
                f"{canonical_index + 1:02d} | Train image "
                f"{train_images[image_id]['id']}",
                fill="black",
            )
        output_path = OUTPUT_STEM.with_name(
            f"{OUTPUT_STEM.name}_page_{page_index + 1:02d}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        page.save(output_path, optimize=True)
        outputs.append(
            {
                "path": str(output_path.relative_to(PROJECT_ROOT)).replace(
                    "\\",
                    "/",
                ),
                "sha256": _sha256(output_path),
            }
        )
    return outputs


def main() -> None:
    outputs = render_separated_pages()
    print(
        json.dumps(
            {
                "status": "separated_review_pages_rendered",
                "source": str(FIGURE_PATH),
                "pages": outputs,
                "model_inference_run": False,
                "validation_images_read": 0,
                "test_images_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
