"""Render every consumed v6 audit image for the required human review."""

from __future__ import annotations

import argparse
import gc
import json
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.train_supervised_labeler import (
    FIGURE_PATH,
    REPORT_PATH,
    _aggregate,
    _build_datasets,
    _evaluation_collate,
    _predict,
    _render_audit,
    _sha256,
)


def split_review_sheet(
    source: Path,
    *,
    image_count: int = 48,
    page_size: int = 16,
) -> list[Path]:
    """Split the long four-column sheet into zoomable review pages."""

    columns = 4
    panel = 260
    caption = 30
    legend = 58
    if image_count % columns or page_size % columns:
        raise ValueError("Review counts must align to four columns")
    total_rows = image_count // columns
    rows_per_page = page_size // columns
    expected_size = (
        panel * columns,
        legend + total_rows * (panel + caption),
    )
    with Image.open(source) as handle:
        sheet = handle.convert("RGB")
    if sheet.size != expected_size:
        raise RuntimeError(
            f"Unexpected review sheet size {sheet.size}, expected {expected_size}"
        )
    outputs = []
    for page_index, start_row in enumerate(
        range(0, total_rows, rows_per_page),
        start=1,
    ):
        page_rows = min(rows_per_page, total_rows - start_row)
        page = Image.new(
            "RGB",
            (
                expected_size[0],
                legend + page_rows * (panel + caption),
            ),
            "white",
        )
        page.paste(sheet.crop((0, 0, expected_size[0], legend)), (0, 0))
        source_top = legend + start_row * (panel + caption)
        source_bottom = source_top + page_rows * (panel + caption)
        page.paste(
            sheet.crop((0, source_top, expected_size[0], source_bottom)),
            (0, legend),
        )
        output = source.with_name(
            f"{source.stem}_page_{page_index:02d}{source.suffix}"
        )
        page.save(output, optimize=True)
        outputs.append(output)
    return outputs


def main(*, split_only: bool = False) -> None:
    if split_only:
        outputs = split_review_sheet(FIGURE_PATH)
        print(
            json.dumps(
                {
                    "status": "human_review_sheet_split",
                    "source": str(FIGURE_PATH),
                    "pages": [str(path) for path in outputs],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Human-review rendering requires CUDA")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if (
        report["status"] != "supervised_labeler_audit_passed"
        or int(report["untouched_audit_images_read"]) != 48
        or int(report["validation_images_read"]) != 0
        or int(report["test_images_read"]) != 0
    ):
        raise RuntimeError("Expected one passed v6 Train-only audit")
    (
        config,
        split,
        _,
        train_images,
        image_root,
        _,
        _,
        audit,
    ) = _build_datasets()
    if len(split["untouched_audit_image_ids"]) != 48:
        raise RuntimeError("Expected exactly 48 frozen audit images")
    checkpoint = Path(report["checkpoint_path"])
    if _sha256(checkpoint / "model.safetensors") != report["checkpoint_sha256"]:
        raise RuntimeError("Passed v6 checkpoint changed")
    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to("cuda")
    loader = DataLoader(
        audit,
        batch_size=int(config["optimization"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=partial(_evaluation_collate, processor),
    )
    image_ids, truth, predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=min(config["calibration"]["score_thresholds"]),
        geometry_filter=config["postprocessing"],
    )
    best = report["best_calibration"]
    metrics = _aggregate(
        image_ids=image_ids,
        truth=truth,
        predictions=predictions,
        threshold=float(best["threshold"]),
        match_iou=float(config["calibration"]["match_iou"]),
    )
    for key in ("precision", "recall", "median_matched_iou"):
        if abs(float(metrics[key]) - float(report["audit_metrics"][key])) > 1e-12:
            raise RuntimeError(f"Review rerun changed the passed {key}")
    _render_audit(
        rows=image_ids,
        train_images=train_images,
        image_root=image_root,
        truth=truth,
        predictions=predictions,
        threshold=float(best["threshold"]),
    )
    review_pages = split_review_sheet(FIGURE_PATH)
    print(
        json.dumps(
            {
                "status": "human_review_sheet_rendered",
                "figure": str(FIGURE_PATH),
                "images": len(image_ids),
                "pages": [str(path) for path in review_pages],
                "threshold": best["threshold"],
                "metrics_reproduced": metrics,
                "validation_images_read": 0,
                "test_images_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-only", action="store_true")
    arguments = parser.parse_args()
    main(split_only=arguments.split_only)
