"""Run the approved four-case v10 FLUX diagnostic and auto-label its outputs."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.grounded_labeler import load_whole_image_config
from src.synthetic.supervised_labeler import (
    load_audited_supervised_labeler,
    load_supervised_labeler_config,
    predict_helmet_boxes,
    require_verified_audited_checkpoint,
)
from src.synthetic.whole_image import (
    diagnostic_manifest,
    generator_directory,
    load_flux2_text_to_image,
    require_generation_approval,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "whole_image_v10_seed20260808"
WORK_ROOT = PROJECT_ROOT / "outputs" / "whole_image_v10_seed20260808_in_progress"
REPORT_PATH = PROJECT_ROOT / "reports" / "whole_image_v10_diagnostic.json"
FIGURE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "whole_image_v10_diagnostic.png"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filtered_boxes(
    detections: list[tuple[float, list[float]]],
    *,
    threshold: float,
) -> list[tuple[float, list[float]]]:
    return sorted(
        [
            (float(score), [float(value) for value in box])
            for score, box in detections
            if float(score) >= threshold
        ],
        key=lambda item: -item[0],
    )


def _annotated_image(
    image: Image.Image,
    boxes: list[tuple[float, list[float]]],
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for score, box in boxes:
        draw.rectangle(tuple(box), outline=(0, 255, 255), width=3)
        draw.text((box[0] + 3, box[1] + 3), f"{score:.2f}", fill="cyan")
    return result


def _top_crop(
    image: Image.Image,
    boxes: list[tuple[float, list[float]]],
) -> Image.Image:
    if not boxes:
        result = Image.new("RGB", image.size, (40, 40, 40))
        ImageDraw.Draw(result).text((12, 12), "NO AUTO LABEL", fill="white")
        return result
    _, box = boxes[0]
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    padding = 0.25 * max(width, height)
    crop_box = (
        max(0, int(x1 - padding)),
        max(0, int(y1 - padding)),
        min(image.width, int(x2 + padding)),
        min(image.height, int(y2 + padding)),
    )
    return image.crop(crop_box)


def _render_sheet(rows: list[dict[str, Any]]) -> None:
    panel = 320
    header = 44
    columns = ("FLUX IMAGE", "AUTO LABEL (CYAN)", "TOP HELMET CROP")
    sheet = Image.new(
        "RGB",
        (panel * len(columns), (panel + header) * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        y0 = row_index * (panel + header)
        draw.text(
            (5, y0 + 4),
            f"CASE {row['case_index']:02d} | {row['scenario']}",
            fill="black",
        )
        panels = (row["image"], row["annotated"], row["crop"])
        for column_index, (label, source) in enumerate(
            zip(columns, panels, strict=True)
        ):
            x0 = column_index * panel
            draw.text((x0 + 5, y0 + 24), label, fill="black")
            fitted = source.copy()
            fitted.thumbnail((panel, panel), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (panel, panel), (90, 90, 90))
            canvas.paste(
                fitted,
                ((panel - fitted.width) // 2, (panel - fitted.height) // 2),
            )
            sheet.paste(canvas, (x0, y0 + header))
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(FIGURE_PATH, optimize=True)


def main() -> None:
    config = load_whole_image_config()
    registration = config["supervised_labeler"]
    labeler_config = load_supervised_labeler_config(
        PROJECT_ROOT / str(registration["config_path"])
    )
    labeler_report = json.loads(
        (PROJECT_ROOT / str(registration["report_path"])).read_text(
            encoding="utf-8"
        )
    )
    labeler_split = json.loads(
        (PROJECT_ROOT / str(registration["split_path"])).read_text(
            encoding="utf-8"
        )
    )
    manifest = diagnostic_manifest(config)
    require_generation_approval(
        config=config,
        labeler_report=labeler_report,
        manifest=manifest,
    )
    checkpoint_dir = require_verified_audited_checkpoint(
        config=labeler_config,
        registration=registration,
        report=labeler_report,
        split=labeler_split,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("v10 diagnostic requires an available CUDA GPU")
    if (
        OUTPUT_ROOT.exists()
        or WORK_ROOT.exists()
        or REPORT_PATH.exists()
        or FIGURE_PATH.exists()
    ):
        raise RuntimeError(
            "v10 diagnostic output or in-progress evidence already exists; "
            "reruns after model execution are forbidden"
        )

    paths = load_project_paths()
    WORK_ROOT.mkdir(parents=True)
    flux = None
    generated: list[dict[str, Any]] = []
    try:
        flux = load_flux2_text_to_image(
            model_dir=generator_directory(paths, config),
            config=config,
        )
        generator_config = config["generator"]
        for case in manifest["cases"]:
            case_index = int(case["case_index"])
            case_dir = WORK_ROOT / f"case_{case_index:02d}"
            case_dir.mkdir()
            seed = int(case["seed"])
            result = flux(
                prompt=str(case["prompt"]),
                height=int(generator_config["height"]),
                width=int(generator_config["width"]),
                num_inference_steps=int(
                    generator_config["num_inference_steps"]
                ),
                guidance_scale=float(generator_config["guidance_scale"]),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                output_type="pil",
            )
            image = result.images[0].convert("RGB")
            image_path = case_dir / "flux.png"
            image.save(image_path, optimize=True)
            generated.append(
                {
                    **case,
                    "image": image,
                    "image_path": image_path,
                    "image_sha256": _sha256(image_path),
                }
            )
            print(f"Generated FLUX case {case_index:02d}/04", flush=True)
    finally:
        if flux is not None:
            del flux
        gc.collect()
        torch.cuda.empty_cache()

    threshold = float(registration["score_threshold"])
    processor = None
    model = None
    try:
        processor, model = load_audited_supervised_labeler(
            checkpoint_dir=checkpoint_dir,
            config=labeler_config,
            device="cuda",
        )
        detected = predict_helmet_boxes(
            processor=processor,
            model=model,
            images=[record["image"] for record in generated],
            device="cuda",
            score_floor=float(registration["score_floor"]),
            geometry_filter={
                "max_relative_area": float(
                    registration["max_relative_area"]
                ),
                "max_relative_height": float(
                    registration["max_relative_height"]
                ),
            },
        )
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        torch.cuda.empty_cache()

    records = []
    sheet_rows = []
    for generated_record, raw_boxes in zip(
        generated,
        detected,
        strict=True,
    ):
        boxes = _filtered_boxes(raw_boxes, threshold=threshold)
        annotated = _annotated_image(generated_record["image"], boxes)
        crop = _top_crop(generated_record["image"], boxes)
        case_dir = generated_record["image_path"].parent
        annotated_path = case_dir / "auto_label.png"
        crop_path = case_dir / "top_crop.png"
        annotated.save(annotated_path, optimize=True)
        crop.save(crop_path, optimize=True)
        records.append(
            {
                "case_index": int(generated_record["case_index"]),
                "scenario": str(generated_record["scenario"]),
                "seed": int(generated_record["seed"]),
                "prompt": str(generated_record["prompt"]),
                "image_sha256": str(generated_record["image_sha256"]),
                "auto_labels": [
                    {"score": score, "bbox_xyxy": box}
                    for score, box in boxes
                ],
            }
        )
        sheet_rows.append(
            {
                "case_index": int(generated_record["case_index"]),
                "scenario": str(generated_record["scenario"]),
                "image": generated_record["image"],
                "annotated": annotated,
                "crop": crop,
            }
        )
    _render_sheet(sheet_rows)
    WORK_ROOT.rename(OUTPUT_ROOT)
    report = {
        "schema_version": 1,
        "status": "pending_kuotunyu_visual_review",
        "architecture": str(config["architecture"]),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "labeler_audit_status": str(labeler_report["status"]),
        "labeler_experiment_id": str(registration["experiment_id"]),
        "labeler_checkpoint_sha256": str(
            registration["checkpoint_sha256"]
        ),
        "labeler_split_manifest_sha256": str(
            registration["split_manifest_sha256"]
        ),
        "label_score_threshold": threshold,
        "label_score_floor": float(registration["score_floor"]),
        "label_geometry_filter": {
            "max_relative_area": float(
                registration["max_relative_area"]
            ),
            "max_relative_height": float(
                registration["max_relative_height"]
            ),
        },
        "cases": records,
        "cases_without_auto_label": sum(
            not record["auto_labels"] for record in records
        ),
        "validation_images_read": 0,
        "test_images_read": 0,
        "expanded_to_64": False,
        "h4_auc_computed": False,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
