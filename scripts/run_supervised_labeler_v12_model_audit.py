"""Run the preregistered v11 checkpoint on the frozen v12 audit once."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.prepare_supervised_labeler_v12_gt_review import CONFIG_PATH
from scripts.record_supervised_labeler_v12_gt_review import AUDIT_PATH
from scripts.train_supervised_labeler import (
    HelmetDataset,
    _aggregate,
    _evaluation_collate,
    _predict,
)
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.whole_image import canonical_mapping_sha256

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v12_model_audit.json"
)
EVIDENCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v12_model_audit_evidence.json"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "supervised_labeler_v12_model_audit"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    windows_root = os.environ.get("WINDIR")
    if windows_root:
        name = "msjhbd.ttc" if bold else "msjh.ttc"
        path = Path(windows_root) / "Fonts" / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _verified_audit() -> dict[str, Any]:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    canonical = dict(audit)
    embedded_sha = str(canonical.pop("manifest_sha256", ""))
    if (
        canonical_mapping_sha256(canonical) != embedded_sha
        or audit.get("status")
        != "v12_adjudicated_audit_frozen_before_model_inference"
        or audit.get("model_inference_run") is not False
        or int(audit.get("sealed_reserve_pixels_read", -1)) != 0
        or int(audit.get("validation_images_read", -1)) != 0
        or int(audit.get("test_images_read", -1)) != 0
    ):
        raise RuntimeError("Frozen v12 adjudicated audit changed")
    return audit


def _verified_registration(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    registration = dict(config["model_audit_registration"])
    report_path = PROJECT_ROOT / registration["source_training_report"]
    if _sha256(report_path) != registration["source_training_report_sha256"]:
        raise RuntimeError("Registered source training report changed")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checkpoint_dir = Path(str(registration["checkpoint_path"]))
    checkpoint_path = checkpoint_dir / "model.safetensors"
    if (
        registration["status"] != "frozen_before_v12_model_inference"
        or registration["source_experiment"] != "supervised_labeler_v11"
        or report.get("status") != "supervised_labeler_audit_passed"
        or report.get("checkpoint_sha256")
        != registration["checkpoint_sha256"]
        or not checkpoint_path.is_file()
        or _sha256(checkpoint_path) != registration["checkpoint_sha256"]
        or float(report["best_calibration"]["threshold"])
        != float(registration["score_threshold"])
        or {
            key: float(report["postprocessing"][key])
            for key in registration["postprocessing"]
        }
        != {
            key: float(value)
            for key, value in registration["postprocessing"].items()
        }
        or registration["whole_image_generation_run"] is not False
    ):
        raise RuntimeError("v12 model audit registration changed")
    return registration, report, checkpoint_dir


def _draw_boxes(
    panel: Image.Image,
    boxes: Sequence[Sequence[float]],
    *,
    source_size: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(panel)
    scale_x = panel.width / source_size[0]
    scale_y = panel.height / source_size[1]
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box)
        draw.rectangle(
            (
                round(x1 * scale_x),
                round(y1 * scale_y),
                round(x2 * scale_x),
                round(y2 * scale_y),
            ),
            outline=color,
            width=2,
        )


def _panel(
    image: Image.Image,
    *,
    truth: Sequence[Sequence[float]],
    model: Sequence[Sequence[float]],
    show_truth: bool,
    show_model: bool,
    size: int,
) -> Image.Image:
    panel = image.resize((size, size), Image.Resampling.LANCZOS)
    if show_truth:
        _draw_boxes(
            panel,
            truth,
            source_size=image.size,
            color=(0, 230, 0),
        )
    if show_model:
        _draw_boxes(
            panel,
            model,
            source_size=image.size,
            color=(255, 0, 255),
        )
    return panel


def _render_pages(
    *,
    dataset: HelmetDataset,
    audit_cases: Sequence[Mapping[str, Any]],
    predictions: Mapping[int, Sequence[tuple[float, Sequence[float]]]],
    threshold: float,
) -> list[dict[str, Any]]:
    panel_size = 290
    panel_header = 32
    case_caption = 38
    header_height = 150
    cases_per_page = 16
    cases_per_row = 2
    case_width = panel_size * 3
    case_height = panel_header + panel_size + case_caption
    page_width = case_width * cases_per_row
    page_height = header_height + case_height * 8
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pages = []
    for page_index in range(3):
        sheet = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(sheet)
        first_cell = page_index * cases_per_page + 1
        last_cell = first_cell + cases_per_page - 1
        draw.text(
            (14, 10),
            f"V12 模型盲測｜第 {page_index + 1}/3 頁｜"
            f"編號 {first_cell:02d}–{last_cell:02d}",
            fill="black",
            font=_font(27, bold=True),
        )
        draw.text(
            (14, 50),
            "每格從左到右：GT 綠框（正確答案）｜"
            "模型洋紅框（模型預測）｜疊圖（兩者比較）",
            fill="black",
            font=_font(20),
        )
        draw.text(
            (14, 84),
            "請只評模型：漏掉戴帽頭部、框到背景／人臉／"
            "未佩戴安全帽，或把多頂帽子合成一框，都算問題。",
            fill="black",
            font=_font(19),
        )
        draw.text(
            (14, 116),
            f"固定分數門檻 {threshold:.2f}；"
            "此頁在 GT 凍結後才產生，未使用 Val/Test。",
            fill=(70, 70, 70),
            font=_font(17),
        )
        for local_index in range(cases_per_page):
            global_index = page_index * cases_per_page + local_index
            case = audit_cases[global_index]
            item = dataset[global_index]
            image_id = int(case["image_id"])
            if int(item["image_id"]) != image_id:
                raise RuntimeError("v12 model review order changed")
            truth = [
                [float(value) for value in box]
                for box in case["truth_boxes"]
            ]
            model_boxes = [
                [float(value) for value in box]
                for score, box in predictions[image_id]
                if float(score) >= threshold
            ]
            column = local_index % cases_per_row
            row = local_index // cases_per_row
            x0 = column * case_width
            y0 = header_height + row * case_height
            headers = ("GT 綠框", "模型 洋紅框", "疊圖")
            modes = ((True, False), (False, True), (True, True))
            for panel_index, (show_truth, show_model) in enumerate(modes):
                panel_x = x0 + panel_index * panel_size
                draw.text(
                    (panel_x + 6, y0 + 5),
                    headers[panel_index],
                    fill="black",
                    font=_font(16, bold=True),
                )
                rendered = _panel(
                    item["image"],
                    truth=truth,
                    model=model_boxes,
                    show_truth=show_truth,
                    show_model=show_model,
                    size=panel_size,
                )
                sheet.paste(rendered, (panel_x, y0 + panel_header))
            draw.text(
                (x0 + 8, y0 + panel_header + panel_size + 7),
                f"{int(case['audit_cell']):02d}｜"
                f"Train image {image_id}｜"
                f"GT {len(truth)}｜模型 {len(model_boxes)}",
                fill="black",
                font=_font(16),
            )
        path = FIGURE_DIR / f"page_{page_index + 1:02d}.png"
        sheet.save(path, optimize=True)
        pages.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
                "cells": [first_cell, last_cell],
            }
        )
    return pages


def main() -> None:
    if REPORT_PATH.exists() or EVIDENCE_PATH.exists() or FIGURE_DIR.exists():
        raise RuntimeError("v12 model audit evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("v12 model audit requires CUDA")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registration, source_report, checkpoint_dir = _verified_registration(
        config
    )
    audit = _verified_audit()
    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    selected = audit["selected_cases"]
    image_ids = [int(row["image_id"]) for row in selected]
    if (
        test_ids & set(image_ids)
        or len(image_ids) != 48
        or any(
            int(frozen[image_id]["group_id"]) != int(row["group_id"])
            for image_id, row in zip(image_ids, selected, strict=True)
        )
    ):
        raise RuntimeError("Val/Test or group leakage entered v12 model audit")
    helmet_category_id = next(
        int(row["id"])
        for row in coco["categories"]
        if str(row["name"]) == "helmet"
    )
    dataset = HelmetDataset(
        image_ids=image_ids,
        images=train_images,
        annotations=annotations,
        image_root=paths.hardhat_raw,
        helmet_category_id=helmet_category_id,
        input_normalization=config["input_normalization"],
    )
    processor = AutoImageProcessor.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    ).to("cuda")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: _evaluation_collate(processor, batch),
    )
    threshold = float(registration["score_threshold"])
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    predicted_ids, truth, predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=threshold,
        geometry_filter=registration["postprocessing"],
    )
    if predicted_ids != image_ids:
        raise RuntimeError("v12 model audit prediction order changed")
    for row in selected:
        image_id = int(row["image_id"])
        expected = [
            [float(value) for value in box] for box in row["truth_boxes"]
        ]
        if truth[image_id] != expected:
            raise RuntimeError("v12 GT changed between freeze and inference")
    metrics = _aggregate(
        image_ids=image_ids,
        truth=truth,
        predictions=predictions,
        threshold=threshold,
        match_iou=float(registration["match_iou"]),
    )
    gates = registration["numeric_gates"]
    checks = {
        "precision": float(metrics["precision"])
        >= float(gates["min_precision"]),
        "recall": float(metrics["recall"]) >= float(gates["min_recall"]),
        "median_matched_iou": float(metrics["median_matched_iou"])
        >= float(gates["min_median_matched_iou"]),
    }
    evidence = {
        "schema_version": 1,
        "status": "v12_frozen_model_audit_evidence",
        "experiment_id": "supervised_labeler_v12",
        "audit_manifest_sha256": str(audit["manifest_sha256"]),
        "source_experiment": str(registration["source_experiment"]),
        "checkpoint_sha256": str(registration["checkpoint_sha256"]),
        "score_threshold": threshold,
        "postprocessing": registration["postprocessing"],
        "cases": [
            {
                "cell": int(row["audit_cell"]),
                "primary_cell": int(row["primary_cell"]),
                "image_id": int(row["image_id"]),
                "group_id": int(row["group_id"]),
                "stratum": str(row["stratum"]),
                "truth_boxes": row["truth_boxes"],
                "model_predictions": [
                    {
                        "score": float(score),
                        "box": [float(value) for value in box],
                    }
                    for score, box in predictions[int(row["image_id"])]
                    if float(score) >= threshold
                ],
            }
            for row in selected
        ],
        "model_images_read": len(image_ids),
        "sealed_reserve_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    evidence["evidence_sha256"] = canonical_mapping_sha256(evidence)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    pages = _render_pages(
        dataset=dataset,
        audit_cases=selected,
        predictions=predictions,
        threshold=threshold,
    )
    report = {
        "schema_version": 1,
        "status": (
            "v12_numeric_audit_passed_owner_review_pending"
            if all(checks.values())
            else "v12_numeric_audit_failed"
        ),
        "experiment_id": "supervised_labeler_v12",
        "source_experiment": str(registration["source_experiment"]),
        "source_training_report_sha256": str(
            registration["source_training_report_sha256"]
        ),
        "source_numeric_audit_status": str(source_report["status"]),
        "checkpoint_path": str(checkpoint_dir),
        "checkpoint_sha256": str(registration["checkpoint_sha256"]),
        "audit_manifest_sha256": str(audit["manifest_sha256"]),
        "score_threshold": threshold,
        "match_iou": float(registration["match_iou"]),
        "postprocessing": registration["postprocessing"],
        "numeric_gates": gates,
        "metrics": metrics,
        "checks": checks,
        "model_images_read": len(image_ids),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.time() - started,
        "evidence_path": str(
            EVIDENCE_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "evidence_file_sha256": _sha256(EVIDENCE_PATH),
        "evidence_sha256": str(evidence["evidence_sha256"]),
        "pages": pages,
        "owner_review_required": all(checks.values()),
        "owner_problem_count_required": 0,
        "sealed_reserve_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    report["report_sha256"] = canonical_mapping_sha256(report)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": metrics,
                "checks": checks,
                "model_images_read": report["model_images_read"],
                "peak_vram_gib": report["peak_vram_gib"],
                "elapsed_seconds": report["elapsed_seconds"],
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
