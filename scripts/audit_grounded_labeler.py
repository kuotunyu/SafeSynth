"""Calibrate and audit Grounding DINO on group-disjoint Train helmet boxes."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.grounded_labeler import (
    greedy_detection_metrics,
    labeler_directory,
    load_grounding_dino,
    load_whole_image_config,
    predict_single_phrase,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "grounded_labeler_audit.json"
MARKDOWN_PATH = PROJECT_ROOT / "reports" / "grounded_labeler_audit.md"
FIGURE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "grounded_labeler_audit.png"
)


def _stable_rank(root_seed: int, group_id: int) -> str:
    return hashlib.sha256(f"{root_seed}|{group_id}".encode()).hexdigest()


def _xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x, y, x + width, y + height


def _select_images(
    *,
    train_images: Mapping[int, Mapping[str, Any]],
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    frozen: Mapping[int, Mapping[str, Any]],
    helmet_category_id: int,
    root_seed: int,
    calibration_count: int,
    audit_count: int,
) -> tuple[list[int], list[int], dict[int, list[list[float]]]]:
    by_group: dict[int, tuple[int, float, list[list[float]]]] = {}
    for image_id in sorted(train_images):
        helmet_boxes = [
            [float(value) for value in annotation["bbox"]]
            for annotation in annotations[image_id]
            if int(annotation["category_id"]) == helmet_category_id
        ]
        if not helmet_boxes:
            continue
        group_id = int(frozen[image_id]["group_id"])
        areas = [
            box[2] * box[3]
            / (
                int(train_images[image_id]["width"])
                * int(train_images[image_id]["height"])
            )
            for box in helmet_boxes
        ]
        candidate = (image_id, float(np.median(areas)), helmet_boxes)
        previous = by_group.get(group_id)
        if previous is None or _stable_rank(root_seed, image_id) < _stable_rank(
            root_seed,
            previous[0],
        ):
            by_group[group_id] = candidate

    ordered = sorted(
        (
            (group_id, image_id, area, boxes)
            for group_id, (image_id, area, boxes) in by_group.items()
        ),
        key=lambda item: (item[2], _stable_rank(root_seed, item[0])),
    )
    total = calibration_count + audit_count
    if len(ordered) < total:
        raise RuntimeError("Not enough Train helmet groups for labeler audit")
    quartiles = np.array_split(np.asarray(ordered, dtype=object), 4)
    calibration: list[int] = []
    audit: list[int] = []
    truth: dict[int, list[list[float]]] = {}
    calibration_per_bin = calibration_count // 4
    audit_per_bin = audit_count // 4
    for bin_index, source in enumerate(quartiles):
        ranked = sorted(
            source.tolist(),
            key=lambda item: _stable_rank(
                root_seed + bin_index,
                int(item[0]),
            ),
        )
        selected = ranked[: calibration_per_bin + audit_per_bin]
        calibration_rows = selected[:calibration_per_bin]
        audit_rows = selected[calibration_per_bin:]
        calibration.extend(int(row[1]) for row in calibration_rows)
        audit.extend(int(row[1]) for row in audit_rows)
        for row in selected:
            truth[int(row[1])] = [
                list(_xywh_to_xyxy(box)) for box in row[3]
            ]
    if len(calibration) != calibration_count or len(audit) != audit_count:
        raise AssertionError("Stratified labeler sample count changed")
    calibration.sort()
    audit.sort()
    return calibration, audit, truth


def _aggregate(
    image_ids: Sequence[int],
    *,
    truth: Mapping[int, Sequence[Sequence[float]]],
    predictions: Mapping[
        int,
        Mapping[str, Sequence[tuple[float, Sequence[float]]]],
    ],
    phrase: str,
    score_threshold: float,
    match_iou: float,
) -> dict[str, float | int]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_ious: list[float] = []
    for image_id in image_ids:
        metrics = greedy_detection_metrics(
            truth[image_id],
            predictions[image_id].get(phrase, []),
            score_threshold=score_threshold,
            match_iou=match_iou,
        )
        true_positives += int(metrics["true_positives"])
        false_positives += int(metrics["false_positives"])
        false_negatives += int(metrics["false_negatives"])
        matched_ious.extend(float(value) for value in metrics["matched_ious"])
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(
            2 * precision * recall / max(precision + recall, np.finfo(float).eps)
        ),
        "median_matched_iou": (
            float(np.median(matched_ious)) if matched_ious else 0.0
        ),
    }


def _render_audit_sheet(
    image_ids: Sequence[int],
    *,
    paths: Any,
    train_images: Mapping[int, Mapping[str, Any]],
    truth: Mapping[int, Sequence[Sequence[float]]],
    predictions: Mapping[
        int,
        Mapping[str, Sequence[tuple[float, Sequence[float]]]],
    ],
    phrase: str,
    threshold: float,
) -> None:
    panel = 260
    caption = 30
    columns = 4
    selected = list(image_ids[:16])
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (panel * columns, (panel + caption) * rows), "white")
    for index, image_id in enumerate(selected):
        image = Image.open(
            paths.hardhat_raw / str(train_images[image_id]["file_name"])
        ).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in truth[image_id]:
            draw.rectangle(tuple(box), outline=(0, 255, 0), width=2)
        for score, box in predictions[image_id].get(phrase, []):
            if score < threshold:
                continue
            draw.rectangle(tuple(box), outline=(0, 255, 255), width=2)
            draw.text((box[0] + 2, box[1] + 2), f"{score:.2f}", fill="cyan")
        image = image.resize((panel, panel), Image.Resampling.LANCZOS)
        x0 = (index % columns) * panel
        y0 = (index // columns) * (panel + caption)
        sheet.paste(image, (x0, y0))
        ImageDraw.Draw(sheet).text(
            (x0 + 4, y0 + panel + 6),
            f"{image_id} | green=GT cyan=prediction",
            fill="black",
        )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(FIGURE_PATH, optimize=True)


def _write_report(payload: dict[str, Any]) -> None:
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    selected = payload["selected"]
    audit = payload["audit_metrics"]
    lines = [
        "# Grounding DINO Train-only label audit",
        "",
        f"- Status: **{payload['status']}**",
        f"- Phrase: `{selected['phrase']}`",
        f"- Score threshold: **{selected['score_threshold']:.2f}**",
        f"- Untouched audit precision@IoU0.50: **{audit['precision']:.4f}**",
        f"- Untouched audit recall@IoU0.50: **{audit['recall']:.4f}**",
        (
            "- Untouched audit median matched IoU: "
            f"**{audit['median_matched_iou']:.4f}**"
        ),
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image generation run: **no**",
        "",
        (
            "Green boxes in the contact sheet are frozen Train helmet "
            "annotations; cyan boxes are Grounding DINO predictions."
        ),
        "",
    ]
    MARKDOWN_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    paths = load_project_paths()
    config = load_whole_image_config()
    audit_config = config["labeler_audit"]
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco["categories"]
    }
    helmet_category_id = next(
        category_id
        for category_id, name in categories.items()
        if name == "helmet"
    )
    calibration_ids, audit_ids, truth = _select_images(
        train_images=train_images,
        annotations=annotations,
        frozen=frozen,
        helmet_category_id=helmet_category_id,
        root_seed=int(audit_config["root_seed"]),
        calibration_count=int(audit_config["calibration_images"]),
        audit_count=int(audit_config["untouched_audit_images"]),
    )
    selected_ids = set(calibration_ids) | set(audit_ids)
    if test_ids & selected_ids:
        raise AssertionError("Test leakage entered the labeler audit")
    if {
        int(frozen[image_id]["group_id"]) for image_id in calibration_ids
    } & {int(frozen[image_id]["group_id"]) for image_id in audit_ids}:
        raise AssertionError("Calibration and audit groups overlap")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Grounding DINO audit requires an available CUDA GPU; "
            "CPU fallback is intentionally disabled"
        )
    device = "cuda"
    processor, model = load_grounding_dino(
        model_dir=labeler_directory(paths, config),
        config=config,
        device=device,
    )
    phrases = [str(value) for value in audit_config["phrases"]]
    predictions: dict[
        int,
        dict[str, list[tuple[float, list[float]]]],
    ] = {}
    all_ids = calibration_ids + audit_ids
    batch_size = 4
    try:
        for start in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[start : start + batch_size]
            images = [
                Image.open(
                    paths.hardhat_raw
                    / str(train_images[image_id]["file_name"])
                ).convert("RGB")
                for image_id in batch_ids
            ]
            for image_id in batch_ids:
                predictions[image_id] = {}
            for phrase in phrases:
                detected = predict_single_phrase(
                    processor=processor,
                    model=model,
                    images=images,
                    phrase=phrase,
                    device=device,
                    score_floor=0.05,
                    text_threshold=float(audit_config["text_threshold"]),
                )
                for image_id, boxes in zip(
                    batch_ids,
                    detected,
                    strict=True,
                ):
                    predictions[image_id][phrase] = boxes
            print(
                f"Processed {min(start + batch_size, len(all_ids))}/"
                f"{len(all_ids)} Train images",
                flush=True,
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    for phrase in phrases:
        for threshold in audit_config["score_thresholds"]:
            metrics = _aggregate(
                calibration_ids,
                truth=truth,
                predictions=predictions,
                phrase=phrase,
                score_threshold=float(threshold),
                match_iou=float(audit_config["match_iou"]),
            )
            rows.append(
                {
                    "phrase": phrase,
                    "score_threshold": float(threshold),
                    **metrics,
                }
            )
    precision_floor = float(audit_config["calibration_min_precision"])
    eligible = [row for row in rows if row["precision"] >= precision_floor]
    ranked = eligible if eligible else rows
    ranked.sort(
        key=lambda row: (
            -float(row["f1"]),
            -float(row["precision"]),
            -float(row["recall"]),
            -float(row["median_matched_iou"]),
            -float(row["score_threshold"]),
            str(row["phrase"]),
        )
    )
    selected = ranked[0]
    audit_metrics = _aggregate(
        audit_ids,
        truth=truth,
        predictions=predictions,
        phrase=str(selected["phrase"]),
        score_threshold=float(selected["score_threshold"]),
        match_iou=float(audit_config["match_iou"]),
    )
    checks = {
        "calibration_precision": float(selected["precision"])
        >= float(audit_config["calibration_min_precision"]),
        "audit_precision": float(audit_metrics["precision"])
        >= float(audit_config["audit_min_precision"]),
        "audit_recall": float(audit_metrics["recall"])
        >= float(audit_config["audit_min_recall"]),
        "audit_median_matched_iou": float(
            audit_metrics["median_matched_iou"]
        )
        >= float(audit_config["audit_min_median_matched_iou"]),
    }
    status = (
        "labeler_audit_passed"
        if all(checks.values())
        else "labeler_audit_failed"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "model": config["labeler"],
        "scope": "two frozen group-disjoint Train subsets",
        "calibration_image_ids": calibration_ids,
        "untouched_audit_image_ids": audit_ids,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
        "selected": selected,
        "audit_metrics": audit_metrics,
        "checks": checks,
        "calibration_grid": rows,
    }
    _render_audit_sheet(
        audit_ids,
        paths=paths,
        train_images=train_images,
        truth=truth,
        predictions=predictions,
        phrase=str(selected["phrase"]),
        threshold=float(selected["score_threshold"]),
    )
    _write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
