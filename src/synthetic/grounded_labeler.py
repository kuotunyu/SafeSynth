"""Pinned Grounding DINO loading and box metrics for synthetic auto-labeling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from PIL import Image

from src.data.paths import PROJECT_ROOT, ProjectPaths

CONFIG_PATH = PROJECT_ROOT / "configs" / "whole_image_generation.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_whole_image_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the preregistered whole-image configuration."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    labeler = config["labeler"]
    if labeler["license"] != "apache-2.0":
        raise RuntimeError("The automatic labeler must remain Apache-2.0")
    if labeler["local_files_only"] is not True:
        raise RuntimeError("The automatic labeler runtime must remain local-only")
    expected = sum(int(value) for value in labeler["allow_files"].values())
    if expected != int(labeler["required_download_bytes"]):
        raise RuntimeError("Grounding DINO registered file sizes disagree")
    supervised = config["supervised_labeler"]
    if (
        supervised["experiment_id"] != "supervised_labeler_v6"
        or supervised["architecture"] != "rtdetr_v2_r50vd_helmet_only"
        or int(supervised["validation_images_read"]) != 0
        or int(supervised["test_images_read"]) != 0
        or int(supervised["audit_images"]) != 48
        or not 0 < float(supervised["score_floor"])
        <= float(supervised["score_threshold"])
        or not 0 < float(supervised["max_relative_area"]) <= 1
        or not 0 < float(supervised["max_relative_height"]) <= 1
    ):
        raise RuntimeError("Supervised v6 labeler registration changed")
    labeler_review = supervised["human_review"]
    page_records = labeler_review.get("pages", [])
    if (
        labeler_review.get("required_reviewer") != "kuotunyu"
        or len(str(labeler_review.get("figure_sha256", ""))) != 64
        or len(page_records) != 3
        or any(
            not record.get("path")
            or len(str(record.get("sha256", ""))) != 64
            for record in page_records
        )
        or not labeler_review.get("evidence_path")
    ):
        raise RuntimeError("Supervised v6 human-review registration changed")
    gate = config["generation_gate"]
    if gate.get("allowed") not in (True, False):
        raise RuntimeError("Whole-image generation gate must be boolean")
    if gate.get("required_reviewer") != "kuotunyu":
        raise RuntimeError("Whole-image generation reviewer changed")
    if gate["allowed"] is True:
        review = config["diagnostic"]["input_review"]
        if (
            review.get("required_reviewer") != "kuotunyu"
            or review.get("status") != "approved_by_kuotunyu"
            or not review.get("approved_manifest_sha256")
            or labeler_review.get("required_reviewer") != "kuotunyu"
            or labeler_review.get("status") != "approved_by_kuotunyu"
        ):
            raise RuntimeError(
                "Whole-image generation gate cannot open without owner approval"
            )
    output_review = config["diagnostic"]["output_review"]
    scaleup_gate = config["scaleup_gate"]
    if (
        output_review.get("required_reviewer") != "kuotunyu"
        or int(output_review.get("required_problem_count", -1)) != 0
        or not output_review.get("evidence_path")
        or scaleup_gate.get("allowed") not in (True, False)
        or scaleup_gate.get("required_reviewer") != "kuotunyu"
    ):
        raise RuntimeError("Whole-image v10 output-review gate changed")
    if scaleup_gate["allowed"] is True and output_review.get(
        "status"
    ) != "approved_by_kuotunyu":
        raise RuntimeError(
            "Whole-image scale-up cannot open without owner output review"
        )
    return config


def labeler_directory(paths: ProjectPaths, config: dict[str, Any]) -> Path:
    """Return the project-isolated Grounding DINO directory."""

    labeler = config["labeler"]
    slug = str(labeler["repo_id"]).replace("/", "--")
    return paths.cache / "models" / slug / str(labeler["revision"])


def require_verified_labeler(
    model_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Hard fail unless the labeler manifest matches the frozen registration."""

    manifest_path = model_dir / "SAFESYNTH_MODEL_MANIFEST.json"
    if not manifest_path.exists():
        raise RuntimeError("Pinned automatic labeler is not downloaded")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config["labeler"]
    if (
        manifest.get("repo_id") != expected["repo_id"]
        or manifest.get("revision") != expected["revision"]
        or manifest.get("license") != expected["license"]
        or int(manifest.get("download_bytes", -1))
        != int(expected["required_download_bytes"])
    ):
        raise RuntimeError("Automatic labeler manifest does not match registration")
    registered_files = {
        str(record["path"]): record for record in manifest.get("files", [])
    }
    if set(registered_files) != set(expected["allow_files"]):
        raise RuntimeError("Automatic labeler file list changed")
    for name, expected_size in expected["allow_files"].items():
        path = model_dir / str(name)
        record = registered_files[str(name)]
        if (
            not path.is_file()
            or path.stat().st_size != int(expected_size)
            or int(record.get("bytes", -1)) != int(expected_size)
            or _sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(
                f"Automatic labeler file failed integrity check: {name}"
            )
    return manifest


def load_grounding_dino(
    *,
    model_dir: Path,
    config: dict[str, Any],
    device: str,
) -> tuple[Any, Any]:
    """Load the pinned zero-shot detector without network access."""

    require_verified_labeler(model_dir, config)
    if transformers.__version__ != str(
        config["labeler"]["transformers_version"]
    ):
        raise RuntimeError(
            "Expected transformers "
            f"{config['labeler']['transformers_version']}, "
            f"got {transformers.__version__}"
        )
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    return processor, model


def predict_single_phrase(
    *,
    processor: Any,
    model: Any,
    images: Sequence[Image.Image],
    phrase: str,
    device: str,
    score_floor: float,
    text_threshold: float,
) -> list[list[tuple[float, list[float]]]]:
    """Predict one phrase independently for each image in a batch."""

    if not images:
        return []
    normalized_phrase = " ".join(str(phrase).split())
    if not normalized_phrase:
        raise ValueError("Grounding phrase cannot be empty")
    inputs = processor(
        images=list(images),
        text=[[normalized_phrase] for _ in images],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=float(score_floor),
        text_threshold=float(text_threshold),
        target_sizes=[image.size[::-1] for image in images],
    )
    detections: list[list[tuple[float, list[float]]]] = []
    for result in results:
        detections.append(
            [
                (
                    float(score.item()),
                    [float(value) for value in box.tolist()],
                )
                for score, box in zip(
                    result["scores"],
                    result["boxes"],
                    strict=True,
                )
            ]
        )
    return detections


def box_iou_xyxy(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    """Return intersection-over-union for two xyxy boxes."""

    first_x1, first_y1, first_x2, first_y2 = (
        float(value) for value in first
    )
    second_x1, second_y1, second_x2, second_y2 = (
        float(value) for value in second
    )
    width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = width * height
    first_area = max(0.0, first_x2 - first_x1) * max(
        0.0, first_y2 - first_y1
    )
    second_area = max(0.0, second_x2 - second_x1) * max(
        0.0, second_y2 - second_y1
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_detection_metrics(
    ground_truth_boxes: Sequence[Sequence[float]],
    predictions: Sequence[tuple[float, Sequence[float]]],
    *,
    score_threshold: float,
    match_iou: float,
) -> dict[str, float | int | list[float]]:
    """Greedily match score-sorted predictions to ground-truth boxes."""

    remaining = set(range(len(ground_truth_boxes)))
    matched_ious: list[float] = []
    false_positives = 0
    filtered = sorted(
        (
            (float(score), tuple(float(value) for value in box))
            for score, box in predictions
            if float(score) >= score_threshold
        ),
        key=lambda item: -item[0],
    )
    for _, predicted_box in filtered:
        if not remaining:
            false_positives += 1
            continue
        best_index, best_iou = max(
            (
                (index, box_iou_xyxy(predicted_box, ground_truth_boxes[index]))
                for index in remaining
            ),
            key=lambda item: item[1],
        )
        if best_iou >= match_iou:
            remaining.remove(best_index)
            matched_ious.append(float(best_iou))
        else:
            false_positives += 1
    true_positives = len(matched_ious)
    false_negatives = len(remaining)
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
        "matched_ious": matched_ious,
    }
