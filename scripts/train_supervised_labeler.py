"""Smoke-test or train the preregistered Train-only RT-DETRv2 labeler."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    get_cosine_schedule_with_warmup,
)

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import (
    _load_context,
    normalize_reflected_padding,
    reflected_padding_guard,
)
from src.synthetic.grounded_labeler import greedy_detection_metrics
from src.synthetic.supervised_labeler import (
    CONFIG_PATH,
    SPLIT_PATH,
    filter_prediction_geometry,
    load_supervised_labeler_config,
    model_directory,
    require_verified_model,
    supervised_sampling_weights,
)

EXPERIMENT_STEM = CONFIG_PATH.stem
REPORT_PATH = PROJECT_ROOT / "reports" / f"{EXPERIMENT_STEM}_training.json"
SMOKE_REPORT_PATH = PROJECT_ROOT / "reports" / f"{EXPERIMENT_STEM}_smoke.json"
MARKDOWN_PATH = PROJECT_ROOT / "reports" / f"{EXPERIMENT_STEM}_training.md"
FIGURE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / f"{EXPERIMENT_STEM}_audit.png"
)
AUDIT_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / f"{EXPERIMENT_STEM}_audit_evidence.json"
)


def configure_experiment_paths(config_path: Path) -> None:
    """Point run evidence at the experiment named by a selected config."""

    global EXPERIMENT_STEM
    global REPORT_PATH
    global SMOKE_REPORT_PATH
    global MARKDOWN_PATH
    global FIGURE_PATH
    global AUDIT_EVIDENCE_PATH

    EXPERIMENT_STEM = config_path.stem
    REPORT_PATH = (
        PROJECT_ROOT / "reports" / f"{EXPERIMENT_STEM}_training.json"
    )
    SMOKE_REPORT_PATH = (
        PROJECT_ROOT / "reports" / f"{EXPERIMENT_STEM}_smoke.json"
    )
    MARKDOWN_PATH = (
        PROJECT_ROOT / "reports" / f"{EXPERIMENT_STEM}_training.md"
    )
    FIGURE_PATH = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / f"{EXPERIMENT_STEM}_audit.png"
    )
    AUDIT_EVIDENCE_PATH = (
        PROJECT_ROOT
        / "reports"
        / f"{EXPERIMENT_STEM}_audit_evidence.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


class HelmetDataset(Dataset):
    """Lazy PIL images plus COCO targets, restricted to one frozen ID list."""

    def __init__(
        self,
        *,
        image_ids: Sequence[int],
        images: Mapping[int, Mapping[str, Any]],
        annotations: Mapping[int, Sequence[Mapping[str, Any]]],
        image_root: Path,
        helmet_category_id: int,
        input_normalization: Mapping[str, Any] | None = None,
    ) -> None:
        self.image_ids = [int(value) for value in image_ids]
        self.images = images
        self.annotations = annotations
        self.image_root = image_root
        self.helmet_category_id = int(helmet_category_id)
        self.input_normalization = input_normalization

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        image_record = self.images[image_id]
        path = self.image_root / str(image_record["file_name"])
        with Image.open(path) as handle:
            image = handle.convert("RGB").copy()
        helmet_annotations = [
            dict(annotation)
            for annotation in self.annotations[image_id]
            if int(annotation["category_id"]) == self.helmet_category_id
        ]
        normalization_record = {
            "applied": False,
            "detected_sides": [],
            "crop_xyxy": [0, 0, image.width, image.height],
        }
        if self.input_normalization is not None:
            image_rgb = np.asarray(image, dtype=np.uint8)
            reflection = reflected_padding_guard(
                image_rgb,
                guard_config=self.input_normalization["guard"],
            )
            normalized, transformed, _, normalization = (
                normalize_reflected_padding(
                    image_rgb,
                    annotations=helmet_annotations,
                    pass1={
                        int(annotation["id"]): {}
                        for annotation in helmet_annotations
                    },
                    reflection=reflection,
                    output_shape=image_rgb.shape[:2],
                    transform_masks=False,
                )
            )
            image = Image.fromarray(normalized, mode="RGB")
            helmet_annotations = list(transformed)
            normalization_record = {
                "applied": bool(normalization.applied),
                "detected_sides": list(normalization.detected_sides),
                "crop_xyxy": list(normalization.crop_xyxy),
            }
        coco_annotations = []
        truth = []
        for annotation_index, annotation in enumerate(helmet_annotations):
            box = [float(value) for value in annotation["bbox"]]
            coco_annotations.append(
                {
                    "id": annotation_index,
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
            )
            truth.append(_xywh_to_xyxy(box))
        return {
            "image_id": image_id,
            "image": image,
            "processor_target": {
                "image_id": image_id,
                "annotations": coco_annotations,
            },
            "truth": truth,
            "input_normalization": normalization_record,
        }


def _training_collate(
    processor: Any,
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    encoded = processor(
        images=[item["image"] for item in batch],
        annotations=[item["processor_target"] for item in batch],
        return_tensors="pt",
    )
    return {
        "pixel_values": encoded["pixel_values"],
        "labels": encoded["labels"],
        "helmet_boxes": sum(len(item["truth"]) for item in batch),
    }


def _evaluation_collate(
    processor: Any,
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    encoded = processor(
        images=[item["image"] for item in batch],
        return_tensors="pt",
    )
    return {
        "pixel_values": encoded["pixel_values"],
        "image_ids": [int(item["image_id"]) for item in batch],
        "target_sizes": torch.tensor(
            [
                [item["image"].height, item["image"].width]
                for item in batch
            ],
            dtype=torch.int64,
        ),
        "truth": [item["truth"] for item in batch],
    }


def _move_labels(
    labels: Sequence[Mapping[str, torch.Tensor]],
    device: str,
) -> list[dict[str, torch.Tensor]]:
    return [
        {key: value.to(device) for key, value in label.items()}
        for label in labels
    ]


def _optimizer(model: Any, config: Mapping[str, Any]) -> torch.optim.AdamW:
    settings = config["optimization"]
    base_lr = float(settings["learning_rate"])
    backbone_lr = base_lr * float(settings["backbone_lr_multiplier"])
    weight_decay = float(settings["weight_decay"])
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = "backbone" in name
        no_decay = (
            parameter.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        )
        groups[(is_backbone, no_decay)].append(parameter)
    parameter_groups = []
    for (is_backbone, no_decay), parameters in groups.items():
        if not parameters:
            continue
        parameter_groups.append(
            {
                "params": parameters,
                "lr": backbone_lr if is_backbone else base_lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return torch.optim.AdamW(
        parameter_groups,
        betas=(0.9, 0.999),
    )


def _aggregate(
    *,
    image_ids: Sequence[int],
    truth: Mapping[int, Sequence[Sequence[float]]],
    predictions: Mapping[int, Sequence[tuple[float, Sequence[float]]]],
    threshold: float,
    match_iou: float,
) -> dict[str, float | int]:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_ious = []
    for image_id in image_ids:
        metrics = greedy_detection_metrics(
            truth[image_id],
            predictions[image_id],
            score_threshold=threshold,
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
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(
            precision + recall,
            np.finfo(float).eps,
        ),
        "median_matched_iou": (
            float(np.median(matched_ious)) if matched_ious else 0.0
        ),
    }


def select_calibration_candidate(
    rows: Sequence[Mapping[str, Any]],
    *,
    precision_floor: float,
) -> dict[str, Any] | None:
    """Select maximum F1 only among rows satisfying the frozen precision floor."""

    eligible = [
        dict(row)
        for row in rows
        if float(row["precision"]) >= float(precision_floor)
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            -float(row["f1"]),
            -float(row["recall"]),
            -float(row["median_matched_iou"]),
            -float(row["threshold"]),
            int(row["epoch"]),
        )
    )
    return eligible[0]


def _predict(
    *,
    model: Any,
    processor: Any,
    loader: DataLoader,
    device: str,
    score_floor: float,
    geometry_filter: Mapping[str, Any] | None = None,
) -> tuple[
    list[int],
    dict[int, list[list[float]]],
    dict[int, list[tuple[float, list[float]]]],
]:
    model.eval()
    image_ids = []
    truth = {}
    predictions = {}
    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(pixel_values=pixel_values)
            results = processor.post_process_object_detection(
                outputs,
                threshold=float(score_floor),
                target_sizes=batch["target_sizes"],
            )
            for image_id, boxes, target_size, result in zip(
                batch["image_ids"],
                batch["truth"],
                batch["target_sizes"],
                results,
                strict=True,
            ):
                image_ids.append(int(image_id))
                truth[int(image_id)] = [
                    [float(value) for value in box] for box in boxes
                ]
                rows = [
                    (
                        float(score.item()),
                        [float(value) for value in box.tolist()],
                    )
                    for score, label, box in zip(
                        result["scores"],
                        result["labels"],
                        result["boxes"],
                        strict=True,
                    )
                    if int(label.item()) == 0
                ]
                if geometry_filter is not None:
                    rows = filter_prediction_geometry(
                        rows,
                        image_width=int(target_size[1].item()),
                        image_height=int(target_size[0].item()),
                        max_relative_area=float(
                            geometry_filter["max_relative_area"]
                        ),
                        max_relative_height=float(
                            geometry_filter["max_relative_height"]
                        ),
                        min_aspect_ratio=float(
                            geometry_filter.get("min_aspect_ratio", 0.0)
                        ),
                        max_aspect_ratio=float(
                            geometry_filter.get(
                                "max_aspect_ratio",
                                math.inf,
                            )
                        ),
                    )
                predictions[int(image_id)] = rows
    return image_ids, truth, predictions


def _calibration_rows(
    *,
    epoch: int,
    image_ids: Sequence[int],
    truth: Mapping[int, Sequence[Sequence[float]]],
    predictions: Mapping[int, Sequence[tuple[float, Sequence[float]]]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "epoch": int(epoch),
            "threshold": float(threshold),
            **_aggregate(
                image_ids=image_ids,
                truth=truth,
                predictions=predictions,
                threshold=float(threshold),
                match_iou=float(config["calibration"]["match_iou"]),
            ),
        }
        for threshold in config["calibration"]["score_thresholds"]
    ]


def _load_model_and_processor(
    *,
    model_dir: Path,
    config: Mapping[str, Any],
) -> tuple[Any, Any]:
    require_verified_model(model_dir, config)
    processor = AutoImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model_config.num_labels = 1
    model_config.id2label = {0: "helmet"}
    model_config.label2id = {"helmet": 0}
    model = AutoModelForObjectDetection.from_pretrained(
        model_dir,
        config=model_config,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    return processor, model


def _render_audit(
    *,
    rows: Sequence[int],
    train_images: Mapping[int, Mapping[str, Any]],
    image_root: Path,
    truth: Mapping[int, Sequence[Sequence[float]]],
    predictions: Mapping[int, Sequence[tuple[float, Sequence[float]]]],
    threshold: float,
    split_manifest_sha256: str,
) -> None:
    panel = 260
    caption = 30
    legend = 58
    columns = 4
    selected = list(rows)
    row_count = math.ceil(len(selected) / columns)
    sheet = Image.new(
        "RGB",
        (panel * columns, legend + (panel + caption) * row_count),
        "white",
    )
    draw_sheet = ImageDraw.Draw(sheet)
    draw_sheet.text(
        (8, 7),
        "SUPERVISED RT-DETRv2 | NEW TRAIN-ONLY AUDIT | "
        "GREEN=DATASET GT | CYAN=MODEL",
        fill="black",
    )
    draw_sheet.text(
        (8, 30),
        f"frozen score threshold={threshold:.3f}",
        fill="black",
    )
    for index, image_id in enumerate(selected):
        with Image.open(
            image_root / str(train_images[image_id]["file_name"])
        ) as handle:
            image = handle.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        for box in truth[image_id]:
            draw.rectangle(tuple(box), outline=(0, 255, 0), width=2)
        for score, box in predictions[image_id]:
            if score < threshold:
                continue
            draw.rectangle(tuple(box), outline=(0, 255, 255), width=2)
            draw.text((box[0] + 2, box[1] + 2), f"{score:.2f}", fill="cyan")
        image = image.resize((panel, panel), Image.Resampling.LANCZOS)
        x0 = (index % columns) * panel
        y0 = legend + (index // columns) * (panel + caption)
        sheet.paste(image, (x0, y0))
        draw_sheet.text(
            (x0 + 4, y0 + panel + 6),
            f"{index + 1:02d} | Train image {image_id}",
            fill="black",
        )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(FIGURE_PATH, optimize=True)
    evidence = build_audit_evidence(
        rows=selected,
        truth=truth,
        predictions=predictions,
        threshold=threshold,
        split_manifest_sha256=split_manifest_sha256,
    )
    AUDIT_EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_audit_evidence(
    *,
    rows: Sequence[int],
    truth: Mapping[int, Sequence[Sequence[float]]],
    predictions: Mapping[int, Sequence[tuple[float, Sequence[float]]]],
    threshold: float,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    """Store exact review boxes so later rendering never parses a raster."""

    cases = []
    for cell, image_id in enumerate(rows, start=1):
        cases.append(
            {
                "cell": cell,
                "image_id": int(image_id),
                "truth_boxes": [
                    [float(value) for value in box]
                    for box in truth[int(image_id)]
                ],
                "model_predictions": [
                    {
                        "score": float(score),
                        "box": [float(value) for value in box],
                    }
                    for score, box in predictions[int(image_id)]
                    if float(score) >= float(threshold)
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "frozen_one_shot_audit_review_evidence",
        "experiment_id": EXPERIMENT_STEM,
        "split_manifest_sha256": split_manifest_sha256,
        "score_threshold": float(threshold),
        "cases": cases,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }


def _build_datasets(
    *,
    config_path: Path = CONFIG_PATH,
    split_path: Path = SPLIT_PATH,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    dict[int, Mapping[str, Any]],
    Path,
    HelmetDataset,
    HelmetDataset,
    HelmetDataset,
]:
    config = load_supervised_labeler_config(config_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if config.get("split_manifest_sha256") != split.get("manifest_sha256"):
        raise RuntimeError("Supervised config and split manifest disagree")
    paths = load_project_paths()
    coco, _, train_images, annotations, _, test_ids = _load_context(paths)
    all_selected = (
        set(split["training_image_ids"])
        | set(split["calibration_image_ids"])
        | set(split["untouched_audit_image_ids"])
    )
    if test_ids & all_selected:
        raise AssertionError("Test leakage entered supervised labeler")
    helmet_category_id = next(
        int(category["id"])
        for category in coco["categories"]
        if str(category["name"]) == "helmet"
    )

    def dataset(key: str) -> HelmetDataset:
        return HelmetDataset(
            image_ids=split[key],
            images=train_images,
            annotations=annotations,
            image_root=paths.hardhat_raw,
            helmet_category_id=helmet_category_id,
            input_normalization=config.get("input_normalization"),
        )

    return (
        config,
        split,
        paths,
        train_images,
        paths.hardhat_raw,
        dataset("training_image_ids"),
        dataset("calibration_image_ids"),
        dataset("untouched_audit_image_ids"),
    )


def _smoke(
    *,
    config_path: Path = CONFIG_PATH,
    split_path: Path = SPLIT_PATH,
) -> None:
    (
        config,
        _,
        paths,
        _,
        _,
        training,
        _,
        _,
    ) = _build_datasets(config_path=config_path, split_path=split_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Supervised labeler smoke requires CUDA")
    seed = int(config.get("training_seed", config.get("root_seed", -1)))
    if seed < 0:
        raise RuntimeError("Supervised training seed is missing")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    processor, model = _load_model_and_processor(
        model_dir=model_directory(paths, config),
        config=config,
    )
    model.to("cuda").train()
    loader = DataLoader(
        training,
        batch_size=int(config["optimization"]["train_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: _training_collate(processor, batch),
    )
    batch = next(iter(loader))
    optimizer = _optimizer(model, config)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(
            pixel_values=batch["pixel_values"].to("cuda"),
            labels=_move_labels(batch["labels"], "cuda"),
        )
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise RuntimeError("Supervised smoke produced non-finite loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float(config["optimization"]["max_grad_norm"]),
    )
    optimizer.step()
    payload = {
        "status": "smoke_passed",
        "loss": float(loss.detach().cpu()),
        "batch_size": int(batch["pixel_values"].shape[0]),
        "helmet_boxes": int(batch["helmet_boxes"]),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "validation_images_read": 0,
        "test_images_read": 0,
        "untouched_audit_images_read": 0,
    }
    SMOKE_REPORT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _train(
    *,
    config_path: Path = CONFIG_PATH,
    split_path: Path = SPLIT_PATH,
) -> None:
    (
        config,
        split,
        paths,
        train_images,
        image_root,
        training,
        calibration,
        audit,
    ) = _build_datasets(config_path=config_path, split_path=split_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Supervised labeler training requires CUDA")
    seed = int(config.get("training_seed", config.get("root_seed", -1)))
    split_seed = int(config.get("split_seed", config.get("root_seed", -1)))
    if seed < 0 or split_seed < 0:
        raise RuntimeError("Supervised training/split seed is missing")
    run_root = (
        paths.runs
        / f"{config.get('experiment_id', EXPERIMENT_STEM)}"
        f"_train{seed}_split{split_seed}"
    )
    if (
        run_root.exists()
        or REPORT_PATH.exists()
        or FIGURE_PATH.exists()
        or AUDIT_EVIDENCE_PATH.exists()
    ):
        raise RuntimeError("Supervised labeler run evidence already exists")
    run_root.mkdir(parents=True)
    best_dir = run_root / "best"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    processor, model = _load_model_and_processor(
        model_dir=model_directory(paths, config),
        config=config,
    )
    model.to("cuda")
    generator = torch.Generator().manual_seed(seed)
    sampling = config.get("sampling")
    sampler = None
    if sampling is not None:
        weights = supervised_sampling_weights(
            image_ids=training.image_ids,
            annotations=training.annotations,
            image_records=training.images,
            helmet_category_id=training.helmet_category_id,
            empty_image_weight=float(sampling["empty_image_weight"]),
            close_helmet_pair_weight=float(
                sampling["close_helmet_pair_weight"]
            ),
            close_pair_ratio_max=float(
                sampling[
                    "close_pair_center_distance_over_mean_sqrt_area_max"
                ]
            ),
            small_helmet_weight=float(
                sampling.get("small_helmet_weight", 1.0)
            ),
            small_helmet_relative_area_max=float(
                sampling.get("small_helmet_relative_area_max", 0.0)
            ),
            large_helmet_weight=float(
                sampling.get("large_helmet_weight", 1.0)
            ),
            large_helmet_relative_area_min=float(
                sampling.get("large_helmet_relative_area_min", 1.0)
            ),
            near_image_edge_helmet_weight=float(
                sampling.get("near_image_edge_helmet_weight", 1.0)
            ),
            near_image_edge_margin_fraction=float(
                sampling.get("near_image_edge_margin_fraction", 0.0)
            ),
            owner_miss_replay_image_ids=tuple(
                int(value)
                for value in sampling.get("owner_miss_replay_image_ids", [])
            ),
            owner_miss_replay_weight=float(
                sampling.get("owner_miss_replay_weight", 1.0)
            ),
        )
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(training),
            replacement=True,
            generator=generator,
        )
    training_loader = DataLoader(
        training,
        batch_size=int(config["optimization"]["train_batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator if sampler is None else None,
        num_workers=int(config["optimization"]["dataloader_num_workers"]),
        collate_fn=lambda batch: _training_collate(processor, batch),
    )
    evaluation_collate = lambda batch: _evaluation_collate(processor, batch)
    calibration_loader = DataLoader(
        calibration,
        batch_size=int(config["optimization"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=evaluation_collate,
    )
    optimizer = _optimizer(model, config)
    epochs = int(config["optimization"]["epochs"])
    total_steps = epochs * len(training_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["optimization"]["warmup_steps"]),
        num_training_steps=total_steps,
    )
    all_calibration_rows = []
    epoch_records = []
    best = None
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(training_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    pixel_values=batch["pixel_values"].to("cuda"),
                    labels=_move_labels(batch["labels"], "cuda"),
                )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}, step {step}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["optimization"]["max_grad_norm"]),
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 50 == 0 or step == len(training_loader):
                print(
                    f"epoch={epoch:02d}/{epochs} "
                    f"step={step:04d}/{len(training_loader)} "
                    f"loss={np.mean(losses[-50:]):.4f}",
                    flush=True,
                )
        image_ids, truth, predictions = _predict(
            model=model,
            processor=processor,
            loader=calibration_loader,
            device="cuda",
            score_floor=min(config["calibration"]["score_thresholds"]),
            geometry_filter=config.get("postprocessing"),
        )
        rows = _calibration_rows(
            epoch=epoch,
            image_ids=image_ids,
            truth=truth,
            predictions=predictions,
            config=config,
        )
        all_calibration_rows.extend(rows)
        candidate = select_calibration_candidate(
            rows,
            precision_floor=float(config["calibration"]["min_precision"]),
        )
        epoch_record = {
            "epoch": epoch,
            "mean_train_loss": float(np.mean(losses)),
            "calibration_candidate": candidate,
        }
        epoch_records.append(epoch_record)
        if candidate is not None and (
            best is None
            or select_calibration_candidate(
                [candidate, best],
                precision_floor=float(config["calibration"]["min_precision"]),
            )
            == candidate
        ):
            best = dict(candidate)
            model.save_pretrained(best_dir, safe_serialization=True)
            processor.save_pretrained(best_dir)
        (run_root / "training_progress.json").write_text(
            json.dumps(
                {
                    "epochs_completed": epoch,
                    "epoch_records": epoch_records,
                    "best": best,
                    "untouched_audit_images_read": 0,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(
            f"epoch={epoch:02d} calibration={json.dumps(candidate)}",
            flush=True,
        )

    if best is None:
        report = {
            "schema_version": 1,
            "status": "supervised_calibration_failed",
            "split_manifest_sha256": split["manifest_sha256"],
            "epoch_records": epoch_records,
            "calibration_grid": all_calibration_rows,
            "validation_images_read": 0,
            "test_images_read": 0,
            "untouched_audit_images_read": 0,
            "whole_image_generation_run": False,
        }
    else:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForObjectDetection.from_pretrained(
            best_dir,
            local_files_only=True,
        ).to("cuda")
        processor = AutoImageProcessor.from_pretrained(
            best_dir,
            local_files_only=True,
        )
        audit_loader = DataLoader(
            audit,
            batch_size=int(config["optimization"]["eval_batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: _evaluation_collate(processor, batch),
        )
        audit_ids, audit_truth, audit_predictions = _predict(
            model=model,
            processor=processor,
            loader=audit_loader,
            device="cuda",
            score_floor=min(config["calibration"]["score_thresholds"]),
            geometry_filter=config.get("postprocessing"),
        )
        audit_metrics = _aggregate(
            image_ids=audit_ids,
            truth=audit_truth,
            predictions=audit_predictions,
            threshold=float(best["threshold"]),
            match_iou=float(config["calibration"]["match_iou"]),
        )
        checks = {
            "audit_precision": float(audit_metrics["precision"])
            >= float(config["audit_gate"]["min_precision"]),
            "audit_recall": float(audit_metrics["recall"])
            >= float(config["audit_gate"]["min_recall"]),
            "audit_median_matched_iou": float(
                audit_metrics["median_matched_iou"]
            )
            >= float(config["audit_gate"]["min_median_matched_iou"]),
        }
        status = (
            "supervised_labeler_audit_passed"
            if all(checks.values())
            else "supervised_labeler_audit_failed"
        )
        _render_audit(
            rows=audit_ids,
            train_images=train_images,
            image_root=image_root,
            truth=audit_truth,
            predictions=audit_predictions,
            threshold=float(best["threshold"]),
            split_manifest_sha256=str(split["manifest_sha256"]),
        )
        checkpoint_path = best_dir / "model.safetensors"
        report = {
            "schema_version": 1,
            "status": status,
            "split_manifest_sha256": split["manifest_sha256"],
            "best_calibration": best,
            "audit_metrics": audit_metrics,
            "checks": checks,
            "epoch_records": epoch_records,
            "calibration_grid": all_calibration_rows,
            "checkpoint_path": str(best_dir),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "elapsed_minutes": (time.time() - started) / 60,
            "validation_images_read": 0,
            "test_images_read": 0,
            "untouched_audit_images_read": len(audit_ids),
            "whole_image_generation_run": False,
            "postprocessing": config.get("postprocessing"),
            "sampling": config.get("sampling"),
            "audit_evidence_path": str(
                AUDIT_EVIDENCE_PATH.relative_to(PROJECT_ROOT)
            ).replace("\\", "/"),
            "audit_evidence_sha256": _sha256(AUDIT_EVIDENCE_PATH),
        }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised RT-DETRv2 labeler",
        "",
        f"- Status: **{report['status']}**",
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image FLUX generations: **0**",
    ]
    if best is not None:
        audit_metrics = report["audit_metrics"]
        lines.extend(
            [
                f"- Best calibration epoch: **{best['epoch']}**",
                f"- Frozen score threshold: **{best['threshold']:.4f}**",
                f"- New audit precision: **{audit_metrics['precision']:.4f}**",
                f"- New audit recall: **{audit_metrics['recall']:.4f}**",
                (
                    "- New audit median matched IoU: "
                    f"**{audit_metrics['median_matched_iou']:.4f}**"
                ),
            ]
        )
    lines.append("")
    MARKDOWN_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "train"))
    parser.add_argument(
        "--config-path",
        type=Path,
        default=CONFIG_PATH,
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=SPLIT_PATH,
    )
    args = parser.parse_args()
    configure_experiment_paths(args.config_path)
    if args.action == "smoke":
        _smoke(
            config_path=args.config_path,
            split_path=args.split_path,
        )
    else:
        _train(
            config_path=args.config_path,
            split_path=args.split_path,
        )


if __name__ == "__main__":
    main()
