"""One arm, end to end. Shared by the local smoke test and the Colab notebook.

TRAIN-13 asks for a local 1-step smoke test, which is only meaningful if the
smoke test exercises the same code path Colab will. So the notebook calls this
function too, with different paths and a real step budget.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.training.arms import ArmComposition
from src.training.data import (
    DetectionDataset,
    assert_eval_batch_remainder_is_safe,
    build_transform,
    collate,
    load_coco_samples,
)
from src.training.metrics import (
    build_coco_ground_truth,
    evaluate_detections,
    extract_logits_and_boxes,
    predictions_to_coco,
)
from src.training.trainer import (
    RTDetrTrainer,
    build_training_arguments,
    find_resumable_checkpoint,
)

CLASS_NAMES = ("helmet", "head", "person")


@dataclass(frozen=True)
class RunPaths:
    real_images: Path
    real_coco: Path
    synthetic_images: Path
    synthetic_coco: Path | None
    output_dir: Path


def load_model_and_processor(checkpoint: str, *, dtype=torch.float32):
    """Auto classes only. ADR-014: the named RTDetrV2ImageProcessor does not exist."""

    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=len(CLASS_NAMES),
        # Mandatory: the 80-class head cannot be reused for 3 classes.
        ignore_mismatched_sizes=True,
        id2label=dict(enumerate(CLASS_NAMES)),
        label2id={name: index for index, name in enumerate(CLASS_NAMES)},
        dtype=dtype,
    )
    return model, processor


def build_datasets(
    composition: ArmComposition,
    paths: RunPaths,
    processor,
    config: Mapping[str, Any],
):
    train_samples = load_coco_samples(
        paths.real_coco,
        paths.real_images,
        keep_names=composition.real_train,
    )
    if composition.synthetic and paths.synthetic_coco is not None:
        train_samples += load_coco_samples(
            paths.synthetic_coco,
            paths.synthetic_images,
            keep_names=composition.synthetic,
            is_synthetic=True,
        )
    # TRAIN-23: validation is real only, always.
    val_samples = load_coco_samples(
        paths.real_coco, paths.real_images, keep_names=composition.real_val
    )

    transform = build_transform(composition.augmentation_profile, config)
    return (
        DetectionDataset(train_samples, processor, transform),
        DetectionDataset(val_samples, processor, None),
        val_samples,
    )


def make_compute_metrics(processor, val_samples):
    """COCOeval over the whole val set; AP_small is a headline metric."""

    ground_truth = build_coco_ground_truth(val_samples, CLASS_NAMES)
    image_ids = [int(sample.image_id) for sample in val_samples]
    # (height, width) per image, in the order the eval dataloader yields them.
    # Trainer's eval loader does not shuffle, so this order is the dataset order.
    target_sizes_all = [[sample.height, sample.width] for sample in val_samples]

    def compute_metrics(eval_prediction) -> dict[str, float]:
        from types import SimpleNamespace

        predictions = eval_prediction.predictions
        # eval_do_concat_batches=False means predictions arrive as a list of
        # per-batch outputs rather than one concatenated array.
        batches = (
            predictions
            if isinstance(predictions, list)
            else [predictions]
        )
        logits_batches, boxes_batches = [], []
        for batch in batches:
            logits, boxes = extract_logits_and_boxes(
                batch, num_labels=len(CLASS_NAMES)
            )
            logits_batches.append(torch.as_tensor(logits))
            boxes_batches.append(torch.as_tensor(boxes))
        logits = torch.cat(logits_batches, dim=0).float()
        boxes = torch.cat(boxes_batches, dim=0).float()

        count = int(logits.shape[0])
        outputs = SimpleNamespace(logits=logits, pred_boxes=boxes)
        target_sizes = torch.tensor(target_sizes_all[:count], dtype=torch.float32)
        processed = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target_sizes
        )
        detections = predictions_to_coco(processed, image_ids[:count])
        return evaluate_detections(ground_truth, detections)

    return compute_metrics


def run_arm(
    composition: ArmComposition,
    paths: RunPaths,
    *,
    config: Mapping[str, Any],
    total_steps: int,
    seed: int,
    resume: bool = True,
) -> dict[str, Any]:
    """Train one arm, resuming automatically if a checkpoint is present."""

    run_config = config["run"]
    eval_batch = int(run_config["per_device_eval_batch_size"])
    assert_eval_batch_remainder_is_safe(len(composition.real_val), eval_batch)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model, processor = load_model_and_processor(config["model"]["checkpoint"])
    train_dataset, val_dataset, val_samples = build_datasets(
        composition, paths, processor, config
    )

    workers = int(
        run_config["dataloader_num_workers_colab"]
        if os.environ.get("COLAB_GPU") or Path("/content").exists()
        else run_config["dataloader_num_workers_windows"]
    )
    arguments = build_training_arguments(
        output_dir=str(paths.output_dir),
        config=config,
        total_steps=total_steps,
        seed=seed,
        use_bf16=use_bf16,
        dataloader_num_workers=workers,
    )

    trainer = RTDetrTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate,
        compute_metrics=make_compute_metrics(processor, val_samples),
        optimizer_config=config["optimizer"],
    )

    checkpoint = find_resumable_checkpoint(paths.output_dir) if resume else None
    result = trainer.train(resume_from_checkpoint=checkpoint)
    # Evaluate explicitly at the end so the record always carries metrics, even
    # when eval_strategy would not have fired on the final step.
    #
    # K-20: treat what this returns as EXISTENCE EVIDENCE, not as a measurement.
    # Measured against the checkpoints on disk, it matches neither the best nor
    # the last one, so it describes an in-memory model that was never saved -
    # load_best_model_at_end seems to leave some weights behind on this
    # architecture (see also K-17 on its missing/unexpected keys). Report numbers
    # only from scripts/eval.py, which loads a checkpoint directory (EVAL-12).
    final_metrics = trainer.evaluate()

    record = {
        "arm": composition.arm,
        "seed": seed,
        "total_steps": total_steps,
        "resumed_from": checkpoint,
        "train_runtime_seconds": float(result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(result.metrics.get("train_loss", float("nan"))),
        "eval_metrics": {
            key: float(value)
            for key, value in final_metrics.items()
            if isinstance(value, (int, float))
        },
        "composition": composition.summary(),
        "bf16": bool(use_bf16),
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "run_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return record
