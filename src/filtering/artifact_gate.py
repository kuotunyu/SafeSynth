"""HOG + logistic-regression paste-artifact gate for M11."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import rankdata
from skimage.feature import hog

from src.data.paths import ProjectPaths


@dataclass(frozen=True)
class PatchExample:
    feature: np.ndarray
    label: int
    group_key: str
    class_name: str
    example_id: str


def _stable_fold(key: str, *, seed: int, folds: int = 5) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _context_crop(
    image_rgb: np.ndarray,
    bbox_xywh: Sequence[float],
    *,
    context_scale: float,
) -> np.ndarray:
    image_height, image_width = image_rgb.shape[:2]
    x, y, width, height = (float(value) for value in bbox_xywh)
    center_x = x + width / 2
    center_y = y + height / 2
    crop_width = max(width * context_scale, 8)
    crop_height = max(height * context_scale, 8)
    left = max(0, int(np.floor(center_x - crop_width / 2)))
    top = max(0, int(np.floor(center_y - crop_height / 2)))
    right = min(image_width, int(np.ceil(center_x + crop_width / 2)))
    bottom = min(image_height, int(np.ceil(center_y + crop_height / 2)))
    return image_rgb[top:bottom, left:right]


def patch_feature(patch_rgb: np.ndarray, *, size: int) -> np.ndarray:
    """Extract scale-normalized colour HOG plus coarse HSV statistics."""

    resized = cv2.resize(patch_rgb, (size, size), interpolation=cv2.INTER_AREA)
    hog_feature = hog(
        resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        channel_axis=-1,
        feature_vector=True,
    ).astype(np.float32)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    histograms = [
        np.histogram(hsv[..., channel], bins=8, range=(0, 256), density=True)[0]
        for channel in range(3)
    ]
    return np.concatenate((hog_feature, *histograms)).astype(np.float32)


def _geometry_descriptor(bbox_xywh: Sequence[float]) -> np.ndarray:
    width = max(float(bbox_xywh[2]), 1.0)
    height = max(float(bbox_xywh[3]), 1.0)
    return np.log(np.asarray((width, height), dtype=np.float64))


def _match_real_annotations(
    target_boxes: Sequence[Sequence[float]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Match controls without replacement by log pixel width and height."""

    if not candidates:
        raise ValueError("Cannot geometry-match without real candidates")
    candidate_descriptors = np.stack(
        [_geometry_descriptor(candidate["bbox"]) for candidate in candidates]
    )
    available = np.ones(len(candidates), dtype=bool)
    matched: list[Mapping[str, Any]] = []
    for target_box in target_boxes:
        if not available.any():
            available[:] = True
        delta = candidate_descriptors - _geometry_descriptor(target_box)
        distances = np.einsum("ij,ij->i", delta, delta)
        distances[~available] = np.inf
        index = int(np.argmin(distances))
        available[index] = False
        matched.append(candidates[index])
    return matched


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_patch_examples(
    *,
    paths: ProjectPaths,
    run_dir: Path,
    config: Mapping[str, Any],
    seed: int,
) -> list[PatchExample]:
    """Build class/geometry/fold-matched real and pasted examples."""

    gate = config["artifact_gate"]
    context_scale = float(gate["patch_context_scale"])
    patch_size = int(gate["patch_size_px"])
    records = _read_jsonl(run_dir / "records.jsonl")
    frozen = {
        int(record["image_id"]): record
        for record in _read_json(paths.splits / "split_manifest.json")["images"]
    }
    coco = _read_json(paths.interim / "coco_all.json")
    categories = {
        int(record["id"]): str(record["name"]) for record in coco["categories"]
    }
    images = {int(record["id"]): record for record in coco["images"]}
    annotations = {int(record["id"]): record for record in coco["annotations"]}
    real_by_class_fold: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if frozen[image_id]["split"] != "train":
            continue
        class_name = categories[int(annotation["category_id"])]
        group_key = f"frozen-group:{int(frozen[image_id]['group_id'])}"
        fold = _stable_fold(group_key, seed=seed)
        real_by_class_fold[(class_name, fold)].append(annotation)
    for candidates in real_by_class_fold.values():
        candidates.sort(key=lambda item: int(item["id"]))

    examples: list[PatchExample] = []
    synthetic_targets: dict[
        tuple[str, int], list[tuple[Sequence[float], str]]
    ] = defaultdict(list)
    for record in records:
        image = np.asarray(
            Image.open(run_dir / record["file_name"]).convert("RGB")
        )
        for instance in record["instances"]:
            if instance["kind"] != "pasted":
                continue
            class_name = str(instance["class_name"])
            patch = _context_crop(
                image,
                instance["bbox_xywh"],
                context_scale=context_scale,
            )
            if patch.size == 0:
                continue
            source_annotation_id = int(instance["source_annotation_id"])
            source_annotation = annotations[source_annotation_id]
            source_image_id = int(source_annotation["image_id"])
            source_group_id = int(frozen[source_image_id]["group_id"])
            if frozen[source_image_id]["split"] != "train":
                raise RuntimeError("H4 paired control accessed a non-Train source")
            if source_group_id != int(instance["source_group_id"]):
                raise RuntimeError("H4 source group disagrees with the frozen manifest")
            if categories[int(source_annotation["category_id"])] != class_name:
                raise RuntimeError("H4 source class disagrees with the pasted class")
            group_key = f"frozen-group:{source_group_id}"
            fold = _stable_fold(group_key, seed=seed)
            synthetic_targets[(class_name, fold)].append(
                (instance["bbox_xywh"], str(record["sample_id"]))
            )
            examples.append(
                PatchExample(
                    feature=patch_feature(patch, size=patch_size),
                    label=1,
                    group_key=group_key,
                    class_name=class_name,
                    example_id=f"{record['sample_id']}:{instance['instance_id']}",
                )
            )

    for key, targets in sorted(synthetic_targets.items()):
        class_name, fold = key
        target_boxes = [target[0] for target in targets]
        candidates = real_by_class_fold[key]
        matched_annotations = _match_real_annotations(target_boxes, candidates)
        for target_index, annotation in enumerate(matched_annotations):
            image_id = int(annotation["image_id"])
            group_key = f"frozen-group:{int(frozen[image_id]['group_id'])}"
            if _stable_fold(group_key, seed=seed) != fold:
                raise AssertionError("Geometry match crossed an H4 fold")
            real_image = np.asarray(
                Image.open(paths.hardhat_raw / images[image_id]["file_name"]).convert(
                    "RGB"
                )
            )
            real_patch = _context_crop(
                real_image,
                annotation["bbox"],
                context_scale=context_scale,
            )
            examples.append(
                PatchExample(
                    feature=patch_feature(real_patch, size=patch_size),
                    label=0,
                    group_key=group_key,
                    class_name=class_name,
                    example_id=(
                        f"real:{int(annotation['id'])}:"
                        f"match:{targets[target_index][1]}"
                    ),
                )
            )
    return examples


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute tie-corrected binary ROC AUC without an extra dependency."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC AUC requires both labels")
    ranks = rankdata(scores, method="average")
    return float(
        (
            ranks[positive].sum()
            - n_positive * (n_positive + 1) / 2
        )
        / (n_positive * n_negative)
    )


def train_artifact_classifier(
    examples: Sequence[PatchExample],
    *,
    seed: int,
    bootstrap_samples: int,
    logistic_c: float = 1.0,
) -> dict[str, Any]:
    """Fit a deterministic linear classifier with group-disjoint evaluation."""

    train_examples = [
        example
        for example in examples
        if _stable_fold(example.group_key, seed=seed) != 0
    ]
    test_examples = [
        example
        for example in examples
        if _stable_fold(example.group_key, seed=seed) == 0
    ]
    for partition_name, partition in (
        ("train", train_examples),
        ("test", test_examples),
    ):
        if {example.label for example in partition} != {0, 1}:
            raise RuntimeError(f"H4 {partition_name} partition lacks one label")
    train_x = np.stack([example.feature for example in train_examples])
    train_y = np.asarray([example.label for example in train_examples], dtype=np.float32)
    test_x = np.stack([example.feature for example in test_examples])
    test_y = np.asarray([example.label for example in test_examples], dtype=np.int64)
    mean = train_x.mean(axis=0)
    standard_deviation = np.maximum(train_x.std(axis=0), 1e-5)
    train_x = (train_x - mean) / standard_deviation
    test_x = (test_x - mean) / standard_deviation

    torch.manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], 1)
    features = torch.from_numpy(train_x)
    targets = torch.from_numpy(train_y)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=200,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = model(features).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        # Match the usual C=1 L2-logistic objective while using a mean BCE:
        #   mean(log-loss) + ||w||^2 / (2 * C * n)
        # The unregularized high-dimensional HOG fit saturates and learns
        # unstable pixel-level shortcuts.
        loss = loss + model.weight.square().sum() / (
            2 * float(logistic_c) * len(train_examples)
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.from_numpy(test_x)).squeeze(1)).numpy()
    auc = roc_auc(test_y, scores)
    rng = np.random.default_rng(seed)
    bootstrap_auc: list[float] = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(test_y), len(test_y))
        sampled_labels = test_y[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        bootstrap_auc.append(roc_auc(sampled_labels, scores[indices]))
    ci_low, ci_high = np.percentile(bootstrap_auc, (2.5, 97.5))
    return {
        "n_examples": len(examples),
        "n_train": len(train_examples),
        "n_test": len(test_examples),
        "train_label_counts": {
            str(label): int((train_y == label).sum()) for label in (0, 1)
        },
        "test_label_counts": {
            str(label): int((test_y == label).sum()) for label in (0, 1)
        },
        "auc": auc,
        "auc_ci95": [float(ci_low), float(ci_high)],
        "test_labels": test_y.tolist(),
        "test_scores": scores.astype(float).tolist(),
        "test_example_ids": [example.example_id for example in test_examples],
        "test_classes": [example.class_name for example in test_examples],
        "test_group_keys": [example.group_key for example in test_examples],
        "feature_dimensions": int(train_x.shape[1]),
        "logistic_c": float(logistic_c),
        "seed": seed,
    }
