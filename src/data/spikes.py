"""Exploratory data spikes that precede the frozen group split."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
import matplotlib
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.data.voc_to_coco import DataInvariantError, write_canonical_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt

PHASH_THRESHOLDS = (4, 6, 8, 10)
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


@dataclass(frozen=True)
class GroupingResult:
    threshold: int
    labels: np.ndarray
    group_sizes: tuple[int, ...]
    split_simulations: dict[int, dict[str, int]]


@dataclass(frozen=True)
class ClipGroupingResult:
    cosine_threshold: float
    phash_guard: int
    labels: np.ndarray
    group_sizes: tuple[int, ...]
    split_simulations: dict[int, dict[str, int]]


def load_grouping_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return config


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def image_paths_from_coco(coco: dict[str, Any], dataset_root: Path) -> list[Path]:
    paths = [dataset_root / item["file_name"] for item in coco["images"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise DataInvariantError(f"Missing {len(missing)} COCO image files: {missing[:10]}")
    return paths


def box_iou_xywh(left: Sequence[float], right: Sequence[float]) -> float:
    left_x1, left_y1, left_w, left_h = left
    right_x1, right_y1, right_w, right_h = right
    left_x2, left_y2 = left_x1 + left_w, left_y1 + left_h
    right_x2, right_y2 = right_x1 + right_w, right_y1 + right_h
    intersection_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_w * intersection_h
    union = left_w * left_h + right_w * right_h - intersection
    return intersection / union if union > 0 else 0.0


def count_cross_class_pairs(
    coco: dict[str, Any],
    left_class: str,
    right_class: str,
    *,
    iou_threshold: float,
) -> tuple[int, int]:
    category_names = {item["id"]: item["name"] for item in coco["categories"]}
    boxes_by_image_class: dict[tuple[int, str], list[Sequence[float]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        boxes_by_image_class[
            (annotation["image_id"], category_names[annotation["category_id"]])
        ].append(annotation["bbox"])

    total_pairs = 0
    matching_pairs = 0
    for image in coco["images"]:
        left_boxes = boxes_by_image_class[(image["id"], left_class)]
        right_boxes = boxes_by_image_class[(image["id"], right_class)]
        for left in left_boxes:
            for right in right_boxes:
                total_pairs += 1
                if box_iou_xywh(left, right) > iou_threshold:
                    matching_pairs += 1
    return matching_pairs, total_pairs


def _context_crop(
    image: Image.Image,
    bbox: Sequence[float],
    *,
    context_fraction: float,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    x, y, width, height = bbox
    padding = context_fraction * max(width, height)
    left = max(0, math.floor(x - padding))
    top = max(0, math.floor(y - padding))
    right = min(image.width, math.ceil(x + width + padding))
    bottom = min(image.height, math.ceil(y + height + padding))
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    relative_bbox = (x - left, y - top, width, height)
    return crop, relative_bbox


def create_contact_sheet(
    *,
    coco: dict[str, Any],
    dataset_root: Path,
    class_name: str,
    output_path: Path,
    sample_size: int,
    seed: int,
    context_fraction: float = 0.50,
    columns: int = 8,
    cell_size: int = 144,
) -> list[int]:
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    images = {item["id"]: item for item in coco["images"]}
    candidates = [
        item for item in coco["annotations"] if categories[item["category_id"]] == class_name
    ]
    if len(candidates) < sample_size:
        raise DataInvariantError(
            f"Requested {sample_size} {class_name} samples from {len(candidates)} candidates"
        )
    selected = random.Random(seed).sample(candidates, sample_size)
    rows = math.ceil(sample_size / columns)
    caption_height = 18
    canvas = Image.new(
        "RGB",
        (columns * cell_size, rows * (cell_size + caption_height)),
        "white",
    )
    font = ImageFont.load_default()

    for index, annotation in enumerate(selected):
        image_record = images[annotation["image_id"]]
        image_path = dataset_root / image_record["file_name"]
        with Image.open(image_path) as image:
            crop, relative_bbox = _context_crop(
                image,
                annotation["bbox"],
                context_fraction=context_fraction,
            )
        scale = min(cell_size / crop.width, cell_size / crop.height)
        resized_size = (
            max(1, round(crop.width * scale)),
            max(1, round(crop.height * scale)),
        )
        resized = crop.resize(resized_size, Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (cell_size, cell_size), (235, 235, 235))
        x_offset = (cell_size - resized.width) // 2
        y_offset = (cell_size - resized.height) // 2
        cell.paste(resized, (x_offset, y_offset))
        draw = ImageDraw.Draw(cell)
        box_x, box_y, box_w, box_h = relative_bbox
        draw.rectangle(
            (
                x_offset + box_x * scale,
                y_offset + box_y * scale,
                x_offset + (box_x + box_w) * scale,
                y_offset + (box_y + box_h) * scale,
            ),
            outline=(255, 0, 0),
            width=2,
        )
        column = index % columns
        row = index // columns
        canvas.paste(cell, (column * cell_size, row * (cell_size + caption_height)))
        caption = f"a{annotation['id']} i{annotation['image_id']}"
        ImageDraw.Draw(canvas).text(
            (column * cell_size + 3, row * (cell_size + caption_height) + cell_size + 2),
            caption,
            fill="black",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return [int(item["id"]) for item in selected]


def save_aspect_ratio_histogram(coco: dict[str, Any], output_path: Path) -> dict[str, Any]:
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    ratios: dict[str, list[float]] = defaultdict(list)
    for annotation in coco["annotations"]:
        width, height = annotation["bbox"][2:]
        ratios[categories[annotation["category_id"]]].append(width / height)

    fig, axes = plt.subplots(1, len(ratios), figsize=(13, 4), constrained_layout=True)
    summary: dict[str, Any] = {}
    for axis, class_name in zip(axes, sorted(ratios), strict=True):
        values = np.asarray(ratios[class_name], dtype=np.float64)
        clipped = values[(values >= 0.1) & (values <= 4.0)]
        axis.hist(clipped, bins=40, color="#2364aa", alpha=0.9)
        axis.axvline(float(np.median(values)), color="#d1495b", linestyle="--")
        axis.set_title(f"{class_name} (n={len(values):,})")
        axis.set_xlabel("bbox width / height")
        axis.set_ylabel("instances")
        summary[class_name] = {
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return summary


def compute_phashes(
    image_paths: Sequence[Path],
    *,
    workers: int = 8,
) -> list[str]:
    def hash_one(path: Path) -> str:
        with Image.open(path) as image:
            return str(imagehash.phash(image.convert("RGB"), hash_size=8))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(hash_one, image_paths))


def hamming_distance_matrix(phashes: Sequence[str]) -> np.ndarray:
    """Compute all 64-bit Hamming distances with two dense byte matrix products."""

    if not phashes:
        return np.empty((0, 0), dtype=np.uint8)
    integers = np.asarray([int(value, 16) for value in phashes], dtype=">u8")
    bits = np.unpackbits(integers.view(np.uint8).reshape(-1, 8), axis=1)
    inverse = 1 - bits
    matches = bits @ bits.T + inverse @ inverse.T
    return (64 - matches).astype(np.uint8, copy=False)


def canonical_group_labels(labels: np.ndarray) -> np.ndarray:
    """Remap component labels by their earliest image index for stable group IDs."""

    first_index: dict[int, int] = {}
    for index, label in enumerate(labels.tolist()):
        first_index.setdefault(label, index)
    order = sorted(first_index, key=first_index.get)
    remap = {old: new for new, old in enumerate(order)}
    return np.asarray([remap[int(label)] for label in labels], dtype=np.int32)


def group_from_distances(distances: np.ndarray, threshold: int) -> np.ndarray:
    adjacency = csr_matrix(distances <= threshold)
    _, labels = connected_components(adjacency, directed=False, return_labels=True)
    return canonical_group_labels(labels)


def stable_group_split(
    labels: Sequence[int],
    *,
    seed: int,
    fractions: Sequence[float] = SPLIT_FRACTIONS,
) -> dict[int, str]:
    """Greedily place whole groups near target image counts with seeded tie-breaking."""

    group_sizes = Counter(int(label) for label in labels)
    rng = random.Random(seed)
    tie_breakers = {group_id: rng.random() for group_id in group_sizes}
    ordered_groups = sorted(
        group_sizes,
        key=lambda group_id: (-group_sizes[group_id], tie_breakers[group_id], group_id),
    )
    target = {
        name: len(labels) * fraction for name, fraction in zip(SPLIT_NAMES, fractions, strict=True)
    }
    current = dict.fromkeys(SPLIT_NAMES, 0)
    assignments: dict[int, str] = {}
    for group_id in ordered_groups:
        size = group_sizes[group_id]
        relative_deficits = {
            name: (target[name] - current[name]) / target[name] for name in SPLIT_NAMES
        }
        best_deficit = max(relative_deficits.values())
        candidates = [
            name for name, deficit in relative_deficits.items() if deficit == best_deficit
        ]
        chosen = rng.choice(candidates)
        assignments[group_id] = chosen
        current[chosen] += size
    return assignments


def split_counts(labels: Sequence[int], assignments: dict[int, str]) -> dict[str, int]:
    counts = dict.fromkeys(SPLIT_NAMES, 0)
    for label in labels:
        counts[assignments[int(label)]] += 1
    return counts


def evaluate_grouping_thresholds(
    distances: np.ndarray,
    *,
    thresholds: Sequence[int] = PHASH_THRESHOLDS,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
) -> list[GroupingResult]:
    results: list[GroupingResult] = []
    for threshold in thresholds:
        labels = group_from_distances(distances, threshold)
        size_counter = Counter(labels.tolist())
        simulations = {
            seed: split_counts(labels, stable_group_split(labels, seed=seed)) for seed in seeds
        }
        results.append(
            GroupingResult(
                threshold=threshold,
                labels=labels,
                group_sizes=tuple(sorted(size_counter.values(), reverse=True)),
                split_simulations=simulations,
            )
        )
    return results


def choose_grouping_threshold(
    results: Sequence[GroupingResult],
    *,
    max_group_size: int = 250,
    split_tolerance: float = 0.02,
) -> GroupingResult:
    """Choose the most merging threshold that passes both frozen guardrails."""

    passing: list[GroupingResult] = []
    for result in results:
        if result.group_sizes[0] > max_group_size:
            continue
        total = len(result.labels)
        simulations_pass = True
        for counts in result.split_simulations.values():
            for name, fraction in zip(SPLIT_NAMES, SPLIT_FRACTIONS, strict=True):
                if abs(counts[name] / total - fraction) > split_tolerance:
                    simulations_pass = False
        if simulations_pass:
            passing.append(result)
    if not passing:
        raise DataInvariantError(
            "No pHash threshold satisfies max group <= 250 and split ratios within +/-2%"
        )
    return max(passing, key=lambda item: item.threshold)


def grouping_report(results: Sequence[GroupingResult], selected: GroupingResult) -> str:
    lines = [
        "# Spike H3 — pHash Grouping Structure",
        "",
        "| Hamming threshold | Groups | Largest group | Singleton groups |",
        "|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.threshold} | {len(result.group_sizes):,} | "
            f"{result.group_sizes[0]:,} | {sum(size == 1 for size in result.group_sizes):,} |"
        )
    lines.extend(["", "## Five seeded split simulations", ""])
    for result in results:
        lines.append(f"### Threshold {result.threshold}")
        lines.append("")
        lines.append("| Seed | Train | Validation | Test |")
        lines.append("|---:|---:|---:|---:|")
        for seed, counts in result.split_simulations.items():
            lines.append(
                f"| {seed} | {counts['train']:,} | {counts['val']:,} | {counts['test']:,} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Selected pHash Hamming threshold: `{selected.threshold}`",
            f"- Selected group count: `{len(selected.group_sizes):,}`",
            f"- Selected maximum group size: `{selected.group_sizes[0]:,}`",
            (
                f"- CLIP trigger (`> 2,000` groups): "
                f"`{'yes' if len(selected.group_sizes) > 2_000 else 'no'}`"
            ),
            "",
            (
                "The final M4 split must still use class-stratified allocation; these simulations "
                "only test whether group geometry makes the requested image ratios feasible."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def compute_clip_embeddings(
    image_paths: Sequence[Path],
    *,
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Compute normalized OpenCLIP image embeddings with deterministic preprocessing."""

    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    torch.backends.cudnn.benchmark = False
    batches: list[np.ndarray] = []
    for start in range(0, len(image_paths), batch_size):
        tensors = []
        for path in image_paths[start : start + batch_size]:
            with Image.open(path) as image:
                tensors.append(preprocess(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model.encode_image(batch)
            else:
                features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        batches.append(features.float().cpu().numpy())
        print(
            f"CLIP embeddings: {min(start + batch_size, len(image_paths)):,}/{len(image_paths):,}"
        )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0)


def clip_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got shape {embeddings.shape}")
    return np.clip(embeddings @ embeddings.T, -1.0, 1.0)


def group_with_clip_guard(
    distances: np.ndarray,
    similarities: np.ndarray,
    *,
    phash_threshold: int,
    cosine_threshold: float,
    phash_guard: int,
) -> np.ndarray:
    if distances.shape != similarities.shape:
        raise ValueError("pHash distance and CLIP similarity matrices must have the same shape")
    adjacency = (distances <= phash_threshold) | (
        (distances <= phash_guard) & (similarities >= cosine_threshold)
    )
    _, labels = connected_components(
        csr_matrix(adjacency),
        directed=False,
        return_labels=True,
    )
    return canonical_group_labels(labels)


def evaluate_clip_candidates(
    *,
    distances: np.ndarray,
    similarities: np.ndarray,
    phash_threshold: int,
    cosine_thresholds: Sequence[float],
    phash_guards: Sequence[int],
    seeds: Sequence[int],
) -> list[ClipGroupingResult]:
    results: list[ClipGroupingResult] = []
    for guard in phash_guards:
        for cosine in cosine_thresholds:
            labels = group_with_clip_guard(
                distances,
                similarities,
                phash_threshold=phash_threshold,
                cosine_threshold=cosine,
                phash_guard=guard,
            )
            group_sizes = tuple(sorted(Counter(labels.tolist()).values(), reverse=True))
            simulations = {
                seed: split_counts(labels, stable_group_split(labels, seed=seed)) for seed in seeds
            }
            results.append(
                ClipGroupingResult(
                    cosine_threshold=float(cosine),
                    phash_guard=int(guard),
                    labels=labels,
                    group_sizes=group_sizes,
                    split_simulations=simulations,
                )
            )
    return results


def choose_clip_candidate(
    results: Sequence[ClipGroupingResult],
    *,
    max_group_size: int,
    split_tolerance: float,
) -> ClipGroupingResult:
    passing: list[ClipGroupingResult] = []
    for result in results:
        if result.group_sizes[0] > max_group_size:
            continue
        simulations_pass = True
        for counts in result.split_simulations.values():
            for name, fraction in zip(SPLIT_NAMES, SPLIT_FRACTIONS, strict=True):
                if abs(counts[name] / len(result.labels) - fraction) > split_tolerance:
                    simulations_pass = False
        if simulations_pass:
            passing.append(result)
    if not passing:
        raise DataInvariantError(
            "No guarded CLIP candidate satisfies max-group and split-ratio guardrails"
        )
    return min(
        passing,
        key=lambda item: (
            len(item.group_sizes),
            item.group_sizes[0],
            item.phash_guard,
            -item.cosine_threshold,
        ),
    )


def clip_grouping_report(
    results: Sequence[ClipGroupingResult],
    selected: ClipGroupingResult,
    *,
    model_name: str,
    pretrained: str,
    phash_threshold: int,
) -> str:
    lines = [
        "# Spike H3 — Guarded CLIP Extension",
        "",
        f"- OpenCLIP model: `{model_name}`",
        f"- Pretrained tag: `{pretrained}`",
        f"- Base pHash edge: Hamming `<= {phash_threshold}`",
        "",
        "| Cosine threshold | pHash guard | Groups | Largest group | Singletons |",
        "|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: (item.phash_guard, item.cosine_threshold)):
        lines.append(
            f"| {result.cosine_threshold:.3f} | {result.phash_guard} | "
            f"{len(result.group_sizes):,} | {result.group_sizes[0]:,} | "
            f"{sum(size == 1 for size in result.group_sizes):,} |"
        )
    lines.extend(
        [
            "",
            "## Selected candidate (visual review passed)",
            "",
            f"- Cosine threshold: `{selected.cosine_threshold:.3f}`",
            f"- pHash guard: `{selected.phash_guard}`",
            f"- Groups: `{len(selected.group_sizes):,}`",
            f"- Largest group: `{selected.group_sizes[0]:,}`",
            "",
            (
                "The largest groups are rendered in `h3_clip_largest_groups.png`. "
                "The grid was visually checked before M4 froze this candidate."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def save_clip_group_contact_sheet(
    *,
    coco: dict[str, Any],
    dataset_root: Path,
    labels: Sequence[int],
    output_path: Path,
    rows: int,
    columns: int,
    cell_size: int,
) -> list[dict[str, Any]]:
    image_records = {int(item["id"]): item for item in coco["images"]}
    image_ids = [int(item["id"]) for item in coco["images"]]
    members: dict[int, list[int]] = defaultdict(list)
    for image_id, label in zip(image_ids, labels, strict=True):
        members[int(label)].append(image_id)
    selected_groups = sorted(
        members.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[:rows]
    caption_height = 20
    canvas = Image.new(
        "RGB",
        (columns * cell_size, rows * (cell_size + caption_height)),
        "white",
    )
    font = ImageFont.load_default()
    rendered: list[dict[str, Any]] = []
    for row, (group_id, group_image_ids) in enumerate(selected_groups):
        shown = group_image_ids[:columns]
        rendered.append(
            {
                "group_id": group_id,
                "size": len(group_image_ids),
                "shown_image_ids": shown,
            }
        )
        for column, image_id in enumerate(shown):
            record = image_records[image_id]
            with Image.open(dataset_root / record["file_name"]) as image:
                thumbnail = image.convert("RGB")
                thumbnail.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
            cell = Image.new("RGB", (cell_size, cell_size), (235, 235, 235))
            cell.paste(
                thumbnail,
                ((cell_size - thumbnail.width) // 2, (cell_size - thumbnail.height) // 2),
            )
            canvas.paste(
                cell,
                (column * cell_size, row * (cell_size + caption_height)),
            )
            ImageDraw.Draw(canvas).text(
                (
                    column * cell_size + 3,
                    row * (cell_size + caption_height) + cell_size + 3,
                ),
                f"g{group_id} i{image_id}",
                fill="black",
                font=font,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)
    return rendered


def save_placement_prior_heatmap(
    *,
    coco: dict[str, Any],
    dataset_root: Path,
    selected_image_ids: set[int],
    output_path: Path,
    bins: int = 16,
) -> dict[str, Any]:
    images_by_id = {item["id"]: item for item in coco["images"]}
    category_names = {item["id"]: item["name"] for item in coco["categories"]}
    selected_images = [images_by_id[image_id] for image_id in sorted(selected_image_ids)]
    if not selected_images:
        raise DataInvariantError("Placement-prior candidate split is empty")

    average_size = (208, 208)
    average = np.zeros((average_size[1], average_size[0], 3), dtype=np.float64)
    for image_record in selected_images:
        with Image.open(dataset_root / image_record["file_name"]) as image:
            resized = image.convert("RGB").resize(average_size, Image.Resampling.BILINEAR)
            average += np.asarray(resized, dtype=np.float64)
    average = np.clip(average / len(selected_images), 0, 255).astype(np.uint8)

    histograms: dict[str, np.ndarray] = {
        name: np.ones((bins, bins), dtype=np.float64) for name in category_names.values()
    }
    counts = Counter()
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in selected_image_ids:
            continue
        image_record = images_by_id[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        center_x = min(1.0 - np.finfo(float).eps, max(0.0, (x + width / 2) / image_record["width"]))
        center_y = min(
            1.0 - np.finfo(float).eps,
            max(0.0, (y + height / 2) / image_record["height"]),
        )
        column = int(center_x * bins)
        row = int(center_y * bins)
        class_name = category_names[annotation["category_id"]]
        histograms[class_name][row, column] += 1
        counts[class_name] += 1

    fig, axes = plt.subplots(1, len(histograms), figsize=(15, 5), constrained_layout=True)
    summary: dict[str, Any] = {}
    for axis, class_name in zip(axes, sorted(histograms), strict=True):
        probability = histograms[class_name] / histograms[class_name].sum()
        entropy = -float(np.sum(probability * np.log(probability))) / math.log(bins * bins)
        smoothed = gaussian_filter(probability, sigma=0.75)
        axis.imshow(average)
        overlay = axis.imshow(
            smoothed,
            cmap="magma",
            alpha=0.58,
            interpolation="bilinear",
            extent=(0, average_size[0], average_size[1], 0),
        )
        axis.set_title(f"{class_name} (n={counts[class_name]:,})\nentropy={entropy:.3f}")
        axis.set_axis_off()
        fig.colorbar(overlay, ax=axis, fraction=0.046, pad=0.04)
        summary[class_name] = {
            "annotation_count": counts[class_name],
            "normalized_entropy": entropy,
            "peak_probability": float(probability.max()),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return summary


def write_spike_artifacts(
    *,
    interim_path: Path,
    phashes: Sequence[str],
    image_ids: Sequence[int],
    selected: GroupingResult,
    clip_selected: ClipGroupingResult | None = None,
    clip_model_name: str | None = None,
    clip_pretrained: str | None = None,
) -> None:
    write_canonical_json(
        interim_path / "phash_spike.json",
        {
            "hash_size": 8,
            "records": [
                {"image_id": image_id, "phash": phash}
                for image_id, phash in zip(image_ids, phashes, strict=True)
            ],
        },
    )
    final_labels = clip_selected.labels if clip_selected is not None else selected.labels
    payload: dict[str, Any] = {
        "selected_phash_hamming_threshold": selected.threshold,
        "clip_required": clip_selected is not None,
        "clip_model_name": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "clip_cosine_threshold": (
            clip_selected.cosine_threshold if clip_selected is not None else None
        ),
        "clip_phash_guard": clip_selected.phash_guard if clip_selected is not None else None,
        "group_count": len(set(final_labels.tolist())),
        "max_group_size": max(Counter(final_labels.tolist()).values()),
        "group_ids": [
            {"image_id": image_id, "group_id": int(group_id)}
            for image_id, group_id in zip(image_ids, final_labels, strict=True)
        ],
    }
    write_canonical_json(interim_path / "h3_spike.json", payload)


def image_ids_for_split(
    image_ids: Sequence[int],
    labels: Sequence[int],
    assignments: dict[int, str],
    split: str,
) -> set[int]:
    return {
        image_id
        for image_id, label in zip(image_ids, labels, strict=True)
        if assignments[int(label)] == split
    }


def summarize_group_sizes(sizes: Iterable[int]) -> dict[str, int]:
    counter = Counter(sizes)
    return {str(size): count for size, count in sorted(counter.items())}
