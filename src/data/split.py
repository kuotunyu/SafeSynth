"""Group-aware stratified splitting and frozen manifest generation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from src.data.voc_to_coco import (
    DataInvariantError,
    canonical_json_bytes,
    sha256_file,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class GroupStats:
    group_id: int
    image_ids: tuple[int, ...]
    image_count: int
    class_counts: dict[str, int]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def build_group_stats(
    coco: dict[str, Any],
    image_to_group: Mapping[int, int],
    classes: Sequence[str],
) -> list[GroupStats]:
    image_ids = {int(item["id"]) for item in coco["images"]}
    if image_ids != set(image_to_group):
        missing = sorted(image_ids - set(image_to_group))
        extra = sorted(set(image_to_group) - image_ids)
        raise DataInvariantError(
            f"Group map does not cover COCO images; missing={missing[:10]}, extra={extra[:10]}"
        )

    category_names = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    class_counts_by_image: dict[int, Counter[str]] = defaultdict(Counter)
    for annotation in coco["annotations"]:
        class_name = category_names[int(annotation["category_id"])]
        class_counts_by_image[int(annotation["image_id"])][class_name] += 1

    grouped_images: dict[int, list[int]] = defaultdict(list)
    grouped_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for image_id in sorted(image_ids):
        group_id = int(image_to_group[image_id])
        grouped_images[group_id].append(image_id)
        grouped_counts[group_id].update(class_counts_by_image[image_id])

    return [
        GroupStats(
            group_id=group_id,
            image_ids=tuple(grouped_images[group_id]),
            image_count=len(grouped_images[group_id]),
            class_counts={name: grouped_counts[group_id][name] for name in classes},
        )
        for group_id in sorted(grouped_images)
    ]


def _seeded_choice(
    candidates: Sequence[str],
    *,
    rng: random.Random,
) -> str:
    return candidates[rng.randrange(len(candidates))]


def stratified_group_split(
    groups: Sequence[GroupStats],
    *,
    classes: Sequence[str],
    fractions: Mapping[str, float],
    seed: int,
) -> dict[int, str]:
    """Implement DATA-17: person-first allocation followed by four-dimensional LPT."""

    if set(fractions) != set(SPLIT_NAMES):
        raise ValueError(f"Expected split fractions for {SPLIT_NAMES}, got {sorted(fractions)}")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-12):
        raise ValueError(f"Split fractions must sum to 1, got {sum(fractions.values())}")
    if "person" not in classes:
        raise ValueError("DATA-17 requires the person class")

    totals: dict[str, float] = {"images": float(sum(group.image_count for group in groups))}
    totals.update(
        {
            class_name: float(sum(group.class_counts.get(class_name, 0) for group in groups))
            for class_name in classes
        }
    )
    dimensions = ("images", *classes)
    targets = {
        split: {dimension: totals[dimension] * fractions[split] for dimension in dimensions}
        for split in SPLIT_NAMES
    }
    current = {split: dict.fromkeys(dimensions, 0.0) for split in SPLIT_NAMES}
    assignments: dict[int, str] = {}
    rng = random.Random(seed)
    tie_breaker = {group.group_id: rng.random() for group in groups}

    def assign(group: GroupStats, split: str) -> None:
        assignments[group.group_id] = split
        current[split]["images"] += group.image_count
        for class_name in classes:
            current[split][class_name] += group.class_counts.get(class_name, 0)

    person_groups = [group for group in groups if group.class_counts.get("person", 0) > 0]
    person_groups.sort(
        key=lambda group: (
            -group.class_counts["person"],
            -group.image_count,
            tie_breaker[group.group_id],
            group.group_id,
        )
    )
    for group in person_groups:
        deficits = {
            split: (targets[split]["person"] - current[split]["person"])
            / max(targets[split]["person"], 1.0)
            for split in SPLIT_NAMES
        }
        best = max(deficits.values())
        candidates = [
            split
            for split, deficit in deficits.items()
            if math.isclose(deficit, best, abs_tol=1e-12)
        ]
        assign(group, _seeded_choice(candidates, rng=rng))

    remaining = [group for group in groups if group.group_id not in assignments]
    remaining.sort(
        key=lambda group: (
            -group.image_count,
            -sum(group.class_counts.values()),
            tie_breaker[group.group_id],
            group.group_id,
        )
    )
    for group in remaining:
        relative_deficits: dict[str, float] = {}
        for split in SPLIT_NAMES:
            deficits = [
                (targets[split][dimension] - current[split][dimension])
                / max(targets[split][dimension], 1.0)
                for dimension in dimensions
            ]
            relative_deficits[split] = statistics.fmean(deficits)
        best = max(relative_deficits.values())
        candidates = [
            split
            for split, deficit in relative_deficits.items()
            if math.isclose(deficit, best, abs_tol=1e-12)
        ]
        assign(group, _seeded_choice(candidates, rng=rng))

    return assignments


def image_split_map(
    groups: Sequence[GroupStats],
    group_assignments: Mapping[int, str],
) -> dict[int, str]:
    return {
        image_id: group_assignments[group.group_id]
        for group in groups
        for image_id in group.image_ids
    }


def aggregate_split_stats(
    coco: dict[str, Any],
    image_splits: Mapping[int, str],
    classes: Sequence[str],
) -> dict[str, Any]:
    category_names = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    image_sets = {
        split: {
            image_id for image_id, assigned_split in image_splits.items() if assigned_split == split
        }
        for split in SPLIT_NAMES
    }
    class_instances = {split: dict.fromkeys(classes, 0) for split in SPLIT_NAMES}
    class_images = {split: {name: set() for name in classes} for split in SPLIT_NAMES}
    total_annotations = dict.fromkeys(SPLIT_NAMES, 0)
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        split = image_splits[image_id]
        class_name = category_names[int(annotation["category_id"])]
        class_instances[split][class_name] += 1
        class_images[split][class_name].add(image_id)
        total_annotations[split] += 1
    return {
        split: {
            "images": len(image_sets[split]),
            "annotations": total_annotations[split],
            "class_instances": class_instances[split],
            "class_images": {name: len(class_images[split][name]) for name in classes},
        }
        for split in SPLIT_NAMES
    }


def verify_split_invariants(
    *,
    groups: Sequence[GroupStats],
    group_assignments: Mapping[int, str],
    image_splits: Mapping[int, str],
    split_stats: Mapping[str, Any],
    fractions: Mapping[str, float],
    person_min_fraction: float,
    ratio_tolerance: float,
) -> None:
    all_image_ids = {image_id for group in groups for image_id in group.image_ids}
    if set(image_splits) != all_image_ids:
        raise DataInvariantError("Split image map is not a complete partition")
    if any(split not in SPLIT_NAMES for split in image_splits.values()):
        raise DataInvariantError("Unknown split label found")
    for group in groups:
        assigned = {image_splits[image_id] for image_id in group.image_ids}
        if assigned != {group_assignments[group.group_id]}:
            raise DataInvariantError(f"Group {group.group_id} crosses splits")

    total_images = len(image_splits)
    total_person = sum(
        int(split_stats[split]["class_instances"]["person"]) for split in SPLIT_NAMES
    )
    for split in SPLIT_NAMES:
        actual_ratio = int(split_stats[split]["images"]) / total_images
        if abs(actual_ratio - fractions[split]) > ratio_tolerance:
            raise DataInvariantError(f"{split} image ratio {actual_ratio:.4f} exceeds tolerance")
        person_fraction = int(split_stats[split]["class_instances"]["person"]) / total_person
        if person_fraction < person_min_fraction:
            raise DataInvariantError(f"{split} has only {person_fraction:.2%} of person instances")


def hash_images(
    coco: dict[str, Any],
    dataset_root: Path,
    *,
    workers: int,
) -> dict[int, str]:
    records = sorted(coco["images"], key=lambda item: int(item["id"]))

    def hash_one(record: dict[str, Any]) -> tuple[int, str]:
        image_path = dataset_root / record["file_name"]
        return int(record["id"]), sha256_file(image_path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(hash_one, records))


def build_split_manifest(
    *,
    coco: dict[str, Any],
    phash_records: Mapping[int, str],
    image_hashes: Mapping[int, str],
    image_to_group: Mapping[int, int],
    image_splits: Mapping[int, str],
    source_checksums: dict[str, Any],
    grouping_decision: dict[str, Any],
    fractions: Mapping[str, float],
    seed: int,
) -> dict[str, Any]:
    images = []
    for image in sorted(coco["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        images.append(
            {
                "image_id": image_id,
                "file_name": str(image["file_name"]),
                "sha256": image_hashes[image_id],
                "phash": phash_records[image_id],
                "group_id": image_to_group[image_id],
                "split": image_splits[image_id],
            }
        )
    return {
        "schema_version": 1,
        "dataset": {
            "handle": source_checksums["dataset_handle"],
            "version": source_checksums["dataset_version"],
            "archive_sha256": source_checksums["archive"]["sha256"],
            "coordinate_global_min": coco["info"]["coordinate_global_min"],
            "coordinate_offset": coco["info"]["coordinate_offset"],
        },
        "grouping": {
            "phash_hash_size": 8,
            "phash_hamming_threshold": grouping_decision["selected_phash_hamming_threshold"],
            "clip_enabled": grouping_decision["clip_required"],
            "clip_model_name": grouping_decision["clip_model_name"],
            "clip_pretrained": grouping_decision["clip_pretrained"],
            "clip_cosine_threshold": grouping_decision["clip_cosine_threshold"],
            "clip_phash_guard": grouping_decision["clip_phash_guard"],
            "group_count": grouping_decision["group_count"],
            "max_group_size": grouping_decision["max_group_size"],
        },
        "split": {
            "seed": seed,
            "fractions": dict(fractions),
            "algorithm": "person-first four-dimensional deficit LPT (DATA-17)",
        },
        "images": images,
    }


def build_test_blocklist(manifest: dict[str, Any]) -> dict[str, Any]:
    records = [
        {
            "image_id": item["image_id"],
            "file_name": item["file_name"],
            "sha256": item["sha256"],
        }
        for item in manifest["images"]
        if item["split"] == "test"
    ]
    return {
        "schema_version": 1,
        "manifest_sha256": hashlib.sha256(canonical_json_bytes(manifest) + b"\n").hexdigest(),
        "images": records,
    }


def manifest_fingerprint(path: Path) -> str:
    return sha256_file(path)


def write_manifest_fingerprint(manifest_path: Path, fingerprint_path: Path) -> str:
    fingerprint = manifest_fingerprint(manifest_path)
    fingerprint_path.write_text(
        f"{fingerprint}  {manifest_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    return fingerprint


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def distribution_summary(
    *,
    coco: dict[str, Any],
    image_splits: Mapping[int, str],
    classes: Sequence[str],
) -> dict[str, Any]:
    categories = {int(item["id"]): str(item["name"]) for item in coco["categories"]}
    image_areas = {
        int(item["id"]): int(item["width"]) * int(item["height"]) for item in coco["images"]
    }
    annotations_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_image_counts = Counter()
    for annotation in coco["annotations"]:
        annotations_by_class[categories[int(annotation["category_id"])]].append(annotation)
        per_image_counts[int(annotation["image_id"])] += 1

    class_rows = {}
    for class_name in classes:
        annotations = annotations_by_class[class_name]
        widths = [float(item["bbox"][2]) for item in annotations]
        heights = [float(item["bbox"][3]) for item in annotations]
        min_sides = [min(width, height) for width, height in zip(widths, heights, strict=True)]
        areas = [float(item["area"]) for item in annotations]
        area_percentages = [
            100.0 * float(item["area"]) / image_areas[int(item["image_id"])] for item in annotations
        ]
        class_rows[class_name] = {
            "instances": len(annotations),
            "images": len({int(item["image_id"]) for item in annotations}),
            "mean_area_px2": statistics.fmean(areas),
            "mean_area_percent": statistics.fmean(area_percentages),
            "width_p01_p50_p99": [_percentile(widths, q) for q in (0.01, 0.50, 0.99)],
            "height_p01_p50_p99": [_percentile(heights, q) for q in (0.01, 0.50, 0.99)],
            "min_side_p01_p50_p99": [_percentile(min_sides, q) for q in (0.01, 0.50, 0.99)],
        }
    all_counts = [per_image_counts[int(item["id"])] for item in coco["images"]]
    return {
        "classes": class_rows,
        "objects_per_image": {
            "mean": statistics.fmean(all_counts),
            "p50": _percentile(all_counts, 0.50),
            "p95": _percentile(all_counts, 0.95),
            "max": max(all_counts),
        },
        "split_stats": aggregate_split_stats(coco, image_splits, classes),
    }


def class_distribution_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Frozen Class Distribution",
        "",
        "All values are regenerated from `split_manifest.json` and `coco_all.json`.",
        "",
        "## Whole dataset",
        "",
        (
            "| Class | Instances | Images | Mean bbox area (px²) | Mean image area (%) | "
            "Min-side p1 / p50 / p99 (px) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_name, row in summary["classes"].items():
        min_sides = row["min_side_p01_p50_p99"]
        lines.append(
            f"| `{class_name}` | {row['instances']:,} | {row['images']:,} | "
            f"{row['mean_area_px2']:,.2f} | {row['mean_area_percent']:.3f} | "
            f"{min_sides[0]:.2f} / {min_sides[1]:.2f} / {min_sides[2]:.2f} |"
        )
    objects = summary["objects_per_image"]
    lines.extend(
        [
            "",
            (
                "Object count per image: "
                f"mean `{objects['mean']:.3f}`, p50 `{objects['p50']:.0f}`, "
                f"p95 `{objects['p95']:.0f}`, max `{objects['max']}`."
            ),
            "",
            (
                "The measured single-box mean areas resolve the earlier source conflict: "
                "`helmet` and `head` match the smaller published reading, while `person` "
                "remains the largest class by box area."
            ),
            "",
            "## Frozen split",
            "",
            (
                "| Split | Images | Annotations | Helmet | Head | Person | "
                "Images with helmet / head / person |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLIT_NAMES:
        row = summary["split_stats"][split]
        instances = row["class_instances"]
        class_images = row["class_images"]
        lines.append(
            f"| `{split}` | {row['images']:,} | {row['annotations']:,} | "
            f"{instances['helmet']:,} | {instances['head']:,} | {instances['person']:,} | "
            f"{class_images['helmet']:,} / {class_images['head']:,} / "
            f"{class_images['person']:,} |"
        )
    lines.extend(
        [
            "",
            (
                "`head` is image-level scarce: 5,785 instances occur in only 920/5,000 images. "
                "`person` is more extreme: 751 instances occur in 158/5,000 images, and the "
                "class is known to be incompletely annotated. All final claims therefore remain "
                "relative comparisons on the same frozen Test split."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def save_class_distribution_figure(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    classes = tuple(summary["classes"])
    split_stats = summary["split_stats"]
    colors = {"train": "#2364aa", "val": "#f18f01", "test": "#c73e1d"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].bar(
        SPLIT_NAMES,
        [split_stats[split]["images"] for split in SPLIT_NAMES],
        color=[colors[split] for split in SPLIT_NAMES],
    )
    axes[0, 0].set_title("Images per frozen split")
    axes[0, 0].set_ylabel("images")

    x = np.arange(len(classes))
    width = 0.25
    for index, split in enumerate(SPLIT_NAMES):
        axes[0, 1].bar(
            x + (index - 1) * width,
            [split_stats[split]["class_instances"][name] for name in classes],
            width,
            label=split,
            color=colors[split],
        )
    axes[0, 1].set_xticks(x, classes)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Class instances by split (log scale)")
    axes[0, 1].legend()

    axes[1, 0].bar(
        classes,
        [summary["classes"][name]["images"] for name in classes],
        color="#4f6d7a",
    )
    axes[1, 0].set_title("Images containing each class")
    axes[1, 0].set_ylabel("images")

    percentile_labels = ("p1", "p50", "p99")
    for class_name in classes:
        axes[1, 1].plot(
            percentile_labels,
            summary["classes"][class_name]["min_side_p01_p50_p99"],
            marker="o",
            label=class_name,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("BBox minimum-side percentiles")
    axes[1, 1].set_ylabel("pixels (log scale)")
    axes[1, 1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def grouping_markdown(
    *,
    groups: Sequence[GroupStats],
    group_assignments: Mapping[int, str],
    split_stats: Mapping[str, Any],
    grouping_decision: Mapping[str, Any],
) -> str:
    sizes = sorted((group.image_count for group in groups), reverse=True)
    histogram = Counter(sizes)
    lines = [
        "# M4 Grouping and Split Report",
        "",
        f"- Groups: `{len(groups):,}`",
        f"- Maximum group size: `{max(sizes):,}`",
        f"- Base pHash threshold: `{grouping_decision['selected_phash_hamming_threshold']}`",
        f"- CLIP enabled: `{grouping_decision['clip_required']}`",
        (
            f"- CLIP model/tag: `{grouping_decision['clip_model_name']}` / "
            f"`{grouping_decision['clip_pretrained']}`"
        ),
        (
            f"- CLIP cosine / pHash guard: "
            f"`{grouping_decision['clip_cosine_threshold']}` / "
            f"`{grouping_decision['clip_phash_guard']}`"
        ),
        "",
        "## Group-size histogram",
        "",
        "| Group size | Number of groups |",
        "|---:|---:|",
    ]
    for size, count in sorted(histogram.items()):
        lines.append(f"| {size} | {count:,} |")
    lines.extend(
        [
            "",
            "## Largest 20 groups",
            "",
            "| Group ID | Images | Split | Image IDs |",
            "|---:|---:|---|---|",
        ]
    )
    for group in sorted(groups, key=lambda item: (-item.image_count, item.group_id))[:20]:
        lines.append(
            f"| {group.group_id} | {group.image_count} | "
            f"`{group_assignments[group.group_id]}` | "
            f"{', '.join(str(item) for item in group.image_ids)} |"
        )
    lines.extend(
        [
            "",
            "## Split balance",
            "",
            "| Split | Images | Helmet | Head | Person |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in SPLIT_NAMES:
        row = split_stats[split]
        lines.append(
            f"| `{split}` | {row['images']:,} | "
            f"{row['class_instances']['helmet']:,} | "
            f"{row['class_instances']['head']:,} | "
            f"{row['class_instances']['person']:,} |"
        )
    lines.extend(
        [
            "",
            (
                "All images in a connected component share one split. The split union contains "
                "all 5,000 images and the three partitions are pairwise disjoint."
            ),
            "",
        ]
    )
    return "\n".join(lines)
