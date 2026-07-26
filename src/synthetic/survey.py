"""Frozen-Train SAM2 H2 spike and resumable Pass 1 survey orchestration."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.data.paths import ProjectPaths
from src.synthetic.mask_ops import decode_rle, encode_rle, quality_failures
from src.synthetic.sam2_runner import MaskPrediction, Sam2BoxSegmenter, xywh_to_xyxy

H2_MODES = ("full", "crop_1024", "crop_512")
H2_BINS = ("very_small", "medium", "larger")
PASS1_SCHEMA_VERSION = 2


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_compose_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return config


def train_context(paths: ProjectPaths) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    dict[int, str],
    dict[int, dict[str, Any]],
]:
    """Load COCO plus frozen Train metadata and reject any split drift."""

    coco = load_json(paths.interim / "coco_all.json")
    split_manifest = load_json(paths.splits / "split_manifest.json")
    frozen = {int(item["image_id"]): item for item in split_manifest["images"]}
    images = {int(item["id"]): item for item in coco["images"]}
    if set(images) != set(frozen):
        raise RuntimeError("Frozen split image IDs no longer match coco_all.json")
    train_images = {
        image_id: image
        for image_id, image in images.items()
        if frozen[image_id]["split"] == "train"
    }
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in train_images:
            annotations[image_id].append(annotation)
    for image_annotations in annotations.values():
        image_annotations.sort(key=lambda item: int(item["id"]))
    category_names = {
        int(category["id"]): str(category["name"]) for category in coco["categories"]
    }
    return coco, train_images, annotations, category_names, frozen


def select_h2_candidates(
    annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    category_names: Mapping[int, str],
    *,
    per_bin: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select exactly three equal-count shortest-side strata, class-aware when possible."""

    instances: list[dict[str, Any]] = []
    for image_id, image_annotations in annotations_by_image.items():
        for annotation in image_annotations:
            width, height = (float(value) for value in annotation["bbox"][2:4])
            instances.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "image_id": int(image_id),
                    "category_id": int(annotation["category_id"]),
                    "class_name": category_names[int(annotation["category_id"])],
                    "bbox": [float(value) for value in annotation["bbox"]],
                    "min_side_px": min(width, height),
                }
            )
    if len(instances) < per_bin * len(H2_BINS):
        raise ValueError("Not enough Train annotations for H2")
    instances.sort(key=lambda item: (item["min_side_px"], item["annotation_id"]))
    thirds = np.array_split(np.arange(len(instances)), len(H2_BINS))
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for bin_name, indexes in zip(H2_BINS, thirds, strict=True):
        pool = [instances[int(index)] for index in indexes]
        by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in pool:
            by_class[item["class_name"]].append(item)
        for values in by_class.values():
            rng.shuffle(values)

        chosen: list[dict[str, Any]] = []
        chosen_ids: set[int] = set()
        minimum_per_class = min(4, per_bin // max(len(by_class), 1))
        for class_name in sorted(by_class):
            for item in by_class[class_name][:minimum_per_class]:
                chosen.append(item)
                chosen_ids.add(item["annotation_id"])
        remaining = [item for item in pool if item["annotation_id"] not in chosen_ids]
        rng.shuffle(remaining)
        chosen.extend(remaining[: per_bin - len(chosen)])
        if len(chosen) != per_bin:
            raise RuntimeError(f"H2 selection failed for {bin_name}")
        chosen.sort(key=lambda item: (item["min_side_px"], item["annotation_id"]))
        for item in chosen:
            selected.append({**item, "size_bin": bin_name})
    return selected


def _prediction_record(prediction: MaskPrediction) -> dict[str, Any]:
    return {
        "iou_score": prediction.iou_score,
        "object_score_logit": prediction.object_score_logit,
        "metrics": prediction.metrics,
        "segmentation": encode_rle(prediction.mask),
    }


def run_h2(
    *,
    paths: ProjectPaths,
    config: Mapping[str, Any],
    segmenter: Sam2BoxSegmenter,
    output_json: Path,
    figure_dir: Path,
) -> dict[str, Any]:
    """Run the 60×3 H2 comparison and save masks, metrics, and three-column grids."""

    _, images, annotations, category_names, _ = train_context(paths)
    candidates = select_h2_candidates(annotations, category_names, seed=paths.seed)
    crop_config = config["sam2"]["pass2_bank"]
    records: list[dict[str, Any]] = []
    for candidate in tqdm(candidates, desc="SAM2 H2", unit="box"):
        image_record = images[candidate["image_id"]]
        image = Image.open(paths.hardhat_raw / image_record["file_name"]).convert("RGB")
        box_xyxy = xywh_to_xyxy(candidate["bbox"])
        modes = {
            "full": segmenter.predict_full(image, [box_xyxy])[0],
            "crop_1024": segmenter.predict_crop(
                image,
                box_xyxy,
                context_pad_frac=float(crop_config["context_pad_frac"]),
                min_crop_side_px=int(crop_config["min_crop_side_px"]),
                target_size=1024,
            ),
            "crop_512": segmenter.predict_crop(
                image,
                box_xyxy,
                context_pad_frac=float(crop_config["context_pad_frac"]),
                min_crop_side_px=int(crop_config["min_crop_side_px"]),
                target_size=512,
            ),
        }
        records.append(
            {
                **candidate,
                "file_name": image_record["file_name"],
                "modes": {name: _prediction_record(value) for name, value in modes.items()},
            }
        )

    result = {
        "schema_version": 1,
        "seed": paths.seed,
        "model_id": config["sam2"]["model_id"],
        "modes": list(H2_MODES),
        "records": records,
        "summary": summarize_h2(records),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    render_h2_grids(records, paths=paths, figure_dir=figure_dir)
    return result


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        f"p{percentile}": float(np.percentile(array, percentile))
        for percentile in (10, 50, 90)
    }


def summarize_h2(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate H2 confidence and structural metrics by size stratum and mode."""

    summary: dict[str, Any] = {}
    for bin_name in H2_BINS:
        summary[bin_name] = {}
        bin_records = [record for record in records if record["size_bin"] == bin_name]
        summary[bin_name]["min_side_range_px"] = [
            min(float(record["min_side_px"]) for record in bin_records),
            max(float(record["min_side_px"]) for record in bin_records),
        ]
        summary[bin_name]["class_counts"] = dict(
            sorted(Counter(str(record["class_name"]) for record in bin_records).items())
        )
        for mode in H2_MODES:
            summary[bin_name][mode] = {
                metric: _percentiles(
                    float(record["modes"][mode]["metrics"][metric])
                    for record in bin_records
                )
                for metric in (
                    "iou_score",
                    "object_score_logit",
                    "mask_to_box_coverage",
                    "component_count",
                    "solidity",
                )
            }
    return summary


def _overlay_cell(
    image: Image.Image,
    box_xywh: Sequence[float],
    mask: np.ndarray,
    *,
    title: str,
    subtitle: str,
    size: int = 256,
) -> Image.Image:
    """Render a contextual crop with cyan box and translucent magenta mask."""

    x, y, width, height = (float(value) for value in box_xywh)
    pad = max(width, height) * 1.1
    left = int(np.floor(x - pad))
    top = int(np.floor(y - pad))
    right = int(np.ceil(x + width + pad))
    bottom = int(np.ceil(y + height + pad))
    crop = image.crop((left, top, right, bottom)).resize((size, size), Image.Resampling.BICUBIC)
    mask_crop = Image.fromarray(mask.astype(np.uint8) * 255).crop(
        (left, top, right, bottom)
    ).resize((size, size), Image.Resampling.NEAREST)
    rgba = crop.convert("RGBA")
    tint = Image.new("RGBA", (size, size), (255, 0, 255, 0))
    tint.putalpha(mask_crop.point(lambda value: 105 if value else 0))
    rgba = Image.alpha_composite(rgba, tint)

    scale_x = size / max(right - left, 1)
    scale_y = size / max(bottom - top, 1)
    draw = ImageDraw.Draw(rgba)
    draw.rectangle(
        (
            (x - left) * scale_x,
            (y - top) * scale_y,
            (x + width - left) * scale_x,
            (y + height - top) * scale_y,
        ),
        outline=(0, 255, 255, 255),
        width=2,
    )
    draw.rectangle((0, 0, size, 30), fill=(0, 0, 0, 190))
    font = ImageFont.load_default()
    draw.text((4, 3), title, fill="white", font=font)
    draw.text((4, 16), subtitle, fill="white", font=font)
    return rgba.convert("RGB")


def render_h2_grids(
    records: Sequence[Mapping[str, Any]], *, paths: ProjectPaths, figure_dir: Path
) -> None:
    """Render one 20-row, three-column comparison grid per H2 size bin."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    cell_size = 256
    for bin_name in H2_BINS:
        bin_records = [record for record in records if record["size_bin"] == bin_name]
        sheet = Image.new("RGB", (cell_size * 3, cell_size * len(bin_records)), "black")
        for row, record in enumerate(bin_records):
            image = Image.open(paths.hardhat_raw / str(record["file_name"])).convert("RGB")
            for column, mode in enumerate(H2_MODES):
                mode_record = record["modes"][mode]
                metrics = mode_record["metrics"]
                title = (
                    f"{mode} | ann={record['annotation_id']} {record['class_name']} "
                    f"side={record['min_side_px']:.0f}"
                )
                subtitle = (
                    f"IoU={metrics['iou_score']:.3f} obj={metrics['object_score_logit']:.2f} "
                    f"cov={metrics['mask_to_box_coverage']:.2f}"
                )
                cell = _overlay_cell(
                    image,
                    record["bbox"],
                    decode_rle(mode_record["segmentation"]),
                    title=title,
                    subtitle=subtitle,
                    size=cell_size,
                )
                sheet.paste(cell, (column * cell_size, row * cell_size))
        sheet.save(figure_dir / f"h2_sam2_{bin_name}.png", optimize=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, record: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run_pass1(
    *,
    paths: ProjectPaths,
    config: Mapping[str, Any],
    segmenter: Sam2BoxSegmenter,
    max_images: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run or resume full-image SAM2 prompting for every frozen Train annotation."""

    _, images, annotations, category_names, frozen = train_context(paths)
    records_dir = paths.masks_pass1 / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    selected_images = sorted(images.items())
    if max_images is not None:
        selected_images = selected_images[:max_images]
    processed = skipped = annotation_count = qc_pass_count = 0

    for image_id, image_record in tqdm(selected_images, desc="SAM2 Pass 1", unit="image"):
        output_path = records_dir / f"{image_id:06d}.json"
        expected_annotations = annotations.get(image_id, [])
        if output_path.exists() and not force:
            existing = load_json(output_path)
            if (
                existing.get("schema_version") == PASS1_SCHEMA_VERSION
                and existing.get("model_id") == config["sam2"]["model_id"]
                and len(existing.get("annotations", [])) == len(expected_annotations)
                and existing.get("image_sha256") == frozen[image_id]["sha256"]
            ):
                skipped += 1
                annotation_count += len(existing["annotations"])
                qc_pass_count += sum(
                    bool(item["qc_pass"]) for item in existing["annotations"]
                )
                continue

        image_path = paths.hardhat_raw / image_record["file_name"]
        image = Image.open(image_path).convert("RGB")
        boxes = [xywh_to_xyxy(annotation["bbox"]) for annotation in expected_annotations]
        predictions = segmenter.predict_full(image, boxes)
        output_annotations: list[dict[str, Any]] = []
        for annotation, prediction in zip(expected_annotations, predictions, strict=True):
            class_name = category_names[int(annotation["category_id"])]
            pass1_config = config["sam2"]["pass1_survey"]
            failures = quality_failures(
                prediction.metrics,
                class_name=class_name,
                config=config,
                min_iou_score=float(pass1_config["min_iou_score"]),
                min_object_score_logit=float(pass1_config["min_object_score_logit"]),
            )
            output_annotations.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "category_id": int(annotation["category_id"]),
                    "class_name": class_name,
                    "bbox": [float(value) for value in annotation["bbox"]],
                    "segmentation": encode_rle(prediction.mask),
                    "metrics": prediction.metrics,
                    "qc_failures": failures,
                    "qc_pass": not failures,
                }
            )
        record = {
            "schema_version": PASS1_SCHEMA_VERSION,
            "model_id": config["sam2"]["model_id"],
            "image_id": image_id,
            "file_name": image_record["file_name"],
            "image_sha256": frozen[image_id]["sha256"],
            "group_id": int(frozen[image_id]["group_id"]),
            "annotations": output_annotations,
        }
        _write_json_atomic(output_path, record)
        processed += 1
        annotation_count += len(output_annotations)
        qc_pass_count += sum(bool(item["qc_pass"]) for item in output_annotations)

    summary = rebuild_pass1_index(paths=paths, config=config)
    summary.update(
        {
            "run_processed_images": processed,
            "run_skipped_images": skipped,
            "run_selected_images": len(selected_images),
            "run_selected_annotations": annotation_count,
            "run_selected_qc_pass": qc_pass_count,
        }
    )
    return summary


def rebuild_pass1_index(
    *, paths: ProjectPaths, config: Mapping[str, Any], recheck_qc: bool = False
) -> dict[str, Any]:
    """Rebuild the deterministic JSONL index and optionally apply new QC thresholds."""

    records_dir = paths.masks_pass1 / "records"
    index_path = paths.masks_pass1 / "manifest.jsonl"
    class_totals: Counter[str] = Counter()
    class_pass: Counter[str] = Counter()
    image_count = annotation_count = 0
    lines: list[str] = []
    for record_path in sorted(records_dir.glob("*.json")):
        record = load_json(record_path)
        if recheck_qc:
            pass1_config = config["sam2"]["pass1_survey"]
            for annotation in record["annotations"]:
                failures = quality_failures(
                    annotation["metrics"],
                    class_name=annotation["class_name"],
                    config=config,
                    min_iou_score=float(pass1_config["min_iou_score"]),
                    min_object_score_logit=float(pass1_config["min_object_score_logit"]),
                )
                annotation["qc_failures"] = failures
                annotation["qc_pass"] = not failures
            _write_json_atomic(record_path, record)
        image_count += 1
        annotation_count += len(record["annotations"])
        for annotation in record["annotations"]:
            class_name = str(annotation["class_name"])
            class_totals[class_name] += 1
            class_pass[class_name] += int(bool(annotation["qc_pass"]))
        lines.append(
            json.dumps(
                {
                    "image_id": record["image_id"],
                    "file_name": record["file_name"],
                    "group_id": record["group_id"],
                    "image_sha256": record["image_sha256"],
                    "record": f"records/{record_path.name}",
                    "record_sha256": _sha256(record_path),
                    "annotation_count": len(record["annotations"]),
                    "qc_pass_count": sum(
                        bool(item["qc_pass"]) for item in record["annotations"]
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return {
        "indexed_images": image_count,
        "indexed_annotations": annotation_count,
        "qc_pass_by_class": dict(sorted(class_pass.items())),
        "total_by_class": dict(sorted(class_totals.items())),
        "manifest_sha256": _sha256(index_path),
    }
