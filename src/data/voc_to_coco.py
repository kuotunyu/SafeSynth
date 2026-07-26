"""Deterministic PASCAL VOC to COCO conversion for Hard Hat Workers."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

EXPECTED_IMAGE_COUNT = 5_000
EXPECTED_ANNOTATION_COUNT = 25_502
EXPECTED_CLASS_INSTANCES = {"helmet": 18_966, "head": 5_785, "person": 751}
EXPECTED_CLASS_IMAGES = {"helmet": 4_581, "head": 920, "person": 158}
LICENSE_URL = "https://creativecommons.org/publicdomain/zero/1.0/"


class DataInvariantError(RuntimeError):
    """Raised when upstream data differs from the frozen dataset facts."""


@dataclass
class ConversionStats:
    """Auditable counters collected while parsing and normalizing VOC records."""

    label_counts: Counter[str] = field(default_factory=Counter)
    class_image_stems: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    difficult_counts: Counter[int] = field(default_factory=Counter)
    truncated_counts: Counter[int] = field(default_factory=Counter)
    corrections: Counter[str] = field(default_factory=Counter)
    xml_size_mismatches: list[str] = field(default_factory=list)
    parse_failures: list[dict[str, str]] = field(default_factory=list)
    unknown_labels: Counter[str] = field(default_factory=Counter)
    raw_coordinates: list[float] = field(default_factory=list)
    boxes_by_class: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    objects_per_image: list[int] = field(default_factory=list)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file in binary mode so Windows newline conversion cannot interfere."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize canonical ASCII JSON for stable manifests and checksums."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def write_canonical_json(path: Path, value: Any) -> None:
    """Write deterministic JSON with LF line endings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value).decode("ascii") + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def discover_pairs(dataset_root: Path) -> list[tuple[Path, Path]]:
    """Pair XML and PNG files by stem, never by filesystem enumeration order."""

    xml_paths = sorted(dataset_root.rglob("*.xml"), key=lambda path: path.as_posix().lower())
    png_paths = sorted(dataset_root.rglob("*.png"), key=lambda path: path.as_posix().lower())

    def index_unique(paths: Iterable[Path], kind: str) -> dict[str, Path]:
        indexed: dict[str, Path] = {}
        duplicates: list[str] = []
        for path in paths:
            key = path.stem.casefold()
            if key in indexed:
                duplicates.append(path.stem)
            indexed[key] = path
        if duplicates:
            raise DataInvariantError(f"Duplicate {kind} stems: {sorted(duplicates)[:20]}")
        return indexed

    xml_by_stem = index_unique(xml_paths, "XML")
    png_by_stem = index_unique(png_paths, "PNG")
    if xml_by_stem.keys() != png_by_stem.keys():
        missing_xml = sorted(png_by_stem.keys() - xml_by_stem.keys())
        missing_png = sorted(xml_by_stem.keys() - png_by_stem.keys())
        raise DataInvariantError(
            "XML/PNG stem mismatch. "
            f"Missing XML ({len(missing_xml)}): {missing_xml[:20]}; "
            f"missing PNG ({len(missing_png)}): {missing_png[:20]}"
        )
    return [(xml_by_stem[key], png_by_stem[key]) for key in sorted(xml_by_stem)]


def _parse_float(text: str | None, *, field_name: str, xml_path: Path) -> float:
    if text is None:
        raise ValueError(f"Missing {field_name} in {xml_path.name}")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field_name}={text!r} in {xml_path.name}")
    return value


def _parse_flag(text: str | None) -> int:
    value = int(text or "0")
    return 1 if value else 0


def _percentile(values: Sequence[float], probability: float) -> float:
    """Linear percentile compatible with NumPy's default for report-only statistics."""

    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _find_global_min(pairs: Sequence[tuple[Path, Path]], stats: ConversionStats) -> float:
    """Parse every raw coordinate once to select the DATA-05 coordinate offset."""

    for xml_path, _ in pairs:
        try:
            root = ET.parse(xml_path).getroot()
            for obj in root.findall("object"):
                box = obj.find("bndbox")
                if box is None:
                    continue
                for name in ("xmin", "ymin", "xmax", "ymax"):
                    stats.raw_coordinates.append(
                        _parse_float(box.findtext(name), field_name=name, xml_path=xml_path)
                    )
        except (ET.ParseError, OSError, ValueError) as exc:
            stats.parse_failures.append({"file": xml_path.name, "error": str(exc)})
    if stats.parse_failures:
        raise DataInvariantError(
            f"Unable to inspect {len(stats.parse_failures)} XML files; see parse_failures.json"
        )
    if not stats.raw_coordinates:
        raise DataInvariantError("No bounding-box coordinates found")
    return min(stats.raw_coordinates)


def detect_coordinate_offset(global_min: float) -> tuple[int, str]:
    """Apply the runtime-only coordinate indexing decision required by DATA-05."""

    if global_min == 0:
        return 0, "global minimum is 0; treating coordinates as zero-based"
    if global_min == 1:
        return 1, "global minimum is 1; converting VOC one-based minima to zero-based"
    return 0, (
        f"unexpected global minimum {global_min:g}; defaulting to zero offset and recording warning"
    )


def _normalize_box(
    box: ET.Element,
    *,
    xml_path: Path,
    image_width: int,
    image_height: int,
    coordinate_offset: int,
    stats: ConversionStats,
) -> tuple[float, float, float, float] | None:
    raw = {
        name: _parse_float(box.findtext(name), field_name=name, xml_path=xml_path)
        for name in ("xmin", "ymin", "xmax", "ymax")
    }
    rounded = {name: round(value) for name, value in raw.items()}
    if any(not value.is_integer() for value in raw.values()):
        stats.corrections["float_coordinates_rounded"] += 1

    if rounded["xmin"] > rounded["xmax"]:
        rounded["xmin"], rounded["xmax"] = rounded["xmax"], rounded["xmin"]
        stats.corrections["x_coordinates_swapped"] += 1
    if rounded["ymin"] > rounded["ymax"]:
        rounded["ymin"], rounded["ymax"] = rounded["ymax"], rounded["ymin"]
        stats.corrections["y_coordinates_swapped"] += 1

    # DATA-06: x/y minima shift for one-based VOC; maxima remain the COCO-exclusive edge.
    x0 = rounded["xmin"] - coordinate_offset
    y0 = rounded["ymin"] - coordinate_offset
    x1 = rounded["xmax"]
    y1 = rounded["ymax"]
    clipped = (
        min(max(x0, 0), image_width),
        min(max(y0, 0), image_height),
        min(max(x1, 0), image_width),
        min(max(y1, 0), image_height),
    )
    if clipped != (x0, y0, x1, y1):
        stats.corrections["boxes_clipped_to_image"] += 1
    x0, y0, x1, y1 = clipped
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        stats.corrections["zero_area_boxes_dropped"] += 1
        return None
    return float(x0), float(y0), float(width), float(height)


def convert_voc_dataset(
    dataset_root: Path,
    *,
    classes: Sequence[str],
    kaggle_handle: str,
    kaggle_version: int,
    archive_sha256: str,
) -> tuple[dict[str, Any], ConversionStats, dict[str, Any]]:
    """Convert a paired VOC tree into deterministic COCO and an audit summary."""

    pairs = discover_pairs(dataset_root)
    stats = ConversionStats()
    global_min = _find_global_min(pairs, stats)
    coordinate_offset, offset_message = detect_coordinate_offset(global_min)
    class_to_id = {name: index + 1 for index, name in enumerate(classes)}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, (xml_path, png_path) in enumerate(pairs, start=1):
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            stats.parse_failures.append({"file": xml_path.name, "error": str(exc)})
            continue

        with Image.open(png_path) as image:
            image_width, image_height = image.size
        xml_width_text = root.findtext("size/width")
        xml_height_text = root.findtext("size/height")
        try:
            xml_size = (int(xml_width_text or "0"), int(xml_height_text or "0"))
        except ValueError:
            xml_size = (0, 0)
        if xml_size != (image_width, image_height):
            stats.xml_size_mismatches.append(xml_path.name)

        try:
            relative_name = png_path.relative_to(dataset_root).as_posix()
        except ValueError as exc:
            raise DataInvariantError(f"{png_path} is not inside {dataset_root}") from exc
        images.append(
            {
                "id": image_id,
                "file_name": relative_name,
                "width": image_width,
                "height": image_height,
            }
        )

        image_annotation_count = 0
        for obj in root.findall("object"):
            label = (obj.findtext("name") or "").strip().lower()
            if label not in class_to_id:
                stats.unknown_labels[label or "<empty>"] += 1
                continue
            box_element = obj.find("bndbox")
            if box_element is None:
                stats.corrections["objects_missing_bndbox"] += 1
                continue
            try:
                bbox = _normalize_box(
                    box_element,
                    xml_path=xml_path,
                    image_width=image_width,
                    image_height=image_height,
                    coordinate_offset=coordinate_offset,
                    stats=stats,
                )
            except ValueError as exc:
                stats.parse_failures.append({"file": xml_path.name, "error": str(exc)})
                continue
            if bbox is None:
                continue

            difficult = _parse_flag(obj.findtext("difficult"))
            truncated = _parse_flag(obj.findtext("truncated"))
            stats.label_counts[label] += 1
            stats.class_image_stems[label].add(png_path.stem)
            stats.difficult_counts[difficult] += 1
            stats.truncated_counts[truncated] += 1
            stats.boxes_by_class[label].append((bbox[2], bbox[3]))
            image_annotation_count += 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_to_id[label],
                    "bbox": list(bbox),
                    "area": bbox[2] * bbox[3],
                    "segmentation": [],
                    "iscrowd": 0,
                    "difficult": difficult,
                    "truncated": truncated,
                }
            )
            annotation_id += 1
        stats.objects_per_image.append(image_annotation_count)

    if stats.unknown_labels:
        raise DataInvariantError(f"Unknown labels found: {dict(stats.unknown_labels)}")
    if stats.parse_failures:
        raise DataInvariantError(
            f"Failed to parse {len(stats.parse_failures)} records; see parse_failures.json"
        )

    coco = {
        "info": {
            "description": "Hard Hat Workers converted deterministically from PASCAL VOC",
            "version": str(kaggle_version),
            "source": f"{kaggle_handle}/versions/{kaggle_version}",
            "source_archive_sha256": archive_sha256,
            "coordinate_global_min": global_min,
            "coordinate_offset": coordinate_offset,
            "bbox_width_convention": "xmax-xmin without +1 (DATA-06)",
        },
        "licenses": [{"id": 1, "name": "CC0 1.0 Universal", "url": LICENSE_URL}],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name, "supercategory": "ppe"}
            for name, category_id in class_to_id.items()
        ],
    }
    audit = {
        "global_min_coordinate": global_min,
        "coordinate_offset": coordinate_offset,
        "coordinate_offset_message": offset_message,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "class_instances": dict(stats.label_counts),
        "class_images": {name: len(stats.class_image_stems.get(name, set())) for name in classes},
        "difficult_histogram": dict(stats.difficult_counts),
        "truncated_histogram": dict(stats.truncated_counts),
        "xml_size_mismatch_count": len(stats.xml_size_mismatches),
        "corrections": dict(stats.corrections),
    }
    return coco, stats, audit


def assert_expected_facts(coco: dict[str, Any], classes: Sequence[str]) -> dict[str, Any]:
    """Hard-fail if the upstream release differs from DATA-03."""

    categories = {item["id"]: item["name"] for item in coco["categories"]}
    class_instances = Counter(categories[item["category_id"]] for item in coco["annotations"])
    class_images: dict[str, set[int]] = defaultdict(set)
    for item in coco["annotations"]:
        class_images[categories[item["category_id"]]].add(item["image_id"])
    observed_class_images = {name: len(class_images[name]) for name in classes}

    errors: list[str] = []
    if len(coco["images"]) != EXPECTED_IMAGE_COUNT:
        errors.append(f"images: expected {EXPECTED_IMAGE_COUNT}, got {len(coco['images'])}")
    annotation_delta = len(coco["annotations"]) - EXPECTED_ANNOTATION_COUNT
    if abs(annotation_delta) > 1:
        errors.append(
            f"annotations: expected {EXPECTED_ANNOTATION_COUNT} +/- 1, "
            f"got {len(coco['annotations'])}"
        )
    if dict(class_instances) != EXPECTED_CLASS_INSTANCES:
        errors.append(
            f"class instances: expected {EXPECTED_CLASS_INSTANCES}, got {dict(class_instances)}"
        )
    if observed_class_images != EXPECTED_CLASS_IMAGES:
        errors.append(
            f"class images: expected {EXPECTED_CLASS_IMAGES}, got {observed_class_images}"
        )
    if errors:
        raise DataInvariantError("Upstream dataset facts changed:\n- " + "\n- ".join(errors))

    return {
        "image_count": len(coco["images"]),
        "annotation_count": len(coco["annotations"]),
        "annotation_delta_from_documented_total": annotation_delta,
        "class_instances": dict(class_instances),
        "class_images": observed_class_images,
    }


def verify_coco_schema(coco: dict[str, Any]) -> None:
    """Validate the schema invariants that commonly fail silently."""

    if not coco.get("info") or not coco.get("licenses"):
        raise DataInvariantError("COCO must include non-empty info and licenses")
    image_ids = [item["id"] for item in coco["images"]]
    annotation_ids = [item["id"] for item in coco["annotations"]]
    category_ids = {item["id"] for item in coco["categories"]}
    if len(image_ids) != len(set(image_ids)):
        raise DataInvariantError("Duplicate COCO image IDs")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise DataInvariantError("Duplicate COCO annotation IDs")
    if any("\\" in item["file_name"] for item in coco["images"]):
        raise DataInvariantError("COCO file_name contains Windows backslashes")
    valid_image_ids = set(image_ids)
    for annotation in coco["annotations"]:
        if annotation["image_id"] not in valid_image_ids:
            raise DataInvariantError(f"Orphan annotation {annotation['id']}")
        if annotation["category_id"] not in category_ids:
            raise DataInvariantError(f"Unknown category on annotation {annotation['id']}")
        if annotation["iscrowd"] != 0:
            raise DataInvariantError(f"iscrowd must be zero on annotation {annotation['id']}")
        if annotation["segmentation"] != []:
            raise DataInvariantError(
                f"Expected empty segmentation on annotation {annotation['id']}"
            )
        x, y, width, height = annotation["bbox"]
        if min(x, y, width, height) < 0 or width <= 0 or height <= 0:
            raise DataInvariantError(f"Invalid bbox on annotation {annotation['id']}")
        if annotation["area"] != width * height:
            raise DataInvariantError(f"Incorrect area on annotation {annotation['id']}")


def coco_self_evaluation(coco_path: Path) -> float:
    """Evaluate ground truth against itself to catch bbox/schema mismatches."""

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(coco_path))
    detections = [
        {
            "image_id": item["image_id"],
            "category_id": item["category_id"],
            "bbox": item["bbox"],
            "score": 1.0,
        }
        for item in ground_truth.dataset["annotations"]
    ]
    result = ground_truth.loadRes(detections)
    evaluator = COCOeval(ground_truth, result, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    map_value = float(evaluator.stats[0])
    if not math.isclose(map_value, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise DataInvariantError(f"COCO self-evaluation mAP must be 1.000, got {map_value:.12f}")
    return map_value


def _format_number(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:,.2f}"


def build_conversion_report(
    *,
    coco: dict[str, Any],
    stats: ConversionStats,
    audit: dict[str, Any],
    archive_path: Path,
    archive_sha256: str,
    archive_size: int,
    official_total_bytes: int,
    kaggle_version: int,
    self_map: float,
    classes: Sequence[str],
) -> str:
    """Render a tracked Markdown audit report from conversion outputs."""

    lines = [
        "# M2 Conversion Report",
        "",
        "Generated deterministically by `scripts/prepare_data.py`.",
        "",
        "## Source",
        "",
        f"- Kaggle version: `{kaggle_version}`",
        f"- Official uncompressed file total: `{official_total_bytes:,}` bytes",
        f"- Downloaded archive: `{archive_path.name}` (`{archive_size:,}` bytes)",
        f"- Archive SHA256: `{archive_sha256}`",
        f"- Global minimum coordinate: `{audit['global_min_coordinate']:g}`",
        f"- Applied xmin/ymin offset: `{audit['coordinate_offset']}`",
        f"- Decision: {audit['coordinate_offset_message']}",
        "",
        "## Dataset facts",
        "",
        (
            "| Class | Instances | Images | Mean area (px²) | Mean area (%) | "
            "Min-side p1 / p50 / p99 (px) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    image_areas = {item["id"]: item["width"] * item["height"] for item in coco["images"]}
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    annotations_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_class[categories[annotation["category_id"]]].append(annotation)
    for name in classes:
        annotations = annotations_by_class[name]
        areas = [float(item["area"]) for item in annotations]
        area_percentages = [
            100.0 * float(item["area"]) / image_areas[item["image_id"]] for item in annotations
        ]
        min_sides = [min(item["bbox"][2], item["bbox"][3]) for item in annotations]
        lines.append(
            f"| `{name}` | {len(annotations):,} | "
            f"{len(stats.class_image_stems[name]):,} | "
            f"{_format_number(statistics.fmean(areas))} | "
            f"{_format_number(statistics.fmean(area_percentages))} | "
            f"{_format_number(_percentile(min_sides, 0.01))} / "
            f"{_format_number(_percentile(min_sides, 0.50))} / "
            f"{_format_number(_percentile(min_sides, 0.99))} |"
        )
    lines.extend(
        [
            "",
            f"- Images: `{len(coco['images']):,}`",
            f"- Annotations: `{len(coco['annotations']):,}`",
            f"- XML/PNG size mismatches: `{len(stats.xml_size_mismatches)}`",
            f"- Unknown labels: `{sum(stats.unknown_labels.values())}`",
            f"- `iscrowd != 0`: `{sum(item['iscrowd'] != 0 for item in coco['annotations'])}`",
            f"- Difficult histogram: `{dict(sorted(stats.difficult_counts.items()))}`",
            f"- Truncated histogram: `{dict(sorted(stats.truncated_counts.items()))}`",
            "",
            "## Recorded corrections",
            "",
        ]
    )
    if stats.corrections:
        for key, value in sorted(stats.corrections.items()):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Verification",
            "",
            f"- COCO self-evaluation mAP: `{self_map:.3f}`",
            "- Bounding-box convention: `w = xmax - xmin`, without `+1` (DATA-06).",
            "- All paths use forward slashes and all `iscrowd` values are zero.",
            "",
        ]
    )
    return "\n".join(lines)


def copy_file_preserving_bytes(source: Path, destination: Path) -> None:
    """Small seam used by the downloader and isolated in tests."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
