"""Deterministic, provenance-complete synthetic composition preview engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.data.paths import PROJECT_ROOT, ProjectPaths, load_project_paths
from src.filtering.rules import filter_sample
from src.synthetic.composition import (
    Layer,
    alpha_composite,
    annulus_mask,
    apply_postfx,
    box_to_mask,
    decontaminate_soft_edge,
    feather_alpha,
    harmonize_lab,
    inpaint_masked_object,
    match_high_frequency_noise,
    placement_slices,
    poisson_composite,
    recompute_visible_annotations,
    seam_energy_ratio,
    tight_bbox,
    warp_rgba,
)

if TYPE_CHECKING:
    from src.synthetic.generative_inpaint import GenerativeBoundaryInpainter

SCENARIO_ORDER = (
    "small_distant",
    "head_no_helmet",
    "partial_occlusion",
    "crowded",
    "hard_negative",
    "low_light_blur",
)
EXPERIMENTAL_SCENARIOS = ("context_replacement",)

# ---------------------------------------------------------------------------
# ADR-011 scale cap.
#
# H4 did not pass (paste-artifact AUC 0.9159 on the 1x pool against a pre-registered
# maximum of 0.60; M11 measured 0.7964 on 300 images, and the estimate rose once the
# sample grew to 75,106 patches) and this project does not claim otherwise. The consequence was changed
# from "block indefinitely" to "1x is allowed, 2x is forbidden": there is no
# reason to spend a large generation budget on data whose artifacts are known to
# be detectable.
#
# 1x means ACCEPTED samples reaching parity with the 3,500-image real Train split.
#
# The cap that matters is therefore on ACCEPTED samples, not on the candidate pool.
# An earlier version of this constant capped the pool at 6,000 and claimed that
# made 2x unreachable; that conflated the two. The pool is just candidates, and a
# measured 47.4% acceptance rate on the first full run meant a 6,000 pool could
# not even reach 1x.
#
# The guarantee now lives where it belongs: the 1x export takes the first
# TARGET_ACCEPTED_1X accepted samples by stable rank and never more, so 2x is
# impossible regardless of how large the pool is. MAX_POOL_IMAGES is only a
# runaway guard, sized from the measured acceptance rate with headroom.
# ---------------------------------------------------------------------------
TARGET_ACCEPTED_1X = 3_500
# Sized from the measured acceptance rate with headroom. Reweighting the scenarios
# by inverse pass rate (so the POST-FILTER mix hits its target) deliberately shifts
# generation toward the scenarios that survive least often, which drags the overall
# rate down: 41.7% before reweighting, 38.3% after. A 9,000 pool yielded only 3,449
# accepted against the 3,500 needed for 1x, hence 10,000.
MAX_POOL_IMAGES = 10_000


@dataclass
class Paste:
    layer: Layer
    rgba: np.ndarray
    frame_slice: tuple[slice, slice]
    patch_slice: tuple[slice, slice]
    bank: dict[str, Any]
    bbox_preclip: list[float]


@dataclass(frozen=True)
class ReflectedBorderScore:
    """High-confidence reflected-padding score for one image border."""

    pair_mae: float
    pair_correlation: float
    texture_std: float
    pad_px: int
    detected: bool


@dataclass(frozen=True)
class ReflectedAxisScore:
    """Independent reflected-padding scores for both ends of one image axis."""

    start: ReflectedBorderScore
    end: ReflectedBorderScore

    @property
    def detected(self) -> bool:
        return self.start.detected or self.end.detected


@dataclass(frozen=True)
class ReflectedPaddingResult:
    """Pixel-level reflected-padding evidence for both image axes."""

    detected: bool
    detected_axes: tuple[str, ...]
    top_bottom: ReflectedAxisScore
    left_right: ReflectedAxisScore


@dataclass(frozen=True)
class ReflectedPaddingNormalization:
    """Deterministic crop/resize transform used to remove mirrored borders."""

    applied: bool
    crop_xyxy: tuple[int, int, int, int]
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int
    output_width: int
    output_height: int
    detected_sides: tuple[str, ...]


@dataclass(frozen=True)
class ContextReplacementGuardResult:
    """CPU-only eligibility decision for one context-replacement background."""

    accepted: bool
    eligible_annotation_ids: tuple[int, ...]
    anchor_margins: tuple[tuple[int, float, int], ...]
    background_min_headlike_edge_margin_px: float | None
    reflected_padding: ReflectedPaddingResult | None
    reject_reason: str | None


@dataclass(frozen=True)
class PreparedContextBackground:
    """Normalized context-replacement pixels, labels, masks, and guard result."""

    image_rgb: np.ndarray
    annotations: tuple[dict[str, Any], ...]
    pass1: dict[int, dict[str, Any]]
    guard: ContextReplacementGuardResult
    normalization: ReflectedPaddingNormalization


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    path.write_text(text + ("\n" if records else ""), encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_rle(segmentation: Mapping[str, Any]) -> np.ndarray:
    rle = {
        "size": [int(value) for value in segmentation["size"]],
        "counts": str(segmentation["counts"]).encode("ascii"),
    }
    return mask_utils.decode(rle).astype(bool)


def _encode_rle(mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(
        np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    )
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": encoded["counts"].decode("ascii"),
    }


def _phash_hex(image_rgb: np.ndarray) -> str:
    return str(imagehash.phash(Image.fromarray(image_rgb), hash_size=8))


def _hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _sample_seed(root_seed: int, sample_index: int) -> int:
    """Derive a stable independent RNG seed for one sample."""

    digest = hashlib.sha256(f"{root_seed}|{sample_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _generative_seed(root_seed: int, sample_index: int, instance_id: str) -> int:
    """Derive a stable torch-compatible seed for one generative boundary edit."""

    payload = f"generative|{root_seed}|{sample_index}|{instance_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        2**63 - 1
    )


def _archive_existing(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}_stale_{timestamp}")
    suffix = 1
    while target.exists():
        target = path.with_name(f"{path.name}_stale_{timestamp}_{suffix}")
        suffix += 1
    path.replace(target)


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "compose.yaml").read_text(encoding="utf-8")
    )
    filtering = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "filtering.yaml").read_text(encoding="utf-8")
    )
    return compose, filtering


def _load_context(
    paths: ProjectPaths,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    dict[int, dict[str, Any]],
    set[int],
]:
    coco = _read_json(paths.interim / "coco_all.json")
    frozen_manifest = _read_json(paths.splits / "split_manifest.json")
    frozen = {
        int(record["image_id"]): record for record in frozen_manifest["images"]
    }
    train_images = {
        int(record["id"]): record
        for record in coco["images"]
        if frozen[int(record["id"])]["split"] == "train"
    }
    annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in train_images:
            annotations[image_id].append(annotation)
    for records in annotations.values():
        records.sort(key=lambda item: int(item["id"]))
    bank = _read_jsonl(paths.cutouts / "bank_manifest.jsonl")
    test_blocklist = _read_json(paths.splits / "test_blocklist.json")
    test_ids = {int(record["image_id"]) for record in test_blocklist["images"]}
    if test_ids & set(train_images):
        raise RuntimeError("Train backgrounds intersect the frozen Test blocklist")
    hardneg_path = paths.cutouts / "hardneg_bank_manifest.jsonl"
    hardneg_bank = _read_jsonl(hardneg_path) if hardneg_path.exists() else []
    for item in (*bank, *hardneg_bank):
        if str(item["src_split"]) != "train" or int(item["src_image_id"]) in test_ids:
            raise RuntimeError("Cutout bank contains a Val/Test source")
    if any(item.get("annotated", False) for item in hardneg_bank):
        raise RuntimeError("Hard negatives must never be annotated (ADR-004)")
    return coco, bank, hardneg_bank, train_images, annotations, frozen, test_ids


def _load_pass1(
    paths: ProjectPaths, image_id: int
) -> dict[int, dict[str, Any]]:
    path = paths.masks_pass1 / "records" / f"{image_id:06d}.json"
    record = _read_json(path)
    return {
        int(annotation["annotation_id"]): annotation
        for annotation in record["annotations"]
    }


def _bbox_edge_margin(
    bbox_xywh: Sequence[float],
    *,
    image_shape: tuple[int, int],
) -> float:
    """Return the smallest half-open distance from a box to the image frame."""

    height, width = image_shape
    x, y, box_width, box_height = (float(value) for value in bbox_xywh)
    return min(
        x,
        y,
        float(width) - (x + box_width),
        float(height) - (y + box_height),
    )


def _flat_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = np.asarray(first, dtype=np.float32).ravel()
    second_flat = np.asarray(second, dtype=np.float32).ravel()
    first_centered = first_flat - float(first_flat.mean())
    second_centered = second_flat - float(second_flat.mean())
    denominator = float(
        np.sqrt(
            np.dot(first_centered, first_centered)
            * np.dot(second_centered, second_centered)
        )
    )
    if denominator <= 1e-12:
        return -1.0
    return float(np.dot(first_centered, second_centered) / denominator)


def _reflected_axis_score(
    grayscale: np.ndarray,
    *,
    guard_config: Mapping[str, Any],
) -> ReflectedAxisScore:
    image = np.asarray(grayscale, dtype=np.float32)
    length = image.shape[0]
    orthogonal_size = int(guard_config["orthogonal_sample_size"])
    sampled = cv2.resize(
        image,
        (orthogonal_size, length),
        interpolation=cv2.INTER_AREA,
    )
    min_pad = int(guard_config["min_pad_px"])
    max_pad = min(
        int(np.floor(length * float(guard_config["max_pad_fraction"]))),
        (length - 1) // 2,
    )
    if min_pad > max_pad:
        raise ValueError("Invalid reflected-padding search bounds")

    def border_score(*, start: bool) -> ReflectedBorderScore:
        candidates: list[tuple[float, float, float, int]] = []
        for pad in range(min_pad, max_pad + 1):
            if start:
                border = sampled[:pad]
                interior = sampled[pad : 2 * pad][::-1]
            else:
                border = sampled[-pad:]
                interior = sampled[-2 * pad : -pad][::-1]
            candidates.append(
                (
                    float(np.abs(border - interior).mean()),
                    _flat_correlation(border, interior),
                    float(border.std()),
                    pad,
                )
            )
        pair_mae, pair_correlation, texture_std, pad_px = min(
            candidates,
            key=lambda item: item[0],
        )
        detected = (
            pair_mae <= float(guard_config["max_pair_mae"])
            and pair_correlation
            >= float(guard_config["min_pair_correlation"])
            and texture_std >= float(guard_config["min_texture_std"])
        )
        return ReflectedBorderScore(
            pair_mae=pair_mae,
            pair_correlation=pair_correlation,
            texture_std=texture_std,
            pad_px=pad_px,
            detected=detected,
        )

    return ReflectedAxisScore(
        start=border_score(start=True),
        end=border_score(start=False),
    )


def reflected_padding_guard(
    image_rgb: np.ndarray,
    *,
    guard_config: Mapping[str, Any],
) -> ReflectedPaddingResult:
    """Detect near-exact mirrored padding independently at all four borders."""

    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    top_bottom = _reflected_axis_score(grayscale, guard_config=guard_config)
    left_right = _reflected_axis_score(
        grayscale.T,
        guard_config=guard_config,
    )
    detected_axes = tuple(
        axis
        for axis, score in (
            ("top_bottom", top_bottom),
            ("left_right", left_right),
        )
        if score.detected
    )
    return ReflectedPaddingResult(
        detected=bool(detected_axes),
        detected_axes=detected_axes,
        top_bottom=top_bottom,
        left_right=left_right,
    )


def normalize_reflected_padding(
    image_rgb: np.ndarray,
    *,
    annotations: Sequence[Mapping[str, Any]],
    pass1: Mapping[int, Mapping[str, Any]],
    reflection: ReflectedPaddingResult,
    output_shape: tuple[int, int],
    transform_masks: bool = True,
) -> tuple[
    np.ndarray,
    tuple[dict[str, Any], ...],
    dict[int, dict[str, Any]],
    ReflectedPaddingNormalization,
]:
    """Remove detected mirrored borders, then resize-cover and center-crop."""

    image = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    output_height, output_width = output_shape
    top = (
        reflection.top_bottom.start.pad_px
        if reflection.top_bottom.start.detected
        else 0
    )
    bottom = (
        height - reflection.top_bottom.end.pad_px
        if reflection.top_bottom.end.detected
        else height
    )
    left = (
        reflection.left_right.start.pad_px
        if reflection.left_right.start.detected
        else 0
    )
    right = (
        width - reflection.left_right.end.pad_px
        if reflection.left_right.end.detected
        else width
    )
    detected_sides = tuple(
        name
        for name, detected in (
            ("top", reflection.top_bottom.start.detected),
            ("bottom", reflection.top_bottom.end.detected),
            ("left", reflection.left_right.start.detected),
            ("right", reflection.left_right.end.detected),
        )
        if detected
    )
    if not detected_sides and (height, width) == output_shape:
        normalization = ReflectedPaddingNormalization(
            applied=False,
            crop_xyxy=(0, 0, width, height),
            resized_width=width,
            resized_height=height,
            offset_x=0,
            offset_y=0,
            output_width=output_width,
            output_height=output_height,
            detected_sides=(),
        )
        return (
            image.copy(),
            tuple(dict(annotation) for annotation in annotations),
            {
                int(annotation_id): dict(record)
                for annotation_id, record in pass1.items()
            },
            normalization,
        )
    if left >= right or top >= bottom:
        raise ValueError("Reflected-padding crop removed the whole image")

    crop_width = right - left
    crop_height = bottom - top
    scale = max(output_width / crop_width, output_height / crop_height)
    resized_width = max(output_width, round(crop_width * scale))
    resized_height = max(output_height, round(crop_height * scale))
    offset_x = (resized_width - output_width) // 2
    offset_y = (resized_height - output_height) // 2
    cropped = image[top:bottom, left:right]
    resized = cv2.resize(
        cropped,
        (resized_width, resized_height),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized = resized[
        offset_y : offset_y + output_height,
        offset_x : offset_x + output_width,
    ]
    scale_x = resized_width / crop_width
    scale_y = resized_height / crop_height

    def transform_mask(mask: np.ndarray) -> np.ndarray:
        cropped_mask = np.asarray(mask, dtype=np.uint8)[top:bottom, left:right]
        resized_mask = cv2.resize(
            cropped_mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized_mask[
            offset_y : offset_y + output_height,
            offset_x : offset_x + output_width,
        ].astype(bool)

    transformed_annotations: list[dict[str, Any]] = []
    transformed_pass1: dict[int, dict[str, Any]] = {}
    for source in annotations:
        annotation = dict(source)
        annotation_id = int(annotation["id"])
        x, y, box_width, box_height = (
            float(value) for value in annotation["bbox"]
        )
        x1 = (x - left) * scale_x - offset_x
        y1 = (y - top) * scale_y - offset_y
        x2 = (x + box_width - left) * scale_x - offset_x
        y2 = (y + box_height - top) * scale_y - offset_y
        clipped_x1 = min(max(0.0, x1), float(output_width))
        clipped_y1 = min(max(0.0, y1), float(output_height))
        clipped_x2 = min(max(0.0, x2), float(output_width))
        clipped_y2 = min(max(0.0, y2), float(output_height))
        if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
            continue
        transformed_bbox = [
            clipped_x1,
            clipped_y1,
            clipped_x2 - clipped_x1,
            clipped_y2 - clipped_y1,
        ]
        annotation["bbox"] = transformed_bbox
        annotation["area"] = transformed_bbox[2] * transformed_bbox[3]
        transformed_annotations.append(annotation)

        mask_record = dict(pass1[annotation_id])
        if transform_masks and bool(mask_record["qc_pass"]):
            transformed_mask = transform_mask(
                _decode_rle(mask_record["segmentation"])
            )
            if transformed_mask.any():
                mask_record["segmentation"] = _encode_rle(transformed_mask)
            else:
                mask_record["qc_pass"] = False
        transformed_pass1[annotation_id] = mask_record

    normalization = ReflectedPaddingNormalization(
        applied=bool(detected_sides),
        crop_xyxy=(left, top, right, bottom),
        resized_width=resized_width,
        resized_height=resized_height,
        offset_x=offset_x,
        offset_y=offset_y,
        output_width=output_width,
        output_height=output_height,
        detected_sides=detected_sides,
    )
    return (
        normalized,
        tuple(transformed_annotations),
        transformed_pass1,
        normalization,
    )


def context_replacement_input_guard(
    *,
    image_rgb: np.ndarray | None = None,
    image_shape: tuple[int, int],
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    pass1: Mapping[int, Mapping[str, Any]],
    guard_config: Mapping[str, Any],
) -> ContextReplacementGuardResult:
    """Reject reflected-border backgrounds and unsafe replacement anchors."""

    reflection: ReflectedPaddingResult | None = None
    reflection_config = guard_config.get("reflected_padding")
    if reflection_config is not None:
        if image_rgb is None:
            raise ValueError("Pixel reflection guard requires image_rgb")
        reflection = reflected_padding_guard(
            image_rgb,
            guard_config=reflection_config,
        )
        if reflection.detected:
            return ContextReplacementGuardResult(
                accepted=False,
                eligible_annotation_ids=(),
                anchor_margins=(),
                background_min_headlike_edge_margin_px=None,
                reflected_padding=reflection,
                reject_reason="BACKGROUND_REFLECTED_PADDING",
            )

    headlike = [
        annotation
        for annotation in annotations
        if categories[int(annotation["category_id"])] in {"helmet", "head"}
    ]
    if not headlike:
        return ContextReplacementGuardResult(
            accepted=False,
            eligible_annotation_ids=(),
            anchor_margins=(),
            background_min_headlike_edge_margin_px=None,
            reflected_padding=reflection,
            reject_reason="NO_CONTEXT_REPLACEMENT_ANCHOR",
        )

    margins = {
        int(annotation["id"]): _bbox_edge_margin(
            annotation["bbox"],
            image_shape=image_shape,
        )
        for annotation in headlike
    }
    background_min = min(margins.values())
    background_required = float(
        guard_config["background_headlike_min_edge_margin_px"]
    )
    if background_min < background_required:
        return ContextReplacementGuardResult(
            accepted=False,
            eligible_annotation_ids=(),
            anchor_margins=(),
            background_min_headlike_edge_margin_px=float(background_min),
            reflected_padding=reflection,
            reject_reason="BACKGROUND_HEADLIKE_NEAR_FRAME_EDGE",
        )

    fixed_margin = int(guard_config["anchor_min_edge_margin_px"])
    fraction = float(guard_config["anchor_min_edge_margin_long_side_fraction"])
    candidates: list[tuple[int, float, int]] = []
    for annotation in headlike:
        annotation_id = int(annotation["id"])
        _, _, box_width, box_height = (
            float(value) for value in annotation["bbox"]
        )
        required = max(fixed_margin, ceil(fraction * max(box_width, box_height)))
        if (
            bool(pass1[annotation_id]["qc_pass"])
            and margins[annotation_id] >= required
        ):
            candidates.append((annotation_id, float(margins[annotation_id]), required))
    candidates.sort()
    if not candidates:
        return ContextReplacementGuardResult(
            accepted=False,
            eligible_annotation_ids=(),
            anchor_margins=(),
            background_min_headlike_edge_margin_px=float(background_min),
            reflected_padding=reflection,
            reject_reason="NO_SAFE_CONTEXT_REPLACEMENT_ANCHOR",
        )
    return ContextReplacementGuardResult(
        accepted=True,
        eligible_annotation_ids=tuple(item[0] for item in candidates),
        anchor_margins=tuple(candidates),
        background_min_headlike_edge_margin_px=float(background_min),
        reflected_padding=reflection,
        reject_reason=None,
    )


def prepare_context_replacement_background(
    *,
    image_rgb: np.ndarray,
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    pass1: Mapping[int, Mapping[str, Any]],
    guard_config: Mapping[str, Any],
    output_shape: tuple[int, int],
    transform_masks: bool,
) -> PreparedContextBackground:
    """Normalize mirrored borders before applying label and anchor guards."""

    reflection_config = guard_config.get("reflected_padding")
    if reflection_config is None:
        raise ValueError("Context normalization requires reflected-padding config")
    reflection = reflected_padding_guard(
        image_rgb,
        guard_config=reflection_config,
    )
    normalized, transformed_annotations, transformed_pass1, normalization = (
        normalize_reflected_padding(
            image_rgb,
            annotations=annotations,
            pass1=pass1,
            reflection=reflection,
            output_shape=output_shape,
            transform_masks=transform_masks,
        )
    )
    post_normalization_config = {
        key: value
        for key, value in guard_config.items()
        if key != "reflected_padding"
    }
    post_guard = context_replacement_input_guard(
        image_shape=normalized.shape[:2],
        annotations=transformed_annotations,
        categories=categories,
        pass1=transformed_pass1,
        guard_config=post_normalization_config,
    )
    guard = ContextReplacementGuardResult(
        accepted=post_guard.accepted,
        eligible_annotation_ids=post_guard.eligible_annotation_ids,
        anchor_margins=post_guard.anchor_margins,
        background_min_headlike_edge_margin_px=(
            post_guard.background_min_headlike_edge_margin_px
        ),
        reflected_padding=reflection,
        reject_reason=post_guard.reject_reason,
    )
    return PreparedContextBackground(
        image_rgb=normalized,
        annotations=transformed_annotations,
        pass1=transformed_pass1,
        guard=guard,
        normalization=normalization,
    )


def _guarded_context_background_ids(
    *,
    paths: ProjectPaths,
    train_images: Mapping[int, Mapping[str, Any]],
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Mapping[int, str],
    config: Mapping[str, Any],
) -> np.ndarray:
    """Build the safe context-replacement pool before sampling pilot inputs."""

    accepted: list[int] = []
    guard_config = config["compose"]["context_replacement"]["input_guard"]
    for image_id in sorted(train_images):
        image = train_images[image_id]
        image_rgb = np.asarray(
            Image.open(paths.hardhat_raw / str(image["file_name"])).convert("RGB")
        )
        prepared = prepare_context_replacement_background(
            image_rgb=image_rgb,
            annotations=annotations[image_id],
            categories=categories,
            pass1=_load_pass1(paths, image_id),
            guard_config=guard_config,
            output_shape=image_rgb.shape[:2],
            transform_masks=False,
        )
        if prepared.guard.accepted:
            accepted.append(image_id)
    return np.asarray(accepted, dtype=np.int64)


def _existing_layers(
    *,
    image_shape: tuple[int, int],
    annotations: Sequence[Mapping[str, Any]],
    pass1: Mapping[int, Mapping[str, Any]],
    categories: Mapping[int, str],
    intentional_removals: set[int],
) -> list[Layer]:
    layers: list[Layer] = []
    for annotation in annotations:
        annotation_id = int(annotation["id"])
        if annotation_id in intentional_removals:
            continue
        mask_record = pass1[annotation_id]
        if mask_record["qc_pass"]:
            mask = _decode_rle(mask_record["segmentation"])
            mask_source = "sam2_pass1"
        else:
            mask = box_to_mask(image_shape, list(annotation["bbox"]))
            mask_source = "box_fallback"
        bbox = list(annotation["bbox"])
        layers.append(
            Layer(
                instance_id=f"real:{annotation_id}",
                class_name=categories[int(annotation["category_id"])],
                kind="existing",
                mask=mask,
                bbox_xywh_original=bbox,
                y_bottom=float(bbox[1]) + float(bbox[3]),
                existing_mask_source=mask_source,
            )
        )
    return layers


def _choose_bank_item(
    *,
    bank_by_class: Mapping[str, list[dict[str, Any]]],
    class_name: str,
    background_id: int,
    background_group: int,
    used_source_images: Counter[int],
    use_counts: Counter[str],
    config: Mapping[str, Any],
    rng: np.random.Generator,
    target_bbox_xywh: Sequence[float] | None = None,
) -> dict[str, Any] | None:
    compose = config["compose"]
    candidates = [
        item
        for item in bank_by_class.get(class_name, [])
        if (
            (not compose["forbid_same_image"])
            or int(item["src_image_id"]) != background_id
        )
        and (
            (not compose["forbid_same_group"])
            or int(item["src_group_id"]) != background_group
        )
        and used_source_images[int(item["src_image_id"])]
        < int(compose["max_same_source_image_per_composite"])
        and use_counts[str(item["cutout_id"])] < int(compose["max_uses_per_cutout"])
    ]
    if not candidates:
        return None
    if target_bbox_xywh is not None:
        target_descriptor = _log_size_descriptor(target_bbox_xywh)
        candidates.sort(
            key=lambda item: (
                float(
                    np.square(
                        _log_size_descriptor(item["src_bbox_xywh"])
                        - target_descriptor
                    ).sum()
                ),
                str(item["cutout_id"]),
            )
        )
        pool_size = int(compose["context_replacement"]["size_match_pool"])
        population = candidates[:pool_size]
        return population[int(rng.integers(0, len(population)))]
    preferred = [item for item in candidates if item["preferred_tier"]]
    population = preferred if preferred and rng.random() < 0.85 else candidates
    return population[int(rng.integers(0, len(population)))]


def _paste_hard_negatives(
    *,
    rendered: np.ndarray,
    kept_boxes: Sequence[Sequence[float]],
    bank: Sequence[Mapping[str, Any]],
    cutouts_root: Path,
    settings: Mapping[str, Any],
    config: Mapping[str, Any],
    use_counts: Counter[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Composite unannotated distractors after the annotated stack (COMP-20..24).

    Hard negatives carry no annotation by construction: the label space is
    {helmet, head, person} and a yellow dome is none of them, so under COCO
    semantics the absence of a box is the correct and complete labelling
    (ADR-004). FILT-10 additionally forbids them from overlapping any kept
    annotation - occluding a real labelled object with an unlabelled blob would
    corrupt the label instead of sharpening the decision boundary. Because they
    are non-overlapping by rule, they can be composited in their own pass
    without disturbing the annotated instance stack or its z-order.
    """

    if not bank:
        return rendered, []
    rules = config["filtering"]["rules"]["hard_negative_no_overlap"]
    max_iou = float(rules["max_iou_with_annotation"])
    low, high = (int(value) for value in settings["n_pasted"])
    wanted = int(rng.integers(low, high + 1))
    scale_low, scale_high = (float(value) for value in settings["scale_range"])
    # Rigid distractors tolerate more rotation than heads do, so the limit lives in
    # the shared per-class table rather than in the scenario block.
    max_rotation = float(config["compose"]["rotation_deg"]["hard_negative"])
    height, width = rendered.shape[:2]
    output = rendered.copy()
    placed: list[dict[str, Any]] = []

    for index in range(wanted):
        eligible = [
            item
            for item in bank
            if use_counts[str(item["cutout_id"])] < int(item["max_uses"])
        ]
        if not eligible:
            break
        item = eligible[int(rng.integers(0, len(eligible)))]
        rgba = np.asarray(Image.open(cutouts_root / item["file"]).convert("RGBA"))
        scale = float(rng.uniform(scale_low, scale_high))
        angle = float(rng.uniform(-max_rotation, max_rotation))
        hflip = bool(rng.random() < 0.5)
        warped = warp_rgba(rgba, scale=scale, rotation_deg=angle, hflip=hflip)
        patch_h, patch_w = warped.shape[:2]
        if patch_h < 4 or patch_w < 4 or patch_h >= height or patch_w >= width:
            continue
        cy_low, cy_high = (float(value) for value in settings["cy_range"])
        for _ in range(24):  # bounded retries; a rejected placement is not a failure
            left = int(rng.integers(0, width - patch_w))
            # Ground-plane band: a dome floating in the sky is not a hard negative.
            center_y = float(rng.uniform(cy_low, cy_high)) * height
            top = int(np.clip(center_y - patch_h / 2, 0, height - patch_h))
            box = [float(left), float(top), float(patch_w), float(patch_h)]
            if all(_iou_xywh(box, kept) <= max_iou for kept in kept_boxes):
                break
        else:
            continue
        region = output[top : top + patch_h, left : left + patch_w]
        alpha = (warped[..., 3:4].astype(np.float32) / 255.0)
        output[top : top + patch_h, left : left + patch_w] = np.clip(
            region.astype(np.float32) * (1.0 - alpha)
            + warped[..., :3].astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
        use_counts[str(item["cutout_id"])] += 1
        placed.append(
            {
                "annotated": False,
                "bbox_xywh": box,
                "class_name": None,
                "cutout_id": str(item["cutout_id"]),
                "instance_id": f"hardneg:{index}",
                "kind": "hard_negative",
                "max_iou_with_annotation": max(
                    (_iou_xywh(box, kept) for kept in kept_boxes), default=0.0
                ),
                "negative_source": str(item["negative_source"]),
                "src_group_id": int(item["src_group_id"]),
                "src_image_id": int(item["src_image_id"]),
                "transform": {"hflip": hflip, "rotation_deg": angle, "scale": scale},
            }
        )
    return output, placed


def _iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _log_size_descriptor(bbox_xywh: Sequence[float]) -> np.ndarray:
    width = max(float(bbox_xywh[2]), 1)
    height = max(float(bbox_xywh[3]), 1)
    return np.log(np.asarray((width, height), dtype=np.float64))


def _requested_classes(
    scenario: str,
    count: int,
    rng: np.random.Generator,
    *,
    person_crowded_fallback: bool,
) -> list[str]:
    if scenario == "small_distant":
        return [
            str(rng.choice(("helmet", "head"), p=(0.75, 0.25)))
            for _ in range(count)
        ]
    if scenario == "hard_negative":
        # Distractors are composited in their own unannotated pass, so this
        # scenario contributes no annotated pastes of its own. The background's
        # real annotations are still carried through unchanged.
        return []
    if scenario == "head_no_helmet":
        return ["head"] * count
    if scenario == "partial_occlusion":
        targets = [
            str(rng.choice(("helmet", "head"), p=(0.75, 0.25)))
            for _ in range(count)
        ]
        return [*targets, "person"]
    if scenario == "crowded":
        if person_crowded_fallback:
            return [
                str(rng.choice(("helmet", "head"), p=(0.70, 0.30)))
                for _ in range(count)
            ]
        return [
            str(rng.choice(("helmet", "head", "person"), p=(0.60, 0.25, 0.15)))
            for _ in range(count)
        ]
    return [
        str(rng.choice(("helmet", "head", "person"), p=(0.65, 0.25, 0.10)))
        for _ in range(count)
    ]


def _scenario_count(
    scenario: str, settings: Mapping[str, Any], rng: np.random.Generator
) -> int:
    key = "n_targets" if scenario == "partial_occlusion" else "n_pasted"
    low, high = settings[key]
    return int(rng.integers(int(low), int(high) + 1))


def _transform_scale(
    *,
    scenario: str,
    settings: Mapping[str, Any],
    class_name: str,
    rgba: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    target_bbox_xywh: Sequence[float] | None = None,
) -> float:
    if target_bbox_xywh is not None:
        alpha_bbox = tight_bbox(rgba[..., 3] >= 128)
        if alpha_bbox is None:
            return 0
        width_scale = float(target_bbox_xywh[2]) / max(float(alpha_bbox[2]), 1)
        height_scale = float(target_bbox_xywh[3]) / max(float(alpha_bbox[3]), 1)
        scale = float(np.sqrt(width_scale * height_scale))
    elif scenario == "small_distant":
        target_min_side = float(rng.uniform(*settings["target_min_side_px"]))
        alpha_bbox = tight_bbox(rgba[..., 3] >= 128)
        source_min_side = (
            min(float(alpha_bbox[2]), float(alpha_bbox[3]))
            if alpha_bbox is not None
            else min(rgba.shape[:2])
        )
        scale = target_min_side / max(source_min_side, 1)
    else:
        scale_range = settings.get("scale_range", (0.60, 1.00))
        scale = float(rng.uniform(*scale_range))
    return min(scale, float(config["compose"]["max_paste_scale"][class_name]))


def _placement_center(
    *,
    scenario: str,
    settings: Mapping[str, Any],
    image_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    prior_centers: Sequence[tuple[float, float]],
    prior_boxes: Sequence[Sequence[float]],
    person_boxes: Sequence[Sequence[float]],
    headlike_boxes: Sequence[Sequence[float]],
    empirical_prior: Sequence[tuple[float, float]],
    class_name: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    height, width = image_shape
    patch_height, patch_width = patch_shape
    margin_x = min(patch_width / 2, width / 2)
    margin_y = min(patch_height / 2, height / 2)
    if scenario == "small_distant":
        cy_low, cy_high = settings["cy_range"]
        if empirical_prior:
            normalized_x, normalized_y = empirical_prior[
                int(rng.integers(0, len(empirical_prior)))
            ]
            normalized_y = float(np.clip(normalized_y, cy_low, cy_high))
            return (
                float(
                    np.clip(
                        (normalized_x + rng.normal(0, 0.01)) * width,
                        margin_x,
                        max(width - margin_x, margin_x),
                    )
                ),
                float(
                    np.clip(
                        (normalized_y + rng.normal(0, 0.01)) * height,
                        cy_low * height,
                        cy_high * height,
                    )
                ),
            )
        return (
            float(rng.uniform(margin_x, max(width - margin_x, margin_x + 1))),
            float(rng.uniform(cy_low * height, cy_high * height)),
        )
    if scenario == "head_no_helmet" and person_boxes:
        person = person_boxes[int(rng.integers(0, len(person_boxes)))]
        return (
            float(person[0] + rng.uniform(0.35, 0.65) * person[2]),
            float(person[1] + rng.uniform(0.08, 0.25) * person[3]),
        )
    if scenario == "crowded" and headlike_boxes:
        anchor = headlike_boxes[int(rng.integers(0, len(headlike_boxes)))]
        anchor_x = float(anchor[0]) + float(anchor[2]) / 2
        anchor_y = float(anchor[1]) + float(anchor[3]) / 2
        return (
            float(
                np.clip(
                    anchor_x + rng.normal(0, max(patch_width, float(anchor[2])) * 1.2),
                    margin_x,
                    max(width - margin_x, margin_x),
                )
            ),
            float(
                np.clip(
                    anchor_y + rng.normal(0, max(patch_height, float(anchor[3])) * 0.8),
                    margin_y,
                    max(height - margin_y, margin_y),
                )
            ),
        )
    if scenario == "partial_occlusion" and class_name == "person" and prior_boxes:
        target_x, target_y, target_width, target_height = (
            float(value) for value in prior_boxes[0]
        )
        occlusion = float(rng.uniform(*settings["target_occlusion_ratio"]))
        occluder_top = target_y + (1 - occlusion) * target_height
        return (
            float(
                np.clip(
                    target_x + target_width / 2,
                    margin_x,
                    max(width - margin_x, margin_x),
                )
            ),
            float(
                np.clip(
                    occluder_top + patch_height / 2,
                    margin_y,
                    max(height - margin_y, margin_y),
                )
            ),
        )
    if empirical_prior:
        normalized_x, normalized_y = empirical_prior[
            int(rng.integers(0, len(empirical_prior)))
        ]
        return (
            float(
                np.clip(
                    (normalized_x + rng.normal(0, 0.015)) * width,
                    margin_x,
                    max(width - margin_x, margin_x),
                )
            ),
            float(
                np.clip(
                    (normalized_y + rng.normal(0, 0.015)) * height,
                    margin_y,
                    max(height - margin_y, margin_y),
                )
            ),
        )
    return (
        float(rng.uniform(margin_x, max(width - margin_x, margin_x + 1))),
        float(rng.uniform(margin_y, max(height - margin_y, margin_y + 1))),
    )


def _make_paste(
    *,
    item: dict[str, Any],
    class_name: str,
    scenario: str,
    settings: Mapping[str, Any],
    image_shape: tuple[int, int],
    center_override: tuple[float, float] | None,
    prior_centers: Sequence[tuple[float, float]],
    prior_boxes: Sequence[Sequence[float]],
    person_boxes: Sequence[Sequence[float]],
    headlike_boxes: Sequence[Sequence[float]],
    empirical_prior: Sequence[tuple[float, float]],
    paste_index: int,
    paths: ProjectPaths,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    target_bbox_xywh: Sequence[float] | None = None,
) -> Paste | None:
    rgba = np.asarray(Image.open(paths.cutouts / item["file"]).convert("RGBA"))
    scale = _transform_scale(
        scenario=scenario,
        settings=settings,
        class_name=class_name,
        rgba=rgba,
        config=config,
        rng=rng,
        target_bbox_xywh=target_bbox_xywh,
    )
    rotation_limit = float(
        settings.get("rotation_deg", config["compose"]["rotation_deg"][class_name])
    )
    transformed = warp_rgba(
        rgba,
        scale=scale,
        rotation_deg=float(rng.uniform(-rotation_limit, rotation_limit)),
        hflip=bool(rng.random() < float(config["compose"]["hflip_prob"])),
    )
    if scenario == "small_distant":
        alpha_bbox = tight_bbox(transformed[..., 3] >= 128)
        if alpha_bbox is not None:
            actual_min_side = min(float(alpha_bbox[2]), float(alpha_bbox[3]))
            low, high = (float(value) for value in settings["target_min_side_px"])
            desired_min_side = float(np.clip(actual_min_side, low, high))
            correction = desired_min_side / max(actual_min_side, 1)
            if not np.isclose(correction, 1):
                corrected_width = max(1, round(transformed.shape[1] * correction))
                corrected_height = max(1, round(transformed.shape[0] * correction))
                transformed = cv2.resize(
                    transformed,
                    (corrected_width, corrected_height),
                    interpolation=(
                        cv2.INTER_AREA if correction < 1 else cv2.INTER_LINEAR
                    ),
                )
    center = center_override or _placement_center(
        scenario=scenario,
        settings=settings,
        image_shape=image_shape,
        patch_shape=transformed.shape[:2],
        prior_centers=prior_centers,
        prior_boxes=prior_boxes,
        person_boxes=person_boxes,
        headlike_boxes=headlike_boxes,
        empirical_prior=empirical_prior,
        class_name=class_name,
        rng=rng,
    )
    frame_slice, patch_slice, inside_ratio = placement_slices(
        frame_shape=image_shape,
        patch_shape=transformed.shape[:2],
        center_xy=center,
    )
    if inside_ratio < float(config["compose"]["bbox"]["min_inside_ratio"]):
        return None
    full_alpha = np.zeros(image_shape, dtype=np.uint8)
    full_alpha[frame_slice] = transformed[..., 3][patch_slice]
    full_mask = full_alpha >= 128
    bbox = tight_bbox(full_mask)
    if bbox is None:
        return None
    bbox_preclip = [
        float(center[0] - transformed.shape[1] / 2),
        float(center[1] - transformed.shape[0] / 2),
        float(transformed.shape[1]),
        float(transformed.shape[0]),
    ]
    layer = Layer(
        instance_id=f"paste:{paste_index}",
        class_name=class_name,
        kind="pasted",
        mask=full_mask,
        bbox_xywh_original=bbox,
        y_bottom=float(bbox[1]) + float(bbox[3]),
    )
    return Paste(
        layer=layer,
        rgba=transformed,
        frame_slice=frame_slice,
        patch_slice=patch_slice,
        bank=item,
        bbox_preclip=bbox_preclip,
    )


def _render_pastes(
    *,
    base_rgb: np.ndarray,
    pastes: Sequence[Paste],
    existing_layers: Sequence[Layer],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    output = base_rgb.copy()
    paste_by_id = {paste.layer.instance_id: paste for paste in pastes}
    all_layers = sorted(
        [*existing_layers, *(paste.layer for paste in pastes)],
        key=lambda layer: (layer.y_bottom, layer.instance_id),
    )
    for layer in all_layers:
        if layer.kind != "pasted":
            continue
        paste = paste_by_id[layer.instance_id]
        rgb = paste.rgba[..., :3].copy()
        if config["compose"]["blending"]["edge_decontamination"]:
            rgb = decontaminate_soft_edge(
                rgb,
                paste.rgba[..., 3],
                core_alpha_min=int(
                    config["compose"]["blending"]["edge_core_alpha_min"]
                ),
            )
        alpha = feather_alpha(
            paste.rgba[..., 3], config=config["compose"]["blending"]
        )
        patch_alpha = alpha[paste.patch_slice]
        target = output[paste.frame_slice]
        target_ring = annulus_mask(
            patch_alpha.shape,
            patch_alpha >= 128,
            outer_scale=float(
                config["compose"]["harmonization"]["annulus_outer_scale"]
            ),
        )
        if target_ring.sum() < int(
            config["compose"]["harmonization"]["annulus_min_valid_px"]
        ):
            target_ring = patch_alpha < 128
        patch_rgb = rgb[paste.patch_slice]
        if config["compose"]["harmonization"]["enabled"]:
            patch_rgb = harmonize_lab(
                patch_rgb,
                patch_alpha,
                target,
                target_ring,
                config=config["compose"]["harmonization"],
            )
        if config["compose"]["harmonization"]["noise_match"]:
            patch_rgb = match_high_frequency_noise(
                patch_rgb,
                patch_alpha,
                target,
                target_ring,
                sigma_cap=float(
                    config["compose"]["harmonization"]["noise_match_sigma_cap"]
                ),
                rng=rng,
            )
        rgb[paste.patch_slice] = patch_rgb
        blending_method = str(config["compose"]["blending"]["method"])
        if blending_method == "feathered_alpha":
            output = alpha_composite(
                output,
                rgb,
                alpha,
                frame_slice=paste.frame_slice,
                patch_slice=paste.patch_slice,
            )
        elif blending_method == "poisson":
            output = poisson_composite(
                output,
                rgb,
                paste.rgba[..., 3],
                frame_slice=paste.frame_slice,
                patch_slice=paste.patch_slice,
            )
        else:
            raise ValueError(f"Unknown blending method: {blending_method}")
        # The real object is already in the background. Restore it whenever its
        # geometric layer is in front of this paste.
        for existing in existing_layers:
            if existing.y_bottom > layer.y_bottom:
                output[existing.mask] = base_rgb[existing.mask]
    seams = {
        paste.layer.instance_id: seam_energy_ratio(
            output,
            paste.layer.mask,
            band_px=int(config["filtering"]["rules"]["clipping_artifact"]["seam_band_px"]),
        )
        for paste in pastes
    }
    return output, seams


def _instance_records(
    *,
    visibility: Sequence[Mapping[str, Any]],
    pastes: Sequence[Paste],
    annotations_by_id: Mapping[int, Mapping[str, Any]],
    seams: Mapping[str, float],
    generative: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    paste_by_id = {paste.layer.instance_id: paste for paste in pastes}
    records: list[dict[str, Any]] = []
    for visible in visibility:
        instance_id = str(visible["instance_id"])
        record = dict(visible)
        record["bbox_xywh_preclip"] = list(record["bbox_xywh"])
        if instance_id.startswith("real:"):
            annotation_id = int(instance_id.split(":", 1)[1])
            record["source_annotation_id"] = annotation_id
            record["source_image_id"] = int(annotations_by_id[annotation_id]["image_id"])
            record["sam2_qc_pass"] = True
        else:
            paste = paste_by_id[instance_id]
            item = paste.bank
            record.update(
                {
                    "bbox_xywh_preclip": paste.bbox_preclip,
                    "cutout_id": item["cutout_id"],
                    "source_annotation_id": int(
                        item["cutout_id"].rsplit("ann", 1)[1]
                    ),
                    "source_image_id": int(item["src_image_id"]),
                    "source_group_id": int(item["src_group_id"]),
                    "sam2_qc_pass": True,
                    "mask_to_box_coverage": float(
                        item["sam2"]["mask_to_box_coverage"]
                    ),
                    "sam2_iou_score": float(item["sam2"]["iou_score"]),
                    "seam_energy_ratio": float(seams.get(instance_id, 0)),
                }
            )
            if generative and instance_id in generative:
                record["generative_inpaint"] = dict(generative[instance_id])
        records.append(record)
    records.sort(
        key=lambda item: (
            float(item["bbox_xywh"][1]) + float(item["bbox_xywh"][3]),
            str(item["instance_id"]),
        )
    )
    for z_index, record in enumerate(records):
        record["z_index"] = z_index
        record["y_bottom"] = float(record["bbox_xywh"][1]) + float(
            record["bbox_xywh"][3]
        )
    return records


def _visible_paste_masks(
    *,
    existing_layers: Sequence[Layer],
    pastes: Sequence[Paste],
) -> dict[str, np.ndarray]:
    """Compute final visible support for each paste without editing occluders."""

    layers = sorted(
        [*existing_layers, *(paste.layer for paste in pastes)],
        key=lambda layer: (layer.z_index, layer.instance_id),
    )
    if not layers:
        return {}
    union_above = np.zeros_like(layers[0].mask, dtype=bool)
    visible: dict[str, np.ndarray] = {}
    for layer in reversed(layers):
        support = np.asarray(layer.mask, dtype=bool)
        if layer.kind == "pasted":
            visible[layer.instance_id] = support & ~union_above
        union_above |= support
    return visible


def _build_sample(
    *,
    sample_index: int,
    scenario: str,
    background: Mapping[str, Any],
    background_group: int,
    background_annotations: Sequence[dict[str, Any]],
    categories: Mapping[int, str],
    bank_by_class: Mapping[str, list[dict[str, Any]]],
    placement_priors: Mapping[str, Sequence[tuple[float, float]]],
    person_crowded_fallback: bool,
    paths: ProjectPaths,
    config: Mapping[str, Any],
    filter_config: Mapping[str, Any],
    use_counts: Counter[str],
    real_phashes: Mapping[int, str],
    accepted_phashes: Sequence[str],
    rng: np.random.Generator,
    generative_inpainter: GenerativeBoundaryInpainter | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    image_path = paths.hardhat_raw / str(background["file_name"])
    original = np.asarray(Image.open(image_path).convert("RGB"))
    image_shape = original.shape[:2]
    pass1 = _load_pass1(paths, int(background["id"]))
    settings = config["scenarios"][scenario]
    context_guard: ContextReplacementGuardResult | None = None
    context_normalization: ReflectedPaddingNormalization | None = None
    if scenario == "context_replacement":
        prepared = prepare_context_replacement_background(
            image_rgb=original,
            annotations=background_annotations,
            categories=categories,
            pass1=pass1,
            guard_config=config["compose"]["context_replacement"]["input_guard"],
            output_shape=image_shape,
            transform_masks=True,
        )
        original = prepared.image_rgb
        image_shape = original.shape[:2]
        background_annotations = prepared.annotations
        pass1 = prepared.pass1
        context_guard = prepared.guard
        context_normalization = prepared.normalization
        if not context_guard.accepted:
            raise RuntimeError(
                "Could not place scenario=context_replacement "
                f"on image={background['id']}: {context_guard.reject_reason}"
            )
    annotations_by_id = {
        int(annotation["id"]): annotation for annotation in background_annotations
    }
    max_attempts = int(config["compose"]["bbox"]["max_placement_retries"])
    last_reason = "PLACEMENT_RETRIES_EXHAUSTED"
    for _ in range(max_attempts):
        attempt_state = rng.bit_generator.state
        base = original.copy()
        intentional_removals: set[int] = set()
        center_override: tuple[float, float] | None = None
        replacement_anchor: dict[str, Any] | None = None
        target_bbox_xywh: Sequence[float] | None = None
        person_boxes = [
            annotation["bbox"]
            for annotation in background_annotations
            if categories[int(annotation["category_id"])] == "person"
        ]
        headlike_boxes = [
            annotation["bbox"]
            for annotation in background_annotations
            if categories[int(annotation["category_id"])] in {"helmet", "head"}
        ]
        eligible_swap_helmets = [
            annotation
            for annotation in background_annotations
            if categories[int(annotation["category_id"])] == "helmet"
            and pass1[int(annotation["id"])]["qc_pass"]
        ]
        if scenario == "context_replacement":
            if context_guard is None:
                raise AssertionError("Context-replacement input guard was not evaluated")
            eligible_ids = set(context_guard.eligible_annotation_ids)
            eligible_replacements = [
                annotation
                for annotation in background_annotations
                if int(annotation["id"]) in eligible_ids
            ]
            if not eligible_replacements:
                raise AssertionError(
                    "Accepted context-replacement guard has no eligible anchor"
                )
            replacement_anchor = eligible_replacements[
                int(rng.integers(0, len(eligible_replacements)))
            ]
            removed_mask = _decode_rle(
                pass1[int(replacement_anchor["id"])]["segmentation"]
            )
            replacement_config = config["compose"]["context_replacement"]
            base, _ = inpaint_masked_object(
                base,
                removed_mask,
                dilate_px=int(replacement_config["inpaint_dilate_px"]),
                radius=int(replacement_config["inpaint_radius"]),
            )
            intentional_removals.add(int(replacement_anchor["id"]))
            target_bbox_xywh = replacement_anchor["bbox"]
            x, y, width, height = (
                float(value) for value in target_bbox_xywh
            )
            center_override = (x + width / 2, y + height / 2)
        do_swap = (
            scenario == "head_no_helmet"
            and bool(eligible_swap_helmets)
            and (
                not person_boxes
                or rng.random()
                < float(settings["submode_helmet_to_head_swap_prob"])
            )
        )
        if scenario == "context_replacement":
            pass
        elif do_swap:
            removed = eligible_swap_helmets[
                int(rng.integers(0, len(eligible_swap_helmets)))
            ]
            removed_mask = _decode_rle(
                pass1[int(removed["id"])]["segmentation"]
            )
            base, _ = inpaint_masked_object(
                base,
                removed_mask,
                dilate_px=int(settings["swap_inpaint_dilate_px"]),
                radius=int(settings["swap_inpaint_radius"]),
            )
            intentional_removals.add(int(removed["id"]))
            x, y, width, height = (float(value) for value in removed["bbox"])
            center_override = (x + width / 2, y + height / 2)
        elif scenario == "head_no_helmet" and not person_boxes:
            last_reason = "NO_VALID_HEAD_ANCHOR"
            continue
        existing_layers = _existing_layers(
            image_shape=image_shape,
            annotations=background_annotations,
            pass1=pass1,
            categories=categories,
            intentional_removals=intentional_removals,
        )
        if replacement_anchor is not None:
            count = 1
            classes = [
                categories[int(replacement_anchor["category_id"])]
            ]
        else:
            count = 1 if do_swap and center_override else _scenario_count(
                scenario, settings, rng
            )
            classes = _requested_classes(
                scenario,
                count,
                rng,
                person_crowded_fallback=person_crowded_fallback,
            )
        pastes: list[Paste] = []
        prior_centers: list[tuple[float, float]] = []
        prior_boxes: list[list[float]] = []
        used_source_images: Counter[int] = Counter()
        for paste_index, class_name in enumerate(classes):
            item = _choose_bank_item(
                bank_by_class=bank_by_class,
                class_name=class_name,
                background_id=int(background["id"]),
                background_group=background_group,
                used_source_images=used_source_images,
                use_counts=use_counts,
                config=config,
                rng=rng,
                target_bbox_xywh=(
                    target_bbox_xywh if paste_index == 0 else None
                ),
            )
            if item is None and class_name == "person":
                class_name = "helmet"
                item = _choose_bank_item(
                    bank_by_class=bank_by_class,
                    class_name=class_name,
                    background_id=int(background["id"]),
                    background_group=background_group,
                    used_source_images=used_source_images,
                    use_counts=use_counts,
                    config=config,
                    rng=rng,
                )
            if item is None:
                continue
            paste = _make_paste(
                item=item,
                class_name=class_name,
                scenario=scenario,
                settings=settings,
                image_shape=image_shape,
                center_override=(center_override if paste_index == 0 else None),
                prior_centers=prior_centers,
                prior_boxes=prior_boxes,
                person_boxes=person_boxes,
                headlike_boxes=headlike_boxes,
                empirical_prior=placement_priors[class_name],
                paste_index=paste_index,
                paths=paths,
                config=config,
                rng=rng,
                target_bbox_xywh=(
                    target_bbox_xywh if paste_index == 0 else None
                ),
            )
            if paste is None:
                continue
            pastes.append(paste)
            used_source_images[int(item["src_image_id"])] += 1
            bbox = paste.layer.bbox_xywh_original
            prior_centers.append(
                (float(bbox[0]) + float(bbox[2]) / 2, float(bbox[1]) + float(bbox[3]) / 2)
            )
            prior_boxes.append(list(bbox))
        # low_light_blur may legitimately paste nothing (it is a whole-image effect),
        # and hard_negative never produces an ANNOTATED paste at all - its distractors
        # are composited in a separate unannotated pass further down.
        if not pastes and scenario not in {"low_light_blur", "hard_negative"}:
            rng.bit_generator.state = attempt_state
            rng.random()
            continue
        visibility = recompute_visible_annotations(
            [*existing_layers, *(paste.layer for paste in pastes)],
            min_visible_fraction_pasted=float(
                config["compose"]["bbox"]["min_visible_fraction_pasted"]
            ),
            existing_keep_original_above=float(
                config["compose"]["bbox"]["existing_keep_original_above"]
            ),
            existing_recompute_above=float(
                config["compose"]["bbox"]["existing_recompute_above"]
            ),
        )
        if not visibility.accepted or visibility.rejected_instance_ids:
            last_reason = visibility.reason or "PASTED_OBJECT_TOO_OCCLUDED"
            continue
        if scenario == "small_distant" and any(
            float(instance["visible_fraction"]) < 0.80
            for instance in visibility.annotations
            if instance["kind"] == "pasted"
        ):
            last_reason = "SMALL_DISTANT_OCCLUDED"
            continue
        render_config = dict(config)
        render_config["filtering"] = filter_config
        rendered, seams = _render_pastes(
            base_rgb=base,
            pastes=pastes,
            existing_layers=existing_layers,
            config=render_config,
            rng=rng,
        )
        hard_negatives: list[dict[str, Any]] = []
        if scenario == "hard_negative":
            rendered, hard_negatives = _paste_hard_negatives(
                rendered=rendered,
                kept_boxes=[
                    [float(value) for value in instance["bbox_xywh"]]
                    for instance in visibility.annotations
                ],
                bank=bank_by_class.get("hard_negative", []),
                cutouts_root=paths.cutouts,
                settings=settings,
                config=render_config,
                use_counts=use_counts,
                rng=rng,
            )
            if not hard_negatives:
                last_reason = "NO_HARD_NEGATIVE_PLACEMENT"
                continue
        generative_records: dict[str, dict[str, Any]] = {}
        if generative_inpainter is not None:
            visible_masks = _visible_paste_masks(
                existing_layers=existing_layers,
                pastes=pastes,
            )
            for paste in sorted(
                pastes,
                key=lambda item: (item.layer.z_index, item.layer.instance_id),
            ):
                visible_mask = visible_masks[paste.layer.instance_id]
                if not visible_mask.any():
                    raise AssertionError("Accepted pasted layer has no visible mask")
                edit_seed = _generative_seed(
                    int(config["seed"]),
                    sample_index,
                    paste.layer.instance_id,
                )
                inpainted = generative_inpainter.generate(
                    draft_rgb=rendered,
                    object_mask=visible_mask,
                    reference_rgba=paste.rgba,
                    class_name=paste.layer.class_name,
                    seed=edit_seed,
                )
                rendered = inpainted.image_rgb
                generative_records[paste.layer.instance_id] = inpainted.provenance
            seams = {
                paste.layer.instance_id: seam_energy_ratio(
                    rendered,
                    visible_masks[paste.layer.instance_id],
                    band_px=int(
                        filter_config["rules"]["clipping_artifact"]["seam_band_px"]
                    ),
                )
                for paste in pastes
            }
        postfx_applied: dict[str, Any] = {}
        apply_fx = scenario == "low_light_blur" or rng.random() < float(
            settings["postfx_prob"]
        )
        if apply_fx:
            rendered, postfx_applied = apply_postfx(
                rendered, config=config["postfx"], rng=rng
            )
        instances = _instance_records(
            visibility=visibility.annotations,
            pastes=pastes,
            annotations_by_id=annotations_by_id,
            seams=seams,
            generative=generative_records,
        )
        output_phash = _phash_hex(rendered)
        changed = np.any(
            np.abs(rendered.astype(np.int16) - original.astype(np.int16)) > 2,
            axis=2,
        )
        record: dict[str, Any] = {
            "schema_version": 1,
            "sample_id": f"s{int(config['seed'])}_{sample_index:06d}",
            "scenario": scenario,
            "width": int(image_shape[1]),
            "height": int(image_shape[0]),
            "background": {
                "image_id": int(background["id"]),
                "group_id": background_group,
                "file_name": str(background["file_name"]),
            },
            "instances": instances,
            "pairs": [],
            # Kept OUT of "instances" on purpose: everything in that list becomes a
            # COCO annotation, and hard negatives must never produce one (ADR-004).
            "hard_negatives": hard_negatives,
            "intentional_removals": sorted(intentional_removals),
            "replacement_anchor_annotation_id": (
                int(replacement_anchor["id"])
                if replacement_anchor is not None
                else None
            ),
            "postfx": postfx_applied,
            "dedup": {
                "phash": output_phash,
                "changed_pixel_ratio": float(changed.mean()),
                "min_hamming_to_accepted_synthetic": min(
                    (_hamming(output_phash, value) for value in accepted_phashes),
                    default=10**9,
                ),
                "min_hamming_to_other_real_image": min(
                    (
                        _hamming(output_phash, value)
                        for image_id, value in real_phashes.items()
                        if image_id != int(background["id"])
                    ),
                    default=10**9,
                ),
                "excluded_background_image_id": int(background["id"]),
            },
            "invariants": {
                "n_real_ann_in": len(background_annotations),
                "n_real_ann_out": sum(
                    instance["kind"] == "existing" for instance in instances
                ),
                "intentional_removals": sorted(intentional_removals),
                "test_blocklist_untouched": True,
            },
            "generation": {
                "seed": int(config["seed"]),
                "attempts_limit": max_attempts,
                "hard_negative_signoff_required": True,
                "hard_negative_used": bool(hard_negatives),
                "generative_inpaint_used": generative_inpainter is not None,
            },
        }
        if context_guard is not None and replacement_anchor is not None:
            if context_normalization is None:
                raise AssertionError("Context normalization provenance is missing")
            anchor_id = int(replacement_anchor["id"])
            _, anchor_margin, required_margin = next(
                item for item in context_guard.anchor_margins if item[0] == anchor_id
            )
            record["context_replacement_input_guard"] = {
                "version": "v5",
                "background_min_headlike_edge_margin_px": (
                    context_guard.background_min_headlike_edge_margin_px
                ),
                "background_reflected_padding_detected": bool(
                    context_guard.reflected_padding
                    and context_guard.reflected_padding.detected
                ),
                "background_normalization": {
                    "applied": context_normalization.applied,
                    "crop_xyxy": list(context_normalization.crop_xyxy),
                    "resized_width": context_normalization.resized_width,
                    "resized_height": context_normalization.resized_height,
                    "offset_x": context_normalization.offset_x,
                    "offset_y": context_normalization.offset_y,
                    "output_width": context_normalization.output_width,
                    "output_height": context_normalization.output_height,
                    "detected_sides": list(context_normalization.detected_sides),
                },
                "selected_anchor_edge_margin_px": anchor_margin,
                "selected_anchor_required_edge_margin_px": required_margin,
                "selected_anchor_sam2_qc_pass": True,
            }
        result = filter_sample(record, filter_config)
        record["passed"] = result.passed
        record["reject_reasons"] = list(result.reject_reasons)
        record["first_reject_reason"] = result.first_reason
        for paste in pastes:
            use_counts[str(paste.bank["cutout_id"])] += 1
        return rendered, record
    raise RuntimeError(
        f"Could not place scenario={scenario} on image={background['id']}: {last_reason}"
    )


def _scenario_sequence(
    n: int,
    config: Mapping[str, Any],
    selected: Sequence[str] | None,
    rng: np.random.Generator,
) -> list[str]:
    if selected:
        unknown = set(selected) - set(SCENARIO_ORDER + EXPERIMENTAL_SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
        names = list(selected)
    else:
        # M9 closed: kuotunyu signed off the H6 sheet at 0/64 real helmets and the
        # bank is materialised, so hard_negative now participates like any other
        # scenario. Its distractors are unannotated by construction (ADR-004).
        names = list(SCENARIO_ORDER)
    weights = np.asarray([float(config["scenarios"][name]["weight"]) for name in names])
    weights /= weights.sum()
    if n >= len(names):
        sampled = [*names]
        sampled.extend(
            str(name) for name in rng.choice(names, size=n - len(names), p=weights)
        )
        rng.shuffle(sampled)
        return sampled
    sampled = rng.choice(names, size=n, replace=False, p=weights)
    return [str(name) for name in sampled]


def _write_image(path: Path, image_rgb: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower()
    encode_options = (
        [cv2.IMWRITE_JPEG_QUALITY, quality] if extension in {".jpg", ".jpeg"} else []
    )
    success, encoded = cv2.imencode(
        extension,
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        encode_options,
    )
    if not success:
        raise RuntimeError(f"Could not encode {path}")
    path.write_bytes(encoded.tobytes())


def _write_preview_grid(
    records: Sequence[Mapping[str, Any]],
    images: Sequence[np.ndarray],
    output_path: Path,
) -> None:
    if not images:
        return
    cell_width, cell_height = 416, 456
    columns = 4
    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    colors = {"helmet": "yellow", "head": "cyan", "person": "lime"}
    for index, (record, image_rgb) in enumerate(zip(records, images, strict=True)):
        panel = Image.fromarray(image_rgb).convert("RGB")
        draw = ImageDraw.Draw(panel)
        for instance in record["instances"]:
            if not instance.get("kept", True):
                continue
            x, y, width, height = instance["bbox_xywh"]
            draw.rectangle(
                (x, y, x + width, y + height),
                outline=colors[str(instance["class_name"])],
                width=2,
            )
        x0 = (index % columns) * cell_width
        y0 = (index // columns) * cell_height
        canvas.paste(panel, (x0, y0))
        caption = (
            f"{record['sample_id']} | {record['scenario']} | "
            f"{'PASS' if record['passed'] else record['first_reject_reason']}"
        )
        ImageDraw.Draw(canvas).text((x0 + 4, y0 + 420), caption, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _self_eval(coco_path: Path) -> float:
    coco = COCO(str(coco_path))
    detections = [
        {
            "image_id": int(annotation["image_id"]),
            "category_id": int(annotation["category_id"]),
            "bbox": list(annotation["bbox"]),
            "score": 1.0,
        }
        for annotation in coco.dataset["annotations"]
    ]
    result = coco.loadRes(detections)
    evaluation = COCOeval(coco, result, "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    value = float(evaluation.stats[0])
    if not np.isclose(value, 1.0, atol=1e-9):
        raise AssertionError(f"COCO self-evaluation mAP must be 1.0, got {value}")
    return value


def generate(
    *,
    paths: ProjectPaths,
    n: int,
    seed: int,
    output_tag: str,
    selected_scenarios: Sequence[str] | None,
    draw_boxes: bool,
    blending_method: str | None = None,
    generative_inpainter: GenerativeBoundaryInpainter | None = None,
) -> dict[str, Any]:
    config, filter_config = _load_configs()
    config["seed"] = seed
    if blending_method is not None:
        config["compose"]["blending"]["method"] = blending_method
    rng = np.random.default_rng(seed)
    coco, bank, hardneg_bank, train_images, annotations, frozen, _ = _load_context(paths)
    categories = {
        int(record["id"]): str(record["name"]) for record in coco["categories"]
    }
    bank_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in bank:
        bank_by_class[str(item["class_name"])].append(item)
    for items in bank_by_class.values():
        items.sort(key=lambda item: str(item["cutout_id"]))
    missing = {"helmet", "head"} - set(bank_by_class)
    if missing:
        raise RuntimeError(f"Cutout bank lacks required classes: {sorted(missing)}")
    # Distractors live under their own key so they can never be selected as an
    # annotated paste: _requested_classes only ever yields helmet/head/person.
    bank_by_class["hard_negative"] = sorted(
        hardneg_bank, key=lambda item: str(item["cutout_id"])
    )
    person_groups = {
        int(item["src_group_id"]) for item in bank_by_class.get("person", [])
    }
    person_crowded_fallback = len(person_groups) < int(
        config["cutout_bank"]["min_distinct_person_groups"]
    )
    placement_priors: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for image_id, image_annotations in annotations.items():
        image = train_images[image_id]
        for annotation in image_annotations:
            x, y, width, height = (float(value) for value in annotation["bbox"])
            class_name = categories[int(annotation["category_id"])]
            placement_priors[class_name].append(
                (
                    (x + width / 2) / float(image["width"]),
                    (y + height / 2) / float(image["height"]),
                )
            )
    train_ids = np.asarray(sorted(train_images))
    if (
        selected_scenarios is not None
        and tuple(selected_scenarios) == ("context_replacement",)
    ):
        train_ids = _guarded_context_background_ids(
            paths=paths,
            train_images=train_images,
            annotations=annotations,
            categories=categories,
            config=config,
        )
        if len(train_ids) < n:
            raise RuntimeError(
                "Guarded context-replacement pool is smaller than the "
                f"registered run: {len(train_ids)} < {n}"
            )
    background_ids = rng.choice(train_ids, size=n, replace=n > len(train_ids))
    scenarios = _scenario_sequence(n, config, selected_scenarios, rng)
    real_phashes = {
        int(image_id): str(record["phash"]) for image_id, record in frozen.items()
    }
    output_dir = paths.synthetic / output_tag
    _archive_existing(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    use_counts: Counter[str] = Counter()
    accepted_phashes: list[str] = []
    records: list[dict[str, Any]] = []
    preview_images: list[np.ndarray] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    annotation_id = 1
    category_ids = {name: category_id for category_id, name in categories.items()}
    jpeg_config = config["postfx"]["jpeg"]
    for sample_index, (background_id, scenario) in enumerate(
        zip(background_ids.tolist(), scenarios, strict=True), start=1
    ):
        sample_seed = _sample_seed(seed, sample_index)
        sample_rng = np.random.default_rng(sample_seed)
        background_retries = 0
        while True:
            background = train_images[int(background_id)]
            try:
                image, record = _build_sample(
                    sample_index=sample_index,
                    scenario=scenario,
                    background=background,
                    background_group=int(frozen[int(background_id)]["group_id"]),
                    background_annotations=annotations[int(background_id)],
                    categories=categories,
                    bank_by_class=bank_by_class,
                    placement_priors=placement_priors,
                    person_crowded_fallback=person_crowded_fallback,
                    paths=paths,
                    config=config,
                    filter_config=filter_config,
                    use_counts=use_counts,
                    real_phashes=real_phashes,
                    accepted_phashes=accepted_phashes,
                    rng=sample_rng,
                    generative_inpainter=generative_inpainter,
                )
                break
            except RuntimeError as error:
                if not str(error).startswith("Could not place scenario="):
                    raise
                background_retries += 1
                if background_retries >= 20:
                    raise RuntimeError(
                        f"Could not generate sample {sample_index} after "
                        f"{background_retries} Train backgrounds"
                    ) from error
                background_id = int(sample_rng.choice(train_ids))
        record["generation"]["background_retries"] = background_retries
        record["generation"]["root_seed"] = seed
        record["generation"]["sample_seed"] = sample_seed
        quality = (
            int(
                sample_rng.integers(
                    int(jpeg_config["quality"][0]),
                    int(jpeg_config["quality"][1]) + 1,
                )
            )
            if jpeg_config["always"]
            else 100
        )
        extension = ".jpg" if jpeg_config["always"] else ".png"
        file_name = f"{record['sample_id']}{extension}"
        output_path = image_dir / file_name
        _write_image(output_path, image, quality)
        record["file_name"] = f"images/{file_name}"
        record["image_sha256"] = _sha256_file(output_path)
        record["jpeg_quality"] = quality if jpeg_config["always"] else None
        records.append(record)
        preview_images.append(image)
        if record["passed"]:
            accepted_phashes.append(str(record["dedup"]["phash"]))
        coco_images.append(
            {
                "id": sample_index,
                "file_name": record["file_name"],
                "width": int(record["width"]),
                "height": int(record["height"]),
                "scenario": scenario,
                "sample_id": record["sample_id"],
            }
        )
        for instance in record["instances"]:
            bbox = [float(value) for value in instance["bbox_xywh"]]
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": sample_index,
                    "category_id": category_ids[str(instance["class_name"])],
                    "bbox": bbox,
                    "area": float(bbox[2] * bbox[3]),
                    "iscrowd": 0,
                    "segmentation": [],
                    "instance_id": instance["instance_id"],
                }
            )
            annotation_id += 1
    coco_output = {
        "info": {
            "description": "SafeSynth M10 deterministic preview",
            "seed": seed,
            "hard_negative_status": "H6 approved (0/64); procedural bank wired, distractors unannotated by construction",
            "generation_method": (
                "reference_conditioned_boundary_inpaint_v1"
                if generative_inpainter is not None
                else "registered_feathered_alpha_draft"
            ),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco["categories"],
    }
    coco_path = output_dir / "annotations.json"
    _write_json(coco_path, coco_output)
    _write_jsonl(output_dir / "records.jsonl", records)
    self_map = _self_eval(coco_path)
    scenario_class_size: Counter[tuple[str, str, str]] = Counter()
    small_distant_sides: list[float] = []
    for record in records:
        for instance in record["instances"]:
            if instance["kind"] != "pasted":
                continue
            bbox = instance["bbox_xywh"]
            area = float(bbox[2]) * float(bbox[3])
            size_bucket = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            scenario_class_size[
                (str(record["scenario"]), str(instance["class_name"]), size_bucket)
            ] += 1
            if record["scenario"] == "small_distant":
                small_distant_sides.append(min(float(bbox[2]), float(bbox[3])))
    target_low, target_high = (
        float(value)
        for value in config["scenarios"]["small_distant"]["target_min_side_px"]
    )
    summary = {
        "output_dir": str(output_dir),
        "n_images": len(records),
        "n_annotations": len(coco_annotations),
        "passed": sum(bool(record["passed"]) for record in records),
        "rejected": sum(not bool(record["passed"]) for record in records),
        "scenario_counts": dict(sorted(Counter(scenarios).items())),
        "first_reject_reasons": dict(
            sorted(
                Counter(
                    record["first_reject_reason"]
                    for record in records
                    if record["first_reject_reason"]
                ).items()
            )
        ),
        "coco_self_map": self_map,
        "hard_negative_status": "h6_approved_procedural_bank_wired",
        "blending_method": str(config["compose"]["blending"]["method"]),
        "generative_inpaint_used": generative_inpainter is not None,
        "person_crowded_fallback": person_crowded_fallback,
        "scenario_class_size": {
            "|".join(key): value
            for key, value in sorted(scenario_class_size.items())
        },
        "small_distant": {
            "pasted_instances": len(small_distant_sides),
            "min_side_min": min(small_distant_sides, default=None),
            "min_side_max": max(small_distant_sides, default=None),
            "target_range": [target_low, target_high],
            "within_target_range": sum(
                target_low <= value <= target_high for value in small_distant_sides
            ),
        },
        "image_hashes": {
            record["sample_id"]: record["image_sha256"] for record in records
        },
    }
    _write_json(output_dir / "summary.json", summary)
    if draw_boxes:
        _write_preview_grid(
            records,
            preview_images,
            paths.figures / f"preview_{output_tag}.png",
        )
    return summary


def _write_stats_report(
    summary: Mapping[str, Any],
    paths: ProjectPaths,
    *,
    report_tag: str = "synthetic_stats",
) -> None:
    lines = [
        "# M10 synthetic preview statistics",
        "",
        f"- Images: {summary['n_images']}",
        f"- Annotations: {summary['n_annotations']}",
        f"- Filter pass/reject: {summary['passed']} / {summary['rejected']}",
        f"- COCO self-evaluation bbox mAP: `{summary['coco_self_map']:.3f}`",
        f"- Output: `{summary['output_dir']}`",
        "- Hard negatives: procedural bank wired; distractors carry no annotation (ADR-004).",
        (
            "- Generative boundary inpainting: "
            f"**{'used' if summary['generative_inpaint_used'] else 'not used'}**."
        ),
        "",
        "## Scenario counts",
        "",
        "| scenario | count |",
        "|---|---:|",
    ]
    for scenario, count in summary["scenario_counts"].items():
        lines.append(f"| {scenario} | {count} |")
    lines.extend(
        [
            "",
            "## First filter rejection reason",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
    )
    for reason, count in summary["first_reject_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Pasted instances by scenario, class, and COCO size bucket",
            "",
            "| scenario | class | size | count |",
            "|---|---|---|---:|",
        ]
    )
    for key, count in summary["scenario_class_size"].items():
        scenario, class_name, size = key.split("|")
        lines.append(f"| {scenario} | {class_name} | {size} | {count} |")
    small = summary["small_distant"]
    lines.extend(
        [
            "",
            "## `small_distant` contract",
            "",
            f"- Pasted instances: {small['pasted_instances']}",
            f"- Observed min-side range: {small['min_side_min']}–{small['min_side_max']} px",
            f"- Config target range: {small['target_range']} px",
            (
                f"- Within target range: {small['within_target_range']} / "
                f"{small['pasted_instances']}"
            ),
            "",
        ]
    )
    (paths.reports / f"{report_tag}.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-tag", default="m10_seed42")
    parser.add_argument("--draw-boxes", action="store_true")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=(*SCENARIO_ORDER, *EXPERIMENTAL_SCENARIOS),
    )
    parser.add_argument("--stats-report-tag", default="synthetic_stats")
    parser.add_argument(
        "--blending-method",
        choices=("feathered_alpha", "poisson"),
    )
    parser.add_argument(
        "--generative-inpaint",
        action="store_true",
        help="run the pinned local-only Option A boundary inpainter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0 or args.n > MAX_POOL_IMAGES:
        raise ValueError(
            f"ADR-011 pool guard: requested {args.n} candidates, allowed range is "
            f"[1, {MAX_POOL_IMAGES}]. The binding rule is the ACCEPTED cap of "
            f"{TARGET_ACCEPTED_1X} (1x); H4 did not pass so 2x is forbidden."
        )
    paths = load_project_paths()
    generative_inpainter = None
    if args.generative_inpaint:
        from src.synthetic.generative_inpaint import (
            GenerativeBoundaryInpainter,
            load_flux2_pipeline,
            load_generative_config,
            model_directory,
        )

        generative_config = load_generative_config()
        local_model_dir = model_directory(paths, generative_config)
        generative_inpainter = GenerativeBoundaryInpainter(
            load_flux2_pipeline(
                model_dir=local_model_dir,
                config=generative_config,
            ),
            generative_config,
        )
    summary = generate(
        paths=paths,
        n=args.n,
        seed=args.seed,
        output_tag=args.output_tag,
        selected_scenarios=args.scenario,
        draw_boxes=args.draw_boxes,
        blending_method=args.blending_method,
        generative_inpainter=generative_inpainter,
    )
    _write_stats_report(summary, paths, report_tag=args.stats_report_tag)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
