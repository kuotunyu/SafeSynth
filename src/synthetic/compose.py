"""Deterministic, provenance-complete synthetic composition preview engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    recompute_visible_annotations,
    seam_energy_ratio,
    tight_bbox,
    warp_rgba,
)

SCENARIO_ORDER = (
    "small_distant",
    "head_no_helmet",
    "partial_occlusion",
    "crowded",
    "hard_negative",
    "low_light_blur",
)


@dataclass
class Paste:
    layer: Layer
    rgba: np.ndarray
    frame_slice: tuple[slice, slice]
    patch_slice: tuple[slice, slice]
    bank: dict[str, Any]
    bbox_preclip: list[float]


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


def _phash_hex(image_rgb: np.ndarray) -> str:
    return str(imagehash.phash(Image.fromarray(image_rgb), hash_size=8))


def _hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


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
    if any(
        str(item["src_split"]) != "train" or int(item["src_image_id"]) in test_ids
        for item in bank
    ):
        raise RuntimeError("Cutout bank contains a Val/Test source")
    return coco, bank, train_images, annotations, frozen, test_ids


def _load_pass1(
    paths: ProjectPaths, image_id: int
) -> dict[int, dict[str, Any]]:
    path = paths.masks_pass1 / "records" / f"{image_id:06d}.json"
    record = _read_json(path)
    return {
        int(annotation["annotation_id"]): annotation
        for annotation in record["annotations"]
    }


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
    preferred = [item for item in candidates if item["preferred_tier"]]
    population = preferred if preferred and rng.random() < 0.85 else candidates
    return population[int(rng.integers(0, len(population)))]


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
) -> float:
    if scenario == "small_distant":
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
) -> Paste | None:
    rgba = np.asarray(Image.open(paths.cutouts / item["file"]).convert("RGBA"))
    scale = _transform_scale(
        scenario=scenario,
        settings=settings,
        class_name=class_name,
        rgba=rgba,
        config=config,
        rng=rng,
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
        output = alpha_composite(
            output,
            rgb,
            alpha,
            frame_slice=paste.frame_slice,
            patch_slice=paste.patch_slice,
        )
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
) -> tuple[np.ndarray, dict[str, Any]]:
    image_path = paths.hardhat_raw / str(background["file_name"])
    original = np.asarray(Image.open(image_path).convert("RGB"))
    image_shape = original.shape[:2]
    pass1 = _load_pass1(paths, int(background["id"]))
    settings = config["scenarios"][scenario]
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
        do_swap = (
            scenario == "head_no_helmet"
            and bool(eligible_swap_helmets)
            and (
                not person_boxes
                or rng.random()
                < float(settings["submode_helmet_to_head_swap_prob"])
            )
        )
        if do_swap:
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
        if not pastes and scenario != "low_light_blur":
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
            "intentional_removals": sorted(intentional_removals),
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
                "hard_negative_used": False,
            },
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
        unknown = set(selected) - set(SCENARIO_ORDER)
        if unknown:
            raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
        names = list(selected)
    else:
        # M9 remains explicitly blocked on kuotunyu's contact-sheet signoff.
        names = [name for name in SCENARIO_ORDER if name != "hard_negative"]
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
) -> dict[str, Any]:
    config, filter_config = _load_configs()
    config["seed"] = seed
    rng = np.random.default_rng(seed)
    coco, bank, train_images, annotations, frozen, _ = _load_context(paths)
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
                    rng=rng,
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
                background_id = int(rng.choice(train_ids))
        record["generation"]["background_retries"] = background_retries
        quality = (
            int(rng.integers(int(jpeg_config["quality"][0]), int(jpeg_config["quality"][1]) + 1))
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
            "hard_negative_status": "pending kuotunyu signoff; excluded",
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
        "hard_negative_status": "blocked_pending_kuotunyu_signoff",
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


def _write_stats_report(summary: Mapping[str, Any], paths: ProjectPaths) -> None:
    lines = [
        "# M10 synthetic preview statistics",
        "",
        f"- Images: {summary['n_images']}",
        f"- Annotations: {summary['n_annotations']}",
        f"- Filter pass/reject: {summary['passed']} / {summary['rejected']}",
        f"- COCO self-evaluation bbox mAP: `{summary['coco_self_map']:.3f}`",
        f"- Output: `{summary['output_dir']}`",
        "- Hard negatives: **not used**; M9 remains blocked on kuotunyu signoff.",
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
    (paths.reports / "synthetic_stats.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-tag", default="m10_seed42")
    parser.add_argument("--draw-boxes", action="store_true")
    parser.add_argument("--scenario", action="append", choices=SCENARIO_ORDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0 or args.n > 300:
        raise ValueError("M11 hard gate: preview count must be in [1, 300]")
    if args.scenario and "hard_negative" in args.scenario:
        raise RuntimeError(
            "Hard-negative generation is blocked until kuotunyu signs "
            "reports/h6_hard_negative_spike.md."
        )
    paths = load_project_paths()
    summary = generate(
        paths=paths,
        n=args.n,
        seed=args.seed,
        output_tag=args.output_tag,
        selected_scenarios=args.scenario,
        draw_boxes=args.draw_boxes,
    )
    _write_stats_report(summary, paths)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
