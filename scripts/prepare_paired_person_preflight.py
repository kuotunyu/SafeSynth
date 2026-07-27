"""Render a fixed CPU-only preflight for coupled person + headlike units."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT, ProjectPaths, load_project_paths
from src.synthetic.compose import (
    Paste,
    _archive_existing,
    _existing_layers,
    _load_configs,
    _load_context,
    _load_pass1,
    _render_pastes,
    _visible_paste_masks,
    _write_image,
    prepare_context_replacement_background,
)
from src.synthetic.composition import Layer, recompute_visible_annotations, tight_bbox
from src.synthetic.paired_person import (
    PairedPersonUnit,
    intersection_over_smaller_box,
    paired_headlike_annotations,
    strict_unit_reject_reasons,
    transform_linked_box,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    text = "\n".join(
        json.dumps(record, sort_keys=True, separators=(",", ":"))
        for record in records
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def load_train_only_clip_embeddings(
    path: Path,
    *,
    coco_images: Sequence[Mapping[str, Any]],
    train_image_ids: set[int],
) -> tuple[np.ndarray, dict[int, int]]:
    """Materialize only frozen-Train rows from the pre-split feature artifact."""

    full = np.load(path, mmap_mode="r", allow_pickle=False)
    if full.shape[0] != len(coco_images):
        raise RuntimeError("CLIP embedding row count changed")
    source_row_by_image = {
        int(image["id"]): index
        for index, image in enumerate(coco_images)
    }
    if not train_image_ids <= set(source_row_by_image):
        raise RuntimeError("A frozen Train image lacks a CLIP feature row")
    ordered_train_ids = sorted(train_image_ids)
    train_only = np.stack(
        [
            np.asarray(full[source_row_by_image[image_id]])
            for image_id in ordered_train_ids
        ]
    )
    row_by_image = {
        image_id: index
        for index, image_id in enumerate(ordered_train_ids)
    }
    if set(row_by_image) != train_image_ids:
        raise AssertionError("Non-Train CLIP row entered the candidate mapping")
    return train_only, row_by_image


def eligible_units(
    *,
    bank: Sequence[Mapping[str, Any]],
    annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    categories: Mapping[int, str],
    image_height_by_id: Mapping[int, float],
    donor_config: Mapping[str, Any],
) -> tuple[list[PairedPersonUnit], Counter[str]]:
    """Apply the frozen donor gates and retain one linked label per person."""

    units: list[PairedPersonUnit] = []
    rejected: Counter[str] = Counter()
    for source in bank:
        if str(source["class_name"]) != "person":
            continue
        pairs = paired_headlike_annotations(
            source["src_bbox_xywh"],
            annotations=annotations_by_image[int(source["src_image_id"])],
            categories=categories,
        )
        if len(pairs) != int(donor_config["paired_headlike_count"]):
            rejected["PAIRED_HEADLIKE_COUNT"] += 1
            continue
        reasons = strict_unit_reject_reasons(
            source,
            pairs[0],
            preferred_tier_required=bool(
                donor_config["preferred_tier_required"]
            ),
            min_person_height_px=float(donor_config["min_person_height_px"]),
            min_person_aspect_height_over_width=float(
                donor_config["min_person_aspect_height_over_width"]
            ),
            max_head_center_y_fraction=float(
                donor_config["max_head_center_y_fraction"]
            ),
            max_head_width_fraction=float(
                donor_config["max_head_width_fraction"]
            ),
            max_edge_touch_top=float(donor_config["max_edge_touch_top"]),
            max_edge_touch_side=float(donor_config["max_edge_touch_side"]),
            min_source_person_bottom_fraction=float(
                donor_config["min_source_person_bottom_fraction"]
            ),
            source_image_height=float(
                image_height_by_id[int(source["src_image_id"])]
            ),
        )
        if reasons:
            rejected[reasons[0]] += 1
            continue
        units.append(
            PairedPersonUnit(
                person=dict(source),
                headlike=pairs[0],
                headlike_class=categories[int(pairs[0]["category_id"])],
            )
        )
    units.sort(key=lambda unit: str(unit.person["cutout_id"]))
    return units, rejected


def _containing_person(
    anchor: Mapping[str, Any],
    *,
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
) -> Mapping[str, Any] | None:
    anchor_x, anchor_y, anchor_width, anchor_height = (
        float(value) for value in anchor["bbox"]
    )
    center_x = anchor_x + anchor_width / 2
    center_y = anchor_y + anchor_height / 2
    candidates = []
    for source in annotations:
        if categories[int(source["category_id"])] != "person":
            continue
        x, y, width, height = (float(value) for value in source["bbox"])
        if x <= center_x <= x + width and y <= center_y <= y + 0.55 * height:
            candidates.append((width * height, int(source["id"]), source))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _transformed_rgba(
    rgba: np.ndarray,
    *,
    scale: float,
    hflip: bool,
) -> np.ndarray:
    width = max(1, round(rgba.shape[1] * scale))
    height = max(1, round(rgba.shape[0] * scale))
    resized = cv2.resize(
        rgba,
        (width, height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    if hflip:
        resized = np.ascontiguousarray(resized[:, ::-1])
    return resized


def _candidate_paste(
    *,
    unit: PairedPersonUnit,
    anchor: Mapping[str, Any],
    anchor_person: Mapping[str, Any] | None,
    annotations: Sequence[Mapping[str, Any]],
    image_shape: tuple[int, int],
    placement_config: Mapping[str, Any],
    paths: ProjectPaths,
    hflip: bool,
    gap_px: int,
    side: str,
) -> tuple[Paste, list[float], list[float], float] | None:
    donor_head_width = float(unit.headlike["bbox"][2])
    donor_head_height = float(unit.headlike["bbox"][3])
    anchor_x, anchor_y, anchor_width, anchor_height = (
        float(value) for value in anchor["bbox"]
    )
    head_scale = float(
        np.sqrt(
            (anchor_width * anchor_height)
            / max(donor_head_width * donor_head_height, 1.0)
        )
    )
    scale = head_scale
    if anchor_person is not None:
        target_person_height = float(anchor_person["bbox"][3])
        donor_person_height = float(unit.person["src_bbox_xywh"][3])
        person_scale = target_person_height / max(donor_person_height, 1.0)
        scale = float(np.sqrt(head_scale * person_scale))
    if not (
        float(placement_config["min_scale"])
        <= scale
        <= float(placement_config["max_scale"])
    ):
        return None

    rgba = np.asarray(
        Image.open(paths.cutouts / str(unit.person["file"])).convert("RGBA")
    )
    transformed = _transformed_rgba(rgba, scale=scale, hflip=hflip)
    patch_height, patch_width = transformed.shape[:2]
    local_head = transform_linked_box(
        unit.headlike["bbox"],
        source_person_box_xywh=unit.person["src_bbox_xywh"],
        scale=scale,
        hflip=hflip,
        patch_width=patch_width,
        patch_left=0,
        patch_top=0,
    )

    if anchor_person is None:
        patch_top = round(
            anchor_y + anchor_height / 2
            - (local_head[1] + local_head[3] / 2)
        )
        obstacle_left = anchor_x
        obstacle_right = anchor_x + anchor_width
    else:
        person_x, person_y, person_width, person_height = (
            float(value) for value in anchor_person["bbox"]
        )
        patch_top = round(person_y + person_height - patch_height)
        obstacle_left = person_x
        obstacle_right = person_x + person_width
        transformed_head_center_y = (
            patch_top + local_head[1] + local_head[3] / 2
        )
        anchor_center_y = anchor_y + anchor_height / 2
        if abs(transformed_head_center_y - anchor_center_y) > 1.5 * max(
            anchor_height,
            local_head[3],
        ):
            return None
    patch_left = (
        round(obstacle_left - gap_px - patch_width)
        if side == "left"
        else round(obstacle_right + gap_px)
    )

    margin = int(placement_config["frame_margin_px"])
    image_height, image_width = image_shape
    if (
        patch_left < margin
        or patch_top < margin
        or patch_left + patch_width > image_width - margin
        or patch_top + patch_height > image_height - margin
    ):
        return None

    alpha_mask = transformed[..., 3] >= 128
    full_mask = np.zeros(image_shape, dtype=bool)
    full_mask[
        patch_top : patch_top + patch_height,
        patch_left : patch_left + patch_width,
    ] = alpha_mask
    person_box = tight_bbox(full_mask)
    if person_box is None:
        return None
    person_bottom_fraction = (
        float(person_box[1]) + float(person_box[3])
    ) / image_height
    if not (
        float(placement_config["min_person_bottom_fraction"])
        <= person_bottom_fraction
        <= float(placement_config["max_person_bottom_fraction"])
    ):
        return None
    maximum_overlap = max(
        (
            intersection_over_smaller_box(person_box, source["bbox"])
            for source in annotations
        ),
        default=0.0,
    )
    if maximum_overlap > float(
        placement_config["max_annotation_overlap_fraction"]
    ):
        return None

    head_box = transform_linked_box(
        unit.headlike["bbox"],
        source_person_box_xywh=unit.person["src_bbox_xywh"],
        scale=scale,
        hflip=hflip,
        patch_width=patch_width,
        patch_left=patch_left,
        patch_top=patch_top,
    )
    layer = Layer(
        instance_id="paste:paired_person",
        class_name="person",
        kind="pasted",
        mask=full_mask,
        bbox_xywh_original=person_box,
        y_bottom=float(person_box[1]) + float(person_box[3]),
    )
    paste = Paste(
        layer=layer,
        rgba=transformed,
        frame_slice=(
            slice(patch_top, patch_top + patch_height),
            slice(patch_left, patch_left + patch_width),
        ),
        patch_slice=(slice(0, patch_height), slice(0, patch_width)),
        bank=unit.person,
        bbox_preclip=[
            float(patch_left),
            float(patch_top),
            float(patch_width),
            float(patch_height),
        ],
    )
    return paste, person_box, head_box, scale


def _source_position_paste(
    *,
    unit: PairedPersonUnit,
    annotations: Sequence[Mapping[str, Any]],
    image_shape: tuple[int, int],
    source_image_size: tuple[int, int],
    placement_config: Mapping[str, Any],
    paths: ProjectPaths,
    hflip: bool,
    scale: float,
    horizontal_jitter_fraction: float,
) -> tuple[Paste, list[float], list[float], float] | None:
    rgba = np.asarray(
        Image.open(paths.cutouts / str(unit.person["file"])).convert("RGBA")
    )
    transformed = _transformed_rgba(rgba, scale=scale, hflip=hflip)
    patch_height, patch_width = transformed.shape[:2]
    image_height, image_width = image_shape
    source_width, source_height = source_image_size
    source_x, source_y, _, source_person_height = (
        float(value) for value in unit.person["src_bbox_xywh"]
    )
    position_scale_x = image_width / max(float(source_width), 1.0)
    position_scale_y = image_height / max(float(source_height), 1.0)
    patch_left = round(
        source_x * position_scale_x
        + horizontal_jitter_fraction * image_width
    )
    source_bottom = (
        source_y + source_person_height
    ) * position_scale_y
    patch_top = round(source_bottom - patch_height)

    margin = int(placement_config["frame_margin_px"])
    if (
        patch_left < margin
        or patch_top < margin
        or patch_left + patch_width > image_width - margin
        or patch_top + patch_height > image_height - margin
    ):
        return None
    alpha_mask = transformed[..., 3] >= 128
    full_mask = np.zeros(image_shape, dtype=bool)
    full_mask[
        patch_top : patch_top + patch_height,
        patch_left : patch_left + patch_width,
    ] = alpha_mask
    person_box = tight_bbox(full_mask)
    if person_box is None:
        return None
    person_bottom_fraction = (
        float(person_box[1]) + float(person_box[3])
    ) / image_height
    if not (
        float(placement_config["min_person_bottom_fraction"])
        <= person_bottom_fraction
        <= float(placement_config["max_person_bottom_fraction"])
    ):
        return None
    maximum_overlap = max(
        (
            intersection_over_smaller_box(person_box, source["bbox"])
            for source in annotations
        ),
        default=0.0,
    )
    if maximum_overlap > float(
        placement_config["max_annotation_overlap_fraction"]
    ):
        return None
    head_box = transform_linked_box(
        unit.headlike["bbox"],
        source_person_box_xywh=unit.person["src_bbox_xywh"],
        scale=scale,
        hflip=hflip,
        patch_width=patch_width,
        patch_left=patch_left,
        patch_top=patch_top,
    )
    layer = Layer(
        instance_id="paste:paired_person",
        class_name="person",
        kind="pasted",
        mask=full_mask,
        bbox_xywh_original=person_box,
        y_bottom=float(person_box[1]) + float(person_box[3]),
    )
    paste = Paste(
        layer=layer,
        rgba=transformed,
        frame_slice=(
            slice(patch_top, patch_top + patch_height),
            slice(patch_left, patch_left + patch_width),
        ),
        patch_slice=(slice(0, patch_height), slice(0, patch_width)),
        bank=unit.person,
        bbox_preclip=[
            float(patch_left),
            float(patch_top),
            float(patch_width),
            float(patch_height),
        ],
    )
    return paste, person_box, head_box, scale


def _square_crop(
    image: Image.Image,
    bbox_xywh: Sequence[float],
    *,
    padding_fraction: float = 0.35,
) -> Image.Image:
    x, y, width, height = (float(value) for value in bbox_xywh)
    side = min(
        max(width, height) * (1 + 2 * padding_fraction),
        min(image.size),
    )
    center_x = x + width / 2
    center_y = y + height / 2
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
    left = min(max(0, left), image.width - int(side))
    top = min(max(0, top), image.height - int(side))
    return image.crop((left, top, left + int(side), top + int(side)))


def _marked(
    image_rgb: np.ndarray,
    *,
    person_bbox_xywh: Sequence[float],
    head_bbox_xywh: Sequence[float],
) -> Image.Image:
    image = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    person_x, person_y, person_width, person_height = person_bbox_xywh
    head_x, head_y, head_width, head_height = head_bbox_xywh
    draw.rectangle(
        (
            person_x,
            person_y,
            person_x + person_width,
            person_y + person_height,
        ),
        outline=(0, 255, 255),
        width=3,
    )
    draw.rectangle(
        (head_x, head_y, head_x + head_width, head_y + head_height),
        outline=(255, 230, 0),
        width=3,
    )
    return image


def render_sheet(
    *,
    images: Sequence[np.ndarray],
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Show each full draft and an enlarged coupled-person crop."""

    if len(images) != 64 or len(records) != 64:
        raise ValueError("Paired-person preflight must contain exactly 64 cells")
    panel = 224
    caption = 28
    cell_width = panel * 2
    cell_height = panel + caption
    sheet = Image.new("RGB", (cell_width * 8, cell_height * 8), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image_rgb, record) in enumerate(
        zip(images, records, strict=True),
        start=1,
    ):
        marked = _marked(
            image_rgb,
            person_bbox_xywh=record["person_bbox_xywh"],
            head_bbox_xywh=record["head_bbox_xywh"],
        )
        full = marked.resize((panel, panel), Image.Resampling.LANCZOS)
        crop = _square_crop(
            marked,
            record["person_bbox_xywh"],
        ).resize((panel, panel), Image.Resampling.LANCZOS)
        x0 = ((index - 1) % 8) * cell_width
        y0 = ((index - 1) // 8) * cell_height
        sheet.paste(full, (x0, y0))
        sheet.paste(crop, (x0 + panel, y0))
        draw.text(
            (x0 + 4, y0 + panel + 6),
            (
                f"{index:02d} | {record['sample_id']} | "
                f"{record['headlike_class']}"
            ),
            fill="black",
        )
        draw.rectangle(
            (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
            outline=(60, 60, 60),
            width=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)


def _try_background(
    *,
    background_id: int,
    units: Sequence[PairedPersonUnit],
    unit_use_counts: Counter[str],
    train_images: Mapping[int, Mapping[str, Any]],
    annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    frozen: Mapping[int, Mapping[str, Any]],
    categories: Mapping[int, str],
    compose_config: Mapping[str, Any],
    filter_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    clip_embeddings: np.ndarray,
    clip_row_by_image: Mapping[int, int],
    paths: ProjectPaths,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    image_record = train_images[background_id]
    raw = np.asarray(
        Image.open(
            paths.hardhat_raw / str(image_record["file_name"])
        ).convert("RGB")
    )
    prepared = prepare_context_replacement_background(
        image_rgb=raw,
        annotations=annotations_by_image[background_id],
        categories=categories,
        pass1=_load_pass1(paths, background_id),
        guard_config=compose_config["compose"]["context_replacement"][
            "input_guard"
        ],
        output_shape=raw.shape[:2],
        transform_masks=True,
    )
    if not prepared.guard.accepted:
        return None
    annotations = list(prepared.annotations)
    annotation_by_id = {
        int(annotation["id"]): annotation for annotation in annotations
    }
    anchor_ids = list(prepared.guard.eligible_annotation_ids)
    rng.shuffle(anchor_ids)
    background_embedding = clip_embeddings[clip_row_by_image[background_id]]
    scene_candidates = []
    for unit_index, unit in enumerate(units):
        donor_image_id = int(unit.person["src_image_id"])
        similarity = float(
            background_embedding
            @ clip_embeddings[clip_row_by_image[donor_image_id]]
        )
        if similarity >= float(
            candidate_config["scene_matching"][
                "min_full_image_cosine_similarity"
            ]
        ):
            scene_candidates.append((similarity, unit_index))
    scene_candidates.sort(
        key=lambda item: (
            -item[0],
            str(units[item[1]].person["cutout_id"]),
        )
    )
    sides = ["left", "right"]
    rng.shuffle(sides)
    hflips = [
        bool(rng.random() < float(candidate_config["placement"]["hflip_prob"])),
        bool(rng.random() >= float(candidate_config["placement"]["hflip_prob"])),
    ]
    if (
        candidate_config["placement"]["mode"]
        == "source_normalized_position"
    ):
        anchor_ids = [None]
    for anchor_id in anchor_ids:
        if anchor_id is None:
            anchor = None
            anchor_person = None
        else:
            anchor = annotation_by_id[anchor_id]
            anchor_person = _containing_person(
                anchor,
                annotations=annotations,
                categories=categories,
            )
        for scene_similarity, unit_index in scene_candidates:
            unit = units[unit_index]
            cutout_id = str(unit.person["cutout_id"])
            if unit_use_counts[cutout_id] >= int(
                candidate_config["donor"]["max_uses_per_cutout"]
            ):
                continue
            if int(unit.person["src_group_id"]) == int(
                frozen[background_id]["group_id"]
            ):
                continue
            placement_mode = str(candidate_config["placement"]["mode"])
            if placement_mode == "source_normalized_position":
                placement_attempts = [
                    (
                        hflip,
                        float(scale),
                        float(jitter),
                        None,
                        None,
                    )
                    for scale in candidate_config["placement"][
                        "scale_candidates"
                    ]
                    for jitter in candidate_config["placement"][
                        "horizontal_jitter_fractions"
                    ]
                    for hflip in hflips
                ]
            else:
                placement_attempts = [
                    (hflip, 1.0, 0.0, side, gap)
                    for hflip in hflips
                    for side in sides
                    for gap in (6, 12, 20)
                ]
            for hflip, scale, jitter, side, gap in placement_attempts:
                        if placement_mode == "source_normalized_position":
                            source_image = train_images[
                                int(unit.person["src_image_id"])
                            ]
                            candidate = _source_position_paste(
                                unit=unit,
                                annotations=annotations,
                                image_shape=prepared.image_rgb.shape[:2],
                                source_image_size=(
                                    int(source_image["width"]),
                                    int(source_image["height"]),
                                ),
                                placement_config=candidate_config["placement"],
                                paths=paths,
                                hflip=hflip,
                                scale=scale,
                                horizontal_jitter_fraction=jitter,
                            )
                        else:
                            if anchor is None or side is None or gap is None:
                                raise AssertionError(
                                    "Anchor placement inputs are missing"
                                )
                            candidate = _candidate_paste(
                                unit=unit,
                                anchor=anchor,
                                anchor_person=anchor_person,
                                annotations=annotations,
                                image_shape=prepared.image_rgb.shape[:2],
                                placement_config=candidate_config["placement"],
                                paths=paths,
                                hflip=hflip,
                                gap_px=gap,
                                side=side,
                            )
                        if candidate is None:
                            continue
                        paste, person_box, head_box, scale = candidate
                        existing_layers = _existing_layers(
                            image_shape=prepared.image_rgb.shape[:2],
                            annotations=annotations,
                            pass1=prepared.pass1,
                            categories=categories,
                            intentional_removals=set(),
                        )
                        visibility = recompute_visible_annotations(
                            [*existing_layers, paste.layer],
                            min_visible_fraction_pasted=0.95,
                            existing_keep_original_above=float(
                                compose_config["compose"]["bbox"][
                                    "existing_keep_original_above"
                                ]
                            ),
                            existing_recompute_above=float(
                                compose_config["compose"]["bbox"][
                                    "existing_recompute_above"
                                ]
                            ),
                        )
                        if not visibility.accepted:
                            continue
                        render_config = json.loads(
                            json.dumps(compose_config)
                        )
                        render_config["filtering"] = filter_config
                        render_config["compose"]["harmonization"][
                            "enabled"
                        ] = bool(
                            candidate_config["placement"][
                                "whole_person_lab_harmonization"
                            ]
                        )
                        render_config["compose"]["harmonization"][
                            "noise_match"
                        ] = bool(
                            candidate_config["placement"][
                                "whole_person_noise_matching"
                            ]
                        )
                        rendered, _ = _render_pastes(
                            base_rgb=prepared.image_rgb,
                            pastes=[paste],
                            existing_layers=existing_layers,
                            config=render_config,
                            rng=rng,
                        )
                        visible_mask = _visible_paste_masks(
                            existing_layers=existing_layers,
                            pastes=[paste],
                        )[paste.layer.instance_id]
                        if int(visible_mask.sum()) < 250:
                            continue
                        unit_use_counts[cutout_id] += 1
                        record = {
                            "sample_id": "",
                            "background_image_id": background_id,
                            "background_group_id": int(
                                frozen[background_id]["group_id"]
                            ),
                            "background_normalization_applied": bool(
                                prepared.normalization.applied
                            ),
                            "anchor_annotation_id": anchor_id,
                            "anchor_person_annotation_id": (
                                int(anchor_person["id"])
                                if anchor_person is not None
                                else None
                            ),
                            "donor_cutout_id": cutout_id,
                            "donor_image_id": int(unit.person["src_image_id"]),
                            "donor_group_id": int(unit.person["src_group_id"]),
                            "donor_headlike_annotation_id": int(
                                unit.headlike["id"]
                            ),
                            "headlike_class": unit.headlike_class,
                            "person_bbox_xywh": person_box,
                            "head_bbox_xywh": head_box,
                            "scale": scale,
                            "hflip": hflip,
                            "placement_mode": placement_mode,
                            "horizontal_jitter_fraction": jitter,
                            "gap_px": gap,
                            "side": side,
                            "scene_clip_cosine_similarity": scene_similarity,
                            "visible_person_mask_pixels": int(
                                visible_mask.sum()
                            ),
                            "model_inference_run": False,
                            "test_image_read": False,
                        }
                        return rendered, record
    return None


def main() -> None:
    candidate_config = yaml.safe_load(
        (
            PROJECT_ROOT / "configs" / "paired_person_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    if (
        candidate_config["status"]
        != "candidate_v7_cpu_preflight_no_model_output"
        or candidate_config["architecture"]
        != "paired_person_scene_position_insert_v7"
        or int(candidate_config["n_images"]) != 64
        or bool(candidate_config["model_gate"]["model_inference_allowed"])
    ):
        raise RuntimeError(
            "Paired-person CPU candidate is not an active frozen run; "
            "see its archived outcome"
        )

    paths = load_project_paths()
    compose_config, filter_config = _load_configs()
    (
        coco,
        bank,
        train_images,
        annotations_by_image,
        frozen,
        test_ids,
    ) = _load_context(paths)
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco["categories"]
    }
    units, donor_rejections = eligible_units(
        bank=bank,
        annotations_by_image=annotations_by_image,
        categories=categories,
        image_height_by_id={
            int(image["id"]): float(image["height"])
            for image in coco["images"]
        },
        donor_config=candidate_config["donor"],
    )
    if not units:
        raise RuntimeError("No strict paired-person units passed")
    if any(int(unit.person["src_image_id"]) in test_ids for unit in units):
        raise RuntimeError("Paired-person candidate leaked Test")
    clip_embeddings, clip_row_by_image = load_train_only_clip_embeddings(
        paths.interim
        / str(candidate_config["scene_matching"]["feature_file"]),
        coco_images=coco["images"],
        train_image_ids=set(train_images),
    )
    if set(clip_row_by_image) & test_ids:
        raise RuntimeError("Test CLIP row entered the candidate mapping")

    seed = int(candidate_config["root_seed"])
    rng = np.random.default_rng(seed)
    background_ids = np.asarray(sorted(train_images), dtype=np.int64)
    rng.shuffle(background_ids)
    unit_use_counts: Counter[str] = Counter()
    images: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    attempted_backgrounds = 0
    for background_id_raw in background_ids.tolist():
        if len(records) >= int(candidate_config["n_images"]):
            break
        attempted_backgrounds += 1
        background_id = int(background_id_raw)
        result = _try_background(
            background_id=background_id,
            units=units,
            unit_use_counts=unit_use_counts,
            train_images=train_images,
            annotations_by_image=annotations_by_image,
            frozen=frozen,
            categories=categories,
            compose_config=compose_config,
            filter_config=filter_config,
            candidate_config=candidate_config,
            clip_embeddings=clip_embeddings,
            clip_row_by_image=clip_row_by_image,
            paths=paths,
            rng=rng,
        )
        if result is None:
            continue
        rendered, record = result
        sample_index = len(records) + 1
        record["sample_id"] = f"paired_v6_{seed}_{sample_index:02d}"
        images.append(rendered)
        records.append(record)
    if len(records) != int(candidate_config["n_images"]):
        failure = {
            "schema_version": 1,
            "status": "capacity_infeasible_before_kuotunyu_review",
            "architecture": candidate_config["architecture"],
            "root_seed": seed,
            "required_images": int(candidate_config["n_images"]),
            "built_images": len(records),
            "attempted_train_backgrounds": attempted_backgrounds,
            "available_train_backgrounds": len(train_images),
            "strict_donor_cutouts": len(units),
            "strict_donor_groups": len(
                {int(unit.person["src_group_id"]) for unit in units}
            ),
            "max_uses_per_cutout": int(
                candidate_config["donor"]["max_uses_per_cutout"]
            ),
            "theoretical_maximum_images": len(units)
            * int(candidate_config["donor"]["max_uses_per_cutout"]),
            "used_donor_cutouts": len(unit_use_counts),
            "used_donor_groups": len(
                {int(record["donor_group_id"]) for record in records}
            ),
            "model_inference_run": False,
            "h4_auc_computed": False,
            "validation_images_read": 0,
            "test_images_read": 0,
            "test_feature_rows_read": 0,
            "reviewed_by": "mechanical_capacity_gate",
        }
        failure_stem = (
            f"h4_paired_person_input_preflight_seed{seed}_capacity_failed"
        )
        _write_json(paths.reports / f"{failure_stem}.json", failure)
        (
            paths.reports / f"{failure_stem}.md"
        ).write_text(
            "\n".join(
                [
                    "# H4 paired-person v7 capacity failure",
                    "",
                    "- Status: **capacity infeasible before kuotunyu review**",
                    (
                        f"- Built/required: **{len(records)} / "
                        f"{candidate_config['n_images']}**"
                    ),
                    (
                        "- Attempted Train backgrounds: "
                        f"**{attempted_backgrounds} / {len(train_images)}**"
                    ),
                    (
                        "- Strict donor cutouts/groups: "
                        f"**{len(units)} / "
                        f"{failure['strict_donor_groups']}**"
                    ),
                    (
                        "- Maximum uses per cutout: "
                        f"**{failure['max_uses_per_cutout']}**"
                    ),
                    (
                        "- Theoretical maximum: "
                        f"**{failure['theoretical_maximum_images']}**"
                    ),
                    "- Model inference run: **no**",
                    "- H4 AUC computed: **no**",
                    "- Validation/Test images read: **0 / 0**",
                    "",
                    (
                        "The fixed search exhausted every frozen Train "
                        "background and still produced only 63 of 64 required "
                        "drafts. Raising the reuse cap after this result would "
                        "hide the source-diversity failure, so the whole-person "
                        "paste architecture is stopped before human review or "
                        "GPU inference."
                    ),
                    "",
                ]
            ),
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(failure, indent=2, sort_keys=True))
        raise RuntimeError(
            f"Only built {len(records)}/{candidate_config['n_images']} "
            f"paired-person drafts from {attempted_backgrounds} backgrounds"
        )

    output_dir = paths.synthetic / f"h4_paired_person_preflight_seed{seed}"
    _archive_existing(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for image_rgb, record in zip(images, records, strict=True):
        file_name = f"{record['sample_id']}.png"
        _write_image(image_dir / file_name, image_rgb, 100)
        record["file_name"] = f"images/{file_name}"
        record["image_sha256"] = _sha256(image_dir / file_name)
    _write_jsonl(output_dir / "records.jsonl", records)

    artifact_stem = f"h4_paired_person_input_preflight_seed{seed}"
    sheet_path = paths.figures / f"{artifact_stem}.png"
    render_sheet(images=images, records=records, output_path=sheet_path)
    geometry = [
        {
            key: record[key]
            for key in (
                "sample_id",
                "background_image_id",
                "anchor_annotation_id",
                "donor_cutout_id",
                "person_bbox_xywh",
                "head_bbox_xywh",
                "scale",
                "hflip",
                "scene_clip_cosine_similarity",
            )
        }
        for record in records
    ]
    payload = {
        "schema_version": 1,
        "status": "pending_kuotunyu_input_review",
        "architecture": candidate_config["architecture"],
        "root_seed": seed,
        "n_images": len(records),
        "input_issue_max_count": int(
            candidate_config["input_issue_max_count"]
        ),
        "observed_input_issue_count": None,
        "reviewed_by": None,
        "model_inference_run": False,
        "h4_auc_computed": False,
        "test_images_read": 0,
        "test_feature_rows_read": 0,
        "available_feature_rows": len(clip_row_by_image),
        "feature_scope": "frozen Train rows only",
        "strict_donor_cutouts": len(units),
        "strict_donor_groups": len(
            {int(unit.person["src_group_id"]) for unit in units}
        ),
        "used_donor_cutouts": len(unit_use_counts),
        "used_donor_groups": len(
            {
                int(record["donor_group_id"])
                for record in records
            }
        ),
        "distinct_backgrounds": len(
            {int(record["background_image_id"]) for record in records}
        ),
        "attempted_backgrounds": attempted_backgrounds,
        "scene_matching": {
            "model_name": candidate_config["scene_matching"]["model_name"],
            "pretrained": candidate_config["scene_matching"]["pretrained"],
            "min_full_image_cosine_similarity": candidate_config[
                "scene_matching"
            ]["min_full_image_cosine_similarity"],
            "observed_min_cosine_similarity": min(
                float(record["scene_clip_cosine_similarity"])
                for record in records
            ),
        },
        "donor_first_reject_reasons": dict(sorted(donor_rejections.items())),
        "geometry_fingerprint_sha256": _canonical_sha256(geometry),
        "contact_sheet": str(sheet_path),
        "contact_sheet_sha256": _sha256(sheet_path),
        "output_dir": str(output_dir),
    }
    _write_json(
        paths.reports / f"{artifact_stem}.json",
        payload,
    )
    markdown = [
        "# H4 paired-person v6 CPU input preflight",
        "",
        "- Status: **pending kuotunyu input review**",
        f"- Inputs: **{payload['n_images']}**",
        f"- Root seed: `{payload['root_seed']}`",
        "- Model inference run: **no**",
        "- H4 AUC computed: **no**",
        "- Validation/Test images read: **0 / 0**",
        (
            f"- Strict donor cutouts/groups: **{len(units)} / "
            f"{payload['strict_donor_groups']}**"
        ),
        (
            "- Used donor cutouts/groups: "
            f"**{payload['used_donor_cutouts']} / "
            f"{payload['used_donor_groups']}**"
        ),
        f"- Distinct Train backgrounds: **{payload['distinct_backgrounds']}**",
        (
            "- Scene CLIP cosine threshold/observed minimum: "
            f"**{payload['scene_matching']['min_full_image_cosine_similarity']}"
            " / "
            f"{payload['scene_matching']['observed_min_cosine_similarity']:.4f}**"
        ),
        (
            "- Geometry fingerprint: "
            f"`{payload['geometry_fingerprint_sha256']}`"
        ),
        f"- Contact sheet: `{sheet_path}`",
        "",
        (
            "Each cell shows the full CPU draft on the left and an enlarged "
            "person crop on the right. Cyan is the coupled person support; "
            "yellow is the linked helmet/head label transported with that "
            "same person."
        ),
        "",
        (
            "This candidate cannot call FLUX. Approval requires zero floating, "
            "composited, implausible, truncated, or misplaced people and zero "
            "misplaced yellow labels. Approval would freeze the architecture "
            "but still would not pass the later GPU identity or H4 gates."
        ),
        "",
    ]
    (
        paths.reports / f"{artifact_stem}.md"
    ).write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
