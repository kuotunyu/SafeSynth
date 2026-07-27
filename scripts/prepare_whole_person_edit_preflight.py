"""Render the exact four CPU-only inputs for the v8 FLUX diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import (
    _archive_existing,
    _decode_rle,
    _load_configs,
    _load_context,
    _load_pass1,
    normalize_reflected_padding,
    reflected_padding_guard,
)
from src.synthetic.generative_inpaint import reference_canvas
from src.synthetic.region_inpaint import whole_person_edit_mask

CONFIG_PATH = PROJECT_ROOT / "configs" / "whole_person_edit_diagnostic.yaml"
AUDIT_PATH = PROJECT_ROOT / "reports" / "whole_person_edit_candidate_audit.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "whole_person_edit_preflight_seed20260804"
FIGURE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "whole_person_edit_preflight_v8.png"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "whole_person_edit_preflight_v8.json"


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


def _select_targets(
    candidates: Sequence[Mapping[str, Any]],
    *,
    height_bands: Sequence[Sequence[float | None]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_groups: set[int] = set()
    for lower, upper in height_bands:
        lower_bound = float(lower) if lower is not None else float("-inf")
        upper_bound = float(upper) if upper is not None else float("inf")
        eligible = [
            dict(candidate)
            for candidate in candidates
            if (
                lower_bound
                <= float(candidate["person_height_px"])
                < upper_bound
                and int(candidate["group_id"]) not in used_groups
            )
        ]
        eligible.sort(
            key=lambda item: (
                -int(item["edit_edge_margin_px"]),
                -float(item["person_height_px"]),
                -int(item["person_mask_area_px"]),
                str(item["cutout_id"]),
            )
        )
        if not eligible:
            raise RuntimeError(
                f"No v8 target candidate remains in band [{lower}, {upper})"
            )
        selected.append(eligible[0])
        used_groups.add(int(eligible[0]["group_id"]))
    return selected


def _select_references(
    candidates: Sequence[Mapping[str, Any]],
    *,
    bank_by_cutout: Mapping[str, Mapping[str, Any]],
    target_groups: set[int],
    count: int,
    min_person_height_px: float,
) -> list[dict[str, Any]]:
    eligible = [
        dict(candidate)
        for candidate in candidates
        if (
            int(candidate["group_id"]) not in target_groups
            and float(candidate["person_height_px"]) >= min_person_height_px
        )
    ]
    eligible.sort(
        key=lambda item: (
            -float(bank_by_cutout[str(item["cutout_id"])]["sam2"]["iou_score"]),
            -int(item["person_mask_area_px"]),
            -int(item["edit_edge_margin_px"]),
            str(item["cutout_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    used_groups: set[int] = set()
    for candidate in eligible:
        group_id = int(candidate["group_id"])
        if group_id in used_groups:
            continue
        selected.append(candidate)
        used_groups.add(group_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("Not enough distinct Train reference groups for v8")
    return selected


def _materialize_target(
    target: Mapping[str, Any],
    *,
    paths: Any,
    train_images: Mapping[int, Mapping[str, Any]],
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    reflection_config: Mapping[str, Any],
    outer_dilate_px: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    image_id = int(target["image_id"])
    raw = np.asarray(
        Image.open(
            paths.hardhat_raw / str(train_images[image_id]["file_name"])
        ).convert("RGB")
    )
    reflection = reflected_padding_guard(
        raw,
        guard_config=reflection_config,
    )
    normalized, _, transformed_pass1, normalization = (
        normalize_reflected_padding(
            raw,
            annotations=annotations[image_id],
            pass1=_load_pass1(paths, image_id),
            reflection=reflection,
            output_shape=raw.shape[:2],
            transform_masks=True,
        )
    )
    person_mask = _decode_rle(
        transformed_pass1[int(target["person_annotation_id"])]["segmentation"]
    )
    headlike_mask = _decode_rle(
        transformed_pass1[int(target["headlike_annotation_id"])]["segmentation"]
    )
    edit_mask = whole_person_edit_mask(
        person_mask,
        headlike_mask,
        outer_dilate_px=outer_dilate_px,
    )
    return (
        normalized,
        edit_mask,
        {
            "applied": normalization.applied,
            "crop_xyxy": list(normalization.crop_xyxy),
            "resized_width": normalization.resized_width,
            "resized_height": normalization.resized_height,
            "offset_x": normalization.offset_x,
            "offset_y": normalization.offset_y,
            "detected_sides": list(normalization.detected_sides),
        },
    )


def _square_crop(
    image: Image.Image,
    mask: np.ndarray,
    *,
    padding_fraction: float = 0.12,
) -> Image.Image:
    rows, columns = np.nonzero(mask)
    x1, x2 = int(columns.min()), int(columns.max()) + 1
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    extent = max(x2 - x1, y2 - y1)
    padding = max(8, round(extent * padding_fraction))
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    half = (extent + 2 * padding) // 2
    left = max(0, center_x - half)
    top = max(0, center_y - half)
    right = min(image.width, center_x + half)
    bottom = min(image.height, center_y + half)
    return image.crop((left, top, right, bottom))


def _mask_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    image = np.asarray(image_rgb, dtype=np.uint8)
    overlay = image.copy()
    overlay[mask] = (
        0.58 * image[mask].astype(np.float32)
        + 0.42 * np.asarray([0, 235, 235], dtype=np.float32)
    ).astype(np.uint8)
    contour_source = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        contour_source,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return Image.fromarray(overlay)


def _render_sheet(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
) -> None:
    panel = 240
    header = 52
    labels = (
        "ORIGINAL (model input)",
        "CYAN = only editable pixels",
        "ZOOM of editable worker",
        "BINARY EDIT MASK",
        "REFERENCE worker",
    )
    sheet = Image.new(
        "RGB",
        (panel * len(labels), (panel + header) * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        draft = Image.open(row["draft_path"]).convert("RGB")
        mask = np.asarray(Image.open(row["mask_path"]).convert("L")) > 0
        overlay = _mask_overlay(np.asarray(draft), mask)
        zoom = _square_crop(overlay, mask)
        binary = Image.fromarray(mask.astype(np.uint8) * 255).convert("RGB")
        reference = Image.open(row["reference_path"]).convert("RGB")
        panels = (draft, overlay, zoom, binary, reference)
        y0 = row_index * (panel + header)
        for column, (label, source) in enumerate(
            zip(labels, panels, strict=True)
        ):
            resized = source.resize((panel, panel), Image.Resampling.LANCZOS)
            sheet.paste(resized, (column * panel, y0 + header))
            draw.text((column * panel + 5, y0 + 30), label, fill="black")
        draw.text(
            (5, y0 + 5),
            (
                f"{row_index + 1:02d} | target {row['target_cutout_id']} | "
                f"reference {row['reference_cutout_id']}"
            ),
            fill="black",
        )
        draw.line(
            (0, y0 + panel + header - 1, sheet.width, y0 + panel + header - 1),
            fill=(80, 80, 80),
            width=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)


def main() -> None:
    paths = load_project_paths()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    compose_config, _ = _load_configs()
    reflection_config = compose_config["compose"]["context_replacement"][
        "input_guard"
    ]["reflected_padding"]
    _, bank, train_images, annotations, frozen, test_ids = _load_context(paths)
    bank_by_cutout = {
        str(item["cutout_id"]): item
        for item in bank
        if str(item["class_name"]) == "person"
    }
    candidates = audit["candidates"]
    targets = _select_targets(
        candidates,
        height_bands=config["selection"]["target_height_bands_px"],
    )
    target_groups = {int(target["group_id"]) for target in targets}
    references = _select_references(
        candidates,
        bank_by_cutout=bank_by_cutout,
        target_groups=target_groups,
        count=len(targets),
        min_person_height_px=float(config["reference"]["min_person_height_px"]),
    )
    if target_groups & {int(reference["group_id"]) for reference in references}:
        raise AssertionError("Target/reference frozen groups overlap")
    if len({int(reference["group_id"]) for reference in references}) != len(
        references
    ):
        raise AssertionError("Reference frozen groups are not distinct")

    _archive_existing(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    case_records: list[dict[str, Any]] = []
    sheet_rows: list[dict[str, Any]] = []
    for case_index, (target, reference) in enumerate(
        zip(targets, references, strict=True),
        start=1,
    ):
        draft, edit_mask, normalization = _materialize_target(
            target,
            paths=paths,
            train_images=train_images,
            annotations=annotations,
            reflection_config=reflection_config,
            outer_dilate_px=int(config["selection"]["outer_dilate_px"]),
        )
        reference_item = bank_by_cutout[str(reference["cutout_id"])]
        reference_rgba = np.asarray(
            Image.open(paths.cutouts / str(reference_item["file"])).convert("RGBA")
        )
        reference_image = reference_canvas(
            reference_rgba,
            canvas_size=int(config["reference"]["canvas_size"]),
            max_fill=float(config["reference"]["max_fill"]),
            background_rgb=tuple(
                int(value) for value in config["reference"]["background_rgb"]
            ),
        )
        case_dir = OUTPUT_ROOT / f"case_{case_index:02d}"
        case_dir.mkdir()
        draft_path = case_dir / "draft.png"
        mask_path = case_dir / "edit_mask.png"
        reference_path = case_dir / "reference.png"
        Image.fromarray(draft).save(draft_path, optimize=True)
        Image.fromarray(edit_mask.astype(np.uint8) * 255).save(
            mask_path,
            optimize=True,
        )
        reference_image.save(reference_path, optimize=True)
        record = {
            "case_index": case_index,
            "target": dict(target),
            "reference": {
                **dict(reference),
                "sam2_iou_score": float(reference_item["sam2"]["iou_score"]),
            },
            "normalization": normalization,
            "files": {
                "draft": str(draft_path.relative_to(OUTPUT_ROOT)),
                "edit_mask": str(mask_path.relative_to(OUTPUT_ROOT)),
                "reference": str(reference_path.relative_to(OUTPUT_ROOT)),
            },
            "sha256": {
                "draft": _sha256(draft_path),
                "edit_mask": _sha256(mask_path),
                "reference": _sha256(reference_path),
            },
            "model_inference_run": False,
        }
        _write_json(case_dir / "record.json", record)
        case_records.append(record)
        sheet_rows.append(
            {
                "draft_path": draft_path,
                "mask_path": mask_path,
                "reference_path": reference_path,
                "target_cutout_id": target["cutout_id"],
                "reference_cutout_id": reference["cutout_id"],
            }
        )

    frozen_payload = {
        "architecture": config["architecture"],
        "root_seed": int(config["root_seed"]),
        "method": config["method"],
        "cases": case_records,
    }
    input_manifest_sha256 = _canonical_sha256(frozen_payload)
    manifest = {
        "schema_version": 1,
        "status": "pending_kuotunyu_input_review",
        "diagnostic_only": True,
        "architecture": config["architecture"],
        "root_seed": int(config["root_seed"]),
        "n_cases": len(case_records),
        "scope": "frozen Train pixels, labels, masks, and cutouts only",
        "validation_images_read": 0,
        "test_images_read": 0,
        "model_inference_run": False,
        "h4_auc_computed": False,
        "input_manifest_sha256": input_manifest_sha256,
        "cases": case_records,
        "model_gate": config["model_gate"],
    }
    if test_ids & {
        int(record["target"]["image_id"]) for record in case_records
    }:
        raise AssertionError("Test target entered the v8 preflight")
    for record in case_records:
        if frozen[int(record["target"]["image_id"])]["split"] != "train":
            raise AssertionError("Non-Train target entered the v8 preflight")
        if frozen[int(record["reference"]["image_id"])]["split"] != "train":
            raise AssertionError("Non-Train reference entered the v8 preflight")
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    _write_json(REPORT_PATH, manifest)
    _render_sheet(sheet_rows, output_path=FIGURE_PATH)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
