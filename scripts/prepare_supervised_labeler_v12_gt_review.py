"""Freeze and render the semantics-correct v12 GT-only review pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

from scripts.train_supervised_labeler import HelmetDataset
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "supervised_labeler_v12_gt_review.yaml"
)
POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v12_gt_pool.json"
)
EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v12_gt_review.json"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "supervised_labeler_v12_gt_review"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(seed: int, namespace: str, value: int) -> str:
    material = f"{seed}|{namespace}|{value}".encode()
    return hashlib.sha256(material).hexdigest()


def _read_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config["source_split"] != "Train"
        or config["protocol"]["label_semantics"]
        != "class_direct_helmeted_head_region"
        or config["review_stages"]["gt_only"]["model_boxes_allowed"] is not False
        or config["review_stages"]["model_review"][
            "allowed_before_gt_freeze"
        ]
        is not False
        or config["generation_gate"]["allowed"] is not False
        or int(config["history_boundary"]["validation_images_read"]) != 0
        or int(config["history_boundary"]["test_images_read"]) != 0
    ):
        raise RuntimeError("v12 GT-only safety registration changed")
    return config


def _prior_revealed_groups(
    config: Mapping[str, Any],
) -> tuple[set[int], list[dict[str, Any]]]:
    groups: set[int] = set()
    records = []
    for experiment_id in config["history_boundary"][
        "exclude_revealed_experiments"
    ]:
        split_path = PROJECT_ROOT / "splits" / f"{experiment_id}_split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        experiment_groups = {
            int(value)
            for key in (
                "calibration_group_ids",
                "untouched_audit_group_ids",
                "quarantined_gt_defect_group_ids",
            )
            for value in split.get(key, [])
        }
        groups.update(experiment_groups)
        records.append(
            {
                "experiment_id": str(experiment_id),
                "path": str(split_path.relative_to(PROJECT_ROOT)).replace(
                    "\\",
                    "/",
                ),
                "file_sha256": _sha256(split_path),
                "revealed_or_quarantined_groups": len(experiment_groups),
            }
        )
    return groups, records


def _split_quartiles(
    rows: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["median_relative_helmet_area"]),
            int(row["group_id"]),
        ),
    )
    quotient, remainder = divmod(len(ordered), 4)
    quartiles = []
    offset = 0
    for index in range(4):
        size = quotient + (1 if index < remainder else 0)
        quartiles.append(ordered[offset : offset + size])
        offset += size
    if offset != len(ordered) or any(not rows for rows in quartiles):
        raise RuntimeError("Cannot form four positive-area quartiles")
    return quartiles


def _candidate_record(
    *,
    image_id: int,
    stratum: str,
    train_images: Mapping[int, Mapping[str, Any]],
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    frozen: Mapping[int, Mapping[str, Any]],
    helmet_category_id: int,
) -> dict[str, Any]:
    image = train_images[image_id]
    helmet_annotations = [
        row
        for row in annotations[image_id]
        if int(row["category_id"]) == helmet_category_id
    ]
    image_area = max(int(image["width"]) * int(image["height"]), 1)
    relative_areas = [
        float(row["bbox"][2]) * float(row["bbox"][3]) / image_area
        for row in helmet_annotations
    ]
    return {
        "image_id": int(image_id),
        "group_id": int(frozen[image_id]["group_id"]),
        "file_name": str(image["file_name"]),
        "source_image_sha256": str(frozen[image_id]["sha256"]),
        "width": int(image["width"]),
        "height": int(image["height"]),
        "stratum": str(stratum),
        "source_helmet_gt_count": len(helmet_annotations),
        "median_relative_helmet_area": (
            float(statistics.median(relative_areas))
            if relative_areas
            else None
        ),
    }


def freeze_pool() -> dict[str, Any]:
    """Freeze 64 primary and 32 sealed reserve Train-only cases."""

    config = _read_config()
    seed = int(config["split_seed"])
    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    helmet_category_id = next(
        int(row["id"])
        for row in coco["categories"]
        if str(row["name"]) == "helmet"
    )
    excluded_groups, history = _prior_revealed_groups(config)
    train_ids_by_group: dict[int, list[int]] = defaultdict(list)
    for image_id in sorted(train_images):
        train_ids_by_group[int(frozen[image_id]["group_id"])].append(image_id)

    positive_candidates = []
    empty_candidates = []
    for group_id, image_ids in sorted(train_ids_by_group.items()):
        if group_id in excluded_groups:
            continue
        positive_ids = [
            image_id
            for image_id in image_ids
            if any(
                int(row["category_id"]) == helmet_category_id
                for row in annotations[image_id]
            )
        ]
        if positive_ids:
            image_id = min(
                positive_ids,
                key=lambda value: _rank(
                    seed,
                    "positive_representative",
                    value,
                ),
            )
            positive_candidates.append(
                _candidate_record(
                    image_id=image_id,
                    stratum="pending_positive_quartile",
                    train_images=train_images,
                    annotations=annotations,
                    frozen=frozen,
                    helmet_category_id=helmet_category_id,
                )
            )
        elif all(
            not any(
                int(row["category_id"]) == helmet_category_id
                for row in annotations[image_id]
            )
            for image_id in image_ids
        ):
            image_id = min(
                image_ids,
                key=lambda value: _rank(
                    seed,
                    "empty_representative",
                    value,
                ),
            )
            empty_candidates.append(
                _candidate_record(
                    image_id=image_id,
                    stratum="dataset_gt_empty",
                    train_images=train_images,
                    annotations=annotations,
                    frozen=frozen,
                    helmet_category_id=helmet_category_id,
                )
            )

    primary_quotas = {
        str(key): int(value)
        for key, value in config["candidate_pool"]["primary_quotas"].items()
    }
    reserve_quotas = {
        str(key): int(value)
        for key, value in config["candidate_pool"][
            "sealed_reserve_quotas"
        ].items()
    }
    primary_by_stratum: dict[str, list[dict[str, Any]]] = {}
    reserve_by_stratum: dict[str, list[dict[str, Any]]] = {}
    eligible_counts = {}
    for quartile_index, quartile in enumerate(
        _split_quartiles(positive_candidates),
        start=1,
    ):
        stratum = f"positive_area_q{quartile_index}"
        ranked = sorted(
            quartile,
            key=lambda row: _rank(
                seed + quartile_index,
                stratum,
                int(row["group_id"]),
            ),
        )
        needed = primary_quotas[stratum] + reserve_quotas[stratum]
        if len(ranked) < needed:
            raise RuntimeError(f"Not enough candidates for {stratum}")
        chosen = [dict(row, stratum=stratum) for row in ranked[:needed]]
        primary_by_stratum[stratum] = chosen[: primary_quotas[stratum]]
        reserve_by_stratum[stratum] = chosen[primary_quotas[stratum] :]
        eligible_counts[stratum] = len(ranked)

    empty_ranked = sorted(
        empty_candidates,
        key=lambda row: _rank(
            seed,
            "dataset_gt_empty",
            int(row["group_id"]),
        ),
    )
    empty_needed = (
        primary_quotas["dataset_gt_empty"]
        + reserve_quotas["dataset_gt_empty"]
    )
    if len(empty_ranked) < empty_needed:
        raise RuntimeError("Not enough dataset-GT-empty groups")
    primary_by_stratum["dataset_gt_empty"] = empty_ranked[
        : primary_quotas["dataset_gt_empty"]
    ]
    reserve_by_stratum["dataset_gt_empty"] = empty_ranked[
        primary_quotas["dataset_gt_empty"] : empty_needed
    ]
    eligible_counts["dataset_gt_empty"] = len(empty_ranked)

    primary = [
        row for rows in primary_by_stratum.values() for row in rows
    ]
    reserve = [
        row for rows in reserve_by_stratum.values() for row in rows
    ]
    primary.sort(
        key=lambda row: _rank(seed, "primary_sheet_order", row["image_id"])
    )
    reserve.sort(
        key=lambda row: _rank(seed, "sealed_reserve_order", row["image_id"])
    )
    for cell, row in enumerate(primary, start=1):
        row["cell"] = cell
    for reserve_index, row in enumerate(reserve, start=1):
        row["reserve_index"] = reserve_index

    primary_ids = {int(row["image_id"]) for row in primary}
    reserve_ids = {int(row["image_id"]) for row in reserve}
    selected_groups = {
        int(row["group_id"]) for row in [*primary, *reserve]
    }
    if (
        len(primary) != int(config["candidate_pool"]["primary_images"])
        or len(reserve)
        != int(config["candidate_pool"]["sealed_reserve_images"])
        or primary_ids & reserve_ids
        or selected_groups & excluded_groups
        or test_ids & (primary_ids | reserve_ids)
    ):
        raise RuntimeError("v12 GT-only pool violates a frozen boundary")
    if len(selected_groups) != len(primary) + len(reserve):
        raise RuntimeError("v12 GT-only pool repeats a source group")

    split_manifest_sha256 = (
        PROJECT_ROOT / "splits" / "MANIFEST.sha256"
    ).read_text(encoding="utf-8").split()[0]
    payload = {
        "schema_version": 1,
        "status": "v12_gt_only_pool_frozen_before_pixel_review",
        "experiment_id": str(config["experiment_id"]),
        "split_seed": seed,
        "source_split": "Train",
        "source_split_manifest_sha256": split_manifest_sha256,
        "protocol_path": str(config["protocol"]["path"]),
        "label_semantics": str(config["protocol"]["label_semantics"]),
        "history": history,
        "excluded_revealed_group_count": len(excluded_groups),
        "eligible_group_counts": eligible_counts,
        "primary_cases": primary,
        "sealed_reserve_cases": reserve,
        "primary_images": len(primary),
        "sealed_reserve_images": len(reserve),
        "primary_pixels_read": 0,
        "sealed_reserve_pixels_read": 0,
        "model_inference_run": False,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    payload["manifest_sha256"] = canonical_mapping_sha256(payload)
    POOL_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    windows_root = os.environ.get("WINDIR")
    if windows_root:
        name = "msjhbd.ttc" if bold else "msjh.ttc"
        path = Path(windows_root) / "Fonts" / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_truth(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    *,
    source_size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    scale_x = image.width / source_size[0]
    scale_y = image.height / source_size[1]
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box)
        draw.rectangle(
            (
                round(x1 * scale_x),
                round(y1 * scale_y),
                round(x2 * scale_x),
                round(y2 * scale_y),
            ),
            outline=(0, 230, 0),
            width=2,
        )


def _render_pages(
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    panel_size = 480
    columns = 2
    rows = 8
    header_height = 190
    caption_height = 40
    page_width = panel_size * columns
    page_height = header_height + rows * (panel_size + caption_height)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for page_index in range(4):
        page = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(page)
        draw.text(
            (16, 10),
            "V12 第 1 階段：只檢查資料綠框（目前沒有模型框）",
            fill="black",
            font=_font(28, bold=True),
        )
        draw.text(
            (16, 52),
            "正確目標：戴著安全帽的頭部；一人一框，可包含安全帽＋頭／部分臉。",
            fill="black",
            font=_font(20),
        )
        draw.text(
            (16, 84),
            "孤立、放在地上／桌上、未戴在人頭上的安全帽：不要框。",
            fill="black",
            font=_font(20, bold=True),
        )
        draw.text(
            (16, 116),
            "請只回報：GT 多框、GT 漏框、GT 框歪、不確定；其他就是 PASS。",
            fill="black",
            font=_font(20),
        )
        draw.text(
            (16, 151),
            f"第 {page_index + 1}/4 頁｜格號 "
            f"{page_index * 16 + 1:02d}–{page_index * 16 + 16:02d}",
            fill=(70, 70, 70),
            font=_font(18),
        )
        for local_index in range(16):
            case = cases[page_index * 16 + local_index]
            column = local_index % columns
            row = local_index // columns
            x0 = column * panel_size
            y0 = header_height + row * (panel_size + caption_height)
            page.paste(case["panel"], (x0, y0))
            draw.text(
                (x0 + 8, y0 + panel_size + 6),
                f"{case['cell']:02d}｜Train image {case['image_id']}｜"
                f"綠框 {len(case['truth_boxes'])} 個",
                fill="black",
                font=_font(17),
            )
        output = FIGURE_DIR / f"page_{page_index + 1:02d}.png"
        page.save(output, optimize=True)
        outputs.append(
            {
                "path": str(output.relative_to(PROJECT_ROOT)).replace(
                    "\\",
                    "/",
                ),
                "sha256": _sha256(output),
                "cells": [
                    page_index * 16 + 1,
                    page_index * 16 + 16,
                ],
            }
        )
    return outputs


def render_primary() -> dict[str, Any]:
    """Read and render only the 64 preregistered primary images."""

    config = _read_config()
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    canonical = dict(pool)
    embedded_sha = canonical.pop("manifest_sha256")
    if (
        canonical_mapping_sha256(canonical) != embedded_sha
        or pool["status"] != "v12_gt_only_pool_frozen_before_pixel_review"
        or int(pool["primary_pixels_read"]) != 0
        or int(pool["sealed_reserve_pixels_read"]) != 0
        or pool["model_inference_run"] is not False
        or int(pool["validation_images_read"]) != 0
        or int(pool["test_images_read"]) != 0
    ):
        raise RuntimeError("Frozen v12 GT-only pool changed before rendering")

    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    helmet_category_id = next(
        int(row["id"])
        for row in coco["categories"]
        if str(row["name"]) == "helmet"
    )
    primary_ids = [
        int(row["image_id"]) for row in pool["primary_cases"]
    ]
    reserve_ids = {
        int(row["image_id"]) for row in pool["sealed_reserve_cases"]
    }
    if test_ids & (set(primary_ids) | reserve_ids):
        raise RuntimeError("Val/Test leakage entered v12 GT-only review")
    dataset = HelmetDataset(
        image_ids=primary_ids,
        images=train_images,
        annotations=annotations,
        image_root=paths.hardhat_raw,
        helmet_category_id=helmet_category_id,
        input_normalization=config["input_normalization"],
    )

    rendered_cases = []
    evidence_cases = []
    normalized_images = 0
    for index, item in enumerate(dataset):
        registration = pool["primary_cases"][index]
        image_id = int(item["image_id"])
        if (
            image_id != int(registration["image_id"])
            or int(frozen[image_id]["group_id"])
            != int(registration["group_id"])
        ):
            raise RuntimeError("v12 primary review order changed")
        image = item["image"].convert("RGB")
        if item["input_normalization"]["applied"]:
            normalized_images += 1
        panel = image.resize(
            (480, 480),
            Image.Resampling.LANCZOS,
        )
        _draw_truth(
            panel,
            item["truth"],
            source_size=image.size,
        )
        rendered_cases.append(
            {
                "cell": index + 1,
                "image_id": image_id,
                "truth_boxes": item["truth"],
                "panel": panel,
            }
        )
        evidence_cases.append(
            {
                "cell": index + 1,
                "image_id": image_id,
                "group_id": int(registration["group_id"]),
                "stratum": str(registration["stratum"]),
                "truth_boxes": item["truth"],
                "input_normalization": item["input_normalization"],
            }
        )

    pages = _render_pages(rendered_cases)
    evidence = {
        "schema_version": 1,
        "status": "v12_gt_only_primary_review_rendered",
        "experiment_id": str(config["experiment_id"]),
        "pool_manifest_sha256": str(pool["manifest_sha256"]),
        "protocol_path": str(config["protocol"]["path"]),
        "label_semantics": str(config["protocol"]["label_semantics"]),
        "review_stage": "gt_only",
        "model_boxes_present": False,
        "model_inference_run": False,
        "cases": evidence_cases,
        "primary_images_read": len(evidence_cases),
        "primary_images_normalized": normalized_images,
        "sealed_reserve_images": len(reserve_ids),
        "sealed_reserve_pixels_read": 0,
        "pages": pages,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    evidence["evidence_sha256"] = canonical_mapping_sha256(evidence)
    EVIDENCE_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("freeze", "render", "verify"),
    )
    args = parser.parse_args()
    if args.action == "freeze":
        payload = freeze_pool()
    elif args.action == "render":
        payload = render_primary()
    else:
        pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
        canonical = dict(pool)
        embedded_sha = canonical.pop("manifest_sha256")
        payload = {
            "status": "v12_gt_only_pool_verified",
            "manifest_sha256": embedded_sha,
            "canonical_hash_matches": (
                canonical_mapping_sha256(canonical) == embedded_sha
            ),
            "validation_images_read": 0,
            "test_images_read": 0,
        }
        if not payload["canonical_hash_matches"]:
            raise RuntimeError("v12 GT-only pool manifest hash changed")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
