"""Freeze and render the independent v13 GT-only review pool."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from PIL import Image, ImageDraw

from scripts.prepare_supervised_labeler_v12_gt_review import (
    _candidate_record,
    _draw_truth,
    _font,
    _rank,
    _sha256,
    _split_quartiles,
)
from scripts.train_supervised_labeler import HelmetDataset
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "supervised_labeler_v13_gt_review.yaml"
)
POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v13_gt_pool.json"
)
EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_gt_review.json"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "supervised_labeler_v13_gt_review"
)
V12_POOL_PATH = (
    PROJECT_ROOT / "splits" / "supervised_labeler_v12_gt_pool.json"
)


def _read_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config["status"] != "preregistered_before_pool_freeze"
        or config["source_split"] != "Train"
        or config["protocol"]["label_semantics"]
        != "class_direct_helmeted_head_region"
        or config["review_stage"]["model_boxes_allowed"] is not False
        or config["independence_boundary"][
            "required_future_model_initialization"
        ]
        != "pinned_base_checkpoint_only"
        or config["generation_gate"]["allowed"] is not False
        or int(config["independence_boundary"]["validation_images_read"]) != 0
        or int(config["independence_boundary"]["test_images_read"]) != 0
    ):
        raise RuntimeError("v13 GT-only preregistration changed")
    return config


def _verified_v12_pool() -> dict[str, Any]:
    pool = json.loads(V12_POOL_PATH.read_text(encoding="utf-8"))
    canonical = dict(pool)
    embedded_sha = str(canonical.pop("manifest_sha256", ""))
    if (
        canonical_mapping_sha256(canonical) != embedded_sha
        or pool.get("status")
        != "v12_gt_only_pool_frozen_before_pixel_review"
    ):
        raise RuntimeError("Frozen v12 pool changed")
    return pool


def _excluded_groups(
    config: Mapping[str, Any],
) -> tuple[set[int], list[dict[str, Any]]]:
    excluded: set[int] = set()
    history = []
    for version in config["independence_boundary"][
        "prior_revealed_split_versions"
    ]:
        path = (
            PROJECT_ROOT
            / "splits"
            / f"supervised_labeler_v{int(version)}_split.json"
        )
        split = json.loads(path.read_text(encoding="utf-8"))
        groups = {
            int(value)
            for key in (
                "calibration_group_ids",
                "untouched_audit_group_ids",
                "quarantined_gt_defect_group_ids",
            )
            for value in split.get(key, [])
        }
        excluded.update(groups)
        history.append(
            {
                "source": f"supervised_labeler_v{int(version)}",
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "file_sha256": _sha256(path),
                "excluded_group_count": len(groups),
            }
        )
    v12 = _verified_v12_pool()
    v12_groups = {
        int(row["group_id"])
        for row in [
            *v12["primary_cases"],
            *v12["sealed_reserve_cases"],
        ]
    }
    excluded.update(v12_groups)
    history.append(
        {
            "source": "supervised_labeler_v12_primary_and_reserve",
            "path": str(V12_POOL_PATH.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "manifest_sha256": str(v12["manifest_sha256"]),
            "excluded_group_count": len(v12_groups),
        }
    )
    return excluded, history


def freeze_pool() -> dict[str, Any]:
    """Freeze 64 primary and 32 reserve cases before reading pixels."""

    config = _read_config()
    seed = int(config["split_seed"])
    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    helmet_category_id = next(
        int(row["id"])
        for row in coco["categories"]
        if str(row["name"]) == "helmet"
    )
    excluded_groups, history = _excluded_groups(config)
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
                    seed, "v13_positive_representative", value
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
        else:
            image_id = min(
                image_ids,
                key=lambda value: _rank(
                    seed, "v13_empty_representative", value
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

    strata: dict[str, list[dict[str, Any]]] = {
        "dataset_gt_empty": empty_candidates
    }
    for index, rows in enumerate(
        _split_quartiles(positive_candidates),
        start=1,
    ):
        name = f"positive_area_q{index}"
        for row in rows:
            row["stratum"] = name
        strata[name] = rows

    primary = []
    reserve = []
    eligible_counts = {}
    for stratum in sorted(strata):
        rows = sorted(
            strata[stratum],
            key=lambda row: _rank(
                seed, f"v13_stratum_{stratum}", int(row["group_id"])
            ),
        )
        primary_quota = int(
            config["candidate_pool"]["primary_quotas"][stratum]
        )
        reserve_quota = int(
            config["candidate_pool"]["sealed_reserve_quotas"][stratum]
        )
        if len(rows) < primary_quota + reserve_quota:
            raise RuntimeError(f"Insufficient fresh v13 cases: {stratum}")
        primary.extend(rows[:primary_quota])
        reserve.extend(
            rows[primary_quota : primary_quota + reserve_quota]
        )
        eligible_counts[stratum] = len(rows)
    primary.sort(
        key=lambda row: _rank(seed, "v13_primary_sheet", row["image_id"])
    )
    reserve.sort(
        key=lambda row: _rank(seed, "v13_sealed_reserve", row["image_id"])
    )
    for cell, row in enumerate(primary, start=1):
        row["cell"] = cell
    for index, row in enumerate(reserve, start=1):
        row["reserve_index"] = index

    all_cases = [*primary, *reserve]
    groups = {int(row["group_id"]) for row in all_cases}
    ids = {int(row["image_id"]) for row in all_cases}
    if (
        len(primary) != 64
        or len(reserve) != 32
        or len(groups) != 96
        or groups & excluded_groups
        or ids & test_ids
    ):
        raise RuntimeError("v13 pool violates its frozen boundary")
    payload = {
        "schema_version": 1,
        "status": "v13_gt_only_pool_frozen_before_pixel_review_or_training",
        "experiment_id": str(config["experiment_id"]),
        "split_seed": seed,
        "source_split": "Train",
        "protocol_path": str(config["protocol"]["path"]),
        "label_semantics": str(config["protocol"]["label_semantics"]),
        "history_exclusions": history,
        "excluded_group_count": len(excluded_groups),
        "eligible_group_counts": eligible_counts,
        "primary_cases": primary,
        "sealed_reserve_cases": reserve,
        "future_v13_training_exclusion_group_ids": sorted(groups),
        "primary_images": len(primary),
        "sealed_reserve_images": len(reserve),
        "primary_pixels_read": 0,
        "sealed_reserve_pixels_read": 0,
        "v13_training_started": False,
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


def _render_pages(
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    panel_size = 480
    columns = 2
    rows = 8
    header_height = 180
    caption_height = 40
    page_width = panel_size * columns
    page_height = header_height + rows * (panel_size + caption_height)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for page_index in range(4):
        page = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(page)
        first = page_index * 16 + 1
        last = first + 15
        draw.text(
            (16, 10),
            f"V13 GT-only 審核｜第 {page_index + 1}/4 頁｜"
            f"編號 {first:02d}–{last:02d}",
            fill="black",
            font=_font(27, bold=True),
        )
        draw.text(
            (16, 51),
            "綠框只有資料集 GT，完全沒有模型框。"
            "請判斷綠框是否符合戴安全帽頭部規則。",
            fill="black",
            font=_font(20),
        )
        draw.text(
            (16, 84),
            "散放／手拿的安全帽、Logo、背景、普通人臉不應有框；"
            "戴帽者應一人一框。",
            fill="black",
            font=_font(19),
        )
        draw.text(
            (16, 116),
            "邊界截斷或看不清楚請標 UNCERTAIN，整張隔離。"
            "不要猜，也不要只刪單一框。",
            fill="black",
            font=_font(19, bold=True),
        )
        draw.text(
            (16, 148),
            "此池在 v13 訓練前凍結；未使用 Val/Test，未執行模型。",
            fill=(70, 70, 70),
            font=_font(17),
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
        path = FIGURE_DIR / f"page_{page_index + 1:02d}.png"
        page.save(path, optimize=True)
        outputs.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
                "cells": [first, last],
            }
        )
    return outputs


def render_primary() -> dict[str, Any]:
    """Render only the 64 frozen v13 primary cases with GT boxes."""

    config = _read_config()
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    canonical = dict(pool)
    embedded_sha = str(canonical.pop("manifest_sha256", ""))
    if (
        canonical_mapping_sha256(canonical) != embedded_sha
        or pool.get("status")
        != "v13_gt_only_pool_frozen_before_pixel_review_or_training"
        or int(pool["primary_pixels_read"]) != 0
        or int(pool["sealed_reserve_pixels_read"]) != 0
        or pool["v13_training_started"] is not False
        or pool["model_inference_run"] is not False
    ):
        raise RuntimeError("Frozen v13 pool changed before rendering")
    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    helmet_category_id = next(
        int(row["id"])
        for row in coco["categories"]
        if str(row["name"]) == "helmet"
    )
    primary_ids = [int(row["image_id"]) for row in pool["primary_cases"]]
    reserve_ids = {
        int(row["image_id"]) for row in pool["sealed_reserve_cases"]
    }
    if test_ids & (set(primary_ids) | reserve_ids):
        raise RuntimeError("Val/Test leakage entered v13 review")
    dataset = HelmetDataset(
        image_ids=primary_ids,
        images=train_images,
        annotations=annotations,
        image_root=paths.hardhat_raw,
        helmet_category_id=helmet_category_id,
        input_normalization=config["input_normalization"],
    )
    rendered = []
    evidence_cases = []
    normalized = 0
    for index, item in enumerate(dataset):
        registration = pool["primary_cases"][index]
        image_id = int(item["image_id"])
        if (
            image_id != int(registration["image_id"])
            or int(frozen[image_id]["group_id"])
            != int(registration["group_id"])
        ):
            raise RuntimeError("v13 primary review order changed")
        image = item["image"].convert("RGB")
        normalized += int(item["input_normalization"]["applied"])
        panel = image.resize((480, 480), Image.Resampling.LANCZOS)
        _draw_truth(panel, item["truth"], source_size=image.size)
        rendered.append(
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
    evidence = {
        "schema_version": 1,
        "status": "v13_gt_only_primary_review_rendered_before_training",
        "experiment_id": str(config["experiment_id"]),
        "pool_manifest_sha256": str(pool["manifest_sha256"]),
        "label_semantics": str(config["protocol"]["label_semantics"]),
        "review_stage": "gt_only",
        "model_boxes_present": False,
        "model_inference_run": False,
        "v13_training_started": False,
        "cases": evidence_cases,
        "primary_images_read": len(evidence_cases),
        "primary_images_normalized": normalized,
        "sealed_reserve_images": len(reserve_ids),
        "sealed_reserve_pixels_read": 0,
        "pages": _render_pages(rendered),
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
    parser.add_argument("action", choices=("freeze", "render", "verify"))
    args = parser.parse_args()
    if args.action == "freeze":
        if POOL_PATH.exists():
            raise RuntimeError(f"v13 pool already exists: {POOL_PATH}")
        payload = freeze_pool()
    elif args.action == "render":
        if EVIDENCE_PATH.exists() or FIGURE_DIR.exists():
            raise RuntimeError("v13 GT review evidence already exists")
        payload = render_primary()
    else:
        payload = json.loads(POOL_PATH.read_text(encoding="utf-8"))
        canonical = dict(payload)
        embedded_sha = str(canonical.pop("manifest_sha256", ""))
        if canonical_mapping_sha256(canonical) != embedded_sha:
            raise RuntimeError("v13 pool hash changed")
        payload = {
            "status": "v13_gt_pool_verified",
            "manifest_sha256": embedded_sha,
            "validation_images_read": 0,
            "test_images_read": 0,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
