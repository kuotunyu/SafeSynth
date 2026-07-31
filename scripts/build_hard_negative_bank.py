"""Materialise the hard-negative cutout bank (M9 completion, COMP-20..COMP-24).

Hard negatives are distractors that look like helmets and must NOT fire the
detector. They carry no annotation by construction: the label space is
{helmet, head, person}, a yellow drum is none of those, and under COCO/VOC
semantics the absence of a box is a positive assertion that no listed class
occupies that region (ADR-004).

Material comes from two sources, and ADR-012 flipped which one dominates:

* procedural (primary) - safety-coloured domes/ellipses/cylinders/arcs whose
  shading modulates real Train background texture. These are rendered AT
  helmet-typical size and aspect, which is what makes them *hard*, and they
  cannot possibly be a real unlabelled helmet.
* mined (supplementary) - real yellow/orange objects cut from unannotated
  regions of Train images. Domain-real texture, but supply-limited: the
  COMP-21 guards deliberately exclude helmet-typical geometry, so most mined
  candidates are too small to be mistaken for a helmet and are dropped here.

Mining may only be used once kuotunyu has signed off the H6 contact sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.data.paths import ProjectPaths, load_project_paths
from src.synthetic.cutout_bank import appearance_statistics, soft_alpha
from src.synthetic.hard_negatives import (
    procedural_hard_negative,
    select_h6_images,
    validate_human_signoff,
)
from src.synthetic.sam2_runner import Sam2BoxSegmenter, xywh_to_xyxy
from src.synthetic.survey import train_context

BANK_DIRNAME = "hardneg"
MANIFEST_NAME = "hardneg_bank_manifest.jsonl"
# The exact H6 contact sheet kuotunyu approved at 0/64 real helmets.
EXPECTED_H6_GRID_SHA256 = "0e385d857067aa293c5e3d0dd43ad84b4141ff9bac5c8d4aefed187ee9c45739"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_rgba(path: Path, rgba: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)
    return _sha256_file(path)


def _qualifying_mined(
    candidates: list[dict[str, Any]], *, min_side_px: int
) -> list[dict[str, Any]]:
    """Drop candidates too small to be mistaken for a helmet (ADR-012)."""

    kept = [
        candidate
        for candidate in candidates
        if min(candidate["bbox"][2], candidate["bbox"][3]) >= min_side_px
    ]
    return sorted(kept, key=lambda candidate: candidate["candidate_id"])


def _build_mined(
    *,
    paths: ProjectPaths,
    compose_config: dict[str, Any],
    candidates: list[dict[str, Any]],
    bank_dir: Path,
    built_at: str,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    sam2_config = compose_config["sam2"]
    pass2 = sam2_config["pass2_bank"]
    segmenter = Sam2BoxSegmenter(
        model_id=str(sam2_config["model_id"]),
        dtype=str(sam2_config["dtype"]),
        morph_close_kernel=int(sam2_config["cleanup"]["morph_close_kernel"]),
        morph_close_iterations=int(sam2_config["cleanup"]["morph_close_iterations"]),
    )
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        image = Image.open(paths.hardhat_raw / candidate["file_name"]).convert("RGB")
        box_xyxy = xywh_to_xyxy(candidate["bbox"])
        prediction = segmenter.predict_crop(
            image,
            box_xyxy,
            context_pad_frac=float(pass2["context_pad_frac"]),
            min_crop_side_px=int(pass2["min_crop_side_px"]),
            target_size=512,
        )
        x1, y1, x2, y2 = (round(value) for value in box_xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        rgb = np.asarray(image)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        patch = rgb[y1:y2, x1:x2]
        mask = prediction.mask[y1:y2, x1:x2]
        if not mask.any():
            continue
        alpha = soft_alpha(mask, config=compose_config)
        rgba = np.dstack((patch, alpha))
        relative = f"{BANK_DIRNAME}/mined_{candidate['candidate_id']}.png"
        file_sha256 = _write_rgba(bank_dir.parent / relative, rgba)
        records.append(
            {
                "appearance": appearance_statistics(patch, mask),
                "annotated": False,
                "build": {"built_at_utc": built_at, "seed": int(compose_config["seed"])},
                "class_name": None,
                "cutout_id": f"hn_mined_{candidate['candidate_id']}",
                "file": relative,
                "file_sha256": file_sha256,
                "kind": "hard_negative",
                "max_uses": int(compose_config["compose"]["max_uses_per_cutout"]),
                "negative_source": "mined_hsv_yellow",
                "sam2": {
                    "effective_crop_size": 512,
                    "iou_score": float(prediction.iou_score),
                    "object_score_logit": float(prediction.object_score_logit),
                },
                "size_wh": [int(x2 - x1), int(y2 - y1)],
                "src_bbox_xywh": [float(value) for value in candidate["bbox"]],
                "src_group_id": int(candidate["group_id"]),
                "src_image_id": int(candidate["image_id"]),
                "src_split": str(candidate["src_split"]),
            }
        )
    return records


def _build_procedural(
    *,
    paths: ProjectPaths,
    compose_config: dict[str, Any],
    count: int,
    bank_dir: Path,
    built_at: str,
) -> list[dict[str, Any]]:
    _, images, _, _, frozen = train_context(paths)
    seed = int(compose_config["seed"])
    shapes = list(compose_config["hard_negatives"]["procedural"]["shapes"])
    # Draw source textures from Train only. select_h6_images is reused so the
    # sampling is the same deterministic, frozen-split-aware helper as H6.
    pool = select_h6_images(sorted(images), seed=seed + 7, n=min(count, len(images)))
    records: list[dict[str, Any]] = []
    for index in range(count):
        image_id = pool[index % len(pool)]
        image = np.asarray(
            Image.open(paths.hardhat_raw / images[image_id]["file_name"]).convert("RGB")
        )
        rng = np.random.default_rng(seed + 100_003 * index)
        # Helmet-typical geometry is what makes these hard: real helmet cutouts
        # have min_side p10=22 / median=34 / p90=74 in this dataset.
        side = int(rng.integers(24, 80))
        side = min(side, min(image.shape[0], image.shape[1]))
        left = int(rng.integers(0, image.shape[1] - side + 1))
        top = int(rng.integers(0, image.shape[0] - side + 1))
        texture = image[top : top + side, left : left + side]
        shape = shapes[index % len(shapes)]
        rgba = procedural_hard_negative(texture, shape=shape, seed=seed + index)
        mask = rgba[..., 3] >= 128
        if not mask.any():
            continue
        relative = f"{BANK_DIRNAME}/proc_{index:04d}_{shape}.png"
        file_sha256 = _write_rgba(bank_dir.parent / relative, rgba)
        records.append(
            {
                "appearance": appearance_statistics(rgba[..., :3], mask),
                "annotated": False,
                "build": {"built_at_utc": built_at, "seed": seed},
                "class_name": None,
                "cutout_id": f"hn_proc_{index:04d}",
                "file": relative,
                "file_sha256": file_sha256,
                "kind": "hard_negative",
                "max_uses": int(compose_config["compose"]["max_uses_per_cutout"]),
                "negative_source": "procedural",
                "procedural": {"shape": shape, "texture_side_px": side},
                "size_wh": [int(rgba.shape[1]), int(rgba.shape[0])],
                "src_group_id": int(frozen[image_id]["group_id"]),
                "src_image_id": int(image_id),
                "src_split": "train",
            }
        )
    return records


def _render_contact_sheet(
    *, paths: ProjectPaths, records: list[dict[str, Any]], bank_root: Path
) -> dict[str, Any]:
    """Composite on magenta so alpha leaks and halos are impossible to miss."""

    sample = records[:64]
    cell = 150
    sheet = Image.new("RGB", (8 * cell, 8 * cell), (24, 24, 24))
    for index, record in enumerate(sample):
        rgba = Image.open(bank_root / record["file"]).convert("RGBA")
        rgba.thumbnail((cell - 14, cell - 26), Image.Resampling.LANCZOS)
        backdrop = Image.new("RGBA", (cell, cell), (255, 0, 255, 255))
        backdrop.alpha_composite(rgba, (7, 7))
        draw = ImageDraw.Draw(backdrop)
        draw.rectangle((0, cell - 16, cell, cell), fill=(0, 0, 0, 200))
        label = "mined" if record["negative_source"] != "procedural" else "proc"
        draw.text(
            (3, cell - 14),
            f"{index + 1:02d} {label} {record['size_wh'][0]}x{record['size_wh'][1]}",
            fill="white",
        )
        sheet.paste(backdrop.convert("RGB"), ((index % 8) * cell, (index // 8) * cell))
    output = paths.figures / "hard_negative_bank_grid.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    # Repo-relative: absolute paths embed the local username (publish-repo gate 1).
    try:
        rendered = output.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        rendered = output.name
    return {"path": rendered, "sha256": _sha256_file(output), "cells": len(sample)}


def build(*, target_total: int) -> dict[str, Any]:
    paths = load_project_paths()
    project_root = Path(__file__).resolve().parents[1]
    compose_config = _load_yaml(project_root / "configs" / "compose.yaml")
    hard_negatives = compose_config["hard_negatives"]

    # Hard-fails unless kuotunyu approved the exact H6 grid that is on disk. Mining
    # is only safe because that review found 0/64 real helmets; if the sheet ever
    # changes, the approval no longer covers it.
    validate_human_signoff(
        signoff_path=project_root / "reports" / "hard_negative_signoff.json",
        expected_grid_sha256=EXPECTED_H6_GRID_SHA256,
    )

    bank_root = paths.cutouts
    bank_dir = bank_root / BANK_DIRNAME
    bank_dir.mkdir(parents=True, exist_ok=True)
    built_at = datetime.now(UTC).isoformat()

    candidates = [
        json.loads(line)
        for line in (bank_root / "hardneg_candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    min_side = int(hard_negatives["mined_min_side_px"])
    qualifying = _qualifying_mined(candidates, min_side_px=min_side)

    # ADR-012 set mined_fraction to 0. Visual review of the SAM2 cutouts from the
    # 16 qualifying candidates found no object-like material at all: elongated
    # background strips (42x158, 51x167), thin pipes, and one flat 94x90 texture
    # patch. HSV+contour mining finds COLOUR REGIONS, not objects, and the
    # COMP-21 guards independently exclude helmet-like geometry, so mining cannot
    # produce a helmet-lookalike in this dataset. The code path is kept so the
    # decision stays reproducible rather than merely asserted.
    mined: list[dict[str, Any]] = []
    if float(hard_negatives["mined_fraction"]) > 0:
        mined = _build_mined(
            paths=paths,
            compose_config=compose_config,
            candidates=qualifying,
            bank_dir=bank_dir,
            built_at=built_at,
        )
    # Mined material is supply-limited by the COMP-21 guards, so the configured
    # fraction is an upper bound rather than a quota. Procedural fills the rest
    # and the ACTUAL split is reported instead of the aspirational one.
    procedural_count = max(target_total - len(mined), 0)
    procedural = _build_procedural(
        paths=paths,
        compose_config=compose_config,
        count=procedural_count,
        bank_dir=bank_dir,
        built_at=built_at,
    )

    records = mined + procedural
    manifest_path = bank_root / MANIFEST_NAME
    manifest_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )
    sheet = _render_contact_sheet(paths=paths, records=records, bank_root=bank_root)

    sides = sorted(min(record["size_wh"]) for record in records)
    summary = {
        "built_at_utc": built_at,
        "contact_sheet": sheet,
        "h6_signoff_approved": True,
        "manifest": manifest_path.as_posix(),
        "mined_accepted": len(mined),
        "mined_candidates_total": len(candidates),
        "mined_min_side_px": min_side,
        "mined_qualifying": len(qualifying),
        "procedural_accepted": len(procedural),
        "size_min_side": {
            "max": sides[-1],
            "median": statistics.median(sides),
            "min": sides[0],
        },
        "total": len(records),
        "validation_test_images_read": 0,
    }
    report = project_root / "reports" / "hard_negative_bank.md"
    report.write_text(
        "\n".join(
            [
                "# Hard-negative cutout bank",
                "",
                f"- Total cutouts: **{summary['total']}**",
                f"- Procedural (primary, ADR-012): **{summary['procedural_accepted']}**",
                (
                    f"- Mined (supplementary): **{summary['mined_accepted']}** "
                    f"of {summary['mined_qualifying']} qualifying "
                    f"({summary['mined_candidates_total']} mined, "
                    f"min_side >= {min_side} px)"
                ),
                (
                    f"- Cutout min_side: min {sides[0]} / "
                    f"median {statistics.median(sides):.0f} / max {sides[-1]} px"
                ),
                "- Annotations emitted by these cutouts: **0** (correct by construction)",
                f"- Validation/Test images read: **{summary['validation_test_images_read']}**",
                f"- Contact sheet: `{Path(sheet['path']).name}` (magenta backdrop)",
                "",
                "Real helmet cutouts in this dataset have min_side p10=22 / median=34 /",
                "p90=74, so procedural material is rendered into the same range on",
                "purpose: a distractor only counts as *hard* if it could plausibly be",
                "mistaken for a helmet.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-total", type=int, default=240)
    args = parser.parse_args()
    print(json.dumps(build(target_total=args.target_total), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
