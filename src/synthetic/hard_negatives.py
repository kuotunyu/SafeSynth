"""M9/H6 guarded hard-negative mining and texture-modulated procedural distractors."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.paths import ProjectPaths
from src.synthetic.survey import load_json, train_context


def _canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_key(value: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _xywh_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(right - left, 0) * max(bottom - top, 0)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _has_head_or_person_below(
    bbox: Sequence[float],
    annotations: Sequence[Mapping[str, Any]],
    category_names: Mapping[int, str],
) -> bool:
    """Conservative annotation-based 'worn helmet' rejection test."""

    x, y, width, height = (float(value) for value in bbox)
    center_x = x + width / 2
    bottom = y + height
    for annotation in annotations:
        class_name = category_names[int(annotation["category_id"])]
        if class_name not in {"head", "person"}:
            continue
        ox, oy, ow, oh = (float(value) for value in annotation["bbox"])
        horizontal_match = ox - 0.25 * ow <= center_x <= ox + 1.25 * ow
        if class_name == "head":
            vertical_match = bottom - 0.25 * height <= oy <= bottom + 2.5 * height
        else:
            vertical_match = oy - 0.25 * height <= bottom <= oy + 0.50 * oh
        if horizontal_match and vertical_match:
            return True
    return False


def _skin_like_below(image_hsv: np.ndarray, bbox: Sequence[float]) -> bool:
    """Reject candidates with a substantial skin-like patch immediately below."""

    x, y, width, height = (float(value) for value in bbox)
    image_height, image_width = image_hsv.shape[:2]
    left = max(0, int(np.floor(x - 0.25 * width)))
    right = min(image_width, int(np.ceil(x + 1.25 * width)))
    # Start below the candidate; including its lower 30% would classify the
    # safety-yellow/orange pixels themselves as "skin" and reject everything.
    top = max(0, int(np.floor(y + height)))
    bottom = min(image_height, int(np.ceil(y + 3.0 * height)))
    if right <= left or bottom <= top:
        return False
    region = image_hsv[top:bottom, left:right]
    # Broad skin window: red/orange hue, enough saturation, not extremely dark.
    hue = region[..., 0]
    saturation = region[..., 1]
    value = region[..., 2]
    skin = ((hue <= 25) | (hue >= 170)) & (saturation >= 35) & (value >= 45)
    return float(skin.mean()) >= 0.18


def _outside_helmet_typical_range(
    bbox: Sequence[float], helmet_geometry: Mapping[str, Any]
) -> bool:
    _, _, width, height = (float(value) for value in bbox)
    area = width * height
    aspect = width / height
    area_distribution = helmet_geometry["bbox_area_px"]
    aspect_distribution = helmet_geometry["aspect_ratio"]
    return not (
        float(area_distribution["p5"]) <= area <= float(area_distribution["p95"])
        and float(aspect_distribution["p5"])
        <= aspect
        <= float(aspect_distribution["p95"])
    )


def select_h6_images(image_ids: Sequence[int], *, seed: int, n: int = 200) -> list[int]:
    """Select a stable, order-independent H6 image sample."""

    return sorted(image_ids, key=lambda image_id: _stable_key(image_id, seed))[:n]


def mine_hard_negatives(
    *,
    paths: ProjectPaths,
    compose_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    n_images: int = 200,
) -> dict[str, Any]:
    """Mine yellow/orange distractors with all automatic COMP-21 guards."""

    _, images, annotations, category_names, frozen = train_context(paths)
    mining = compose_config["hard_negatives"]["mining"]
    selected_ids = select_h6_images(sorted(images), seed=int(compose_config["seed"]), n=n_images)
    helmet_geometry = calibration["geometry"]["per_class"]["helmet"]
    candidates: list[dict[str, Any]] = []
    pre_guard_count = 0
    guard_rejects: Counter[str] = Counter()

    hue_low, hue_high = (float(value) / 2 for value in mining["hue_deg"])
    saturation_low = round(float(mining["min_saturation"]) * 255)
    value_low = round(float(mining["min_value"]) * 255)
    contour_area_low, contour_area_high = (
        float(value) for value in mining["contour_area_px"]
    )
    circularity_low, circularity_high = (
        float(value) for value in mining["circularity"]
    )

    for image_id in selected_ids:
        image_record = images[image_id]
        image_path = paths.hardhat_raw / image_record["file_name"]
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        color_mask = cv2.inRange(
            hsv,
            np.array([hue_low, saturation_low, value_low], dtype=np.uint8),
            np.array([hue_high, 255, 255], dtype=np.uint8),
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour_index, contour in enumerate(contours):
            contour_area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (
                4 * math.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
            )
            if not contour_area_low <= contour_area <= contour_area_high:
                continue
            if not circularity_low <= circularity <= circularity_high:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            bbox = [float(x), float(y), float(width), float(height)]
            pre_guard_count += 1
            image_annotations = annotations.get(image_id, [])
            max_iou = max(
                (_xywh_iou(bbox, annotation["bbox"]) for annotation in image_annotations),
                default=0.0,
            )
            no_annotation_overlap = max_iou <= float(
                mining["max_iou_with_any_annotation"]
            )
            has_head_below = _has_head_or_person_below(
                bbox, image_annotations, category_names
            ) or _skin_like_below(hsv, bbox)
            outside_typical = _outside_helmet_typical_range(bbox, helmet_geometry)
            worn_test_failed = (
                (not has_head_below if mining["require_no_head_below"] else True)
                and (
                    outside_typical
                    if mining["require_outside_helmet_size_range"]
                    else True
                )
            )
            if not (no_annotation_overlap and worn_test_failed):
                if not no_annotation_overlap:
                    guard_rejects["annotation_overlap"] += 1
                elif has_head_below:
                    guard_rejects["head_like_region_below"] += 1
                elif not outside_typical:
                    guard_rejects["inside_helmet_typical_range"] += 1
                continue
            region = hsv[y : y + height, x : x + width]
            region_mask = np.zeros((height, width), dtype=np.uint8)
            shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(region_mask, [shifted], -1, 1, thickness=cv2.FILLED)
            saturation_mean = float(region[..., 1][region_mask.astype(bool)].mean()) / 255
            fill_ratio = contour_area / max(width * height, 1)
            score = 0.40 * circularity + 0.30 * saturation_mean + 0.30 * fill_ratio
            candidates.append(
                {
                    "candidate_id": f"{image_id:06d}_c{contour_index:03d}",
                    "image_id": image_id,
                    "file_name": image_record["file_name"],
                    "image_sha256": frozen[image_id]["sha256"],
                    "group_id": int(frozen[image_id]["group_id"]),
                    "src_split": frozen[image_id]["split"],
                    "bbox": bbox,
                    "contour_area_px": contour_area,
                    "circularity": circularity,
                    "saturation_mean": saturation_mean,
                    "fill_ratio": fill_ratio,
                    "score": score,
                    "automatic_guards": {
                        "max_iou_with_annotation": max_iou,
                        "no_annotation_overlap": no_annotation_overlap,
                        "has_head_like_region_below": has_head_below,
                        "outside_helmet_typical_range": outside_typical,
                        "worn_helmet_test_failed": worn_test_failed,
                    },
                }
            )

    candidates.sort(key=lambda record: (-float(record["score"]), record["candidate_id"]))
    output_path = paths.cutouts / "hardneg_candidates.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(_canonical_json(record) for record in candidates)
        + ("\n" if candidates else ""),
        encoding="utf-8",
        newline="\n",
    )
    grid_path = paths.figures / "h6_hard_negative_candidates.png"
    render_h6_grid(paths=paths, candidates=candidates[:64], output_path=grid_path)
    grid_sha256 = _sha256_file(grid_path)
    write_h6_report(
        paths=paths,
        n_images=len(selected_ids),
        pre_guard_count=pre_guard_count,
        candidates=candidates,
        grid_sha256=grid_sha256,
        compose_config=compose_config,
        guard_rejects=guard_rejects,
    )
    return {
        "sampled_images": len(selected_ids),
        "pre_guard_candidates": pre_guard_count,
        "guarded_candidates": len(candidates),
        "guard_rejects": dict(sorted(guard_rejects.items())),
        "grid_candidates": min(64, len(candidates)),
        "grid_sha256": grid_sha256,
        "human_signoff": "pending",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_h6_grid(
    *,
    paths: ProjectPaths,
    candidates: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Render the exact 8×8 sheet the user must inspect before bank freeze."""

    cell = 192
    sheet = Image.new("RGB", (8 * cell, 8 * cell), "black")
    font = ImageFont.load_default()
    for index, candidate in enumerate(candidates):
        image = Image.open(paths.hardhat_raw / candidate["file_name"]).convert("RGB")
        x, y, width, height = (float(value) for value in candidate["bbox"])
        pad = max(width, height) * 1.2
        bounds = (
            int(np.floor(x - pad)),
            int(np.floor(y - pad)),
            int(np.ceil(x + width + pad)),
            int(np.ceil(y + height + pad)),
        )
        crop = image.crop(bounds).resize((cell, cell), Image.Resampling.BICUBIC)
        scale_x = cell / max(bounds[2] - bounds[0], 1)
        scale_y = cell / max(bounds[3] - bounds[1], 1)
        draw = ImageDraw.Draw(crop)
        draw.rectangle(
            (
                (x - bounds[0]) * scale_x,
                (y - bounds[1]) * scale_y,
                (x + width - bounds[0]) * scale_x,
                (y + height - bounds[1]) * scale_y,
            ),
            outline=(0, 255, 255),
            width=3,
        )
        draw.rectangle((0, 0, cell, 24), fill=(0, 0, 0))
        draw.text(
            (3, 3),
            f"{index + 1:02d} {candidate['candidate_id']} s={candidate['score']:.2f}",
            fill="white",
            font=font,
        )
        sheet.paste(crop, ((index % 8) * cell, (index // 8) * cell))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)


def write_h6_report(
    *,
    paths: ProjectPaths,
    n_images: int,
    pre_guard_count: int,
    candidates: Sequence[Mapping[str, Any]],
    grid_sha256: str,
    compose_config: Mapping[str, Any],
    guard_rejects: Mapping[str, int],
) -> None:
    """Record H6 evidence while leaving user approval explicitly unresolved."""

    max_rate = float(
        compose_config["hard_negatives"]["mining"]["max_tolerated_helmet_rate"]
    )
    report = [
        "# M9 / Spike H6 — hard-negative mining purity",
        "",
        f"- Frozen Train images sampled: {n_images}",
        f"- HSV/shape candidates before semantic guards: {pre_guard_count}",
        f"- Candidates after IoU + worn-helmet guards: {len(candidates)}",
        f"- Guard rejections: `{dict(sorted(guard_rejects.items()))}`",
        f"- Contact-sheet cells: {min(64, len(candidates))}",
        f"- Maximum tolerated real-helmet rate: {max_rate:.0%}",
        f"- Contact-sheet SHA256: `{grid_sha256}`",
        "",
        "## Human gate",
        "",
        "**PENDING USER SIGNOFF.** Count cyan-boxed regions that are actual helmets in",
        "`reports/figures/h6_hard_negative_candidates.png`. The mined bank must not be",
        "frozen until the user supplies that count and explicitly approves it.",
        "",
        "If the real-helmet count exceeds the configured tolerance, the mined/procedural",
        "mix flips to procedural-primary and every retained mined item requires review.",
        "",
        "Automatic safeguards already applied:",
        "",
        "1. IoU with every existing annotation is below the fixed limit.",
        "2. The region has no annotation/skin-like head below and falls outside the",
        "   frozen Train helmet p5–p95 size/aspect envelope.",
        "3. The contact sheet and its SHA are immutable inputs to the pending signoff.",
        "",
    ]
    (paths.reports / "h6_hard_negative_spike.md").write_text(
        "\n".join(report), encoding="utf-8", newline="\n"
    )


def validate_human_signoff(
    *, signoff_path: Path, expected_grid_sha256: str, required_user: str = "kuotunyu"
) -> dict[str, Any]:
    """Hard fail unless the user explicitly approved the exact H6 grid."""

    if not signoff_path.exists():
        raise RuntimeError(f"Human signoff is required and missing: {signoff_path}")
    signoff = load_json(signoff_path)
    if signoff.get("approved_by") != required_user or signoff.get("approved") is not True:
        raise RuntimeError("Hard-negative signoff is not explicitly approved by kuotunyu")
    if signoff.get("grid_sha256") != expected_grid_sha256:
        raise RuntimeError("Hard-negative contact sheet changed after signoff")
    if not isinstance(signoff.get("real_helmet_count"), int):
        raise TypeError("Signoff must include integer real_helmet_count")
    return signoff


def _shape_mask(shape: str, size: int, rng: np.random.Generator) -> np.ndarray:
    scale = 4
    canvas = np.zeros((size * scale, size * scale), dtype=np.uint8)
    height, width = canvas.shape
    margin = round(size * scale * 0.10)
    center = (width // 2, height // 2)
    if shape == "ellipse":
        axes = (width // 2 - margin, round(height * 0.35))
        cv2.ellipse(canvas, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    elif shape == "dome":
        axes = (width // 2 - margin, round(height * 0.38))
        cv2.ellipse(canvas, (center[0], round(height * 0.55)), axes, 0, 180, 360, 255, -1)
        cv2.rectangle(
            canvas,
            (margin, round(height * 0.55)),
            (width - margin, round(height * 0.68)),
            255,
            -1,
        )
    elif shape == "rounded_cylinder":
        radius = round(height * 0.16)
        cv2.rectangle(canvas, (margin, radius + margin), (width - margin, height - radius - margin), 255, -1)
        cv2.ellipse(canvas, (center[0], radius + margin), (center[0] - margin, radius), 0, 0, 360, 255, -1)
        cv2.ellipse(canvas, (center[0], height - radius - margin), (center[0] - margin, radius), 0, 0, 360, 255, -1)
    elif shape == "arc":
        axes = (width // 2 - margin, round(height * 0.35))
        thickness = max(scale * 3, round(size * scale * 0.12))
        cv2.ellipse(canvas, center, axes, 0, 190, 350, 255, thickness, cv2.LINE_AA)
    else:
        raise ValueError(f"Unsupported procedural hard-negative shape: {shape}")
    angle = float(rng.uniform(-18, 18))
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    canvas = cv2.warpAffine(canvas, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def procedural_hard_negative(
    texture_rgb: np.ndarray,
    *,
    shape: str,
    seed: int,
) -> np.ndarray:
    """Create an RGBA distractor whose shading modulates real background texture."""

    texture = np.asarray(texture_rgb, dtype=np.uint8)
    if texture.ndim != 3 or texture.shape[2] != 3 or texture.shape[0] != texture.shape[1]:
        raise ValueError("texture_rgb must be square H×H×3 RGB")
    size = texture.shape[0]
    rng = np.random.default_rng(seed)
    alpha = _shape_mask(shape, size, rng)
    safety_colors = np.asarray(
        [[245, 185, 20], [255, 130, 15], [235, 210, 35], [250, 155, 25]],
        dtype=np.float32,
    )
    color = safety_colors[int(rng.integers(len(safety_colors)))]
    texture_float = texture.astype(np.float32)
    base = 0.38 * texture_float + 0.62 * color

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    normalized_x = (xx - size / 2) / max(size / 2, 1)
    normalized_y = (yy - size / 2) / max(size / 2, 1)
    radial = np.sqrt(normalized_x**2 + normalized_y**2)
    shading = np.clip(1.10 - 0.30 * radial - 0.12 * normalized_y, 0.65, 1.15)
    highlight_center = (-0.30, -0.35)
    highlight = np.exp(
        -(
            (normalized_x - highlight_center[0]) ** 2
            + (normalized_y - highlight_center[1]) ** 2
        )
        / 0.035
    )
    shading += 0.18 * highlight
    rgb = np.clip(base * shading[..., None], 0, 255).astype(np.uint8)
    return np.dstack((rgb, alpha))


def render_procedural_grid(
    *, paths: ProjectPaths, compose_config: Mapping[str, Any], n: int = 64
) -> dict[str, Any]:
    """Render 64 texture-modulated procedural examples from frozen Train backgrounds."""

    _, images, _, _, _ = train_context(paths)
    selected_ids = select_h6_images(sorted(images), seed=int(compose_config["seed"]) + 1, n=n)
    shapes = list(compose_config["hard_negatives"]["procedural"]["shapes"])
    cell = 160
    sheet = Image.new("RGB", (8 * cell, 8 * cell), (32, 32, 32))
    within_shape_std: list[float] = []
    for index, image_id in enumerate(selected_ids):
        image = np.asarray(
            Image.open(paths.hardhat_raw / images[image_id]["file_name"]).convert("RGB")
        )
        rng = np.random.default_rng(int(compose_config["seed"]) + image_id)
        side = int(rng.integers(48, 97))
        left = int(rng.integers(0, image.shape[1] - side + 1))
        top = int(rng.integers(0, image.shape[0] - side + 1))
        texture = image[top : top + side, left : left + side]
        shape = shapes[index % len(shapes)]
        rgba = procedural_hard_negative(
            texture, shape=shape, seed=int(compose_config["seed"]) + index
        )
        mask = rgba[..., 3] >= 128
        within_shape_std.append(float(rgba[..., :3][mask].std()))
        foreground = Image.fromarray(rgba, mode="RGBA").resize(
            (cell - 12, cell - 12), Image.Resampling.LANCZOS
        )
        backdrop = Image.new("RGBA", (cell, cell), (255, 0, 255, 255))
        backdrop.alpha_composite(foreground, (6, 6))
        draw = ImageDraw.Draw(backdrop)
        draw.rectangle((0, cell - 18, cell, cell), fill=(0, 0, 0, 190))
        draw.text((3, cell - 15), f"{shape} src={image_id}", fill="white")
        sheet.paste(backdrop.convert("RGB"), ((index % 8) * cell, (index // 8) * cell))
    output_path = paths.figures / "procedural_hard_negative_grid.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)
    return {
        "rendered": len(selected_ids),
        "mean_within_shape_rgb_std": float(np.mean(within_shape_std)),
        "grid_sha256": _sha256_file(output_path),
        "texture_modulation_asserted": min(within_shape_std) > 0,
    }
