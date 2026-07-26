"""M8 two-pass SAM2 cutout bank with auditable gates and deterministic manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from src.data.paths import ProjectPaths
from src.synthetic.mask_ops import decode_rle, encode_rle, quality_failures
from src.synthetic.sam2_runner import Sam2BoxSegmenter, xywh_to_xyxy
from src.synthetic.survey import load_json, train_context

BANK_STATE_SCHEMA_VERSION = 1


def _canonical_json(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_sha256(mask: np.ndarray) -> str:
    header = np.asarray(mask.shape, dtype=np.int32).tobytes()
    return _sha256_bytes(header + np.packbits(mask, axis=None).tobytes())


def _write_json_atomic(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(record), encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(_canonical_json(record) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
        newline="\n",
    )


def _box_intersection(first: Sequence[float], second: Sequence[float]) -> float:
    first_xyxy = xywh_to_xyxy(first)
    second_xyxy = xywh_to_xyxy(second)
    width = max(min(first_xyxy[2], second_xyxy[2]) - max(first_xyxy[0], second_xyxy[0]), 0)
    height = max(min(first_xyxy[3], second_xyxy[3]) - max(first_xyxy[1], second_xyxy[1]), 0)
    return width * height


def _occlusion_ratio(
    annotation: Mapping[str, Any],
    image_annotations: Sequence[Mapping[str, Any]],
    category_names: Mapping[int, str],
    *,
    image_shape: tuple[int, int],
) -> float:
    """Measure union coverage by relevant annotated objects without double counting."""

    class_name = category_names[int(annotation["category_id"])]
    relevant: list[Mapping[str, Any]] = []
    for other in image_annotations:
        if int(other["id"]) == int(annotation["id"]):
            continue
        other_class = category_names[int(other["category_id"])]
        if class_name == "person" and other_class != "person":
            continue
        if class_name != "person" and other_class == "person":
            continue
        if _box_intersection(annotation["bbox"], other["bbox"]) > 0:
            relevant.append(other)
    if not relevant:
        return 0.0

    height, width = image_shape
    candidate = np.zeros((height, width), dtype=bool)
    x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
    left = max(0, int(np.floor(x)))
    top = max(0, int(np.floor(y)))
    right = min(width, int(np.ceil(x + box_width)))
    bottom = min(height, int(np.ceil(y + box_height)))
    candidate[top:bottom, left:right] = True
    occupied = np.zeros_like(candidate)
    for other in relevant:
        ox, oy, ow, oh = (float(value) for value in other["bbox"])
        other_left = max(0, int(np.floor(ox)))
        other_top = max(0, int(np.floor(oy)))
        other_right = min(width, int(np.ceil(ox + ow)))
        other_bottom = min(height, int(np.ceil(oy + oh)))
        occupied[other_top:other_bottom, other_left:other_right] = True
    candidate_area = int(candidate.sum())
    return float((candidate & occupied).sum()) / max(candidate_area, 1)


def cheap_gate_record(
    *,
    annotation: Mapping[str, Any],
    image_record: Mapping[str, Any],
    image_annotations: Sequence[Mapping[str, Any]],
    category_names: Mapping[int, str],
    frozen_record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply CUT-06 gates in order and retain every measured value."""

    bank = config["cutout_bank"]
    class_name = category_names[int(annotation["category_id"])]
    x, y, width, height = (float(value) for value in annotation["bbox"])
    area = width * height
    min_side = min(width, height)
    aspect = width / height
    image_width = int(image_record["width"])
    image_height = int(image_record["height"])
    edge_distance = min(x, y, image_width - (x + width), image_height - (y + height))
    occlusion = _occlusion_ratio(
        annotation,
        image_annotations,
        category_names,
        image_shape=(image_height, image_width),
    )
    failures: list[str] = []
    if bank["respect_voc_flags"] and (
        int(annotation.get("truncated", 0)) != 0
        or int(annotation.get("difficult", 0)) != 0
    ):
        failures.append("G1_VOC_FLAG")
    hard_floor = bank["hard_floor"]
    if (
        min_side < float(hard_floor["min_side_px"])
        or area < float(hard_floor["min_area_px"])
    ):
        failures.append("G2_HARD_FLOOR")
    low_aspect, high_aspect = bank["aspect_ratio"][class_name]
    if not float(low_aspect) <= aspect <= float(high_aspect):
        failures.append("G4_ASPECT_RATIO")
    if edge_distance < float(bank["min_distance_to_image_edge_px"]):
        failures.append("G5_IMAGE_EDGE")
    if occlusion > float(bank["max_occlusion_by_others"]):
        failures.append("G6_OCCLUDED")

    preferred = bank["preferred_tier"]
    return {
        "annotation_id": int(annotation["id"]),
        "image_id": int(annotation["image_id"]),
        "file_name": str(image_record["file_name"]),
        "image_sha256": str(frozen_record["sha256"]),
        "group_id": int(frozen_record["group_id"]),
        "split": str(frozen_record["split"]),
        "category_id": int(annotation["category_id"]),
        "class_name": class_name,
        "bbox": [float(value) for value in annotation["bbox"]],
        "voc_flags": {
            "truncated": int(annotation.get("truncated", 0)),
            "difficult": int(annotation.get("difficult", 0)),
        },
        "measurements": {
            "area_px": area,
            "min_side_px": min_side,
            "aspect_ratio": aspect,
            "distance_to_edge_px": edge_distance,
            "occlusion_by_others": occlusion,
        },
        "preferred_tier": (
            min_side >= float(preferred["min_side_px"])
            and area >= float(preferred["min_area_px"])
        ),
        "cheap_gate_failures": failures,
        "cheap_gate_pass": not failures,
    }


def prepare_candidates(
    *, paths: ProjectPaths, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze every Train candidate and its cheap-gate decision before GPU work."""

    _, images, annotations, category_names, frozen = train_context(paths)
    test_blocklist = load_json(paths.splits / "test_blocklist.json")
    test_ids = {int(item["image_id"]) for item in test_blocklist["images"]}
    if test_ids & set(images):
        raise RuntimeError("Train candidate set intersects the frozen Test blocklist")

    records: list[dict[str, Any]] = []
    for image_id in sorted(images):
        for annotation in annotations.get(image_id, []):
            records.append(
                cheap_gate_record(
                    annotation=annotation,
                    image_record=images[image_id],
                    image_annotations=annotations[image_id],
                    category_names=category_names,
                    frozen_record=frozen[image_id],
                    config=config,
                )
            )
    records.sort(key=lambda item: int(item["annotation_id"]))
    _write_jsonl(paths.cutouts / "_candidates.jsonl", records)
    failures = Counter(
        failure
        for record in records
        for failure in record["cheap_gate_failures"][:1]
    )
    return {
        "train_candidates": len(records),
        "cheap_gate_pass": sum(bool(record["cheap_gate_pass"]) for record in records),
        "cheap_first_reject": dict(sorted(failures.items())),
        "preferred_among_pass": sum(
            bool(record["cheap_gate_pass"] and record["preferred_tier"]) for record in records
        ),
    }


def _candidate_fingerprint(candidate: Mapping[str, Any], config_sha256: str) -> str:
    inputs = {
        "schema": BANK_STATE_SCHEMA_VERSION,
        "annotation_id": candidate["annotation_id"],
        "image_sha256": candidate["image_sha256"],
        "bbox": candidate["bbox"],
        "config_sha256": config_sha256,
    }
    return _sha256_bytes(_canonical_json(inputs).encode("utf-8"))


def run_pass2(
    *,
    paths: ProjectPaths,
    config: Mapping[str, Any],
    segmenter: Sam2BoxSegmenter,
    config_path: Path,
    max_candidates: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run or resume crop-mode SAM2 for every cheap-gate candidate."""

    candidate_path = paths.cutouts / "_candidates.jsonl"
    if not candidate_path.exists():
        prepare_candidates(paths=paths, config=config)
    candidates = [
        json.loads(line)
        for line in candidate_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    candidates = [record for record in candidates if record["cheap_gate_pass"]]
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    state_dir = paths.cutouts / "_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = _sha256_file(config_path)
    crop_config = config["sam2"]["pass2_bank"]
    processed = skipped = passed = 0

    for candidate in tqdm(candidates, desc="SAM2 Pass 2", unit="candidate"):
        state_path = state_dir / f"{int(candidate['annotation_id']):06d}.json"
        fingerprint = _candidate_fingerprint(candidate, config_sha256)
        if state_path.exists() and not force:
            state = load_json(state_path)
            if state.get("fingerprint") == fingerprint:
                skipped += 1
                passed += int(bool(state["mask_quality_pass"]))
                continue
        image = Image.open(paths.hardhat_raw / candidate["file_name"]).convert("RGB")
        prediction = segmenter.predict_crop(
            image,
            xywh_to_xyxy(candidate["bbox"]),
            context_pad_frac=float(crop_config["context_pad_frac"]),
            min_crop_side_px=int(crop_config["min_crop_side_px"]),
            target_size=int(crop_config["resize_to"]),
        )
        failures = quality_failures(
            prediction.metrics,
            class_name=str(candidate["class_name"]),
            config=config,
        )
        state = {
            "schema_version": BANK_STATE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "annotation_id": int(candidate["annotation_id"]),
            "model_id": config["sam2"]["model_id"],
            "effective_crop_size": int(crop_config["resize_to"]),
            "model_canvas_size": int(crop_config["model_canvas_size"]),
            "metrics": prediction.metrics,
            "mask_quality_failures": failures,
            "mask_quality_pass": not failures,
            "segmentation": encode_rle(prediction.mask),
            "mask_sha256": _mask_sha256(prediction.mask),
        }
        _write_json_atomic(state_path, state)
        processed += 1
        passed += int(not failures)
    return {
        "selected_candidates": len(candidates),
        "processed": processed,
        "skipped": skipped,
        "mask_quality_pass": passed,
    }


def soft_alpha(mask: np.ndarray, *, config: Mapping[str, Any]) -> np.ndarray:
    """Create CUT-10's eroded, scale-aware Gaussian-feathered alpha."""

    blending = config["compose"]["blending"]
    binary = mask.astype(np.uint8)
    erode_px = int(blending["erode_before_feather_px"])
    if erode_px > 0 and binary.any():
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=erode_px)
        if eroded.any():
            binary = eroded
    min_side = min(mask.shape)
    sigma = float(blending["feather_sigma_base"]) + float(
        blending["feather_sigma_per_px"]
    ) * min_side
    sigma = float(np.clip(sigma, *blending["feather_sigma_clip"]))
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    if float(blurred.max()) > 0:
        blurred /= float(blurred.max())
    return np.clip(np.rint(blurred * 255), 0, 255).astype(np.uint8)


def appearance_statistics(rgb: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Compute reusable Lab, high-frequency noise, and hue statistics."""

    if not mask.any():
        raise ValueError("Cannot measure an empty cutout mask")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pixels = lab[mask]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    high_frequency = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0][mask]
    histogram = np.bincount(hue, minlength=180)
    dominant_hue_deg = int(np.argmax(histogram)) * 2
    return {
        "lab_mean": [float(value) for value in pixels.mean(axis=0)],
        "lab_std": [float(value) for value in pixels.std(axis=0)],
        "hf_noise_sigma": float(high_frequency[mask].std()),
        "dominant_hue_deg": dominant_hue_deg,
        "lab_encoding": "OpenCV uint8 Lab scale",
    }


def _crop_bounds(
    bbox: Sequence[float], image_size: tuple[int, int], *, pad_fraction: float = 0.0
) -> tuple[int, int, int, int]:
    x, y, width, height = (float(value) for value in bbox)
    pad = max(width, height) * pad_fraction
    image_width, image_height = image_size
    left = max(0, int(np.floor(x - pad)))
    top = max(0, int(np.floor(y - pad)))
    right = min(image_width, int(np.ceil(x + width + pad)))
    bottom = min(image_height, int(np.ceil(y + height + pad)))
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _git_sha(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _render_cutout(
    *,
    paths: ProjectPaths,
    candidate: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    built_at: str,
    git_sha: str,
) -> dict[str, Any]:
    image = Image.open(paths.hardhat_raw / candidate["file_name"]).convert("RGB")
    rgb = np.asarray(image)
    mask = decode_rle(state["segmentation"])
    bounds = _crop_bounds(candidate["bbox"], image.size)
    left, top, right, bottom = bounds
    crop_rgb = rgb[top:bottom, left:right]
    crop_mask = mask[top:bottom, left:right]
    alpha = soft_alpha(crop_mask, config=config)
    rgba = np.dstack((crop_rgb, alpha))

    class_name = str(candidate["class_name"])
    cutout_id = f"{int(candidate['image_id']):06d}_ann{int(candidate['annotation_id']):06d}"
    relative_png = Path(class_name) / f"{cutout_id}.png"
    png_path = paths.cutouts / relative_png
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(png_path, optimize=True)

    context_bounds = _crop_bounds(candidate["bbox"], image.size, pad_fraction=0.60)
    context = image.crop(context_bounds)
    relative_context = Path("_ctx") / f"{cutout_id}_ctx.png"
    context_path = paths.cutouts / relative_context
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context.save(context_path, optimize=True)

    stats = appearance_statistics(crop_rgb, crop_mask)
    return {
        "schema_version": 1,
        "cutout_id": cutout_id,
        "class_name": class_name,
        "category_id": int(candidate["category_id"]),
        "src_image_id": int(candidate["image_id"]),
        "src_image_sha256": candidate["image_sha256"],
        "src_split": candidate["split"],
        "src_group_id": int(candidate["group_id"]),
        "src_bbox_xywh": candidate["bbox"],
        "voc_flags": candidate["voc_flags"],
        "preferred_tier": bool(candidate["preferred_tier"]),
        "sam2": {
            "model_id": state["model_id"],
            "effective_crop_size": state["effective_crop_size"],
            "model_canvas_size": state["model_canvas_size"],
            "mask_sha256": state["mask_sha256"],
            **state["metrics"],
        },
        "appearance": stats,
        "file": relative_png.as_posix(),
        "context_file": relative_context.as_posix(),
        "file_sha256": _sha256_file(png_path),
        "gates": {"cheap": "pass", "mask_quality": "pass"},
        "max_uses": int(config["compose"]["max_uses_per_cutout"]),
        "use_count": 0,
        "build": {
            "git_sha": git_sha,
            "seed": int(config["seed"]),
            "built_at_utc": built_at,
        },
    }


def finalize_bank(
    *,
    paths: ProjectPaths,
    config: Mapping[str, Any],
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Apply person diversity cap, render RGBA/contexts, and rebuild both ledgers."""

    candidates = [
        json.loads(line)
        for line in (paths.cutouts / "_candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    candidate_by_id = {int(record["annotation_id"]): record for record in candidates}
    states = {
        int(path.stem): load_json(path)
        for path in sorted((paths.cutouts / "_state").glob("*.json"))
    }
    expected_state_ids = {
        int(record["annotation_id"]) for record in candidates if record["cheap_gate_pass"]
    }
    missing = sorted(expected_state_ids - set(states))
    if missing and not allow_incomplete:
        raise RuntimeError(f"Pass 2 is incomplete: {len(missing)} candidate states are missing")

    quality_pass_ids = [
        annotation_id
        for annotation_id, state in states.items()
        if state["mask_quality_pass"] and annotation_id in candidate_by_id
    ]
    person_cap = int(config["cutout_bank"]["max_person_cutouts_per_group"])
    person_by_group: dict[int, list[int]] = defaultdict(list)
    for annotation_id in quality_pass_ids:
        candidate = candidate_by_id[annotation_id]
        if candidate["class_name"] == "person":
            person_by_group[int(candidate["group_id"])].append(annotation_id)
    capped_person_ids = {
        annotation_id
        for annotation_ids in person_by_group.values()
        for annotation_id in sorted(annotation_ids)[person_cap:]
    }
    accepted_ids = sorted(set(quality_pass_ids) - capped_person_ids)

    built_at = datetime.now(UTC).isoformat()
    git_sha = _git_sha(paths.project_root)
    manifest: list[dict[str, Any]] = []
    for annotation_id in tqdm(accepted_ids, desc="Render cutout bank", unit="cutout"):
        manifest.append(
            _render_cutout(
                paths=paths,
                candidate=candidate_by_id[annotation_id],
                state=states[annotation_id],
                config=config,
                built_at=built_at,
                git_sha=git_sha,
            )
        )

    rejects: list[dict[str, Any]] = []
    for candidate in candidates:
        annotation_id = int(candidate["annotation_id"])
        if candidate["cheap_gate_failures"]:
            rejects.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": candidate["image_id"],
                    "class_name": candidate["class_name"],
                    "stage": "cheap_gate",
                    "first_reason": candidate["cheap_gate_failures"][0],
                    "reasons": candidate["cheap_gate_failures"],
                }
            )
        elif annotation_id not in states:
            if allow_incomplete:
                continue
            raise AssertionError("Unreachable missing state")
        elif not states[annotation_id]["mask_quality_pass"]:
            reasons = [
                f"SAM2_{reason.upper()}"
                for reason in states[annotation_id]["mask_quality_failures"]
            ]
            rejects.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": candidate["image_id"],
                    "class_name": candidate["class_name"],
                    "stage": "mask_quality",
                    "first_reason": reasons[0],
                    "reasons": reasons,
                    "metrics": states[annotation_id]["metrics"],
                }
            )
        elif annotation_id in capped_person_ids:
            rejects.append(
                {
                    "annotation_id": annotation_id,
                    "image_id": candidate["image_id"],
                    "class_name": candidate["class_name"],
                    "stage": "person_group_cap",
                    "first_reason": "PERSON_GROUP_CAP",
                    "reasons": ["PERSON_GROUP_CAP"],
                }
            )

    _write_jsonl(paths.cutouts / "bank_manifest.jsonl", manifest)
    _write_jsonl(paths.cutouts / "bank_rejects.jsonl", rejects)
    stale = _move_stale_pngs(paths, manifest)
    summary = validate_bank(
        paths=paths,
        expected_total=(len(candidates) if not allow_incomplete else None),
    )
    summary["stale_pngs_moved"] = stale
    render_bank_grids(paths=paths, manifest=manifest)
    write_bank_report(paths=paths, manifest=manifest, rejects=rejects, summary=summary)
    return summary


def _move_stale_pngs(paths: ProjectPaths, manifest: Sequence[Mapping[str, Any]]) -> int:
    expected = {(paths.cutouts / str(record["file"])).resolve() for record in manifest}
    stale_paths = [
        path.resolve()
        for class_name in ("helmet", "head", "person")
        for path in (paths.cutouts / class_name).glob("*.png")
        if path.resolve() not in expected
    ]
    if not stale_paths:
        return 0
    stale_dir = paths.cutouts / "_stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    for path in stale_paths:
        if paths.cutouts.resolve() not in path.parents:
            raise RuntimeError(f"Refusing to move path outside cutouts: {path}")
        shutil.move(str(path), str(stale_dir / path.name))
    return len(stale_paths)


def validate_bank(
    *, paths: ProjectPaths, expected_total: int | None = None
) -> dict[str, Any]:
    """Verify M8's ledger, leakage, RGBA, and person-diversity invariants."""

    manifest = [
        json.loads(line)
        for line in (paths.cutouts / "bank_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    rejects = [
        json.loads(line)
        for line in (paths.cutouts / "bank_rejects.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    test_ids = {
        int(item["image_id"])
        for item in load_json(paths.splits / "test_blocklist.json")["images"]
    }
    leaked = [record["cutout_id"] for record in manifest if record["src_image_id"] in test_ids]
    if leaked:
        raise RuntimeError(f"Cutout bank leaked Test sources: {leaked[:10]}")
    png_paths = [
        path
        for class_name in ("helmet", "head", "person")
        for path in (paths.cutouts / class_name).glob("*.png")
    ]
    if len(png_paths) != len(manifest):
        raise RuntimeError(
            f"Manifest/PNG mismatch: manifest={len(manifest)}, PNG={len(png_paths)}"
        )
    for record in manifest:
        path = paths.cutouts / record["file"]
        rgba = np.asarray(Image.open(path))
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError(f"Cutout is not RGBA: {path}")
        alpha = rgba[..., 3]
        if not alpha.any() or np.all(alpha == 255):
            raise RuntimeError(f"Cutout has degenerate alpha: {path}")
        if _sha256_file(path) != record["file_sha256"]:
            raise RuntimeError(f"Cutout checksum drift: {path}")
    if expected_total is not None and len(manifest) + len(rejects) != expected_total:
        raise RuntimeError(
            f"Funnel mismatch: {len(manifest)} + {len(rejects)} != {expected_total}"
        )
    person_records = [record for record in manifest if record["class_name"] == "person"]
    return {
        "manifest_count": len(manifest),
        "reject_count": len(rejects),
        "png_count": len(png_paths),
        "test_leak_count": len(leaked),
        "person_cutouts": len(person_records),
        "distinct_person_groups": len(
            {int(record["src_group_id"]) for record in person_records}
        ),
        "manifest_sha256": _sha256_file(paths.cutouts / "bank_manifest.jsonl"),
        "rejects_sha256": _sha256_file(paths.cutouts / "bank_rejects.jsonl"),
    }


def render_bank_grids(
    *, paths: ProjectPaths, manifest: Sequence[Mapping[str, Any]], per_class: int = 64
) -> None:
    """Render CUT-10 RGBA cutouts over the required magenta checker-proof backdrop."""

    font = ImageFont.load_default()
    for class_name in ("helmet", "head", "person"):
        records = [record for record in manifest if record["class_name"] == class_name]
        records = sorted(
            records, key=lambda record: _sha256_bytes(record["cutout_id"].encode())
        )[:per_class]
        if not records:
            continue
        columns = 8
        cell = 160
        rows = int(np.ceil(len(records) / columns))
        sheet = Image.new("RGB", (columns * cell, rows * cell), (255, 0, 255))
        for index, record in enumerate(records):
            rgba = Image.open(paths.cutouts / record["file"]).convert("RGBA")
            scale = min((cell - 8) / rgba.width, (cell - 25) / rgba.height, 1.0)
            resized = rgba.resize(
                (max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale))),
                Image.Resampling.LANCZOS,
            )
            backdrop = Image.new("RGBA", (cell, cell), (255, 0, 255, 255))
            x = (cell - resized.width) // 2
            y = (cell - resized.height) // 2
            backdrop.alpha_composite(resized, (x, y))
            draw = ImageDraw.Draw(backdrop)
            draw.rectangle((0, cell - 18, cell, cell), fill=(0, 0, 0, 190))
            draw.text((3, cell - 15), record["cutout_id"], fill="white", font=font)
            sheet.paste(backdrop.convert("RGB"), ((index % columns) * cell, (index // columns) * cell))
        output = paths.figures / f"bank_{class_name}_grid.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, optimize=True)


def write_bank_report(
    *,
    paths: ProjectPaths,
    manifest: Sequence[Mapping[str, Any]],
    rejects: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Write the reproducible M8 funnel and diversity report."""

    accepted_by_class = Counter(str(record["class_name"]) for record in manifest)
    preferred_by_class = Counter(
        str(record["class_name"]) for record in manifest if record["preferred_tier"]
    )
    first_rejects = Counter(str(record["first_reason"]) for record in rejects)
    lines = [
        "# M8 cutout bank report",
        "",
        "Input is frozen Train only. Val/Test sources are a hard failure.",
        "",
        "## Funnel",
        "",
        f"- Candidates: {len(manifest) + len(rejects)}",
        f"- Accepted: {len(manifest)}",
        f"- Rejected: {len(rejects)}",
        "",
        "| first rejection reason | count |",
        "|---|---:|",
        *[f"| {reason} | {count} |" for reason, count in sorted(first_rejects.items())],
        "",
        "## Accepted material",
        "",
        "| class | accepted | preferred tier |",
        "|---|---:|---:|",
        *[
            f"| {class_name} | {accepted_by_class[class_name]} | "
            f"{preferred_by_class[class_name]} |"
            for class_name in ("helmet", "head", "person")
        ],
        "",
        f"- `n_person_cutouts`: {summary['person_cutouts']}",
        f"- `n_distinct_person_groups`: {summary['distinct_person_groups']}",
        f"- Test blocklist hits: {summary['test_leak_count']}",
        f"- Manifest rows == PNG files: {summary['manifest_count']} == {summary['png_count']}",
        f"- Manifest SHA256: `{summary['manifest_sha256']}`",
        f"- Reject ledger SHA256: `{summary['rejects_sha256']}`",
        "",
        (
            "The funnel above is aggregated directly from `bank_rejects.jsonl`; "
            "accepted + rejected equals the frozen candidate count."
        ),
        "",
    ]
    (paths.reports / "bank_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def verify_mask_reproducibility(
    *,
    paths: ProjectPaths,
    config: Mapping[str, Any],
    segmenter: Sam2BoxSegmenter,
    n: int = 100,
) -> dict[str, Any]:
    """Rerun N accepted masks and require byte-identical packed binary masks."""

    candidates = {
        int(record["annotation_id"]): record
        for record in (
            json.loads(line)
            for line in (paths.cutouts / "_candidates.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        )
    }
    manifest = [
        json.loads(line)
        for line in (paths.cutouts / "bank_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ][:n]
    crop_config = config["sam2"]["pass2_bank"]
    mismatches: list[int] = []
    for record in tqdm(manifest, desc="Reproducibility", unit="mask"):
        annotation_id = int(record["cutout_id"].split("_ann")[-1])
        candidate = candidates[annotation_id]
        image = Image.open(paths.hardhat_raw / candidate["file_name"]).convert("RGB")
        prediction = segmenter.predict_crop(
            image,
            xywh_to_xyxy(candidate["bbox"]),
            context_pad_frac=float(crop_config["context_pad_frac"]),
            min_crop_side_px=int(crop_config["min_crop_side_px"]),
            target_size=int(crop_config["resize_to"]),
        )
        if _mask_sha256(prediction.mask) != record["sam2"]["mask_sha256"]:
            mismatches.append(annotation_id)
    if mismatches:
        raise RuntimeError(f"SAM2 mask reproducibility failed: {mismatches[:10]}")
    result = {
        "rerun_masks": len(manifest),
        "mismatches": len(mismatches),
        "model_id": str(config["sam2"]["model_id"]),
        "dtype": str(config["sam2"]["dtype"]),
        "manifest_sha256": _sha256_file(paths.cutouts / "bank_manifest.jsonl"),
    }
    _write_json_atomic(paths.reports / "bank_reproducibility.json", result)
    report_path = paths.reports / "bank_report.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "## Mask reproducibility"
    section = (
        f"\n{marker}\n\n"
        f"- Re-run accepted masks: {result['rerun_masks']}\n"
        f"- Byte-identical masks: {result['rerun_masks'] - result['mismatches']}\n"
        f"- Mismatches: {result['mismatches']}\n"
        f"- Model: `{result['model_id']}` ({result['dtype']})\n"
    )
    if marker in report:
        report = report.split(marker, 1)[0].rstrip() + section
    else:
        report = report.rstrip() + "\n" + section
    report_path.write_text(report, encoding="utf-8", newline="\n")
    return result
