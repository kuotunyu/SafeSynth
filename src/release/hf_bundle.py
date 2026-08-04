"""Build the small, reviewable payloads that an owner can upload to Hugging Face."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any


class ReleaseBundleError(RuntimeError):
    """The source artifacts cannot produce an unambiguous release bundle."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseBundleError(f"{path.name} must contain a JSON object")
    return value


def _release_name(value: object) -> str:
    name = PurePosixPath(str(value).replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts or len(name.parts) != 2:
        raise ReleaseBundleError(f"unsafe COCO file_name: {value!r}")
    if name.parts[0] != "images" or not name.parts[1]:
        raise ReleaseBundleError(f"COCO file_name must be images/<name>: {value!r}")
    return name.as_posix()


def _image_names(coco: dict[str, Any], *, label: str) -> set[str]:
    rows = coco.get("images")
    if not isinstance(rows, list):
        raise ReleaseBundleError(f"{label} COCO has no images list")
    names = [_release_name(row.get("file_name")) for row in rows if isinstance(row, dict)]
    if len(names) != len(rows) or len(set(names)) != len(names):
        raise ReleaseBundleError(f"{label} COCO has malformed or duplicate image rows")
    return set(names)


def _prepare_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ReleaseBundleError(f"refusing to overwrite non-empty release directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _copy_or_link(source: Path, destination: Path, *, link_mode: str) -> None:
    if link_mode == "copy":
        shutil.copy2(source, destination)
    elif link_mode == "hardlink":
        os.link(source, destination)
    else:
        raise ReleaseBundleError(f"unsupported link mode: {link_mode!r}")


def prepare_dataset_bundle(
    source_root: Path,
    output_root: Path,
    card_path: Path,
    *,
    link_mode: str = "hardlink",
) -> dict[str, Any]:
    """Package the exact union referenced by the two equal-size COCO releases."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    card_path = Path(card_path)
    if link_mode not in {"copy", "hardlink"}:
        raise ReleaseBundleError(f"unsupported link mode: {link_mode!r}")
    if not card_path.is_file():
        raise ReleaseBundleError(f"missing dataset card: {card_path}")
    filtered_path = source_root / "annotations_filtered_1x.json"
    unfiltered_path = source_root / "annotations_unfiltered_1x.json"
    records_path = source_root / "records.jsonl"
    filtered = _read_json(filtered_path)
    unfiltered = _read_json(unfiltered_path)
    filtered_names = _image_names(filtered, label="filtered")
    unfiltered_names = _image_names(unfiltered, label="unfiltered")
    if len(filtered_names) != len(unfiltered_names):
        raise ReleaseBundleError(
            "filtered and unfiltered releases must contain the same number of images"
        )
    release_names = filtered_names | unfiltered_names

    records_by_name: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ReleaseBundleError(f"records.jsonl line {line_number} is not an object")
        name = _release_name(record.get("file_name"))
        if name in records_by_name:
            raise ReleaseBundleError(f"duplicate provenance for {name}")
        records_by_name[name] = record
    missing_records = sorted(release_names - records_by_name.keys())
    if missing_records:
        raise ReleaseBundleError(f"missing provenance for {missing_records[0]}")

    source_images: dict[str, Path] = {}
    for name in sorted(release_names):
        source = source_root / PurePosixPath(name)
        if not source.is_file():
            raise ReleaseBundleError(f"missing source image: {name}")
        expected_hash = records_by_name[name].get("image_sha256")
        if not isinstance(expected_hash, str) or _sha256(source) != expected_hash.lower():
            raise ReleaseBundleError(f"image hash disagrees with provenance: {name}")
        source_images[name] = source

    _prepare_empty_directory(output_root)
    output_images = output_root / "images"
    output_images.mkdir()
    for name, source in sorted(source_images.items()):
        _copy_or_link(
            source,
            output_images / PurePosixPath(name).name,
            link_mode=link_mode,
        )

    shutil.copy2(filtered_path, output_root / "annotations_filtered.json")
    shutil.copy2(unfiltered_path, output_root / "annotations_unfiltered.json")
    shutil.copy2(card_path, output_root / "README.md")
    released_records = "".join(
        json.dumps(records_by_name[name], ensure_ascii=False, sort_keys=True) + "\n"
        for name in sorted(release_names)
    )
    (output_root / "records.jsonl").write_text(
        released_records, encoding="utf-8", newline="\n"
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "filtered_images": len(filtered_names),
            "unfiltered_images": len(unfiltered_names),
            "shared_images": len(filtered_names & unfiltered_names),
            "unique_images": len(release_names),
            "provenance_records": len(release_names),
        },
        "artifacts": {
            name: {"sha256": _sha256(output_root / name)}
            for name in (
                "README.md",
                "annotations_filtered.json",
                "annotations_unfiltered.json",
                "records.jsonl",
            )
        },
    }
    (output_root / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def prepare_model_bundle(
    checkpoint_dir: Path,
    processor_config: Path,
    output_root: Path,
    card_path: Path,
) -> dict[str, Any]:
    """Copy only the files needed to load the selected inference checkpoint."""

    checkpoint_dir = Path(checkpoint_dir)
    processor_config = Path(processor_config)
    output_root = Path(output_root)
    published = ("config.json", "model.safetensors", "preprocessor_config.json")
    sources = {
        "config.json": checkpoint_dir / "config.json",
        "model.safetensors": checkpoint_dir / "model.safetensors",
        "preprocessor_config.json": processor_config,
    }
    for name, source in sources.items():
        if not source.is_file():
            raise ReleaseBundleError(f"missing model release file {name}: {source}")
    if not Path(card_path).is_file():
        raise ReleaseBundleError(f"missing model card: {card_path}")

    _prepare_empty_directory(output_root)
    for name, source in sources.items():
        shutil.copy2(source, output_root / name)
    shutil.copy2(card_path, output_root / "README.md")

    trainer_state_names = (
        "optimizer.pt",
        "rng_state.pth",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "published_files": list(published),
            "excluded_trainer_state": any(
                (checkpoint_dir / name).exists() for name in trainer_state_names
            ),
        },
        "artifacts": {
            name: {"sha256": _sha256(output_root / name)}
            for name in ("README.md", *published)
        },
    }
    (output_root / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _records_by_name(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReleaseBundleError(
                f"records.jsonl line {line_number} is invalid JSON"
            ) from error
        if not isinstance(record, dict):
            raise ReleaseBundleError(f"records.jsonl line {line_number} is not an object")
        name = _release_name(record.get("file_name"))
        if name in records:
            raise ReleaseBundleError(f"duplicate provenance for {name}")
        records[name] = record
    return records


def _verify_artifact_hashes(root: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ReleaseBundleError("release manifest has no artifact hashes")
    for name, evidence in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise ReleaseBundleError(f"unsafe manifest artifact name: {name!r}")
        path = root / name
        if not path.is_file():
            raise ReleaseBundleError(f"manifest artifact is missing: {name}")
        expected = evidence.get("sha256") if isinstance(evidence, dict) else None
        if not isinstance(expected, str) or _sha256(path) != expected.lower():
            raise ReleaseBundleError(f"artifact hash disagrees with manifest: {name}")


def _reject_local_paths(paths: tuple[Path, ...]) -> None:
    windows_drive = re.compile(r"(?i)(?:^|[^a-z0-9])[a-z]:[\\/]")
    forbidden = ("/users/", "\\users\\", "appdata", "sdg-data")
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        if windows_drive.search(text) or any(token in text for token in forbidden):
            raise ReleaseBundleError(f"local absolute path leaked into {path.name}")


def verify_dataset_bundle(root: Path) -> dict[str, Any]:
    """Re-read a prepared dataset payload and verify the owner upload boundary."""

    root = Path(root)
    expected_top_level = {
        "README.md",
        "annotations_filtered.json",
        "annotations_unfiltered.json",
        "records.jsonl",
        "release_manifest.json",
        "images",
    }
    actual_top_level = {path.name for path in root.iterdir()} if root.is_dir() else set()
    if actual_top_level != expected_top_level:
        extras = sorted(actual_top_level - expected_top_level)
        missing = sorted(expected_top_level - actual_top_level)
        raise ReleaseBundleError(
            f"dataset payload boundary mismatch; unexpected={extras}, missing={missing}"
        )

    manifest = _read_json(root / "release_manifest.json")
    if manifest.get("schema_version") != 1:
        raise ReleaseBundleError("unsupported dataset release manifest schema")
    _verify_artifact_hashes(root, manifest)

    filtered = _read_json(root / "annotations_filtered.json")
    unfiltered = _read_json(root / "annotations_unfiltered.json")
    filtered_names = _image_names(filtered, label="filtered")
    unfiltered_names = _image_names(unfiltered, label="unfiltered")
    if len(filtered_names) != len(unfiltered_names):
        raise ReleaseBundleError(
            "filtered and unfiltered releases must contain the same number of images"
        )
    release_names = filtered_names | unfiltered_names
    records = _records_by_name(root / "records.jsonl")
    if set(records) != release_names:
        raise ReleaseBundleError("dataset provenance set does not equal annotation union")

    images_dir = root / "images"
    image_paths = [path for path in images_dir.iterdir() if path.is_file()]
    if any(path.is_dir() for path in images_dir.iterdir()):
        raise ReleaseBundleError("dataset images directory must not contain subdirectories")
    actual_images = {f"images/{path.name}" for path in image_paths}
    if actual_images != release_names:
        raise ReleaseBundleError("dataset image files do not equal annotation union")
    for name in sorted(release_names):
        expected_hash = records[name].get("image_sha256")
        image_path = root / PurePosixPath(name)
        if not isinstance(expected_hash, str) or _sha256(image_path) != expected_hash.lower():
            raise ReleaseBundleError(f"image hash disagrees with provenance: {name}")

    expected_counts = {
        "filtered_images": len(filtered_names),
        "unfiltered_images": len(unfiltered_names),
        "shared_images": len(filtered_names & unfiltered_names),
        "unique_images": len(release_names),
        "provenance_records": len(records),
    }
    if manifest.get("dataset") != expected_counts:
        raise ReleaseBundleError("dataset manifest counts do not match payload")

    card = (root / "README.md").read_text(encoding="utf-8").lower()
    for required in ("license: cc0-1.0", "sam2", "ground truth", "filtered", "unfiltered"):
        if required not in card:
            raise ReleaseBundleError(f"dataset card is missing required disclosure: {required}")
    filtered_categories = filtered.get("categories")
    if not isinstance(filtered_categories, list) or filtered_categories != unfiltered.get(
        "categories"
    ):
        raise ReleaseBundleError("filtered and unfiltered COCO categories disagree")
    if re.search(r"\|\s*\d+\s*\|\s*`[^`]+`", card):
        for category in filtered_categories:
            if not isinstance(category, dict):
                raise ReleaseBundleError("COCO category row is malformed")
            expected_row = f"| {category.get('id')} | `{category.get('name')}` |".lower()
            if expected_row not in card:
                raise ReleaseBundleError("dataset card category table disagrees with COCO")
    _reject_local_paths(
        (
            root / "README.md",
            root / "annotations_filtered.json",
            root / "annotations_unfiltered.json",
            root / "records.jsonl",
            root / "release_manifest.json",
        )
    )
    return manifest


def verify_model_bundle(root: Path) -> dict[str, Any]:
    """Re-read a prepared model payload and reject non-inference state or drift."""

    root = Path(root)
    expected = {
        "README.md",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "release_manifest.json",
    }
    actual = {path.name for path in root.iterdir()} if root.is_dir() else set()
    extras = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extras:
        raise ReleaseBundleError(f"unexpected model payload file: {extras[0]}")
    if missing:
        raise ReleaseBundleError(f"model payload file is missing: {missing[0]}")

    manifest = _read_json(root / "release_manifest.json")
    if manifest.get("schema_version") != 1:
        raise ReleaseBundleError("unsupported model release manifest schema")
    _verify_artifact_hashes(root, manifest)
    if (root / "model.safetensors").stat().st_size <= 0:
        raise ReleaseBundleError("model.safetensors is empty")

    config = _read_json(root / "config.json")
    labels = config.get("id2label")
    expected_labels = {"0": "helmet", "1": "head", "2": "person"}
    if labels != expected_labels:
        raise ReleaseBundleError("model config label IDs are not exact")
    processor = _read_json(root / "preprocessor_config.json")
    if processor.get("image_processor_type") != "RTDetrImageProcessor":
        raise ReleaseBundleError("model processor type is not RTDetrImageProcessor")
    if processor.get("do_normalize") is not False:
        raise ReleaseBundleError("RT-DETR processor must not enable ImageNet normalization")
    if processor.get("size") != {"height": 640, "width": 640}:
        raise ReleaseBundleError("RT-DETR processor size must be 640x640")

    card = (root / "README.md").read_text(encoding="utf-8").lower()
    for required in (
        "license: apache-2.0",
        "base_model: pekingu/rtdetr_v2_r18vd",
        "absolute",
        "synthetic",
        "four-arm",
    ):
        if required not in card:
            raise ReleaseBundleError(f"model card is missing required disclosure: {required}")
    _reject_local_paths(
        (
            root / "README.md",
            root / "config.json",
            root / "preprocessor_config.json",
            root / "release_manifest.json",
        )
    )
    return manifest
