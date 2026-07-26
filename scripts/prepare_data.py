"""Download, convert, and verify the frozen Hard Hat Workers release."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Running a file sets sys.path[0] to scripts/, so add the discovered project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import ProjectPaths, load_project_paths, pin_dataset_version
from src.data.voc_to_coco import (
    DataInvariantError,
    assert_expected_facts,
    build_conversion_report,
    coco_self_evaluation,
    convert_voc_dataset,
    copy_file_preserving_bytes,
    sha256_file,
    verify_coco_schema,
    write_canonical_json,
)

KAGGLE_DATASET_METADATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/view/andrewmvd/hard-hat-detection"
)


def fetch_official_metadata() -> dict[str, Any]:
    """Read current version and byte count from Kaggle's official API."""

    request = urllib.request.Request(
        KAGGLE_DATASET_METADATA_URL,
        headers={"User-Agent": "SafeSynth-PPE/0.1 data-preparation"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.load(response)
    required = ("currentVersionNumber", "totalBytes", "licenseName", "ref")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise DataInvariantError(f"Kaggle metadata response is missing: {missing}")
    if metadata["ref"] != "andrewmvd/hard-hat-detection":
        raise DataInvariantError(f"Unexpected Kaggle dataset ref: {metadata['ref']}")
    return metadata


def _archive_capture_name(version: int) -> str:
    return f"hard-hat-detection-v{version}.zip"


def download_dataset(
    paths: ProjectPaths,
    *,
    version: int,
    force_download: bool,
) -> tuple[Path, Path]:
    """Use kagglehub while retaining the archive it normally deletes after extraction."""

    load_dotenv(paths.dotenv)
    if not os.getenv("KAGGLE_API_TOKEN"):
        raise DataInvariantError(f"KAGGLE_API_TOKEN is not available after loading {paths.dotenv}")

    import kagglehub
    from kagglehub import http_resolver

    archive_copy = paths.raw / _archive_capture_name(version)
    versioned_handle = f"{paths.kaggle_handle}/versions/{version}"
    if paths.hardhat_raw.exists() and archive_copy.exists() and not force_download:
        return paths.hardhat_raw, archive_copy

    original_extract = http_resolver._extract_archive
    capture_count = 0

    def capture_then_extract(archive_path: str, output_path: str) -> None:
        nonlocal capture_count
        capture_count += 1
        copy_file_preserving_bytes(Path(archive_path), archive_copy)
        original_extract(archive_path, output_path)

    http_resolver._extract_archive = capture_then_extract
    try:
        resolved = Path(
            kagglehub.dataset_download(
                versioned_handle,
                output_dir=str(paths.hardhat_raw),
                force_download=force_download,
            )
        ).resolve()
    finally:
        http_resolver._extract_archive = original_extract

    if resolved != paths.hardhat_raw:
        raise DataInvariantError(
            f"kagglehub resolved unexpected path {resolved}; expected {paths.hardhat_raw}"
        )
    if capture_count != 1 or not archive_copy.is_file():
        raise DataInvariantError(
            "kagglehub did not expose exactly one archive extraction; "
            "its resolver behavior may have changed"
        )
    return resolved, archive_copy


def write_source_checksums(
    *,
    paths: ProjectPaths,
    archive_path: Path,
    official_metadata: dict[str, Any],
    version: int,
) -> dict[str, Any]:
    record = {
        "dataset_handle": paths.kaggle_handle,
        "dataset_version": version,
        "official_total_bytes": int(official_metadata["totalBytes"]),
        "official_last_updated": official_metadata.get("lastUpdated"),
        "official_license": official_metadata["licenseName"],
        "archive": {
            "file_name": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
        },
    }
    write_canonical_json(paths.splits / "source_checksums.json", record)
    return record


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def convert_and_verify(
    *,
    paths: ProjectPaths,
    archive_path: Path,
    official_metadata: dict[str, Any],
    version: int,
) -> None:
    paths.interim.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.splits.mkdir(parents=True, exist_ok=True)

    source_record = write_source_checksums(
        paths=paths,
        archive_path=archive_path,
        official_metadata=official_metadata,
        version=version,
    )
    archive_sha256 = source_record["archive"]["sha256"]
    try:
        coco, stats, audit = convert_voc_dataset(
            paths.hardhat_raw,
            classes=paths.classes,
            kaggle_handle=paths.kaggle_handle,
            kaggle_version=version,
            archive_sha256=archive_sha256,
        )
    except DataInvariantError:
        # An empty list is still useful evidence that failures happened before conversion.
        failures = locals().get("stats")
        write_canonical_json(
            paths.reports / "parse_failures.json",
            failures.parse_failures if failures is not None else [],
        )
        raise

    write_canonical_json(paths.reports / "parse_failures.json", stats.parse_failures)
    verify_coco_schema(coco)
    facts = assert_expected_facts(coco, paths.classes)
    coco_path = paths.interim / "coco_all.json"
    write_canonical_json(coco_path, coco)
    self_map = coco_self_evaluation(coco_path)

    report = build_conversion_report(
        coco=coco,
        stats=stats,
        audit=audit,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_size=archive_path.stat().st_size,
        official_total_bytes=int(official_metadata["totalBytes"]),
        kaggle_version=version,
        self_map=self_map,
        classes=paths.classes,
    )
    (paths.reports / "conversion_report.md").write_text(report, encoding="utf-8", newline="\n")

    print(
        f"global min coordinate = {audit['global_min_coordinate']:g} "
        f"-> offset {audit['coordinate_offset']}"
    )
    print(f"images = {facts['image_count']:,}")
    print(f"annotations = {facts['annotation_count']:,}")
    print(f"class instances = {facts['class_instances']}")
    print(f"class images = {facts['class_images']}")
    print(f"unknown labels = {sum(stats.unknown_labels.values())}")
    print(f"iscrowd != 0 = {sum(item['iscrowd'] != 0 for item in coco['annotations'])}")
    print(f"COCO self-evaluation mAP = {self_map:.3f}")


def verify_existing(paths: ProjectPaths) -> None:
    """Verify existing artifacts without downloading or rewriting them."""

    if paths.pinned_version is None:
        raise DataInvariantError("dataset.pinned_version is null")
    source_path = paths.splits / "source_checksums.json"
    coco_path = paths.interim / "coco_all.json"
    for required in (source_path, coco_path, paths.reports / "conversion_report.md"):
        if not required.is_file():
            raise DataInvariantError(f"Missing M2 artifact: {required}")

    source = _load_json(source_path)
    archive_path = paths.raw / source["archive"]["file_name"]
    if not archive_path.is_file():
        raise DataInvariantError(f"Missing retained source archive: {archive_path}")
    if archive_path.stat().st_size != source["archive"]["size_bytes"]:
        raise DataInvariantError("Retained source archive size changed")
    actual_sha256 = sha256_file(archive_path)
    if actual_sha256 != source["archive"]["sha256"]:
        raise DataInvariantError("Retained source archive SHA256 changed")
    if source["dataset_version"] != paths.pinned_version:
        raise DataInvariantError("Config and source checksum dataset versions differ")

    coco = _load_json(coco_path)
    verify_coco_schema(coco)
    facts = assert_expected_facts(coco, paths.classes)
    if coco["info"]["coordinate_offset"] not in (0, 1):
        raise DataInvariantError("Invalid recorded coordinate offset")
    self_map = coco_self_evaluation(coco_path)
    print(
        f"global min coordinate = {coco['info']['coordinate_global_min']:g} "
        f"-> offset {coco['info']['coordinate_offset']}"
    )
    print(f"archive SHA256 = {actual_sha256}")
    print(f"images = {facts['image_count']:,}")
    print(f"annotations = {facts['annotation_count']:,}")
    print(f"class instances = {facts['class_instances']}")
    print(f"class images = {facts['class_images']}")
    print("unknown labels = 0")
    print(f"iscrowd != 0 = {sum(item['iscrowd'] != 0 for item in coco['annotations'])}")
    print(f"COCO self-evaluation mAP = {self_map:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing M2 artifacts without downloading or rewriting them.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace an existing local extraction with the same pinned upstream version.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = load_project_paths()
        if args.verify:
            verify_existing(paths)
            return 0

        metadata = fetch_official_metadata()
        version = int(metadata["currentVersionNumber"])
        if paths.pinned_version is not None and paths.pinned_version != version:
            raise DataInvariantError(
                f"Pinned version {paths.pinned_version} differs from current Kaggle version {version}"
            )
        pin_dataset_version(paths.config_path, version)
        paths = load_project_paths()
        dataset_root, archive_path = download_dataset(
            paths,
            version=version,
            force_download=args.force_download,
        )
        if dataset_root != paths.hardhat_raw:
            raise DataInvariantError("Resolved dataset root differs from configured hardhat_raw")
        convert_and_verify(
            paths=paths,
            archive_path=archive_path,
            official_metadata=metadata,
            version=version,
        )
        return 0
    except (DataInvariantError, OSError, ValueError, re.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
