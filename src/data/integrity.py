"""Leakage guardrails shared by every synthetic-data entry point."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.paths import ProjectPaths, load_project_paths
from src.data.voc_to_coco import DataInvariantError, sha256_file


def assert_test_untouched(
    *,
    paths: ProjectPaths | None = None,
    blocklist_path: Path | None = None,
) -> None:
    """Re-hash every Test image and fail before generation if any byte changed."""

    resolved_paths = paths or load_project_paths()
    resolved_blocklist = blocklist_path or (resolved_paths.splits / "test_blocklist.json")
    if not resolved_blocklist.is_file():
        raise DataInvariantError(f"Missing Test blocklist: {resolved_blocklist}")
    with resolved_blocklist.open("r", encoding="utf-8") as stream:
        blocklist = json.load(stream)

    mismatches: list[str] = []
    for record in blocklist["images"]:
        image_path = resolved_paths.hardhat_raw / record["file_name"]
        if not image_path.is_file():
            mismatches.append(f"missing:{record['file_name']}")
            continue
        actual = sha256_file(image_path)
        if actual != record["sha256"]:
            mismatches.append(f"sha256:{record['file_name']}")
    if mismatches:
        raise DataInvariantError(
            f"Test blocklist verification failed for {len(mismatches)} files: {mismatches[:20]}"
        )
