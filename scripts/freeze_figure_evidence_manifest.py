"""Publish a canonical figure commitment manifest from a verified recovery archive."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.repository_archive import (
    MANIFEST_NAME,
    ArchiveError,
    load_and_verify_manifest,
    sha256_file,
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class FreezeFigureEvidenceError(RuntimeError):
    """The requested frozen manifest could not be verified and published safely."""


def freeze(archive: Path, expected_manifest_sha256: str, output: Path) -> tuple[str, int]:
    """Verify an archive completely, then publish its manifest without overwriting."""

    archive = Path(archive)
    output = Path(output)
    if _SHA256_PATTERN.fullmatch(expected_manifest_sha256) is None:
        raise FreezeFigureEvidenceError("expected manifest SHA-256 is invalid")
    if os.path.lexists(output):
        raise FileExistsError(output)
    try:
        entries = load_and_verify_manifest(archive)
        manifest_path = archive / MANIFEST_NAME
        manifest_sha256 = sha256_file(manifest_path)
        manifest_bytes = manifest_path.read_bytes()
    except (ArchiveError, OSError) as error:
        raise FreezeFigureEvidenceError(str(error)) from error
    if manifest_sha256 != expected_manifest_sha256:
        raise FreezeFigureEvidenceError("verified manifest SHA-256 differs from expected value")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(manifest_bytes)
        if os.path.lexists(output):
            raise FileExistsError(output)
        os.rename(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.lexists(temporary_path):
            temporary_path.unlink()
    return manifest_sha256, len(entries)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze verified figure commitments without archive bytes.")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest_sha256, count = freeze(
            arguments.archive, arguments.expected_manifest_sha256, arguments.output
        )
    except (ArchiveError, FreezeFigureEvidenceError, FileExistsError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"manifest_sha256={manifest_sha256} entries={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
