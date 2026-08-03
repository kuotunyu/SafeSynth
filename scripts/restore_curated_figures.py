"""Restore verified KEEP figures after a repository history rewrite."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.archive_repository_curation import load_and_verify_receipt
from src.release.repository_archive import (
    ArchiveError,
    restore_keep_files,
)


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_empty_figures_target(project_root: Path) -> None:
    figures = project_root / "reports" / "figures"
    if not _path_exists(figures):
        return
    if figures.is_symlink() or not figures.is_dir():
        raise ArchiveError(f"restore target is not a directory: {figures}")
    if next(figures.iterdir(), None) is not None:
        raise ArchiveError(f"restore target must be empty: {figures}")
    figures.rmdir()


def restore(project_root: Path, archive_root: Path) -> tuple[str, ...]:
    """Verify all archived content, then restore only the KEEP inventory."""

    load_and_verify_receipt(archive_root)
    _require_empty_figures_target(project_root)
    return tuple(sorted(restore_keep_files(project_root, archive_root)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restore only verified KEEP figures from a completed curation archive."
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        for path in restore(arguments.project_root, arguments.archive):
            print(path)
    except (ArchiveError, FileExistsError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
