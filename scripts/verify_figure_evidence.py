"""Fail closed unless the repository has exactly the approved figure state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.figure_evidence import (
    FigureEvidenceError,
    load_repository_figure_manifest,
    verify_approved_figure_manifest,
    verify_repository_figure_state,
)
from src.release.repository_archive import ArchiveError, load_manifest_commitments


def verify(project_root: Path, expected_state: str) -> str:
    """Check the pinned canonical manifest and requested complete repository state."""

    manifest_path = Path(project_root) / "reports" / "figure_curation_manifest.json"
    try:
        source_commit, entries, manifest_sha256 = load_manifest_commitments(manifest_path)
    except ArchiveError as error:
        raise FigureEvidenceError(str(error)) from error
    verify_approved_figure_manifest(source_commit, entries, manifest_sha256)
    entries = load_repository_figure_manifest(
        project_root, commitments=(source_commit, entries, manifest_sha256)
    )
    keep_count = sum(entry.disposition == "KEEP" for entry in entries)
    drop_count = sum(entry.disposition == "DROP" for entry in entries)
    state = verify_repository_figure_state(project_root, entries, expected_state)  # type: ignore[arg-type]
    return f"manifest_sha256={manifest_sha256} total={len(entries)} keep={keep_count} drop={drop_count} state={state}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the complete curated or source figure evidence state.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--expected-state", choices=("source", "curated", "auto"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        print(verify(arguments.project_root, arguments.expected_state))
    except (ArchiveError, FigureEvidenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
