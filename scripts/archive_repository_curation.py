"""Create a verified recovery archive before its owner rewrites repository history."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.markdown_links import RepositoryLinkError
from src.release.repository_archive import ArchiveError, create_recovery_package, sha256_file
from src.release.repository_curation import plan_figure_curation, tracked_files

BUNDLE_NAME = "SafeSynth-pre-filter-repo.bundle"
RECEIPT_NAME = "archive_receipt.json"
RUNBOOK_NAME = "OWNER_HISTORY_REWRITE_RUNBOOK.txt"


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _power_shell_literal(value: str) -> str:
    return value.replace("'", "''")


def _write_canonical_json(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(
        (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )


def _runbook(owner_project_root: str, archive_root: str) -> str:
    owner = _power_shell_literal(owner_project_root)
    archive = _power_shell_literal(archive_root)
    lines = (
        f"Set-Location -LiteralPath '{owner}'",
        "git status --short --branch",
        "uvx git-filter-repo --version",
        "uvx git-filter-repo --path reports/figures/ --invert-paths --force",
        f"uv run python scripts/restore_curated_figures.py --archive '{archive}'",
        "git add -- reports/figures",
        "git diff --cached --check",
        "git commit -m 'docs: restore curated figure evidence'",
    )
    return "\n".join(lines) + "\n"


def _verify_renamed_bundle(project_root: Path, bundle_path: Path, expected_digest: str) -> None:
    if not bundle_path.is_file() or sha256_file(bundle_path) != expected_digest:
        raise ArchiveError("published Git bundle digest does not match the verified bundle")
    completed = subprocess.run(
        ["git", "bundle", "verify", str(bundle_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArchiveError(f"published Git bundle verification failed: {detail}")


def archive(project_root: Path, destination: Path, owner_project_root: str, archive_argument: str) -> None:
    """Build a recovery package and write owner instructions after every verification."""

    if _path_exists(destination):
        raise FileExistsError(f"destination already exists: {destination}")
    plan = plan_figure_curation(project_root, tracked_files(project_root))
    recovery = create_recovery_package(project_root, destination, plan)

    published_bundle = destination / BUNDLE_NAME
    recovery.bundle_path.rename(published_bundle)
    _verify_renamed_bundle(project_root, published_bundle, recovery.bundle_sha256)

    keep_count = sum(entry.disposition == "KEEP" for entry in recovery.entries)
    drop_count = sum(entry.disposition == "DROP" for entry in recovery.entries)
    receipt = {
        "schema_version": 1,
        "source_commit": recovery.source_commit,
        "manifest": recovery.manifest_path.name,
        "manifest_sha256": recovery.manifest_sha256,
        "bundle": published_bundle.name,
        "bundle_sha256": recovery.bundle_sha256,
        "keep_count": keep_count,
        "drop_count": drop_count,
    }
    _write_canonical_json(destination / RECEIPT_NAME, receipt)
    (destination / RUNBOOK_NAME).write_text(
        _runbook(owner_project_root, archive_argument), encoding="utf-8", newline="\n"
    )
    print(
        f"archive={destination} source_commit={recovery.source_commit} "
        f"KEEP={keep_count} DROP={drop_count} "
        f"manifest_sha256={recovery.manifest_sha256} "
        f"bundle_sha256={recovery.bundle_sha256}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a verified repository curation recovery archive; never rewrite history."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--owner-project-root",
        required=True,
        help="Exact owner repository root to place in the generated PowerShell runbook.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        archive(
            Path(arguments.project_root),
            Path(arguments.destination),
            arguments.owner_project_root,
            arguments.destination,
        )
    except (
        ArchiveError,
        FileExistsError,
        OSError,
        RepositoryLinkError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
