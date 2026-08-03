"""Create a verified recovery archive before its owner rewrites repository history."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.markdown_links import RepositoryLinkError
from src.release.repository_archive import (
    MANIFEST_NAME,
    ArchiveEntry,
    ArchiveError,
    ArchiveReceipt,
    create_recovery_package,
    load_and_verify_manifest,
    sha256_file,
)
from src.release.repository_curation import plan_figure_curation, tracked_files

BUNDLE_NAME = "SafeSynth-pre-filter-repo.bundle"
RECEIPT_NAME = "archive_receipt.json"
RUNBOOK_NAME = "OWNER_HISTORY_REWRITE_RUNBOOK.txt"
RECEIPT_SCHEMA_VERSION = 1
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "source_commit",
        "manifest",
        "manifest_sha256",
        "bundle",
        "bundle_sha256",
        "entries",
        "keep_count",
        "drop_count",
    }
)
_RECEIPT_ENTRY_KEYS = frozenset(
    {"path", "size_bytes", "sha256", "disposition", "reference_sources"}
)
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class CurationReceipt:
    """The immutable package facts that bind a restore to its recovery archive."""

    source_commit: str
    manifest_sha256: str
    bundle_sha256: str
    entries: tuple[ArchiveEntry, ...]
    keep_count: int
    drop_count: int


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _power_shell_literal(value: str) -> str:
    return value.replace("'", "''")


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _write_canonical_json(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(_canonical_json_bytes(value))


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError(f"receipt contains duplicate key: {key}")
        result[key] = value
    return result


def _canonical_receipt_object(path: Path) -> dict[str, Any]:
    try:
        receipt_bytes = path.read_bytes()
        receipt = json.loads(
            receipt_bytes.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveError(f"receipt is not valid UTF-8 JSON: {error}") from error
    if not isinstance(receipt, dict) or receipt_bytes != _canonical_json_bytes(receipt):
        raise ArchiveError("receipt is not canonical JSON")
    return receipt


def _receipt_entry(raw_entry: object) -> ArchiveEntry:
    if not isinstance(raw_entry, dict) or set(raw_entry) != _RECEIPT_ENTRY_KEYS:
        raise ArchiveError("receipt entry has invalid fields")
    path = raw_entry["path"]
    size_bytes = raw_entry["size_bytes"]
    digest = raw_entry["sha256"]
    disposition = raw_entry["disposition"]
    references = raw_entry["reference_sources"]
    if (
        not isinstance(path, str)
        or type(size_bytes) is not int
        or size_bytes < 0
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
        or disposition not in {"KEEP", "DROP"}
        or not isinstance(references, list)
        or any(not isinstance(reference, str) or not reference for reference in references)
        or references != sorted(set(references))
    ):
        raise ArchiveError("receipt entry has invalid values")
    return ArchiveEntry(path, size_bytes, digest, disposition, tuple(references))


def _entry_object(entry: ArchiveEntry) -> dict[str, object]:
    return {
        "path": entry.path,
        "size_bytes": entry.size_bytes,
        "sha256": entry.sha256,
        "disposition": entry.disposition,
        "reference_sources": list(entry.reference_sources),
    }


def _manifest_source_commit(archive_root: Path) -> str:
    try:
        manifest = json.loads((archive_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArchiveError(f"cannot read verified manifest source commit: {error}") from error
    source_commit = manifest.get("source_commit") if isinstance(manifest, dict) else None
    if not isinstance(source_commit, str) or _COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ArchiveError("verified manifest has invalid source commit")
    return source_commit


def _receipt_object(receipt: CurationReceipt) -> dict[str, object]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_commit": receipt.source_commit,
        "manifest": MANIFEST_NAME,
        "manifest_sha256": receipt.manifest_sha256,
        "bundle": BUNDLE_NAME,
        "bundle_sha256": receipt.bundle_sha256,
        "entries": [_entry_object(entry) for entry in receipt.entries],
        "keep_count": receipt.keep_count,
        "drop_count": receipt.drop_count,
    }


def _receipt_from_recovery(recovery: ArchiveReceipt) -> CurationReceipt:
    entries = tuple(recovery.entries)
    return CurationReceipt(
        source_commit=recovery.source_commit,
        manifest_sha256=recovery.manifest_sha256,
        bundle_sha256=recovery.bundle_sha256,
        entries=entries,
        keep_count=sum(entry.disposition == "KEEP" for entry in entries),
        drop_count=sum(entry.disposition == "DROP" for entry in entries),
    )


def load_and_verify_receipt(archive_root: Path) -> CurationReceipt:
    """Verify the complete package against a strict receipt before restoration."""

    archive_root = Path(archive_root)
    entries = load_and_verify_manifest(archive_root)
    manifest_digest = sha256_file(archive_root / MANIFEST_NAME)
    source_commit = _manifest_source_commit(archive_root)
    receipt = _canonical_receipt_object(archive_root / RECEIPT_NAME)
    if set(receipt) != _RECEIPT_KEYS:
        raise ArchiveError("receipt has invalid fields")
    raw_entries = receipt["entries"]
    if not isinstance(raw_entries, list):
        raise ArchiveError("receipt entries must be a list")
    receipt_entries = tuple(_receipt_entry(entry) for entry in raw_entries)
    keep_count = receipt["keep_count"]
    drop_count = receipt["drop_count"]
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != RECEIPT_SCHEMA_VERSION
        or receipt["manifest"] != MANIFEST_NAME
        or receipt["bundle"] != BUNDLE_NAME
        or not isinstance(receipt["source_commit"], str)
        or _COMMIT_PATTERN.fullmatch(receipt["source_commit"]) is None
        or not isinstance(receipt["manifest_sha256"], str)
        or _SHA256_PATTERN.fullmatch(receipt["manifest_sha256"]) is None
        or not isinstance(receipt["bundle_sha256"], str)
        or _SHA256_PATTERN.fullmatch(receipt["bundle_sha256"]) is None
        or type(keep_count) is not int
        or keep_count < 0
        or type(drop_count) is not int
        or drop_count < 0
    ):
        raise ArchiveError("receipt has invalid values")
    bundle_path = archive_root / BUNDLE_NAME
    if bundle_path.is_symlink() or not bundle_path.is_file():
        raise ArchiveError("receipt bundle is missing or unsupported")
    if receipt["source_commit"] != source_commit:
        raise ArchiveError("receipt source commit does not match manifest")
    if receipt["manifest_sha256"] != manifest_digest:
        raise ArchiveError("receipt manifest SHA-256 does not match manifest")
    if receipt_entries != entries:
        raise ArchiveError("receipt entries do not match manifest")
    if keep_count != sum(entry.disposition == "KEEP" for entry in entries):
        raise ArchiveError("receipt KEEP count does not match manifest")
    if drop_count != sum(entry.disposition == "DROP" for entry in entries):
        raise ArchiveError("receipt DROP count does not match manifest")
    if receipt["bundle_sha256"] != sha256_file(bundle_path):
        raise ArchiveError("receipt bundle SHA-256 does not match bundle")
    return CurationReceipt(
        source_commit=receipt["source_commit"],
        manifest_sha256=receipt["manifest_sha256"],
        bundle_sha256=receipt["bundle_sha256"],
        entries=receipt_entries,
        keep_count=keep_count,
        drop_count=drop_count,
    )


def _runbook(owner_project_root: str, source_commit: str) -> str:
    if not isinstance(source_commit, str) or _COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ArchiveError("runbook requires a canonical source commit")
    owner = _power_shell_literal(owner_project_root)
    lines = (
        "$ErrorActionPreference = 'Stop'",
        f"$OwnerProjectRoot = '{owner}'",
        f"$ExpectedSourceCommit = '{source_commit}'",
        (
            "if (-not (Test-Path -LiteralPath $OwnerProjectRoot -PathType Container)) { "
            "throw 'Owner project root is not a directory. STOP.' }"
        ),
        "Set-Location -LiteralPath $OwnerProjectRoot",
        "$GitStatus = @(git status --porcelain=v1 --untracked-files=all)",
        (
            'if ($LASTEXITCODE -ne 0) { throw "git status failed with exit code '
            '$LASTEXITCODE. STOP." }'
        ),
        (
            'if ($GitStatus.Count -ne 0) { throw "Owner repository is not clean. STOP. '
            'Status:`n$($GitStatus -join [Environment]::NewLine)" }'
        ),
        "$ActualSourceCommit = @(git rev-parse --verify HEAD)",
        (
            'if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed with exit code '
            '$LASTEXITCODE. STOP." }'
        ),
        (
            'if ($ActualSourceCommit.Count -ne 1) { throw "git rev-parse must return '
            'exactly one source commit line. STOP." }'
        ),
        (
            "if ($ActualSourceCommit[0] -cne $ExpectedSourceCommit) { throw "
            "'Owner repository HEAD does not match archived source commit. STOP.' }"
        ),
        "uvx git-filter-repo --version",
        (
            'if ($LASTEXITCODE -ne 0) { throw "git-filter-repo availability check failed '
            'with exit code $LASTEXITCODE. STOP." }'
        ),
        "uvx git-filter-repo --path reports/figures/ --invert-paths --force",
        (
            'if ($LASTEXITCODE -ne 0) { throw "git-filter-repo rewrite failed with exit code '
            '$LASTEXITCODE. STOP." }'
        ),
        (
            "Write-Host 'STOP: Stage 1 history rewrite finished. Do not restore, stage, or "
            "commit. Report the full output to the controller and wait for the Task 7 "
            "read-only checkpoint.'"
        ),
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


def _private_stage(destination: Path) -> tuple[Path, Path]:
    """Atomically reserve an invocation-owned root and its unpublished package path."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=".cs-", dir=destination.parent))
    return root, root / "p"


def _remove_private_stage(root: Path) -> None:
    if not _path_exists(root):
        return
    if root.is_dir() and not root.is_symlink():
        shutil.rmtree(root)
    else:
        root.unlink()


def _publish_complete_stage(stage: Path, destination: Path) -> None:
    if _path_exists(destination):
        raise FileExistsError(f"destination already exists: {destination}")
    try:
        os.rename(stage, destination)
    except OSError as error:
        if _path_exists(destination):
            raise FileExistsError(f"destination already exists: {destination}") from error
        raise ArchiveError(f"cannot publish verified recovery package: {error}") from error


def archive(project_root: Path, destination: Path, owner_project_root: str) -> None:
    """Build a recovery package and write owner instructions after every verification."""

    if _path_exists(destination):
        raise FileExistsError(f"destination already exists: {destination}")
    plan = plan_figure_curation(project_root, tracked_files(project_root))
    stage_root, stage = _private_stage(destination)
    published = False
    try:
        recovery = create_recovery_package(project_root, stage, plan)
        published_bundle = stage / BUNDLE_NAME
        recovery.bundle_path.rename(published_bundle)
        _verify_renamed_bundle(project_root, published_bundle, recovery.bundle_sha256)

        receipt = _receipt_from_recovery(recovery)
        _write_canonical_json(stage / RECEIPT_NAME, _receipt_object(receipt))
        (stage / RUNBOOK_NAME).write_text(
            _runbook(owner_project_root, recovery.source_commit), encoding="utf-8", newline="\n"
        )
        verified_receipt = load_and_verify_receipt(stage)
        _verify_renamed_bundle(project_root, published_bundle, verified_receipt.bundle_sha256)
        if verified_receipt != receipt:
            raise ArchiveError("staged receipt differs from the verified recovery package")
        _publish_complete_stage(stage, destination)
        published = True
    finally:
        if not published:
            _remove_private_stage(stage_root)
    _remove_private_stage(stage_root)

    print(
        f"archive={destination} source_commit={receipt.source_commit} "
        f"KEEP={receipt.keep_count} DROP={receipt.drop_count} "
        f"manifest_sha256={receipt.manifest_sha256} "
        f"bundle_sha256={receipt.bundle_sha256}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a verified repository curation recovery archive; never rewrite history."
    )
    parser.add_argument("--project-root", default=PROJECT_ROOT, type=Path)
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
            arguments.project_root,
            Path(arguments.destination),
            arguments.owner_project_root,
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
