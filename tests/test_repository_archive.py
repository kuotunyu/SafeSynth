"""Recovery archives must be complete, deterministic, and fail closed."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.release.repository_archive import (
    ArchiveError,
    create_recovery_package,
    create_verified_archive,
    create_verified_git_bundle,
    load_and_verify_manifest,
    restore_keep_files,
    sha256_file,
)
from src.release.repository_curation import FigureDisposition, FigureReference


@dataclass(frozen=True)
class _Project:
    root: Path
    plan: tuple[FigureDisposition, ...]
    commit: str


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _write(root: Path, relative_path: str, contents: bytes) -> None:
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contents)


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        message,
    )
    return _git(root, "rev-parse", "HEAD")


def _project_with_keep_and_drop(root: Path) -> _Project:
    _write(root, "README.md", b"![keep](reports/figures/keep.png)\n")
    _write(root, "reports/figures/keep.png", b"keep")
    _write(root, "reports/figures/drop.png", b"drop")
    _git(root, "init", "-q")
    commit = _commit_all(root, "fixture")
    return _Project(
        root=root,
        plan=(
            FigureDisposition(
                path="reports/figures/drop.png",
                size_bytes=4,
                keep=False,
                references=(),
            ),
            FigureDisposition(
                path="reports/figures/keep.png",
                size_bytes=4,
                keep=True,
                references=(FigureReference("README.md", 1),),
            ),
        ),
        commit=commit,
    )


def test_archive_copies_every_entry_and_hash_verifies(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"

    manifest_path = create_verified_archive(project.root, destination, project.plan)
    entries = load_and_verify_manifest(destination)

    assert manifest_path == destination / "figure_manifest.json"
    assert {entry.path for entry in entries} == {
        "reports/figures/keep.png",
        "reports/figures/drop.png",
    }
    assert (destination / "figures/reports/figures/keep.png").read_bytes() == b"keep"
    assert (destination / "figures/reports/figures/drop.png").read_bytes() == b"drop"


def test_manifest_is_canonical_and_records_source_commit(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"

    manifest_path = create_verified_archive(project.root, destination, project.plan)

    manifest_bytes = manifest_path.read_bytes()
    parsed = json.loads(manifest_bytes)
    assert manifest_bytes.endswith(b"\n")
    assert manifest_bytes == (
        json.dumps(parsed, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert parsed["schema_version"] == 1
    assert parsed["source_commit"] == project.commit
    assert [entry["path"] for entry in parsed["entries"]] == [
        "reports/figures/drop.png",
        "reports/figures/keep.png",
    ]
    assert parsed["entries"][1]["reference_sources"] == ["README.md:1"]


def test_archive_tampering_is_a_blocking_failure(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"
    create_verified_archive(project.root, destination, project.plan)
    (destination / "figures" / "reports" / "figures" / "drop.png").write_bytes(b"changed")

    with pytest.raises(ArchiveError, match="SHA-256 mismatch"):
        load_and_verify_manifest(destination)


def test_missing_archived_file_is_a_blocking_failure(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"
    create_verified_archive(project.root, destination, project.plan)
    (destination / "figures/reports/figures/drop.png").unlink()

    with pytest.raises(ArchiveError, match="archived file set does not match manifest"):
        load_and_verify_manifest(destination)


def test_extra_archived_file_is_a_blocking_failure(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"
    create_verified_archive(project.root, destination, project.plan)
    _write(destination / "figures", "unexpected.png", b"unexpected")

    with pytest.raises(ArchiveError, match="archived file set does not match manifest"):
        load_and_verify_manifest(destination)


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "archive"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        create_verified_archive(tmp_path / "project", destination, ())


def test_missing_source_is_rejected_without_publishing_archive(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    (project.root / "reports/figures/drop.png").unlink()
    destination = tmp_path / "archive"

    with pytest.raises(ArchiveError, match="missing source"):
        create_verified_archive(project.root, destination, project.plan)

    assert not destination.exists()


def test_source_size_mismatch_is_rejected(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    wrong_plan = (
        FigureDisposition("reports/figures/drop.png", 99, False, ()),
    )

    with pytest.raises(ArchiveError, match="source size mismatch"):
        create_verified_archive(project.root, tmp_path / "archive", wrong_plan)


def test_duplicate_source_paths_are_rejected(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")

    with pytest.raises(ArchiveError, match="duplicate archive path"):
        create_verified_archive(
            project.root,
            tmp_path / "archive",
            (project.plan[0], project.plan[0]),
        )


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.png", "/absolute.png", r"C:\outside.png", "reports/../outside.png"],
)
def test_unsafe_source_paths_are_rejected(tmp_path: Path, unsafe_path: str) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    unsafe = FigureDisposition(unsafe_path, 4, False, ())

    with pytest.raises(ArchiveError, match="unsafe repository path"):
        create_verified_archive(project.root, tmp_path / "archive", (unsafe,))


def test_manifest_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    archive = tmp_path / "archive"
    manifest_path = create_verified_archive(project.root, archive, project.plan)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveError, match="manifest schema"):
        load_and_verify_manifest(archive)


def test_manifest_path_escape_is_rejected(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    archive = tmp_path / "archive"
    manifest_path = create_verified_archive(project.root, archive, project.plan)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["path"] = "../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArchiveError, match="unsafe repository path"):
        load_and_verify_manifest(archive)


def test_restore_copies_only_verified_keep_files(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    archive = tmp_path / "archive"
    create_verified_archive(project.root, archive, project.plan)
    restored_root = tmp_path / "restored"

    restored = restore_keep_files(restored_root, archive)

    assert restored == ("reports/figures/keep.png",)
    assert (restored_root / "reports/figures/keep.png").read_bytes() == b"keep"
    assert not (restored_root / "reports/figures/drop.png").exists()


def test_restore_refuses_existing_keep_destination_before_copying(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    archive = tmp_path / "archive"
    create_verified_archive(project.root, archive, project.plan)
    restored_root = tmp_path / "restored"
    _write(restored_root, "reports/figures/keep.png", b"do not overwrite")

    with pytest.raises(FileExistsError):
        restore_keep_files(restored_root, archive)

    assert (restored_root / "reports/figures/keep.png").read_bytes() == b"do not overwrite"


def test_restore_verifies_all_entries_before_copying_keep_files(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    archive = tmp_path / "archive"
    create_verified_archive(project.root, archive, project.plan)
    (archive / "figures/reports/figures/drop.png").write_bytes(b"tampered")
    restored_root = tmp_path / "restored"

    with pytest.raises(ArchiveError, match="SHA-256 mismatch"):
        restore_keep_files(restored_root, archive)

    assert not restored_root.exists()


def test_verified_git_bundle_contains_two_commit_repository(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    _write(project.root, "second.txt", b"second commit")
    second_commit = _commit_all(project.root, "second")
    bundle_path = tmp_path / "repository.bundle"

    digest = create_verified_git_bundle(project.root, bundle_path)

    assert bundle_path.is_file()
    assert digest == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    verified = subprocess.run(
        ["git", "bundle", "verify", str(bundle_path)],
        cwd=project.root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    heads = _git(project.root, "bundle", "list-heads", str(bundle_path))
    assert second_commit in heads
    restored_repository = tmp_path / "restored-repository"
    subprocess.run(
        ["git", "clone", "-q", str(bundle_path), str(restored_repository)],
        check=True,
    )
    assert _git(restored_repository, "rev-list", "--count", "HEAD") == "2"
    assert _git(restored_repository, "rev-parse", "HEAD^") == project.commit


def test_existing_bundle_is_never_overwritten(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    bundle_path = tmp_path / "repository.bundle"
    bundle_path.write_bytes(b"do not overwrite")

    with pytest.raises(FileExistsError):
        create_verified_git_bundle(project.root, bundle_path)

    assert bundle_path.read_bytes() == b"do not overwrite"


def test_recovery_package_returns_verified_archive_and_bundle(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "recovery"

    receipt = create_recovery_package(project.root, destination, project.plan)

    assert receipt.source_commit == project.commit
    assert receipt.manifest_path == destination / "figure_manifest.json"
    assert receipt.manifest_sha256 == sha256_file(receipt.manifest_path)
    assert receipt.bundle_path == destination / "repository.bundle"
    assert receipt.bundle_sha256 == sha256_file(receipt.bundle_path)
    assert receipt.entries == load_and_verify_manifest(destination)
