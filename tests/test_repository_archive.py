"""Recovery archives must be complete, deterministic, and fail closed."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.release import repository_archive
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


def test_recovery_package_bundle_failure_leaves_no_published_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "recovery"
    real_create_bundle = repository_archive.create_verified_git_bundle

    def fail_bundle(_project_root: Path, _bundle_path: Path) -> str:
        raise ArchiveError("injected bundle failure")

    monkeypatch.setattr(repository_archive, "create_verified_git_bundle", fail_bundle)
    with pytest.raises(ArchiveError, match="injected bundle failure"):
        create_recovery_package(project.root, destination, project.plan)

    assert not destination.exists()
    monkeypatch.setattr(repository_archive, "create_verified_git_bundle", real_create_bundle)
    assert create_recovery_package(project.root, destination, project.plan).bundle_path.is_file()


def test_restore_copy_failure_leaves_no_final_figures_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    _write(project.root, "reports/figures/second-keep.png", b"second")
    plan = project.plan + (
        FigureDisposition(
            "reports/figures/second-keep.png",
            6,
            True,
            (FigureReference("README.md", 2),),
        ),
    )
    archive = tmp_path / "archive"
    create_verified_archive(project.root, archive, plan)
    restored_root = tmp_path / "restored"
    real_copy = repository_archive.shutil.copyfileobj
    calls = 0

    def fail_second_copy(source: object, target: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected restore failure")
        real_copy(source, target)

    monkeypatch.setattr(repository_archive.shutil, "copyfileobj", fail_second_copy)
    with pytest.raises(OSError, match="injected restore failure"):
        restore_keep_files(restored_root, archive)

    assert not (restored_root / "reports/figures").exists()
    reports = restored_root / "reports"
    assert not reports.exists() or not list(reports.glob(".figures.staging-*"))


def test_recovery_receipt_uses_the_single_verified_manifest_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    real_loads = repository_archive.json.loads
    calls = 0

    def allow_one_parse(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("manifest was reopened with an unverified parser")
        return real_loads(*args, **kwargs)

    monkeypatch.setattr(repository_archive.json, "loads", allow_one_parse)

    receipt = create_recovery_package(project.root, tmp_path / "recovery", project.plan)

    assert receipt.source_commit == project.commit
    assert calls == 1


def test_original_source_alias_is_rejected_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")

    monkeypatch.setattr(
        repository_archive,
        "_is_path_alias",
        lambda path: path.name == "keep.png",
        raising=False,
    )

    with pytest.raises(ArchiveError, match="alias"):
        create_verified_archive(project.root, tmp_path / "archive", (project.plan[1],))


def test_duplicate_resolved_source_targets_are_rejected(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    alias = FigureDisposition(
        "reports/figures/KEEP.png",
        4,
        True,
        (FigureReference("README.md", 1),),
    )

    with pytest.raises(ArchiveError, match="duplicate resolved source"):
        create_verified_archive(
            project.root,
            tmp_path / "archive",
            (project.plan[1], alias),
        )


def test_archive_rejects_non_windows_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "outputs" / "archive"
    monkeypatch.setattr(repository_archive, "_is_windows_native", lambda: False, raising=False)

    with pytest.raises(ArchiveError, match="Windows"):
        create_verified_archive(project.root, destination, project.plan)

    assert not destination.parent.exists()


def test_bundle_rejects_non_windows_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    bundle_path = tmp_path / "outputs" / "repository.bundle"
    monkeypatch.setattr(repository_archive, "_is_windows_native", lambda: False, raising=False)

    with pytest.raises(ArchiveError, match="Windows"):
        create_verified_git_bundle(project.root, bundle_path)

    assert not bundle_path.parent.exists()


def test_package_rejects_non_windows_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "outputs" / "recovery"
    monkeypatch.setattr(repository_archive, "_is_windows_native", lambda: False, raising=False)

    with pytest.raises(ArchiveError, match="Windows"):
        create_recovery_package(project.root, destination, project.plan)

    assert not destination.parent.exists()


def test_restore_rejects_non_windows_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    archive = tmp_path / "archive"
    create_verified_archive(project.root, archive, project.plan)
    restored_root = tmp_path / "outputs" / "restored"
    monkeypatch.setattr(repository_archive, "_is_windows_native", lambda: False, raising=False)

    with pytest.raises(ArchiveError, match="Windows"):
        restore_keep_files(restored_root, archive)

    assert not restored_root.parent.exists()


def test_post_publication_bundle_digest_failure_removes_only_new_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    bundle_path = tmp_path / "repository.bundle"
    real_digest = repository_archive.sha256_file

    def fail_final_digest(path: Path) -> str:
        if Path(path) == bundle_path:
            return "0" * 64
        return real_digest(path)

    monkeypatch.setattr(repository_archive, "sha256_file", fail_final_digest)

    with pytest.raises(ArchiveError, match="published Git bundle SHA-256 mismatch"):
        create_verified_git_bundle(project.root, bundle_path)

    assert not bundle_path.exists()


def test_bundle_digest_failure_does_not_remove_a_replacement_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    bundle_path = tmp_path / "repository.bundle"
    real_digest = repository_archive.sha256_file

    def replace_before_failed_digest(path: Path) -> str:
        if Path(path) == bundle_path:
            bundle_path.unlink()
            bundle_path.write_bytes(b"replacement owned elsewhere")
            return "0" * 64
        return real_digest(path)

    monkeypatch.setattr(repository_archive, "sha256_file", replace_before_failed_digest)

    with pytest.raises(ArchiveError, match="published Git bundle SHA-256 mismatch"):
        create_verified_git_bundle(project.root, bundle_path)

    assert bundle_path.read_bytes() == b"replacement owned elsewhere"


def test_recovery_bundle_must_contain_manifest_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "recovery"
    real_create_bundle = repository_archive.create_verified_git_bundle

    def replace_only_ref_then_bundle(project_root: Path, bundle_path: Path) -> str:
        tree = subprocess.run(
            ["git", "mktree"],
            cwd=project_root,
            input="",
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        replacement = _git(
            project_root,
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit-tree",
            tree,
            "-m",
            "replacement root",
        )
        _git(project_root, "update-ref", "HEAD", replacement)
        return real_create_bundle(project_root, bundle_path)

    monkeypatch.setattr(
        repository_archive,
        "create_verified_git_bundle",
        replace_only_ref_then_bundle,
    )

    with pytest.raises(ArchiveError, match="does not contain source commit"):
        create_recovery_package(project.root, destination, project.plan)

    assert not destination.exists()
