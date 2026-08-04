"""Recovery archives must be complete, deterministic, and fail closed."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from scripts import archive_repository_curation as archive_command
from scripts import restore_curated_figures as restore_command
from src.release import repository_archive
from src.release.figure_evidence import FigureEvidenceError, FigureManifestApproval
from src.release.markdown_links import RepositoryLinkError
from src.release.repository_archive import (
    ArchiveError,
    _create_recovery_package,
    create_verified_archive,
    create_verified_git_bundle,
    load_and_verify_manifest,
    restore_keep_files,
    sha256_file,
)
from src.release.repository_curation import FigureDisposition, FigureReference

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_COMMAND = WORKSPACE_ROOT / "scripts" / "archive_repository_curation.py"
RESTORE_COMMAND = WORKSPACE_ROOT / "scripts" / "restore_curated_figures.py"
EXPECTED_SOURCE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_TEST_APPROVAL = FigureManifestApproval(
    manifest_sha256="668147987292b42323df710e8be3846fc1f3fe36a000eee11daeba9131acbc8c",
    source_commit=EXPECTED_SOURCE_COMMIT,
    total=2,
    keep=1,
    drop=1,
)


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


def _run_command(command: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    if command == ARCHIVE_COMMAND:
        parsed = archive_command._parser().parse_args(arguments)
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                archive_command._archive(
                    parsed.project_root,
                    Path(parsed.destination),
                    parsed.owner_project_root,
                    _TEST_APPROVAL,
                )
        except (
            ArchiveError,
            FigureEvidenceError,
            FileExistsError,
            OSError,
            RepositoryLinkError,
            subprocess.CalledProcessError,
        ) as error:
            print(f"error: {error}", file=stderr)
            return subprocess.CompletedProcess(
                [sys.executable, str(command), *arguments], 1, stdout.getvalue(), stderr.getvalue()
            )
        return subprocess.CompletedProcess(
            [sys.executable, str(command), *arguments], 0, stdout.getvalue(), stderr.getvalue()
        )
    return subprocess.run(
        [sys.executable, str(command), *arguments],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _project_with_keep_and_drop(root: Path) -> _Project:
    _write(root, "README.md", b"![keep](reports/figures/keep.png)\n")
    _write(root, "reports/figures/keep.png", b"keep")
    _write(root, "reports/figures/drop.png", b"drop")
    _git(root, "init", "-q")
    _write(
        root,
        "reports/figure_curation_manifest.json",
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "source_commit": EXPECTED_SOURCE_COMMIT,
                    "entries": [
                        {
                            "path": "reports/figures/drop.png",
                            "size_bytes": 4,
                            "sha256": hashlib.sha256(b"drop").hexdigest(),
                            "disposition": "DROP",
                            "reference_sources": [],
                        },
                        {
                            "path": "reports/figures/keep.png",
                            "size_bytes": 4,
                            "sha256": hashlib.sha256(b"keep").hexdigest(),
                            "disposition": "KEEP",
                            "reference_sources": ["README.md:1"],
                        },
                    ],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )
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


def _create_command_archive(project: _Project, destination: Path) -> None:
    completed = _run_command(
        ARCHIVE_COMMAND,
        "--project-root",
        str(project.root),
        "--destination",
        str(destination),
        "--owner-project-root",
        str(destination.parent / "owner"),
    )
    assert completed.returncode == 0, completed.stderr


def _archive_with_test_approval(project_root: Path, destination: Path, owner_project_root: str) -> None:
    """Exercise archive internals with the minimal reviewed fixture contract."""

    archive_command._archive(project_root, destination, owner_project_root, _TEST_APPROVAL)


def test_raw_recovery_package_builder_is_internal_to_the_guarded_archive_command() -> None:
    """Catch a public raw-builder alias that can bypass manifest and clean-tree gates."""

    assert not hasattr(repository_archive, "create_recovery_package")
    assert archive_command._create_recovery_package is repository_archive._create_recovery_package


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

    receipt = _create_recovery_package(project.root, destination, project.plan)

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
        _create_recovery_package(project.root, destination, project.plan)

    assert not destination.exists()
    monkeypatch.setattr(repository_archive, "create_verified_git_bundle", real_create_bundle)
    assert _create_recovery_package(project.root, destination, project.plan).bundle_path.is_file()


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

    receipt = _create_recovery_package(project.root, tmp_path / "recovery", project.plan)

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
        _create_recovery_package(project.root, destination, project.plan)

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


def test_bundle_verification_failure_leaves_no_final_or_staging_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    bundle_path = tmp_path / "repository.bundle"
    real_run = repository_archive.subprocess.run

    def fail_bundle_verify(command: list[str], *args: object, **kwargs: object) -> object:
        if command[:3] == ["git", "bundle", "verify"]:
            return subprocess.CompletedProcess(command, 1, "", "injected verify failure")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(repository_archive.subprocess, "run", fail_bundle_verify)

    with pytest.raises(ArchiveError, match="injected verify failure"):
        create_verified_git_bundle(project.root, bundle_path)

    assert not bundle_path.exists()
    assert not list(tmp_path.glob(".repository.bundle.staging-*"))


def test_bundle_digest_failure_before_rename_leaves_no_final_or_staging_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    bundle_path = tmp_path / "repository.bundle"
    real_digest = repository_archive.sha256_file

    def fail_staged_digest(path: Path) -> str:
        if Path(path).name == bundle_path.name:
            raise ArchiveError("injected staged digest failure")
        return real_digest(path)

    monkeypatch.setattr(repository_archive, "sha256_file", fail_staged_digest)

    with pytest.raises(ArchiveError, match="injected staged digest failure"):
        create_verified_git_bundle(project.root, bundle_path)

    assert not bundle_path.exists()
    assert not list(tmp_path.glob(".repository.bundle.staging-*"))


def test_successful_bundle_publication_never_reopens_the_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    bundle_path = tmp_path / "repository.bundle"
    real_digest = repository_archive.sha256_file
    digested_paths: list[Path] = []

    def reject_final_path_digest(path: Path) -> str:
        path = Path(path)
        digested_paths.append(path)
        if path == bundle_path:
            raise AssertionError("published bundle path was reopened")
        return real_digest(path)

    monkeypatch.setattr(repository_archive, "sha256_file", reject_final_path_digest)

    digest = create_verified_git_bundle(project.root, bundle_path)

    assert bundle_path.is_file()
    assert len(digested_paths) == 1
    assert digested_paths[0] != bundle_path
    assert digest == hashlib.sha256(bundle_path.read_bytes()).hexdigest()


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
        _create_recovery_package(project.root, destination, project.plan)

    assert not destination.exists()


def test_archive_command_refuses_an_existing_destination(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"
    destination.mkdir()

    completed = _run_command(
        ARCHIVE_COMMAND,
        "--project-root",
        str(project.root),
        "--destination",
        str(destination),
        "--owner-project-root",
        str(tmp_path / "owner"),
    )

    assert completed.returncode != 0
    assert "already exists" in completed.stderr


def test_archive_command_refuses_unresolved_figure_links(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    _write(
        project.root,
        "README.md",
        b"![keep](reports/figures/keep.png)\n![missing](reports/figures/missing.png)\n",
    )
    _commit_all(project.root, "add unresolved link")
    destination = tmp_path / "archive"

    completed = _run_command(
        ARCHIVE_COMMAND,
        "--project-root",
        str(project.root),
        "--destination",
        str(destination),
        "--owner-project-root",
        str(tmp_path / "owner"),
    )

    assert completed.returncode != 0
    assert "local Markdown destination" in completed.stderr
    assert not destination.exists()


def test_archive_command_writes_verified_receipt_and_exact_owner_runbook(
    tmp_path: Path,
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"
    owner_root = tmp_path / "owner project"
    _write(owner_root, "owner-sentinel.txt", b"unchanged")

    completed = _run_command(
        ARCHIVE_COMMAND,
        "--project-root",
        str(project.root),
        "--destination",
        str(destination),
        "--owner-project-root",
        str(owner_root),
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((destination / "archive_receipt.json").read_text(encoding="utf-8"))
    assert receipt["source_commit"] == project.commit
    assert receipt["manifest_sha256"] == sha256_file(destination / "figure_manifest.json")
    bundle = destination / "SafeSynth-pre-filter-repo.bundle"
    assert receipt["bundle_sha256"] == sha256_file(bundle)
    assert receipt["keep_count"] == 1
    assert receipt["drop_count"] == 1
    assert not (destination / "repository.bundle").exists()
    verified = subprocess.run(
        ["git", "bundle", "verify", str(bundle)],
        cwd=project.root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    runbook = (destination / "OWNER_HISTORY_REWRITE_RUNBOOK.txt").read_text(encoding="utf-8")
    assert runbook == "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            f"$OwnerProjectRoot = '{owner_root}'",
            f"$ExpectedSourceCommit = '{project.commit}'",
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
            "",
        )
    )
    assert "restore_curated_figures" not in runbook
    assert "git add" not in runbook
    assert "git commit" not in runbook
    assert owner_root.joinpath("owner-sentinel.txt").read_bytes() == b"unchanged"
    assert completed.stdout.index("KEEP=1") < completed.stdout.index("DROP=1")


def _execute_runbook_with_fake_native_commands(
    tmp_path: Path,
    *,
    expected_commit: str = EXPECTED_SOURCE_COMMIT,
    status_exit: int = 0,
    rev_parse_exit: int = 0,
    rev_parse_lines: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess[str], list[str], str, Path]:
    owner_root = tmp_path / "owner's project"
    owner_root.mkdir()
    fake_commands = tmp_path / "fake-commands"
    fake_commands.mkdir()
    command_log = tmp_path / "commands.log"
    _write(
        fake_commands,
        "git.cmd",
        (
            '@echo off\r\necho git %*>>"%COMMAND_LOG%"\r\n'
            'if "%1"=="status" exit /b %STATUS_EXIT%\r\n'
            'if "%1"=="rev-parse" (\r\n'
            '  if defined REV_PARSE_LINE_1 echo %REV_PARSE_LINE_1%\r\n'
            '  if defined REV_PARSE_LINE_2 echo %REV_PARSE_LINE_2%\r\n'
            '  exit /b %REV_PARSE_EXIT%\r\n'
            ')\r\nexit /b 99\r\n'
        ).encode("ascii"),
    )
    _write(
        fake_commands,
        "uvx.cmd",
        b'@echo off\r\necho uvx %*>>"%COMMAND_LOG%"\r\nexit /b 0\r\n',
    )
    runbook = tmp_path / "runbook.ps1"
    runbook_text = archive_command._runbook(str(owner_root), expected_commit)
    runbook.write_bytes(runbook_text.encode("utf-8"))
    environment = os.environ.copy()
    environment["COMMAND_LOG"] = str(command_log)
    environment["PATH"] = f"{fake_commands}{os.pathsep}{environment['PATH']}"
    environment["STATUS_EXIT"] = str(status_exit)
    environment["REV_PARSE_EXIT"] = str(rev_parse_exit)
    for index, line in enumerate(rev_parse_lines, start=1):
        environment[f"REV_PARSE_LINE_{index}"] = line
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(runbook),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    commands = (
        command_log.read_text(encoding="utf-8").splitlines() if command_log.exists() else []
    )
    return completed, commands, runbook_text, owner_root


def test_owner_runbook_quotes_root_and_stops_after_native_status_failure(tmp_path: Path) -> None:
    completed, commands, runbook, owner_root = _execute_runbook_with_fake_native_commands(
        tmp_path, status_exit=17
    )

    assert completed.returncode != 0
    assert "git status failed with exit code 17" in completed.stderr
    assert commands == ["git status --porcelain=v1 --untracked-files=all"]
    assert f"$OwnerProjectRoot = '{str(owner_root).replace("'", "''")}'" in runbook
    assert f"$ExpectedSourceCommit = '{EXPECTED_SOURCE_COMMIT}'" in runbook
    assert runbook == archive_command._runbook(str(owner_root), EXPECTED_SOURCE_COMMIT)


def test_owner_runbook_rev_parse_native_failure_never_reaches_uvx(tmp_path: Path) -> None:
    completed, commands, _runbook, _owner = _execute_runbook_with_fake_native_commands(
        tmp_path, rev_parse_exit=23
    )

    assert completed.returncode != 0
    assert "git rev-parse failed with exit code 23" in completed.stderr
    assert commands == [
        "git status --porcelain=v1 --untracked-files=all",
        "git rev-parse --verify HEAD",
    ]


def test_owner_runbook_wrong_clean_head_never_reaches_uvx(tmp_path: Path) -> None:
    completed, commands, _runbook, _owner = _execute_runbook_with_fake_native_commands(
        tmp_path, rev_parse_lines=("fedcba9876543210fedcba9876543210fedcba98",)
    )

    assert completed.returncode != 0
    assert "HEAD does not match archived source commit" in completed.stderr
    assert commands[-1] == "git rev-parse --verify HEAD"
    assert all(not command.startswith("uvx ") for command in commands)


@pytest.mark.parametrize(
    "rev_parse_lines",
    [(), (EXPECTED_SOURCE_COMMIT, "fedcba9876543210fedcba9876543210fedcba98")],
    ids=["empty", "multiline"],
)
def test_owner_runbook_rejects_non_scalar_rev_parse_output(
    tmp_path: Path, rev_parse_lines: tuple[str, ...]
) -> None:
    completed, commands, _runbook, _owner = _execute_runbook_with_fake_native_commands(
        tmp_path, rev_parse_lines=rev_parse_lines
    )

    assert completed.returncode != 0
    assert "exactly one source commit line" in completed.stderr
    assert commands[-1] == "git rev-parse --verify HEAD"
    assert all(not command.startswith("uvx ") for command in commands)


def test_owner_runbook_matching_head_reaches_rewrite_then_stop(tmp_path: Path) -> None:
    completed, commands, runbook, _owner = _execute_runbook_with_fake_native_commands(
        tmp_path, rev_parse_lines=(EXPECTED_SOURCE_COMMIT,)
    )

    assert completed.returncode == 0, completed.stderr
    assert commands == [
        "git status --porcelain=v1 --untracked-files=all",
        "git rev-parse --verify HEAD",
        "uvx git-filter-repo --version",
        "uvx git-filter-repo --path reports/figures/ --invert-paths --force",
    ]
    assert "STOP: Stage 1 history rewrite finished" in completed.stdout
    assert "restore_curated_figures" not in runbook
    assert "git add" not in runbook
    assert "git commit" not in runbook


@pytest.mark.parametrize(
    "unsafe_commit",
    [
        "A" * 40,
        "a" * 39,
        "a" * 40 + "'; Write-Host injected",
    ],
)
def test_owner_runbook_rejects_noncanonical_expected_commit(unsafe_commit: str) -> None:
    with pytest.raises(ArchiveError, match="canonical source commit"):
        archive_command._runbook("C:/owner", unsafe_commit)


def test_restore_command_rejects_tampered_archive_before_copying(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    archive = tmp_path / "archive"
    _create_command_archive(project, archive)
    (archive / "figures/reports/figures/drop.png").write_bytes(b"tampered")
    target = tmp_path / "target"

    completed = _run_command(
        RESTORE_COMMAND, "--project-root", str(target), "--archive", str(archive)
    )

    assert completed.returncode != 0
    assert "SHA-256 mismatch" in completed.stderr
    assert not target.exists()


def test_restore_command_refuses_nonempty_figures_target(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    archive = tmp_path / "archive"
    _create_command_archive(project, archive)
    target = tmp_path / "target"
    _write(target, "reports/figures/do-not-overwrite.png", b"existing")

    completed = _run_command(
        RESTORE_COMMAND, "--project-root", str(target), "--archive", str(archive)
    )

    assert completed.returncode != 0
    assert "must be empty" in completed.stderr
    assert (target / "reports/figures/do-not-overwrite.png").read_bytes() == b"existing"


def test_restore_command_restores_only_keep_files_into_clean_target(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    archive = tmp_path / "archive"
    _create_command_archive(project, archive)
    target = tmp_path / "target"
    (target / "reports/figures").mkdir(parents=True)

    completed = _run_command(
        RESTORE_COMMAND, "--project-root", str(target), "--archive", str(archive)
    )

    assert completed.returncode == 0, completed.stderr
    assert (target / "reports/figures/keep.png").read_bytes() == b"keep"
    assert not (target / "reports/figures/drop.png").exists()
    assert completed.stdout.splitlines() == ["reports/figures/keep.png"]


def test_commands_default_project_root_to_their_script_repository() -> None:
    archive_arguments = archive_command._parser().parse_args(
        ["--destination", "archive", "--owner-project-root", "owner"]
    )
    restore_arguments = restore_command._parser().parse_args(["--archive", "archive"])

    assert archive_arguments.project_root == archive_command.PROJECT_ROOT
    assert restore_arguments.project_root == restore_command.PROJECT_ROOT


def test_restore_command_rejects_canonical_manifest_disposition_tampering(
    tmp_path: Path,
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    archive = tmp_path / "archive"
    owner_root = tmp_path / "owner"
    completed = _run_command(
        ARCHIVE_COMMAND,
        "--project-root",
        str(project.root),
        "--destination",
        str(archive),
        "--owner-project-root",
        str(owner_root),
    )
    assert completed.returncode == 0, completed.stderr
    manifest_path = archive / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["disposition"] = "KEEP"
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    receipt_path = archive / "archive_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = sha256_file(manifest_path)
    receipt_path.write_bytes(
        (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    target = tmp_path / "target"

    completed = _run_command(
        RESTORE_COMMAND, "--project-root", str(target), "--archive", str(archive)
    )

    assert completed.returncode != 0
    assert "receipt entries do not match manifest" in completed.stderr
    assert not target.exists()


def test_archive_command_removes_its_private_stage_when_receipt_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"

    def fail_receipt_write(_path: Path, _value: dict[str, object]) -> None:
        raise OSError("injected receipt write failure")

    monkeypatch.setattr(archive_command, "_write_canonical_json", fail_receipt_write)

    with pytest.raises(OSError, match="injected receipt write failure"):
        _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    assert not destination.exists()
    assert not list(tmp_path.glob(".*"))


def test_archive_private_root_collision_preserves_unowned_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"
    colliding_root = tmp_path / ".cs-collision"
    _write(colliding_root, "sentinel.txt", b"another process owns this")
    owned_root = tmp_path / ".cs-owned"

    monkeypatch.setattr(
        archive_command.tempfile,
        "_get_candidate_names",
        lambda: iter(("collision", "owned")),
    )

    def fail_package(_project_root: Path, stage: Path, _plan: object) -> object:
        assert stage == owned_root / "p"
        raise ArchiveError("injected package failure")

    monkeypatch.setattr(archive_command, "_create_recovery_package", fail_package)

    with pytest.raises(ArchiveError, match="injected package failure"):
        _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    assert colliding_root.joinpath("sentinel.txt").read_bytes() == b"another process owns this"
    assert not owned_root.exists()
    assert not destination.exists()


def test_archive_keyboard_interrupt_cleans_owned_root_without_touching_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"
    colliding_root = tmp_path / ".cs-collision"
    _write(colliding_root, "sentinel.txt", b"another process owns this")
    owned_root = tmp_path / ".cs-owned"

    monkeypatch.setattr(
        archive_command.tempfile,
        "_get_candidate_names",
        lambda: iter(("collision", "owned")),
    )

    def interrupt_package(_project_root: Path, stage: Path, _plan: object) -> object:
        assert stage == owned_root / "p"
        raise KeyboardInterrupt

    monkeypatch.setattr(archive_command, "_create_recovery_package", interrupt_package)

    with pytest.raises(KeyboardInterrupt):
        _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    assert colliding_root.joinpath("sentinel.txt").read_bytes() == b"another process owns this"
    assert not owned_root.exists()
    assert not destination.exists()


def _assert_unpublished(destination: Path) -> None:
    """The archive boundary must expose neither a package nor its private staging root."""

    assert not destination.exists()
    assert not list(destination.parent.glob(".cs-*"))


def test_archive_command_rejects_missing_tracked_canonical_manifest_before_staging(
    tmp_path: Path,
) -> None:
    """Catch publication proceeding when the source lacks its Git-tracked evidence index."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    (project.root / "reports/figure_curation_manifest.json").unlink()

    completed = _run_command(
        ARCHIVE_COMMAND,
        "--project-root",
        str(project.root),
        "--destination",
        str(destination),
        "--owner-project-root",
        str(tmp_path / "owner"),
    )

    assert completed.returncode != 0
    assert "manifest" in completed.stderr.lower()
    _assert_unpublished(destination)


def test_archive_command_rejects_plan_reference_mismatch_before_staging(tmp_path: Path) -> None:
    """Catch a Markdown reference moving without a matching canonical-plan update."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    _write(project.root, "README.md", b"\n![keep](reports/figures/keep.png)\n")
    _commit_all(project.root, "move reference")

    completed = _run_command(
        ARCHIVE_COMMAND,
        "--project-root",
        str(project.root),
        "--destination",
        str(destination),
        "--owner-project-root",
        str(tmp_path / "owner"),
    )

    assert completed.returncode != 0
    assert "references differ from manifest" in completed.stderr
    _assert_unpublished(destination)


def test_archive_rejects_plan_disposition_mismatch_before_staging(
    tmp_path: Path,
) -> None:
    """Catch a canonical KEEP/DROP decision changing while source bytes remain exact."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    manifest_path = project.root / "reports/figure_curation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["disposition"] = "KEEP"
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    _commit_all(project.root, "change disposition")

    source_commit, _entries, manifest_sha256 = repository_archive.load_manifest_commitments(manifest_path)
    changed_approval = FigureManifestApproval(
        manifest_sha256=manifest_sha256,
        source_commit=source_commit,
        total=2,
        keep=2,
        drop=0,
    )

    with pytest.raises(FigureEvidenceError, match="disposition differs from manifest"):
        archive_command._archive(
            project.root, destination, str(tmp_path / "owner"), changed_approval
        )
    _assert_unpublished(destination)


def test_archive_command_rejects_source_byte_mismatch_before_staging(tmp_path: Path) -> None:
    """Catch a source figure changing after the canonical evidence index was frozen."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    _write(project.root, "reports/figures/drop.png", b"gone")
    _commit_all(project.root, "change source byte")

    completed = _run_command(
        ARCHIVE_COMMAND,
        "--project-root",
        str(project.root),
        "--destination",
        str(destination),
        "--owner-project-root",
        str(tmp_path / "owner"),
    )

    assert completed.returncode != 0
    assert "SHA-256 mismatch" in completed.stderr
    _assert_unpublished(destination)


def test_archive_rejects_produced_entry_tuple_mismatch_and_cleans_private_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch publication accepting package entries other than the tracked evidence tuple."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    original_create = archive_command._create_recovery_package

    def create_with_changed_entries(
        project_root: Path, stage: Path, plan: object
    ) -> repository_archive.ArchiveReceipt:
        recovery = original_create(project_root, stage, plan)
        changed = replace(recovery.entries[0], disposition="KEEP")
        return replace(recovery, entries=(changed, *recovery.entries[1:]))

    monkeypatch.setattr(archive_command, "_create_recovery_package", create_with_changed_entries)

    with pytest.raises(ArchiveError, match="entries differ from tracked canonical manifest"):
        _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    _assert_unpublished(destination)


def test_archive_command_publishes_a_canonical_two_entry_manifest_fixture(tmp_path: Path) -> None:
    """Catch a valid tracked two-entry evidence manifest being rejected by archive binding."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"

    _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    assert archive_command.load_and_verify_receipt(destination).entries == (
        repository_archive.ArchiveEntry(
            "reports/figures/drop.png", 4, hashlib.sha256(b"drop").hexdigest(), "DROP", ()
        ),
        repository_archive.ArchiveEntry(
            "reports/figures/keep.png",
            4,
            hashlib.sha256(b"keep").hexdigest(),
            "KEEP",
            ("README.md:1",),
        ),
    )


@pytest.mark.parametrize(
    ("name", "mutate", "expected_error"),
    [
        (
            "digest",
            lambda manifest: manifest["entries"][0].update({"sha256": "0" * 64}),
            "manifest SHA-256 differs",
        ),
        (
            "historical source",
            lambda manifest: manifest.update({"source_commit": "a" * 40}),
            "source commit differs",
        ),
        (
            "counts",
            lambda manifest: manifest["entries"][0].update({"disposition": "KEEP"}),
            "counts differ",
        ),
    ],
)
def test_archive_rejects_clean_tracked_manifest_outside_approved_contract_before_staging(
    tmp_path: Path, name: str, mutate: object, expected_error: str
) -> None:
    """Catch an approved-looking tracked manifest drifting in the named commitment."""

    project = _project_with_keep_and_drop(tmp_path / name)
    destination = tmp_path / "archive"
    manifest_path = project.root / "reports/figure_curation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    _commit_all(project.root, f"replace {name} commitment")

    with pytest.raises(archive_command.FigureEvidenceError, match=expected_error):
        if name in {"historical source", "counts"}:
            source_commit, _entries, manifest_sha256 = repository_archive.load_manifest_commitments(
                manifest_path
            )
            approval_source = EXPECTED_SOURCE_COMMIT if name == "historical source" else source_commit
            archive_command._archive(
                project.root,
                destination,
                str(tmp_path / "owner"),
                FigureManifestApproval(manifest_sha256, approval_source, 2, 1, 1),
            )
        else:
            archive_command.archive(project.root, destination, str(tmp_path / "owner"))

    _assert_unpublished(destination)


def test_archive_rejects_a_dirty_but_manifest_consistent_source_before_staging(tmp_path: Path) -> None:
    """Catch a package binding HEAD while silently reading a modified working tree."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    _write(project.root, "README.md", b"![keep](reports/figures/keep.png)\n\n")

    with pytest.raises(ArchiveError, match="worktree is not clean"):
        _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    _assert_unpublished(destination)


def test_archive_rejects_a_staged_source_change_before_staging(tmp_path: Path) -> None:
    """Catch a staged change that would otherwise be absent from the recorded HEAD."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    _write(project.root, "README.md", b"![keep](reports/figures/keep.png)\n\n")
    _git(project.root, "add", "README.md")

    with pytest.raises(ArchiveError, match="worktree is not clean"):
        _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    _assert_unpublished(destination)


def test_archive_rejects_an_untracked_source_file_before_staging(tmp_path: Path) -> None:
    """Catch an untracked source file being omitted from a HEAD-bound publication."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    _write(project.root, "untracked-proof.txt", b"must block")

    with pytest.raises(ArchiveError, match="worktree is not clean"):
        _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    _assert_unpublished(destination)


def test_archive_rechecks_cleanliness_before_publication_and_cleans_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a concurrent source mutation that appears after the initial clean-tree gate."""

    project = _project_with_keep_and_drop(tmp_path / "source")
    destination = tmp_path / "archive"
    original_write = archive_command._write_canonical_json

    def write_receipt_then_dirty(path: Path, value: dict[str, object]) -> None:
        original_write(path, value)
        _write(project.root, "concurrent-untracked.txt", b"appeared after archive build")

    monkeypatch.setattr(archive_command, "_write_canonical_json", write_receipt_then_dirty)

    with pytest.raises(ArchiveError, match="worktree is not clean"):
        _archive_with_test_approval(project.root, destination, str(tmp_path / "owner"))

    _assert_unpublished(destination)


_CANONICAL_COMMITMENT_MANIFEST = b'''{
  "entries": [
    {
      "disposition": "DROP",
      "path": "reports/figures/drop.png",
      "reference_sources": [],
      "sha256": "d90ee9ccf6bea1d2942a7b21319338198dec2a746f8a0d0771621f00da2e0864",
      "size_bytes": 4
    },
    {
      "disposition": "KEEP",
      "path": "reports/figures/keep.png",
      "reference_sources": [
        "README.md:1"
      ],
      "sha256": "6ca7ea2feefc88ecb5ed6356ed963f47dc9137f82526fdd25d618ea626d0803f",
      "size_bytes": 4
    }
  ],
  "schema_version": 1,
  "source_commit": "0123456789abcdef0123456789abcdef01234567"
}
'''


def test_load_manifest_commitments_reads_canonical_manifest_without_payload(tmp_path: Path) -> None:
    """Catch a parser regression that requires an adjacent figures payload."""

    manifest_path = tmp_path / "figure_manifest.json"
    manifest_path.write_bytes(_CANONICAL_COMMITMENT_MANIFEST)

    source_commit, entries, manifest_sha256 = repository_archive.load_manifest_commitments(
        manifest_path
    )

    assert source_commit == "0123456789abcdef0123456789abcdef01234567"
    assert entries == (
        repository_archive.ArchiveEntry(
            path="reports/figures/drop.png",
            size_bytes=4,
            sha256="d90ee9ccf6bea1d2942a7b21319338198dec2a746f8a0d0771621f00da2e0864",
            disposition="DROP",
            reference_sources=(),
        ),
        repository_archive.ArchiveEntry(
            path="reports/figures/keep.png",
            size_bytes=4,
            sha256="6ca7ea2feefc88ecb5ed6356ed963f47dc9137f82526fdd25d618ea626d0803f",
            disposition="KEEP",
            reference_sources=("README.md:1",),
        ),
    )
    assert manifest_sha256 == "668147987292b42323df710e8be3846fc1f3fe36a000eee11daeba9131acbc8c"


@pytest.mark.parametrize(
    ("name", "manifest_bytes"),
    [
        ("noncanonical JSON", _CANONICAL_COMMITMENT_MANIFEST.replace(b"\n", b"", 1)),
        (
            "duplicate JSON key",
            _CANONICAL_COMMITMENT_MANIFEST.replace(
                b'  "schema_version": 1,', b'  "schema_version": 1,\n  "schema_version": 1,'
            ),
        ),
        (
            "duplicate path",
            _CANONICAL_COMMITMENT_MANIFEST.replace(
                b"reports/figures/keep.png", b"reports/figures/drop.png"
            ),
        ),
        (
            "unsorted entries",
            _CANONICAL_COMMITMENT_MANIFEST.replace(
                b"reports/figures/drop.png", b"reports/figures/zrop.png"
            ),
        ),
        (
            "unsafe path",
            _CANONICAL_COMMITMENT_MANIFEST.replace(
                b"reports/figures/drop.png", b"reports/figures/../drop.png"
            ),
        ),
        (
            "invalid SHA-256",
            _CANONICAL_COMMITMENT_MANIFEST.replace(
                b"d90ee9ccf6bea1d2942a7b21319338198dec2a746f8a0d0771621f00da2e0864",
                b"x" * 64,
            ),
        ),
        ("invalid size", _CANONICAL_COMMITMENT_MANIFEST.replace(b'"size_bytes": 4', b'"size_bytes": -1', 1)),
        (
            "invalid disposition",
            _CANONICAL_COMMITMENT_MANIFEST.replace(b'"disposition": "DROP"', b'"disposition": "MAYBE"'),
        ),
        (
            "invalid references",
            _CANONICAL_COMMITMENT_MANIFEST.replace(b"\"README.md:1\"", b'"README.md:2", "README.md:1"'),
        ),
    ],
)
def test_load_manifest_commitments_rejects_noncanonical_or_invalid_schema(
    tmp_path: Path, name: str, manifest_bytes: bytes
) -> None:
    """Catch the named schema mutation being accepted as a commitment."""

    manifest_path = tmp_path / "figure_manifest.json"
    manifest_path.write_bytes(manifest_bytes)

    with pytest.raises(ArchiveError):
        repository_archive.load_manifest_commitments(manifest_path)
