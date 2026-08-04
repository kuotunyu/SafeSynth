"""The tracked figure manifest is the fail-closed curation evidence index."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import freeze_figure_evidence_manifest as freeze_command
from scripts import verify_figure_evidence as verify_command
from src.release.figure_evidence import (
    FigureEvidenceError,
    verify_curation_plan_matches_manifest,
    verify_frozen_figure,
    verify_repository_figure_state,
)
from src.release.repository_archive import ArchiveEntry, create_verified_archive, sha256_file
from src.release.repository_curation import FigureDisposition, FigureReference

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FREEZE_COMMAND = WORKSPACE_ROOT / "scripts" / "freeze_figure_evidence_manifest.py"
VERIFY_COMMAND = WORKSPACE_ROOT / "scripts" / "verify_figure_evidence.py"

_KEEP_SHA256 = "6ca7ea2feefc88ecb5ed6356ed963f47dc9137f82526fdd25d618ea626d0803f"
_DROP_SHA256 = "d90ee9ccf6bea1d2942a7b21319338198dec2a746f8a0d0771621f00da2e0864"


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True, text=True)


def _commit_all(root: Path, message: str) -> None:
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


def _entries() -> tuple[ArchiveEntry, ...]:
    return (
        ArchiveEntry("reports/figures/drop.png", 4, _DROP_SHA256, "DROP", ()),
        ArchiveEntry("reports/figures/keep.png", 4, _KEEP_SHA256, "KEEP", ("README.md:1",)),
    )


def _project(root: Path) -> tuple[Path, tuple[ArchiveEntry, ...]]:
    keep = root / "reports/figures/keep.png"
    drop = root / "reports/figures/drop.png"
    keep.parent.mkdir(parents=True)
    keep.write_bytes(b"keep")
    drop.write_bytes(b"drop")
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(root, "init", "-q")
    _commit_all(root, "fixture")
    return root, _entries()


def _curate(root: Path) -> None:
    _git(root, "rm", "-q", "reports/figures/drop.png")
    _commit_all(root, "remove drop")


def _two_entry_manifest(drop_sha256: str = _DROP_SHA256) -> bytes:
    return f'''{{
  "entries": [
    {{
      "disposition": "DROP",
      "path": "reports/figures/drop.png",
      "reference_sources": [],
      "sha256": "{drop_sha256}",
      "size_bytes": 4
    }},
    {{
      "disposition": "KEEP",
      "path": "reports/figures/keep.png",
      "reference_sources": [
        "README.md:1"
      ],
      "sha256": "{_KEEP_SHA256}",
      "size_bytes": 4
    }}
  ],
  "schema_version": 1,
  "source_commit": "0123456789abcdef0123456789abcdef01234567"
}}
'''.encode()


def _configure_two_entry_verifier(
    monkeypatch: pytest.MonkeyPatch, manifest: Path, entries: tuple[ArchiveEntry, ...]
) -> None:
    monkeypatch.setattr(
        verify_command, "APPROVED_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(verify_command, "EXPECTED_TOTAL", len(entries))
    monkeypatch.setattr(verify_command, "EXPECTED_KEEP", 1)
    monkeypatch.setattr(verify_command, "EXPECTED_DROP", 1)


def test_source_state_requires_and_hashes_every_manifest_entry(tmp_path: Path) -> None:
    """Catch source verification accepting a corrupt tracked figure."""

    root, entries = _project(tmp_path / "project")

    assert verify_repository_figure_state(root, entries, "source") == "source"
    (root / "reports/figures/drop.png").write_bytes(b"bad!")

    with pytest.raises(FigureEvidenceError, match="SHA-256 mismatch"):
        verify_repository_figure_state(root, entries, "source")


def test_curated_state_requires_only_keep_and_rejects_present_drop(tmp_path: Path) -> None:
    """Catch curated verification accepting a tracked or present DROP byte."""

    root, entries = _project(tmp_path / "project")

    with pytest.raises(FigureEvidenceError, match="expected curated"):
        verify_repository_figure_state(root, entries, "curated")

    _curate(root)
    assert verify_repository_figure_state(root, entries, "curated") == "curated"
    (root / "reports/figures/drop.png").write_bytes(b"drop")

    with pytest.raises(FigureEvidenceError, match="expected curated"):
        verify_repository_figure_state(root, entries, "curated")


def test_auto_accepts_only_exact_source_or_curated_inventory(tmp_path: Path) -> None:
    """Catch auto detection accepting an inventory other than the two legal states."""

    root, entries = _project(tmp_path / "project")

    assert verify_repository_figure_state(root, entries, "auto") == "source"
    _curate(root)
    assert verify_repository_figure_state(root, entries, "auto") == "curated"


def test_partial_mixed_extra_untracked_and_corrupt_states_fail(tmp_path: Path) -> None:
    """Catch exact-inventory verification accepting an incomplete or altered repository."""

    root, entries = _project(tmp_path / "partial")
    (root / "reports/figures/drop.png").unlink()
    with pytest.raises(FigureEvidenceError):
        verify_repository_figure_state(root, entries, "auto")

    root, entries = _project(tmp_path / "extra")
    (root / "reports/figures/extra.png").write_bytes(b"extra")
    with pytest.raises(FigureEvidenceError):
        verify_repository_figure_state(root, entries, "auto")

    root, entries = _project(tmp_path / "untracked")
    _git(root, "rm", "-q", "reports/figures/drop.png")
    (root / "reports/figures/drop.png").write_bytes(b"drop")
    with pytest.raises(FigureEvidenceError):
        verify_repository_figure_state(root, entries, "auto")

    root, entries = _project(tmp_path / "corrupt")
    (root / "reports/figures/keep.png").write_bytes(b"bad!")
    with pytest.raises(FigureEvidenceError, match="SHA-256 mismatch"):
        verify_repository_figure_state(root, entries, "auto")


def test_missing_keep_and_path_alias_fail(tmp_path: Path) -> None:
    """Catch verification accepting a missing KEEP byte or a linked path alias."""

    root, entries = _project(tmp_path / "missing")
    (root / "reports/figures/keep.png").unlink()
    with pytest.raises(FigureEvidenceError):
        verify_repository_figure_state(root, entries, "source")

    root, entries = _project(tmp_path / "alias")
    figure_root = root / "reports/figures"
    target = root / "figure-target"
    figure_root.rename(target)
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(figure_root), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    with pytest.raises(FigureEvidenceError, match="path alias"):
        verify_repository_figure_state(root, entries, "source")


def test_frozen_drop_may_be_archived_only_when_metadata_sha_matches(tmp_path: Path) -> None:
    """Catch missing DROP acceptance without matching the metadata SHA commitment."""

    root, entries = _project(tmp_path / "project")
    _curate(root)
    by_path = {entry.path: entry for entry in entries}

    verify_frozen_figure(root, by_path, "reports/figures/drop.png", _DROP_SHA256)


def test_unknown_path_and_metadata_sha_conflict_fail(tmp_path: Path) -> None:
    """Catch metadata binding an unknown path or a digest different from the manifest."""

    root, entries = _project(tmp_path / "project")
    by_path = {entry.path: entry for entry in entries}

    with pytest.raises(FigureEvidenceError, match="not indexed"):
        verify_frozen_figure(root, by_path, "reports/figures/unknown.png", _DROP_SHA256)
    with pytest.raises(FigureEvidenceError, match="metadata SHA-256"):
        verify_frozen_figure(root, by_path, "reports/figures/drop.png", _KEEP_SHA256)


def test_curation_plan_must_match_every_manifest_commitment(tmp_path: Path) -> None:
    """Catch a current plan changing a disposition, reference, or source byte from the manifest."""

    root, entries = _project(tmp_path / "project")
    plan = (
        FigureDisposition("reports/figures/drop.png", 4, False, ()),
        FigureDisposition(
            "reports/figures/keep.png", 4, True, (FigureReference("README.md", 1),)
        ),
    )

    verify_curation_plan_matches_manifest(root, plan, entries)
    changed_plan = (
        FigureDisposition("reports/figures/drop.png", 4, True, ()),
        plan[1],
    )
    with pytest.raises(FigureEvidenceError, match="disposition differs"):
        verify_curation_plan_matches_manifest(root, changed_plan, entries)
    (root / "reports/figures/keep.png").write_bytes(b"bad!")
    with pytest.raises(FigureEvidenceError, match="SHA-256 mismatch"):
        verify_curation_plan_matches_manifest(root, plan, entries)


def test_freeze_command_verifies_archive_digest_and_never_leaves_partial_output(tmp_path: Path) -> None:
    """Catch a freeze operation that publishes bytes before every archive check passes."""

    root, _ = _project(tmp_path / "project")
    archive = tmp_path / "archive"
    plan = (
        FigureDisposition("reports/figures/drop.png", 4, False, ()),
        FigureDisposition(
            "reports/figures/keep.png", 4, True, (FigureReference("README.md", 1),)
        ),
    )
    create_verified_archive(root, archive, plan)
    output = tmp_path / "reports/figure_curation_manifest.json"
    output.parent.mkdir()

    with pytest.raises(freeze_command.FreezeFigureEvidenceError):
        freeze_command.freeze(archive, "0" * 64, output)
    assert not output.exists()

    freeze_command.freeze(archive, sha256_file(archive / "figure_manifest.json"), output)
    assert output.read_bytes() == (archive / "figure_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        freeze_command.freeze(archive, sha256_file(archive / "figure_manifest.json"), output)


def test_verifier_command_enforces_expected_counts_hash_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch the public gate proceeding when its pinned manifest facts do not match."""

    root, entries = _project(tmp_path / "project")
    manifest = root / "reports/figure_curation_manifest.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_bytes(_two_entry_manifest())
    _commit_all(root, "track canonical manifest")
    _configure_two_entry_verifier(monkeypatch, manifest, entries)

    assert verify_command.main(["--project-root", str(root), "--expected-state", "source"]) == 0
    monkeypatch.setattr(verify_command, "EXPECTED_TOTAL", 3)
    assert verify_command.main(["--project-root", str(root), "--expected-state", "source"]) == 1
    monkeypatch.setattr(verify_command, "EXPECTED_TOTAL", len(entries))
    monkeypatch.setattr(verify_command, "APPROVED_MANIFEST_SHA256", "0" * 64)
    assert verify_command.main(["--project-root", str(root), "--expected-state", "source"]) == 1
    monkeypatch.setattr(
        verify_command, "APPROVED_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    _curate(root)
    assert verify_command.main(["--project-root", str(root), "--expected-state", "source"]) == 1


def test_verifier_command_rejects_a_correct_but_untracked_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catch the public verifier accepting canonical bytes absent from Git's exact manifest path."""

    root, entries = _project(tmp_path / "project")
    manifest = root / "reports/figure_curation_manifest.json"
    manifest.parent.mkdir(exist_ok=True)
    manifest.write_bytes(_two_entry_manifest())
    _configure_two_entry_verifier(monkeypatch, manifest, entries)

    assert verify_command.main(["--project-root", str(root), "--expected-state", "source"]) == 1

    assert "manifest is not Git-tracked" in capsys.readouterr().err


def test_verifier_uses_the_entries_from_the_manifest_bytes_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a second manifest parse accepting replacement entries after the approved bytes were hashed."""

    root, entries = _project(tmp_path / "project")
    manifest = root / "reports/figure_curation_manifest.json"
    manifest.parent.mkdir(exist_ok=True)
    original_bytes = _two_entry_manifest()
    manifest.write_bytes(original_bytes)
    _commit_all(root, "track canonical manifest")
    _configure_two_entry_verifier(monkeypatch, manifest, entries)
    original_load = verify_command.load_manifest_commitments

    def replace_after_first_verified_read(manifest_path: Path) -> tuple[str, tuple[ArchiveEntry, ...], str]:
        commitments = original_load(manifest_path)
        manifest.write_bytes(_two_entry_manifest("0" * 64))
        return commitments

    monkeypatch.setattr(verify_command, "load_manifest_commitments", replace_after_first_verified_read)

    assert verify_command.main(["--project-root", str(root), "--expected-state", "source"]) == 0


def test_figure_evidence_scripts_run_directly_without_pytest_path_injection() -> None:
    """Catch direct CLI execution failing because the repository package is not importable."""

    for command in (FREEZE_COMMAND, VERIFY_COMMAND):
        completed = subprocess.run(
            [sys.executable, str(command), "--help"],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
