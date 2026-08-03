# Repository Curation and History Slimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the unpublished SafeSynth Git repository to a verified pack below 120 MiB while retaining every figure linked by surviving documentation, archiving every current figure externally, and preserving `kuotunyu` as the sole contributor.

**Architecture:** A shared Markdown-link resolver produces exact repository-relative targets. A curation planner uses those targets to generate a deterministic KEEP/DROP inventory, while a separate archive module copies and hashes the complete figure tree, creates a verified Git bundle, and restores only KEEP files after the owner-operated history rewrite. A repository-wide link verifier supplies the final fail-closed acceptance gate.

**Tech Stack:** Python 3.12 standard library, dataclasses, pathlib/PurePosixPath, hashlib, json, shutil, subprocess/Git, pytest, Ruff, PowerShell, `git-filter-repo` (owner-operated only).

## Final-review amendment — archive rebound and staged owner gate

The v1 archive at
`D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation`
remains a recovery snapshot only for source commit
`2c2d3ff5198ff600220e5b1e1c606ebc80e07a98`. The v2 archive at
`D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v2`
remains a recovery snapshot only for source commit
`f514950b142da95bb4c71d3626b9417fb25a3bff`. These v1 and v2 recovery snapshots
are forbidden for the owner gate after this amendment. The next archive attempt
must use the non-overwriting destination
`D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3`
and bind its receipt and stage-1 owner runbook to the current clean branch HEAD.

## Global Constraints

- Keep only files under `reports/figures/` that a surviving tracked Markdown document links to through an exact normalized path.
- Exclude generated `reports/repo_slimming_plan.md` from reference inputs so it cannot promote its own DROP entries to KEEP.
- Treat unresolved, escaping, malformed, or ambiguous local links as blocking failures; never silently classify them as DROP.
- Archive every current tracked file under `reports/figures/`, including both KEEP and DROP, under `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3`.
- Record byte size and SHA-256 for every archived file and verify source/archive equality before allowing history rewrite.
- Create and verify a complete pre-rewrite Git bundle outside the repository.
- Do not overwrite or delete an existing archive destination.
- The agent must not run `git filter-repo`, create a remote, push, force-push, or upload to Hugging Face; the owner runs the exact reviewed history-rewrite command.
- Integrate the development branch and remove the linked worktree before the owner rewrites history.
- Restore only the manifest KEEP set after rewriting and verify every tracked Markdown link, not only README links.
- Require a post-rewrite Git object pack below 120 MiB; investigate rather than weakening this threshold.
- Git author and committer must remain `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` with no `Co-Authored-By` trailer.
- Do not publish fine-tuned RF-DETR latency results that have not passed the contention gate.
- No task in this plan requires GPU or loads a model.

## File Map

- Create `src/release/__init__.py`: package marker for release-only repository tooling.
- Create `src/release/markdown_links.py`: extract real Markdown destinations and resolve safe local repository paths.
- Create `src/release/repository_curation.py`: tracked-file inventory, exact figure references, KEEP/DROP dataclasses, and deterministic planning.
- Modify `scripts/plan_repo_slimming.py`: render the human report from the shared curation plan.
- Expand `tests/test_plan_repo_slimming.py`: parser, classification, self-reference, deterministic rendering, and direct-script coverage.
- Create `src/release/repository_archive.py`: SHA-256 manifest, fail-closed archive, Git bundle verification, and KEEP-only restoration.
- Create `scripts/archive_repository_curation.py`: safe archive/bundle command-line entry point.
- Create `scripts/restore_curated_figures.py`: post-rewrite KEEP-only restoration entry point.
- Create `tests/test_repository_archive.py`: archive, tamper detection, bundle, and restoration tests.
- Create `scripts/verify_repository_links.py`: all-tracked-Markdown local-link verifier.
- Create `tests/test_verify_repository_links.py`: valid, missing, escaping, external, and command-line exit-code tests.
- Regenerate `reports/repo_slimming_plan.md`: stable exact-path KEEP/DROP report.
- Use the approved design at `docs/superpowers/specs/2026-08-04-repository-curation-history-slimming-design.md` as the requirements source.

---

### Task 1: Exact Markdown link resolution

**Files:**
- Create: `src/release/__init__.py`
- Create: `src/release/markdown_links.py`
- Modify: `tests/test_plan_repo_slimming.py`

**Interfaces:**
- Produces: `RepositoryLinkError(ValueError)`.
- Produces: `MarkdownDestination(source_path: str, line_number: int, raw_target: str, resolved_path: str | None)`.
- Produces: `extract_markdown_destinations(text: str, source_path: str) -> tuple[tuple[int, str], ...]`.
- Produces: `resolve_local_target(source_path: str, raw_target: str) -> str | None` where `None` means an external URL or same-document anchor.
- Produces: `collect_local_destinations(root: Path, markdown_paths: Sequence[str]) -> tuple[MarkdownDestination, ...]`.
- Consumes: only text and repository-relative POSIX paths; no Git or filesystem mutation.

- [ ] **Step 1: Add failing tests for real links and ignored mentions**

Add tests that demonstrate link extraction ignores fenced code, inline code, and plain prose while retaining inline Markdown links, images, angle-bracket destinations, and reference definitions:

```python
from src.release.markdown_links import collect_local_destinations


def test_only_real_markdown_destinations_are_collected(tmp_path: Path) -> None:
    doc = tmp_path / "docs" / "audit.md"
    doc.parent.mkdir()
    doc.write_text(
        """# Audit
![inline](../reports/figures/a.png)
[angle](<../reports/figures/with space.png>)
[ref]: ../reports/figures/reference.png
`reports/figures/code.png`
plain reports/figures/prose.png
```text
![fenced](../reports/figures/fenced.png)
```
""",
        encoding="utf-8",
    )

    links = collect_local_destinations(tmp_path, ["docs/audit.md"])

    assert [link.resolved_path for link in links] == [
        "reports/figures/a.png",
        "reports/figures/with space.png",
        "reports/figures/reference.png",
    ]
```

- [ ] **Step 2: Add failing tests for normalization and fail-closed paths**

```python
import pytest

from src.release.markdown_links import RepositoryLinkError, resolve_local_target


def test_relative_paths_are_resolved_from_the_document_directory() -> None:
    assert (
        resolve_local_target("docs/audit/report.md", "../../reports/figures/a.png")
        == "reports/figures/a.png"
    )


def test_external_urls_and_anchors_are_not_local_targets() -> None:
    assert resolve_local_target("README.md", "https://example.com/x.png") is None
    assert resolve_local_target("README.md", "#results") is None


def test_path_escape_fails_closed() -> None:
    with pytest.raises(RepositoryLinkError, match="escapes repository"):
        resolve_local_target("README.md", "../outside.png")
```

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run:

```powershell
uv run pytest tests/test_plan_repo_slimming.py -q
```

Expected: FAIL because `src.release.markdown_links` does not exist.

- [ ] **Step 4: Implement the minimal shared resolver**

Implement `src/release/markdown_links.py` with focused helpers. Strip fenced blocks line-by-line, remove inline code spans before matching, support inline link/image destinations and reference-definition destinations, URL-decode local paths, drop query/fragment suffixes, normalize with `posixpath.normpath`, and reject absolute or `..`-escaping local paths.

Core definitions:

```python
from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


class RepositoryLinkError(ValueError):
    """A local Markdown destination cannot be resolved safely."""


@dataclass(frozen=True)
class MarkdownDestination:
    source_path: str
    line_number: int
    raw_target: str
    resolved_path: str | None


def resolve_local_target(source_path: str, raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    parsed = urlsplit(unquote(target))
    if parsed.scheme or parsed.netloc or target.startswith("#"):
        return None
    if not parsed.path:
        return None
    if PurePosixPath(parsed.path).is_absolute():
        raise RepositoryLinkError(f"absolute local path is forbidden: {raw_target}")
    base = PurePosixPath(source_path).parent.as_posix()
    resolved = posixpath.normpath(posixpath.join(base, parsed.path.replace("\\", "/")))
    if resolved == ".." or resolved.startswith("../"):
        raise RepositoryLinkError(f"link escapes repository: {raw_target}")
    return resolved
```

Keep the extraction expressions private and expose only the interfaces named above.

- [ ] **Step 5: Run focused tests and Ruff**

Run:

```powershell
uv run pytest tests/test_plan_repo_slimming.py -q
uv run ruff check src/release/markdown_links.py tests/test_plan_repo_slimming.py
```

Expected: both commands PASS.

- [ ] **Step 6: Commit the resolver**

```powershell
git add src/release/__init__.py src/release/markdown_links.py tests/test_plan_repo_slimming.py
git diff --cached --check
git commit -m "feat(release): resolve exact markdown links"
```

Verify the commit author/committer and absence of co-author trailers before proceeding.

---

### Task 2: Deterministic figure curation plan

**Files:**
- Create: `src/release/repository_curation.py`
- Modify: `scripts/plan_repo_slimming.py`
- Modify: `tests/test_plan_repo_slimming.py`
- Regenerate: `reports/repo_slimming_plan.md`

**Interfaces:**
- Consumes: `collect_local_destinations()` from Task 1.
- Produces: `FigureReference(source_path: str, line_number: int)`.
- Produces: `FigureDisposition(path: str, size_bytes: int, keep: bool, references: tuple[FigureReference, ...])`.
- Produces: `tracked_files(root: Path) -> list[str]`.
- Produces: `plan_figure_curation(root: Path, files: Sequence[str]) -> tuple[FigureDisposition, ...]`.
- Produces: `render(root: Path) -> str` in `scripts.plan_repo_slimming` using the shared plan.

- [ ] **Step 1: Add failing tests for self-reference and duplicate basenames**

```python
from src.release.repository_curation import plan_figure_curation


def test_generated_report_cannot_promote_drop_entries(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "![keep](reports/figures/a/keep.png)\n")
    _write(
        tmp_path,
        "reports/repo_slimming_plan.md",
        "`reports/figures/drop.png`\n",
    )
    _write_bytes(tmp_path, "reports/figures/a/keep.png", b"keep")
    _write_bytes(tmp_path, "reports/figures/drop.png", b"drop")
    files = [
        "README.md",
        "reports/repo_slimming_plan.md",
        "reports/figures/a/keep.png",
        "reports/figures/drop.png",
    ]

    planned = {item.path: item for item in plan_figure_curation(tmp_path, files)}

    assert planned["reports/figures/a/keep.png"].keep is True
    assert planned["reports/figures/drop.png"].keep is False


def test_same_basename_in_two_directories_is_not_conflated(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "![keep](reports/figures/a/same.png)\n")
    _write_bytes(tmp_path, "reports/figures/a/same.png", b"a")
    _write_bytes(tmp_path, "reports/figures/b/same.png", b"b")
    files = [
        "README.md",
        "reports/figures/a/same.png",
        "reports/figures/b/same.png",
    ]

    planned = {item.path: item for item in plan_figure_curation(tmp_path, files)}

    assert planned["reports/figures/a/same.png"].keep is True
    assert planned["reports/figures/b/same.png"].keep is False
```

- [ ] **Step 2: Add a failing determinism test**

Monkeypatch `history_bytes_by_area()` to a fixed value and assert two renders are byte-identical and include source-document evidence for each KEEP entry:

```python
def test_report_is_deterministic_and_names_keep_sources(
    fixture_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        plan_repo_slimming,
        "history_bytes_by_area",
        lambda _root: (100.0, {"reports/figures/": 75.0, "everything else": 25.0}),
    )

    first = plan_repo_slimming.render(fixture_repo)
    second = plan_repo_slimming.render(fixture_repo)

    assert first == second
    assert "README.md:3" in first
```

- [ ] **Step 3: Run focused tests and confirm they fail**

```powershell
uv run pytest tests/test_plan_repo_slimming.py -q
```

Expected: FAIL because `repository_curation` and exact-path rendering are absent.

- [ ] **Step 4: Implement the planner and refactor the renderer**

Use these data structures in `src/release/repository_curation.py`:

```python
GENERATED_REFERENCE_INPUTS = frozenset({"reports/repo_slimming_plan.md"})
FIGURE_ROOT = "reports/figures/"


@dataclass(frozen=True)
class FigureReference:
    source_path: str
    line_number: int


@dataclass(frozen=True)
class FigureDisposition:
    path: str
    size_bytes: int
    keep: bool
    references: tuple[FigureReference, ...]
```

`plan_figure_curation()` must sort figures by full path and sort references by `(source_path, line_number)`. It must raise `RepositoryLinkError` if a local figure link is missing from the tracked inventory instead of silently dropping it.

Refactor `scripts/plan_repo_slimming.py` to remove `IMAGE_PATTERN`, basename matching, and `referenced_image_names()`. Render KEEP/DROP counts from `FigureDisposition`, list exact source lines for KEEP, and state that the complete current tree is archived before owner action.

- [ ] **Step 5: Run focused tests and regenerate twice**

```powershell
uv run pytest tests/test_plan_repo_slimming.py -q
uv run python scripts/plan_repo_slimming.py
$firstDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath reports/repo_slimming_plan.md).Hash
uv run python scripts/plan_repo_slimming.py
$secondDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath reports/repo_slimming_plan.md).Hash
if ($firstDigest -ne $secondDigest) { throw "slimming report is not deterministic" }
```

Expected: tests PASS; both hashes are identical. The corrected real-repository report should be close to the read-only baseline of 32 KEEP / 118 DROP; any difference must be explained by exact resolved links before committing.

- [ ] **Step 6: Run Ruff and commit**

```powershell
uv run ruff check src/release/repository_curation.py scripts/plan_repo_slimming.py tests/test_plan_repo_slimming.py
git add src/release/repository_curation.py scripts/plan_repo_slimming.py tests/test_plan_repo_slimming.py reports/repo_slimming_plan.md
git diff --cached --check
git commit -m "fix(release): make figure curation deterministic"
```

Verify the approved identity and no co-author trailer.

---

### Task 3: Verified external archive and Git bundle

**Files:**
- Create: `src/release/repository_archive.py`
- Create: `tests/test_repository_archive.py`

**Interfaces:**
- Consumes: `FigureDisposition` from Task 2.
- Produces: `ArchiveError(RuntimeError)`.
- Produces: `ArchiveEntry(path: str, size_bytes: int, sha256: str, disposition: str, reference_sources: tuple[str, ...])`.
- Produces: `ArchiveReceipt(source_commit: str, manifest_path: Path, manifest_sha256: str, bundle_path: Path, bundle_sha256: str, entries: tuple[ArchiveEntry, ...])`.
- Produces: `sha256_file(path: Path) -> str`.
- Produces: `create_verified_archive(project_root: Path, destination: Path, dispositions: Sequence[FigureDisposition]) -> Path` returning the manifest path.
- Produces: `create_verified_git_bundle(project_root: Path, bundle_path: Path) -> str` returning the bundle SHA-256.
- Produces: `create_recovery_package(project_root: Path, destination: Path, dispositions: Sequence[FigureDisposition]) -> ArchiveReceipt`.
- Produces: `load_and_verify_manifest(archive_root: Path) -> tuple[ArchiveEntry, ...]`.
- Produces: `restore_keep_files(project_root: Path, archive_root: Path) -> tuple[str, ...]`.

- [ ] **Step 1: Add failing archive and tamper tests**

```python
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


def test_archive_tampering_is_a_blocking_failure(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "project")
    destination = tmp_path / "archive"
    create_verified_archive(project.root, destination, project.plan)
    (destination / "figures" / "reports" / "figures" / "drop.png").write_bytes(b"changed")

    with pytest.raises(ArchiveError, match="SHA-256 mismatch"):
        load_and_verify_manifest(destination)
```

- [ ] **Step 2: Add failing no-overwrite and KEEP-only restoration tests**

```python
def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "archive"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        create_verified_archive(tmp_path / "project", destination, ())


def test_restore_copies_only_verified_keep_files(tmp_path: Path) -> None:
    project = _project_with_keep_and_drop(tmp_path / "source")
    archive = tmp_path / "archive"
    create_verified_archive(project.root, archive, project.plan)
    restored_root = tmp_path / "restored"

    restored = restore_keep_files(restored_root, archive)

    assert restored == ("reports/figures/keep.png",)
    assert (restored_root / "reports/figures/keep.png").read_bytes() == b"keep"
    assert not (restored_root / "reports/figures/drop.png").exists()
```

- [ ] **Step 3: Add a failing verified Git bundle test**

Create a two-commit temporary Git repository, call `create_verified_git_bundle()`, assert the bundle exists, has the returned digest, and passes `git bundle verify`.

- [ ] **Step 4: Run focused tests and confirm they fail**

```powershell
uv run pytest tests/test_repository_archive.py -q
```

Expected: FAIL because `repository_archive` does not exist.

- [ ] **Step 5: Implement canonical manifests and fail-closed copy/restore**

Write manifest JSON with `sort_keys=True`, `indent=2`, UTF-8, and a trailing newline. Use schema:

```json
{
  "schema_version": 1,
  "source_commit": "40-character Git object id",
  "entries": [
    {
      "path": "reports/figures/example.png",
      "size_bytes": 123,
      "sha256": "64 lowercase hex characters",
      "disposition": "KEEP",
      "reference_sources": ["README.md:10"]
    }
  ]
}
```

Copy files to `destination/figures/<repository-relative-path>`. Reject existing destinations, missing sources, duplicate paths, path escapes, size mismatches, digest mismatches, manifest schema mismatches, and destination KEEP files that already exist. Invoke `git bundle create <path> --all`, then `git bundle verify <path>`, and delete only the newly created invalid bundle if verification fails. `create_recovery_package()` composes the verified archive and bundle and returns the canonical data used to write the receipt.

- [ ] **Step 6: Run tests, Ruff, and commit**

```powershell
uv run pytest tests/test_repository_archive.py -q
uv run ruff check src/release/repository_archive.py tests/test_repository_archive.py
git add src/release/repository_archive.py tests/test_repository_archive.py
git diff --cached --check
git commit -m "feat(release): archive repository evidence safely"
```

Verify identity and trailers.

---

### Task 4: Safe archive and restoration commands

**Files:**
- Create: `scripts/archive_repository_curation.py`
- Create: `scripts/restore_curated_figures.py`
- Modify: `tests/test_repository_archive.py`

**Interfaces:**
- Consumes: Task 2 planning and Task 3 archive functions.
- Produces: archive CLI arguments `--project-root`, `--destination`, and `--owner-project-root`.
- Produces: restore CLI arguments `--project-root` and required `--archive`.
- Produces externally: `archive_receipt.json`, `SafeSynth-pre-filter-repo.bundle`, and `OWNER_HISTORY_REWRITE_RUNBOOK.txt` inside the archive directory.

- [ ] **Step 1: Add failing command-line tests**

Test that archive CLI:

- exits nonzero when destination exists;
- exits nonzero if planning finds an unresolved figure link;
- archives KEEP and DROP;
- creates and verifies the bundle;
- writes a receipt containing source commit and manifest/bundle digests; and
- writes a PowerShell runbook pointing to the passed owner project root without modifying that root.

Test that restore CLI:

- exits nonzero on a tampered manifest/file;
- exits nonzero if `reports/figures/` is nonempty; and
- restores only KEEP files when the target is clean.

- [ ] **Step 2: Run focused tests and confirm they fail**

```powershell
uv run pytest tests/test_repository_archive.py -q
```

Expected: FAIL because both scripts are absent.

- [ ] **Step 3: Implement the archive command**

The command must compute the real tracked inventory, plan it, call `create_recovery_package()`, verify the returned archive and bundle, and only then write `archive_receipt.json` and the runbook. Its success output must state counts, digests, source commit, and destination. It must not print or execute `git-filter-repo` before every verification passes.

The generated PowerShell runbook is Stage 1 only. It must use these owner steps
in this order, explicitly checking `$LASTEXITCODE` after every native command:

```powershell
$ErrorActionPreference = 'Stop'
$OwnerProjectRoot = 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth'
$ExpectedSourceCommit = '<recovery.source_commit>'
if (-not (Test-Path -LiteralPath $OwnerProjectRoot -PathType Container)) { throw 'Owner project root is not a directory. STOP.' }
Set-Location -LiteralPath $OwnerProjectRoot
$GitStatus = @(git status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) { throw "git status failed with exit code $LASTEXITCODE. STOP." }
if ($GitStatus.Count -ne 0) { throw "Owner repository is not clean. STOP. Status:`n$($GitStatus -join [Environment]::NewLine)" }
$ActualSourceCommit = @(git rev-parse --verify HEAD)
if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed with exit code $LASTEXITCODE. STOP." }
if ($ActualSourceCommit.Count -ne 1) { throw "git rev-parse must return exactly one source commit line. STOP." }
if ($ActualSourceCommit[0] -cne $ExpectedSourceCommit) { throw 'Owner repository HEAD does not match archived source commit. STOP.' }
uvx git-filter-repo --version
if ($LASTEXITCODE -ne 0) { throw "git-filter-repo availability check failed with exit code $LASTEXITCODE. STOP." }
uvx git-filter-repo --path reports/figures/ --invert-paths --force
if ($LASTEXITCODE -ne 0) { throw "git-filter-repo rewrite failed with exit code $LASTEXITCODE. STOP." }
Write-Host 'STOP: Stage 1 history rewrite finished. Do not restore, stage, or commit. Report the full output to the controller and wait for the Task 7 read-only checkpoint.'
```

The implementation embeds the exact passed owner-root argument and the verified
canonical recovery source commit safely. The Stage 1 runbook contains no archive
path or restoration authority.

- [ ] **Step 4: Implement the restore command**

Require an empty/nonexistent `reports/figures/` target, verify the complete external archive before copying, restore only `disposition == "KEEP"`, re-hash restored files, and print the sorted restored inventory. Do not delete DROP files from the archive.

- [ ] **Step 5: Run tests, direct help checks, and Ruff**

```powershell
uv run pytest tests/test_repository_archive.py -q
uv run python scripts/archive_repository_curation.py --help
uv run python scripts/restore_curated_figures.py --help
uv run ruff check scripts/archive_repository_curation.py scripts/restore_curated_figures.py tests/test_repository_archive.py
```

Expected: all commands PASS.

- [ ] **Step 6: Commit the command-line safeguards**

```powershell
git add scripts/archive_repository_curation.py scripts/restore_curated_figures.py tests/test_repository_archive.py
git diff --cached --check
git commit -m "feat(release): guard repository curation workflow"
```

Verify identity and trailers.

---

### Task 5: Repository-wide Markdown link verification

**Files:**
- Create: `scripts/verify_repository_links.py`
- Create: `tests/test_verify_repository_links.py`

**Interfaces:**
- Consumes: `collect_local_destinations()` from Task 1 and `tracked_files()` from Task 2.
- Produces: `LinkFailure(source_path: str, line_number: int, target: str, reason: str)`.
- Produces: `verify_repository_links(root: Path, files: Sequence[str]) -> tuple[LinkFailure, ...]`.
- Produces: CLI exit 0 on complete success and 1 for any broken/unsafe local link.

- [ ] **Step 1: Add failing verifier tests**

```python
def test_all_tracked_markdown_local_links_are_checked(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "[ok](docs/ok.md)\n")
    _write(tmp_path, "docs/ok.md", "![missing](../reports/figures/missing.png)\n")
    files = ["README.md", "docs/ok.md"]

    failures = verify_repository_links(tmp_path, files)

    assert [(item.source_path, item.line_number) for item in failures] == [
        ("docs/ok.md", 1)
    ]


def test_external_links_are_out_of_scope(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "[site](https://example.com)\n")
    assert verify_repository_links(tmp_path, ["README.md"]) == ()
```

Add CLI tests for exit codes and sorted diagnostics.

- [ ] **Step 2: Run focused tests and confirm they fail**

```powershell
uv run pytest tests/test_verify_repository_links.py -q
```

Expected: FAIL because the verifier is absent.

- [ ] **Step 3: Implement the minimal verifier**

Check every tracked `.md` returned by Git. A local target passes if it is an existing file, or if it is an existing directory explicitly linked as a directory. Catch `RepositoryLinkError` and return it as a failure. Print every failure, then a final PASS/FAILED count. Never perform network checks.

- [ ] **Step 4: Run focused tests, full direct verification, and Ruff**

```powershell
uv run pytest tests/test_verify_repository_links.py -q
uv run python scripts/verify_repository_links.py
uv run ruff check scripts/verify_repository_links.py tests/test_verify_repository_links.py
```

Expected: tests and the real-repository verifier PASS. If the real verifier identifies an existing broken link, stop this task and amend this plan with the exact document paths and corrections before changing those documents; do not weaken the verifier or add an allowlist.

- [ ] **Step 5: Commit verifier and any exact link corrections**

```powershell
git add scripts/verify_repository_links.py tests/test_verify_repository_links.py
git diff --cached --check
git commit -m "feat(release): verify all repository links"
```

Before committing, inspect `git diff --cached --name-only` and confirm only the new verifier and its tests are staged. Verify identity and trailers after committing.

---

### Task 6: Build and verify the real recovery archive

**Files:**
- Read: `reports/repo_slimming_plan.md`
- External create: `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3\`
- No tracked repository changes expected.

**Interfaces:**
- Consumes: all completed commands and the clean branch HEAD.
- Produces: complete figure archive, manifest, receipt, verified Git bundle, and owner runbook outside Git.

- [ ] **Step 1: Run the complete pre-archive verification suite**

```powershell
uv run pytest -q
uv run ruff check .
uv run python scripts/verify_readme.py
uv run python scripts/verify_repository_links.py
uv lock --check
git diff --check
git status --short --branch
```

Expected: every verifier passes and the branch is clean.

- [ ] **Step 2: Confirm the approved destination does not exist**

```powershell
$safeSynthArchive = 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3'
if (Test-Path -LiteralPath $safeSynthArchive) { throw "approved archive destination already exists" }
```

- [ ] **Step 3: Create the complete verified archive and bundle**

```powershell
uv run python scripts/archive_repository_curation.py --destination 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3' --owner-project-root 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth'
```

Expected: exit 0 and a receipt reporting all tracked `reports/figures/` files, KEEP/DROP counts, source commit, manifest SHA-256, and bundle SHA-256.

- [ ] **Step 4: Independently re-verify archive and bundle**

```powershell
$safeSynthArchive = 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3'
Get-FileHash -Algorithm SHA256 -LiteralPath "$safeSynthArchive\figure_manifest.json"
Get-FileHash -Algorithm SHA256 -LiteralPath "$safeSynthArchive\SafeSynth-pre-filter-repo.bundle"
git bundle verify "$safeSynthArchive\SafeSynth-pre-filter-repo.bundle"
```

Compare both hashes with `archive_receipt.json`. Any mismatch blocks Task 7.

- [ ] **Step 5: Verify contributor identity and clean state**

```powershell
git log --format='%an <%ae>|%cn <%ce>' | Sort-Object -Unique
git log --format='%B' | Select-String -Pattern 'Co-Authored-By|Co-authored-by'
git status --short --branch
```

Expected: the identity command prints only `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` for author and committer, the trailer scan prints nothing, and the branch is clean.

---

### Task 7: Fast-forward integration and owner history-rewrite gate

**Files:**
- Worktree to retire after integration: `C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth\.worktrees\rfdetr-four-arm`
- Main repository: `C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth`
- External read: `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3\OWNER_HISTORY_REWRITE_RUNBOOK.txt`

**Interfaces:**
- Consumes: verified Task 6 archive/bundle and clean branch.
- Produces: main fast-forwarded to the completed branch, no linked worktree, and an explicit owner action request.
- Owner produces: rewritten local history with `reports/figures/` absent before restoration.

- [ ] **Step 1: Verify fast-forward safety without changing either worktree**

```powershell
git -C 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth' status --short --branch
git merge-base --is-ancestor main codex/rfdetr-four-arm
```

Expected: main is clean and the ancestor command exits 0. If either check fails, stop and reconcile without deleting either worktree or rewriting history.

- [ ] **Step 2: Fast-forward main**

```powershell
git -C 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth' merge --ff-only codex/rfdetr-four-arm
```

Expected: fast-forward succeeds without a merge commit, so the verified bundle already contains the resulting HEAD object.

- [ ] **Step 3: Verify main, then remove only the clean linked worktree**

```powershell
git -C 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth' status --short --branch
git -C 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth\.worktrees\rfdetr-four-arm' status --short --branch
git -C 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth' worktree remove 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth\.worktrees\rfdetr-four-arm'
git -C 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth' worktree list
```

Expected: both statuses are clean before removal, and only the intended linked worktree is removed. Never use `--force`.

- [ ] **Step 4: Run the stage-1 owner runbook, stop and report**

The agent must not execute the next command. Ask the owner to open the verified
stage-1 owner runbook and run it in Windows PowerShell. The runbook checks the
clean state, requires `HEAD` to equal the archive's exact source commit, checks
every native exit code, performs only the exact rewrite below, and ends with a
mandatory STOP/report-back instruction. It contains no restoration, staging, or
commit step. Explain that `uvx` downloads and runs the missing history tool
without installing a persistent global command, while the exact rewrite is:

```powershell
uvx git-filter-repo --path reports/figures/ --invert-paths --force
```

The owner must stop and report the full output or any error. Wait for that report
before continuing; do not infer completion from process disappearance and do not
start Task 8.

- [ ] **Step 5: Confirm the owner rewrite before restoration**

After owner confirmation, read-only check:

```powershell
git -C 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth' status --short --branch
git -C 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth' ls-files reports/figures
```

Expected: main is clean and `git ls-files reports/figures` prints nothing. If not, stop and diagnose before any copy.

---

### Task 8: Restore curated evidence and run post-rewrite acceptance

**Files:**
- Restore: exact manifest KEEP set under `reports/figures/`
- External read-only recovery source: `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3\`

**Interfaces:**
- Consumes: owner-rewritten main and verified archive.
- Produces: one `kuotunyu` restoration commit, pack below 120 MiB, complete passing verification, and a publication-ready local repository.

- [ ] **Step 1: Restore only verified KEEP files**

```powershell
Set-Location -LiteralPath 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth'
uv run python scripts/restore_curated_figures.py --archive 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3'
```

Expected: only manifest KEEP paths are restored, and every restored digest matches.

- [ ] **Step 2: Verify staged inventory before committing**

```powershell
git add -- reports/figures
git diff --cached --check
git diff --cached --name-status
```

Compare the staged paths exactly with the KEEP entries in `figure_manifest.json`. Any extra or missing path blocks the commit.

- [ ] **Step 3: Commit using only the approved identity**

```powershell
git config user.name kuotunyu
git config user.email 61350295+kuotunyu@users.noreply.github.com
git commit -m 'docs: restore curated figure evidence'
```

Verify author, committer, and trailers immediately.

- [ ] **Step 4: Run the complete post-rewrite verification suite**

```powershell
uv run pytest -q
uv run ruff check .
uv run python scripts/verify_readme.py
uv run python scripts/verify_repository_links.py
uv lock --check
git diff --check
git status --short --branch
```

Expected: all commands PASS and the tree is clean.

- [ ] **Step 5: Verify size and contributor invariants**

```powershell
git count-objects -vH
git log --format='%an <%ae>|%cn <%ce>' | Sort-Object -Unique
git log --format='%B' | Select-String -Pattern 'Co-Authored-By|Co-authored-by'
git remote -v
```

Expected: `size-pack` is below 120 MiB; only the approved identity appears; no co-author trailer appears; no remote exists. A size above 120 MiB blocks publication and requires an object-size investigation.

- [ ] **Step 6: Verify recovery assets remain intact**

```powershell
$safeSynthArchive = 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3'
Get-FileHash -Algorithm SHA256 -LiteralPath "$safeSynthArchive\figure_manifest.json"
Get-FileHash -Algorithm SHA256 -LiteralPath "$safeSynthArchive\SafeSynth-pre-filter-repo.bundle"
git bundle verify "$safeSynthArchive\SafeSynth-pre-filter-repo.bundle"
```

Expected: hashes still match `archive_receipt.json` and the bundle verifies. Do not delete the archive.

- [ ] **Step 7: Record completion and start the next separate release design**

Update the project worklog/plan to record the post-rewrite commit, final pack size, KEEP/DROP totals, verification results, and archive manifest digest without committing the machine-local absolute archive path. Commit that documentation as `kuotunyu`, then begin separate specifications for dataset/model cards and external GitHub/Hugging Face publication.
