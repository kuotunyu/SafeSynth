# Repository Curation and History Slimming Implementation Plan

> **Historical implementation plan (completed):** The unchecked boxes below preserve the
> drafted execution sequence; they are not a live backlog. Authoritative completion evidence
> is recorded in [PLAN_PHASE2.md](../../../PLAN_PHASE2.md) and
> [docs/worklog.md](../../worklog.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the unpublished SafeSynth Git repository to a verified pack below 120 MiB while retaining every figure linked by surviving documentation, archiving every current figure externally, and preserving `kuotunyu` as the sole contributor.

**Architecture:** A shared Markdown-link resolver produces exact repository-relative targets. A curation planner uses those targets to generate a deterministic KEEP/DROP inventory, while a separate archive module copies and hashes the complete figure tree, creates a verified Git bundle, and restores only KEEP files after the owner-operated history rewrite. A repository-wide link verifier supplies the final fail-closed acceptance gate.

**Tech Stack:** Python 3.12 standard library, dataclasses, pathlib/PurePosixPath, hashlib, json, shutil, subprocess/Git, pytest, Ruff, PowerShell, `git-filter-repo` (owner-operated in the formal repository; agent-operated only through a generated runbook in a disposable rehearsal clone).

## Final-review amendment — archive rebound and staged owner gate

The v1 archive at
`D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation`
is an immutable recovery-only package for source commit
`2c2d3ff5198ff600220e5b1e1c606ebc80e07a98`. The v2 archive at
`D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v2`
is an immutable recovery-only package for source commit
`f514950b142da95bb4c71d3626b9417fb25a3bff`. The v3 and v4 packages are
also immutable recovery-only packages. The v1, v2, and v3 immutable recovery snapshots and the v4 immutable recovery-only package are forbidden for the owner gate: the v1-v4 runbooks must not be executed, and none of those packages may be
offered as an owner gate. The sole owner-gate package is the non-overwriting v5 destination
`D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5`.

The archive command must load the tracked canonical manifest, verify exact source
bytes and every curation-plan field before staging or publication, and require the
produced recovery entry tuple to exactly match the tracked manifest entries before
publication. The manifest's historical source commit remains evidence history;
the v5 recovery package records the current clean HEAD. The 21-test dry-run
finding established that this binding is required before an owner may be offered
the Stage 1 runbook.

## Global Constraints

- Keep only files under `reports/figures/` that a surviving tracked Markdown document links to through an exact normalized path.
- Exclude generated `reports/repo_slimming_plan.md` from reference inputs so it cannot promote its own DROP entries to KEEP.
- Treat unresolved, escaping, malformed, or ambiguous local links as blocking failures; never silently classify them as DROP.
- Archive every current tracked file under `reports/figures/`, including both KEEP and DROP, only under `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5`.
- Record byte size and SHA-256 for every archived file and verify source/archive equality before allowing history rewrite.
- Create and verify a complete pre-rewrite Git bundle outside the repository.
- Do not overwrite or delete an existing archive destination.
- No agent may run the packaged owner runbook or `git-filter-repo` in the formal repository. An agent may run only a newly generated rehearsal runbook pointed at a disposable clone; `git-filter-repo` may execute only inside that disposable rehearsal. The owner alone runs the exact reviewed packaged command in the formal repository.
- The agent must not create a remote, push, force-push, or upload to Hugging Face.
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
- Internal only: the guarded `scripts/archive_repository_curation.py` command
  uses its internal `_create_recovery_package` builder to produce an
  `ArchiveReceipt`; there is no public raw-package-builder contract.
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

Copy files to `destination/figures/<repository-relative-path>`. Reject existing destinations, missing sources, duplicate paths, path escapes, size mismatches, digest mismatches, manifest schema mismatches, and destination KEEP files that already exist. Invoke `git bundle create <path> --all`, then `git bundle verify <path>`, and delete only the newly created invalid bundle if verification fails. The internal builder used only by the guarded archive command composes the verified archive and bundle and returns the canonical data used to write the receipt.

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

The guarded `scripts/archive_repository_curation.py` command must compute the real tracked inventory, plan it, use its internal `_create_recovery_package` builder, verify the returned archive and bundle, and only then write `archive_receipt.json` and the runbook. Its success output must state counts, digests, source commit, and destination. It must not print or execute `git-filter-repo` before every verification passes. This guarded command is the sole supported archive-creation interface.

The generated PowerShell runbook is Stage 1 only. It must use these owner steps
in this order, explicitly checking `$LASTEXITCODE` after every native command:

```powershell
$ErrorActionPreference = 'Stop'
$OwnerProjectRoot = '<project_root>'
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

### Task 6: Final tracked review, local integration, and linked-worktree retirement

Tasks 6 through 8 below supersede the earlier archive-before-integration sequence.
The single detailed executable source of truth for the destructive boundary is
`docs/superpowers/plans/2026-08-04-repository-curation-v5-tree-ref-safety.md`,
Tasks 4 and 5. If these summary gates and that v5 safety plan ever differ, stop;
do not improvise or fall back to a v1-v4 runbook.

**Files:**
- Final-review branch: `codex/repository-curation-v5`
- Linked worktree to retire: `<project_root>/.worktrees/repository-curation-v5`
- Formal repository: `<project_root>`

- [ ] **Step 1: Complete final tracked review and clean-state verification**

Review the complete tracked diff and commit sequence from `main` through
`codex/repository-curation-v5`. Require both the formal repository and linked
worktree to be clean, the feature branch to descend from `main`, all author and
committer identities to be `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`,
and no co-author trailer. Any unexplained file, commit, identity, or trailer
blocks integration.

- [ ] **Step 2: Fast-forward the feature branch into local main**

```powershell
git -C '<project_root>' merge --ff-only codex/repository-curation-v5
```

Require a fast-forward with no merge commit. Do not create a remote or publish.

- [ ] **Step 3: Rerun the complete verification suite on the merged result**

```powershell
Set-Location -LiteralPath '<project_root>'
uv run python scripts/verify_figure_evidence.py --expected-state source
uv run pytest -q
uv run ruff check .
uv run python scripts/verify_readme.py
uv run python scripts/verify_repository_links.py
uv run python scripts/check_forbidden_licences.py
uv lock --check
git diff --check
git status --short --branch
```

Every command must pass and the merged formal repository must remain clean.

- [ ] **Step 4: Remove only the clean linked worktree and merged feature branch**

Reconfirm the linked worktree is clean, remove exactly
`.worktrees/repository-curation-v5` without `--force`, delete the merged
`codex/repository-curation-v5` branch with the safe merged-branch deletion, and
require `git worktree list --porcelain` to contain exactly one `worktree ` record.
Do not proceed while any other registered worktree or the merged branch remains.

```powershell
git -C '<project_root>/.worktrees/repository-curation-v5' status --short --branch
git -C '<project_root>' worktree remove '<project_root>/.worktrees/repository-curation-v5'
git -C '<project_root>' branch -d codex/repository-curation-v5
git -C '<project_root>' worktree list --porcelain
```

---

### Task 7: Create v5 only after integration and prove it in a disposable all-refs clone

Follow Task 4 of the v5 tree-ref safety plan for the exact operational procedure.
This task is a mandatory rehearsal gate, not permission to mutate the formal
repository. The v1-v4 packages remain immutable recovery-only assets and none of
their runbooks may be executed or offered.

**Files:**
- Formal repository: `<project_root>`
- External create: `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5\`
- Disposable rehearsal: one unique clone outside the formal repository

- [ ] **Step 1: Verify the formal source state before creating v5**

Require all of the following together: clean status; exactly one worktree;
source-state figure evidence passes; `git remote -v` is empty; and the exact v5
destination does not exist. A mismatch is a blocking failure and must not be
worked around by overwriting, renaming, or reusing an archive.

- [ ] **Step 2: Create the non-overwriting v5 package with all roots explicit**

```powershell
uv run python scripts/archive_repository_curation.py --project-root '<project_root>' --destination 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5' --owner-project-root '<project_root>'
```

- [ ] **Step 3: Independently verify every receipt commitment**

Using the strict receipt loader and independent digest/count checks, require
exactly 150 archived entries, `KEEP=14`, `DROP=136`, payload size `422839340`
bytes, every entry size and SHA-256, the receipt-matching manifest and bundle
SHA-256 values, a passing `git bundle verify`, and receipt source HEAD equal to
the current clean formal `HEAD`. Rerun source-state verification and prove that
archive creation did not change the formal repository.

- [ ] **Step 4: Rehearse the generated logic in a disposable all-refs clone**

Build a normal disposable clone from the verified bundle. Materialize every
ordinary bundled branch, including `main` and `codex/rfdetr-four-arm`, and add
one or more exact `refs/codex/turn-diffs/` refs pointing to the approved source
tree. Generate a fresh runbook with that disposable clone as its root and run it
only there. This generated rehearsal runbook is the sole agent exception for
executing `git-filter-repo`; never run or edit the packaged owner runbook and
never point a rehearsal runbook at the formal repository.

Require conditional deletion of the exact Codex refs, successful rewriting of
every ordinary ref, zero reachable historical `reports/figures/` paths across
all refs, `git fsck --full --strict` success, a reported pack below 120 MiB, and
the mandatory STOP as the last action.

- [ ] **Step 5: Restore exact KEEP in the rehearsal and run curated acceptance**

Restore from v5 into the disposable clone, stage exactly the 14 manifest KEEP
paths and zero DROP paths, commit once as the approved identity, and run the
complete curated suite specified by the v5 safety plan, including the forbidden-
license scan, strict fsck, and count/pack threshold. Require no remote, only the
approved author/committer identity, and no co-author trailer.

- [ ] **Step 6: Preserve evidence and reverify the formal repository**

Quarantine the rehearsal clone non-destructively if deletion is not permitted.
Reverify the unchanged formal HEAD, clean source state, exactly one worktree, and
no remote. Only a fully passing rehearsal unlocks Task 8.

---

### Task 8: Owner-only formal rewrite, controller checkpoint, and final acceptance

Follow Task 5 of the v5 tree-ref safety plan as the single detailed source of
truth. Nothing in this task authorizes an agent to execute the packaged owner
runbook or `git-filter-repo` in the formal repository.

- [ ] **Step 1: Offer the owner-only formal runbook command after rehearsal passes**

Tell the owner to copy the reviewed v5 command, fully close Codex and all
editors, run it in an external Windows PowerShell 5.1 process, wait for its
mandatory STOP, reopen Codex only after STOP, and return the complete output.
Do not offer any v1-v4 command and do not supply the raw internal
`git-filter-repo` line as a standalone command.

```powershell
& ([scriptblock]::Create([System.IO.File]::ReadAllText('D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5\OWNER_HISTORY_REWRITE_RUNBOOK.txt', [System.Text.UTF8Encoding]::new($false))))
```

- [ ] **Step 2: Stop at the controller boundary**

The runbook must finish with its mandatory STOP and contain no restoration,
staging, commit, remote, push, or publication action. Wait for the full owner
output. Do not infer success from process disappearance and do not restore any
figure before the read-only checkpoint passes.

- [ ] **Step 3: Require the complete read-only pre-restoration checkpoint**

Before any restoration, independently require every item below:

- clean status;
- exactly one registered worktree;
- an empty `refs/codex/turn-diffs/` namespace;
- zero reachable `reports/figures/` paths across `git rev-list --objects --all`;
- passing `git fsck --full --strict`;
- successful `git count-objects -vH` with the reported pack below 120 MiB;
- no remote;
- every author and committer equal to `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`; and
- no co-author trailer.

Any failed or ambiguous check stops the workflow without restoring files.

- [ ] **Step 4: Restore and commit the exact KEEP set**

Use the strict v5 restore command, stage only `reports/figures/`, compare the
staged paths and digests exactly with the 14 manifest KEEP entries, require zero
DROP paths, and make the single restoration commit as the approved identity.

```powershell
uv run python scripts/restore_curated_figures.py --project-root '<project_root>' --archive 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5'
git add -- reports/figures
git diff --cached --check
git diff --cached --name-only
```

- [ ] **Step 5: Run complete final acceptance**

```powershell
uv run python scripts/verify_figure_evidence.py --expected-state curated
uv run pytest -q
uv run ruff check .
uv run python scripts/verify_readme.py
uv run python scripts/verify_repository_links.py
uv run python scripts/check_forbidden_licences.py
uv lock --check
git diff --check
git fsck --full --strict
git count-objects -vH
git status --short --branch
```

In addition to command success, require a clean tree, a pack below 120 MiB,
exactly 14 KEEP and zero DROP figure paths, no remote, only the approved author
and committer identity, no co-author trailer, and unchanged v5 manifest and
bundle hashes with a still-verifying bundle. Only then record final acceptance;
GitHub, Hugging Face, model/dataset cards, and latency publication remain
separate future work.
