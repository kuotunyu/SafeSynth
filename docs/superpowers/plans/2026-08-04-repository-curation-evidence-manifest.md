# Repository Curation Evidence Manifest Implementation Plan

> **Historical implementation plan (completed):** The unchecked boxes below preserve the
> drafted execution sequence; they are not a live backlog. Authoritative completion evidence
> is recorded in [PLAN_PHASE2.md](../../../PLAN_PHASE2.md) and
> [docs/worklog.md](../../worklog.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve exact SHA-256 auditability for all 150 curated figures while allowing the 136 approved DROP PNGs to disappear from Git and keeping the post-rewrite suite green.

**Architecture:** A byte-for-byte tracked copy of the verified v3 figure manifest becomes the repository's compact scientific commitment. A strict manifest parser and figure-state verifier accept only the complete pre-rewrite source state or exact post-rewrite 14-KEEP state; scientific tests verify their existing metadata hashes against this manifest. The archive command cross-checks the entry tuple before publishing a new v4 recovery package.

**Tech Stack:** Python 3.12 standard library, dataclasses, pathlib/PurePosixPath, hashlib, JSON, subprocess/Git, pytest, Ruff, PowerShell 5.1.

## Global Constraints

- Do not execute `git filter-repo`; the history rewrite remains owner-only.
- Do not overwrite, rename, move, or delete recovery archives v1, v2, or v3.
- The tracked manifest must be byte-identical to `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3\figure_manifest.json` with SHA-256 `aa39003c3189278eda178a39514bfa7f640655f8ddf5f1e3c2bad99380751fd5`.
- The approved manifest contains 150 entries: 14 KEEP and 136 DROP, sourced from commit `c2c6059987e71142d7c5524a52ecc2b0c4afcee5`.
- Local development may auto-detect only exact `source` or exact `curated` state; public CI and final acceptance must explicitly require `curated`.
- Missing bytes are acceptable only for manifest DROP entries whose SHA equals the calling scientific metadata SHA. KEEP bytes are always required.
- No public test or runtime verifier may read the machine-local `D:\` archive; owner-only recovery plans may name it as an explicit operational input.
- The new v4 package must be non-overwriting and its manifest entry tuple must equal the tracked canonical manifest. Its runbook source commit must equal the new exact HEAD.
- Post-rewrite Git pack target remains below 120 MiB.
- Sole Git author/committer/contributor remains `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`; no `Co-Authored-By:` trailer.
- Work is CPU-only. Do not inspect, stop, restart, or modify Docker or other projects.

---

### Task 1: Canonical manifest parser and repository-state verifier

**Files:**
- Modify: `src/release/repository_archive.py`
- Create: `src/release/figure_evidence.py`
- Create: `scripts/freeze_figure_evidence_manifest.py`
- Create: `scripts/verify_figure_evidence.py`
- Create: `tests/test_figure_evidence.py`
- Modify: `tests/test_repository_archive.py`
- Create: `reports/figure_curation_manifest.json`
- Modify: `.github/workflows/ci.yml`
- Create: `docs/superpowers/specs/2026-08-04-repository-curation-evidence-manifest-design.md`
- Create: `docs/superpowers/plans/2026-08-04-repository-curation-evidence-manifest.md`

**Interfaces:**
- Produces: `load_manifest_commitments(manifest_path: Path) -> tuple[str, tuple[ArchiveEntry, ...], str]` in `repository_archive.py`; returned values are source commit, sorted entries, and canonical manifest SHA-256.
- Produces: `FigureEvidenceError` in `figure_evidence.py`.
- Produces: `load_repository_figure_manifest(project_root: Path) -> tuple[ArchiveEntry, ...]`.
- Produces: `verify_frozen_figure(project_root: Path, entries_by_path: Mapping[str, ArchiveEntry], relative_path: str, expected_sha256: str) -> None`.
- Produces: `verify_repository_figure_state(project_root: Path, entries: Sequence[ArchiveEntry], expected_state: Literal["auto", "source", "curated"] = "auto") -> Literal["source", "curated"]`.
- Produces: `verify_curation_plan_matches_manifest(project_root: Path, dispositions: Sequence[FigureDisposition], entries: Sequence[ArchiveEntry]) -> None`.
- Produces: CLI `uv run python scripts/verify_figure_evidence.py --expected-state {source,curated,auto}`.

- [ ] **Step 1: Add strict parser tests before changing production code**

Add tests to `tests/test_repository_archive.py` proving that
`load_manifest_commitments()` accepts a canonical manifest without an adjacent
`figures/` payload and returns the literal source commit, entries, and digest.
Add mutation cases for noncanonical JSON, duplicate JSON keys, duplicate paths,
unsorted entries, unsafe paths, invalid SHA, invalid size, invalid disposition,
and invalid references.

- [ ] **Step 2: Run the focused parser tests and observe RED**

Run:

```powershell
uv run pytest tests/test_repository_archive.py -q
```

Expected: FAIL because `load_manifest_commitments` does not exist.

- [ ] **Step 3: Extract the existing strict manifest parser**

Refactor `_load_verified_manifest()` so canonical schema parsing is shared by
the new public function. Do not weaken any existing archive-payload checks.
The public function returns:

```python
(source_commit, entries, hashlib.sha256(manifest_bytes).hexdigest())
```

- [ ] **Step 4: Add figure-evidence behavior tests before implementation**

In `tests/test_figure_evidence.py`, build a real temporary Git repository with
one KEEP and one DROP entry. Use literal bytes and literal SHA-256 values. Cover:

```python
def test_source_state_requires_and_hashes_every_manifest_entry(...): ...
def test_curated_state_requires_only_keep_and_rejects_present_drop(...): ...
def test_auto_accepts_only_exact_source_or_curated_inventory(...): ...
def test_partial_mixed_extra_untracked_and_corrupt_states_fail(...): ...
def test_missing_keep_and_path_alias_fail(...): ...
def test_frozen_drop_may_be_archived_only_when_metadata_sha_matches(...): ...
def test_unknown_path_and_metadata_sha_conflict_fail(...): ...
```

The expected values must be hand-derived literals; do not compute both sides
with the implementation under test.

- [ ] **Step 5: Run the focused evidence tests and observe RED**

Run:

```powershell
uv run pytest tests/test_figure_evidence.py -q
```

Expected: FAIL because `src.release.figure_evidence` does not exist.

- [ ] **Step 6: Implement minimal evidence verification**

The verifier must compare both Git-tracked paths and the filesystem inventory.
It accepts only these path sets:

```python
source_paths = {entry.path for entry in entries}
curated_paths = {
    entry.path for entry in entries if entry.disposition == "KEEP"
}
```

For every present path, reject aliases and verify exact size and SHA-256.
`auto` returns `source` or `curated` only for an exact match. Explicit state
never falls back. Unknown, partial, mixed, extra, corrupt, or Git/filesystem
disagreement raises `FigureEvidenceError` with sorted diagnostics.

`verify_curation_plan_matches_manifest()` compares exact normalized path,
size, KEEP/DROP disposition, sorted `source_path:line_number` references, and
the SHA-256 of every current source byte. It rejects missing, duplicate, or
additional entries.

- [ ] **Step 7: Add CLI and no-overwrite freeze tests, then observe RED**

Test that the freeze command verifies the complete external archive and an
explicit expected manifest SHA before writing, refuses an existing output, and
leaves no partial file on any failure. Test verifier exit 0 for expected state
and nonzero for wrong counts/hash/state. Direct-script execution must work
without pytest path injection.

- [ ] **Step 8: Implement both CLIs and materialize the approved manifest**

The release verifier pins:

```python
APPROVED_MANIFEST_SHA256 = "aa39003c3189278eda178a39514bfa7f640655f8ddf5f1e3c2bad99380751fd5"
EXPECTED_TOTAL = 150
EXPECTED_KEEP = 14
EXPECTED_DROP = 136
```

Materialize with:

```powershell
uv run python scripts/freeze_figure_evidence_manifest.py `
  --archive 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v3' `
  --expected-manifest-sha256 'aa39003c3189278eda178a39514bfa7f640655f8ddf5f1e3c2bad99380751fd5' `
  --output 'reports\figure_curation_manifest.json'
```

Then independently run `Get-FileHash` and require the same digest. The command
must not contain a default machine-local archive path.

- [ ] **Step 9: Add the public curated-state gate to CI**

Add this CPU-only step to `.github/workflows/ci.yml`:

```yaml
- name: Verify curated figure evidence
  run: uv run python scripts/verify_figure_evidence.py --expected-state curated
```

It is intentionally release-state strict; local pre-rewrite verification uses
`--expected-state source`.

- [ ] **Step 10: Run focused tests, source-state gate, Ruff, and commit**

Run:

```powershell
uv run pytest tests/test_repository_archive.py tests/test_figure_evidence.py -q
uv run python scripts/verify_figure_evidence.py --expected-state source
uv run ruff check src/release/repository_archive.py src/release/figure_evidence.py scripts/freeze_figure_evidence_manifest.py scripts/verify_figure_evidence.py tests/test_repository_archive.py tests/test_figure_evidence.py
git diff --check
```

Expected: all PASS, exact manifest digest and counts printed. Commit as
`feat(release): track archived figure commitments` using only the approved
`kuotunyu` identity and no co-author trailer.

---

### Task 2: Migrate frozen scientific page checks without weakening SHA gates

**Files:**
- Modify: `tests/test_supervised_labeler.py`
- Test: `tests/test_figure_evidence.py`

**Interfaces:**
- Consumes: Task 1 `load_repository_figure_manifest()` and
  `verify_frozen_figure()`.
- Produces: all 21 formerly failing tests pass in exact curated state without
  accessing an external archive.

- [ ] **Step 1: Add a regression test for the exact missing-DROP contract**

Extend `tests/test_figure_evidence.py` with a hand-written metadata SHA and an
absent DROP path. Prove that identical SHA passes while a one-character digest
mutation fails. Name the production mutation it catches: accepting structured
metadata that disagrees with the canonical manifest.

- [ ] **Step 2: Run the regression test and observe RED if the Task 1 API is incomplete**

Run:

```powershell
uv run pytest tests/test_figure_evidence.py -q
```

Expected: the new test must either expose the missing behavior and fail before
the minimal fix, or document that Task 1 already supplied the exact behavior;
in the latter case, use the already observed 21-test post-rewrite failure as
the RED evidence and do not add a tautological test.

- [ ] **Step 3: Load the canonical entries once in the supervised-labeler tests**

Create one test-local mapping from
`load_repository_figure_manifest(PROJECT_ROOT)` and one small wrapper that
normalizes `Path` to repository-relative POSIX form before calling
`verify_frozen_figure()`. Do not copy parser or hash-policy logic into the test.

- [ ] **Step 4: Replace only the 78 physical-PNG assertions**

In the 20 versioned GT/model review tests, keep every existing semantic,
registration, independence, count, path, and exact SHA assertion. Replace only
`path.is_file()` / `path.read_bytes()` page checks with the common verifier and
the existing metadata SHA. Do not alter JSON/YAML evidence files.

Replace
`test_figure_cleanup_keeps_current_evidence_and_removes_legacy_duplicates`
with an exact call to `verify_repository_figure_state(..., "auto")`. This gate
must cover the complete indexed inventory rather than six filenames and globs.

- [ ] **Step 5: Run the focused source-state tests**

Run:

```powershell
uv run pytest tests/test_supervised_labeler.py tests/test_figure_evidence.py -q
```

Expected: PASS with every present source PNG re-hashed.

- [ ] **Step 6: Rehearse curated state in a disposable detached worktree**

Create a short-path disposable worktree from the task HEAD, remove
`reports/figures/` in a detached rehearsal commit, restore exactly the 14 KEEP
paths from v3, and run:

```powershell
uv run pytest tests/test_supervised_labeler.py tests/test_figure_evidence.py -q
uv run python scripts/verify_figure_evidence.py --expected-state curated
```

Expected: PASS with no `D:\` access by tests. Remove the disposable worktree
without `--force`; never modify the implementation branch during rehearsal.

- [ ] **Step 7: Run Ruff, diff check, and commit**

Run:

```powershell
uv run ruff check tests/test_supervised_labeler.py tests/test_figure_evidence.py
git diff --check
```

Commit as `test(release): verify archived figure commitments` using only the
approved identity and no co-author trailer.

---

### Task 3: Bind v4 archive creation and the formal owner plan to the manifest

**Files:**
- Modify: `scripts/archive_repository_curation.py`
- Modify: `tests/test_repository_archive.py`
- Modify: `docs/superpowers/specs/2026-08-04-repository-curation-history-slimming-design.md`
- Modify: `docs/superpowers/plans/2026-08-04-repository-curation-history-slimming.md`
- Modify: `tests/test_plan_repo_slimming.py`

**Interfaces:**
- Consumes: Task 1 manifest parser, plan comparison, and source-state verifier.
- Produces: a staged recovery package is publishable only when its entry tuple
  exactly matches the tracked canonical manifest.
- Produces: the owner plan names only non-overwriting v4 for Stage 1 and marks
  v1/v2/v3 as recovery-only and forbidden for the owner gate.

- [ ] **Step 1: Add archive mismatch tests before implementation**

Add direct and CLI tests showing archive publication fails without a tracked
manifest, on plan disposition/reference mismatch, on current-byte/hash
mismatch, and when the produced archive entry tuple differs. Assert no
destination or private staging root remains. Add a success test with a
canonical two-entry fixture manifest.

- [ ] **Step 2: Run focused archive tests and observe RED**

Run:

```powershell
uv run pytest tests/test_repository_archive.py -q
```

Expected: FAIL because archive creation does not yet bind the tracked manifest.

- [ ] **Step 3: Implement pre-publication and post-build equality gates**

Before staging, load `reports/figure_curation_manifest.json`, require exact
`source` state, and compare every plan field with the manifest. After the
guarded archive command's internal builder completes and before publishing,
require `recovery.entries == tracked_manifest_entries`. Compare entries only; the
tracked manifest's historical source commit remains v3 while the recovery
package source commit is the current v4 HEAD.

- [ ] **Step 4: Amend the original design and owner plan**

Record the evidence-manifest contract, the 21-test dry-run finding, and the
required curated-state verifier. Replace every owner-gate v3 path with:

```text
D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v4
```

State that v1, v2, and v3 are immutable recovery snapshots and forbidden for
the owner gate. Task 7 must verify the v4 exact source commit and stop. Task 8
must restore from v4 and run the curated-state verifier before tests.

- [ ] **Step 5: Update formal-plan behavior tests**

Change `tests/test_plan_repo_slimming.py` to require v1/v2/v3 retirement,
v4-only Task 7/8 paths, tracked manifest verification, and explicit curated
state. Test behavior and required gates, not incidental prose layout.

- [ ] **Step 6: Run focused and full pre-rewrite gates**

Run:

```powershell
uv run pytest tests/test_repository_archive.py tests/test_plan_repo_slimming.py -q
uv run pytest -q
uv run ruff check .
uv run python scripts/verify_figure_evidence.py --expected-state source
uv run python scripts/verify_readme.py
uv run python scripts/verify_repository_links.py
uv lock --check
git diff --check
```

Expected: all PASS. Commit as
`fix(release): bind curation archive to evidence manifest` using only the
approved identity and no co-author trailer.

---

### Task 4: Independent review, post-rewrite rehearsal, and v4 recovery package

**Files:**
- No implementation files expected unless review finds a defect.
- External create only: `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v4\`

**Interfaces:**
- Consumes: reviewed Task 1-3 commits.
- Produces: independently approved branch, passing disposable post-rewrite
  rehearsal, and independently verified non-overwriting v4 recovery package.

- [ ] **Step 1: Complete task-scoped and whole-branch reviews**

Require both spec-compliance and code-quality approval for each task, then a
whole-branch review of the complete base-to-head diff. Resolve every Critical
or Important finding through the bounded review loop.

- [ ] **Step 2: Run a complete disposable Task 8 rehearsal**

In a short-path detached worktree, remove all figures, restore only v3 KEEP
bytes, and run the full pytest, Ruff, source-independent curated figure gate,
README verifier, repository-link verifier, lock check, and diff check. Require
zero failures and exact 14 KEEP paths. Remove the worktree without force.

- [ ] **Step 3: Integrate the reviewed branch and re-run all gates**

Fast-forward `main` only after review passes. Verify clean status, exact commit
identity, no co-author trailer, and no remote changes.

- [ ] **Step 4: Create v4 without overwriting older archives**

Run the archive command with v4 destination and exact owner root. Require 150
entries, 14 KEEP, 136 DROP, the new exact source commit, canonical manifest,
verified Git bundle, tracked-manifest entry equality, and a Stage 1-only
runbook. Independently verify all hashes and bundle contents.

- [ ] **Step 5: Stop at the owner gate**

Do not execute history rewriting. Provide the user the exact v4 runbook path
and ask them to run only Stage 1 and return the complete output. v1-v3 remain
untouched.
