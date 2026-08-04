# SafeSynth repository curation and history slimming design

**Date:** 2026-08-04

**Status:** Approved written specification

**Scope:** Repository figure curation, recoverable archival, and pre-publication Git history slimming only

## 1. Purpose

SafeSynth is not yet connected to a remote repository, but its local Git object pack is approximately 631 MiB. Historical versions of `reports/figures/` account for approximately 629.7 MB (94%) of all historical file bytes. The current checkout contains approximately 403.2 MiB of tracked figures.

The repository should remain technically auditable without forcing every future clone to download hundreds of megabytes of obsolete review sheets. The selected approach is **curated evidence**:

- keep figures that a surviving tracked Markdown document actually links to;
- remove unreferenced figures from Git while retaining a verified external archive;
- remove old versions of the entire figure directory from Git history;
- restore only the curated current figures in one fresh commit; and
- preserve the invariant that every Git/GitHub contributor is `kuotunyu`.

This work does not publish to GitHub or Hugging Face. Repository creation, remote configuration, push, Hugging Face upload, and history rewriting remain separate owner-operated release actions.

## 2. Current evidence and resolved defects

A canonical exact-path manifest and a fresh source-state verifier currently find:

| Classification | Files | Current size |
|---|---:|---:|
| Referenced evidence to retain | 14 | 2.147 MiB |
| Unreferenced evidence to archive outside Git | 136 | 401.104 MiB |
| Total tracked figure entries | 150 | 403.251 MiB |

The tracked manifest SHA-256 is
`aa39003c3189278eda178a39514bfa7f640655f8ddf5f1e3c2bad99380751fd5`.
The deterministic manifest and verifier, not a hand-maintained list, are the
source of truth.

The original audit had a self-reference defect. `reports/repo_slimming_plan.md`
listed every KEEP and DROP filename, while the audit scanned tracked Markdown
for figure-like filenames. On the next run, names in the report itself made DROP
files look referenced. It also identified references by basename, which could
confuse two files with the same name in different directories. The implemented
exact-path collector, canonical manifest, and regression tests resolve both
defects.

A later all-refs rehearsal found a separate history-slimming defect. The Codex
desktop app maintains a direct tree ref under `refs/codex/turn-diffs/`. The ref
points to the exact current `HEAD^{tree}`, but `git-filter-repo` does not rewrite
tree-only refs. Leaving it in place kept 170 historical figure paths reachable
and left a 406.31 MiB pack even though the rewritten branch history itself was
correct. A guarded rehearsal that removed only refs equal to the approved source
tree reduced reachable historical figure paths to zero and the pack to 4.59 MiB.
The v4 archive remains a valid immutable recovery package, but its Stage 1
runbook is superseded and must not be executed. History rewriting may proceed
only from a newly verified, non-overwriting v5 package and runbook.

## 3. Alternatives considered

### A. Keep all current figures in Git

This preserves every visual review artifact online, but a clean clone would still carry approximately 403 MiB of current figures after the history rewrite. Most files are intermediate labeler and review iterations already summarized by searchable technical records.

### B. Keep only evidence linked by surviving documents — selected

This retains the figures that support readable reports and decisions, archives
the remainder outside Git, and keeps the final pack below the 120 MiB acceptance
ceiling. It balances auditability, clone size, and reversibility.

### C. Keep only README presentation images

This would minimize size, but it would break or weaken technical reports and remove evidence needed to understand model, labeler, and synthesis decisions. It is too aggressive.

## 4. Boundaries and responsibilities

The work is divided into four isolated components.

### 4.1 Reference collector

The collector reads tracked Markdown files and returns exact repository-relative figure paths that appear in real Markdown links or image destinations. It must:

- resolve relative links from the directory containing the Markdown file;
- normalize separators and harmless URL encoding without leaving the repository root;
- recognize links to tracked files under `reports/figures/`;
- use the full normalized path rather than a basename;
- ignore ordinary prose, code spans, and code fences that merely mention a filename;
- exclude the generated `reports/repo_slimming_plan.md` from reference inputs; and
- produce sorted deterministic results.

Tracked Markdown under `reports/figures/` remains a valid reference source unless it is generated output. This preserves deliberately maintained figure indexes.

### 4.2 Curation planner

The planner compares the tracked figure inventory with the collector output. For every entry, it records:

- repository-relative path;
- byte size;
- KEEP or DROP disposition;
- the document links that justify KEEP; and
- a clear reason for DROP when no surviving document links to it.

The generated human-readable report is derived output and must never become an input to its own next run. Re-running the planner twice without repository changes must produce byte-identical output.

### 4.3 External archive builder

Before anything is removed or rewritten, all current tracked figure entries are
copied outside the repository to the approved, non-overwriting owner-controlled
recovery destination. Its machine-local location belongs only in the owner plan,
not in public documentation or tests.

The archive includes both KEEP and DROP files, plus a manifest containing normalized path, size, SHA-256 digest, and planned disposition. Archive creation succeeds only when every copied file hashes identically to its source and the total inventory matches the tracked inventory.

The archive command is bound to the tracked canonical manifest. Before staging or
publication it loads that manifest, verifies the complete source state, and
requires every current curation-plan field (path, byte size, disposition, and
reference sources) to agree with it. After building the recovery package and
before publication, its entry tuple must exactly match the tracked manifest
entries. The historical source commit in the manifest is evidence provenance,
not a requirement that the new package source equal it. This contract follows the
21-test dry-run finding that archive contents otherwise could drift from the
reviewed evidence index.

The implementation also creates a complete pre-rewrite Git bundle outside the
repository. The bundle is independently verified before the owner is offered any
history-rewrite command. It preserves the approved source commit and all refs
present at archive time, including any direct tree refs. Existing v1-v4 recovery
packages remain immutable; the safety correction creates a new non-overwriting
v5 package. No recovery package is deleted as part of this project.

### 4.4 Owner-operated history rewrite and restoration

The development branch is fully integrated into the main branch and every linked
worktree is removed before rewriting history. The owner copies the reviewed v5
command, fully exits Codex and any editor that may modify the repository, and
then runs the v5 Stage 1 runbook in an external Windows PowerShell process, in
accordance with `CLAUDE.md`.

The v5 runbook is bound to the exact archived source commit and its source tree.
Before any ref deletion, it performs one complete preflight over every ref under
`refs/codex/turn-diffs/`. Zero or more refs are allowed, but every discovered ref
must:

- remain in the exact `refs/codex/turn-diffs/` namespace;
- resolve to an object of type `tree`; and
- equal the archived source commit's exact `HEAD^{tree}` object.

If any ref fails any condition, the runbook stops before changing a ref or
rewriting history. After the complete preflight succeeds, each redundant ref is
deleted conditionally with both its exact ref name and expected old object ID.
A concurrent ref change therefore makes Git reject the deletion. Any deletion
failure stops before history rewriting. The runbook then verifies that the
namespace is empty before invoking the exact reviewed `git-filter-repo` command.

After rewriting, Stage 1 must verify across all remaining refs that no reachable
object path begins with `reports/figures/`, print the object-pack report, and end
with the mandatory STOP/report-back instruction. It contains no restoration,
staging, commit, remote, or push operation. A residual figure path blocks
restoration and is diagnosed from the verified bundle.

Only after the owner reports the complete Stage 1 output does the agent perform
a read-only checkpoint. The deterministic KEEP set is then restored from the
verified v5 archive and committed with the configured identity:

`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`

No automated author, committer, co-author trailer, or generated identity is permitted.

## 5. Data flow

1. Enumerate tracked Markdown and tracked `reports/figures/` files from Git.
2. Resolve actual Markdown destinations to exact tracked figure paths.
3. Produce a deterministic KEEP/DROP plan and reviewer-readable report.
4. Copy the complete figure inventory to the external archive.
5. Finish all ordinary implementation, documentation, and branch integration
   while history is unchanged.
6. Build and hash-verify the v5 archive from that final clean source commit and
   verify its complete all-refs Git bundle.
7. Rehearse the exact all-refs path in an isolated mirror, including the Codex
   tree-ref guard, and require zero reachable historical figure paths.
8. Give the owner an exact, copyable v5 Stage 1 procedure and require Codex and
   repository-writing editors to be closed before execution.
9. Guard and remove only redundant Codex tree refs, rewrite history, verify all
   refs, stop, and obtain the owner's complete output.
10. After the read-only checkpoint, restore only KEEP files from the verified v5
    archive.
11. Verify `verify_figure_evidence.py --expected-state curated`, then run link,
    repository, test, licensing, identity, recovery, and size verification.
12. Only after all gates pass may the separate GitHub publication project begin.

## 6. Failure handling and rollback

The workflow fails closed:

- An unresolved, escaping, malformed, or ambiguous figure link is reported and blocks approval; it is not silently treated as DROP.
- A missing tracked canonical manifest, source-state mismatch, curation-plan mismatch, entry-tuple mismatch, count mismatch, copy error, or digest mismatch blocks archive completion before publication.
- A Git bundle that cannot be verified blocks history rewriting.
- The v4 runbook is superseded and must not be offered or executed.
- A history rewrite is never started by the agent and is never suggested while another linked worktree remains active.
- A Codex turn-diff ref outside the expected namespace, with a non-tree object,
  or with an object different from the archived source tree blocks all ref
  deletion and history rewriting.
- A conditional ref deletion failure or a non-empty turn-diff namespace after
  deletion blocks history rewriting.
- Any `reports/figures/` path reachable from any ref after rewriting blocks
  restoration, committing, and publication.
- If owner-operated rewriting fails, the original repository remains recoverable from the verified Git bundle, while all current figures remain recoverable from the SHA-256 archive.
- If a required figure was misclassified, it is restored from the archive in a corrective `kuotunyu` commit before publication.

No command in the automated preparation phase deletes the external archive, force-pushes, creates a remote, or uploads data.

## 7. Testing strategy

### 7.1 Unit tests

Tests will cover:

- the generated slimming report cannot promote its own DROP entries to KEEP;
- relative Markdown links resolve from the containing document;
- identical basenames in different directories remain distinct;
- code-fence, code-span, and plain-text mentions do not count as links;
- Markdown links and image links to figures count as references;
- path traversal and links outside the repository fail closed;
- output ordering and repeated generation are deterministic;
- manifest hashes, complete source state, curation-plan fields, and source/archive entry tuples must match;
- the v5 runbook accepts zero, one, or multiple exact source-tree Codex refs;
- it rejects a wrong tree, non-tree object, malformed ref result, or native Git
  failure before invoking `git-filter-repo`;
- it validates the complete ref set before deleting any ref;
- conditional deletion detects a concurrent ref update and fails closed; and
- post-rewrite all-ref verification rejects any surviving historical figure
  path before the mandatory STOP.

### 7.2 Pre-rewrite acceptance gates

- The curation report is stable across two consecutive runs.
- Every KEEP entry has at least one exact source document.
- Every tracked figure is present in the external archive with a matching SHA-256 digest.
- The Git bundle verifies successfully.
- A disposable all-refs mirror reproduces the source refs, safely handles the
  Codex tree ref, reaches zero historical figure paths, passes strict Git fsck,
  and reports a pack below 120 MiB before the owner receives the runbook.
- The worktree passes the complete test suite, Ruff, README verification, license checks, lock-file checks, and `git diff --check`.
- `git log` and commit trailers show only the approved `kuotunyu` identity.

### 7.3 Post-rewrite acceptance gates

- Every local Markdown link or image destination resolves; this check covers all tracked Markdown, not only README.
- No DROP figure remains tracked in Git, while all DROP files remain in the external archive.
- `git rev-list --objects --all` exposes no historical `reports/figures/` path
  other than the exact KEEP objects introduced by the single restoration commit.
- The restored KEEP inventory matches the approved manifest exactly.
- `verify_figure_evidence.py --expected-state curated` passes before the post-rewrite test suite.
- `git count-objects -vH` reports a target pack size below 120 MiB. A larger pack blocks publication and triggers investigation rather than weakening the target silently.
- The complete verification suite passes again.
- All rewritten commits and the restoration commit preserve the contributor invariant.

## 8. Non-goals

This design does not cover:

- GitHub repository creation, push, branch protection, or Actions secrets;
- Hugging Face dataset or model publication;
- dataset-card or model-card content;
- RF-DETR retraining or additional GPU experiments;
- publishing fine-tuned RF-DETR latency numbers that have not passed the contention gate; or
- deleting the external recovery archive after publication.

Those concerns require separate designs or release checklists after repository curation is complete.

## 9. Completion criteria

Repository curation and history slimming are complete only when:

1. reference classification is exact-path, deterministic, and tested;
2. every pre-rewrite figure and the full Git history have verified external recovery copies;
3. the owner has completed the reviewed v5 history rewrite after the guarded
   Codex tree-ref preflight;
4. immediately before restoration, all refs are free of every
   `reports/figures/` path;
5. after restoration, the only figure history is one new commit containing the
   approved KEEP set;
6. every surviving Markdown link resolves;
7. the Git pack is below 120 MiB;
8. all project and recovery verification gates pass; and
9. Git authors, committers, and trailers preserve `kuotunyu` as the sole contributor.
