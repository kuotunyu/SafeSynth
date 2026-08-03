# SafeSynth repository curation and history slimming design

**Date:** 2026-08-04

**Status:** Approved design, awaiting written-spec review

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

## 2. Current evidence and defect

A read-only audit, excluding the generated slimming report as an input, currently finds:

| Classification | Files | Current size |
|---|---:|---:|
| Referenced evidence to retain | 32 | 57.155 MiB |
| Unreferenced evidence to archive outside Git | 118 | 346.096 MiB |
| Total tracked figure entries | 150 | 403.251 MiB |

These counts are provisional until the corrected exact-path parser runs. The deterministic corrected audit, not a hand-maintained number, will be the source of truth.

The existing audit has a self-reference defect. `reports/repo_slimming_plan.md` lists every KEEP and DROP filename, while the audit scans tracked Markdown for figure-like filenames. On the next run, names in the report itself make DROP files look referenced, promoting almost every figure to KEEP. It also identifies references by basename, which can confuse two files with the same name in different directories.

History rewriting must not proceed until this defect is fixed and the audit output is stable.

## 3. Alternatives considered

### A. Keep all current figures in Git

This preserves every visual review artifact online, but a clean clone would still carry approximately 403 MiB of current figures after the history rewrite. Most files are intermediate labeler and review iterations already summarized by searchable technical records.

### B. Keep only evidence linked by surviving documents — selected

This retains the figures that support readable reports and decisions, archives the remainder outside Git, and is expected to reduce the pack to approximately 100–120 MiB including non-figure history. It balances auditability, clone size, and reversibility.

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

Before anything is removed or rewritten, all 150 current tracked figure entries are copied outside the repository to a date-stamped directory under:

`D:\sdg-data\02-safesynth\release_archive\`

The archive includes both KEEP and DROP files, plus a manifest containing normalized path, size, SHA-256 digest, and planned disposition. Archive creation succeeds only when every copied file hashes identically to its source and the total inventory matches the tracked inventory.

The implementation also creates a complete pre-rewrite Git bundle outside the repository. The bundle is independently verified before the owner is offered any history-rewrite command. Neither archive is deleted as part of this project.

### 4.4 Owner-operated history rewrite and restoration

The development branch is fully integrated into the main branch and the linked worktree is removed before rewriting history. The owner then runs the exact reviewed `git filter-repo` command on the main repository, in accordance with `CLAUDE.md`.

The rewrite removes `reports/figures/` from all historical commits. Only the deterministic KEEP set is restored from the verified external archive and committed with the configured identity:

`kuotunyu <61350295+kuotunyu@users.noreply.github.com>`

No automated author, committer, co-author trailer, or generated identity is permitted.

## 5. Data flow

1. Enumerate tracked Markdown and tracked `reports/figures/` files from Git.
2. Resolve actual Markdown destinations to exact tracked figure paths.
3. Produce a deterministic KEEP/DROP plan and reviewer-readable report.
4. Copy the complete figure inventory to the external archive.
5. Hash-verify the archive and verify a complete Git bundle.
6. Finish all ordinary branch integration while history is unchanged.
7. Give the owner an exact, copyable history-rewrite procedure and stop for owner confirmation.
8. After the owner rewrite, restore only KEEP files from the verified archive.
9. Run link, repository, test, licensing, identity, and size verification.
10. Only after all gates pass may the separate GitHub publication project begin.

## 6. Failure handling and rollback

The workflow fails closed:

- An unresolved, escaping, malformed, or ambiguous figure link is reported and blocks approval; it is not silently treated as DROP.
- A missing source, count mismatch, copy error, or digest mismatch blocks archive completion.
- A Git bundle that cannot be verified blocks history rewriting.
- A history rewrite is never started by the agent and is never suggested while another linked worktree remains active.
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
- output ordering and repeated generation are deterministic; and
- manifest hashes and source/archive inventories must match.

### 7.2 Pre-rewrite acceptance gates

- The curation report is stable across two consecutive runs.
- Every KEEP entry has at least one exact source document.
- Every tracked figure is present in the external archive with a matching SHA-256 digest.
- The Git bundle verifies successfully.
- The worktree passes the complete test suite, Ruff, README verification, license checks, lock-file checks, and `git diff --check`.
- `git log` and commit trailers show only the approved `kuotunyu` identity.

### 7.3 Post-rewrite acceptance gates

- Every local Markdown link or image destination resolves; this check covers all tracked Markdown, not only README.
- No DROP figure remains tracked in Git, while all DROP files remain in the external archive.
- The restored KEEP inventory matches the approved manifest exactly.
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
3. the owner has completed the reviewed history rewrite;
4. only the approved KEEP set is restored;
5. every surviving Markdown link resolves;
6. the Git pack is below 120 MiB;
7. all project verification gates pass; and
8. Git authors, committers, and trailers preserve `kuotunyu` as the sole contributor.
