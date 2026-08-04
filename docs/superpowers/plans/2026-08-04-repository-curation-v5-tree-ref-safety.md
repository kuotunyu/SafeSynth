# Repository Curation v5 Tree-Ref Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and independently prove a fail-closed v5 owner runbook that safely removes only approved Codex tree refs, rewrites every ordinary Git ref free of historical `reports/figures/` paths, and stops before the curated evidence restoration.

**Architecture:** Keep the archive format and `_runbook(owner_project_root: str, source_commit: str) -> str` interface stable. Strengthen the generated PowerShell runbook with a complete pre-mutation ref/worktree/tool preflight, conditional tree-ref deletion, all-ref post-rewrite checks, strict object verification, and a mandatory STOP. Bind the resulting runbook and the full current history to a new immutable v5 recovery package, then validate it in a disposable all-refs clone before offering the owner-only command.

**Tech Stack:** Python 3.12 standard library, pytest, Ruff, PowerShell 5.1, native Git, `git-filter-repo` through `uvx`, SHA-256, and the existing SafeSynth archive/restore verification modules.

## Global Constraints

- Use the approved design at `docs/superpowers/specs/2026-08-04-repository-curation-history-slimming-design.md` as the source of truth.
- Preserve `_runbook(owner_project_root: str, source_commit: str) -> str`; derive the expected source tree inside the generated runbook from the bound source commit.
- Accept zero or more refs only inside `refs/codex/turn-diffs/`; every accepted object must be a tree equal to the exact archived `HEAD^{tree}`.
- Validate the complete discovered ref set, clean worktree, exact HEAD, exactly one registered worktree, and `git-filter-repo` availability before deleting any ref.
- Delete each approved ref with `git update-ref -d <ref> <expected-old-oid>`; any concurrent update or native-command failure stops before history rewriting.
- Do not use `git update-ref --stdin`; Windows PowerShell 5.1 can prepend a BOM to the transaction input.
- After rewriting, require zero reachable `reports/figures/` paths across `git rev-list --objects --all`, a passing `git fsck --full --strict`, and a successful `git count-objects -vH` before the mandatory STOP message.
- The generated runbook must contain no restore, stage, commit, remote, push, or publication action.
- Treat v1 through v4 packages as immutable recovery-only assets. Never overwrite, edit, delete, or offer their runbooks.
- Create v5 only at `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5`; an existing destination is a blocking failure.
- The agent must never run the owner runbook in the formal SafeSynth repository. Only the owner may do so after fully closing Codex and editors.
- All commits must use `kuotunyu <61350295+kuotunyu@users.noreply.github.com>` as both author and committer and contain no co-author trailer.
- Stage explicit paths only; never use `git add -A` or `git add .`.
- Do not create a Git remote, push, force-push, publish to GitHub/Hugging Face, touch Docker/other projects, or use GPU.

---

### Task 1: Guard Codex tree refs before any rewrite

**Files:**
- Modify: `tests/test_repository_archive.py`
- Modify: `scripts/archive_repository_curation.py`

**Interfaces:**
- Preserve: `_runbook(owner_project_root: str, source_commit: str) -> str`.
- Extend test helper: `_execute_runbook_with_fake_native_commands` with explicit worktree, source-tree, `for-each-ref`, conditional-deletion, and `uvx` scenario inputs.
- Generated PowerShell state: `$ExpectedSourceTree`, `$WorktreeRows`, `$TurnDiffRows`, and an in-memory validated collection of ref name/object/type triples.

- [ ] **Step 1: Add failing tests for clean preflight ordering**

Add tests named:

- `test_owner_runbook_accepts_zero_exact_codex_tree_refs`
- `test_owner_runbook_accepts_one_exact_codex_tree_ref`
- `test_owner_runbook_accepts_multiple_exact_codex_tree_refs`
- `test_owner_runbook_rejects_multiple_registered_worktrees_before_ref_deletion`
- `test_owner_runbook_checks_filter_repo_availability_before_ref_deletion`

The fake command log must prove this order: clean status, exact HEAD, source tree, worktree inventory, complete turn-diff inventory, tool availability, conditional deletions, empty-namespace verification, rewrite. For zero refs, no `update-ref` command appears.

- [ ] **Step 2: Run the new preflight tests and confirm RED**

```powershell
uv run pytest tests/test_repository_archive.py -q -k "codex_tree_ref or registered_worktrees or availability_before_ref_deletion"
```

Expected: FAIL because the current runbook neither inspects worktrees/source trees nor enumerates/deletes Codex refs.

- [ ] **Step 3: Add failing tests for fail-closed ref validation**

Add tests named:

- `test_owner_runbook_rejects_wrong_codex_tree_before_any_deletion`
- `test_owner_runbook_rejects_non_tree_codex_object_before_any_deletion`
- `test_owner_runbook_rejects_malformed_codex_ref_row_before_any_deletion`
- `test_owner_runbook_rejects_out_of_namespace_ref_before_any_deletion`
- `test_owner_runbook_validates_all_codex_refs_before_any_deletion`
- `test_owner_runbook_stops_before_rewrite_when_conditional_ref_delete_fails`
- `test_owner_runbook_stops_when_codex_namespace_is_not_empty_after_deletion`

Each rejection must assert that no `git update-ref` occurs when complete-set validation fails and no `git-filter-repo --path ...` occurs for any deletion or empty-namespace failure.

- [ ] **Step 4: Run the rejection tests and confirm RED**

```powershell
uv run pytest tests/test_repository_archive.py -q -k "wrong_codex_tree or non_tree_codex or malformed_codex or out_of_namespace or validates_all_codex or conditional_ref_delete or namespace_is_not_empty"
```

Expected: FAIL because these guards do not yet exist.

- [ ] **Step 5: Implement the minimum complete preflight and conditional deletion**

Generate PowerShell that:

1. preserves the existing owner-root, clean-status, and exact-HEAD checks;
2. resolves exactly one line from `git rev-parse --verify "$ExpectedSourceCommit^{tree}"` and requires a canonical lowercase 40-hex object ID;
3. reads `git worktree list --porcelain`, checks the native exit code, and requires exactly one line beginning with `worktree `;
4. reads every row from `git for-each-ref --format='%(refname) %(objectname) %(objecttype)' 'refs/codex/turn-diffs/'`, checks the native exit code, parses exactly three fields, and validates every row into memory before any mutation;
5. runs `uvx git-filter-repo --version` and checks its exit code before mutation;
6. conditionally deletes each validated ref with `git update-ref -d $RefName $ExpectedOldObject`, checking each native exit code;
7. re-enumerates the exact namespace and requires it to be empty before invoking `git-filter-repo`.

Do not weaken or remove the current exact source-commit binding.

- [ ] **Step 6: Run focused and complete archive tests**

```powershell
uv run pytest tests/test_repository_archive.py -q
uv run ruff check scripts/archive_repository_curation.py tests/test_repository_archive.py
```

Expected: PASS.

- [ ] **Step 7: Commit the guarded preflight**

```powershell
git diff --check
git status --short
git add -- scripts/archive_repository_curation.py tests/test_repository_archive.py
git diff --cached --check
git commit -m "fix(release): guard owner rewrite tree refs"
```

Verify the commit author, committer, and message before continuing.

---

### Task 2: Enforce all-ref post-rewrite acceptance and mandatory STOP

**Files:**
- Modify: `tests/test_repository_archive.py`
- Modify: `scripts/archive_repository_curation.py`

**Interfaces:**
- Extend the fake native-command harness with deterministic outputs and exit codes for `rev-list`, `fsck`, and `count-objects`.
- Preserve the Stage 1 contract: the successful final action is a STOP message, not restoration.

- [ ] **Step 1: Add failing tests for reachable figure paths and native failures**

Add tests named:

- `test_owner_runbook_rejects_reachable_figure_path_after_rewrite`
- `test_owner_runbook_stops_when_all_ref_scan_fails`
- `test_owner_runbook_stops_when_strict_fsck_fails`
- `test_owner_runbook_stops_when_count_objects_fails`
- `test_owner_runbook_clean_post_rewrite_sequence_reaches_mandatory_stop`

The reachable-path fixture must include ordinary object-only lines and a line of the form `<40-hex> reports/figures/example.png`; only the latter blocks. The clean sequence must prove `rev-list --objects --all`, `fsck --full --strict`, and `count-objects -vH` all occur after rewriting and before STOP.

- [ ] **Step 2: Run the new post-rewrite tests and confirm RED**

```powershell
uv run pytest tests/test_repository_archive.py -q -k "reachable_figure_path or all_ref_scan or strict_fsck or count_objects or clean_post_rewrite_sequence"
```

Expected: FAIL because the current runbook stops immediately after `git-filter-repo`.

- [ ] **Step 3: Implement the minimum post-rewrite acceptance block**

After a successful rewrite, generate PowerShell that:

1. captures `git rev-list --objects --all` and checks the native exit code;
2. rejects any line matching a canonical object ID followed by `reports/figures/` or `reports/figures`;
3. runs `git fsck --full --strict` and checks the native exit code;
4. captures and prints `git count-objects -vH`, checking the native exit code; and
5. prints the mandatory STOP instruction telling the owner not to restore, stage, or commit and to return the full output.

Keep explicit negative assertions in the archive test that the runbook contains no `restore_curated_figures`, `git add`, `git commit`, `git remote`, or `git push` command.

- [ ] **Step 4: Run focused and complete archive tests**

```powershell
uv run pytest tests/test_repository_archive.py -q
uv run ruff check scripts/archive_repository_curation.py tests/test_repository_archive.py
git diff --check
```

Expected: PASS.

- [ ] **Step 5: Commit the post-rewrite acceptance block**

```powershell
git add -- scripts/archive_repository_curation.py tests/test_repository_archive.py
git diff --cached --check
git commit -m "fix(release): verify all refs after owner rewrite"
```

Verify the commit author, committer, and absence of co-author trailers.

---

### Task 3: Rebind canonical documentation to v5 and freeze the tracked source state

**Files:**
- Modify: `docs/superpowers/plans/2026-08-04-repository-curation-history-slimming.md`
- Modify: `docs/worklog.md`

**Interfaces:**
- The long-form curation plan must identify v1-v4 as recovery-only and v5 as the sole owner-gate package.
- The worklog entry must record the approved written specification, the guarded Stage 1 boundary, and that no GPU is involved.

- [ ] **Step 1: Replace every active owner-gate v4 path with the exact v5 path**

Update active commands, archive constraints, restore commands, and integrity checks to use:

```text
D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5
```

Retain v1-v4 references only where they are explicitly labeled immutable recovery-only packages whose runbooks must not be executed.

- [ ] **Step 2: Document the v5 close-Codex and mandatory-STOP boundary**

State that the owner copies the command, fully closes Codex/editors, runs the v5 runbook from an external Windows PowerShell process, reopens Codex only after STOP, and returns the complete output. State that restoration occurs only after a separate read-only checkpoint.

- [ ] **Step 3: Append a concise worklog record without machine-local secrets**

Record the approved written spec, tree-ref risk, complete preflight, conditional deletion, all-ref scan, strict fsck, count report, mandatory STOP, and v5-only owner gate. Do not record a future hash or claim the v5 package exists before Task 4.

- [ ] **Step 4: Run documentation and repository verification**

```powershell
uv run python scripts/verify_repository_links.py
uv run python scripts/verify_readme.py
uv run pytest -q
uv run ruff check .
uv run python scripts/check_forbidden_licences.py
uv lock --check
git diff --check
```

Expected: all commands PASS.

- [ ] **Step 5: Commit the v5 documentation binding**

```powershell
git add -- docs/superpowers/plans/2026-08-04-repository-curation-history-slimming.md docs/worklog.md
git diff --cached --check
git commit -m "docs(release): bind owner gate to v5 safety archive"
```

This is the final tracked commit before the v5 package is created. Verify the tree is clean and record the exact `git rev-parse HEAD` value for the archive receipt.

---

### Task 4: Create immutable v5 and prove it in a disposable all-refs clone

**Files:**
- Create externally: `D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5\`
- Create temporarily: a unique directory under `$env:TEMP\SafeSynth-v5-rehearsal-*`
- Modify tracked files: none

**Interfaces:**
- Consumes the clean final Task 3 HEAD and exact source figure state.
- Produces a non-overwriting v5 package containing the `figures/` payload tree, `figure_manifest.json`, `SafeSynth-pre-filter-repo.bundle`, `archive_receipt.json`, and `OWNER_HISTORY_REWRITE_RUNBOOK.txt`.
- Produces independent evidence that every packaged digest, branch/ref, rewrite condition, curated restore, and contributor invariant passes.

- [ ] **Step 1: Reconfirm the formal repository source state**

```powershell
git status --short
git worktree list --porcelain
git remote -v
uv run python scripts/verify_figure_evidence.py --expected-state source
Test-Path -LiteralPath 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5'
```

Expected: clean tree, exactly one worktree, no remote, source-state PASS, and `False` for the v5 destination. Stop if any condition differs.

- [ ] **Step 2: Create the non-overwriting v5 package through the supported command**

```powershell
uv run python scripts/archive_repository_curation.py --project-root 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth' --destination 'D:\sdg-data\02-safesynth\release_archive\2026-08-04-repository-curation-v5' --owner-project-root 'C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth'
```

Expected: one success line with archive path, exact source commit, `KEEP=14`, `DROP=136`, manifest SHA-256, and bundle SHA-256.

- [ ] **Step 3: Independently verify every v5 commitment**

Load `archive_receipt.json` with the existing strict loader, recompute SHA-256 for the manifest and bundle, verify all 150 archived figure entries and their sizes/digests, require payload size `422839340`, run `git bundle verify`, and prove that the receipt source commit equals the current clean formal HEAD. Run the source-state verifier again and prove archive creation left the formal repository unchanged.

- [ ] **Step 4: Build a normal disposable clone containing every rewrite target**

Clone from the v5 bundle into a unique temporary directory. Materialize every bundled ordinary branch locally, including `main` and `codex/rfdetr-four-arm`. Add one or more refs below `refs/codex/turn-diffs/` whose object IDs equal the clone's exact approved `HEAD^{tree}`. Require a clean worktree and exactly one registered worktree.

- [ ] **Step 5: Run the generated v5 logic only in the disposable clone**

Generate a temporary runbook by calling the same `_runbook(disposable_clone_path, receipt.source_commit)` function, encode it explicitly as UTF-8 without BOM, and run it using external Windows PowerShell 5.1. Never run or edit the packaged owner runbook and never point a rehearsal runbook at the formal repository.

Expected: guarded refs are conditionally deleted, ordinary refs are rewritten, zero historical `reports/figures/` paths remain across all refs, strict fsck passes, the object count is reported, and the mandatory STOP is the final action.

- [ ] **Step 6: Restore and verify the exact KEEP set in the disposable clone**

Run `scripts/restore_curated_figures.py` against v5, stage only `reports/figures`, compare the staged path tuple exactly with the 14 manifest KEEP entries, commit once as `kuotunyu`, and run:

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
```

Expected: all PASS; pack below 120 MiB; exactly one post-rewrite figure commit; only `kuotunyu` author/committer identities; no co-author trailer; exact 14 KEEP paths; no DROP paths.

- [ ] **Step 7: Preserve rehearsal evidence without touching the formal repository**

Move the disposable clone non-destructively into a uniquely named quarantine directory if deletion is not permitted. Reverify that the formal repository is still clean, source-state complete, single-worktree, and without a remote. Do not change or delete v5 after publication.

---

### Task 5: Owner-only rewrite, controller checkpoint, curated restoration, and final acceptance

**Files:**
- Owner mutates: formal Git history only through the immutable v5 runbook
- Restore: exact 14 KEEP files under `reports/figures/`
- Modify after acceptance: `docs/worklog.md`

**Interfaces:**
- Owner input: exact v5 runbook command supplied after Task 4 passes.
- Controller checkpoint: clean rewritten repository with zero reachable figure paths before restoration.
- Final output: one curated figure restoration commit, complete passing suite, pack below 120 MiB, and sole-contributor invariant.

- [ ] **Step 1: Give the owner one exact external PowerShell command**

Tell the owner to copy the command, fully close Codex and all editors, run the immutable v5 `OWNER_HISTORY_REWRITE_RUNBOOK.txt` in Windows PowerShell, wait for its mandatory STOP line, reopen Codex, and paste the complete output. Do not offer the v4 runbook.

- [ ] **Step 2: Perform the read-only zero-figure checkpoint**

Before restoration, verify clean status, exactly one worktree, empty Codex turn-diff namespace, zero `reports/figures/` paths across all refs, strict fsck, the reported object size, no remote, and rewritten authors/committers/trailers limited to the approved identity. Any failure stops the process without restoring files.

- [ ] **Step 3: Restore, stage, and commit exactly 14 KEEP files**

Run the strict v5 restore command, stage only `reports/figures`, compare the staged tuple and digests against the v5 manifest, and commit as:

```powershell
git commit -m "docs: restore curated figure evidence"
```

No other file may enter this commit.

- [ ] **Step 4: Run complete final acceptance**

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
```

Require all PASS, pack below 120 MiB, exact 14 KEEP and zero DROP files, one figure-history commit, only the approved identity, no co-author trailer, no remote, and unchanged v5 hashes/bundle verification.

- [ ] **Step 5: Record completion without broadening publication scope**

Append the final rewritten commit IDs, pack size, KEEP/DROP totals, verification results, and v5 manifest/bundle digests to `docs/worklog.md`; commit only that file as `kuotunyu`. GitHub repository creation, push, model/dataset cards, Hugging Face publication, and fine-tuned latency publication remain separate future work.
