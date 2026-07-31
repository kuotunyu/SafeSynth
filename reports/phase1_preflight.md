# Phase 1 handoff preflight

- Audited HEAD: `d83d4cf7475016d8848acb0963fa28261f119782`
- Commits across all local refs: **231**
- GitHub contributor identity ready: **PASS**
- Pre-publication state (identity + no remote): **PASS**
- Unrestricted scale-up (H4 passed): **FAIL**
- Permitted synthetic scale: **1x**

## Integrity checks

- PASS — `all_authors_and_committers_are_kuotunyu`
- PASS — `no_coauthored_by_trailers`
- PASS — `repo_local_identity_is_kuotunyu_noreply`
- PASS — `single_contributor_commit_hook_is_active`
- PASS — `no_git_remote_before_user_request`
- PASS — `h6_grid_exists_and_sha256_matches`
- PASS — `h4_result_is_internally_consistent`
- PASS — `m12_filter_ledger_is_internally_consistent`

## Contributor evidence

- Authors: `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`
- Committers: `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`
- `Co-Authored-By:` commits: 0
- Configured remotes: 0

## Phase 1 gates

- H6 exact-grid approval: **PASS** (SHA256 `0e385d857067aa293c5e3d0dd43ad84b4141ff9bac5c8d4aefed187ee9c45739`)
- H4 scale-up gate: **FAIL** (AUC 0.7964; maximum 0.60)
- M12 ledger: 300 = 196 pass + 104 reject

## Blocking actions

- (none)

## Known limitations carried forward

- M11/H4 AUC 0.7964 exceeds the 0.60 maximum: paste artifacts are detectable. Accepted as a reported limitation per ADR-011; generation is capped at 1x and 2x is forbidden. Every result table must display this AUC.

A failed H6 line is a hard blocker: do not create a signoff on the user's
behalf. A failed H4 line is NOT a blocker any more — per ADR-011 it is an
accepted, published limitation that caps generation at 1x and forbids 2x.
It is still a failure and must never be reported as a pass.
