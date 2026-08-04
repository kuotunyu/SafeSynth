# Repository curation evidence-manifest design

## 1. Context and blocker

The approved repository-curation design removes every historical path under
`reports/figures/` and restores only the Markdown-linked KEEP set.  A disposable
Task 8 rehearsal proved that the 14 KEEP files restore exactly, but the full
suite then reported 21 failures in `tests/test_supervised_labeler.py`.

Those failures are one contract mismatch:

- 78 unique historical review PNGs are required by direct `is_file()` or
  `read_bytes()` assertions;
- every one of those 78 PNGs is an approved DROP entry;
- their committed JSON/YAML metadata already pins the same SHA-256 recorded by
  the independently verified v3 archive; and
- their combined size is 260.82 MiB, so retaining them is incompatible with
  the approved Git-pack target below 120 MiB.

A wider scan found 104 of the 150 figures referenced by structured metadata,
100 of them DROP.  Current inference and generation code does not consume the
old PNG bytes.  Historical review-replay scripts may consume them and therefore
must restore the required files from an external recovery archive first.

## 2. Selected evidence model

The repository distinguishes three kinds of references:

1. **Markdown display evidence.** A surviving Markdown destination keeps the
   PNG in Git.
2. **Structured scientific commitment.** A JSON/YAML `path + sha256` pair must
   agree with a tracked canonical manifest, but the PNG may live only in the
   recovery archive.
3. **Runtime input.** A current runtime consumer must have local bytes or an
   explicit restore/regeneration contract. No current production consumer was
   found for the DROP review PNGs.

The selected solution tracks
`reports/figure_curation_manifest.json`, byte-for-byte copied from the already
verified v3 `figure_manifest.json`.  It contains all 150 normalized paths,
sizes, SHA-256 digests, KEEP/DROP dispositions, reference sources, and the
pre-curation source commit.  The approved bytes are:

- source commit: `c2c6059987e71142d7c5524a52ecc2b0c4afcee5`;
- manifest SHA-256:
  `aa39003c3189278eda178a39514bfa7f640655f8ddf5f1e3c2bad99380751fd5`;
- entries: 150 total, 14 KEEP, 136 DROP.

The manifest's `source_commit` identifies the immutable pre-curation evidence
snapshot. It is not required to equal a later v4 archive commit or the
post-rewrite HEAD. The entry tuple must, however, match the v4 archive exactly.

## 3. Legal repository states

Verification accepts exactly two complete states:

- **source**: all 150 manifest paths are tracked and present, and every byte
  matches its size and SHA-256 commitment;
- **curated**: exactly the 14 KEEP paths are tracked and present, all 136 DROP
  paths are absent, and every KEEP byte matches.

`auto` may identify either exact state for local development before and after
the rewrite. It must reject every partial, mixed, corrupt, aliased, untracked,
or extra-file state. Public CI and final Task 8 acceptance explicitly request
`curated`; they do not use `auto`.

## 4. Verification interfaces

`src/release/repository_archive.py` exposes a strict manifest-only parser that
checks schema, canonical JSON, path safety, ordering, uniqueness, and field
formats without requiring an adjacent archive payload.

`src/release/figure_evidence.py` provides:

- loading of the tracked canonical manifest;
- repository-state verification for `source`, `curated`, or `auto`;
- one-path verification that requires metadata SHA equality, hashes local
  bytes when present, requires every KEEP byte, and permits a missing byte only
  for an indexed DROP entry; and
- exact comparison between the current KEEP/DROP plan and manifest entries.

`scripts/verify_figure_evidence.py` is the public fail-closed gate. It pins the
approved manifest hash and counts and accepts an explicit expected state.

`scripts/freeze_figure_evidence_manifest.py` is a one-time, no-overwrite
materializer. It verifies every byte in an external archive and a caller-given
manifest digest before publishing the canonical manifest into the repository.
It never rewrites history or deletes archive data.

## 5. Scientific-test migration

The 21 affected tests retain all scientific assertions. Their page checks call
the common one-path verifier with the exact SHA from the existing JSON/YAML
record:

- in source state, the original PNG is re-hashed;
- in curated state, a missing DROP PNG is accepted only when its manifest
  entry has the identical expected SHA;
- a missing KEEP, a SHA conflict, an unknown path, or corrupt present byte
  remains a failure.

The legacy cleanup test becomes an exact repository-state test. This is
stronger than checking six hand-picked filenames and several glob patterns.

## 6. Archive and history-rewrite sequencing

The v3 package remains immutable recovery evidence and is never overwritten or
deleted. Because implementation changes HEAD, v3 becomes forbidden for the
owner gate. A new, non-overwriting v4 recovery package is created after this
work is merged. Its manifest entries must equal the tracked canonical manifest
entry-for-entry; its own `source_commit` binds the new owner runbook to the new
HEAD.

Only the owner executes the v4 Stage 1 runbook. After the rewrite, Task 8
restores 14 KEEP files, runs the verifier with `--expected-state curated`, runs
the full repository gates, and verifies the Git pack below 120 MiB and sole
contributor identity `kuotunyu`.

## 7. Blocking conditions

Completion is blocked by any of the following:

- noncanonical, duplicated, unsorted, path-traversing, or SHA-invalid manifest;
- manifest SHA other than the approved v3 digest;
- counts other than 150 total / 14 KEEP / 136 DROP;
- JSON/YAML scientific SHA conflicting with the manifest;
- missing or corrupt KEEP byte;
- tracked or present DROP byte in curated state;
- partial/mixed inventory, filesystem/Git disagreement, path alias, or extra
  figure file;
- archive entries differing from the tracked manifest;
- any public test depending on `D:\` or another machine-local archive path;
- full pytest, Ruff, README, repository-link, lock, diff, pack-size, or
  contributor gate failure.

## 8. Alternatives rejected

- **Retain all structured-metadata figures:** adds at least 260.82 MiB for the
  currently failing subset and defeats the pack target.
- **Delete the file assertions:** makes tests green by discarding provenance
  verification.
- **Depend on the external archive in tests:** makes public CI machine-local and
  unreproducible.
- **Run the v3 rewrite first and repair afterward:** creates a known-red
  intermediate repository and prevents unattended preflight work. Building and
  independently reviewing v4 first is safer.
