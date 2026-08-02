# RF-DETR four-arm completion design

Date: 2026-08-02
Owner: kuotunyu

## Goal

Replicate the completed RT-DETRv2 four-arm comparison with RF-DETR-Nano on the
local RTX 4090. The experiment must answer whether the synthetic-data result
survives a detector-architecture change without changing the frozen split,
training budget, arm composition, evaluation definitions, or reporting rules.

This is the first remaining subproject in the full-release path. Repository
history slimming, GitHub publication, model and dataset cards, and Hugging Face
upload follow after the RF-DETR result is frozen; they do not run concurrently
with this experiment.

## Frozen experiment

- Execution machine: local Windows workstation, NVIDIA GeForce RTX 4090.
- Model: `Roboflow/rf-detr-nano`, pinned and Apache-2.0 as already recorded by
  the latency harness.
- Arms, in execution order:
  1. `real_only`
  2. `filtered_syn`
  3. `standard_aug`
  4. `unfiltered_syn`
- Seed: 1337 for every arm.
- Budget: 10,900 optimizer steps per arm.
- Split: the existing frozen group split. Training may read Train and
  Validation; it must not read Test.
- Synthetic subsets: the existing equal-size M13 1x exports. Filtered and
  unfiltered arms each receive 3,500 synthetic images.
- Validation remains real-only for every arm.
- The four-arm scope supersedes the temporary two-arm reduction currently
  recorded in `configs/training_rfdetr.yaml`.

The ordering completes the decisive `real_only` versus `filtered_syn` contrast
first. If an external interruption occurs after two arms, the most important
cross-architecture comparison is already recoverable; the remaining two arms
still have to finish before M20 is marked complete.

## Architecture and interfaces

### Training CLI

Add `scripts/train_arms.py` as a thin command-line adapter around the existing
`src.training.run.run_arm()` implementation. It owns orchestration only; model
loading, dataset construction, augmentation, metrics, optimizer grouping, and
checkpoint resume stay in their existing modules.

The CLI will:

1. Resolve an inheritance-aware training config with
   `src.training.config.load_training_config()`.
2. Validate the requested arms against the config's frozen arm list.
3. Build arm compositions from the existing split manifest and M13 annotations.
4. Write RF-DETR runs under a separate root with the layout
   `<runs-root>/<arm>/seed_1337/`, so the existing evaluation driver can consume
   the result without overwriting RT-DETRv2 runs.
5. Run one arm at a time with automatic checkpoint resume.
6. Refuse to overwrite a completed run unless an explicit, separately tested
   override is supplied. The normal path is skip-complete or resume-incomplete.
7. Emit one machine-readable orchestration summary containing config path,
   checkpoint identity, arm order, seed, data digests, start/end times, status,
   and the run-record path of each arm.

The CLI will expose a dry-run/preflight mode. Preflight resolves every input and
output path, validates arm composition and disk availability, and performs no
model download or GPU allocation.

### Configuration

Keep `configs/training_rfdetr.yaml` as a differences-only child of
`configs/training.yaml`. Change only its experiment scope from two arms to four;
do not duplicate inherited training arguments.

RF-DETR-specific preprocessing remains frozen:

- 384 x 384 input
- ImageNet normalization enabled
- padding enabled
- three-class replacement head

No hyperparameter search is allowed. The optimizer values are already labelled
as guesses; changing them after observing RF-DETR results would turn this into a
different experiment.

### Speed probe

Before production training, run the existing two-run slope probe against only
`configs/training_rfdetr.yaml`, using 40 and 140 steps. The probe is valid only
when:

- the long run is slower than the short run;
- derived seconds per step is positive;
- fixed overhead is non-negative;
- both runs finish without NaN/Inf loss or CUDA OOM;
- no unrelated compute process materially occupies the GPU.

The measured projection replaces all earlier planning ranges. Report the
projection to the user in commentary, then continue under the user's existing
authorization unless the probe is invalid or exposes a safety problem.

### Evaluation and reporting

After all four arms finish:

1. Discover each arm's best Validation checkpoint using the existing resolver.
2. Persist Validation and Test predictions in a model-specific output location.
3. Run the existing frozen Test evaluation with per-image original-coordinate
   mapping and 1,000 image-level bootstrap resamples.
4. Write RF-DETR metrics and reports to separate files; do not overwrite the
   RT-DETRv2 source-of-truth CSV.
5. Compare the same four quantities and caveats used in the main experiment:
   primary AP_small, primary mAP, bare-head recall at the frozen/selected
   operating point, and real-image exposure.
6. Re-measure latency using fine-tuned RF-DETR weights. Any publishable latency
   comparison must satisfy the existing contention and SM-clock checks.
7. Update README and result figures only after the model-specific reports pass
   the numerical verifier.

The RF-DETR table is a replication result, not a replacement for the primary
RT-DETRv2 table.

## Failure handling and recovery

- Production arms run sequentially; never allocate four models concurrently.
- Every arm has an isolated output directory and resumable checkpoints.
- A failed arm stops orchestration. Later arms do not start automatically after
  an OOM, NaN loss, corrupt checkpoint, missing input, or validation failure.
- Existing output directories are never recursively deleted by the production
  CLI. Cleanup is a separate, explicit user-authorized operation.
- A completed arm is identified by its run record, expected step count, matching
  config/data provenance, and readable checkpoint; directory existence alone is
  not completion evidence.
- GPU availability is checked immediately before the probe and each production
  arm. Other projects' processes are neither stopped nor modified.
- If Windows, the display driver, or another project interrupts a run, resume
  from the newest valid checkpoint rather than restarting.
- If the speed probe produces a negative intercept or inconsistent rate, repeat
  the probe after checking GPU contention. Do not extrapolate production time
  from an invalid slope.

## Testing strategy

Implementation follows red-green-refactor.

Unit tests will cover:

- default four-arm order and explicit arm subsets;
- inheritance-aware config loading;
- dedicated RF-DETR output-root construction;
- dry-run behavior with zero model/GPU calls;
- rejection of unknown arms and unsafe output collisions;
- skip-complete, resume-incomplete, and stop-on-first-failure behavior;
- orchestration-summary provenance;
- no Test access during training orchestration.

Tests must fail for the missing CLI before production code is added. GPU and
network calls are excluded from unit tests through dependency injection at the
orchestration seam, not by reimplementing training behavior in mocks.

Verification proceeds in this order:

1. Focused CLI tests.
2. Existing training/config/speed-probe tests.
3. Ruff.
4. Full pytest suite.
5. RF-DETR short GPU smoke run with 16 Validation images.
6. Two-run speed probe.
7. Four production arms.
8. Prediction/evaluation verification and README numeric audit.

## Human review boundary

No human image review is needed to launch or validate training. Request owner
review only if the post-training error-analysis grids contain semantically
ambiguous cases that affect a stated conclusion. Any such request must provide:

- numbered cells;
- a plain-language explanation of every column and box colour;
- one precise rule to judge;
- a small enough page set to inspect reliably.

Human feedback may classify reporting examples. It may not change the frozen
Test metric, training data, thresholds, arm selection, or hyperparameters after
results are observed.

## Completion criteria

M20 is complete only when all of the following are true:

- the speed probe has a valid measured slope and archived JSON/Markdown report;
- all four arms have a successful run record at 10,900 steps and the same real
  Train digest;
- synthetic counts are 0, 3,500, 0, and 3,500 for the specified arm order;
- every arm has a resolvable best Validation checkpoint;
- frozen Test predictions and metrics exist for all four arms;
- 1,000-resample confidence intervals are present for the required metrics;
- fine-tuned RF-DETR latency has passed contention and clock-spread checks;
- README claims re-aggregate from tracked result sources;
- ruff, the full test suite, README verification, and licence scan pass;
- the Git worktree is clean and commit author/committer remain `kuotunyu` with
  no `Co-Authored-By:` trailer.

## Explicit non-goals

- No Colab execution unless the local RTX 4090 becomes unavailable.
- No additional seeds in M20.
- No hyperparameter search or tuning on Test.
- No changes to the RT-DETRv2 primary result.
- No repository history rewrite, GitHub push, or Hugging Face upload during GPU
  training. Those release steps begin only after M20 artifacts are frozen.
