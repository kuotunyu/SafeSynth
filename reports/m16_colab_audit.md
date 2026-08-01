# M16 — Colab results audit

**Verdict: PASS** — 0 fatal, 4 warning.

- Arms returned: real_only, standard_aug, unfiltered_syn, filtered_syn
- Shared real-train digest: `b46e8263a9000a2ac019adf5034c100e54e41ef912da2c2edb6f1a2e520fc9b3`
- Optimizer steps per arm: 10900
- Synthetic images per arm: filtered_syn 3500, real_only 0, standard_aug 0, unfiltered_syn 3500

## Findings

| Severity | Check | Detail |
|---|---|---|
| **warning** | `training_log` | real_only has no trainer_state.json, so its training curve cannot be re-aggregated from the raw log (EVAL-12) and would have to be copied off the notebook display, which is not allowed. |
| **warning** | `training_log` | standard_aug has no trainer_state.json, so its training curve cannot be re-aggregated from the raw log (EVAL-12) and would have to be copied off the notebook display, which is not allowed. |
| **warning** | `training_log` | unfiltered_syn has no trainer_state.json, so its training curve cannot be re-aggregated from the raw log (EVAL-12) and would have to be copied off the notebook display, which is not allowed. |
| **warning** | `training_log` | filtered_syn has no trainer_state.json, so its training curve cannot be re-aggregated from the raw log (EVAL-12) and would have to be copied off the notebook display, which is not allowed. |
