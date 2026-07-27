# H4 reflected-padding guard v4 CPU audit

- Status: **complete; no model inference**
- Data scope: Train metadata, Pass-1 QC, and rejected-pilot provenance
- Validation/Test images read: **0 / 0**
- H4 AUC computed: **no**
- Train backgrounds accepted: **99/3,500**
- Eligible anchors remaining: **203**
- Failed v3 input-sheet cells rejected by the guard: **10/64**
- Known v3 failure cell 64 rejected: **yes**
- Known v2 failure cells 11 and 25 rejected: **yes**
- Original known failure cells 10 and 12 rejected: **yes**

## Rejection counts

| reason | backgrounds |
|---|---:|
| `BACKGROUND_HEADLIKE_NEAR_FRAME_EDGE` | 81 |
| `BACKGROUND_REFLECTED_PADDING` | 3,274 |
| `NO_SAFE_CONTEXT_REPLACEMENT_ANCHOR` | 46 |

## Scientific boundary

This audit only checks whether the fixed guard removes unsafe inputs while leaving a usable Train pool. It does not select a model-call variant, generate a new identity pilot, compute H4, or reopen M13.

The next untouched pilot is registered for root seed `20260730` and has not been generated.
