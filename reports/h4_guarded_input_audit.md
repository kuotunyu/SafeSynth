# H4 guarded-input CPU audit

- Status: **complete; no model inference**
- Data scope: Train metadata, Pass-1 QC, and rejected-pilot provenance
- Validation/Test images read: **0 / 0**
- H4 AUC computed: **no**
- Train backgrounds accepted: **1,375/3,500**
- Eligible anchors remaining: **3,583**
- Previous failed-pilot cells rejected by the guard: **37/64**
- Known failure cells 10 and 12 rejected: **yes**

## Rejection counts

| reason | backgrounds |
|---|---:|
| `BACKGROUND_HEADLIKE_NEAR_FRAME_EDGE` | 1,813 |
| `NO_SAFE_CONTEXT_REPLACEMENT_ANCHOR` | 312 |

## Scientific boundary

This audit only checks whether the fixed guard removes unsafe inputs while leaving a usable Train pool. It does not select a model-call variant, generate a new identity pilot, compute H4, or reopen M13.

The next untouched pilot is registered for root seed `20260728` and has not been generated.
