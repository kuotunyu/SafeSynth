# H4 reflected-padding normalization v5 CPU audit

- Status: **complete; no model inference**
- Data scope: Train metadata, Pass-1 QC, and rejected-pilot provenance
- Validation/Test images read: **0 / 0**
- H4 AUC computed: **no**
- Train backgrounds accepted: **1,459/3,500**
- Train backgrounds normalized: **3,479/3,500**
- Eligible anchors remaining: **2,873**
- Known v4 failure cells 7 and 54 normalized or rejected: **yes**
- Known v3 failure cell 64 normalized or rejected: **yes**
- Known v2 failure cells 11 and 25 normalized or rejected: **yes**
- Original known failure cells 10 and 12 normalized or rejected: **yes**

## Rejection counts

| reason | backgrounds |
|---|---:|
| `BACKGROUND_HEADLIKE_NEAR_FRAME_EDGE` | 1,628 |
| `NO_CONTEXT_REPLACEMENT_ANCHOR` | 18 |
| `NO_SAFE_CONTEXT_REPLACEMENT_ANCHOR` | 395 |

## Scientific boundary

This audit only checks deterministic CPU normalization and the post-transform input guards. It does not select a model-call variant, generate a new identity pilot, compute H4, or reopen M13.

The next untouched pilot is registered for root seed `20260731` and has not been generated.
