# Whole-person edit v8 GPU diagnostic

- Status: **rejected by kuotunyu; no 64-case scale-up**
- Model: `black-forest-labs/FLUX.2-klein-base-4B`
- Revision: `a3b4f4849157f664bdbc776fd7453c2783562f4d`
- Exact approved input manifest:
  `a0c0795e856f588d5aab90887057498259ac96e90fcfef8349669bf9df0f0af2`
- Fixed cases generated: **4/4**
- Outside-edit changed pixels: **0 for every case**
- Expanded to 64: **no**
- H4 AUC computed: **no**

## Binding human review

- Case 1: **PASS** — 做得很好。
- Case 2: **PASS** — 做得很好。
- Case 3: **FAIL** — 一看就是後製。
- Case 4: **SEVERE FAIL** — 嚴重失敗。
- Total: **2 pass / 2 fail**, including **1 severe failure**.
- Decision: **do not expand v8 to 64 cases**.

## Scientific boundary

This was a fixed Train-only method diagnostic. It did not read Validation/Test,
select a variant after viewing outputs, compute H4, reopen M13, or authorize a
64-image run.
