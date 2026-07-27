# Whole-person edit v8 GPU diagnostic

- Status: **pending kuotunyu output review**
- Model: `black-forest-labs/FLUX.2-klein-base-4B`
- Revision: `a3b4f4849157f664bdbc776fd7453c2783562f4d`
- Exact approved input manifest:
  `a0c0795e856f588d5aab90887057498259ac96e90fcfef8349669bf9df0f0af2`
- Fixed cases generated: **4/4**
- Outside-edit changed pixels: **0 for every case**
- Expanded to 64: **no**
- H4 AUC computed: **no**

## Preliminary visual assessment

- Cases 1 and 2 remain visually plausible, but the reference produces little
  useful semantic or appearance change.
- Case 3 has a severe head/face/body integration failure.
- Case 4 has a severe nearly uniform red human-shaped output failure.

This assessment is not the binding human gate. The exact four-output sheet is
waiting for `kuotunyu` to accept or reject each case.

## Scientific boundary

This was a fixed Train-only method diagnostic. It did not read Validation/Test,
select a variant after viewing outputs, compute H4, reopen M13, or authorize a
64-image run.
