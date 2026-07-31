# H4 guarded-input v2 preflight — failed

- Status: **failed pre-model reflection review**
- Inputs: **64**
- Root seed: `20260728`
- Model inference run: **no**
- H4 AUC computed: **no**
- Allowed input issues: **0**
- Geometry fingerprint: `beda026df3975f0eee43f8dd1531719824b64bef36a5fd68b53f0375d35a4129`
- Observed input issues: **at least 2**
- Contact sheet: `reports/figures/h4_guarded_input_preflight_v2_failed.png`

Each numbered cell shows the full DRAFT on the left and an enlarged anchor crop on the right. The cyan rectangle marks the exact object support that will receive boundary inpainting.

The pre-model development sanity check found obvious reflected padding in at
least cells 11 (`hard_hat_workers1818.png`) and 25
(`hard_hat_workers4287.png`). The annotation-only v2 guard therefore failed
before asking `kuotunyu` to review it. No FLUX.2 output or H4 score was
computed. A pixel-level high-confidence reflection guard is required before a
new input sheet is frozen.
