# H4 guarded-input preflight

- Status: **rejected by kuotunyu**
- Inputs: **64**
- Root seed: `20260731`
- Model inference run: **no**
- H4 AUC computed: **no**
- Allowed input issues: **0**
- Observed input issues: **33**
- Geometry fingerprint: `0d21edbb2e53f632367ad2b548cf972a89ea85623ea498d7daaede92c2bb6b44`
- Contact sheet: `reports/figures/h4_guarded_input_preflight.png`

Each numbered cell shows the full DRAFT on the left and an enlarged anchor crop on the right. The cyan rectangle marks the exact object support that will receive boundary inpainting.

## Human review result

- Obvious composited face: cells 05, 08, 25, 33
- Floating/blurred helmet without a credible visible head: cells 09, 10, 15,
  16, 20, 21, 24, 26, 28, 34, 37, 45, 50
- Helmet and head both present but the overall result is implausible: cells 29,
  30, 32, 36, 38, 41, 43, 44, 47, 57, 63
- Appearance resembles a failed post-process: cells 51, 55, 59, 62, 64

The 33/64 failure count rejects the input gate. GPU inference, the output
identity gate, H4, M13, and Phase 2 remain locked.
