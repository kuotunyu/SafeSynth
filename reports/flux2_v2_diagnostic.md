# FLUX.2 v2 Colab diagnostic

- Status: **diagnostic complete; no registered variant selected**
- GPU: `NVIDIA A100-SXM4-40GB`
- Execution mode: `full_model_on_cuda`
- Outputs: **12/12**
- Total inference time: **86.70 seconds**
- H4 AUC computed: **no**
- Result archive SHA-256: `33bd82ae1625137b0a42aaf92473e94c95591eb29a1d846bf4833b060003e7c6`

## Aggregate masked changes

| variant | changed pixels | RGB MAE | outside-mask changes |
|---|---:|---:|---:|
| `v1_reference_strength_085` | 1.0000 | 80.0504 | 0 |
| `reference_strength_055` | 1.0000 | 77.6105 | 0 |
| `no_reference_strength_055` | 1.0000 | 77.5989 | 0 |

## Pairwise effects

| comparison | different pixels | RGB MAE |
|---|---:|---:|
| `strength_effect_v1_vs_reference_055` | 0.8904 | 3.0179 |
| `reference_effect_at_strength_055` | 0.4329 | 0.2260 |

- Detail sheet: `C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth\reports\figures\flux2_v2_diagnostic_detail.png`

## Method decision

- Selected variant: **none**. The original v1 method remains rejected by the human identity gate.
- Removing the reference at strength 0.55 is effectively a no-op at inspection scale (masked RGB MAE **0.2260/255**).
- Lowering strength changes some masked pixels but shows no consistent visual improvement across the four fixed cases.
- All three calls preserve every pixel outside the edit mask. They do not fix invalid drafts or mislocalized anchors, the failure modes found in the rejected 64-image pilot.
- Next requirement: preregister an input-validity and anchor-localization guard before any new untouched identity pilot.

- This Train-only diagnostic cannot reopen M13 or replace the 64-image gate.
