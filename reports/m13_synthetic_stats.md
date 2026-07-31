# M10 synthetic preview statistics

- Images: 9000
- Annotations: 69825
- Filter pass/reject: 3749 / 5251
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m13_pool_1x`
- Hard negatives: procedural bank wired; distractors carry no annotation (ADR-004).
- Generative boundary inpainting: **not used**.

## Scenario counts

| scenario | count |
|---|---:|
| crowded | 1127 |
| hard_negative | 1128 |
| head_no_helmet | 2212 |
| low_light_blur | 860 |
| partial_occlusion | 1374 |
| small_distant | 2299 |

## First filter rejection reason

| reason | count |
|---|---:|
| BAD_SIZE_RATIO | 40 |
| BOX_TOO_SMALL | 1092 |
| EXCESSIVE_OVERLAP | 669 |
| LOW_VISIBLE_FRACTION | 225 |
| NEAR_DUPLICATE_REAL | 162 |
| NEAR_DUPLICATE_SYNTHETIC | 2406 |
| NO_CHANGE | 646 |
| SEAM_ARTIFACT | 11 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | large | 2 |
| crowded | head | medium | 539 |
| crowded | head | small | 2164 |
| crowded | helmet | large | 38 |
| crowded | helmet | medium | 1773 |
| crowded | helmet | small | 4253 |
| head_no_helmet | head | large | 6 |
| head_no_helmet | head | medium | 565 |
| head_no_helmet | head | small | 1642 |
| low_light_blur | head | medium | 80 |
| low_light_blur | head | small | 211 |
| low_light_blur | helmet | large | 6 |
| low_light_blur | helmet | medium | 287 |
| low_light_blur | helmet | small | 511 |
| low_light_blur | person | large | 55 |
| low_light_blur | person | medium | 47 |
| low_light_blur | person | small | 7 |
| partial_occlusion | head | large | 2 |
| partial_occlusion | head | medium | 169 |
| partial_occlusion | head | small | 496 |
| partial_occlusion | helmet | large | 17 |
| partial_occlusion | helmet | medium | 681 |
| partial_occlusion | helmet | small | 1325 |
| partial_occlusion | person | large | 442 |
| partial_occlusion | person | medium | 782 |
| partial_occlusion | person | small | 150 |
| small_distant | head | small | 2943 |
| small_distant | helmet | small | 8940 |

## `small_distant` contract

- Pasted instances: 11883
- Observed min-side range: 7.0–21.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 11872 / 11883
