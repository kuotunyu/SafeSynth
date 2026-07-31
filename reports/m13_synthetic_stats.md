# M10 synthetic preview statistics

- Images: 9000
- Annotations: 75993
- Filter pass/reject: 3449 / 5551
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m13_pool_1x`
- Hard negatives: procedural bank wired; distractors carry no annotation (ADR-004).
- Generative boundary inpainting: **not used**.

## Scenario counts

| scenario | count |
|---|---:|
| crowded | 1530 |
| hard_negative | 667 |
| head_no_helmet | 1912 |
| low_light_blur | 658 |
| partial_occlusion | 988 |
| small_distant | 3245 |

## First filter rejection reason

| reason | count |
|---|---:|
| BAD_SIZE_RATIO | 30 |
| BOX_TOO_SMALL | 1393 |
| EXCESSIVE_OVERLAP | 802 |
| LOW_VISIBLE_FRACTION | 255 |
| NEAR_DUPLICATE_REAL | 171 |
| NEAR_DUPLICATE_SYNTHETIC | 2207 |
| NO_CHANGE | 680 |
| SEAM_ARTIFACT | 13 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | large | 3 |
| crowded | head | medium | 716 |
| crowded | head | small | 2916 |
| crowded | helmet | large | 49 |
| crowded | helmet | medium | 2468 |
| crowded | helmet | small | 5686 |
| head_no_helmet | head | large | 6 |
| head_no_helmet | head | medium | 505 |
| head_no_helmet | head | small | 1401 |
| low_light_blur | head | medium | 65 |
| low_light_blur | head | small | 160 |
| low_light_blur | helmet | large | 6 |
| low_light_blur | helmet | medium | 221 |
| low_light_blur | helmet | small | 384 |
| low_light_blur | person | large | 45 |
| low_light_blur | person | medium | 35 |
| low_light_blur | person | small | 4 |
| partial_occlusion | head | large | 3 |
| partial_occlusion | head | medium | 122 |
| partial_occlusion | head | small | 362 |
| partial_occlusion | helmet | large | 11 |
| partial_occlusion | helmet | medium | 484 |
| partial_occlusion | helmet | small | 932 |
| partial_occlusion | person | large | 302 |
| partial_occlusion | person | medium | 577 |
| partial_occlusion | person | small | 109 |
| small_distant | head | small | 4120 |
| small_distant | helmet | small | 12376 |

## `small_distant` contract

- Pasted instances: 16496
- Observed min-side range: 7.0–21.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 16480 / 16496
