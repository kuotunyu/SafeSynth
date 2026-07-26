# M10 synthetic preview statistics

- Images: 300
- Annotations: 2434
- Filter pass/reject: 195 / 105
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m11_h4_seed42`
- Hard negatives: **not used**; M9 remains blocked on kuotunyu signoff.

## Scenario counts

| scenario | count |
|---|---:|
| crowded | 44 |
| head_no_helmet | 88 |
| low_light_blur | 32 |
| partial_occlusion | 56 |
| small_distant | 80 |

## First filter rejection reason

| reason | count |
|---|---:|
| BAD_SIZE_RATIO | 1 |
| BOX_TOO_SMALL | 35 |
| EXCESSIVE_OVERLAP | 27 |
| LOW_VISIBLE_FRACTION | 11 |
| NEAR_DUPLICATE_REAL | 7 |
| NEAR_DUPLICATE_SYNTHETIC | 2 |
| NO_CHANGE | 22 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | medium | 27 |
| crowded | head | small | 72 |
| crowded | helmet | large | 4 |
| crowded | helmet | medium | 82 |
| crowded | helmet | small | 171 |
| head_no_helmet | head | large | 1 |
| head_no_helmet | head | medium | 21 |
| head_no_helmet | head | small | 66 |
| low_light_blur | head | medium | 3 |
| low_light_blur | head | small | 4 |
| low_light_blur | helmet | large | 1 |
| low_light_blur | helmet | medium | 12 |
| low_light_blur | helmet | small | 20 |
| low_light_blur | person | large | 1 |
| low_light_blur | person | medium | 5 |
| low_light_blur | person | small | 1 |
| partial_occlusion | head | medium | 4 |
| partial_occlusion | head | small | 24 |
| partial_occlusion | helmet | large | 1 |
| partial_occlusion | helmet | medium | 27 |
| partial_occlusion | helmet | small | 49 |
| partial_occlusion | person | large | 13 |
| partial_occlusion | person | medium | 36 |
| partial_occlusion | person | small | 7 |
| small_distant | head | small | 108 |
| small_distant | helmet | small | 329 |

## `small_distant` contract

- Pasted instances: 437
- Observed min-side range: 8.0–20.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 437 / 437
