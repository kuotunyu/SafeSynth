# M10 synthetic preview statistics

- Images: 300
- Annotations: 2370
- Filter pass/reject: 196 / 104
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
| BOX_TOO_SMALL | 34 |
| EXCESSIVE_OVERLAP | 27 |
| LOW_VISIBLE_FRACTION | 8 |
| NEAR_DUPLICATE_REAL | 9 |
| NEAR_DUPLICATE_SYNTHETIC | 1 |
| NO_CHANGE | 24 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | medium | 18 |
| crowded | head | small | 81 |
| crowded | helmet | large | 1 |
| crowded | helmet | medium | 88 |
| crowded | helmet | small | 147 |
| head_no_helmet | head | medium | 26 |
| head_no_helmet | head | small | 62 |
| low_light_blur | head | medium | 3 |
| low_light_blur | head | small | 9 |
| low_light_blur | helmet | medium | 11 |
| low_light_blur | helmet | small | 17 |
| low_light_blur | person | large | 2 |
| low_light_blur | person | medium | 1 |
| low_light_blur | person | small | 3 |
| partial_occlusion | head | medium | 6 |
| partial_occlusion | head | small | 22 |
| partial_occlusion | helmet | large | 2 |
| partial_occlusion | helmet | medium | 26 |
| partial_occlusion | helmet | small | 52 |
| partial_occlusion | person | large | 15 |
| partial_occlusion | person | medium | 37 |
| partial_occlusion | person | small | 4 |
| small_distant | head | small | 108 |
| small_distant | helmet | small | 273 |

## `small_distant` contract

- Pasted instances: 381
- Observed min-side range: 8.0–20.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 381 / 381
