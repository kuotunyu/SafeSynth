# M10 synthetic preview statistics

- Images: 32
- Annotations: 278
- Filter pass/reject: 2 / 30
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m10_seed42`
- Hard negatives: **not used**; M9 remains blocked on kuotunyu signoff.

## Scenario counts

| scenario | count |
|---|---:|
| crowded | 6 |
| head_no_helmet | 11 |
| low_light_blur | 1 |
| partial_occlusion | 6 |
| small_distant | 8 |

## First filter rejection reason

| reason | count |
|---|---:|
| BOX_TOO_SMALL | 4 |
| EXCESSIVE_OVERLAP | 5 |
| LOW_VISIBLE_FRACTION | 2 |
| NEAR_DUPLICATE_REAL | 18 |
| NO_CHANGE | 1 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | medium | 1 |
| crowded | head | small | 13 |
| crowded | helmet | medium | 9 |
| crowded | helmet | small | 28 |
| head_no_helmet | head | medium | 6 |
| head_no_helmet | head | small | 5 |
| low_light_blur | head | medium | 1 |
| low_light_blur | helmet | medium | 1 |
| low_light_blur | helmet | small | 1 |
| partial_occlusion | head | small | 2 |
| partial_occlusion | helmet | medium | 3 |
| partial_occlusion | helmet | small | 7 |
| partial_occlusion | person | large | 2 |
| partial_occlusion | person | medium | 4 |
| small_distant | head | small | 12 |
| small_distant | helmet | small | 32 |

## `small_distant` contract

- Pasted instances: 44
- Observed min-side range: 8.0–20.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 44 / 44
