# M10 synthetic preview statistics

- Images: 32
- Annotations: 260
- Filter pass/reject: 20 / 12
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m10_repro_after_h4`
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
| BOX_TOO_SMALL | 3 |
| EXCESSIVE_OVERLAP | 3 |
| LOW_VISIBLE_FRACTION | 1 |
| NEAR_DUPLICATE_REAL | 1 |
| NO_CHANGE | 4 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | medium | 1 |
| crowded | head | small | 10 |
| crowded | helmet | large | 1 |
| crowded | helmet | medium | 6 |
| crowded | helmet | small | 25 |
| head_no_helmet | head | medium | 5 |
| head_no_helmet | head | small | 6 |
| partial_occlusion | head | medium | 1 |
| partial_occlusion | head | small | 4 |
| partial_occlusion | helmet | medium | 2 |
| partial_occlusion | helmet | small | 6 |
| partial_occlusion | person | large | 1 |
| partial_occlusion | person | medium | 4 |
| partial_occlusion | person | small | 1 |
| small_distant | head | small | 10 |
| small_distant | helmet | small | 26 |

## `small_distant` contract

- Pasted instances: 36
- Observed min-side range: 8.0–19.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 36 / 36
