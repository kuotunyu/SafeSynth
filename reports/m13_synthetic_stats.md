# M10 synthetic preview statistics

- Images: 10000
- Annotations: 84063
- Filter pass/reject: 3664 / 6336
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m13_pool_1x`
- Hard negatives: procedural bank wired; distractors carry no annotation (ADR-004).
- Generative boundary inpainting: **not used**.

## Scenario counts

| scenario | count |
|---|---:|
| crowded | 1727 |
| hard_negative | 756 |
| head_no_helmet | 2124 |
| low_light_blur | 744 |
| partial_occlusion | 1109 |
| small_distant | 3540 |

## First filter rejection reason

| reason | count |
|---|---:|
| BAD_SIZE_RATIO | 25 |
| BOX_TOO_SMALL | 1546 |
| EXCESSIVE_OVERLAP | 908 |
| LOW_VISIBLE_FRACTION | 286 |
| NEAR_DUPLICATE_REAL | 188 |
| NEAR_DUPLICATE_SYNTHETIC | 2581 |
| NO_CHANGE | 792 |
| SEAM_ARTIFACT | 10 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | large | 3 |
| crowded | head | medium | 801 |
| crowded | head | small | 3217 |
| crowded | helmet | large | 61 |
| crowded | helmet | medium | 2801 |
| crowded | helmet | small | 6469 |
| head_no_helmet | head | large | 2 |
| head_no_helmet | head | medium | 567 |
| head_no_helmet | head | small | 1558 |
| low_light_blur | head | medium | 70 |
| low_light_blur | head | small | 195 |
| low_light_blur | helmet | large | 8 |
| low_light_blur | helmet | medium | 235 |
| low_light_blur | helmet | small | 408 |
| low_light_blur | person | large | 32 |
| low_light_blur | person | medium | 51 |
| low_light_blur | person | small | 13 |
| partial_occlusion | head | medium | 142 |
| partial_occlusion | head | small | 430 |
| partial_occlusion | helmet | large | 9 |
| partial_occlusion | helmet | medium | 524 |
| partial_occlusion | helmet | small | 1036 |
| partial_occlusion | person | large | 357 |
| partial_occlusion | person | medium | 629 |
| partial_occlusion | person | small | 123 |
| small_distant | head | small | 4408 |
| small_distant | helmet | small | 13404 |

## `small_distant` contract

- Pasted instances: 17812
- Observed min-side range: 7.0–21.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 17798 / 17812
