# M10 synthetic preview statistics

- Images: 6000
- Annotations: 46668
- Filter pass/reject: 2842 / 3158
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m13_pool_1x`
- Hard negatives: procedural bank wired; distractors carry no annotation (ADR-004).
- Generative boundary inpainting: **not used**.

## Scenario counts

| scenario | count |
|---|---:|
| crowded | 739 |
| hard_negative | 749 |
| head_no_helmet | 1497 |
| low_light_blur | 579 |
| partial_occlusion | 921 |
| small_distant | 1515 |

## First filter rejection reason

| reason | count |
|---|---:|
| BAD_SIZE_RATIO | 29 |
| BOX_TOO_SMALL | 729 |
| EXCESSIVE_OVERLAP | 446 |
| LOW_VISIBLE_FRACTION | 149 |
| NEAR_DUPLICATE_REAL | 131 |
| NEAR_DUPLICATE_SYNTHETIC | 1216 |
| NO_CHANGE | 451 |
| SEAM_ARTIFACT | 7 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | large | 2 |
| crowded | head | medium | 348 |
| crowded | head | small | 1388 |
| crowded | helmet | large | 31 |
| crowded | helmet | medium | 1247 |
| crowded | helmet | small | 2642 |
| head_no_helmet | head | large | 2 |
| head_no_helmet | head | medium | 385 |
| head_no_helmet | head | small | 1112 |
| low_light_blur | head | large | 1 |
| low_light_blur | head | medium | 52 |
| low_light_blur | head | small | 173 |
| low_light_blur | helmet | large | 3 |
| low_light_blur | helmet | medium | 185 |
| low_light_blur | helmet | small | 336 |
| low_light_blur | person | large | 34 |
| low_light_blur | person | medium | 43 |
| low_light_blur | person | small | 5 |
| partial_occlusion | head | medium | 104 |
| partial_occlusion | head | small | 341 |
| partial_occlusion | helmet | large | 12 |
| partial_occlusion | helmet | medium | 438 |
| partial_occlusion | helmet | small | 858 |
| partial_occlusion | person | large | 304 |
| partial_occlusion | person | medium | 510 |
| partial_occlusion | person | small | 107 |
| small_distant | head | small | 1907 |
| small_distant | helmet | small | 5961 |

## `small_distant` contract

- Pasted instances: 7868
- Observed min-side range: 7.0–21.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 7866 / 7868
