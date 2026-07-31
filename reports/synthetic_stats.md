# M10 synthetic preview statistics

- Images: 14000
- Annotations: 118990
- Filter pass/reject: 4177 / 9823
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m13_pool_1x`
- Hard negatives: procedural bank wired; distractors carry no annotation (ADR-004).
- Generative boundary inpainting: **not used**.

## Scenario counts

| scenario | count |
|---|---:|
| crowded | 2483 |
| hard_negative | 1141 |
| head_no_helmet | 2644 |
| low_light_blur | 1193 |
| partial_occlusion | 1599 |
| small_distant | 4940 |

## First filter rejection reason

| reason | count |
|---|---:|
| BAD_SIZE_RATIO | 35 |
| BOX_TOO_SMALL | 2161 |
| EXCESSIVE_OVERLAP | 1276 |
| ILLEGIBLE_ANNOTATION | 69 |
| LOW_VISIBLE_FRACTION | 440 |
| NEAR_DUPLICATE_REAL | 261 |
| NEAR_DUPLICATE_SYNTHETIC | 4401 |
| NO_CHANGE | 1164 |
| SEAM_ARTIFACT | 16 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| crowded | head | large | 6 |
| crowded | head | medium | 1103 |
| crowded | head | small | 4701 |
| crowded | helmet | large | 101 |
| crowded | helmet | medium | 4100 |
| crowded | helmet | small | 9212 |
| head_no_helmet | head | large | 155 |
| head_no_helmet | head | medium | 1598 |
| head_no_helmet | head | small | 893 |
| low_light_blur | head | large | 1 |
| low_light_blur | head | medium | 87 |
| low_light_blur | head | small | 307 |
| low_light_blur | helmet | large | 8 |
| low_light_blur | helmet | medium | 369 |
| low_light_blur | helmet | small | 692 |
| low_light_blur | person | large | 47 |
| low_light_blur | person | medium | 100 |
| low_light_blur | person | small | 17 |
| partial_occlusion | head | medium | 158 |
| partial_occlusion | head | small | 619 |
| partial_occlusion | helmet | large | 17 |
| partial_occlusion | helmet | medium | 791 |
| partial_occlusion | helmet | small | 1516 |
| partial_occlusion | person | large | 544 |
| partial_occlusion | person | medium | 875 |
| partial_occlusion | person | small | 180 |
| small_distant | head | small | 6183 |
| small_distant | helmet | small | 18692 |

## `small_distant` contract

- Pasted instances: 24875
- Observed min-side range: 6.0–21.0 px
- Config target range: [8.0, 20.0] px
- Within target range: 24850 / 24875
