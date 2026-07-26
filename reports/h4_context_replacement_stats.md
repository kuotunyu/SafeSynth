# M10 synthetic preview statistics

- Images: 300
- Annotations: 1595
- Filter pass/reject: 214 / 86
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\m11_h4_context_replace`
- Hard negatives: **not used**; M9 remains blocked on kuotunyu signoff.

## Scenario counts

| scenario | count |
|---|---:|
| context_replacement | 300 |

## First filter rejection reason

| reason | count |
|---|---:|
| BOX_TOO_SMALL | 6 |
| EXCESSIVE_OVERLAP | 6 |
| LOW_VISIBLE_FRACTION | 4 |
| NEAR_DUPLICATE_REAL | 7 |
| NEAR_DUPLICATE_SYNTHETIC | 5 |
| NO_CHANGE | 58 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| context_replacement | head | large | 1 |
| context_replacement | head | medium | 14 |
| context_replacement | head | small | 24 |
| context_replacement | helmet | large | 21 |
| context_replacement | helmet | medium | 155 |
| context_replacement | helmet | small | 85 |

## `small_distant` contract

- Pasted instances: 0
- Observed min-side range: None–None px
- Config target range: [8.0, 20.0] px
- Within target range: 0 / 0
