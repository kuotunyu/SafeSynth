# M10 synthetic preview statistics

- Images: 32
- Annotations: 115
- Filter pass/reject: 26 / 6
- COCO self-evaluation bbox mAP: `1.000`
- Output: `D:\sdg-data\02-safesynth\synthetic\h4_context_replace_preview`
- Hard negatives: **not used**; M9 remains blocked on kuotunyu signoff.

## Scenario counts

| scenario | count |
|---|---:|
| context_replacement | 32 |

## First filter rejection reason

| reason | count |
|---|---:|
| EXCESSIVE_OVERLAP | 1 |
| NO_CHANGE | 5 |

## Pasted instances by scenario, class, and COCO size bucket

| scenario | class | size | count |
|---|---|---|---:|
| context_replacement | head | small | 2 |
| context_replacement | helmet | large | 5 |
| context_replacement | helmet | medium | 19 |
| context_replacement | helmet | small | 6 |

## `small_distant` contract

- Pasted instances: 0
- Observed min-side range: None–None px
- Config target range: [8.0, 20.0] px
- Within target range: 0 / 0
