# M8 cutout bank report

Input is frozen Train only. Val/Test sources are a hard failure.

## Funnel

- Candidates: 17815
- Accepted: 7255
- Rejected: 10560

| first rejection reason | count |
|---|---:|
| G2_HARD_FLOOR | 4347 |
| G4_ASPECT_RATIO | 162 |
| G5_IMAGE_EDGE | 2548 |
| G6_OCCLUDED | 1340 |
| PERSON_GROUP_CAP | 22 |
| SAM2_EDGE_TOUCH_TOP | 65 |
| SAM2_HOLE_FILL_RATIO | 6 |
| SAM2_IOU_SCORE | 1298 |
| SAM2_MASK_AREA_PX | 346 |
| SAM2_MASK_TO_BOX_COVERAGE | 189 |
| SAM2_OBJECT_SCORE_LOGIT | 59 |
| SAM2_OUTSIDE_BOX_RATIO | 32 |
| SAM2_SECOND_COMPONENT_RATIO | 72 |
| SAM2_SOLIDITY | 74 |

## Accepted material

| class | accepted | preferred tier |
|---|---:|---:|
| helmet | 5578 | 4609 |
| head | 1564 | 1125 |
| person | 113 | 109 |

- `n_person_cutouts`: 113
- `n_distinct_person_groups`: 77
- Test blocklist hits: 0
- Manifest rows == PNG files: 7255 == 7255
- Manifest SHA256: `7e5b7d66da2b27ddf865e8166bc78aba1bdda4116a0955292dc7384856b17802`
- Reject ledger SHA256: `dd57e0a1f2163111c167602cb85936ac6331c2332066b73487aca46ef66c2e05`

The funnel above is aggregated directly from `bank_rejects.jsonl`; accepted + rejected equals the frozen candidate count.

## Mask reproducibility

- Re-run accepted masks: 100
- Byte-identical masks: 100
- Mismatches: 0
- Model: `facebook/sam2.1-hiera-large` (bfloat16)
