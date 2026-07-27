# Supervised labeler v5 pre-registration

Status at registration: v4 failed the frozen Train-only audit because recall
was 0.5257. A consumed-set diagnostic found that lower score thresholds came
close to the recall gate, but fixed relative-area and relative-height filters
did not recover precision across the v1, v3, and v4 failed audits. No
Validation/Test image or FLUX whole-image output was read.

## Frozen change

v5 changes model capacity, not the target class or evaluation protocol:

- base model: `PekingU/rtdetr_v2_r50vd`;
- immutable revision: `282494075698cab9faa1096ae26856890030c817`;
- license: Apache-2.0;
- model file: 172,176,856 bytes;
- input size: 640;
- training seed: 20260809;
- split seed: 20260813;
- Train-only calibration precision floor: 0.85;
- final untouched Train-only audit gates: precision 0.80, recall 0.70, median
  matched IoU 0.60.

The official RT-DETR comparison reports 42M parameters and 53.4 COCO AP for
RT-DETRv2-L/R50, compared with 20M parameters and 48.1 AP for
RT-DETRv2-S/R18. The pinned Hugging Face model card identifies the repository
as Apache-2.0:

- https://github.com/lyuwenyu/RT-DETR#implementations
- https://huggingface.co/PekingU/rtdetr_v2_r50vd/tree/main

## Data boundary

All v1-v4 calibration and failed-audit groups become v5 calibration history.
A new group-disjoint set of 48 Train images is selected but remains unread
until the single final audit. Validation and Test remain unread. The generation
gate stays locked regardless of smoke or calibration results; only the frozen
numeric audit and subsequent human review can open it.
