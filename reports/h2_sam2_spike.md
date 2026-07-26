# M7 / Spike H2 — SAM2 small-box quality

Date: 2026-07-27. Input: frozen Train only, seed 42.

## Method

- 60 boxes, split evenly by shortest side: 8–20 px, 21–34 px, and 36–133 px.
- Class counts across the sample: 31 helmet, 15 head, 13 person.
- Every box ran in three modes: full image, contextual crop at effective 1024,
  and contextual crop at effective 512 on a centre-padded native 1024 canvas.
- Recorded `iou_scores`, `object_score_logits`, coverage, component count, and
  solidity.
- The three full 20-row comparison sheets were opened and visually inspected.

## Result

| size bin | full IoU p10 / p50 | crop-1024 IoU p10 / p50 | crop-512 IoU p10 / p50 |
|---|---:|---:|---:|
| 8–20 px | 0.815 / 0.877 | 0.799 / 0.865 | **0.850 / 0.887** |
| 21–34 px | 0.812 / 0.902 | 0.695 / 0.926 | **0.769 / 0.914** |
| 36–133 px | 0.630 / 0.914 | 0.793 / 0.920 | **0.655 / 0.910** |

The modes were visually close on most instances. Crop-512 preserved the tiny
8–20 px silhouettes at least as well as the other modes and had the strongest
small-bin p10. It also avoids turning a handful of source pixels into a
needlessly large, blocky prompt. Pass 2 therefore uses **crop-512 centred on an
edge-padded 1024 canvas**.

Five obvious failures were excluded from the visually-good threshold pool:
annotations 24111, 19046, 23896, 23925, and 20327. They showed fragmentation,
near-empty person masks, or a mask that selected only a small sub-region. The
remaining 55 crop-512 masks gave:

- IoU p10 = 0.821875 → configured `min_iou_score = 0.82`
- object-score logit p10 = 19.5 → configured `min_object_score_logit = 19.5`

Those are Pass 2 crop-512 thresholds. The visually-good full-image group used
by Pass 1 has IoU p10 = 0.804688 and object-logit p10 = 18.3, so Pass 1 stores
separate 0.80 / 18.3 thresholds. Sharing the crop object-logit threshold would
make valid full-image masks fall back to boxes for the wrong reason.

There was no small-bin segmentation collapse. The hard floor is not raised:
it remains 16 px and 400 px² because sub-16 px RGB sources are visibly
pixel-limited even when their binary silhouette is plausible. The preferred
tier is separately calibrated at 23 px / 667 px²; this keeps small material
available without making it the default.

## Visual evidence

- `reports/figures/h2_sam2_very_small.png`
- `reports/figures/h2_sam2_medium.png`
- `reports/figures/h2_sam2_larger.png`

Machine-readable masks and metrics are stored outside Git at
`D:/sdg-data/02-safesynth/cache/h2/h2_results.json`.
