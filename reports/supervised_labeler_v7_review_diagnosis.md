# Supervised labeler v7 owner-review diagnosis

- Evidence class: **revealed Train-only audit; not gate-eligible**
- Owner problem cells: **8, 11, 13, 36, 39**
- Unmatched helmet instances: **6**
- Reason counts: **{"matching_box_below_frozen_score_threshold": 1, "removed_by_frozen_geometry_filter": 5}**
- Validation/Test images read: **0 / 0**
- Whole-image generations: **0**

| Cell | Image | Misses | Best raw IoU / score / reason |
|---:|---:|---:|---|
| 8 | 1039 | 1 | 0.960 / 0.0559 / removed_by_frozen_geometry_filter |
| 11 | 1282 | 1 | 0.867 / 0.0052 / removed_by_frozen_geometry_filter |
| 13 | 1357 | 1 | 0.834 / 0.0645 / removed_by_frozen_geometry_filter |
| 36 | 3920 | 2 | 0.925 / 0.0610 / removed_by_frozen_geometry_filter; 0.830 / 0.0684 / removed_by_frozen_geometry_filter |
| 39 | 4236 | 1 | 0.550 / 0.0052 / matching_box_below_frozen_score_threshold |

## Revealed audit threshold grid

| Threshold | Precision | Recall | F1 | Median IoU |
|---:|---:|---:|---:|---:|
| 0.0010 | 0.0439 | 0.9851 | 0.0840 | 0.8474 |
| 0.0020 | 0.0445 | 0.9851 | 0.0852 | 0.8474 |
| 0.0050 | 0.1345 | 0.9801 | 0.2365 | 0.8476 |
| 0.0100 | 0.6846 | 0.9502 | 0.7958 | 0.8476 |
| 0.0150 | 0.8942 | 0.9254 | 0.9095 | 0.8500 |
| 0.0200 | 0.9436 | 0.9154 | 0.9293 | 0.8500 |
| 0.0230 | 0.9579 | 0.9055 | 0.9309 | 0.8511 |
| 0.0250 | 0.9630 | 0.9055 | 0.9333 | 0.8511 |
| 0.0300 | 0.9665 | 0.8607 | 0.9105 | 0.8523 |
| 0.0400 | 0.9737 | 0.5522 | 0.7048 | 0.8613 |
| 0.0500 | 1.0000 | 0.2587 | 0.4111 | 0.8507 |
