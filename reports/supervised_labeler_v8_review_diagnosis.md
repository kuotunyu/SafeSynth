# Supervised labeler v8 owner-review diagnosis

- Evidence class: **revealed Train-only audit; not gate-eligible**
- Owner problem cells: **1, 6, 10, 16, 41, 42**
- Cell 41 category: **severe localization failure**
- Automatically unmatched truths in the six reported cells: **6**
- Miss/localization cause counts: **{"below_score_threshold_and_removed_by_geometry_filter": 1, "matching_box_below_frozen_score_threshold": 3, "no_matching_localization": 2}**
- Highest owner-reported false-positive score: **0.0825**
- Validation/Test images read: **0 / 0**
- Whole-image generations: **0**

| Cell | Owner category | Numeric FP | Numeric FN | Best raw IoU / score / cause |
|---:|---|---:|---:|---|
| 1 | background_or_other_false_positive | 1 | 0 | none |
| 6 | background_or_other_false_positive | 1 | 2 | 0.072 / 0.0200 / no_matching_localization; 0.911 / 0.0200 / matching_box_below_frozen_score_threshold |
| 10 | background_or_other_false_positive | 2 | 0 | none |
| 16 | missed_helmet | 1 | 2 | 0.912 / 0.0272 / matching_box_below_frozen_score_threshold; 0.718 / 0.0317 / matching_box_below_frozen_score_threshold |
| 41 | severe_localization_failure | 1 | 1 | 0.920 / 0.0114 / below_score_threshold_and_removed_by_geometry_filter |
| 42 | missed_helmet | 0 | 1 | 0.373 / 0.0107 / no_matching_localization |

## Revealed audit threshold sensitivity

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.035 | 164 | 14 | 25 | 0.9213 | 0.8677 | 0.8937 |
| 0.040 | 154 | 8 | 35 | 0.9506 | 0.8148 | 0.8775 |
| 0.045 | 142 | 5 | 47 | 0.9660 | 0.7513 | 0.8452 |
| 0.050 | 120 | 4 | 69 | 0.9677 | 0.6349 | 0.7668 |
| 0.060 | 79 | 2 | 110 | 0.9753 | 0.4180 | 0.5852 |
| 0.070 | 33 | 2 | 156 | 0.9429 | 0.1746 | 0.2946 |
| 0.080 | 22 | 1 | 167 | 0.9565 | 0.1164 | 0.2075 |
| 0.083 | 16 | 0 | 173 | 1.0000 | 0.0847 | 0.1561 |
| 0.090 | 10 | 0 | 179 | 1.0000 | 0.0529 | 0.1005 |
| 0.100 | 1 | 0 | 188 | 1.0000 | 0.0053 | 0.0105 |
