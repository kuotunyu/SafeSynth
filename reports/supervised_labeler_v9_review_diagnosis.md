# Supervised labeler v9 owner-review diagnosis

- Evidence class: **revealed Train-only audit; not gate-eligible**
- Owner problem cells: **6, 11, 12, 37**
- Automatically unmatched truths in the four reported cells: **4**
- Miss cause counts: **{"below_score_threshold_and_removed_by_geometry_filter": 2, "matching_box_below_frozen_score_threshold": 2}**
- Highest owner-reported false-positive score: **0.0559**
- Validation/Test images read: **0 / 0**
- Whole-image generations: **0**

| Cell | Owner category | Numeric FP | Numeric FN | Best raw IoU / score / cause |
|---:|---|---:|---:|---|
| 6 | background_or_other_false_positive | 1 | 0 | none |
| 11 | missed_helmet | 0 | 2 | 0.837 / 0.0092 / below_score_threshold_and_removed_by_geometry_filter; 0.874 / 0.0094 / below_score_threshold_and_removed_by_geometry_filter |
| 12 | background_or_other_false_positive | 1 | 0 | none |
| 37 | missed_helmet | 0 | 2 | 0.869 / 0.0069 / matching_box_below_frozen_score_threshold; 0.932 / 0.0059 / matching_box_below_frozen_score_threshold |

## Revealed audit threshold sensitivity

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.010 | 189 | 1675 | 5 | 0.1014 | 0.9742 | 0.1837 |
| 0.015 | 187 | 514 | 7 | 0.2668 | 0.9639 | 0.4179 |
| 0.020 | 186 | 199 | 8 | 0.4831 | 0.9588 | 0.6425 |
| 0.030 | 184 | 60 | 10 | 0.7541 | 0.9485 | 0.8402 |
| 0.040 | 180 | 32 | 14 | 0.8491 | 0.9278 | 0.8867 |
| 0.050 | 176 | 13 | 18 | 0.9312 | 0.9072 | 0.9191 |
| 0.056 | 173 | 9 | 21 | 0.9505 | 0.8918 | 0.9202 |
| 0.060 | 167 | 6 | 27 | 0.9653 | 0.8608 | 0.9101 |
| 0.070 | 149 | 4 | 45 | 0.9739 | 0.7680 | 0.8588 |
| 0.080 | 123 | 3 | 71 | 0.9762 | 0.6340 | 0.7687 |
| 0.100 | 78 | 2 | 116 | 0.9750 | 0.4021 | 0.5693 |
