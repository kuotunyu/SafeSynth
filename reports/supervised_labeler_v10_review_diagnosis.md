# Supervised labeler v10 owner-review diagnosis

- Evidence: **revealed Train-only audit; not gate-eligible**
- Owner-confirmed GT-defect cells: **31, 41, 42**
- Frozen numeric metrics contaminated by GT defects: **yes**
- Problem cells: **6, 7, 10, 27, 29, 31, 34, 39, 40, 41, 42, 47**
- Automatic miss causes against existing GT only: **{"matching_box_below_frozen_score_threshold": 4, "removed_by_frozen_geometry_filter": 4}**
- Highest numerical FP score in owner FP cells: **0.0942**
- Validation/Test images read: **0 / 0**
- Whole-image generations: **0**

| Cell | Owner category | GT | Pred | Numeric FP | Numeric FN |
|---:|---|---:|---:|---:|---:|
| 6 | model_missed_helmet | 1 | 0 | 0 | 1 |
| 7 | model_missed_helmet | 1 | 0 | 0 | 1 |
| 10 | model_missed_helmet | 4 | 2 | 0 | 2 |
| 27 | model_missed_helmet | 7 | 6 | 0 | 1 |
| 29 | model_false_positive | 2 | 6 | 4 | 0 |
| 31 | dataset_gt_false_positive_label | 3 | 2 | 0 | 1 |
| 34 | model_false_positive | 1 | 2 | 1 | 0 |
| 39 | model_missed_helmet | 1 | 0 | 0 | 1 |
| 40 | model_missed_helmet | 2 | 1 | 0 | 1 |
| 41 | dataset_gt_missed_helmet | 4 | 5 | 1 | 0 |
| 42 | model_and_dataset_gt_missed_helmet | 1 | 1 | 0 | 0 |
| 47 | model_false_positive | 0 | 7 | 7 | 0 |
