# Supervised labeler v11 geometry diagnosis

- Evidence: **revealed Train-only history; not gate-eligible**
- Revealed images read: **573**
- Quarantined GT-defect images read: **0**
- Validation/Test images read: **0 / 0**
- Whole-image generations: **0**

| Candidate | Area | Height | Min aspect | Recovered | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v10_frozen | 0.14 | 0.40 | 0.25 | 0/4 | 0.050 | 0.8569 | 0.8862 | 0.8713 |
| height_045 | 0.14 | 0.45 | 0.25 | 2/4 | 0.050 | 0.8581 | 0.8953 | 0.8763 |
| height_050 | 0.14 | 0.50 | 0.25 | 2/4 | 0.050 | 0.8583 | 0.8966 | 0.8770 |
| edge_large_060 | 0.15 | 0.60 | 0.20 | 4/4 | 0.050 | 0.8529 | 0.9030 | 0.8772 |
| edge_large_070 | 0.18 | 0.70 | 0.18 | 4/4 | 0.050 | 0.8499 | 0.9114 | 0.8796 |

Recommended: **edge_large_060**.
