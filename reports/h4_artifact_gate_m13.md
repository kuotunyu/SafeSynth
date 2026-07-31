# Spike H4 — paste-artifact detectability

- Source run: `D:\sdg-data\02-safesynth\synthetic\m13_pool_1x` (300 images)
- Examples: 75106 (60482 train / 14624 group-disjoint test)
- HOG + HSV logistic-regression AUC: **0.9159**
- Bootstrap 95% CI: 0.9115–0.9201
- Scale-up maximum AUC: 0.60
- Decision: **FAIL — scale-up gate closed**

Real controls match class, H4 fold, nearest log pixel width/height.
Both labels use the same frozen-group hash, so video near-duplicates cannot
cross the split; fold-level class and target-resolution counts are paired.
