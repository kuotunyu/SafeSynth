# Spike H4 — paste-artifact detectability

- Source run: `D:\sdg-data\02-safesynth\synthetic\m11_h4_poisson_seed42` (300 images)
- Examples: 2028 (1626 train / 402 group-disjoint test)
- HOG + HSV logistic-regression AUC: **0.8869**
- Bootstrap 95% CI: 0.8551–0.9170
- Scale-up maximum AUC: 0.60
- Decision: **FAIL — scale-up gate closed**

Real controls match class, H4 fold, nearest log pixel width/height.
Both labels use the same frozen-group hash, so video near-duplicates cannot
cross the split; fold-level class and target-resolution counts are paired.
