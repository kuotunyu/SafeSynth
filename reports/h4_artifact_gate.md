# Spike H4 — paste-artifact detectability

- Source run: `D:\sdg-data\02-safesynth\synthetic\m11_h4_seed42` (300 images)
- Examples: 2178 (1748 train / 430 group-disjoint test)
- HOG + HSV logistic-regression AUC: **0.7798**
- Bootstrap 95% CI: 0.7352–0.8202
- Scale-up maximum AUC: 0.60
- Decision: **FAIL — fix blending first**

Real controls match class, H4 fold, and nearest log pixel width/height.
Both labels use the same frozen-group hash, so video near-duplicates cannot
cross the split; fold-level class and target-resolution counts are paired.
