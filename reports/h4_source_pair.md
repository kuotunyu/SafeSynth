# Spike H4 — paste-artifact detectability

- Source run: `D:\sdg-data\02-safesynth\synthetic\m11_h4_seed42` (300 images)
- Examples: 2028 (1622 train / 406 group-disjoint test)
- HOG + HSV logistic-regression AUC: **0.9049**
- Bootstrap 95% CI: 0.8788–0.9289
- Scale-up maximum AUC: 0.60
- Decision: **FAIL — scale-up gate closed**

Real controls use each pasted cutout's exact original source.
Each exact source/paste pair shares one frozen source-group key,
so neither an object nor a video-near-duplicate group can cross the split.
