# Compliance operating point (EVAL-04)

Swept on the **frozen validation split** (756 images) using `filtered_syn` at `checkpoint-3066`. Never swept on Test: `sweep_operating_points` takes the split name and raises on `"test"`, so tuning on the test set fails rather than producing a number.

This threshold is deliberately separate from mAP evaluation, which integrates over every confidence. A deployed compliance check has to commit to one point.

![sweep](reports/figures/compliance_sweep_filtered_syn.png)

| threshold | bare-head recall | compliance precision | pred. compliant | pred. non-compliant |
|---:|---:|---:|---:|---:|
| 0.005 | 0.9760 | 0.0348 | 82747 | 48546 |
| 0.010 | 0.9725 | 0.0725 | 39506 | 21818 |
| 0.015 | 0.9665 | 0.1507 | 18803 | 9679 |
| 0.020 | 0.9557 | 0.2550 | 10987 | 5244 |
| 0.025 | 0.9509 | 0.3535 | 7827 | 2978 |
| 0.030 | 0.9353 | 0.4390 | 6178 | 2041 |
| 0.035 | 0.9246 | 0.5176 | 5091 | 1553 |
| 0.040 | 0.9126 | 0.5821 | 4329 | 1260 |
| 0.045 | 0.8731 | 0.6354 | 3730 | 1067 |
| 0.050 | 0.8395 | 0.6839 | 3243 | 953 |
| 0.055 | 0.7856 | 0.7456 | 2689 | 831 |
| 0.060 | 0.7162 | 0.7829 | 2266 | 728 |
| 0.065 **←** | 0.6395 | 0.8076 | 1933 | 625 |
| 0.070 | 0.5665 | 0.8346 | 1620 | 530 |
| 0.075 | 0.4898 | 0.8620 | 1348 | 449 |
| 0.080 | 0.4240 | 0.8894 | 1112 | 381 |
| 0.085 | 0.3617 | 0.9088 | 921 | 322 |
| 0.090 | 0.2910 | 0.9194 | 744 | 254 |
| 0.095 | 0.2395 | 0.9178 | 608 | 210 |
| 0.100 | 0.1856 | 0.9203 | 477 | 163 |
| 0.105 | 0.1509 | 0.9255 | 376 | 132 |
| 0.110 | 0.1126 | 0.9264 | 299 | 99 |
| 0.115 | 0.0922 | 0.9267 | 232 | 81 |
| 0.120 | 0.0790 | 0.9218 | 179 | 69 |
| 0.125 | 0.0671 | 0.9065 | 139 | 59 |
| 0.130 | 0.0467 | 0.8889 | 108 | 41 |
| 0.135 | 0.0335 | 0.8780 | 82 | 30 |
| 0.140 | 0.0204 | 0.9000 | 60 | 19 |
| 0.145 | 0.0168 | 0.9167 | 48 | 16 |
| 0.150 | 0.0132 | 0.8947 | 38 | 12 |
| 0.155 | 0.0120 | 0.9375 | 32 | 10 |
| 0.160 | 0.0108 | 0.9545 | 22 | 9 |
| 0.165 | 0.0072 | 0.9412 | 17 | 6 |
| 0.170 | 0.0060 | 0.9231 | 13 | 5 |
| 0.175 | 0.0048 | 0.9091 | 11 | 4 |
| 0.180 | 0.0036 | 0.7500 | 4 | 3 |
| 0.185 | 0.0012 | 0.7500 | 4 | 1 |
| 0.190 | 0.0012 | 0.7500 | 4 | 1 |
| 0.195 | 0.0012 | 0.5000 | 2 | 1 |
| 0.200 | 0.0000 | 0.0000 | 1 | 0 |
| 0.205 | 0.0000 | 0.0000 | 1 | 0 |
| 0.210 | 0.0000 | — | 0 | 0 |
| 0.215 | 0.0000 | — | 0 | 0 |
| 0.220 | 0.0000 | — | 0 | 0 |
| 0.225 | 0.0000 | — | 0 | 0 |
| 0.230 | 0.0000 | — | 0 | 0 |
| 0.235 | 0.0000 | — | 0 | 0 |
| 0.240 | 0.0000 | — | 0 | 0 |
| 0.245 | 0.0000 | — | 0 | 0 |
| 0.250 | 0.0000 | — | 0 | 0 |
| 0.255 | 0.0000 | — | 0 | 0 |
| 0.260 | 0.0000 | — | 0 | 0 |
| 0.265 | 0.0000 | — | 0 | 0 |
| 0.270 | 0.0000 | — | 0 | 0 |
| 0.275 | 0.0000 | — | 0 | 0 |
| 0.280 | 0.0000 | — | 0 | 0 |
| 0.285 | 0.0000 | — | 0 | 0 |
| 0.290 | 0.0000 | — | 0 | 0 |
| 0.295 | 0.0000 | — | 0 | 0 |
| 0.300 | 0.0000 | — | 0 | 0 |
| 0.350 | 0.0000 | — | 0 | 0 |
| 0.400 | 0.0000 | — | 0 | 0 |
| 0.500 | 0.0000 | — | 0 | 0 |

## Selected

- `compliance.score_threshold` = **0.07**
- bare-head recall 0.6395
- compliance precision 0.8076 (floor 0.80)
- ground-truth bare heads in validation: 835

Write this value into `configs/evaluation.yaml` under `compliance.score_threshold` and change its `source:` tag from `validation (placeholder)` to `validation`. It is then frozen: re-selecting it after seeing Test results would be tuning on Test.
