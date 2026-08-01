# Bare-head recall against detector confidence (EVAL-05b)

Split: **test**. IoU 0.50, one-to-one greedy matching, highest score first. Re-aggregated from the stored detections (EVAL-12), no inference.

⚠️ **The row at threshold 0 is a recall CEILING, not a result.** The detector emits a fixed 300 queries per image, so with no score floor almost every bare head finds some box and every arm scores near the top. Read the main-table value at the frozen compliance operating point of **0.07**.

| threshold | `real_only` | `standard_aug` | `unfiltered_syn` | `filtered_syn` | spread |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.9875 | 0.9875 | 0.9898 | 0.9886 | 0.0023 |
| 0.02 | 0.9818 | 0.9499 | 0.9602 | 0.9693 | 0.0319 |
| 0.05 | 0.9283 | 0.7440 | 0.7304 | 0.8567 | 0.1980 |
| 0.07 **←** | 0.8931 | 0.4687 | 0.3572 | 0.5575 | 0.5358 |
| 0.10 | 0.7349 | 0.1092 | 0.0523 | 0.1832 | 0.6826 |
| 0.12 | 0.5199 | 0.0239 | 0.0114 | 0.0739 | 0.5085 |
| 0.15 | 0.2253 | 0.0057 | 0.0000 | 0.0216 | 0.2253 |
| 0.20 | 0.0239 | 0.0000 | 0.0000 | 0.0023 | 0.0239 |
| 0.30 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The `spread` column is the point: where it is near zero the metric is not measuring anything that distinguishes these models, and quoting a single number from that region would suggest a tie that the underlying detectors do not have.
