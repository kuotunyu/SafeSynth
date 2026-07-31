# M12 filter threshold sensitivity

- Source records: `D:\sdg-data\02-safesynth\synthetic\m13_pool_1x\records.jsonl`
- Samples: 14000
- Baseline acceptance: 29.84%
- Alarm threshold: 15.0 percentage points
- Thresholds above alarm: 0

Each numeric rule leaf is changed independently by -20% and +20%.

| threshold | base | -20% rate | +20% rate | max swing (pp) | alarm |
|---|---:|---:|---:|---:|:---:|
| `rules.phash_dedup.max_hamming_to_accepted_synthetic` | 6 | 36.37% | 29.84% | 6.54 |  |
| `rules.annotation_legibility.min_object_mean_luma.helmet` | 45.58 | 30.15% | 26.07% | 3.76 |  |
| `rules.min_visible_area.min_area_px` | 99 | 32.94% | 27.07% | 3.10 |  |
| `rules.mask_to_box_coverage.head.1` | 0.95 | 28.26% | 29.84% | 1.57 |  |
| `rules.mask_to_box_coverage.helmet.0` | 0.2456 | 29.84% | 28.40% | 1.44 |  |
| `rules.phash_dedup.min_changed_pixel_ratio` | 0.005 | 31.19% | 28.46% | 1.38 |  |
| `rules.overlap.max_overlap_score_same_class` | 0.6364 | 28.54% | 30.67% | 1.29 |  |
| `rules.annotation_legibility.min_object_mean_luma.head` | 23.19 | 29.85% | 28.92% | 0.91 |  |
| `rules.overlap.max_overlap_iou_same_class` | 0.252 | 29.12% | 30.46% | 0.71 |  |
| `rules.mask_to_box_coverage.head.0` | 0.395 | 29.84% | 29.21% | 0.62 |  |
| `rules.mask_to_box_coverage.person.1` | 0.8118 | 29.36% | 29.84% | 0.48 |  |
| `rules.mask_to_box_coverage.helmet.1` | 0.95 | 29.41% | 29.84% | 0.43 |  |
| `rules.phash_dedup.max_hamming_to_other_real_image` | 6 | 30.16% | 29.84% | 0.32 |  |
| `rules.visible_fraction.helmet` | 0.3 | 30.03% | 29.54% | 0.30 |  |
| `rules.visible_fraction.head` | 0.3 | 29.98% | 29.70% | 0.14 |  |
| `rules.annotation_legibility.min_object_mean_luma.person` | 24.26 | 29.84% | 29.73% | 0.11 |  |
| `rules.size_ratio.containment_threshold` | 0.7 | 29.74% | 29.85% | 0.09 |  |
| `rules.size_ratio.head_over_person_width.0` | 0.2034 | 29.84% | 29.80% | 0.04 |  |
| `rules.bbox_in_bounds.min_box_side_px` | 4 | 29.84% | 29.81% | 0.03 |  |
| `rules.size_ratio.helmet_over_person_area.0` | 0.0153 | 29.84% | 29.81% | 0.02 |  |
| `rules.clipping_artifact.max_seam_energy_ratio` | 5.577 | 29.82% | 29.84% | 0.01 |  |
| `rules.size_ratio.helmet_over_person_area.1` | 0.6953 | 29.82% | 29.84% | 0.01 |  |
| `rules.size_ratio.head_over_person_area.0` | 0.0225 | 29.84% | 29.83% | 0.01 |  |
| `rules.size_ratio.head_over_person_area.1` | 0.3494 | 29.84% | 29.84% | 0.01 |  |
| `rules.bbox_in_bounds.min_inside_ratio` | 0.6 | 29.84% | 29.84% | 0.00 |  |
| `rules.clipping_artifact.poke_through_range.0` | 0.02 | 29.84% | 29.84% | 0.00 |  |
| `rules.clipping_artifact.poke_through_range.1` | 0.2 | 29.84% | 29.84% | 0.00 |  |
| `rules.clipping_artifact.seam_band_px` | 2 | 29.84% | 29.84% | 0.00 |  |
| `rules.hard_negative_no_overlap.max_iou_with_annotation` | 0.02 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.dy_max` | 0.75 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.dy_min` | -0.1 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.k_x` | 0.35 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.min_iou_helmet_head` | 0.05 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.overlap_y_max` | 0.75 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.overlap_y_min` | 0.15 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.r_h.0` | 0.25 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.r_h.1` | 1.1 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.r_w.0` | 0.75 | 29.84% | 29.84% | 0.00 |  |
| `rules.helmet_above_head.r_w.1` | 1.6 | 29.84% | 29.84% | 0.00 |  |
| `rules.mask_to_box_coverage.person.0` | 0.1723 | 29.84% | 29.84% | 0.00 |  |
| `rules.phash_dedup.hash_size` | 8 | 29.84% | 29.84% | 0.00 |  |
| `rules.size_ratio.head_over_person_width.1` | 0.8182 | 29.84% | 29.84% | 0.00 |  |
| `rules.size_ratio.head_top_within_person` | 0.45 | 29.84% | 29.84% | 0.00 |  |
| `rules.visible_fraction.person` | 0.25 | 29.84% | 29.84% | 0.00 |  |
