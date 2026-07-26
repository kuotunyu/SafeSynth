# M12 filter threshold sensitivity

- Source records: `D:\sdg-data\02-safesynth\synthetic\m11_h4_seed42\records.jsonl`
- Samples: 300
- Baseline acceptance: 65.33%
- Alarm threshold: 15.0 percentage points
- Thresholds above alarm: 0

Each numeric rule leaf is changed independently by -20% and +20%.

| threshold | base | -20% rate | +20% rate | max swing (pp) | alarm |
|---|---:|---:|---:|---:|:---:|
| `rules.phash_dedup.min_changed_pixel_ratio` | 0.005 | 68.00% | 59.33% | 6.00 |  |
| `rules.min_visible_area.min_area_px` | 99 | 70.67% | 59.67% | 5.67 |  |
| `rules.overlap.max_overlap_score_same_class` | 0.6364 | 63.00% | 67.33% | 2.33 |  |
| `rules.mask_to_box_coverage.head.1` | 0.95 | 63.33% | 65.33% | 2.00 |  |
| `rules.mask_to_box_coverage.helmet.0` | 0.2456 | 65.33% | 63.67% | 1.67 |  |
| `rules.mask_to_box_coverage.helmet.1` | 0.95 | 64.33% | 65.33% | 1.00 |  |
| `rules.overlap.max_overlap_iou_same_class` | 0.252 | 64.33% | 65.67% | 1.00 |  |
| `rules.mask_to_box_coverage.head.0` | 0.395 | 65.33% | 64.67% | 0.67 |  |
| `rules.mask_to_box_coverage.person.1` | 0.8118 | 64.67% | 65.33% | 0.67 |  |
| `rules.visible_fraction.head` | 0.3 | 66.00% | 64.67% | 0.67 |  |
| `rules.visible_fraction.helmet` | 0.3 | 66.00% | 65.00% | 0.67 |  |
| `rules.size_ratio.containment_threshold` | 0.7 | 65.33% | 65.67% | 0.33 |  |
| `rules.bbox_in_bounds.min_box_side_px` | 4 | 65.33% | 65.33% | 0.00 |  |
| `rules.bbox_in_bounds.min_inside_ratio` | 0.6 | 65.33% | 65.33% | 0.00 |  |
| `rules.clipping_artifact.max_seam_energy_ratio` | 5.577 | 65.33% | 65.33% | 0.00 |  |
| `rules.clipping_artifact.poke_through_range.0` | 0.02 | 65.33% | 65.33% | 0.00 |  |
| `rules.clipping_artifact.poke_through_range.1` | 0.2 | 65.33% | 65.33% | 0.00 |  |
| `rules.clipping_artifact.seam_band_px` | 2 | 65.33% | 65.33% | 0.00 |  |
| `rules.hard_negative_no_overlap.max_iou_with_annotation` | 0.02 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.dy_max` | 0.75 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.dy_min` | -0.1 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.k_x` | 0.35 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.min_iou_helmet_head` | 0.05 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.overlap_y_max` | 0.75 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.overlap_y_min` | 0.15 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.r_h.0` | 0.25 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.r_h.1` | 1.1 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.r_w.0` | 0.75 | 65.33% | 65.33% | 0.00 |  |
| `rules.helmet_above_head.r_w.1` | 1.6 | 65.33% | 65.33% | 0.00 |  |
| `rules.mask_to_box_coverage.person.0` | 0.1723 | 65.33% | 65.33% | 0.00 |  |
| `rules.phash_dedup.hash_size` | 8 | 65.33% | 65.33% | 0.00 |  |
| `rules.phash_dedup.max_hamming_to_accepted_synthetic` | 6 | 65.33% | 65.33% | 0.00 |  |
| `rules.phash_dedup.max_hamming_to_other_real_image` | 6 | 65.33% | 65.33% | 0.00 |  |
| `rules.size_ratio.head_over_person_area.0` | 0.0225 | 65.33% | 65.33% | 0.00 |  |
| `rules.size_ratio.head_over_person_area.1` | 0.3494 | 65.33% | 65.33% | 0.00 |  |
| `rules.size_ratio.head_over_person_width.0` | 0.2034 | 65.33% | 65.33% | 0.00 |  |
| `rules.size_ratio.head_over_person_width.1` | 0.8182 | 65.33% | 65.33% | 0.00 |  |
| `rules.size_ratio.head_top_within_person` | 0.45 | 65.33% | 65.33% | 0.00 |  |
| `rules.size_ratio.helmet_over_person_area.0` | 0.0153 | 65.33% | 65.33% | 0.00 |  |
| `rules.size_ratio.helmet_over_person_area.1` | 0.6953 | 65.33% | 65.33% | 0.00 |  |
| `rules.visible_fraction.person` | 0.25 | 65.33% | 65.33% | 0.00 |  |
