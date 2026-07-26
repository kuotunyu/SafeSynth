# M12 filter threshold sensitivity

- Source records: `D:\sdg-data\02-safesynth\synthetic\m10_seed42\records.jsonl`
- Samples: 32
- Baseline acceptance: 6.25%
- Alarm threshold: 15.0 percentage points
- Thresholds above alarm: 0

Each numeric rule leaf is changed independently by -20% and +20%.

| threshold | base | -20% rate | +20% rate | max swing (pp) | alarm |
|---|---:|---:|---:|---:|:---:|
| `rules.phash_dedup.max_hamming_to_any_real_image` | 6 | 12.50% | 6.25% | 6.25 |  |
| `rules.mask_to_box_coverage.head.1` | 0.8307 | 3.12% | 6.25% | 3.12 |  |
| `rules.visible_fraction.helmet` | 0.3 | 9.38% | 6.25% | 3.12 |  |
| `rules.bbox_in_bounds.min_box_side_px` | 4 | 6.25% | 6.25% | 0.00 |  |
| `rules.bbox_in_bounds.min_inside_ratio` | 0.6 | 6.25% | 6.25% | 0.00 |  |
| `rules.clipping_artifact.max_seam_energy_ratio` | 5.577 | 6.25% | 6.25% | 0.00 |  |
| `rules.clipping_artifact.poke_through_range.0` | 0.02 | 6.25% | 6.25% | 0.00 |  |
| `rules.clipping_artifact.poke_through_range.1` | 0.2 | 6.25% | 6.25% | 0.00 |  |
| `rules.clipping_artifact.seam_band_px` | 2 | 6.25% | 6.25% | 0.00 |  |
| `rules.hard_negative_no_overlap.max_iou_with_annotation` | 0.02 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.dy_max` | 0.75 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.dy_min` | -0.1 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.k_x` | 0.35 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.min_iou_helmet_head` | 0.05 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.overlap_y_max` | 0.75 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.overlap_y_min` | 0.15 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.r_h.0` | 0.25 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.r_h.1` | 1.1 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.r_w.0` | 0.75 | 6.25% | 6.25% | 0.00 |  |
| `rules.helmet_above_head.r_w.1` | 1.6 | 6.25% | 6.25% | 0.00 |  |
| `rules.mask_to_box_coverage.head.0` | 0.395 | 6.25% | 6.25% | 0.00 |  |
| `rules.mask_to_box_coverage.helmet.0` | 0.2456 | 6.25% | 6.25% | 0.00 |  |
| `rules.mask_to_box_coverage.helmet.1` | 0.7988 | 6.25% | 6.25% | 0.00 |  |
| `rules.mask_to_box_coverage.person.0` | 0.1723 | 6.25% | 6.25% | 0.00 |  |
| `rules.mask_to_box_coverage.person.1` | 0.8118 | 6.25% | 6.25% | 0.00 |  |
| `rules.min_visible_area.min_area_px` | 99 | 6.25% | 6.25% | 0.00 |  |
| `rules.overlap.max_overlap_iou_same_class` | 0.252 | 6.25% | 6.25% | 0.00 |  |
| `rules.overlap.max_overlap_score_same_class` | 0.6364 | 6.25% | 6.25% | 0.00 |  |
| `rules.phash_dedup.hash_size` | 8 | 6.25% | 6.25% | 0.00 |  |
| `rules.phash_dedup.max_hamming_to_accepted_synthetic` | 6 | 6.25% | 6.25% | 0.00 |  |
| `rules.phash_dedup.min_changed_pixel_ratio` | 0.005 | 6.25% | 6.25% | 0.00 |  |
| `rules.size_ratio.containment_threshold` | 0.7 | 6.25% | 6.25% | 0.00 |  |
| `rules.size_ratio.head_over_person_area.0` | 0.0225 | 6.25% | 6.25% | 0.00 |  |
| `rules.size_ratio.head_over_person_area.1` | 0.3494 | 6.25% | 6.25% | 0.00 |  |
| `rules.size_ratio.head_over_person_width.0` | 0.2034 | 6.25% | 6.25% | 0.00 |  |
| `rules.size_ratio.head_over_person_width.1` | 0.8182 | 6.25% | 6.25% | 0.00 |  |
| `rules.size_ratio.head_top_within_person` | 0.45 | 6.25% | 6.25% | 0.00 |  |
| `rules.size_ratio.helmet_over_person_area.0` | 0.0153 | 6.25% | 6.25% | 0.00 |  |
| `rules.size_ratio.helmet_over_person_area.1` | 0.6953 | 6.25% | 6.25% | 0.00 |  |
| `rules.visible_fraction.head` | 0.3 | 6.25% | 6.25% | 0.00 |  |
| `rules.visible_fraction.person` | 0.25 | 6.25% | 6.25% | 0.00 |  |
