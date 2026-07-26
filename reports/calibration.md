# M6 / Spike H7 calibration

Input: frozen Train split only. Quantiles are p1, p5, p50, p95, p99.

## Box geometry

| class | n | area p1 / p50 / p99 | min-side p1 / p50 / p99 | aspect p1 / p50 / p99 |
|---|---:|---:|---:|---:|
| head | 4071 | 96.00 / 667.00 / 6125.40 | 7.00 / 23.00 / 71.30 | 0.4260 / 0.8276 / 3.6667 |
| helmet | 13219 | 99.00 / 980.00 / 17668.80 | 7.00 / 27.00 / 122.00 | 0.3913 / 0.8750 / 3.7143 |
| person | 525 | 423.28 / 7021.00 / 81684.24 | 13.00 / 58.00 / 251.28 | 0.1820 / 0.6949 / 4.8577 |

## Child contained in person

Containment threshold: 0.70.

| class | contained n | area ratio p1 / p50 / p99 | width ratio p1 / p50 / p99 | top position p1 / p50 / p99 |
|---|---:|---:|---:|---:|
| head | 22 | 0.0225 / 0.1296 / 0.3494 | 0.2034 / 0.4101 / 0.8182 | -0.0270 / 0.0000 / 0.4670 |
| helmet | 371 | 0.0153 / 0.1423 / 0.6953 | 0.1423 / 0.4600 / 0.9147 | -0.0735 / -0.0049 / 0.7353 |

## Same-class overlap

- Per-instance maximum IoMin p99: 0.636364
- Per-instance maximum IoU p99: 0.251965

## SAM2 mask distributions

Base-valid masks: 11273; base-rejected: 6542; images: 3500. Calibrated coverage/solidity were excluded from the base gate to avoid circular calibration.

| class | n | coverage p1 / p50 / p99 | solidity p1 / p50 / p99 | real-boundary seam p1 / p50 / p95 / p99 |
|---|---:|---:|---:|---:|
| head | 2532 | 0.3950 / 0.6645 / 0.8307 | 0.7508 / 0.9694 / 0.9891 | 0.8400 / 2.0651 / 5.3899 / 9.2458 |
| helmet | 8415 | 0.2456 / 0.5997 / 0.7988 | 0.5863 / 0.9609 / 0.9869 | 0.8079 / 1.8434 / 5.4664 / 10.3471 |
| person | 326 | 0.1723 / 0.4989 / 0.8118 | 0.5653 / 0.8336 / 0.9767 | 0.9277 / 1.9731 / 5.5770 / 12.8653 |

## Derived calibrated values

```json
{
  "aspect_ratio": {
    "head": [
      0.426,
      3.6666666666666665
    ],
    "helmet": [
      0.391304347826087,
      3.7142857142857144
    ],
    "person": [
      0.18195655620738863,
      4.857704918032785
    ]
  },
  "mask_to_box_coverage": {
    "head": [
      0.39496614946919206,
      0.8306557522839357
    ],
    "helmet": [
      0.24564976559037108,
      0.7987943722943724
    ],
    "person": [
      0.17234952688543462,
      0.8117908728415975
    ]
  },
  "max_overlap_iou_same_class": 0.2519650058095824,
  "max_overlap_score_same_class": 0.6363636363636364,
  "max_seam_energy_ratio": 5.576960001449256,
  "min_solidity": {
    "head": 0.7508225108225108,
    "helmet": 0.586314357567444,
    "person": 0.5653475158779077
  },
  "min_visible_area_px": 99,
  "preferred_min_area_px": 667,
  "preferred_min_side_px": 23,
  "size_ratio": {
    "head_over_person_area": [
      0.022494337926923787,
      0.34943816746633644
    ],
    "head_over_person_width": [
      0.2033769538349691,
      0.8182051282051281
    ],
    "helmet_over_person_area": [
      0.015315129036193524,
      0.6953162055335973
    ]
  }
}
```

The scalar preferred tier uses the minimum per-class p50 so scarce, smaller `head` sources are not excluded; it is a preference, not the H2 hard floor.

## Remaining `source: guess` lines (128)

These remain explicit priors and must be listed again in the M13 filter report.

```text
configs/compose.yaml:30: batch_images: 4                          # source: guess   (24 GB VRAM is not the limit here)
configs/compose.yaml:37: context_pad_frac: 0.60                   # source: guess   (fraction of max(w,h) added on every side)
configs/compose.yaml:51: morph_close_kernel: 3                    # source: guess
configs/compose.yaml:52: morph_close_iterations: 1                # source: guess
configs/compose.yaml:87: max_occlusion_by_others: 0.15              # source: guess
configs/compose.yaml:100: max_outside_box_ratio: 0.10              # source: guess   (leakage onto occluder/background)
configs/compose.yaml:101: max_second_component_ratio: 0.20         # source: guess   (>0.2 = SAM2 straddled two objects)
configs/compose.yaml:102: max_hole_fill_ratio: 1.25                # source: guess   (swiss-cheese mask)
configs/compose.yaml:111: helmet: 0.50                           # source: guess
configs/compose.yaml:112: head:   0.50                           # source: guess
configs/compose.yaml:114: person: 0.80                           # source: guess
configs/compose.yaml:120: min_distinct_person_groups: 100            # source: guess
configs/compose.yaml:134: max_composites_per_background: 4           # source: guess
configs/compose.yaml:135: max_uses_per_cutout: 25                    # source: guess
configs/compose.yaml:136: max_person_pastes_per_composite: 4         # source: guess
configs/compose.yaml:137: max_same_source_image_per_composite: 1     # source: guess   (relaxed to 2 for the crowded scenario)
configs/compose.yaml:147: helmet: 12                               # source: guess   (heads are near-upright on sites)
configs/compose.yaml:148: head:   12                               # source: guess
configs/compose.yaml:149: person: 5                                # source: guess
configs/compose.yaml:150: hard_negative: 25                        # source: guess   (rigid objects tolerate more)
configs/compose.yaml:162: feather_sigma_base: 0.60                 # source: guess
configs/compose.yaml:163: feather_sigma_per_px: 0.02               # source: guess   sigma = base + per_px * min(w,h)
configs/compose.yaml:164: feather_sigma_clip: [0.60, 2.00]         # source: guess
configs/compose.yaml:172: annulus_outer_scale: 2.0                 # source: guess   (target stats from a ring around the paste)
configs/compose.yaml:173: annulus_min_valid_px: 200                # source: guess   (below this, fall back to whole-background stats)
configs/compose.yaml:174: L_mean_beta: 0.70                        # source: guess
configs/compose.yaml:175: L_std_lambda_clip: [0.70, 1.40]          # source: guess
configs/compose.yaml:178: max_delta_mean_L: 25                     # source: guess
configs/compose.yaml:179: max_delta_mean_ab: 12                    # source: guess
configs/compose.yaml:183: noise_match_sigma_cap: 8.0               # source: guess
configs/compose.yaml:187: min_visible_fraction_pasted: 0.20        # source: guess
configs/compose.yaml:190: max_placement_retries: 8                 # source: guess
configs/compose.yaml:192: existing_keep_original_above: 0.60       # source: guess   (an annotator would not re-tighten)
configs/compose.yaml:193: existing_recompute_above: 0.20           # source: guess   (recompute from the visible mask)
configs/compose.yaml:196: min_inside_ratio: 0.60                   # source: guess   (before clipping to frame)
configs/compose.yaml:204: weight: 0.25                             # source: guess   (AP_small is headline metric #1)
configs/compose.yaml:208: cy_range: [0.10, 0.60]                   # source: guess   (near-horizon band)
configs/compose.yaml:213: weight: 0.25                             # source: guess   (bare-head recall is headline #2)
configs/compose.yaml:215: submode_paste_head_prob: 0.40            # source: guess
configs/compose.yaml:216: submode_helmet_to_head_swap_prob: 0.60   # source: guess
configs/compose.yaml:220: swap_inpaint_dilate_px: 3                # source: guess
configs/compose.yaml:221: swap_inpaint_radius: 3                   # source: guess
configs/compose.yaml:222: swap_head_oversize_factor: 1.05          # source: guess
configs/compose.yaml:223: swap_max_residual_helmet_fraction: 0.05  # source: guess   (assert; crash if exceeded)
configs/compose.yaml:227: weight: 0.15                             # source: guess
configs/compose.yaml:231: target_occlusion_ratio: [0.25, 0.65]     # source: guess
configs/compose.yaml:235: weight: 0.12                             # source: guess   (capped: the person bank is weak, ADR-003)
configs/compose.yaml:238: min_visible_fraction: 0.35               # source: guess
configs/compose.yaml:239: max_pairwise_iou: 0.65                   # source: guess
configs/compose.yaml:243: weight: 0.13                             # source: guess   (must be large enough to move FP rate)
configs/compose.yaml:249: weight: 0.10                             # source: guess   (dedicated bucket; ALSO applied as a
configs/compose.yaml:262: prob_given_postfx: 0.60                  # source: guess
configs/compose.yaml:263: gamma: [1.8, 3.2]                        # source: guess
configs/compose.yaml:264: gain: [0.45, 0.80]                       # source: guess
configs/compose.yaml:265: noise_sigma: [3, 10]                     # source: guess
configs/compose.yaml:266: wb_gain_r: [0.92, 1.00]                  # source: guess
configs/compose.yaml:267: wb_gain_b: [1.00, 1.12]                  # source: guess
configs/compose.yaml:269: prob_given_postfx: 0.50                  # source: guess
configs/compose.yaml:270: kernel_lengths: [3, 5, 7, 9]             # source: guess
configs/compose.yaml:273: always: true                             # source: guess
configs/compose.yaml:274: quality: [70, 95]                        # source: guess
configs/compose.yaml:280: mined_fraction: 0.70                       # source: guess   (domain texture beats procedural control)
configs/compose.yaml:281: procedural_fraction: 0.30                  # source: guess
configs/compose.yaml:285: hue_deg: [10, 50]                        # source: guess   (H6 widened: 20-40 yielded 36/200)
configs/compose.yaml:286: min_saturation: 0.40                     # source: guess
configs/compose.yaml:287: min_value: 0.35                          # source: guess
configs/compose.yaml:288: contour_area_px: [100, 8000]             # source: guess   (H6: enough guarded items for 8x8)
configs/compose.yaml:289: circularity: [0.25, 0.98]                # source: guess   4*pi*A/P^2
configs/compose.yaml:297: max_tolerated_helmet_rate: 0.10          # source: guess
configs/compose.yaml:302: shapes: ["dome", "ellipse", "rounded_cylinder", "arc"]   # source: guess
configs/compose.yaml:308: specular_highlight: true                 # source: guess
configs/compose.yaml:309: rim_shading: true                        # source: guess
configs/compose.yaml:317: base_multiplier_images: 1.0                # source: guess
configs/evaluation.yaml:25: min_compliance_precision: 0.80             # source: guess
configs/evaluation.yaml:59: slice_small_object_min_count: 1            # source: guess   (>=N boxes with area < 1024)
configs/evaluation.yaml:60: slice_crowded_min_instances: 8             # source: guess
configs/evaluation.yaml:61: slice_low_light_percentile: 25             # source: guess   (darkest N% by mean luminance)
configs/evaluation.yaml:67: warmup_iterations: 20                      # source: guess   (never time the first inference)
configs/evaluation.yaml:68: timed_iterations: 200                      # source: guess
configs/evaluation.yaml:75: dtype: "float16"                           # source: guess   (record it; FPS without dtype is meaningless)
configs/evaluation.yaml:83: samples_per_category: 12                      # source: guess
configs/filtering.yaml:13: sensitivity_alarm_points: 15                 # source: guess
configs/filtering.yaml:20: min_inside_ratio: 0.60                   # source: guess   (measured BEFORE clipping)
configs/filtering.yaml:28: helmet: 0.30                             # source: guess
configs/filtering.yaml:29: head:   0.30                             # source: guess
configs/filtering.yaml:30: person: 0.25                             # source: guess
configs/filtering.yaml:63: k_x: 0.35                                # source: guess   |dx| <= k_x
configs/filtering.yaml:64: dy_min: -0.10                            # source: guess   sitting slightly low over the brow
configs/filtering.yaml:65: dy_max: 0.75                             # source: guess   up to 0.75 head-heights above
configs/filtering.yaml:66: overlap_y_min: 0.15                      # source: guess   <- the contact test
configs/filtering.yaml:67: overlap_y_max: 0.75                      # source: guess   above this the helmet swallows the face
configs/filtering.yaml:68: r_w: [0.75, 1.60]                        # source: guess
configs/filtering.yaml:69: r_h: [0.25, 1.10]                        # source: guess
configs/filtering.yaml:70: min_iou_helmet_head: 0.05                # source: guess   (robust to odd aspect ratios)
configs/filtering.yaml:79: containment_threshold: 0.70              # source: guess   (when to consider it "inside")
configs/filtering.yaml:83: head_top_within_person: 0.45             # source: guess   (head must be in the upper 45%)
configs/filtering.yaml:90: poke_through_range: [0.02, 0.20]         # source: guess
configs/filtering.yaml:108: min_changed_pixel_ratio: 0.005           # source: guess
configs/filtering.yaml:147: grid_rows: 4                               # source: guess
configs/filtering.yaml:148: grid_cols: 4                               # source: guess
configs/filtering.yaml:156: accepted_vs_rejected_n: 12                 # source: guess
configs/training.yaml:50: lr_scheduler_type: "cosine"                # source: guess  (upstream uses MultiStepLR; cosine is
configs/training.yaml:61: per_device_train_batch_size: 16            # source: guess  (24 GB L4 has headroom; try 32)
configs/training.yaml:62: per_device_eval_batch_size: 8              # source: guess
configs/training.yaml:70: num_train_epochs_real_only: 50             # source: guess
configs/training.yaml:71: num_train_epochs_with_synthetic: 75        # source: guess
configs/training.yaml:75: precision: "bf16"                          # source: guess  (L4/A100/4090). Use fp16 on T4 (no bf16).
configs/training.yaml:76: dataloader_num_workers_colab: 2            # source: guess
configs/training.yaml:78: dataloader_prefetch_factor: 2              # source: guess
configs/training.yaml:86: eval_strategy: "epoch"                     # source: guess
configs/training.yaml:87: save_strategy: "epoch"                     # source: guess
configs/training.yaml:88: save_total_limit: 2                        # source: guess
configs/training.yaml:89: load_best_model_at_end: true               # source: guess
configs/training.yaml:90: metric_for_best_model: "eval_map"          # source: guess
configs/training.yaml:97: budget_alignment: "equal_steps"              # source: guess  -> revisit at M15 with real numbers
configs/training.yaml:100: primary: 1337                              # source: guess
configs/training.yaml:101: extra: [1338, 1339]                        # source: guess  only for real_only + best filtered arm
configs/training.yaml:114: bbox_min_area: 4                           # source: guess
configs/training.yaml:122: horizontal_flip: 0.5                     # source: guess
configs/training.yaml:124: perspective: 0.1                         # source: guess
configs/training.yaml:125: random_sized_bbox_safe_crop: 0.0         # source: guess
configs/training.yaml:128: random_brightness_contrast: 0.5          # source: guess
configs/training.yaml:129: hue_saturation_value: 0.1                # source: guess
configs/training.yaml:132: motion_blur: 0.1                         # source: guess
configs/training.yaml:133: blur_limit: 7                            # source: guess
configs/training.yaml:158: recommended_runtime: "L4"                  # source: guess
configs/training.yaml:162: compute_units_per_hour:                    # source: guess  (reported ranges, midpoints)
configs/training.yaml:171: estimated_total_compute_units_8_runs: 113  # source: guess
```
