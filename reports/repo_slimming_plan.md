# Repo slimming plan — the file list to review before `git filter-repo`

> Regenerate with `uv run python -m scripts.plan_repo_slimming`.
> It reads `git ls-files` and every tracked `.md`, and changes nothing.

## Why

| | bytes across ALL history | share |
|---|---:|---:|
| `reports/figures/` | 629.7 MB | 94% |
| `everything else` | 42.3 MB | 6% |

A clone downloads history, not the working tree, so deleting these at HEAD
would change nothing. Rewriting history is the only thing that shrinks it,
and this repo has never been pushed - `git remote -v` is empty - so doing it
now costs nothing and doing it later means force-pushing over published
history.

## KEEP — a document links to these

148 files, 403.2 MB.

- `reports/figures/class_distribution.png` (0.14 MB)
- `reports/figures/compliance_sweep.png` (0.05 MB)
- `reports/figures/compliance_sweep_filtered_syn.png` (0.06 MB)
- `reports/figures/compliance_sweep_real_only.png` (0.05 MB)
- `reports/figures/compliance_sweep_standard_aug.png` (0.05 MB)
- `reports/figures/compliance_sweep_unfiltered_syn.png` (0.05 MB)
- `reports/figures/demo_examples/demo_hard_hat_workers2261.png` (0.33 MB)
- `reports/figures/demo_examples/demo_hard_hat_workers245.png` (0.28 MB)
- `reports/figures/error_analysis/both_wrong.png` (0.13 MB)
- `reports/figures/error_analysis/fixed_false_negative.png` (0.24 MB)
- `reports/figures/error_analysis/fixed_false_positive.png` (0.22 MB)
- `reports/figures/error_analysis/new_false_positive.png` (0.29 MB)
- `reports/figures/exposure_curves.png` (0.17 MB)
- `reports/figures/filter_pass_reject_grid.png` (1.79 MB)
- `reports/figures/flux2_v2_diagnostic_detail.png` (0.42 MB)
- `reports/figures/grounded_labeler_audit.png` (1.78 MB)
- `reports/figures/h1_aspect_ratios.png` (0.06 MB)
- `reports/figures/h1_head_contact_sheet.png` (0.79 MB)
- `reports/figures/h1_helmet_contact_sheet.png` (0.90 MB)
- `reports/figures/h2_sam2_larger.png` (1.42 MB)
- `reports/figures/h2_sam2_medium.png` (1.41 MB)
- `reports/figures/h2_sam2_very_small.png` (1.23 MB)
- `reports/figures/h3_clip_largest_groups.png` (2.06 MB)
- `reports/figures/h4_ablation_no_hard_negative_roc.png` (0.05 MB)
- `reports/figures/h4_artifact_gate_m13_roc.png` (0.05 MB)
- `reports/figures/h4_artifact_roc.png` (0.05 MB)
- `reports/figures/h4_generative_identity_pilot.png` (9.87 MB)
- `reports/figures/h4_generative_identity_pilot_detail.png` (8.21 MB)
- `reports/figures/h4_guarded_input_preflight.png` (8.15 MB)
- `reports/figures/h4_paired_person_input_preflight_seed20260802.png` (10.59 MB)
- `reports/figures/h4_ranked_patches.png` (0.25 MB)
- `reports/figures/h5_placement_priors.png` (0.51 MB)
- `reports/figures/h6_hard_negative_candidates.png` (2.00 MB)
- `reports/figures/hard_negative_bank_grid.png` (0.17 MB)
- `reports/figures/hard_negative_test_regions.png` (0.51 MB)
- `reports/figures/headline.png` (0.11 MB)
- `reports/figures/procedural_hard_negative_grid.png` (0.80 MB)
- `reports/figures/review/k11_hard_negative_before.png` (2.43 MB)
- `reports/figures/review/k12_blackout_evidence.png` (0.11 MB)
- `reports/figures/review/loose_helmet_question.png` (2.12 MB)
- `reports/figures/review/preview_crowded_p1.png` (1.91 MB)
- `reports/figures/review/preview_crowded_p2.png` (1.96 MB)
- `reports/figures/review/preview_hard_negative_p1.png` (2.30 MB)
- `reports/figures/review/preview_hard_negative_p2.png` (1.94 MB)
- `reports/figures/review/preview_head_no_helmet_p1.png` (2.08 MB)
- `reports/figures/review/preview_head_no_helmet_p2.png` (2.04 MB)
- `reports/figures/review/preview_low_light_blur_p1.png` (2.23 MB)
- `reports/figures/review/preview_low_light_blur_p2.png` (2.07 MB)
- `reports/figures/review/preview_partial_occlusion_p1.png` (1.93 MB)
- `reports/figures/review/preview_partial_occlusion_p2.png` (1.96 MB)
- `reports/figures/review/preview_small_distant_p1.png` (1.79 MB)
- `reports/figures/review/preview_small_distant_p2.png` (1.48 MB)
- `reports/figures/supervised_labeler_v12_gt_review/page_01.png` (4.62 MB)
- `reports/figures/supervised_labeler_v12_gt_review/page_02.png` (4.55 MB)
- `reports/figures/supervised_labeler_v12_gt_review/page_03.png` (4.32 MB)
- `reports/figures/supervised_labeler_v12_gt_review/page_04.png` (4.62 MB)
- `reports/figures/supervised_labeler_v12_model_audit/page_01.png` (2.32 MB)
- `reports/figures/supervised_labeler_v12_model_audit/page_02.png` (2.27 MB)
- `reports/figures/supervised_labeler_v12_model_audit/page_03.png` (2.18 MB)
- `reports/figures/supervised_labeler_v13_audit.png` (5.39 MB)
- `reports/figures/supervised_labeler_v13_gt_review/page_01.png` (4.38 MB)
- `reports/figures/supervised_labeler_v13_gt_review/page_02.png` (4.52 MB)
- `reports/figures/supervised_labeler_v13_gt_review/page_03.png` (4.70 MB)
- `reports/figures/supervised_labeler_v13_gt_review/page_04.png` (4.45 MB)
- `reports/figures/supervised_labeler_v13_model_review_page_01.png` (0.88 MB)
- `reports/figures/supervised_labeler_v13_model_review_page_02.png` (0.92 MB)
- `reports/figures/supervised_labeler_v13_model_review_page_03.png` (0.88 MB)
- `reports/figures/supervised_labeler_v14_audit.png` (5.58 MB)
- `reports/figures/supervised_labeler_v14_gt_review/page_01.png` (4.51 MB)
- `reports/figures/supervised_labeler_v14_gt_review/page_02.png` (4.56 MB)
- `reports/figures/supervised_labeler_v14_gt_review/page_03.png` (4.62 MB)
- `reports/figures/supervised_labeler_v14_gt_review/page_04.png` (4.49 MB)
- `reports/figures/supervised_labeler_v14_model_review_page_01.png` (0.89 MB)
- `reports/figures/supervised_labeler_v14_model_review_page_02.png` (0.92 MB)
- `reports/figures/supervised_labeler_v14_model_review_page_03.png` (0.96 MB)
- `reports/figures/supervised_labeler_v15_audit.png` (5.36 MB)
- `reports/figures/supervised_labeler_v15_gt_review/page_01.png` (4.42 MB)
- `reports/figures/supervised_labeler_v15_gt_review/page_02.png` (4.18 MB)
- `reports/figures/supervised_labeler_v15_gt_review/page_03.png` (4.43 MB)
- `reports/figures/supervised_labeler_v15_gt_review/page_04.png` (4.39 MB)
- `reports/figures/supervised_labeler_v15_model_review_page_01.png` (0.91 MB)
- `reports/figures/supervised_labeler_v15_model_review_page_02.png` (0.86 MB)
- `reports/figures/supervised_labeler_v15_model_review_page_03.png` (0.93 MB)
- `reports/figures/supervised_labeler_v16_audit.png` (5.57 MB)
- `reports/figures/supervised_labeler_v16_gt_review/page_01.png` (4.88 MB)
- `reports/figures/supervised_labeler_v16_gt_review/page_02.png` (4.46 MB)
- `reports/figures/supervised_labeler_v16_gt_review/page_03.png` (4.47 MB)
- `reports/figures/supervised_labeler_v16_gt_review/page_04.png` (4.49 MB)
- `reports/figures/supervised_labeler_v17_audit.png` (5.38 MB)
- `reports/figures/supervised_labeler_v17_gt_review/page_01.png` (4.05 MB)
- `reports/figures/supervised_labeler_v17_gt_review/page_02.png` (4.39 MB)
- `reports/figures/supervised_labeler_v17_gt_review/page_03.png` (4.79 MB)
- `reports/figures/supervised_labeler_v17_gt_review/page_04.png` (4.63 MB)
- `reports/figures/supervised_labeler_v17_model_review/review_page_01.png` (0.83 MB)
- `reports/figures/supervised_labeler_v17_model_review/review_page_02.png` (0.91 MB)
- `reports/figures/supervised_labeler_v17_model_review/review_page_03.png` (0.99 MB)
- `reports/figures/supervised_labeler_v18_audit.png` (5.41 MB)
- `reports/figures/supervised_labeler_v18_gt_review/page_01.png` (4.40 MB)
- `reports/figures/supervised_labeler_v18_gt_review/page_02.png` (4.28 MB)
- `reports/figures/supervised_labeler_v18_gt_review/page_03.png` (4.75 MB)
- `reports/figures/supervised_labeler_v18_gt_review/page_04.png` (4.55 MB)
- `reports/figures/supervised_labeler_v18_model_review/review_page_01.png` (0.88 MB)
- `reports/figures/supervised_labeler_v18_model_review/review_page_02.png` (0.88 MB)
- `reports/figures/supervised_labeler_v18_model_review/review_page_03.png` (0.97 MB)
- `reports/figures/supervised_labeler_v19_audit.png` (5.42 MB)
- `reports/figures/supervised_labeler_v19_gt_review/page_01.png` (4.70 MB)
- `reports/figures/supervised_labeler_v19_gt_review/page_02.png` (4.48 MB)
- `reports/figures/supervised_labeler_v19_gt_review/page_03.png` (4.60 MB)
- `reports/figures/supervised_labeler_v19_gt_review/page_04.png` (4.27 MB)
- `reports/figures/supervised_labeler_v19_model_review/review_page_01.png` (0.94 MB)
- `reports/figures/supervised_labeler_v19_model_review/review_page_02.png` (0.90 MB)
- `reports/figures/supervised_labeler_v19_model_review/review_page_03.png` (0.89 MB)
- `reports/figures/supervised_labeler_v20_audit.png` (5.44 MB)
- `reports/figures/supervised_labeler_v20_gt_review/page_01.png` (4.29 MB)
- `reports/figures/supervised_labeler_v20_gt_review/page_02.png` (4.40 MB)
- `reports/figures/supervised_labeler_v20_gt_review/page_03.png` (4.35 MB)
- `reports/figures/supervised_labeler_v20_gt_review/page_04.png` (4.49 MB)
- `reports/figures/supervised_labeler_v21_audit.png` (5.38 MB)
- `reports/figures/supervised_labeler_v21_gt_review/page_01.png` (4.50 MB)
- `reports/figures/supervised_labeler_v21_gt_review/page_02.png` (4.55 MB)
- `reports/figures/supervised_labeler_v21_gt_review/page_03.png` (4.44 MB)
- `reports/figures/supervised_labeler_v21_gt_review/page_04.png` (4.39 MB)
- `reports/figures/supervised_labeler_v22_audit.png` (5.36 MB)
- `reports/figures/supervised_labeler_v22_gt_review/page_01.png` (4.45 MB)
- `reports/figures/supervised_labeler_v22_gt_review/page_02.png` (4.60 MB)
- `reports/figures/supervised_labeler_v22_gt_review/page_03.png` (4.46 MB)
- `reports/figures/supervised_labeler_v22_gt_review/page_04.png` (4.52 MB)
- `reports/figures/supervised_labeler_v22_model_review/review_page_01.png` (0.89 MB)
- `reports/figures/supervised_labeler_v22_model_review/review_page_02.png` (0.89 MB)
- `reports/figures/supervised_labeler_v22_model_review/review_page_03.png` (0.94 MB)
- `reports/figures/supervised_labeler_v23_audit.png` (5.51 MB)
- `reports/figures/supervised_labeler_v23_gt_review/page_01.png` (4.55 MB)
- `reports/figures/supervised_labeler_v23_gt_review/page_02.png` (4.48 MB)
- `reports/figures/supervised_labeler_v23_gt_review/page_03.png` (4.63 MB)
- `reports/figures/supervised_labeler_v23_gt_review/page_04.png` (4.60 MB)
- `reports/figures/supervised_labeler_v23_model_review/review_page_01.png` (0.91 MB)
- `reports/figures/supervised_labeler_v23_model_review/review_page_02.png` (0.92 MB)
- `reports/figures/supervised_labeler_v23_model_review/review_page_03.png` (0.93 MB)
- `reports/figures/supervised_labeler_v6_audit.png` (5.57 MB)
- `reports/figures/supervised_labeler_v6_audit_page_01.png` (1.90 MB)
- `reports/figures/supervised_labeler_v6_audit_page_02.png` (1.85 MB)
- `reports/figures/supervised_labeler_v6_audit_page_03.png` (1.83 MB)
- `reports/figures/supervised_labeler_v6_audit_separated_page_01.png` (0.97 MB)
- `reports/figures/supervised_labeler_v6_audit_separated_page_02.png` (0.96 MB)
- `reports/figures/supervised_labeler_v6_audit_separated_page_03.png` (0.94 MB)
- `reports/figures/training_curves.png` (0.16 MB)
- `reports/figures/whole_person_edit_diagnostic_v8.png` (0.56 MB)
- `reports/figures/whole_person_edit_preflight_v8.png` (0.83 MB)

## DROP — no document links to these

2 files, 0.0 MB. Their only
mention anywhere is inside the script that generated them, which is not a
reference a reader can follow.

**The evidence they represent is not discarded** — the worklog records which
labeler iterations and synthesis routes were tried and what each returned.
That is the form a reader can search.

- `reports/figures/README.md` (0.00 MB)
- `reports/figures/review/README.md` (0.00 MB)

## The command (YOURS to run — CLAUDE.md reserves history rewrites)

```bash
git filter-repo --path reports/figures/ --invert-paths --force
```

That removes the whole folder from history, keepers included. Re-add them
afterwards as one fresh commit — they survive the rewrite because they are
still in the working tree; `filter-repo` edits history, not your files:

```bash
git add reports/figures/ && git commit -m "docs: restore the figures documents reference"
```

## Verify afterwards

```bash
git count-objects -vH
uv run pytest -q
uv run python scripts/verify_readme.py
```

`size-pack` should fall sharply. `verify_readme` is the real check: it fails
if any README figure link is dead.
