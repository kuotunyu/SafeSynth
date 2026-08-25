# SafeSynth evidence index

This is a recruiter- and reviewer-oriented route through the current evidence. Read the
[privacy and responsible-use boundary](../docs/privacy_and_responsible_use.md) before
reusing images or model outputs.

## headline comparison

- [RT-DETRv2 four-arm frozen-Test results](detection_main_table.md)
- [RF-DETR-Nano four-arm replication](rfdetr_detection_main_table.md)
- [Curated headline figure](figures/headline.png)

## calibration and operating point

- [Validation-selected compliance operating point](compliance_operating_point.md)
- [Test recall across detector confidence](bare_head_recall_sweep.md)
- Per-arm validation sweeps: [`real_only`](compliance_operating_point_real_only.md),
  [`standard_aug`](compliance_operating_point_standard_aug.md),
  [`unfiltered_syn`](compliance_operating_point_unfiltered_syn.md), and
  [`filtered_syn`](compliance_operating_point_filtered_syn.md)

## error analysis

- [Four-way false-positive and false-negative analysis](error_analysis.md)
- [Machine-readable error evidence](error_analysis.json)
- [Hard-negative false positives](hard_negative_false_positives.md)

## synthetic composition

- [Final 14,000-candidate composition statistics](synthetic_stats.md)
- [Frozen threshold sensitivity](threshold_sensitivity.md)
- [Class and split distribution](class_distribution.md)

## reproducibility

- [Composition reproducibility audit](composition_reproducibility.json)
- [Cutout-bank reproducibility audit](bank_reproducibility.json)
- [Curated-figure manifest](figure_curation_manifest.json)
- [Colab release-package inventory](colab_package.json)

## historical diagnostics

The complete `h4_*` family, every `supervised_labeler_*` iteration, and every
`*_preflight*_failed*` file in this directory are retained here as historical diagnostic
evidence. They are not the first-stop current result. Useful entry points are the
[final H4 artifact-gate report](h4_artifact_gate_m13.md),
[supervised-labeler failure diagnosis](supervised_labeler_failure_diagnosis.md), and
[generative-model preflight](generative_model_preflight.md). Earlier partial composition
statistics such as [`m13_synthetic_stats.md`](m13_synthetic_stats.md) are also historical;
files remain immutable in place and are not renamed or deleted.
