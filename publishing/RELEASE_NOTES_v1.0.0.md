# SafeSynth v1.0.0

SafeSynth v1.0.0 freezes the complete hard-hat synthetic-data experiment:

- deterministic Train-only generation with per-image provenance;
- equal-sized filtered and unfiltered synthetic views;
- four-arm RT-DETRv2-R18 training and frozen-Test evaluation;
- an independently configured RF-DETR-Nano cross-check;
- image-level bootstrap confidence intervals and qualitative error analysis;
- a Gradio image/video demo; and
- reproducible Windows-native setup, CI, tests, and public release tooling.

The primary scientific result is negative: synthetic data did not produce a
robust, architecture-independent improvement. The pre-registered artifact gate
also failed at AUC 0.9053 against a maximum of 0.60. The release preserves that
result rather than selecting a favourable run.

Companion artifacts:

- Dataset: https://huggingface.co/datasets/steven0226/safesynth-hard-hat
- Model: https://huggingface.co/steven0226/safesynth-rtdetrv2-r18
