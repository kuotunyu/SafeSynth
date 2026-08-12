# SafeSynth v1.0.1

SafeSynth v1.0.1 is a presentation and reproducibility release built on the
unchanged v1.0.0 scientific artifacts. It does not retrain the models, alter the
frozen splits, change the operating point, or replace either Hugging Face
artifact.

Highlights:

- adds the evidence-first Gradio image and video experience;
- makes the public `real_only` checkpoint, frozen threshold and runtime
  conditions explicit in the demo;
- improves Desktop and Mobile layouts without reducing readable text sizes;
- restructures the README around the headline result and real demo evidence;
- replaces the oversized four-arm diagram with a compact comparison table;
- restores the complete `Installation & Reproduction` section in English;
- aligns Mermaid colors with the SafeSynth Morandi evidence palette;
- removes private agent plans and maintainer-only handoff files from the public
  repository; and
- retains Windows-native CI, numerical README verification and licence gates.

Scientific result:

- RT-DETRv2-R18 still selects `real_only` as the primary checkpoint.
- RF-DETR-Nano still shows a positive synthetic-data point estimate with
  overlapping confidence intervals.
- The pre-registered H4 artifact gate remains failed at AUC 0.9053 against a
  maximum of 0.60.

Companion artifacts are unchanged from v1.0.0:

- Dataset: https://huggingface.co/datasets/steven0226/safesynth-hard-hat
- Model: https://huggingface.co/steven0226/safesynth-rtdetrv2-r18
- Previous scientific release: https://github.com/kuotunyu/SafeSynth/releases/tag/v1.0.0
