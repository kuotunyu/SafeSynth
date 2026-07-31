# SafeSynth — Targeted Synthetic Data for Hard-Hat Detection

Detecting whether construction workers wear their hard hats fails in exactly the
situations that matter: workers far from the camera, heads occluded by equipment,
crowded scenes, dusk and motion blur, and the rare-but-critical bare head. Real
datasets are thin precisely where the model is weak, and hand-labelling more of
those cases is expensive.

**This project generates the hard cases instead — with bounding-box labels
produced automatically by the generator, at zero annotation cost — and then
measures, under a controlled five-arm protocol, whether that actually helps.**

This is not a "train a detector on a public dataset" tutorial. The subject of
the experiment is the *data*, not the model:

- **Targeted, not bulk.** Six scenarios are synthesized on purpose — small/distant
  objects, partial occlusion, crowding, low light, motion blur, and bare heads —
  plus hard negatives (yellow machinery, round objects) that look like helmets and
  must *not* fire.
- **Labels are free and exact.** SAM 2.1 produces clean cutout masks; the
  compositor recomputes every bounding box from the visible mask after pasting,
  including for pre-existing real objects that a paste occludes.
- **A filter that earns its place.** Geometric and photometric rules (a worn
  helmet must actually touch a head, visible fraction and size ratios must be
  plausible, no floating helmets, no clipping artifacts) emit a *filtered* and an
  *unfiltered* version of the same generated pool, size-matched, so the ablation
  isolates filtering quality from data quantity.
- **Honest measurement.** Validation and test are real images only; the generator
  and the filter never read the test split; the split manifest is frozen with
  SHA-256 before a single synthetic image is generated.

## Status

**Phase 1 is complete, and it produced a negative result worth stating up front.**

The pipeline works: a guarded, frozen 70/15/15 group split; a 7,255-item SAM 2.1
cutout bank; a deterministic compositor that recomputes every bounding box from
the visible mask; and a geometric/quality filter whose 300-image ledger
reconciles exactly (196 pass / 104 reject, zero threshold-sensitivity alarms).

**But the paste artifacts are detectable.** A pre-registered gate (H4) asked
whether a small classifier could distinguish pasted patches from real object
patches, with a pre-registered maximum of **AUC 0.60**. On a group-disjoint,
class- and size-matched split of 2,028 patches, a HOG+HSV logistic regression
reaches **AUC 0.7964** (bootstrap 95% CI 0.7481-0.8392). The gate did not pass,
and this repository does not claim that it did.

Nine synthesis routes were tried against it and all failed: feather-parameter
search, multiband blending, Poisson blending (AUC 0.8869 - it washes out the
helmet hue that carries the class signal), same-class in-place replacement
(0.8312), an exact-source paired control (0.9049), FLUX.2 reference-conditioned
boundary inpainting, whole-person pasting, regional placement, and whole-image
generation. A feature-family diagnostic shows HOG-only at 0.7792 and HSV-only at
0.6816 - both resampling/boundary *and* photometric signals exceed the bar, so
no single-parameter fix exists. An 18-round supervised auto-labeler effort
(v6 through v23) built to support the last of those routes was also stopped: its
best checkpoint peaks at confidence 0.14 with true- and false-positive score
distributions almost completely overlapping, which is an undertrained model
rather than a labeling-quality problem.

**The decision (ADR-011) is to treat this as a finding and measure what it
costs.** Generation is capped at 1x real-Train size (2x is explicitly not done,
since that is exactly the large investment the gate existed to prevent), and the
four-arm comparison runs with the H4 AUC reported alongside every result. If
synthetic data still helps, the conclusion is that detectable paste artifacts do
not prevent transfer. If it does not, AUC 0.7964 is the mechanism. Both outcomes
are reportable; neither is hidden.

Validation and Test reads remain at zero throughout.

See [PLAN.md](PLAN.md) for milestones and [docs/](docs/) for the specifications
each milestone is implemented against.

## Dataset

[Hard Hat Workers](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
(Kaggle `andrewmvd/hard-hat-detection`, CC0 1.0): 5,000 images at 416x416, PASCAL
VOC boxes, three classes — `helmet` (18,966), `head` (5,785), `person` (751).

A documented caveat shapes the entire evaluation design: SHEL5K
([Sensors 2022](https://www.mdpi.com/1424-8220/22/6/2315)) re-annotated these
same 5,000 images and produced 75,570 labels against the original 25,502, and
describes the `person` class as poorly labelled. Roughly two thirds of true
objects carry no box here. Absolute AP on this benchmark is therefore depressed
for every class, which is why **every claim in this repository is relative** —
arm A versus arm B on one identical frozen test set — and never absolute. See
[docs/data_protocol.md](docs/data_protocol.md).

## Environment

Windows 11 native (no WSL), RTX 4090, Python 3.12, uv. Setup and the pinned
versions are in [docs/environment.md](docs/environment.md).

## License

Code is MIT. The source dataset is CC0 1.0; SAM 2.1 weights are Apache-2.0;
generated images are released as CC0 1.0 to match their source. See
[LICENSE](LICENSE).
