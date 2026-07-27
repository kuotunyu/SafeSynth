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

Phase 1 has frozen the guarded split, built and visually verified a 7,255-item
SAM 2.1 cutout bank, implemented the deterministic compositor and filter, and
verified the 300-image filter ledger (196 pass / 104 reject; zero threshold
sensitivity alarms). The scale-up gate remains intentionally closed: the
group-disjoint H4 paste-artifact classifier reaches AUC 0.7964 against the
pre-registered maximum of 0.60. Full generation will not begin until that is
fixed. The repository owner approved the H6 hard-negative sheet at 0/64 true
helmets and selected the pre-registered Option A generative-inpainting route.
The pinned Apache-2.0 model is now downloaded and SHA-256 verified locally. Its
fixed 64-image identity pilot has zero changes outside the edit masks and zero
changes inside the protected cores, but `kuotunyu` rejected the visual gate
after observing more than three severe failures. Cell 02 exposed an invalid
input draft and cell 04 exposed a mislocalized edit region. Option A therefore
does not proceed to the one-shot H4 artifact gate, and no new H4 AUC has been
computed. Phase 2 remains blocked.

A separate four-case, Train-only Colab diagnostic compared the failed v1
reference-conditioning call against two pre-declared lower-strength variants
on an A100 40 GB. All 12 outputs preserved every pixel outside the edit mask,
but removing the reference changed masked RGB values by only 0.2260/255 on
average and lower strength showed no consistent visual improvement. No variant
was selected. A subsequent CPU-only preflight found reflected padding throughout
the source dataset, so v5 removes each detected mirrored border, resizes the
clean center with preserved aspect ratio, transforms every label and Pass-1
mask identically, and then reapplies the anchor guards. `kuotunyu` rejected the
fixed 64-input v5 sheet at 33/64 issues: isolated helmet/head cutouts frequently
looked composited, floated without a credible anatomical carrier, or produced
an implausible head/helmet structure. FLUX inference was not run because its
five-pixel boundary edit cannot repair an invalid core. Any successor must use
a coupled head + helmet + upper-body synthesis unit and pass a new zero-issue
CPU input sheet. A Train-only CPU audit found that 111/113 existing person
cutouts have a paired helmet/head annotation, spanning 76 source groups (90
people are at least 80 pixels high), but the visual and diversity gates rejected
all three coupled-person paste candidates. Geometry-only v6 visibly mismatched
scenes; scene-matched v6b still mismatched pose and rewrote faces/clothing; v7
preserved source position and core pixels but left only 22 strict donors across
19 groups and exhausted all 3,500 Train backgrounds at 63/64 drafts. Raising
the reuse cap after seeing that result was rejected, and the pool cannot support
the later 300-image H4 gate honestly. Whole-person pasting is therefore stopped.
A final Train-only v8 diagnostic edited an existing worker in place instead of
pasting one. `kuotunyu` passed cases 01 and 02, rejected case 03 for obvious
post-production appearance, and marked case 04 a severe failure. The fixed
four-case result is therefore rejected without a 64-case scale-up. No new H4
AUC has been computed, M13 remains closed, and Phase 2 has not started.

Two final regional-placement alternatives were then rejected on Train-only CPU
capacity evidence: v9 found 123 candidate placements across 17 source groups but
could not supply the required 64 strict cases, while helmet-anchored v9b found
only 92 strict placements across seven groups. The active successor is the
preregistered whole-image v10 route, which removes the empty-background capacity
ceiling. Its exact four prompts and seeds are frozen before output inspection.
The pinned Apache-2.0 Grounding DINO Tiny labeler is downloaded and verified,
but FLUX generation remains hard-locked until the labeler passes a group-disjoint
96-image Train-only calibration/audit and `kuotunyu` approves the exact v10
manifest. Validation/Test reads and new whole-image generations remain zero.

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
