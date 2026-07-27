# H4 guarded v5 rejection analysis

- Reviewer: **kuotunyu**
- Result: **rejected**
- Input failures: **33/64 (51.56%)**
- Failed pasted classes: **32 helmet, 1 head**
- Model inference run: **no**
- H4 AUC computed: **no**

## Failure groups

| human-review description | cells | count |
|---|---|---:|
| face is visibly composited | 05, 08, 25, 33 | 4 |
| blurred/floating helmet without a credible visible head | 09, 10, 15, 16, 20, 21, 24, 26, 28, 34, 37, 45, 50 | 13 |
| helmet and head are present but the whole structure is implausible | 29, 30, 32, 36, 38, 41, 43, 44, 47, 57, 63 | 11 |
| appearance resembles a failed post-process | 51, 55, 59, 62, 64 | 5 |

## Structural cause

The cutout bank treats a helmet, head, or person as an independent object. The
v5 context-replacement path removes one original helmet/head mask and places a
same-class cutout into that box. It does not carry the coupled face, neck,
shoulders, pose, or occlusion relationship with the object. Hard Hat Workers
also has sparse `person` labels, so the compositor cannot reliably verify that
a pasted helmet has a credible anatomical carrier.

FLUX.2 is registered to edit only a five-pixel boundary band while preserving
the pasted core. It can harmonize an edge but cannot reconstruct a missing
head, correct the helmet-to-face pose, or replace an already implausible core.
Running the model on these drafts would therefore spend GPU time on inputs that
cannot pass the human gate.

## Decision

The guarded context-replacement architecture is stopped before GPU inference.
Any successor must synthesize and validate a coupled **head + helmet + upper
body** unit, not an isolated helmet/head cutout, and must again pass a
zero-issue CPU input sheet before model inference or H4.

## Successor feasibility

A Train-only CPU audit initially found 111/113 person cutouts with a paired
helmet/head annotation, spanning 76 source groups; 90 of those people are at
least 80 pixels high. Stricter truncation, pose, mask, and position gates reduced
that pool to 22 cutouts across 19 groups. Geometry-only v6 and scene-matched v6b
were rejected internally for visible scene, pose, and photometric mismatch. v7
preserved source position and core pixels, but an exhaustive search of all
3,500 Train backgrounds produced only 63/64 drafts under the frozen
three-uses-per-cutout cap. The coupled-person paste successor is therefore
capacity-infeasible for both the pilot and the later 300-image H4 gate.
