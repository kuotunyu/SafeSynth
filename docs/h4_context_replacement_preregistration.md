# H4 context-replacement spike pre-registration

Status at registration: the original H4 gate fails at AUC 0.7964. The
context-matched reanalysis registered in `ddec0c7` was infeasible and produced
no AUC. No context-replacement images or classifier results exist yet.

## Hypothesis

Replacing a real head or helmet in place should preserve its genuine scene
context and remove the strongest floating/isolated-object shortcut. This tests
a composition architecture, not another edge-feather parameter.

## Frozen generation method

- Generate exactly **300** lossless PNG composites with root seed **271828**.
- Each image replaces exactly one Train-split `head` or `helmet` annotation
  whose Pass-1 SAM2 mask passed QC.
- Remove that annotation with the existing mask inpainting parameters:
  dilation 3 px and radius 3 px.
- Paste a different Train cutout of the **same class**, excluding the same
  source image and frozen group.
- Candidate selection is uniform among the **8** nearest eligible cutouts by
  Euclidean distance in log(source bbox width, source bbox height).
- Scale the cutout by the geometric mean of target/source width and height,
  capped by the existing class maximum (1.15).
- Centre it on the removed annotation; rotation is fixed to 0 degrees and the
  existing horizontal-flip probability remains 0.5.
- Use the unchanged edge decontamination, feathered alpha, Lab harmonization,
  noise matching, whole-image post-effect probability 0.20, bbox recomputation,
  filtering, provenance, and Test blocklist checks.
- A failed placement retries; a background with no eligible replacement anchor
  is replaced under the existing bounded background-retry rule.

## Frozen evaluation

- H4 classifier seed: **314159**, giving a new stable source-group fold.
- Controls: unchanged same class + same fold + nearest log(pixel width, height).
- Features/classifier: unchanged 64 px HOG+HSV and L2 C=1 logistic regression.
- Primary endpoint: AUC no greater than **0.60**, with the existing bootstrap
  interval reported.
- Run and report names:
  - `m11_h4_context_replace`
  - `reports/h4_context_replacement.{json,md}`

## Interpretation fixed before evaluation

- Failure leaves the original H4 block unchanged and rules out same-class
  in-place replacement as a sufficient fix.
- Passing would justify integrating replacement anchors into the scenario
  composer, but it would not by itself open M13 because a replacement-only run
  does not cover the registered target scenarios.
- No seed reselection, classifier weakening, threshold widening, or post-result
  parameter tuning is allowed.
