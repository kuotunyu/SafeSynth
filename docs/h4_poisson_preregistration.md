# H4 Poisson-blending spike pre-registration

Status at registration: the frozen feathered-alpha H4 gate fails at AUC
0.7964. Controlled alpha/multiband changes and context replacement also fail.
The feature-family diagnostic shows both HOG-only (0.7792) and HSV-only
(0.6816) shortcuts. No Poisson-blended run or result exists yet.

## Frozen method

- Generate exactly **300** images from the unchanged default scenario mixture,
  root seed **42**, and per-sample RNG derivation.
- Keep the same backgrounds, cutouts, transforms, placements, inpainting,
  Lab harmonization, noise matching, whole-image post-effects, annotation
  recomputation, filters, and lossless PNG output.
- Replace only the final feathered-alpha composite operation with OpenCV
  `seamlessClone(..., NORMAL_CLONE)`.
- The clone source is the already decontaminated and harmonized RGB patch.
- The clone mask is the transformed alpha thresholded at 128 and clipped by
  the existing frame/patch slices.
- Any OpenCV placement failure makes the spike infeasible. There is no silent
  alpha fallback.
- Run tag: `m11_h4_poisson_seed42`.

## Frozen evaluation

- H4 classifier seed: **161803**, giving a new stable source-group fold.
- Controls, 64 px HOG+HSV features, L2 C=1 classifier, bootstrap count, and
  class/fold/log-size matching are unchanged.
- Primary endpoint: AUC no greater than **0.60**.
- Report tag: `h4_poisson_gate`.

## Interpretation fixed before evaluation

- Passing makes Poisson the candidate default blend and permits a second-seed
  reproducibility confirmation before changing the registered compositor.
- Failure rejects Poisson as an H4 fix; the feathered-alpha default remains
  unchanged and M13 stays blocked.
- Washed-out helmet hue or clone failure is a method failure, not permission to
  mix methods or tune the held-out fold.
- No threshold widening, feature weakening, seed reselection, or post-result
  parameter tuning is allowed.
