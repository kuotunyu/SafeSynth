# H4 context-matched diagnostic pre-registration

Status at registration: the frozen H4 gate fails at AUC 0.7964. No result from
the diagnostic below has been computed yet.

## Question

Does the existing H4 classifier partly distinguish pasted headlike objects
because its real controls are not matched on nearby `person` context?

## Frozen method

- Input images and pasted instances: the unchanged 300-image
  `m11_h4_seed42` run.
- Features and classifier: unchanged 64 px HOG+HSV, standardized from the
  training partition, L2 logistic regression with C=1.
- New group-fold seed: **1729**; five stable SHA256 folds, fold 0 held out.
- A head or helmet has person context when its centre is horizontally inside a
  `person` box expanded by one headlike width and vertically between one
  headlike height above the person and 65% of the person's height plus one
  headlike height.
- Real controls are matched without replacement where possible by:
  1. class;
  2. source-group fold;
  3. the Boolean person-context state above;
  4. nearest log(pixel width, pixel height).
- `person` examples retain the original class + fold + size matching because
  the context predicate applies only to headlike objects.
- Bootstrap samples and all remaining thresholds stay unchanged.

## Interpretation fixed before evaluation

- The reference maximum remains **AUC 0.60**.
- This is a diagnostic and cannot reopen M13 by itself, even if AUC is at or
  below 0.60, because the method was designed after observing the original H4
  failure.
- A missing class/fold/context control pool is a feasibility failure, not
  permission to silently relax matching.
- If it remains above 0.60, the next implementation experiment is a
  context-anchored replacement composer evaluated on another untouched
  frozen-group fold.
- No feature weakening, threshold widening, fold reselection, or post-result
  parameter tuning is allowed.
