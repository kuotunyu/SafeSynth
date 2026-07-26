# H4 exact-source-pair diagnostic pre-registration

Status at registration: the frozen pooled-control H4 gate fails at AUC 0.7964.
Several blend and context methods also fail. No exact-source-pair classifier
result exists yet.

## Confound being tested

Every pasted positive comes from the SAM2-qualified cutout bank, while pooled
real controls currently come from all Train annotations. A classifier can
therefore exploit source-selection quality (visibility, sharpness, maskability)
instead of a paste artifact. Geometry matching does not remove that bias.

## Frozen method

- Input: the unchanged 300-image final feathered-alpha run
  `m11_h4_seed42`.
- For every pasted instance, use the raw Train image and bbox of that cutout's
  exact `source_annotation_id` as its real negative control.
- Preserve one real control per pasted occurrence, including when a cutout is
  reused; this keeps labels exactly balanced.
- Assign both members of a pair the cutout's frozen source-group key, so no
  source object can cross train/test.
- New five-fold seed: **57721**, with fold 0 held out.
- Crop context scale, 64 px resize, HOG+HSV features, standardization,
  L2 logistic C=1, bootstrap count, and AUC calculation remain unchanged.
- No class/size nearest-neighbour search is used: the exact source object is
  intentionally the control. Any scale/rotation/resampling difference is part
  of the synthesis artifact the gate should measure.
- Report tag: `h4_source_pair`.

## Interpretation fixed before evaluation

- The reference maximum remains **AUC 0.60**.
- Failure leaves H4 and M13 blocked.
- Passing would show that the pooled-control gate was materially confounded,
  but cannot replace it immediately because this control design was proposed
  after observing the failure. It requires one second pre-registered
  confirmation on another untouched fold before any gate decision changes.
- No seed reselection, threshold widening, feature weakening, or result-driven
  control mixing is allowed.

## Registered outcome

The result is **FAIL**: AUC **0.9049**, bootstrap 95% CI 0.8788–0.9289.
Using every cutout's exact raw source object makes the shortcut stronger, not
weaker. Source-bank selection bias therefore does not explain away the frozen
H4 failure; M13 remains blocked. See
[`reports/h4_source_pair.md`](../reports/h4_source_pair.md).
