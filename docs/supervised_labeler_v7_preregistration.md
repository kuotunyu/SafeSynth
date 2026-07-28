# Supervised labeler v7 pre-registration

The v6 numeric audit passed, but `kuotunyu` rejected its 48-case human review.
The consumed evidence records nine problem cells:

- background false positives in 04, 06, 07, 23, 38, 43, and 45;
- one missed helmet in 04;
- merged adjacent helmets in 13 and 27.

Those v6 images are revealed evidence. They may motivate v7, but they cannot
pass any v7 gate. All v1-v6 calibration and audit source groups become v7
calibration history.

## Frozen v7 changes

v7 keeps the pinned RT-DETRv2 R50 base checkpoint, training seed, epoch count,
size filters, threshold grid, calibration precision floor, and numeric audit
gates. It makes two changes before selecting a new audit:

1. Empty Train images and Train images containing a close helmet pair each
   receive weight 2.0 in deterministic replacement sampling. A close pair has
   center distance divided by the mean square-root box area at most 1.0.
   Overlapping categories use the maximum weight rather than multiplying.
2. Predictions outside aspect ratio 0.25–4.0 are rejected. The interval is
   rounded outward beyond the previously frozen Train helmet p1–p99 interval
   0.3913–3.7143 and targets the elongated background predictions revealed by
   v6.

Only Train annotations determine sampling weights. The next 48-image audit is
selected deterministically by source group and helmet-size quartile using split
seed 20260815. Its pixels and metrics remain unread until training and
calibration have finished.

Validation and Test pixels remain unread. Whole-image generation remains
locked. Even a numeric v7 pass cannot unlock it: `kuotunyu` must review all
48 cases, and approval requires zero reported problem cells.

## Frozen split outcome

The method was committed as `9a233d9` before split generation. The subsequent
metadata-only freeze produced manifest
`5c391442b6a873e8d8642b64ff4a3d154381f2f8731ce2efad4426fe7ac292ef`:

- 3,045 training images from 2,936 source groups, with 11,364 helmet boxes;
- 384 revealed v1-v6 images assigned to calibration history;
- 48 new group-disjoint audit images kept sealed;
- zero Validation/Test pixels read and zero new-audit pixels or metrics read.

The audit IDs are stored for the future one-shot evaluation but are deliberately
not printed in the freeze summary or this document.
