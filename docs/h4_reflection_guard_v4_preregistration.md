# H4 reflected-padding guard v4 pre-registration

Status at registration: the CPU-only guarded v3 input sheet was stopped before
`kuotunyu` review because a development sanity check found a low-texture,
near-exact right-edge reflection in canonical cell 64
(`hard_hat_workers2107.png`). No FLUX.2 output or H4 score was computed for v3.
The original failed identity gate and H4 AUC 0.7964 remain binding.

## Why v3 was insufficient

The v3 detector required pair correlation of at least 0.995. Cell 64's
left/right reflected borders had maximum grayscale MAE 0.9756/255 and minimum
texture standard deviation 6.2489, but correlation 0.9790. Its pale background
made correlation unstable even though the duplicated partial person at the
right edge was visually unambiguous.

A Train-wide CPU sweep fixed before a new sheet showed:

- v3 (`MAE <= 2.0`, correlation `>= 0.995`): 3,239/3,500 backgrounds rejected;
- v4 (`MAE <= 2.0`, correlation `>= 0.97`): 3,274/3,500 backgrounds rejected;
- cell 64 is rejected only by the v4 setting.

Only the correlation floor changes. The pad search, MAE floor, minimum texture,
annotation clearance, SAM2 QC, and anchor clearance remain unchanged.

## Frozen v4 rule and untouched input sheet

For each image axis, the existing detector searches equal opposite padding
widths from 16 pixels through 31% of that axis and chooses the lowest
16-pixel reflection-seam error. The whole background is rejected if either axis
has maximum opposite-pair MAE no greater than 2.0/255, minimum pair correlation
at least 0.97, and minimum texture standard deviation at least 5.0.

- Architecture: `guarded_context_replacement_v4`
- Root seed: `20260730`
- Inputs: exactly 64 distinct eligible Train backgrounds
- Input preflight: zero invalid drafts and zero misplaced cyan boxes
- Model inference: forbidden until `kuotunyu` passes the input sheet
- Output identity gate: zero label mismatches, at most 3 severe failures, and
  both pixel-exact invariants equal zero
- H4 remains uncomputed until both human gates pass

The v3 seed `20260729`, v2 seed `20260728`, and original failed seed `20260727`
are historical evidence only and may not be reused for v4.

## Historical outcome

The v4 input sheet was generated with no model inference, then stopped during
development sanity review before `kuotunyu` approval. Cells 7 and 54 still
showed reflected borders. A Train-wide feasibility check found that rejection
alone cannot retain 64 distinct safe backgrounds, so v5 supersedes rejection
with deterministic reflected-border normalization.
