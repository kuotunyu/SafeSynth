# H4 reflected-padding guard v3 pre-registration

Status at registration: the CPU-only guarded v2 input sheet was stopped before
`kuotunyu` review because a development sanity check found obvious mirrored
padding in at least canonical cells 11 and 25. No FLUX.2 output or H4 score was
computed for v2. The original failed identity gate and H4 AUC 0.7964 remain
binding.

## Why v2 was insufficient

The annotation-only guard correctly removed headlike boxes at the image frame,
including the two previously reported failures. However, reflected padding can
contain duplicated people or equipment without a corresponding edge
annotation. The v2 sheet therefore still contained invalid drafts such as
`hard_hat_workers1818.png` and `hard_hat_workers4287.png`.

A Train-wide CPU diagnostic also showed that reflection padding is an upstream
dataset-wide preprocessing pattern, not a rare corruption. An aggressive
detector would reject 3,424/3,500 Train backgrounds and create a severe
distribution shift. The v3 detector is deliberately high precision: it rejects
only borders whose opposite padded regions are near-exact pixel reflections.
The existing annotation and anchor guards remain active.

## Frozen high-confidence reflection rule

For top/bottom and left/right independently:

1. convert the RGB image to grayscale;
2. downsample only the orthogonal axis to 64 samples;
3. search equal opposite padding widths from 16 pixels through 31% of the image
   axis;
4. choose the width with the lowest 16-pixel reflection-seam error;
5. classify that axis as reflected only when both opposite border pairs have:
   maximum MAE no greater than 2.0/255, minimum Pearson correlation at least
   0.995, and minimum texture standard deviation at least 5.0.

If either axis passes all three conditions, the whole background is rejected.
The thresholds live in `configs/compose.yaml` and are frozen before the v3
coverage audit or input sheet.

The annotation rules from v2 remain unchanged:

- every background headlike box has at least 4 pixels of edge clearance;
- the selected anchor passes existing SAM2 QC;
- selected-anchor clearance is at least
  `max(8 px, ceil(0.10 × max(box width, box height)))`.

The complete guarded pool is computed before sampling, so the 64 inputs are
drawn without replacement from eligible backgrounds rather than relying on
repeated random failures.

## New untouched input sheet and pilot

- Architecture: `guarded_context_replacement_v3`
- Root seed: `20260729`
- Inputs: exactly 64 distinct eligible Train backgrounds
- Input preflight: zero invalid drafts and zero misplaced cyan boxes
- Model call: unchanged v1 FLUX.2 settings; the A100 diagnostic selected no
  alternative call
- Output identity gate: zero label mismatches, at most 3 severe failures, and
  both pixel-exact invariants equal zero
- H4 remains uncomputed until `kuotunyu` passes both input and output reviews

The v2 root seed `20260728` and original failed seed `20260727` are historical
evidence only and may not be reused for v3.
