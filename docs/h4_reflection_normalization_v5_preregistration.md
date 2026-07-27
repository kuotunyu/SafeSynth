# H4 reflected-padding normalization v5 pre-registration

Status at registration: the CPU-only v4 input sheet was stopped before
`kuotunyu` review after development sanity checks found additional reflected
padding in cells 7 and 54. No FLUX.2 output or H4 score was computed for v4.
The original failed identity gate and H4 AUC 0.7964 remain binding.

## Why rejection alone cannot supply the registered pilot

The Hard Hat Workers source frequently uses reflected padding to make
non-square photographs 416x416. A CPU sweep found that rejecting every
high-confidence one-sided reflection left fewer than 64 distinct Train
backgrounds with safe headlike anchors. Relaxing the detector repeatedly put
obvious duplicated people, equipment, or rebar back into the input sheet.

The v5 architecture therefore removes detected padding instead of accepting or
rejecting it:

1. evaluate top, bottom, left, and right independently;
2. search mirrored widths from 8 pixels through 31% of the corresponding axis;
3. accept a border match only when grayscale MAE is at most 3.0/255,
   correlation is at least 0.97, and border texture standard deviation is at
   least 5.0;
4. crop all detected reflected borders;
5. resize the remaining center with its aspect ratio preserved until it covers
   416x416, then center-crop to exactly 416x416;
6. apply the identical crop/resize transform to every annotation and Pass-1
   mask;
7. run the existing whole-background edge, SAM2 QC, and anchor-clearance guards
   on the transformed result.

Any transformed headlike label near a new frame edge still rejects the whole
background. The normalization is deterministic, CPU-only, and recorded per
sample.

## Untouched v5 input sheet

- Architecture: `guarded_context_replacement_v5`
- Root seed: `20260731`
- Inputs: exactly 64 distinct eligible Train backgrounds
- Input preflight: zero invalid drafts and zero misplaced cyan boxes
- Model inference: forbidden until `kuotunyu` passes the input sheet
- Output identity gate: zero label mismatches, at most 3 severe failures, and
  both pixel-exact invariants equal zero
- H4 remains uncomputed until both human gates pass

Seeds `20260730`, `20260729`, `20260728`, and `20260727` are failed historical
evidence and may not be reused for v5.

## CPU audit outcome

Before generating the v5 sheet, the frozen transform was applied in a
Train-only audit:

- 3,479/3,500 backgrounds required at least one reflected-border crop;
- 1,459 transformed backgrounds passed every post-transform input guard;
- 2,873 transformed anchors remained eligible;
- every previously identified v4, v3, v2, and original input failure was either
  normalized or rejected;
- Validation/Test images read: 0/0;
- model inference and H4 computation: no/no.

## Historical outcome

`kuotunyu` rejected the v5 CPU input sheet at 33/64 issues. The mirror
normalization worked, but the isolated object-paste architecture did not:
helmets floated without credible heads, faces looked composited, and many
helmet/head relationships remained structurally implausible. Because FLUX.2 was
registered to alter only the five-pixel boundary band, it could not repair
these invalid cores. No GPU inference or H4 computation was run. A successor
must operate on a coupled head + helmet + upper-body unit.
