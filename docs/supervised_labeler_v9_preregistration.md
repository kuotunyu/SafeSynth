# Supervised labeler v9 pre-registration

v8 passed every numeric gate, but `kuotunyu` found four visible false-positive
boxes, missed helmets, and one severe partial-box localization failure in six
of 48 owner-review cells. The now-revealed v8 audit diagnosis also showed that
no global threshold can repair this mixture: threshold 0.083 removes the
highest-score owner false positive but drops revealed recall from 0.8677 to
0.0847.

## Frozen v9 change

v9 changes one model-facing factor: the pinned Apache-2.0 base checkpoint moves
from `PekingU/rtdetr_v2_r50vd` to `PekingU/rtdetr_v2_r101vd`, revision
`2c5dbbd2d4d8c8814827a3b42737ba1afce3cf2a`.

The deeper R101 backbone targets the three observed capabilities jointly:
background/face discrimination, confidence separation, and complete box
localization. Every other setting remains frozen from v8:

- 640-pixel processor size and pinned Transformers 5.14.1;
- initialization from the pinned COCO base checkpoint, not a prior SafeSynth
  checkpoint;
- deterministic Train-only weighted sampling;
- 12 epochs, batch size 8, optimizer, learning rates, warmup, and seed;
- v8 geometry limits and score-threshold grid;
- calibration precision floor and all numeric/human gates.

This single-factor design makes the result attributable. In particular, v9
does not tune its threshold or geometry against the revealed v8 failures.

## Data boundary

All v1-v8 calibration and audit source groups become v9 calibration. A
deterministic group-disjoint Train-only split with seed 20260829 selects 48
new audit images stratified by helmet size. Before training:

- the new audit pixels and metrics remain sealed;
- Validation and Test reads remain zero;
- whole-image generation remains locked.

The R101 batch-8 configuration must pass a one-batch GPU memory smoke test
before full training. After a numeric pass, `kuotunyu` must review 48 exact-box
cases; any reported problem rejects v9.

## Frozen split outcome

The seed-20260829 split contains 2,946 training images, 480 revealed
calibration images, and 48 new sealed audit images. All source groups are
disjoint. Its manifest SHA-256 is
`6e5a806bab1265cc71d09efcf6dfac0d699db6c74d81e7e71423913b1498ff59`.
Validation/Test reads remain zero.

The CPU-only preflight rehashed all three R101 model files and verified the
split without reading any image pixels. The frozen weighted sampler assigns
ordinary weight to 804 training images and weight 2.0 to 2,142 images.

The preregistered batch-8 GPU smoke test completed with finite loss 226.90 and
9.325 GiB peak VRAM. It read one training batch, zero sealed-audit images, and
zero Validation/Test images. Full v9 training may proceed.

## Frozen numeric outcome

The RTX 4090 run completed all 12 epochs in 34.73 minutes. The frozen
calibration rule selected epoch 2 at threshold 0.05. On the one-shot new
48-image sealed audit it produced 176 true positives, 13 false positives, and
18 false negatives:

- precision 0.9312;
- recall 0.9072;
- F1 0.9191;
- median matched IoU 0.8151.

All three numeric gates passed. Relative to v8, precision increased from
0.9213, recall increased from 0.8677, and F1 increased from 0.8937. The
checkpoint SHA-256 is
`fb99ce019f5b88bfbf6be6de34a053765bb9e3aeafac49e059eb69d93db7c22f`.
Validation/Test reads and whole-image generation remained zero. Numeric
success does not open generation; the exact 48-case owner review remains
mandatory.
