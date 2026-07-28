# Supervised labeler v10 pre-registration

v9 passed every numeric gate, but `kuotunyu` found two false positives and
two missed-helmet cells in the exact 48-case owner review. A revealed
Train-only diagnosis showed that raising the threshold enough to remove the
two reported false positives also loses three additional true positives.
Lowering it enough to recover the reported misses creates 1,675 false
positives. A global threshold is therefore not a complete repair.

## Frozen v10 change

v10 changes one model-facing factor: source images with deterministic mirrored
border padding are normalized before training and evaluation. The guard and
normalizer are the already-tested production implementation in
`src.synthetic.compose`. Each detected mirrored side is removed, the clean
center is resized with preserved aspect ratio and center-cropped back to the
source canvas, and helmet boxes receive the identical geometric transform.

This change is supported by the revealed v9 evidence:

- all four owner-failure images trigger the reflection guard;
- the false-positive centers in cells 06 and 12 are outside the clean crop;
- the misses in cells 11 and 37 coexist with reflected object duplicates that
  split confidence between the real center and mirrored borders.

Everything else remains frozen from v9: the pinned R101 base model, optimizer,
12 epochs, batch size 8, deterministic weighted sampler, seed, calibration
grid and precision floor, geometry filter, numeric gates, and zero-problem
owner-review rule.

## Data boundary

All v1-v9 calibration and audit source groups become v10 calibration. A
deterministic group-disjoint Train-only split with seed 20260905 selects 48
new audit images stratified by helmet size. The split must be frozen before
any normalization preflight reads pixels. The preflight may read training and
already-consumed calibration pixels, but the new audit pixels remain sealed.
Validation and Test reads remain zero, and whole-image generation remains
locked.

After a one-batch GPU smoke and the full 12-epoch run, the new 48-image audit
may be read exactly once. A numeric pass still requires `kuotunyu` to review
all exact-box pages. Any reported problem rejects v10.

## Frozen split outcome

The seed-20260905 split contains 2,893 training images, 528 already-consumed
calibration images, and 48 new sealed audit images. All source groups are
disjoint. Its manifest SHA-256 is
`9b36cedc3c159614f668b5fafaa12ad33f36d8ff84d62aba69337e1097aa4d6b`.
Validation/Test reads remain zero. The reflection-normalization preflight may
now inspect only the training and calibration partitions; the new audit
pixels remain sealed.

## Frozen CPU normalization preflight

The CPU preflight read the 2,893 training and 528 already-consumed calibration
images, while reading zero new-audit, Validation, or Test pixels. Reflection
normalization was applied to 2,875 training images and 525 calibration images.
It transformed 10,775 source training helmet boxes into 7,407 clean-canvas
boxes and 2,157 calibration boxes into 1,433. The removed boxes were outside
the clean/cropped canvas; all retained boxes have valid positive geometry.
Seventeen training images and four calibration images became valid empty
examples after their mirrored-only helmet labels were removed.

All four revealed v9 problem images received the preregistered crop, including
top/bottom removal for image IDs 345, 1124, and 3569 and left/right removal
for image 1027. The R101 model files were rehashed, no GPU work ran, and
whole-image generation remained locked.

## Frozen GPU smoke outcome

The preregistered batch-8 R101 smoke completed with finite loss 287.36 and
9.316 GiB peak VRAM. It consumed 32 transformed helmet boxes from the first
training batch and read zero calibration, sealed-audit, Validation, or Test
images. Full v10 training may proceed.

## Frozen numeric outcome

The RTX 4090 run completed all 12 epochs in 40.10 minutes. The frozen
calibration rule selected epoch 2 at threshold 0.05. On the one-shot new
48-image sealed audit it produced 106 true positives, 24 false positives, and
16 false negatives:

- precision 0.8154;
- recall 0.8689;
- F1 0.8413;
- median matched IoU 0.8044.

All three numeric gates passed, although precision is close to the frozen 0.80
floor. The checkpoint SHA-256 is
`e987c97fa72f68a80520afa237c3d7b00ca9d27af10853b95ef154a68a7d35bb`.
Validation/Test reads and whole-image generation remained zero. Numeric
success does not open generation.

The exact 48-case evidence and three separated review pages were frozen in
`reports/supervised_labeler_v10_review_manifest.json`, manifest SHA-256
`81733517ac707e6ae92722e0cff5d6bacbb8cdd037d86f6c78bed1d89c3f4f61`.
`kuotunyu` must judge the magenta model boxes against the green dataset boxes;
any reported problem rejects v10.

## Frozen owner-review outcome

`kuotunyu` reviewed the exact 48 cases and rejected v10 on 2026-07-28.
Cells 06, 07, 10, 27, 39, 40, and 42 contain real helmets without magenta
boxes. Cells 29, 34, and 47 contain magenta boxes on background, faces, or
other non-helmet objects. The review also exposed dataset-GT defects: cell 31
labels a lower-right logo as a helmet, cells 41 and 42 omit visible helmets,
and cell 42 is missed by both GT and the model. The canonical review evidence
SHA-256 is
`e9e707692cc96ec26a46aa4595daf0ea5f83f287d775903674e6013df4394f92`.
Generation remains locked.

The revealed-audit diagnosis reproduced the frozen metrics, but those metrics
are no longer trustworthy as final evidence because cells 31, 41, and 42
contain owner-confirmed GT errors. Against the existing GT, four misses have
matching boxes below threshold and four correct high-score boxes are removed
by the frozen geometry filter. In particular, cells 06, 07, and 40 have
correct raw candidates with IoU 0.81–0.88 and score 0.14–0.20 that are removed
by the 0.40 relative-height cap. Cells 10, 27, and 39 include low-score
localizations. The three owner false-positive cells contain numerical false
positives up to score 0.0942, so a small threshold increase cannot repair the
mixture.

This consumed Train-only evidence may support a preregistered GT-repair and
geometry experiment, but it cannot retroactively pass v10.
