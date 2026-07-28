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
