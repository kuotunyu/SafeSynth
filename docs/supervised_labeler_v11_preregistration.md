# Supervised labeler v11 pre-registration

v10 passed the frozen numeric audit, but `kuotunyu` rejected its exact
48-image review. Six cells missed real helmets, three cells placed model boxes
on non-helmet content, and three cells exposed defects in the source dataset
ground truth. Generation therefore remains locked.

## Frozen v11 change

v11 changes one model-facing factor: the prediction geometry filter. The
revealed Train-only diagnosis found correct high-score raw boxes in v10 cells
06, 07, 10, and 40 that were removed by the frozen height, area, or minimum
aspect limits. Five candidate policies were evaluated against already revealed
Train-only history only. The selected `edge_large_060` policy is the smallest
candidate that recovers all four boxes and stays within 0.003 F1 of the best
tested candidate:

- maximum relative area: 0.14 to 0.15;
- maximum relative height: 0.40 to 0.60;
- minimum aspect ratio: 0.25 to 0.20;
- maximum aspect ratio: unchanged at 4.0.

The larger 0.18-area / 0.70-height candidate had slightly higher revealed
history F1 but admitted more false positives, so it was not selected. The
pinned R101 base model, reflected-padding normalization, optimizer, 12 epochs,
batch size, sampler, seeds, threshold grid, precision floor, audit gates, and
zero-problem owner-review gate remain frozen.

## Ground-truth quarantine

Owner review confirmed that Train image 3060 labels a logo as a helmet, image
4155 omits a visible helmet, and image 4364 omits another visible helmet. Those
three images remain preserved as evidence, and their complete source groups
remain excluded from training and the new audit. Their pixels and annotations
are excluded from v11 calibration metrics. This prevents known-bad labels from
selecting the checkpoint or score threshold.

## Frozen split and CPU preflight

Split seed 20260912 produced 2,842 training images, 573 revealed-history
calibration images, three quarantined GT-defect images, and 48 new sealed audit
images. All source groups are disjoint. The split manifest SHA-256 is
`b45ae47e54de2adce00cede09aea40a249434f44b30044587d34050675f2a4d7`.

The CPU preflight read all training and calibration images and rehashed all
three pinned model files. It found zero invalid transformed boxes. It read zero
quarantined-image pixels, zero sealed-audit pixels, and zero Validation/Test
images. No GPU work or whole-image generation ran.

The one-batch GPU smoke then completed with finite loss 287.36 and 9.316 GiB
peak VRAM. It consumed 32 transformed helmet boxes from eight training images
and read zero calibration, sealed-audit, Validation, or Test images. The frozen
12-epoch run is now the next permitted action.

The 48 new audit images may be read only once after training. Numeric success
still cannot open generation: `kuotunyu` must inspect every exact-box review
case and report zero problems.

## Frozen numeric outcome

The RTX 4090 completed all 12 epochs in 38.96 minutes. The frozen calibration
rule selected epoch 3 at threshold 0.03. On the one-shot 48-image audit it
produced 110 true positives, 12 false positives, and 26 false negatives:

- precision 0.9016;
- recall 0.8088;
- F1 0.8527;
- median matched IoU 0.8430.

All numeric gates passed. The checkpoint SHA-256 is
`9b5ee6d360d3768a52ba9261f444e23b6f1da2c56f2eca682e207160604da5c4`.
The exact predictions and six review renderings are frozen under manifest
SHA-256
`b6c3855ef4620d7a606e83aee32ad4caf98182c3f5f62ae86dd49990160f36cd`.
Validation/Test reads and whole-image generation remain zero. Generation stays
locked until the owner reports zero visual problems.
