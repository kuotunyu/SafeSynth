# Supervised labeler v8 pre-registration

v7 passed every numeric gate, but `kuotunyu` found real helmets without
magenta boxes in cells 08, 11, 13, 36, and 39. The six missed instances and
all v1-v7 calibration/audit groups are now revealed evidence. They may
motivate v8 but cannot pass any v8 gate.

## Frozen v8 changes

v8 changes exactly two model-facing rules:

1. maximum relative prediction area increases from 0.08 to 0.14 and maximum
   relative prediction height increases from 0.35 to 0.40;
2. Train images containing a helmet with relative annotation area at or below
   0.0075 receive sampling weight 2.0.

The first change is the smallest round-number relaxation that recovers all
five v7 misses caused by geometry filtering. The second targets the only
remaining miss: a 0.0063-relative-area helmet whose matching box scored
0.0052. Lowering the global score threshold is forbidden because threshold
0.005 produced 1,268 false positives on the revealed v7 audit.

Everything else remains frozen: pinned Apache-2.0 RT-DETRv2 R50 base
checkpoint, initialization from that base, training seed, 12 epochs,
optimizer, score grid, calibration precision floor, numeric gates, and
owner-only zero-problem visual gate.

## Data boundary

All v7 calibration and audit groups become v8 calibration. A deterministic
group-disjoint Train-only split with seed 20260822 will select 48 new audit
images stratified by helmet size. Before v8 training:

- the new audit pixels and metrics remain sealed;
- Validation and Test reads remain zero;
- generation remains locked.

After a numeric pass, `kuotunyu` must review 48 exact-box cases. Any reported
problem rejects v8.

## Frozen numeric outcome

The RTX 4090 run completed all 12 epochs in 21.74 minutes. The frozen
calibration rule selected epoch 6 at threshold 0.035. On the one-shot new
48-image sealed audit it produced 164 true positives, 14 false positives, and
25 false negatives:

- precision 0.9213;
- recall 0.8677;
- F1 0.8937;
- median matched IoU 0.8442.

All three numeric gates passed. Validation/Test reads and whole-image
generation remained zero. The checkpoint SHA-256 is
`b546c10603abe61bd5e65200e55f29b25fc7874499c309d4cac6b67b84dfb914`.
Numeric success does not open generation; the exact 48-case owner review
remains mandatory.
