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
