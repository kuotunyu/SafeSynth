# H4 paired-person v7 capacity failure

- Status: **capacity infeasible before kuotunyu review**
- Built/required: **63 / 64**
- Attempted Train backgrounds: **3,500 / 3,500**
- Strict donor cutouts/groups: **22 / 19**
- Maximum uses per cutout: **3**
- Theoretical maximum: **66**
- Model inference run: **no**
- H4 AUC computed: **no**
- Validation/Test images read: **0 / 0**
- Test feature rows read: **0**

The fixed search exhausted every frozen Train background and still produced
only 63 of 64 required drafts. Raising the reuse cap after this result would
hide the source-diversity failure, so the whole-person paste architecture is
stopped before human review or GPU inference. It also cannot support the later
300-image H4 gate with credible donor diversity.
