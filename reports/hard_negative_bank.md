# Hard-negative cutout bank

- Total cutouts: **240**
- Procedural (primary, ADR-012): **240**
- Mined (supplementary): **0** of 16 qualifying (80 mined, min_side >= 24 px)
- Cutout min_side: min 24 / median 50 / max 79 px
- Annotations emitted by these cutouts: **0** (correct by construction)
- Validation/Test images read: **0**
- Contact sheet: `hard_negative_bank_grid.png` (magenta backdrop)

Real helmet cutouts in this dataset have min_side p10=22 / median=34 /
p90=74, so procedural material is rendered into the same range on
purpose: a distractor only counts as *hard* if it could plausibly be
mistaken for a helmet.
