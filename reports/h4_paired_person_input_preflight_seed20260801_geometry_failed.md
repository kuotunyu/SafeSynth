# H4 paired-person v6 CPU input preflight

- Status: **rejected before kuotunyu review**
- Inputs: **64**
- Root seed: `20260801`
- Model inference run: **no**
- H4 AUC computed: **no**
- Validation/Test images read: **0 / 0**
- Strict donor cutouts/groups: **30 / 25**
- Used donor cutouts/groups: **23 / 21**
- Distinct Train backgrounds: **64**
- Geometry fingerprint: `554e34c138785b7fd63cde8e50241b3bb1d043acfc94c05ceebd61206c49b22c`
- Contact sheet: `C:\Users\3Hml\Desktop\mySyntheticData\2_SafeSynth\reports\figures\h4_paired_person_input_preflight_seed20260801_geometry_failed.png`

Each cell shows the full CPU draft on the left and an enlarged person crop on the right. Cyan is the coupled person support; yellow is the linked helmet/head label transported with that same person.

Internal CPU review found that geometry-only matching still paired incompatible
scenes, clothing, poses, and capture styles. The candidate was rejected without
asking `kuotunyu` to review it and without running FLUX. v6b adds a frozen
Train-only full-image CLIP similarity floor before placement.
