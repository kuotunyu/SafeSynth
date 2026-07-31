# H4 paired-person v6 CPU input preflight

- Status: **rejected before kuotunyu review**
- Inputs: **64**
- Root seed: `20260802`
- Model inference run: **no**
- H4 AUC computed: **no**
- Validation/Test images read: **0 / 0**
- Strict donor cutouts/groups: **30 / 25**
- Used donor cutouts/groups: **23 / 21**
- Distinct Train backgrounds: **64**
- Scene CLIP cosine threshold/observed minimum: **0.6 / 0.6014**
- Geometry fingerprint: `d12564be5d382c1038002913051b492da76ea2d6c82722beab922352e2c01c12`
- Contact sheet: `reports/figures/h4_paired_person_input_preflight_seed20260802.png`

Each cell shows the full CPU draft on the left and an enlarged person crop on the right. Cyan is the coupled person support; yellow is the linked helmet/head label transported with that same person.

Internal CPU review found that full-image scene matching improved background
compatibility but did not constrain the donor pose to the placement. It still
placed ordinary standing people into elevated work positions. Whole-person Lab
and noise matching also visibly changed faces and clothing. The candidate was
rejected without asking `kuotunyu` to review it and without running FLUX.
