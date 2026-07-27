# Whole-person edit v8 CPU preflight

- Status: **approved by kuotunyu; 0 input issues**
- Architecture: `masked_existing_person_restyle_v8`
- Exact input manifest:
  `a0c0795e856f588d5aab90887057498259ac96e90fcfef8349669bf9df0f0af2`
- Fixed cases: **4**
- Candidate capacity after all CPU guards: **26 candidates / 23 frozen groups**
- Data scope: **Train only**
- Validation/Test images read: **0 / 0**
- Model inference run: **no**
- H4 AUC computed: **no**

## What the cyan region means

Each target is a worker already present in the original Train image. The cyan
region is the union of that worker's Pass-1 SAM mask and its single paired
helmet mask, dilated by three pixels. FLUX may change only those pixels. Every
pixel outside the mask is restored bit-exactly from the original image.

The four cases are selected deterministically from fixed person-height bands by
the largest safe distance from the frame edge. References are distinct
Train-only person cutouts from groups not used by any target.

## GPU lock

The local GPU runner verifies every input file hash and refuses to load FLUX
until `kuotunyu` approves this exact four-case sheet with zero input issues.
Passing this gate starts only a four-case method diagnostic; it does not run H4
or expand to 64 images.
