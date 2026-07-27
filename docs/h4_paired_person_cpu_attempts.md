# H4 paired-person CPU attempts

After `kuotunyu` rejected 33/64 v5 isolated helmet/head drafts, three
Train-only, CPU-only successors tested whether moving an anatomically coupled
person + helmet/head unit could remove the structural failure.

## v6 — geometry only

- Fixed seed: `20260801`
- Inputs produced: 64
- Model inference: no
- Result: internally rejected before user review

The complete person removed the floating-helmet failure, but geometry alone
paired incompatible scenes, clothing, poses, and capture styles.

## v6b — scene matched

- Fixed seed: `20260802`
- Inputs produced: 64
- Frozen full-image CLIP cosine floor: `0.60`
- Feature access: 3,500 frozen Train rows only
- Model inference: no
- Result: internally rejected before user review

Scene matching improved the background type, but it did not match the donor
pose to the insertion location. Whole-person Lab and noise matching also made
faces and clothing look post-processed.

## v7 — source position and core preservation

- Fixed seed: `20260803`
- Required inputs: 64
- Inputs produced: 63
- Backgrounds exhausted: 3,500/3,500 frozen Train images
- Strict donors: 22 cutouts across 19 groups
- Maximum uses per donor: 3
- Model inference: no
- Result: capacity rejected before user review

v7 kept the donor's normalized vertical position, allowed only small horizontal
movement or shrinking, and disabled whole-person Lab/noise changes. Its
theoretical maximum was only 66 images, and the exhaustive placement search
stopped at 63. Increasing the reuse cap after seeing this result would hide the
source-diversity failure. The same pool cannot support the later 300-image H4
gate, so whole-person pasting is closed.

All three attempts read zero Validation/Test images, accessed zero Test feature
rows, ran no GPU model, and computed no H4 AUC.
