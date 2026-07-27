# H4 paired-person unit CPU feasibility audit

- Status: **candidate successor feasible; not preregistered**
- Data scope: **Train cutout bank and Train annotations only**
- Validation/Test images read: **0 / 0**
- Model inference run: **no**
- Person cutouts: **113**
- Person cutouts with an upper-body helmet/head pair: **111**
- Paired source groups: **76**
- Paired person cutouts at least 80 px high: **90**
- Paired headlike annotations: **113** ({'head': 1, 'helmet': 112})

## Interpretation

The existing Train bank can support a successor that moves one anatomically coupled person + helmet/head unit instead of an isolated helmet. This removes the structural cause of the v5 floating-hat failures without downloading data or using a GPU.

This is only a feasibility result. Person masks and paired labels still require stricter truncation, pose, and visual gates followed by a new zero-issue CPU draft sheet before any FLUX call.
