# Supervised labeler v23 CPU preflight

- Status: **passed; waiting for one-batch GPU smoke**
- Split SHA-256: `0bf0ac61aedd57b9b5ff05379b33f5d09963e37ed77d2fa57e70d44b466cffd4`
- Training normalized / read: **2179 / 2193 images**
- Calibration normalized / read: **617 / 621 images**
- Invalid transformed boxes: **0**
- Prior sealed, nonselected v22, and v23 groups in model data: **0**
- Owner-reviewed v22 audit groups in training: **48**
- Historic positive replay: **12.0**
- v22 miss replay: **40.0**
- v22 hard-negative replay: **28.0**
- Replay overlap policy: **maximum, never stacked**
- Calibration / audit precision floors: **0.90 / 0.85**
- Out-of-image raw box-area limit: **0.10**
- Independent-audit pixels read: **0**
- Sealed-reserve pixels read: **0**
- Validation/Test images read: **0 / 0**
- GPU work / whole-image generation: **no / no**
