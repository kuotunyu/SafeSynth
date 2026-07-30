# Supervised labeler v22 CPU preflight

- Status: **passed; waiting for one-batch GPU smoke**
- Split SHA-256: `f0b2c8472931ec97a3c8045051e2a084d0e7d2438e795bace16272c4ffd6065c`
- Training normalized / read: **2228 / 2242 images**
- Calibration normalized / read: **617 / 621 images**
- Invalid transformed boxes: **0**
- Prior sealed, nonselected v21, and v22 groups in model data: **0**
- Owner-reviewed v21 audit groups in training: **48**
- Historic positive replay: **12.0**
- v21 miss replay: **40.0**
- v21 hard-negative replay: **28.0**
- Replay overlap policy: **maximum, never stacked**
- Calibration / audit precision floors: **0.90 / 0.85**
- Out-of-image raw box-area limit: **0.10**
- Independent-audit pixels read: **0**
- Sealed-reserve pixels read: **0**
- Validation/Test images read: **0 / 0**
- GPU work / whole-image generation: **no / no**
