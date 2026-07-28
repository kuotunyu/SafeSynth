# Supervised labeler v10 CPU normalization preflight

- Status: **passed; waiting for one-batch GPU smoke**
- Split SHA-256: `9b36cedc3c159614f668b5fafaa12ad33f36d8ff84d62aba69337e1097aa4d6b`
- Training normalized / read: **2875 / 2893 images**
- Calibration normalized / read: **525 / 528 images**
- Removed mirrored/clipped helmet boxes (training / calibration): **3368 / 724**
- Invalid transformed boxes: **0**
- All four revealed v9 problem images normalized: **yes**
- Sampling weight counts: **{'1.0': 787, '2.0': 2106}**
- Sealed-audit pixels read: **0**
- Validation/Test images read: **0 / 0**
- GPU work / whole-image generation: **no / no**
