# M2 Conversion Report

Generated deterministically by `scripts/prepare_data.py`.

## Source

- Kaggle version: `1`
- Official uncompressed file total: `1,320,154,239` bytes
- Downloaded archive: `hard-hat-detection-v1.zip` (`1,314,241,385` bytes)
- Archive SHA256: `aa5c80a85f9f4bd3b27e44256f8e36f9a32c53ee423132fa6cd5ea603781be62`
- Global minimum coordinate: `0`
- Applied xmin/ymin offset: `0`
- Decision: global minimum is 0; treating coordinates as zero-based

## Dataset facts

| Class | Instances | Images | Mean area (px²) | Mean area (%) | Min-side p1 / p50 / p99 (px) |
|---|---:|---:|---:|---:|---:|
| `helmet` | 18,966 | 4,581 | 2,199.75 | 1.27 | 7.00 / 27.00 / 122.00 |
| `head` | 5,785 | 920 | 1,083.37 | 0.63 | 7.00 / 23.00 / 73.00 |
| `person` | 751 | 158 | 12,175.09 | 7.05 | 8.00 / 58.00 / 247.50 |

- Images: `5,000`
- Annotations: `25,502`
- XML/PNG size mismatches: `0`
- Unknown labels: `0`
- `iscrowd != 0`: `0`
- Difficult histogram: `{0: 25502}`
- Truncated histogram: `{0: 25502}`

## Recorded corrections

- None

## Verification

- COCO self-evaluation mAP: `1.000`
- Bounding-box convention: `w = xmax - xmin`, without `+1` (DATA-06).
- All paths use forward slashes and all `iscrowd` values are zero.
