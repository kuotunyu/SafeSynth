# Frozen Class Distribution

All values are regenerated from `split_manifest.json` and `coco_all.json`.

## Whole dataset

| Class | Instances | Images | Mean bbox area (px²) | Mean image area (%) | Min-side p1 / p50 / p99 (px) |
|---|---:|---:|---:|---:|---:|
| `helmet` | 18,966 | 4,581 | 2,199.75 | 1.273 | 7.00 / 27.00 / 122.00 |
| `head` | 5,785 | 920 | 1,083.37 | 0.627 | 7.00 / 23.00 / 73.00 |
| `person` | 751 | 158 | 12,175.09 | 7.045 | 8.00 / 58.00 / 247.50 |

Object count per image: mean `5.100`, p50 `4`, p95 `14`, max `68`.

The measured single-box mean areas resolve the earlier source conflict: `helmet` and `head` match the smaller published reading, while `person` remains the largest class by box area.

## Frozen split

| Split | Images | Annotations | Helmet | Head | Person | Images with helmet / head / person |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 3,500 | 17,815 | 13,219 | 4,071 | 525 | 3,205 / 648 / 112 |
| `val` | 756 | 3,870 | 2,922 | 835 | 113 | 695 / 131 / 23 |
| `test` | 744 | 3,817 | 2,825 | 879 | 113 | 681 / 141 / 23 |

`head` is image-level scarce: 5,785 instances occur in only 920/5,000 images. `person` is more extreme: 751 instances occur in 158/5,000 images, and the class is known to be incompletely annotated. All final claims therefore remain relative comparisons on the same frozen Test split.
