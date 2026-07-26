# M4 Grouping and Split Report

- Groups: `4,808`
- Maximum group size: `8`
- Base pHash threshold: `10`
- CLIP enabled: `True`
- CLIP model/tag: `ViT-B-32` / `laion2b_s34b_b79k`
- CLIP cosine / pHash guard: `0.85` / `20`

## Group-size histogram

| Group size | Number of groups |
|---:|---:|
| 1 | 4,643 |
| 2 | 149 |
| 3 | 11 |
| 4 | 3 |
| 6 | 1 |
| 8 | 1 |

## Largest 20 groups

| Group ID | Images | Split | Image IDs |
|---:|---:|---|---|
| 284 | 8 | `val` | 285, 1607, 1741, 1824, 2223, 3369, 3784, 3824 |
| 197 | 6 | `test` | 198, 3207, 3383, 3817, 4423, 4513 |
| 596 | 4 | `train` | 597, 3468, 3932, 4418 |
| 1068 | 4 | `train` | 1073, 4075, 4102, 4776 |
| 2442 | 4 | `train` | 2489, 2896, 3171, 4557 |
| 93 | 3 | `val` | 94, 4233, 4769 |
| 200 | 3 | `train` | 201, 1033, 4882 |
| 366 | 3 | `train` | 367, 2594, 4351 |
| 583 | 3 | `train` | 584, 870, 3371 |
| 609 | 3 | `train` | 610, 1247, 2994 |
| 625 | 3 | `train` | 626, 1667, 3291 |
| 891 | 3 | `train` | 894, 2356, 2602 |
| 1294 | 3 | `test` | 1305, 3159, 4101 |
| 1829 | 3 | `train` | 1854, 1956, 2818 |
| 1838 | 3 | `train` | 1863, 3151, 4289 |
| 2325 | 3 | `train` | 2366, 2956, 4183 |
| 3 | 2 | `train` | 4, 1213 |
| 10 | 2 | `val` | 11, 4104 |
| 12 | 2 | `train` | 13, 2735 |
| 13 | 2 | `train` | 14, 2439 |

## Split balance

| Split | Images | Helmet | Head | Person |
|---|---:|---:|---:|---:|
| `train` | 3,500 | 13,219 | 4,071 | 525 |
| `val` | 756 | 2,922 | 835 | 113 |
| `test` | 744 | 2,825 | 879 | 113 |

All images in a connected component share one split. The split union contains all 5,000 images and the three partitions are pairwise disjoint.
