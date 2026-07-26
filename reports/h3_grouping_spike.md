# Spike H3 — pHash Grouping Structure

| Hamming threshold | Groups | Largest group | Singleton groups |
|---:|---:|---:|---:|
| 4 | 4,907 | 4 | 4,820 |
| 6 | 4,900 | 4 | 4,806 |
| 8 | 4,890 | 4 | 4,786 |
| 10 | 4,875 | 4 | 4,757 |

## Five seeded split simulations

### Threshold 4

| Seed | Train | Validation | Test |
|---:|---:|---:|---:|
| 42 | 3,500 | 750 | 750 |
| 43 | 3,500 | 750 | 750 |
| 44 | 3,500 | 750 | 750 |
| 45 | 3,500 | 750 | 750 |
| 46 | 3,500 | 750 | 750 |

### Threshold 6

| Seed | Train | Validation | Test |
|---:|---:|---:|---:|
| 42 | 3,500 | 750 | 750 |
| 43 | 3,500 | 750 | 750 |
| 44 | 3,500 | 750 | 750 |
| 45 | 3,500 | 750 | 750 |
| 46 | 3,500 | 750 | 750 |

### Threshold 8

| Seed | Train | Validation | Test |
|---:|---:|---:|---:|
| 42 | 3,500 | 750 | 750 |
| 43 | 3,500 | 750 | 750 |
| 44 | 3,500 | 750 | 750 |
| 45 | 3,500 | 750 | 750 |
| 46 | 3,500 | 750 | 750 |

### Threshold 10

| Seed | Train | Validation | Test |
|---:|---:|---:|---:|
| 42 | 3,500 | 750 | 750 |
| 43 | 3,500 | 750 | 750 |
| 44 | 3,500 | 750 | 750 |
| 45 | 3,500 | 750 | 750 |
| 46 | 3,500 | 750 | 750 |

## Decision

- Selected pHash Hamming threshold: `10`
- Selected group count: `4,875`
- Selected maximum group size: `4`
- CLIP trigger (`> 2,000` groups): `yes`

The final M4 split must still use class-stratified allocation; these simulations only test whether group geometry makes the requested image ratios feasible.
