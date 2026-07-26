# Spike H3 — Guarded CLIP Extension

- OpenCLIP model: `ViT-B-32`
- Pretrained tag: `laion2b_s34b_b79k`
- Base pHash edge: Hamming `<= 10`

| Cosine threshold | pHash guard | Groups | Largest group | Singletons |
|---:|---:|---:|---:|---:|
| 0.850 | 12 | 4,867 | 4 | 4,742 |
| 0.900 | 12 | 4,871 | 4 | 4,750 |
| 0.950 | 12 | 4,875 | 4 | 4,757 |
| 0.975 | 12 | 4,875 | 4 | 4,757 |
| 0.850 | 16 | 4,854 | 4 | 4,719 |
| 0.900 | 16 | 4,865 | 4 | 4,739 |
| 0.950 | 16 | 4,875 | 4 | 4,757 |
| 0.975 | 16 | 4,875 | 4 | 4,757 |
| 0.850 | 20 | 4,808 | 8 | 4,643 |
| 0.900 | 20 | 4,861 | 4 | 4,731 |
| 0.950 | 20 | 4,875 | 4 | 4,757 |
| 0.975 | 20 | 4,875 | 4 | 4,757 |

## Selected candidate (visual review passed)

- Cosine threshold: `0.850`
- pHash guard: `20`
- Groups: `4,808`
- Largest group: `8`

The largest groups are rendered in `h3_clip_largest_groups.png`. The grid was visually checked before M4 froze this candidate.
