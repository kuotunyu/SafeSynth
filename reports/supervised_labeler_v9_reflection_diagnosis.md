# Supervised labeler v9 reflection diagnosis

- Evidence: **revealed Train-only owner failures; not gate-eligible**
- All four problem images contain detected reflection padding: **yes**
- Validation/Test images read: **0 / 0**
- Whole-image generations: **0**

| Cell | Axis | Clean crop | FP centers outside | Miss centers inside |
|---:|---|---|---:|---:|
| 6 | top_bottom | [0, 69, 416, 346] | 1 | 0 |
| 11 | left_right | [69, 0, 346, 416] | 0 | 1 |
| 12 | top_bottom | [0, 71, 416, 344] | 1 | 0 |
| 37 | top_bottom | [0, 79, 416, 336] | 0 | 2 |
