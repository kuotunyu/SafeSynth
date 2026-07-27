# M9 / Spike H6 — hard-negative mining purity

- Frozen Train images sampled: 200
- HSV/shape candidates before semantic guards: 943
- Candidates after IoU + worn-helmet guards: 80
- Guard rejections: `{'annotation_overlap': 266, 'head_like_region_below': 478, 'inside_helmet_typical_range': 119}`
- Contact-sheet cells: 64
- Maximum tolerated real-helmet rate: 10%
- Contact-sheet SHA256: `0e385d857067aa293c5e3d0dd43ad84b4141ff9bac5c8d4aefed187ee9c45739`

## Human gate

**APPROVED by kuotunyu on 2026-07-27.**

- Real helmets in cyan boxes: **0 / 64 (0.0%)**
- Result: **PASS** against the maximum tolerated rate of 10%
- Exact approval: `真正安全帽 0 格，批准；H4 選 A。`
- Machine-readable record: `reports/hard_negative_signoff.json`

Automatic safeguards already applied:

1. IoU with every existing annotation is below the fixed limit.
2. The region has no annotation/skin-like head below and falls outside the
   frozen Train helmet p5–p95 size/aspect envelope.
3. The contact sheet and its SHA are immutable inputs to the recorded signoff.
