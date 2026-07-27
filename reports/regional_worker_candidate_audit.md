# Regional worker v9 CPU capacity audit

- Status: **rejected before GPU**
- Architecture: `prompt_only_adjacent_worker_inpaint_v9`
- Eligible real-person anchors: **99**
- Strict empty placements: **123**
- Candidate images / frozen groups: **17 / 17**
- Required groups to preserve a distinct-group path to the 64-case gate: **64**
- Validation/Test images read: **0 / 0**
- Model inference run: **no**
- H4 AUC computed: **no**

The prompt-only regional generation API is feasible, but requiring a real
`person` annotation as the adjacent scale and ground anchor collapses the pool
to 17 groups. This is not materially better than the rejected v7 whole-person
pool and cannot support the later gate honestly.

The next CPU-only architecture may use the much larger Train `helmet` pool as
the scale anchor. Its full-person geometry must be calibrated only from
Train person/helmet pairs before any output is generated.
