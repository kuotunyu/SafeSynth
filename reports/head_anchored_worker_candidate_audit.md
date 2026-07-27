# Head-anchored worker v9b CPU capacity audit

- Status: **rejected before GPU**
- Architecture: `prompt_only_head_anchored_worker_inpaint_v9b`
- Train person/helmet calibration pairs: **164**
- Images with a preliminary empty region: **1,883**
- Strict placements after all guards: **92**
- Candidate images / frozen groups: **7 / 7**
- Required groups: **64**
- Validation/Test images read: **0 / 0**
- Model inference run: **no**
- H4 AUC computed: **no**

Train median geometry inferred a person's box from each helmet annotation before
testing adjacent empty regions. The broad anchor pool did not solve the real
constraint: full-body regions either crossed the frame, covered an existing
annotation, were too small/large, or belonged to a reflected-padding image.

This closes regional full-worker insertion on the current 416×416 source
images. Whole-image generation is the remaining architecture without a source
background capacity ceiling, but it requires an independently validated
automatic labeler before scale-up.
