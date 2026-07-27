# H4 guarded-input v2 pre-registration

Status at registration: the original 64-image Option A pilot failed its human
identity gate, and the four-case A100 diagnostic selected none of its three
model-call variants. No output from the guarded v2 architecture exists. The
original H4 AUC 0.7964 and the failed identity decision remain binding.

## Failure addressed

The rejected pilot exposed two upstream failures that model-call parameters
cannot repair:

1. canonical cell 10 used `hard_hat_workers861.png`, whose reflected top border
   contains a duplicated headlike annotation even though the selected central
   anchor itself is valid;
2. canonical cell 12 selected annotation `11122`, whose box ends one pixel from
   the lower frame edge and targets truncated/reflected content.

The v2 change is therefore an input guard. The FLUX.2 model, revision, prompt,
reference canvas, edit mask, strength, step count, and guidance remain exactly
the rejected v1 call. This follows the A100 diagnostic: removing the reference
was effectively a no-op and lower strength showed no consistent improvement.

## Frozen CPU-only guard

The guard runs before cutout selection or model inference and reads Train
metadata plus existing Pass-1 SAM2 QC only.

1. A context-replacement background is rejected if any `helmet` or `head`
   annotation has less than 4 pixels of clearance from any image edge. This
   removes drafts containing labelled reflected/truncated headlike border
   content, including the known invalid draft.
2. A replacement anchor must have `sam2_qc_pass=true`.
3. Its edge clearance must be at least
   `max(8 px, ceil(0.10 × max(box width, box height)))`. Eight pixels leave a
   three-pixel unchanged buffer outside the frozen five-pixel generative
   dilation; the fractional term scales the buffer for large anchors.
4. A background with no eligible anchor is rejected before inference. The
   existing deterministic background-retry limit remains 20.
5. Every generated record stores the measured background and selected-anchor
   margins so the decision can be audited.

The exact values live under
`compose.context_replacement.input_guard` in
[`configs/compose.yaml`](../configs/compose.yaml).

## Untouched identity pilot

After CPU coverage auditing, but not as part of that audit:

- generate exactly 64 new Train-only images with root seed `20260728`;
- use only the guarded `context_replacement` path;
- keep the fixed 8×8 DRAFT / EDIT MASK / REFERENCE / OUTPUT evidence sheet;
- require zero label mismatches, at most 3 severe identity failures, zero
  outside-edit pixel changes, and zero protected-core pixel changes;
- do not compute H4 before `kuotunyu` completes the visual identity decision.

This pilot is new evidence, not a repair or relabelling of the rejected seed
`20260727` outputs. A failure returns to architecture design and does not permit
parameter searching on H4.

## Final H4 remains unchanged

Only a passed identity pilot may use the already registered one-shot 300-image
H4 generation and classifier seeds. The maximum AUC remains 0.60. The guard
does not reopen M13 or Phase 2 by itself.
