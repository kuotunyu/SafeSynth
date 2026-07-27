# FLUX.2 v2 Colab diagnostic

## Status and scientific scope

This is a **method-development diagnostic**, not a replacement identity pilot
and not H4. It uses four Train-only examples and must not be cited as a gate
result.

- It does not read Validation or Test images.
- It does not compute an H4 classifier or AUC.
- It does not reopen M13 or Phase 2.
- Its only purpose is to determine whether the neutral-gray reference canvas or
  the registered `strength=0.85` is causing the visible boundary corruption.
- After choosing a v2 method, its parameters must be frozen before a new,
  untouched 64-image identity pilot is generated.

## Inputs

Run:

```powershell
uv run python -m scripts.prepare_flux2_v2_colab_diagnostic
```

The CPU-only exporter writes four cases to:

```text
outputs/flux2_v2_colab_diagnostic_inputs/
```

Each case contains:

- `draft.png`: the composited scene before generative boundary editing;
- `edit_mask.png`: the only pixels that may be replaced by model output;
- `reference.png`: the current neutral-gray reference canvas;
- `metadata.json`: Train provenance, seed, anchor and file hashes.

`manifest.json` binds the four inputs to the pinned model revision. The input
preview is `input_preview.png`.

## Registered development comparisons

The notebook compares these variants on the same four inputs and the same
per-case seeds:

1. `v1_reference_strength_085`: exact current model-call settings;
2. `reference_strength_055`: keep the reference but reduce edit strength;
3. `no_reference_strength_055`: omit the neutral-gray reference and reduce edit
   strength.

This is a causal diagnostic, not a broad sweep. Do not add variants after
looking at the output. If none is acceptable, v1 remains failed and the method
needs a new design.

## Colab runtime

Use an **A100 40 GB** runtime. The pinned 4B package uses approximately
14.88 GiB of local storage; at 416×416 with one image per call, 40 GB is enough
to keep the model on CUDA without CPU offload. The notebook detects at least
35 GiB of visible VRAM and selects `full_model_on_cuda`; smaller GPUs fall back
to model CPU offload.

An A100 80 GB runtime is not needed for this diagnostic. Its extra capacity is
useful for larger models, larger batches, or substantially larger images, none
of which apply here. Select 80 GB only if Colab offers it at the same displayed
compute-unit rate as 40 GB.

The notebook produces `flux2_v2_diagnostic_results.zip`. Download that single
file and return it unchanged for local verification and a blinded contact
sheet.

The Colab dependency cell pins `safetensors==0.8.0`, matching the resolved
project lock. An earlier `0.7.0` pin conflicted with the current Transformers
dependency and must not be reused after a runtime restart.
