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

Upload `outputs/flux2_v2_colab_diagnostic_inputs_portable.zip`. Its ZIP entry
names use POSIX forward slashes so Linux creates `case_07/`, `case_13/`,
`case_17/`, and `case_52/` directories. The notebook also normalizes literal
Windows backslashes for compatibility with the superseded package.

The Colab dependency cell pins `safetensors==0.8.0`, matching the resolved
project lock. An earlier `0.7.0` pin conflicted with the current Transformers
dependency and must not be reused after a runtime restart.

The notebook disables Hugging Face Xet downloads and raises the HTTP response
timeout to 120 seconds before importing Diffusers. This avoids a Colab failure
observed while Xet remained indefinitely at 4/17 files and 10.3/16.0 GiB
reconstructed. Standard HTTP uses the existing Hugging Face partial cache and
Range requests rather than intentionally deleting or restarting the download.

## Result and method decision

The registered run completed on `NVIDIA A100-SXM4-40GB` with the full model on
CUDA. It produced all 12 expected outputs in 86.70 seconds. The unchanged result
archive has SHA-256
`33bd82ae1625137b0a42aaf92473e94c95591eb29a1d846bf4833b060003e7c6`.

CPU-only verification found:

- every variant changed only pixels inside the edit mask; outside-mask changes
  were exactly zero;
- removing the reference at `strength=0.55` changed masked RGB values by only
  0.2260/255 on average, so the neutral-gray reference is not the operative
  cause of the observed failures;
- lowering strength from 0.85 to 0.55 had an aggregate masked RGB MAE of
  3.0179/255, but the effect varied substantially by case and produced no
  consistent visual improvement;
- the four clean Train examples produced locally coherent helmet edits, but
  none of the registered calls can repair an invalid draft or a mislocalized
  anchor before inference.

No registered variant is selected. The original v1 method remains rejected by
the human identity gate. Before another untouched 64-image identity pilot, a
new method must preregister an input-validity and anchor-localization guard.
This result does not compute H4, reopen M13, or start Phase 2.

Machine-readable metrics are in
[`reports/flux2_v2_diagnostic.json`](../reports/flux2_v2_diagnostic.json), with
the human-readable summary in
[`reports/flux2_v2_diagnostic.md`](../reports/flux2_v2_diagnostic.md).
