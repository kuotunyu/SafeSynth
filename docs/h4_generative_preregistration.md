# H4 generative insertion pre-registration

Status at registration: kuotunyu approved Option A and the exact H6 grid on
2026-07-27. No FLUX.2 weights have been downloaded and no output from this
method exists. The original feathered-alpha H4 result (AUC 0.7964) remains
binding until this one-shot replacement method passes the unchanged 0.60 gate.

## Why this is a new architecture

The failed methods all render source pixels into a target scene with a
hand-designed boundary operator. The new method gives a diffusion model the
draft composite, a reference cutout, and a narrow mask around the object
boundary. It generates the transition between object and scene rather than
choosing another alpha or Poisson parameter.

The protected object core is copied back exactly after inference, and every
pixel outside the registered edit mask is copied from the draft. The model
therefore cannot pass by erasing, recolouring, moving, or globally restyling the
object or background.

## Frozen model and runtime

- Model:
  [`black-forest-labs/FLUX.2-klein-base-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
- Revision: `a3b4f4849157f664bdbc776fd7453c2783562f4d`
- License: Apache-2.0
- Pipeline:
  [Diffusers 0.39.0](https://github.com/huggingface/diffusers/releases/tag/v0.39.0)
  `Flux2KleinInpaintPipeline`
- Required Diffusers-format download: 15,980,131,711 bytes (14.88 GiB)
- Device/dtype: RTX 4090 CUDA, bfloat16, model CPU offload enabled
- Runtime is local-only after an explicit, separately approved download.

The 4B checkpoint was selected over FLUX.1 Fill (non-commercial and much
larger), Qwen-Image-Edit (20B), and SDXL Inpainting (older OpenRAIL++ model).
The selected checkpoint is Apache-2.0, its official card targets consumer GPUs,
and Diffusers exposes both an explicit mask and an object reference image.

## Frozen synthesis method

1. The existing deterministic compositor builds a geometry-correct draft and
   annotation masks without reading Val or Test.
2. Dilate the planned object mask by 5 pixels.
3. Erode it by 2 pixels to define a protected core. If this would preserve less
   than 35% of the planned object, retain the most interior pixels until the
   fixed minimum is met.
4. The white inpaint mask is `dilated support - protected core`.
5. Condition the inpaint pipeline on the draft, the class-specific prompt, and
   a neutral 416-pixel reference canvas containing the original RGBA cutout.
6. Run 50 steps, strength 0.85, guidance 4.0, with a per-sample deterministic
   generator and a 48-pixel mask-crop margin.
7. Copy the protected core and all pixels outside the edit mask back from the
   draft exactly.
8. Preserve the pre-inpaint geometry and provenance. A generated result may not
   introduce a second labelled object; visual and automated identity checks run
   before H4.

All numeric values live in
[`configs/generative_inpaint.yaml`](../configs/generative_inpaint.yaml).

## Pilot identity gate

Before any AUC is computed, render exactly 64 images using root seed 20260727.
The fixed 8x8 sheet shows draft, edit mask, reference, and output for each item.
kuotunyu reviews it without seeing an H4 score.

- Label mismatches are not allowed (0/64).
- At most 3/64 may have a severe identity failure.
- Outside-mask and protected-core pixel changes must both be exactly zero.
- No parameter is selected by an artifact-classifier score.

Failure returns to architecture design. It does not authorize weakening the
identity rules.

## One-shot final H4

Only after the pilot passes:

- generate exactly 300 lossless PNG images with seed 1618033;
- use a new stable source-group fold with classifier seed 1732051;
- keep pooled same-class, same-fold, nearest-log-size real controls;
- keep the 64-pixel HOG+HSV features and L2 C=1 logistic regression;
- pass only if AUC is no greater than 0.60;
- report the unchanged bootstrap interval and all provenance.

This is one-shot. A failure blocks M13. The fold seed, model, prompts, masks,
inference parameters, classifier, and threshold may not be tuned after seeing
the result.
