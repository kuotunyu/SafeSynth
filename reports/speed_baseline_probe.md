# Speed baseline probe (DEMO-03 / DEMO-05)

> **PROVISIONAL - NOT A FINAL RESULT.**
> Both models are measured from their PUBLIC PRETRAINED COCO checkpoints, not
> from this project's fine-tuned 3-class weights, which do not exist yet (the
> four-arm run is still executing). Fine-tuning changes one linear layer at the
> end of the network - 91 or 80 class logits per query become 3 - and nothing
> else, so these numbers should carry over closely. They MUST still be
> re-measured on the final weights before any of them reaches the README.

- Generated: `2026-07-31T18:52:50Z`
- Device: `cuda (NVIDIA GeForce RTX 4090)`
- Requested dtype: `float16` -> effective dtype: `float16`
- Warmup / timed iterations: `20` / `200` (warmup discarded, never timed)
- Reported statistic: `median`; p95 also reported: `True`
- Probe image (VALIDATION split, never Test): `hard_hat_workers1000.png` (416x416)
- End-to-end post-processing threshold: `0.5` (`compliance.score_threshold`)

Every setting above is read from `configs/evaluation.yaml`; the harness in `src/evaluation/benchmark.py` holds no tunable number of its own.

> **Provenance of sections 1-4.** Every number in sections 1 to 4 comes from the single GPU run generated at the timestamp above, on the public pretrained checkpoints; they have not been re-measured since, and they are still pending re-measurement on this project's fine-tuned 3-class weights. Section 5's forbidden-package scan was re-executed separately on 2026-08-01 (see that section), because it is a property of the working tree rather than of that run.

## 1. Latency

| Measurement | Device | Batch | Input | dtype | Median (ms) | p95 (ms) | FPS | Iters (warmup+timed) |
|---|---|---:|---:|---|---:|---:|---:|---|
| RT-DETRv2-R18 [model-only] | cuda | 1 | 640 | float16 | 13.33 | 16.63 | 75.0 | 20+200 |
| RT-DETRv2-R18 [end-to-end] | cuda | 1 | 640 | float16 | 17.56 | 21.98 | 56.9 | 20+200 |
| RF-DETR-Nano [model-only] | cuda | 1 | 640 | float16 | 10.61 | 14.61 | 94.2 | 20+200 |
| RF-DETR-Nano [end-to-end] | cuda | 1 | 640 | float16 | 13.55 | 17.37 | 73.8 | 20+200 |

`model-only` is a forward pass on a tensor already resident on the device. `end-to-end` additionally includes image preprocessing, the host-to-device copy and `post_process_object_detection`; that is what a user feels. Both are wrapped in `torch.cuda.synchronize()` inside the timed region - without it the timer would measure how fast Python enqueues work.

### Contention check (performed, not asserted)

A benchmark taken while the machine was busy shows a long tail: the reported median can look ordinary while p95 blows out. EVERY measured row in this report - the resolution probes in section 4 as well as the headline numbers above - is therefore checked against `benchmark.max_p95_to_statistic_ratio` = `1.3`, and the measured ratio is printed rather than left for the reader to compute.

| Measurement | p95 / median | Verdict |
|---|---:|---|
| RT-DETRv2-R18 [model-only] | 1.25 | ok |
| RT-DETRv2-R18 [end-to-end] | 1.25 | ok |
| RF-DETR-Nano [model-only] | 1.38 | CONTENDED - repeat, do not publish |
| RF-DETR-Nano [end-to-end] | 1.28 | ok |
| RT-DETRv2-R18 [model-only @ 320] | 1.21 | ok |
| RT-DETRv2-R18 [model-only @ 1280] | 1.24 | ok |
| RF-DETR-Nano [model-only @ 320] | 1.26 | ok |
| RF-DETR-Nano [model-only @ 1280] | 1.11 | ok |
| RF-DETR-Nano [model-only @ 384, native preset] | 1.19 | ok |

**FAIL** - 1 of 9 measured rows exceeded the threshold.

## 2. Peak VRAM

| Model | Peak allocated VRAM (MiB) |
|---|---:|
| RT-DETRv2-R18 | 97.9 |
| RF-DETR-Nano | 106.0 |

Measured with `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` immediately before each model's timed region, so the figure covers weights plus activations for that model alone.

## 3. Landing check

| Model | Role | Checkpoint | Model class | Params (M) | Native input | Logits | Boxes | Detections |
|---|---|---|---|---:|---:|---|---|---:|
| RT-DETRv2-R18 | primary detector (Apache-2.0) | `PekingU/rtdetr_v2_r18vd` | `RTDetrV2ForObjectDetection` | 20.17 | 640 | `[1, 300, 80]` | `[1, 300, 4]` | 1 |
| RF-DETR-Nano | speed baseline (Apache-2.0, ADR-005) | `Roboflow/rf-detr-nano` | `RfDetrForObjectDetection` | 30.47 | 384 | `[1, 300, 91]` | `[1, 300, 4]` | 1 |

Every model above completed a real forward pass on a real project image; the shapes are read off the returned tensors, not assumed. `Detections` is the count surviving the post-processing threshold with the pretrained COCO head, and carries no meaning for this project's classes.

## 4. Resolution sensitivity: compute-bound or dispatch-bound?

Section 1 runs BOTH models at the config `input_size`, so that comparison is apples-to-apples. This section re-measures model-only latency at other input sizes, which is the only way to tell whether the section-1 numbers characterise the architecture or the dispatch path. The probe runs in BOTH directions - half the configured size and double it - because a flat line downwards on its own is also what a probe that changed nothing would produce. **These rows are not comparable to section 1.**

| Measurement | Device | Batch | Input | dtype | Median (ms) | p95 (ms) | FPS | Iters (warmup+timed) |
|---|---|---:|---:|---|---:|---:|---:|---|
| RT-DETRv2-R18 [model-only @ 320] | cuda | 1 | 320 | float16 | 13.32 | 16.05 | 75.1 | 20+200 |
| RT-DETRv2-R18 [model-only @ 1280] | cuda | 1 | 1280 | float16 | 13.38 | 16.60 | 74.8 | 20+200 |
| RF-DETR-Nano [model-only @ 320] | cuda | 1 | 320 | float16 | 9.29 | 11.67 | 107.6 | 20+200 |
| RF-DETR-Nano [model-only @ 1280] | cuda | 1 | 1280 | float16 | 12.09 | 13.37 | 82.7 | 20+200 |
| RF-DETR-Nano [model-only @ 384, native preset] | cuda | 1 | 384 | float16 | 8.92 | 10.61 | 112.1 | 20+200 |

- **RT-DETRv2-R18**: 320x320 costs `13.32` ms against `13.33` ms at 640x640 (`-0.1%`), on 4x fewer input pixels.
- **RT-DETRv2-R18**: 1280x1280 costs `13.38` ms against `13.33` ms at 640x640 (`+0.3%`), on 4x more input pixels.
- **RF-DETR-Nano**: 320x320 costs `9.29` ms against `10.61` ms at 640x640 (`-12.5%`), on 4x fewer input pixels.
- **RF-DETR-Nano**: 1280x1280 costs `12.09` ms against `10.61` ms at 640x640 (`+13.9%`), on 4x more input pixels.

MEASURED: across a 16x span of input pixel counts (320x320 to 1280x1280), the largest move any model-only measurement made was `+13.9%` (RF-DETR-Nano at 1280x1280). Compute-bound behaviour would be roughly -75% at the small end and +300% at the large end.

The criterion, stated before the numbers above are read: a compute-bound batch-1 measurement falls towards a quarter of its cost when both input dimensions are halved and rises towards four times it when they are doubled. To the extent a measurement does NOT follow the pixel count, its wall clock is dominated by per-operator Python and CUDA launch overhead in eager mode - it characterises OUR INFERENCE PATH rather than the network, and two models cannot be separated on the strength of it. Read section 1 accordingly: it is the latency a user of this eager-PyTorch demo would feel, not a claim about which architecture is faster.

## 5. Licence evidence

ADR-005 forbids AGPL-3.0 detector stacks in this repository: importing one would make the importing file a derivative work, which is incompatible with an MIT release. The licence is therefore re-verified against the Hub at benchmark time rather than trusted from the ADR text.

| Model | Hub licence | Licence tags | Pinned revision |
|---|---|---|---|
| RT-DETRv2-R18 | `apache-2.0` | `license:apache-2.0` | `5650961749fa93567c0d46fc7f43ea4f9e914107` |
| RF-DETR-Nano | `apache-2.0` | `license:apache-2.0` | `9f201577d9415f0378084748d27e68586e6b3600` |

Only the nano / small / medium / base / large RF-DETR variants are Apache-2.0. XL and 2XL are PML-1.0 and must not be used here.

### Forbidden-package scan (PLAN_PHASE2.md M20)

ADR-005 forbids the AGPL-3.0 detector stack because importing it would make the importing file a derivative work. The counts below are the return value of `scripts/check_forbidden_licences.py`, a standalone checker that exits non-zero when it finds a match - they are not a stored sentence. The search term is assembled from fragments at run time so the literal appears in no source file, which is why NO file is exempt from the scan, the checker included.

- Roots scanned: `src/`, `scripts/`, `notebooks/`
- Files read: `198`
- Matches: `0`

**PASS** - no file under those roots mentions the forbidden package.

Scan re-executed standalone on 2026-08-01, after the scan was moved out of the benchmark script, by `uv run python -m scripts.check_forbidden_licences` (exit status `0`). It is the only section of this report not produced by the GPU run in the header: unlike sections 1-4 it depends on the working tree and not on the hardware, so re-running it needs no GPU. What stood here before was a string constant asserting that the scan "returns NO matches" - it was printed whether or not anything had been scanned, and planting a real import of the forbidden package under `src/` left it printed unchanged. Both directions of the replacement are now proved by `tests/test_forbidden_licences.py`, and the same violation reaching `main()` fails the run.

## 6. What this probe does NOT establish

- These are pretrained-checkpoint numbers. They must be re-measured on the fine-tuned 3-class weights and section 1 replaced wholesale. Until then no number here may appear in the README without the PROVISIONAL label.
- The accuracy half of DEMO-05 (RF-DETR-Nano trained on the same four arms) is out of scope: this is a latency-only probe.
- Batch-1 latency only, because latency is a batch-1 question (`benchmark.batch_size`). Throughput at larger batches is a different measurement and is not reported here.
- A single run on a desktop GPU that also drives the display, so it is exposed to whatever else the machine was doing. That exposure is not argued away here, it is measured: the contention check in section 1 prints the p95 / median ratio of every measured row in the report against `benchmark.max_p95_to_statistic_ratio`. Run-to-run variation is NOT characterised - that would need repeated runs recorded as artefacts, and this report contains one run.
- Eager PyTorch only. Read the measured resolution response in section 4 before treating any row in section 1 as a statement about the architectures: to the extent latency does not track the pixel count, section 1 is measuring this inference path and not the networks. An export path (`torch.compile`, ONNX or TensorRT) would move the bottleneck and change the ordering; nothing here has been exported.
