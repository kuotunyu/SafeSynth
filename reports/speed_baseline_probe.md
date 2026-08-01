# Speed baseline probe (DEMO-03 / DEMO-05)

> **PROVISIONAL - NOT A FINAL RESULT.**
> Still measured from a PUBLIC PRETRAINED COCO checkpoint rather than this
> project's fine-tuned 3-class weights (1 of 2): RF-DETR-Nano (`Roboflow/rf-detr-nano`).
> Fine-tuning changes one linear layer at the end of the network - 80 or 91 class
> logits per query become 3 - and nothing else, so those numbers should carry over
> closely. They MUST still be re-measured before any of them reaches the README.

- Generated: `2026-08-01T14:31:06Z`
- Device: `cuda (NVIDIA GeForce RTX 4090)`
- Requested dtype: `float16` -> effective dtype: `float16`
- Warmup / timed iterations: `20` / `200` (warmup discarded, never timed)
- Reported statistic: `median`; p95 also reported: `True`
- Probe image (VALIDATION split, never Test): `hard_hat_workers1000.png` (416x416)
- End-to-end post-processing threshold: `0.07` (`compliance.score_threshold`)

Every setting above is read from `configs/evaluation.yaml`; the harness in `src/evaluation/benchmark.py` holds no tunable number of its own.

## 1. Latency

| Measurement | Device | Batch | Input | dtype | SM clock (MHz) | Median (ms) | p95 (ms) | FPS | Iters (warmup+timed) |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| RT-DETRv2-R18 [model-only] | cuda | 1 | 640 | float16 | 2520 | 27.70 | 41.56 | 36.1 | 20+199 |
| RT-DETRv2-R18 [end-to-end] | cuda | 1 | 640 | float16 | 780 | 33.03 | 53.12 | 30.3 | 20+199 |
| RF-DETR-Nano [model-only] | cuda | 1 | 640 | float16 | 855 | 19.08 | 28.59 | 52.4 | 20+199 |
| RF-DETR-Nano [end-to-end] | cuda | 1 | 640 | float16 | 690 | 28.52 | 30.92 | 35.1 | 20+199 |

`model-only` is a forward pass on a tensor already resident on the device. `end-to-end` additionally includes image preprocessing, the host-to-device copy and `post_process_object_detection`; that is what a user feels. Both are wrapped in `torch.cuda.synchronize()` inside the timed region - without it the timer would measure how fast Python enqueues work.

**The rows above are not a like-for-like comparison.** RT-DETRv2-R18 ran on fine-tuned 3-class weights while RF-DETR-Nano ran on a public pretrained head. The difference is one linear layer over 300 queries, so it is small - but it is not zero, and section 4 already withdraws any claim that these two models can be separated on this measurement at all.

### Contention check (performed, not asserted)

A benchmark taken while the machine was busy shows a long tail: the reported median can look ordinary while p95 blows out. EVERY measured row in this report - the resolution probes in section 4 as well as the headline numbers above - is therefore checked against `benchmark.max_p95_to_statistic_ratio` = `1.3`, and the measured ratio is printed rather than left for the reader to compute.

| Measurement | p95 / median | Verdict |
|---|---:|---|
| RT-DETRv2-R18 [model-only] | 1.50 | CONTENDED - repeat, do not publish |
| RT-DETRv2-R18 [end-to-end] | 1.61 | CONTENDED - repeat, do not publish |
| RF-DETR-Nano [model-only] | 1.50 | CONTENDED - repeat, do not publish |
| RF-DETR-Nano [end-to-end] | 1.08 | ok |
| RT-DETRv2-R18 [model-only @ 320] | 1.39 | CONTENDED - repeat, do not publish |
| RT-DETRv2-R18 [model-only @ 1280] | 1.64 | CONTENDED - repeat, do not publish |
| RF-DETR-Nano [model-only @ 320] | 1.36 | CONTENDED - repeat, do not publish |
| RF-DETR-Nano [model-only @ 1280] | 1.44 | CONTENDED - repeat, do not publish |
| RF-DETR-Nano [model-only @ 384, native preset] | 1.36 | CONTENDED - repeat, do not publish |

**FAIL** - 8 of 9 measured rows exceeded the threshold.

### GPU clock check (the failure the p95 ratio cannot see)

The p95 / statistic ratio is a WITHIN-row test. On 2026-08-01 two runs of this harness, minutes apart and with no code change, reported `11.81` ms and `26.74` ms for the same model - a 2.26x move in which every row scaled together, so the ratios above stayed unremarkable. The SM clocks during those runs were 2520 MHz and 1215 MHz (2.07x). The clock is therefore recorded per row above, and the spread between rows is checked here: rows timed at clocks this far apart are comparing power states rather than networks.

| Lowest (MHz) | Highest (MHz) | Spread | Limit | Verdict |
|---:|---:|---:|---:|---|
| 690 | 2520 | 3.65 | 1.15 | FAIL - re-measure |

## 2. Peak VRAM

| Model | Peak allocated VRAM (MiB) |
|---|---:|
| RT-DETRv2-R18 | 97.7 |
| RF-DETR-Nano | 106.0 |

Measured with `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` immediately before each model's timed region, so the figure covers weights plus activations for that model alone.

## 3. Landing check

| Model | Role | Weights measured | Head | Model class | Params (M) | Native input | Logits | Boxes | Detections |
|---|---|---|---|---|---:|---:|---|---|---:|
| RT-DETRv2-R18 | primary detector (Apache-2.0) | `D:/sdg-data/02-safesynth/runs/real_only/seed_1337/checkpoint-1752` | fine-tuned, 3 classes | `RTDetrV2ForObjectDetection` | 20.08 | 640 | `[1, 300, 3]` | `[1, 300, 4]` | 2 |
| RF-DETR-Nano | speed baseline (Apache-2.0, ADR-005) | `Roboflow/rf-detr-nano` | pretrained, 91 classes | `RfDetrForObjectDetection` | 30.47 | 384 | `[1, 300, 91]` | `[1, 300, 4]` | 83 |

Every model above completed a real forward pass on a real project image; the shapes are read off the returned tensors, not assumed. `Head` is the class list read back off `config.id2label` after loading, and it is what decides the banner at the top of this report. `Detections` counts what survives the post-processing threshold, and carries meaning for this project's classes only on a fine-tuned row.

## 4. Resolution sensitivity: compute-bound or dispatch-bound?

Section 1 runs BOTH models at the config `input_size`, so that comparison is apples-to-apples. This section re-measures model-only latency at other input sizes, which is the only way to tell whether the section-1 numbers characterise the architecture or the dispatch path. The probe runs in BOTH directions - half the configured size and double it - because a flat line downwards on its own is also what a probe that changed nothing would produce. **These rows are not comparable to section 1.**

| Measurement | Device | Batch | Input | dtype | SM clock (MHz) | Median (ms) | p95 (ms) | FPS | Iters (warmup+timed) |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| RT-DETRv2-R18 [model-only @ 320] | cuda | 1 | 320 | float16 | 570 | 25.83 | 35.99 | 38.7 | 20+199 |
| RT-DETRv2-R18 [model-only @ 1280] | cuda | 1 | 1280 | float16 | 1575 | 27.23 | 44.77 | 36.7 | 20+199 |
| RF-DETR-Nano [model-only @ 320] | cuda | 1 | 320 | float16 | 585 | 18.99 | 25.79 | 52.7 | 20+199 |
| RF-DETR-Nano [model-only @ 1280] | cuda | 1 | 1280 | float16 | 2700 | 18.68 | 26.84 | 53.5 | 20+199 |
| RF-DETR-Nano [model-only @ 384, native preset] | cuda | 1 | 384 | float16 | 1185 | 18.80 | 25.58 | 53.2 | 20+199 |

- **RT-DETRv2-R18**: 320x320 costs `25.83` ms against `27.70` ms at 640x640 (`-6.8%`), on 4x fewer input pixels.
- **RT-DETRv2-R18**: 1280x1280 costs `27.23` ms against `27.70` ms at 640x640 (`-1.7%`), on 4x more input pixels.
- **RF-DETR-Nano**: 320x320 costs `18.99` ms against `19.08` ms at 640x640 (`-0.5%`), on 4x fewer input pixels.
- **RF-DETR-Nano**: 1280x1280 costs `18.68` ms against `19.08` ms at 640x640 (`-2.1%`), on 4x more input pixels.

MEASURED: across a 16x span of input pixel counts (320x320 to 1280x1280), the largest move any model-only measurement made was `-6.8%` (RT-DETRv2-R18 at 320x320). Compute-bound behaviour would be roughly -75% at the small end and +300% at the large end.

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
- Files read: `211`
- Matches: `0`

**PASS** - no file under those roots mentions the forbidden package.

## 6. What this probe does NOT establish

- These are pretrained-checkpoint numbers for RF-DETR-Nano. Those rows must be re-measured on fine-tuned 3-class weights, and until they are, no number from them may appear in the README without the PROVISIONAL label.
- The accuracy half of DEMO-05 (RF-DETR-Nano trained on the same four arms) is out of scope: this is a latency-only probe.
- Batch-1 latency only, because latency is a batch-1 question (`benchmark.batch_size`). Throughput at larger batches is a different measurement and is not reported here.
- A single run on a desktop GPU that also drives the display, so it is exposed to whatever else the machine was doing. That exposure is not argued away here, it is measured: the contention check in section 1 prints the p95 / median ratio of every measured row in the report against `benchmark.max_p95_to_statistic_ratio`. Run-to-run variation is NOT characterised - that would need repeated runs recorded as artefacts, and this report contains one run.
- Eager PyTorch only. Read the measured resolution response in section 4 before treating any row in section 1 as a statement about the architectures: to the extent latency does not track the pixel count, section 1 is measuring this inference path and not the networks. An export path (`torch.compile`, ONNX or TensorRT) would move the bottleneck and change the ordering; nothing here has been exported.
