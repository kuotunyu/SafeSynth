---
license: apache-2.0
library_name: transformers
pipeline_tag: object-detection
base_model: PekingU/rtdetr_v2_r18vd
tags:
  - rt-detr-v2
  - object-detection
  - construction-safety
  - hard-hat
  - safesynth
datasets:
  - kuotunyu/safesynth-hard-hat
---

# SafeSynth RT-DETRv2-R18 Hard-Hat Detector

This is the best validation-selected checkpoint from SafeSynth's controlled
four-arm RT-DETRv2-R18 experiment. The selected arm is deliberately
**`real_only`**: it used all 3,500 frozen real Train images and **zero synthetic
images**. The negative selection result is part of the release, not something
hidden by publishing a synthetic arm.

- Source and reproducibility: [github.com/kuotunyu/SafeSynth](https://github.com/kuotunyu/SafeSynth)
- Synthetic ablation dataset: [kuotunyu/safesynth-hard-hat](https://huggingface.co/datasets/kuotunyu/safesynth-hard-hat)
- Base checkpoint: [PekingU/rtdetr_v2_r18vd](https://huggingface.co/PekingU/rtdetr_v2_r18vd)

## Model details

| Item | Value |
|---|---|
| Architecture | RT-DETRv2 with ResNet-18 backbone |
| Classes | `helmet`, `head`, `person` |
| Training composition | 3,500 real Train, 0 synthetic |
| Validation set | 756 real images |
| Seed | 1337 |
| Optimizer steps | 10,900 |
| Selected checkpoint | step 1,752 (best frozen-validation checkpoint) |
| Input preprocessing | Resize to 640x640, rescale by 1/255, no ImageNet normalization |
| Weight format | Safetensors |

The model predicts a `helmet` box around the helmeted head, a `head` box for a
bare head, and the inherited `person` class. The third class is unreliable
because the source dataset's person annotations are substantially incomplete.

## Four-arm result

All arms used the same real Train set, frozen real Validation/Test sets, seed,
and 10,900-step optimizer budget. Synthetic arms used 3,500 generated images;
their real-image exposure was about half that of the real-only arms.

| Arm | Primary AP_small | Primary mAP | Bare-head recall | Real-image exposures |
|---|---:|---:|---:|---:|
| **real_only** | **0.4511** | **0.5341** | 0.9875 | 49.83 |
| standard_aug | 0.4236 | 0.4958 | 0.9875 | 49.83 |
| unfiltered_syn | 0.3759 | 0.4597 | 0.9898 | 24.91 |
| filtered_syn | 0.3664 | 0.4858 | 0.9886 | 24.91 |

The 1,000-resample, image-level bootstrap intervals for primary AP_small were:

| Arm | AP_small (95% CI) |
|---|---:|
| **real_only** | **0.4511 [0.4307, 0.4753]** |
| standard_aug | 0.4236 [0.3993, 0.4530] |
| unfiltered_syn | 0.3759 [0.3474, 0.4064] |
| filtered_syn | 0.3664 [0.3426, 0.3956] |

`AP_small` is the mean of helmet and bare-head AP for COCO-small objects,
computed in each image's **native original annotation coordinates** (415/416
pixel edges), not after resizing to 640x640. `person` is excluded from the
primary mAP because its source annotations are known to be poor.

## Usage

```python
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

repo_id = "kuotunyu/safesynth-rtdetrv2-r18"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = RTDetrImageProcessor.from_pretrained(repo_id)
model = RTDetrV2ForObjectDetection.from_pretrained(repo_id).to(device).eval()
image = Image.open("construction_site.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.inference_mode():
    outputs = model(**inputs)

target_sizes = torch.tensor([(image.height, image.width)], device=device)
result = processor.post_process_object_detection(
    outputs,
    target_sizes=target_sizes,
    threshold=0.07,
)[0]

for score, label_id, box in zip(
    result["scores"], result["labels"], result["boxes"], strict=True
):
    label = model.config.id2label[int(label_id)]
    print(label, float(score), [round(x, 1) for x in box.tolist()])
```

The 0.07 threshold was selected on the frozen Validation set for the repository's
deployment analysis. Recalibrate it for a new camera, site, class balance, and
false-alarm cost. Never tune it on the Test set.

## Intended use

This checkpoint is intended for reproducible research, teaching, controlled
comparison, and prototyping of hard-hat/bare-head detection. It is not a
certified safety product. A human safety process must remain authoritative.

## Limitations and responsible use

- **Do not treat the reported AP as an absolute quality guarantee.** SHEL5K
  re-annotated the same 5,000 source images with 75,570 labels versus 25,502 in
  the original; all repository claims are relative comparisons on one frozen
  Test set.
- `person` AP is not a reliable measure because the upstream person labels are
  sparse and inconsistent.
- Training and bootstrap evaluation used one seed. This checkpoint is the
  validation-selected model for that protocol, not evidence of universal
  superiority.
- The pre-registered synthetic-artifact gate failed (AUC 0.9053 versus a maximum
  of 0.60). Synthetic augmentation did not robustly improve RT-DETRv2 and showed
  architecture-sensitive, inconclusive behavior in an RF-DETR cross-check.
- Data are drawn from a public construction-worker dataset and may not represent
  every geography, PPE design, skin tone, workplace, lighting condition, camera,
  or occlusion pattern.
- Do not use the model as the sole basis for discipline, access denial,
  surveillance, or any decision that can harm a worker.

## License

The weights are a fine-tune of `PekingU/rtdetr_v2_r18vd`, whose Hugging Face
repository declares **Apache-2.0**. These derived weights are released under
Apache-2.0. SafeSynth source code is MIT; the immediate training dataset is
CC0 1.0. See the linked repositories for the full provenance and notices.
