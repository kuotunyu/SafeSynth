"""Dataset, augmentation and collation for RT-DETRv2 fine-tuning.

The coordinate convention is the whole story here. ADR-014 verified by execution
that the HF image processor converts COCO [x, y, w, h] absolute into normalized
cxcywh, which is exactly what the loss wants. So:

    albumentations stays in COCO absolute xywh from end to end
    the image processor performs the one and only conversion
    nothing in between ever touches cxcywh

Doing that conversion by hand, or twice, does not raise. It produces a model
that trains to a plausible-looking loss curve and detects nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ("helmet", "head", "person")
# TRAIN's "category id remapped to 0..K-1" rule. A non-contiguous COCO
# category_id is the classic cause of "loss falls nicely but mAP is near zero".
NAME_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


class AnnotationConventionError(RuntimeError):
    """Raised when boxes are not in the convention this pipeline requires."""


@dataclass(frozen=True)
class Sample:
    image_path: Path
    image_id: int
    boxes_xywh: tuple[tuple[float, float, float, float], ...]
    class_indices: tuple[int, ...]
    is_synthetic: bool


def load_coco_samples(
    annotations_path: Path,
    images_root: Path,
    *,
    keep_names: Sequence[str] | None = None,
    is_synthetic: bool = False,
) -> list[Sample]:
    """Read a COCO file into samples, remapping category ids to 0..K-1."""

    import json

    payload = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    id_to_name = {int(c["id"]): str(c["name"]) for c in payload["categories"]}
    unknown = set(id_to_name.values()) - set(CLASS_NAMES)
    if unknown:
        raise AnnotationConventionError(f"Unexpected categories: {sorted(unknown)}")

    allow = set(keep_names) if keep_names is not None else None
    images = {}
    for image in payload["images"]:
        name = image["file_name"].split("/")[-1]
        if allow is None or name in allow:
            images[int(image["id"])] = name

    grouped: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in grouped:
            grouped[image_id].append(annotation)

    samples: list[Sample] = []
    for image_id, name in sorted(images.items(), key=lambda item: item[1]):
        boxes: list[tuple[float, float, float, float]] = []
        classes: list[int] = []
        for annotation in sorted(grouped[image_id], key=lambda a: int(a["id"])):
            x, y, w, h = (float(v) for v in annotation["bbox"])
            if w <= 0 or h <= 0:
                continue
            boxes.append((x, y, w, h))
            classes.append(NAME_TO_INDEX[id_to_name[int(annotation["category_id"])]])
        samples.append(
            Sample(
                image_path=Path(images_root) / name,
                image_id=image_id,
                boxes_xywh=tuple(boxes),
                class_indices=tuple(classes),
                is_synthetic=is_synthetic,
            )
        )
    return samples


def build_transform(profile: str, config: Mapping[str, Any]):
    """Albumentations pipeline in COCO absolute xywh (TRAIN-05).

    `real_only` gets nothing at all. `standard_aug` gets geometry plus a
    photometric block whose ranges track configs/compose.yaml, so the baseline
    has seen the same kind of degradation the synthetic images carry.
    """

    import albumentations as A

    settings = config["augmentation"]
    bbox_params = A.BboxParams(
        format=settings["bbox_format"],
        label_fields=["class_indices"],
        clip=bool(settings["bbox_clip"]),
        min_area=float(settings["bbox_min_area"]),
    )

    if profile == "real_only":
        return A.Compose([], bbox_params=bbox_params)

    arm = settings["standard_aug"]
    gamma_low, gamma_high = (float(v) for v in arm["gamma_range"])
    noise_low, noise_high = (float(v) for v in arm["gauss_noise_sigma"])
    jpeg_low, jpeg_high = (int(v) for v in arm["jpeg_quality"])
    return A.Compose(
        [
            A.HorizontalFlip(p=float(arm["horizontal_flip"])),
            A.Perspective(p=float(arm["perspective"])),
            A.RandomBrightnessContrast(p=float(arm["random_brightness_contrast"])),
            A.HueSaturationValue(p=float(arm["hue_saturation_value"])),
            # RandomGamma takes PERCENT. Passing 1.81 here is a silent no-op,
            # which is why the config comment shouts about it.
            A.RandomGamma(gamma_limit=(gamma_low * 100, gamma_high * 100), p=0.5),
            A.GaussNoise(std_range=(noise_low / 255.0, noise_high / 255.0), p=0.3),
            A.MotionBlur(blur_limit=int(arm["blur_limit"]), p=float(arm["motion_blur"])),
            A.ImageCompression(quality_range=(jpeg_low, jpeg_high), p=0.2),
        ],
        bbox_params=bbox_params,
    )


class DetectionDataset(Dataset):
    """Applies augmentation in COCO xywh, then hands the processor the conversion."""

    def __init__(self, samples: Sequence[Sample], processor, transform=None) -> None:
        self.samples = list(samples)
        self.processor = processor
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        boxes = [list(box) for box in sample.boxes_xywh]
        classes = list(sample.class_indices)

        if self.transform is not None:
            augmented = self.transform(
                image=image, bboxes=boxes, class_indices=classes
            )
            image = augmented["image"]
            boxes = [list(box) for box in augmented["bboxes"]]
            classes = list(augmented["class_indices"])

        annotations = {
            "image_id": sample.image_id,
            "annotations": [
                {
                    "image_id": sample.image_id,
                    "category_id": int(category),
                    "bbox": [float(v) for v in box],
                    "area": float(box[2]) * float(box[3]),
                    "iscrowd": 0,
                }
                for box, category in zip(boxes, classes, strict=True)
            ],
        }
        encoded = self.processor(
            images=image, annotations=annotations, return_tensors="pt"
        )
        return {
            "pixel_values": encoded["pixel_values"][0],
            "labels": encoded["labels"][0],
        }


def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """`labels` stays a list, never a stacked tensor.

    Per-image label dicts have different lengths. Stacking them is what
    `eval_do_concat_batches=False` exists to prevent on the eval side; the
    collator has to respect the same shape on the train side.
    """

    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": [item["labels"] for item in batch],
    }


def assert_eval_batch_remainder_is_safe(n_val: int, batch_size: int) -> None:
    """A final eval batch of exactly one crashes collect_targets.

    image_size collapses from tensor([H, W]) to tensor([H]) and unpacking raises
    "not enough values to unpack (expected 2, got 1)". Confirmed upstream; one
    assertion at startup is cheaper than discovering it after an epoch of
    training on Colab.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if n_val % batch_size == 1:
        raise AnnotationConventionError(
            f"Validation size {n_val} leaves a remainder of exactly 1 at eval batch "
            f"size {batch_size}, which crashes collect_targets. Change the batch size "
            f"or move one image between splits."
        )
