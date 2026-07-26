"""SAM2 box-prompt inference for full images and resolution-spike crops."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from src.synthetic.mask_ops import clean_and_measure_mask


@dataclass(frozen=True)
class CropTransform:
    """Mapping between a global image and a square region on a 1024 canvas."""

    crop_left: int
    crop_top: int
    crop_side: int
    target_size: int
    canvas_offset: int
    image_width: int
    image_height: int


@dataclass
class MaskPrediction:
    """One post-processed mask plus confidence and deterministic QC metrics."""

    mask: np.ndarray
    iou_score: float
    object_score_logit: float
    metrics: dict[str, float | int]


def xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def build_crop_canvas(
    image: Image.Image,
    box_xyxy: list[float] | tuple[float, ...],
    *,
    context_pad_frac: float,
    min_crop_side_px: int,
    target_size: int,
    canvas_size: int = 1024,
) -> tuple[Image.Image, list[float], CropTransform]:
    """Upscale a contextual square crop and, for 512 mode, centre-pad to 1024."""

    if target_size not in {512, 1024}:
        raise ValueError(f"Expected target_size 512 or 1024, got {target_size}")
    if canvas_size < target_size:
        raise ValueError("canvas_size cannot be smaller than target_size")

    rgb = image.convert("RGB")
    image_width, image_height = rgb.size
    x1, y1, x2, y2 = (float(value) for value in box_xyxy)
    object_side = max(x2 - x1, y2 - y1)
    crop_side = int(np.ceil(max(min_crop_side_px, object_side * (1 + 2 * context_pad_frac))))
    crop_side = max(crop_side, 2)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    crop_left = int(np.floor(center_x - crop_side / 2))
    crop_top = int(np.floor(center_y - crop_side / 2))
    crop = rgb.crop(
        (crop_left, crop_top, crop_left + crop_side, crop_top + crop_side)
    ).resize((target_size, target_size), Image.Resampling.BICUBIC)

    crop_array = np.asarray(crop)
    canvas_offset = (canvas_size - target_size) // 2
    if target_size == canvas_size:
        canvas_array = crop_array
    else:
        pad_before = canvas_offset
        pad_after = canvas_size - target_size - canvas_offset
        canvas_array = np.pad(
            crop_array,
            ((pad_before, pad_after), (pad_before, pad_after), (0, 0)),
            mode="edge",
        )

    scale = target_size / crop_side
    prompt = [
        (x1 - crop_left) * scale + canvas_offset,
        (y1 - crop_top) * scale + canvas_offset,
        (x2 - crop_left) * scale + canvas_offset,
        (y2 - crop_top) * scale + canvas_offset,
    ]
    prompt = [float(np.clip(value, 0, canvas_size)) for value in prompt]
    transform = CropTransform(
        crop_left=crop_left,
        crop_top=crop_top,
        crop_side=crop_side,
        target_size=target_size,
        canvas_offset=canvas_offset,
        image_width=image_width,
        image_height=image_height,
    )
    return Image.fromarray(canvas_array), prompt, transform


def crop_mask_to_global(mask: np.ndarray, transform: CropTransform) -> np.ndarray:
    """Map a 1024-canvas boolean mask back to the original image."""

    offset = transform.canvas_offset
    target = transform.target_size
    crop_mask = mask[offset : offset + target, offset : offset + target].astype(np.uint8)
    restored = np.asarray(
        Image.fromarray(crop_mask * 255).resize(
            (transform.crop_side, transform.crop_side), Image.Resampling.NEAREST
        )
    ) > 0
    result = np.zeros((transform.image_height, transform.image_width), dtype=bool)

    source_left = max(0, -transform.crop_left)
    source_top = max(0, -transform.crop_top)
    destination_left = max(0, transform.crop_left)
    destination_top = max(0, transform.crop_top)
    copy_width = min(
        transform.crop_side - source_left, transform.image_width - destination_left
    )
    copy_height = min(
        transform.crop_side - source_top, transform.image_height - destination_top
    )
    if copy_width > 0 and copy_height > 0:
        result[
            destination_top : destination_top + copy_height,
            destination_left : destination_left + copy_width,
        ] = restored[
            source_top : source_top + copy_height,
            source_left : source_left + copy_width,
        ]
    return result


class Sam2BoxSegmenter:
    """A small deterministic wrapper around the Transformers SAM2 API."""

    def __init__(
        self,
        *,
        model_id: str,
        dtype: str = "bfloat16",
        device: str | None = None,
        morph_close_kernel: int = 3,
        morph_close_iterations: int = 1,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            dtype = "float32"
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.dtype = dtype_map[dtype]
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id, dtype=self.dtype)
        self.model.to(self.device).eval()
        self.morph_close_kernel = morph_close_kernel
        self.morph_close_iterations = morph_close_iterations

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    def _predict(
        self, image: Image.Image, boxes_xyxy: list[list[float]]
    ) -> tuple[list[np.ndarray], list[float], list[float]]:
        if not boxes_xyxy:
            return [], [], []
        inputs = self.processor(
            images=image.convert("RGB"),
            input_boxes=[boxes_xyxy],
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode(), self._autocast():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
        )[0][:, 0]
        iou_scores = outputs.iou_scores.detach().float().cpu()[0, :, 0].tolist()
        object_scores = (
            outputs.object_score_logits.detach().float().cpu()[0, :, 0].tolist()
        )
        return [mask.numpy().astype(bool) for mask in masks], iou_scores, object_scores

    def _finish(
        self,
        raw_masks: list[np.ndarray],
        boxes_xyxy: list[list[float]],
        iou_scores: list[float],
        object_scores: list[float],
    ) -> list[MaskPrediction]:
        predictions: list[MaskPrediction] = []
        for raw, box, iou_score, object_score in zip(
            raw_masks, boxes_xyxy, iou_scores, object_scores, strict=True
        ):
            cleaned, metrics = clean_and_measure_mask(
                raw,
                box,
                morph_close_kernel=self.morph_close_kernel,
                morph_close_iterations=self.morph_close_iterations,
            )
            metrics["iou_score"] = float(iou_score)
            metrics["object_score_logit"] = float(object_score)
            predictions.append(
                MaskPrediction(
                    mask=cleaned,
                    iou_score=float(iou_score),
                    object_score_logit=float(object_score),
                    metrics=metrics,
                )
            )
        return predictions

    def predict_full(
        self, image: Image.Image, boxes_xyxy: list[list[float]]
    ) -> list[MaskPrediction]:
        """Prompt every box together so the full-image embedding is computed once."""

        raw_masks, iou_scores, object_scores = self._predict(image, boxes_xyxy)
        return self._finish(raw_masks, boxes_xyxy, iou_scores, object_scores)

    def predict_crop(
        self,
        image: Image.Image,
        box_xyxy: list[float],
        *,
        context_pad_frac: float,
        min_crop_side_px: int,
        target_size: int,
    ) -> MaskPrediction:
        """Predict one contextual crop at effective 512 or 1024 resolution."""

        canvas, prompt, transform = build_crop_canvas(
            image,
            box_xyxy,
            context_pad_frac=context_pad_frac,
            min_crop_side_px=min_crop_side_px,
            target_size=target_size,
        )
        masks, iou_scores, object_scores = self._predict(canvas, [prompt])
        global_mask = crop_mask_to_global(masks[0], transform)
        return self._finish(
            [global_mask], [list(box_xyxy)], iou_scores, object_scores
        )[0]

