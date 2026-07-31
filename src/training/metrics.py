"""COCO mAP for the Trainer eval loop, via pycocotools.

pycocotools rather than torchmetrics on purpose: it is already a dependency, the
compositor already self-evaluates with COCOeval, and Colab then needs no extra
install that could resolve to a different version than the local smoke test ran
against. Same evaluator locally and remotely is worth more than convenience.

AP_small is one of the two headline metrics for this project, so it is pulled
out by name rather than left inside the twelve-number COCOeval dump.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from typing import Any

import numpy as np

# COCOeval's stats vector is positional and undocumented at the call site.
COCO_STAT_NAMES = (
    "map",           # AP @ IoU 0.50:0.95, all areas
    "map_50",
    "map_75",
    "map_small",     # headline metric #1
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)


def build_coco_ground_truth(
    samples: Sequence[Any], class_names: Sequence[str]
) -> dict[str, Any]:
    """A COCO dict from the same Sample objects the dataset iterates."""

    images, annotations = [], []
    annotation_id = 1
    for sample in samples:
        images.append({"id": int(sample.image_id), "width": 0, "height": 0})
        for box, category in zip(sample.boxes_xywh, sample.class_indices, strict=True):
            x, y, w, h = (float(v) for v in box)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(sample.image_id),
                    "category_id": int(category),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index, "name": name} for index, name in enumerate(class_names)
        ],
    }


def evaluate_detections(
    ground_truth: dict[str, Any], detections: Sequence[dict[str, Any]]
) -> dict[str, float]:
    """Run COCOeval and return the stats vector under readable names."""

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        # An untrained model legitimately predicts nothing above threshold at
        # step 1. Returning zeros keeps the smoke test meaningful instead of
        # crashing inside pycocotools on an empty result set.
        return dict.fromkeys(COCO_STAT_NAMES, 0.0)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ground_truth
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes([dict(d) for d in detections])
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = np.asarray(evaluator.stats, dtype=float)
    return {
        name: float(stats[index]) if index < len(stats) and np.isfinite(stats[index]) else 0.0
        for index, name in enumerate(COCO_STAT_NAMES)
    }


def predictions_to_coco(
    processed: Sequence[dict[str, Any]], image_ids: Sequence[int]
) -> list[dict[str, Any]]:
    """Convert post_process_object_detection output into COCO detections.

    post_process returns xyxy in the original image scale; COCO wants xywh.
    """

    detections: list[dict[str, Any]] = []
    for result, image_id in zip(processed, image_ids, strict=True):
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy()
        for (x0, y0, x1, y1), score, label in zip(boxes, scores, labels, strict=True):
            width, height = float(x1) - float(x0), float(y1) - float(y0)
            if width <= 0 or height <= 0:
                continue
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": [float(x0), float(y0), width, height],
                    "score": float(score),
                }
            )
    return detections
