"""COCOeval wiring: the failure modes here are all silent-zero, not exceptions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.training.metrics import (
    COCO_STAT_NAMES,
    EvalStructureError,
    build_coco_ground_truth,
    evaluate_detections,
    extract_logits_and_boxes,
    predictions_to_coco,
)

CLASSES = ("helmet", "head", "person")


@dataclass(frozen=True)
class FakeSample:
    image_id: int
    boxes_xywh: tuple
    class_indices: tuple


def test_ground_truth_keeps_coco_xywh_and_contiguous_categories() -> None:
    samples = [
        FakeSample(1, ((10.0, 20.0, 30.0, 40.0),), (0,)),
        FakeSample(2, ((5.0, 5.0, 10.0, 10.0), (50.0, 50.0, 20.0, 20.0)), (1, 2)),
    ]

    gt = build_coco_ground_truth(samples, CLASSES)

    assert [c["id"] for c in gt["categories"]] == [0, 1, 2]
    assert len(gt["annotations"]) == 3
    assert gt["annotations"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert gt["annotations"][0]["area"] == pytest.approx(1200.0)
    assert {a["id"] for a in gt["annotations"]} == {1, 2, 3}


def test_perfect_predictions_score_one() -> None:
    """The self-evaluation check: feeding the ground truth back must give AP 1."""

    samples = [FakeSample(1, ((10.0, 20.0, 30.0, 40.0),), (0,))]
    gt = build_coco_ground_truth(samples, CLASSES)
    detections = [
        {"image_id": 1, "category_id": 0, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.9}
    ]

    stats = evaluate_detections(gt, detections)

    assert stats["map"] == pytest.approx(1.0)
    assert set(stats) == set(COCO_STAT_NAMES)


def test_empty_detections_return_zeros_instead_of_crashing() -> None:
    """An untrained model predicts nothing; the smoke test must still finish."""

    gt = build_coco_ground_truth([FakeSample(1, ((1.0, 1.0, 2.0, 2.0),), (0,))], CLASSES)

    stats = evaluate_detections(gt, [])

    assert stats["map"] == 0.0
    assert set(stats) == set(COCO_STAT_NAMES)


def test_post_process_xyxy_is_converted_to_coco_xywh() -> None:
    """post_process emits xyxy, COCO wants xywh; conversion is not optional."""

    processed = [
        {
            "boxes": torch.tensor([[10.0, 20.0, 40.0, 60.0]]),
            "scores": torch.tensor([0.8]),
            "labels": torch.tensor([1]),
        }
    ]

    detections = predictions_to_coco(processed, [7])

    assert len(detections) == 1
    detection = detections[0]
    assert detection["image_id"] == 7
    assert detection["category_id"] == 1
    assert detection["bbox"] == [10.0, 20.0, 30.0, 40.0]
    # float32 round-trips 0.8 as 0.800000011920929.
    assert detection["score"] == pytest.approx(0.8)


def test_degenerate_boxes_are_dropped() -> None:
    processed = [
        {
            "boxes": torch.tensor([[10.0, 20.0, 10.0, 60.0], [1.0, 1.0, 5.0, 5.0]]),
            "scores": torch.tensor([0.9, 0.7]),
            "labels": torch.tensor([0, 2]),
        }
    ]

    detections = predictions_to_coco(processed, [3])

    assert len(detections) == 1
    assert detections[0]["category_id"] == 2


def _fake_eval_batch(batch_size: int = 2, num_labels: int = 3) -> tuple:
    """The measured transformers 5.14.1 layout: 14-tuple, loss dict at index 0.

    This exact shape is what killed the first Colab run, so it is pinned here.
    """

    import numpy as np

    loss_dict = {
        "loss_vfl": np.array(1.0),
        "loss_bbox": np.array(2.0),
        "loss_giou": np.array(3.0),
    }
    logits = np.zeros((batch_size, 300, num_labels), dtype=np.float32)
    boxes = np.zeros((batch_size, 300, 4), dtype=np.float32)
    filler = [np.zeros((batch_size, 4), dtype=np.float32) for _ in range(11)]
    return (loss_dict, logits, boxes, *filler)


# spec: TRAIN-01
def test_logits_and_boxes_are_found_past_the_loss_dict() -> None:
    """Index 0 is the loss dict, not the logits. Positional access broke on this."""

    logits, boxes = extract_logits_and_boxes(_fake_eval_batch(), num_labels=3)

    assert logits.shape == (2, 300, 3)
    assert boxes.shape == (2, 300, 4)


def test_extraction_survives_a_reordered_tuple() -> None:
    """Selection is by trailing dimension, so layout changes do not break it."""

    import numpy as np

    logits = np.zeros((2, 300, 3), dtype=np.float32)
    boxes = np.zeros((2, 300, 4), dtype=np.float32)
    reordered = (boxes, {"loss": np.array(1.0)}, logits)

    found_logits, found_boxes = extract_logits_and_boxes(reordered, num_labels=3)

    assert found_logits.shape == (2, 300, 3)
    assert found_boxes.shape == (2, 300, 4)


def test_missing_logits_raises_instead_of_producing_zero_map() -> None:
    """A silent zero mAP would read as 'the model learned nothing'."""

    import numpy as np

    with pytest.raises(EvalStructureError, match="Could not locate"):
        extract_logits_and_boxes(
            ({"loss": np.array(1.0)}, np.zeros((2, 4), dtype=np.float32)),
            num_labels=3,
        )


def test_four_class_model_would_not_confuse_boxes_with_logits() -> None:
    """With num_labels=4 both arrays end in 4; the first match must win in order."""

    import numpy as np

    logits = np.zeros((2, 300, 4), dtype=np.float32)
    boxes = np.zeros((2, 300, 4), dtype=np.float32)

    found_logits, found_boxes = extract_logits_and_boxes(
        ({"loss": np.array(0.0)}, logits, boxes), num_labels=4
    )

    assert found_logits is not None
    assert found_boxes is not None
