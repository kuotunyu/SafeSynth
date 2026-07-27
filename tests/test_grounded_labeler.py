from __future__ import annotations

from src.synthetic.grounded_labeler import (
    box_iou_xyxy,
    greedy_detection_metrics,
    load_whole_image_config,
)


def test_whole_image_generation_is_locked_before_label_audit() -> None:
    config = load_whole_image_config()

    assert config["generation_gate"]["allowed"] is False
    assert config["labeler"]["license"] == "apache-2.0"
    assert config["labeler"]["required_download_bytes"] == 690_305_545


def test_box_iou_and_greedy_matching() -> None:
    assert box_iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    metrics = greedy_detection_metrics(
        [(0, 0, 10, 10), (20, 20, 30, 30)],
        [
            (0.9, (0, 0, 10, 10)),
            (0.8, (1, 1, 9, 9)),
            (0.7, (20, 20, 30, 30)),
        ],
        score_threshold=0.5,
        match_iou=0.5,
    )

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
