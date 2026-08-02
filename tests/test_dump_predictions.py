"""Prediction export must be selectable for a second detector architecture."""

from __future__ import annotations

from pathlib import Path

from scripts.dump_predictions import parse_args


def test_dump_predictions_accepts_rf_roots_processor_and_index() -> None:
    """Hard-coded RT values would silently export the wrong model's predictions."""

    args = parse_args(
        [
            "--runs-root",
            "D:/runs_rfdetr",
            "--processor",
            "Roboflow/rf-detr-nano",
            "--index",
            "results/rfdetr_predictions_index.json",
        ]
    )

    assert args.runs_root == Path("D:/runs_rfdetr")
    assert args.processor == "Roboflow/rf-detr-nano"
    assert args.index == Path("results/rfdetr_predictions_index.json")
