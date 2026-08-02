"""Prediction export must be selectable for a second detector architecture."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import dump_predictions as dump_driver
from scripts.dump_predictions import (
    PredictionError,
    load_prediction_index,
    parse_args,
    validate_output_isolation,
)


def test_prediction_json_is_flushed_before_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(
        dump_driver.os, "fsync", lambda descriptor: fsync_calls.append(descriptor)
    )

    dump_driver.atomic_write_json_value(tmp_path / "predictions.json", {"ok": True})

    assert fsync_calls


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


def test_nondefault_runs_root_requires_namespaced_outputs() -> None:
    args = parse_args(["--runs-root", "D:/runs_rfdetr"])

    with pytest.raises(PredictionError, match="--out-root.*--index"):
        validate_output_isolation(
            args,
            default_runs_root=Path("D:/runs"),
            default_out_root=Path("D:/runs/predictions"),
            default_index=Path(args.index),
        )


def test_nondefault_runs_root_rejects_aliases_of_primary_outputs(
    tmp_path: Path,
) -> None:
    primary_out = tmp_path / "predictions"
    primary_index = tmp_path / "predictions_index.json"
    args = parse_args(
        [
            "--runs-root",
            str(tmp_path / "runs_rfdetr"),
            "--out-root",
            str(primary_out / "unused" / ".."),
            "--index",
            str(primary_index.parent / "unused" / ".." / primary_index.name),
        ]
    )

    with pytest.raises(PredictionError, match="--out-root.*--index"):
        validate_output_isolation(
            args,
            default_runs_root=tmp_path / "runs",
            default_out_root=primary_out,
            default_index=primary_index,
        )


def test_corrupt_prediction_index_is_rejected_instead_of_discarded(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(PredictionError, match="prediction index"):
        load_prediction_index(index)
