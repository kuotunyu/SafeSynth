"""Prediction export must be selectable for a second detector architecture."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import (
    append_derived_metrics,
    bare_head_sweep,
    hard_negative_analysis,
    select_operating_point,
)
from scripts import dump_predictions as dump_driver
from scripts.dump_predictions import (
    PredictionError,
    load_prediction_index,
    parse_args,
    validate_output_isolation,
)


def _drive_path(suffix: str) -> str:
    return "".join(("D", ":/", suffix.lstrip("/")))


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
            _drive_path("runs_rfdetr"),
            "--processor",
            "Roboflow/rf-detr-nano",
            "--index",
            "results/rfdetr_predictions_index.json",
        ]
    )

    assert args.runs_root == Path(_drive_path("runs_rfdetr"))
    assert args.processor == "Roboflow/rf-detr-nano"
    assert args.index == Path("results/rfdetr_predictions_index.json")


def test_nondefault_runs_root_requires_namespaced_outputs() -> None:
    args = parse_args(["--runs-root", _drive_path("runs_rfdetr")])

    with pytest.raises(PredictionError, match="--out-root.*--index"):
        validate_output_isolation(
            args,
            default_runs_root=Path(_drive_path("runs")),
            default_out_root=Path(_drive_path("runs/predictions")),
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


def test_external_prediction_index_round_trips_through_every_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    data_root = tmp_path / "external-data"
    out_root = data_root / "runs" / "predictions"
    index_path = repository / "results" / "predictions_index.json"
    runs_root = data_root / "runs"
    checkpoint = runs_root / "real_only" / "seed_1337" / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    paths = SimpleNamespace(data_root=data_root, runs=runs_root)
    records = [{"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [1, 2, 3, 4]}]

    monkeypatch.setattr(dump_driver, "PROJECT_ROOT", repository)
    monkeypatch.setattr(dump_driver, "load_project_paths", lambda: paths)
    monkeypatch.setattr(dump_driver, "split_samples", lambda paths, split, limit: [object()])
    monkeypatch.setattr(dump_driver, "resolve_checkpoint", lambda seed_dir: checkpoint)
    monkeypatch.setattr(dump_driver, "predict_split", lambda *args, **kwargs: records)

    code = dump_driver.main(
        [
            "--splits",
            "test",
            "--arms",
            "real_only",
            "--out-root",
            str(out_root),
            "--runs-root",
            str(runs_root),
            "--index",
            str(index_path),
        ]
    )

    assert code == 0
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = index["real_only/test/seed_1337"]
    assert entry["path"] == (
        "${SAFESYNTH_DATA_ROOT}/runs/predictions/real_only_test_seed1337.json"
    )
    assert str(data_root) not in index_path.read_text(encoding="utf-8")
    assert bare_head_sweep.load_predictions(entry, data_root=data_root) == records
    assert hard_negative_analysis.load_predictions(
        index, "real_only", "test", data_root=data_root
    ) == records
    assert append_derived_metrics.stored(
        index, "real_only", "test", 1337, data_root=data_root
    ) == records

    monkeypatch.setattr(select_operating_point, "PROJECT_ROOT", repository)
    assert select_operating_point.stored_predictions(
        "real_only",
        1337,
        split="test",
        data_root=data_root,
        index_path=index_path,
    ) == records
