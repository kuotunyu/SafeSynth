"""Supervised checkpoint registrations remain reloadable after publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_supervised_labeler_v12_model_audit as audit_driver


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v12_registration_resolves_external_checkpoint_from_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "repository"
    data_root = tmp_path / "external-data"
    checkpoint = data_root / "runs" / "supervised-labeler" / "best"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"frozen-v11-checkpoint")
    report_path = project_root / "reports" / "v11.json"
    report_path.parent.mkdir(parents=True)
    report = {
        "status": "supervised_labeler_audit_passed",
        "checkpoint_sha256": _sha256(weights),
        "best_calibration": {"threshold": 0.03},
        "postprocessing": {"max_relative_area": 0.08},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    registration = {
        "status": "frozen_before_v12_model_inference",
        "source_experiment": "supervised_labeler_v11",
        "source_training_report": "reports/v11.json",
        "source_training_report_sha256": _sha256(report_path),
        "checkpoint_path": (
            "${SAFESYNTH_DATA_ROOT}/runs/supervised-labeler/best"
        ),
        "checkpoint_sha256": _sha256(weights),
        "score_threshold": 0.03,
        "postprocessing": {"max_relative_area": 0.08},
        "whole_image_generation_run": False,
    }
    monkeypatch.setattr(audit_driver, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("SAFESYNTH_DATA_ROOT", str(data_root))

    observed, source_report, resolved = audit_driver._verified_registration(
        {"model_audit_registration": registration}
    )

    assert observed == registration
    assert source_report == report
    assert resolved == checkpoint.resolve()
