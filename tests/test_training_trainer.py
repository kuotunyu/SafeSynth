"""Checkpoint discovery must never resume from a partially serialized directory."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.trainer import find_resumable_checkpoint


def _complete_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    for name in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / name).write_bytes(b"complete")
    return checkpoint


def test_resume_ignores_a_higher_partial_checkpoint(tmp_path: Path) -> None:
    valid = _complete_checkpoint(tmp_path, 500)
    partial = tmp_path / "checkpoint-1000"
    partial.mkdir()
    (partial / "trainer_state.json").write_text(
        json.dumps({"global_step": 1000}), encoding="utf-8"
    )
    (partial / "model.safetensors").write_bytes(b"partial")

    assert find_resumable_checkpoint(tmp_path) == str(valid)


def test_resume_rejects_mismatched_or_empty_checkpoint_evidence(tmp_path: Path) -> None:
    mismatched = _complete_checkpoint(tmp_path, 500)
    (mismatched / "trainer_state.json").write_text(
        json.dumps({"global_step": 499}), encoding="utf-8"
    )
    empty = _complete_checkpoint(tmp_path, 1000)
    (empty / "optimizer.pt").write_bytes(b"")

    assert find_resumable_checkpoint(tmp_path) is None
