"""Durable run records are the resume boundary after a process interruption."""

from __future__ import annotations

import json
from pathlib import Path

from src.training.run import write_run_record_atomic


def test_run_record_replaces_atomically_without_leaving_a_temporary_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "run_record.json"
    destination.write_text('{"status":"old"}\n', encoding="utf-8")

    write_run_record_atomic(destination, {"status": "new", "total_steps": 10_900})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "new",
        "total_steps": 10_900,
    }
    assert not (tmp_path / "run_record.json.tmp").exists()
