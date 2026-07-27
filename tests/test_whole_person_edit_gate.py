from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_whole_person_edit_diagnostic import require_approved_inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    case_dir = tmp_path / "case_01"
    case_dir.mkdir()
    files = {}
    hashes = {}
    for name in ("draft", "edit_mask", "reference"):
        path = case_dir / f"{name}.png"
        path.write_bytes(name.encode())
        files[name] = f"case_01/{name}.png"
        hashes[name] = _sha256(path)
    manifest = {
        "input_manifest_sha256": "fixed-input-sha",
        "cases": [{"files": files, "sha256": hashes}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    config = {
        "model_gate": {
            "allowed": True,
            "review_status": "approved_by_kuotunyu",
            "required_reviewer": "kuotunyu",
            "approved_manifest_sha256": "fixed-input-sha",
        }
    }
    report = {
        "status": "approved_by_kuotunyu",
        "reviewed_by": "kuotunyu",
        "observed_input_issue_count": 0,
        "input_manifest_sha256": "fixed-input-sha",
    }
    return config, report, tmp_path


def test_exact_approved_inputs_unlock_gate(tmp_path: Path) -> None:
    config, report, input_root = _fixture(tmp_path)

    require_approved_inputs(
        config=config,
        report=report,
        input_root=input_root,
    )


def test_changed_input_relocks_gate(tmp_path: Path) -> None:
    config, report, input_root = _fixture(tmp_path)
    (input_root / "case_01" / "draft.png").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="input hash changed"):
        require_approved_inputs(
            config=config,
            report=report,
            input_root=input_root,
        )
