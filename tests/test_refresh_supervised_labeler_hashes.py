"""Regression tests for portable-path frozen-evidence hash refreshes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import refresh_supervised_labeler_hashes as refresh
from src.synthetic.whole_image import canonical_mapping_sha256

SOURCE_PATH = "reports/supervised_labeler_v13_training.json"
MANIFEST_PATH = "reports/supervised_labeler_v13_model_review_manifest.json"
CONFIG_PATH = "configs/supervised_labeler_v13.yaml"


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(root: Path, path: str, value: object) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit_fixture(root: Path) -> None:
    _git(root, "init", "--quiet")
    _git(root, "config", "core.autocrlf", "false")
    source = {
        "checkpoint_path": "D:\\private-data\\runs\\v13\\best",
        "metrics": {"precision": 0.875, "recall": 0.8125},
        "sample_count": 64,
        "seed": 20260809,
    }
    _write_json(root, SOURCE_PATH, source)

    manifest = {
        "checkpoint_path": "D:\\private-data\\runs\\v13\\best",
        "metrics": {"false_negatives": 3, "true_positives": 21},
        "source_training_report_sha256": _sha256(root / SOURCE_PATH),
    }
    manifest["manifest_sha256"] = canonical_mapping_sha256(manifest)
    _write_json(root, MANIFEST_PATH, manifest)

    config = {
        "model_review_registration": {
            "manifest_file_sha256": _sha256(root / MANIFEST_PATH),
            "manifest_sha256": manifest["manifest_sha256"],
        }
    }
    config_target = root / CONFIG_PATH
    config_target.parent.mkdir(parents=True, exist_ok=True)
    config_target.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )


def _scrub_paths(root: Path) -> None:
    for relative in (SOURCE_PATH, MANIFEST_PATH):
        path = root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["checkpoint_path"] = "local_only_not_published"
        _write_json(root, relative, payload)


def _refresh(root: Path, *, dry_run: bool) -> refresh.RefreshResult:
    return refresh.refresh_hash_chain(
        root,
        scrubbed_files=(SOURCE_PATH, MANIFEST_PATH),
        allowed_upper_files=(CONFIG_PATH,),
        dry_run=dry_run,
    )


def test_dry_run_reports_the_dependency_chain_without_writing(tmp_path: Path) -> None:
    _commit_fixture(tmp_path)
    _scrub_paths(tmp_path)
    before = {
        path: (tmp_path / path).read_bytes()
        for path in (SOURCE_PATH, MANIFEST_PATH, CONFIG_PATH)
    }

    result = _refresh(tmp_path, dry_run=True)

    assert result.changed_files == (MANIFEST_PATH, CONFIG_PATH)
    assert result.change_count == 4
    assert {
        path: (tmp_path / path).read_bytes()
        for path in (SOURCE_PATH, MANIFEST_PATH, CONFIG_PATH)
    } == before


def test_refresh_updates_only_hashes_and_preserves_frozen_evidence(
    tmp_path: Path,
) -> None:
    _commit_fixture(tmp_path)
    base_source = json.loads(_git(tmp_path, "show", f"HEAD:{SOURCE_PATH}"))
    base_manifest = json.loads(_git(tmp_path, "show", f"HEAD:{MANIFEST_PATH}"))
    _scrub_paths(tmp_path)

    result = _refresh(tmp_path, dry_run=False)

    source = json.loads((tmp_path / SOURCE_PATH).read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / MANIFEST_PATH).read_text(encoding="utf-8"))
    config = yaml.safe_load((tmp_path / CONFIG_PATH).read_text(encoding="utf-8"))
    embedded_manifest_sha = manifest.pop("manifest_sha256")
    assert result.changed_files == (MANIFEST_PATH, CONFIG_PATH)
    assert source["metrics"] == base_source["metrics"]
    assert source["sample_count"] == base_source["sample_count"]
    assert source["seed"] == base_source["seed"]
    assert manifest["metrics"] == base_manifest["metrics"]
    assert manifest["source_training_report_sha256"] == _sha256(
        tmp_path / SOURCE_PATH
    )
    assert embedded_manifest_sha == canonical_mapping_sha256(manifest)
    assert config["model_review_registration"]["manifest_sha256"] == (
        embedded_manifest_sha
    )
    assert config["model_review_registration"]["manifest_file_sha256"] == (
        _sha256(tmp_path / MANIFEST_PATH)
    )


def test_refresh_is_idempotent_for_dry_run_and_formal_run(tmp_path: Path) -> None:
    _commit_fixture(tmp_path)
    _scrub_paths(tmp_path)
    _refresh(tmp_path, dry_run=False)
    before = _git(tmp_path, "diff")

    dry_run = _refresh(tmp_path, dry_run=True)
    formal_run = _refresh(tmp_path, dry_run=False)

    assert dry_run.change_count == 0
    assert formal_run.change_count == 0
    assert _git(tmp_path, "diff") == before


def test_refresh_fails_closed_when_non_path_evidence_changed(tmp_path: Path) -> None:
    _commit_fixture(tmp_path)
    _scrub_paths(tmp_path)
    source = json.loads((tmp_path / SOURCE_PATH).read_text(encoding="utf-8"))
    source["metrics"]["precision"] = 0.5
    _write_json(tmp_path, SOURCE_PATH, source)

    with pytest.raises(refresh.HashRefreshError, match="non-path/hash field"):
        _refresh(tmp_path, dry_run=True)


def test_refresh_fails_closed_for_an_unknown_hash_reference_schema(
    tmp_path: Path,
) -> None:
    _commit_fixture(tmp_path)
    old_source_sha = _sha256(tmp_path / SOURCE_PATH)
    _write_json(
        tmp_path,
        "reports/unknown_registration.json",
        {"mystery_value": old_source_sha},
    )
    _git(tmp_path, "add", "reports/unknown_registration.json")
    _git(
        tmp_path,
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "unknown schema",
    )
    _scrub_paths(tmp_path)

    with pytest.raises(refresh.HashRefreshError, match="unknown hash reference"):
        _refresh(tmp_path, dry_run=True)
