"""Reloadable artifacts keep portable references without losing integrity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.release import public_paths

DATA_ROOT_REFERENCE = "${SAFESYNTH_DATA_ROOT}"


def test_external_runtime_artifact_is_serialized_relative_to_data_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    data_root = tmp_path / "external-data"
    artifact = data_root / "runs" / "checkpoint" / "model.safetensors"

    reference = public_paths.runtime_artifact_reference(
        artifact,
        project_root=repository,
        data_root=data_root,
    )

    assert reference == f"{DATA_ROOT_REFERENCE}/runs/checkpoint/model.safetensors"
    assert str(data_root) not in reference


def test_runtime_reference_resolves_environment_and_verifies_checkpoint_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    data_root = tmp_path / "external-data"
    checkpoint = data_root / "runs" / "experiment" / "best"
    checkpoint.mkdir(parents=True)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"frozen checkpoint")
    expected_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    monkeypatch.setenv("SAFESYNTH_DATA_ROOT", str(data_root))

    resolved = public_paths.resolve_runtime_artifact(
        f"{DATA_ROOT_REFERENCE}/runs/experiment/best",
        project_root=repository,
        expected_kind="directory",
        expected_sha256=expected_sha,
        hash_relative_path="model.safetensors",
    )

    assert resolved == checkpoint.resolve()


def test_runtime_reference_accepts_explicit_artifact_override_for_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SAFESYNTH_DATA_ROOT", raising=False)
    artifact = tmp_path / "operator-selected" / "predictions.json"
    artifact.parent.mkdir()
    artifact.write_text("[]", encoding="utf-8")

    resolved = public_paths.resolve_runtime_artifact(
        public_paths.LOCAL_ONLY_NOT_PUBLISHED,
        project_root=tmp_path / "repository",
        artifact_override=artifact,
        expected_kind="file",
    )

    assert resolved == artifact.resolve()


def test_marker_without_override_fails_with_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError,
        match="local_only_not_published.*explicit artifact override",
    ):
        public_paths.resolve_runtime_artifact(
            public_paths.LOCAL_ONLY_NOT_PUBLISHED,
            project_root=tmp_path,
        )


def test_missing_data_root_and_reference_escape_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SAFESYNTH_DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="SAFESYNTH_DATA_ROOT.*not set"):
        public_paths.resolve_runtime_artifact(
            f"{DATA_ROOT_REFERENCE}/runs/checkpoint",
            project_root=tmp_path,
        )
    with pytest.raises(RuntimeError, match="escapes SAFESYNTH_DATA_ROOT"):
        public_paths.resolve_runtime_artifact(
            f"{DATA_ROOT_REFERENCE}/../outside/checkpoint",
            project_root=tmp_path,
            data_root=tmp_path / "data",
        )


def test_missing_artifact_and_wrong_checkpoint_sha_fail_closed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    checkpoint = data_root / "runs" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"actual")

    with pytest.raises(RuntimeError, match="does not exist"):
        public_paths.resolve_runtime_artifact(
            f"{DATA_ROOT_REFERENCE}/runs/missing",
            project_root=tmp_path / "repository",
            data_root=data_root,
            expected_kind="directory",
        )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        public_paths.resolve_runtime_artifact(
            f"{DATA_ROOT_REFERENCE}/runs/checkpoint",
            project_root=tmp_path / "repository",
            data_root=data_root,
            expected_kind="directory",
            expected_sha256="0" * 64,
            hash_relative_path="model.safetensors",
        )
