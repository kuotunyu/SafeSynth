"""Portable path serialization for public repository artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Literal

LOCAL_ONLY_NOT_PUBLISHED = "local_only_not_published"
DATA_ROOT_ENVIRONMENT_VARIABLE = "SAFESYNTH_DATA_ROOT"
DATA_ROOT_REFERENCE = "${SAFESYNTH_DATA_ROOT}"


class PortablePathError(RuntimeError):
    """Raised when a reloadable public artifact reference is unsafe or unusable."""


def _relative_to(candidate: Path, root: Path) -> Path | None:
    try:
        return candidate.relative_to(root)
    except ValueError:
        return None


def public_artifact_path(path: Path | str, *, project_root: Path | str) -> str:
    """Return a POSIX repository path, or a marker for local-only inputs.

    Public evidence must not encode the checkout location. A path below the
    repository root therefore becomes repository-relative; a path outside it
    cannot be made portable and is represented by an explicit non-path marker.
    """

    root = Path(project_root).resolve()
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return LOCAL_ONLY_NOT_PUBLISHED


def runtime_artifact_reference(
    path: Path | str,
    *,
    project_root: Path | str,
    data_root: Path | str,
) -> str:
    """Serialize a reloadable artifact without publishing a machine root."""

    repository = Path(project_root).resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    candidate = candidate.resolve()
    data = Path(data_root).resolve()
    repository_relative = _relative_to(candidate, repository)
    if repository_relative is not None:
        return repository_relative.as_posix()
    data_relative = _relative_to(candidate, data)
    if data_relative is not None:
        suffix = data_relative.as_posix()
        return DATA_ROOT_REFERENCE if suffix == "." else f"{DATA_ROOT_REFERENCE}/{suffix}"
    raise PortablePathError(
        f"reloadable artifact is outside project_root and {DATA_ROOT_ENVIRONMENT_VARIABLE}"
    )


def _explicit_path(value: Path | str, *, project_root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _data_root(value: Path | str | None) -> Path:
    configured = value if value is not None else os.environ.get(DATA_ROOT_ENVIRONMENT_VARIABLE)
    if not configured:
        raise PortablePathError(
            f"{DATA_ROOT_ENVIRONMENT_VARIABLE} is not set and no explicit data root was provided"
        )
    root = Path(configured)
    if not root.is_absolute():
        raise PortablePathError(f"{DATA_ROOT_ENVIRONMENT_VARIABLE} must be an absolute path")
    return root.resolve()


def resolve_runtime_artifact(
    reference: Path | str,
    *,
    project_root: Path | str,
    data_root: Path | str | None = None,
    artifact_override: Path | str | None = None,
    expected_kind: Literal["file", "directory"] | None = None,
    expected_sha256: str | None = None,
    hash_relative_path: Path | str | None = None,
) -> Path:
    """Resolve one portable reference and fail closed on scope or integrity errors."""

    repository = Path(project_root).resolve()
    raw_reference = str(reference)
    if artifact_override is not None:
        candidate = _explicit_path(artifact_override, project_root=repository)
    elif raw_reference == LOCAL_ONLY_NOT_PUBLISHED:
        raise PortablePathError(
            f"{LOCAL_ONLY_NOT_PUBLISHED} requires an explicit artifact override"
        )
    else:
        normalized = raw_reference.replace("\\", "/")
        if normalized == DATA_ROOT_REFERENCE or normalized.startswith(
            DATA_ROOT_REFERENCE + "/"
        ):
            root = _data_root(data_root)
            suffix = normalized[len(DATA_ROOT_REFERENCE) :].lstrip("/")
            relative = Path(suffix) if suffix else Path()
            if relative.is_absolute() or ".." in relative.parts:
                raise PortablePathError(
                    f"runtime reference escapes {DATA_ROOT_ENVIRONMENT_VARIABLE}: {raw_reference}"
                )
            candidate = (root / relative).resolve()
            if _relative_to(candidate, root) is None:
                raise PortablePathError(
                    f"runtime reference escapes {DATA_ROOT_ENVIRONMENT_VARIABLE}: {raw_reference}"
                )
        else:
            relative = Path(raw_reference)
            if relative.is_absolute():
                raise PortablePathError(
                    "absolute runtime artifact reference is not portable; use an explicit artifact override"
                )
            candidate = (repository / relative).resolve()
            if _relative_to(candidate, repository) is None:
                raise PortablePathError(
                    f"runtime reference escapes project_root: {raw_reference}"
                )

    if not candidate.exists():
        raise PortablePathError(f"runtime artifact does not exist: {candidate}")
    if expected_kind == "file" and not candidate.is_file():
        raise PortablePathError(f"runtime artifact is not a file: {candidate}")
    if expected_kind == "directory" and not candidate.is_dir():
        raise PortablePathError(f"runtime artifact is not a directory: {candidate}")

    if expected_sha256 is not None:
        hash_target = candidate / hash_relative_path if hash_relative_path else candidate
        if not hash_target.is_file():
            raise PortablePathError(f"integrity target does not exist: {hash_target}")
        observed = hashlib.sha256(hash_target.read_bytes()).hexdigest()
        if observed != expected_sha256:
            raise PortablePathError(
                f"SHA-256 mismatch for {hash_target}: expected {expected_sha256}, observed {observed}"
            )
    return candidate


def resolve_checkpoint_reference(
    reference: Path | str,
    *,
    expected_sha256: str,
    project_root: Path | str,
    data_root: Path | str | None = None,
    checkpoint_override: Path | str | None = None,
) -> Path:
    """Resolve a checkpoint directory and verify its frozen model weights."""

    return resolve_runtime_artifact(
        reference,
        project_root=project_root,
        data_root=data_root,
        artifact_override=checkpoint_override,
        expected_kind="directory",
        expected_sha256=expected_sha256,
        hash_relative_path="model.safetensors",
    )
