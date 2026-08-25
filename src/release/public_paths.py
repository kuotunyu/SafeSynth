"""Portable path serialization for public repository artifacts."""

from __future__ import annotations

from pathlib import Path

LOCAL_ONLY_NOT_PUBLISHED = "local_only_not_published"


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
