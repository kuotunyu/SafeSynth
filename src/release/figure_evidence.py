"""Verify the tracked figure commitments before and after repository curation."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from src.release.repository_archive import (
    ArchiveEntry,
    ArchiveError,
    _inside_root,
    _is_path_alias,
    _path_exists,
    _validated_relative_path,
    load_manifest_commitments,
    sha256_file,
)
from src.release.repository_curation import FigureDisposition

_FIGURE_PREFIX = "reports/figures/"


class FigureEvidenceError(RuntimeError):
    """Figure bytes or their tracked curation commitments do not agree."""


def _raise_diagnostics(diagnostics: Sequence[str]) -> None:
    if diagnostics:
        raise FigureEvidenceError("\n".join(sorted(set(diagnostics))))


def _entries_by_path(entries: Sequence[ArchiveEntry]) -> dict[str, ArchiveEntry]:
    by_path: dict[str, ArchiveEntry] = {}
    diagnostics: list[str] = []
    for entry in entries:
        try:
            normalized = str(_validated_relative_path(entry.path))
        except ArchiveError as error:
            diagnostics.append(str(error))
            continue
        if normalized != entry.path or not entry.path.startswith(_FIGURE_PREFIX):
            diagnostics.append(f"manifest contains unsupported figure path: {entry.path!r}")
        elif entry.path in by_path:
            diagnostics.append(f"manifest contains duplicate figure path: {entry.path}")
        elif entry.disposition not in {"KEEP", "DROP"}:
            diagnostics.append(f"manifest contains invalid disposition: {entry.path}")
        else:
            by_path[entry.path] = entry
    _raise_diagnostics(diagnostics)
    return by_path


def load_repository_figure_manifest(project_root: Path) -> tuple[ArchiveEntry, ...]:
    """Load the canonical, payload-free manifest tracked by this repository."""

    manifest_path = Path(project_root) / "reports" / "figure_curation_manifest.json"
    try:
        _, entries, _ = load_manifest_commitments(manifest_path)
    except ArchiveError as error:
        raise FigureEvidenceError(str(error)) from error
    _entries_by_path(entries)
    return entries


def _verify_present_figure(project_root: Path, entry: ArchiveEntry) -> None:
    try:
        relative_path = _validated_relative_path(entry.path)
        candidate = _inside_root(project_root, relative_path)
    except ArchiveError as error:
        raise FigureEvidenceError(str(error)) from error
    if not _path_exists(candidate) or _is_path_alias(candidate) or not candidate.is_file():
        raise FigureEvidenceError(f"missing required figure byte: {entry.path}")
    if candidate.stat().st_size != entry.size_bytes:
        raise FigureEvidenceError(f"size mismatch for figure: {entry.path}")
    if sha256_file(candidate) != entry.sha256:
        raise FigureEvidenceError(f"SHA-256 mismatch for figure: {entry.path}")


def verify_frozen_figure(
    project_root: Path,
    entries_by_path: Mapping[str, ArchiveEntry],
    relative_path: str,
    expected_sha256: str,
) -> None:
    """Bind one scientific metadata reference to its frozen figure commitment."""

    try:
        normalized = str(_validated_relative_path(relative_path))
    except ArchiveError as error:
        raise FigureEvidenceError(str(error)) from error
    if normalized != relative_path or relative_path not in entries_by_path:
        raise FigureEvidenceError(f"figure path is not indexed: {relative_path}")
    entry = entries_by_path[relative_path]
    if expected_sha256 != entry.sha256:
        raise FigureEvidenceError(f"metadata SHA-256 conflicts with manifest: {relative_path}")
    try:
        candidate = _inside_root(Path(project_root), _validated_relative_path(relative_path))
    except ArchiveError as error:
        raise FigureEvidenceError(str(error)) from error
    if _path_exists(candidate):
        _verify_present_figure(Path(project_root), entry)
    elif entry.disposition == "KEEP":
        raise FigureEvidenceError(f"missing required KEEP figure: {relative_path}")


def _tracked_figure_paths(project_root: Path) -> set[str]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--", "reports/figures"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise FigureEvidenceError(f"cannot inspect Git figure inventory: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git ls-files failed"
        raise FigureEvidenceError(f"cannot inspect Git figure inventory: {detail}")
    paths = {line for line in completed.stdout.splitlines() if line}
    diagnostics: list[str] = []
    for path in paths:
        try:
            normalized = str(_validated_relative_path(path))
        except ArchiveError as error:
            diagnostics.append(str(error))
            continue
        if normalized != path or not path.startswith(_FIGURE_PREFIX):
            diagnostics.append(f"Git inventory contains unsupported figure path: {path!r}")
    _raise_diagnostics(diagnostics)
    return paths


def _filesystem_figure_paths(project_root: Path) -> set[str]:
    figure_root = project_root / "reports" / "figures"
    if not _path_exists(figure_root):
        return set()
    if _is_path_alias(figure_root) or not figure_root.is_dir():
        raise FigureEvidenceError(f"path alias or unsupported figure root: {figure_root}")
    paths: set[str] = set()
    diagnostics: list[str] = []
    for candidate in figure_root.rglob("*"):
        if _is_path_alias(candidate):
            diagnostics.append(f"path alias is not allowed: {candidate}")
            continue
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            diagnostics.append(f"unsupported figure filesystem path: {candidate}")
            continue
        relative_path = f"reports/figures/{candidate.relative_to(figure_root).as_posix()}"
        try:
            normalized = str(_validated_relative_path(relative_path))
        except ArchiveError as error:
            diagnostics.append(str(error))
            continue
        if normalized != relative_path:
            diagnostics.append(f"filesystem contains unsupported figure path: {relative_path!r}")
        else:
            paths.add(relative_path)
    _raise_diagnostics(diagnostics)
    return paths


def _inventory_diagnostic(kind: str, actual: set[str], expected: set[str], state: str) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    detail = ", ".join(
        [*(f"missing {path}" for path in missing), *(f"unexpected {path}" for path in extra)]
    )
    return f"{kind} inventory differs from expected {state}: {detail or 'unknown difference'}"


def verify_repository_figure_state(
    project_root: Path,
    entries: Sequence[ArchiveEntry],
    expected_state: Literal["auto", "source", "curated"] = "auto",
) -> Literal["source", "curated"]:
    """Accept only the complete source or curated inventory and verified bytes."""

    if expected_state not in {"auto", "source", "curated"}:
        raise FigureEvidenceError(f"unsupported expected figure state: {expected_state}")
    project_root = Path(project_root)
    by_path = _entries_by_path(entries)
    source_paths = set(by_path)
    curated_paths = {path for path, entry in by_path.items() if entry.disposition == "KEEP"}
    tracked_paths = _tracked_figure_paths(project_root)
    filesystem_paths = _filesystem_figure_paths(project_root)
    states = {"source": source_paths, "curated": curated_paths}
    candidates = ("source", "curated") if expected_state == "auto" else (expected_state,)
    accepted = next(
        (
            state
            for state in candidates
            if tracked_paths == states[state] and filesystem_paths == states[state]
        ),
        None,
    )
    diagnostics: list[str] = []
    if accepted is None:
        if expected_state == "auto":
            diagnostics.append("repository figure inventory matches neither expected source nor curated state")
            for state in candidates:
                diagnostics.append(_inventory_diagnostic("Git", tracked_paths, states[state], state))
                diagnostics.append(
                    _inventory_diagnostic("filesystem", filesystem_paths, states[state], state)
                )
        else:
            diagnostics.append(
                _inventory_diagnostic("Git", tracked_paths, states[expected_state], expected_state)
            )
            diagnostics.append(
                _inventory_diagnostic(
                    "filesystem", filesystem_paths, states[expected_state], expected_state
                )
            )
    for path in sorted(filesystem_paths & source_paths):
        try:
            _verify_present_figure(project_root, by_path[path])
        except FigureEvidenceError as error:
            diagnostics.append(str(error))
    _raise_diagnostics(diagnostics)
    if accepted is None:
        raise AssertionError("missing inventory diagnostic")
    return accepted


def verify_curation_plan_matches_manifest(
    project_root: Path,
    dispositions: Sequence[FigureDisposition],
    entries: Sequence[ArchiveEntry],
) -> None:
    """Require a current curation plan to exactly reproduce every manifest commitment."""

    project_root = Path(project_root)
    entries_by_path = _entries_by_path(entries)
    dispositions_by_path: dict[str, FigureDisposition] = {}
    diagnostics: list[str] = []
    for disposition in dispositions:
        try:
            normalized = str(_validated_relative_path(disposition.path))
        except ArchiveError as error:
            diagnostics.append(str(error))
            continue
        if normalized != disposition.path:
            diagnostics.append(f"curation plan contains unsupported path: {disposition.path!r}")
        elif disposition.path in dispositions_by_path:
            diagnostics.append(f"curation plan contains duplicate path: {disposition.path}")
        else:
            dispositions_by_path[disposition.path] = disposition
    for path in sorted(set(entries_by_path) - set(dispositions_by_path)):
        diagnostics.append(f"curation plan is missing manifest path: {path}")
    for path in sorted(set(dispositions_by_path) - set(entries_by_path)):
        diagnostics.append(f"curation plan has additional path: {path}")
    for path in sorted(set(entries_by_path) & set(dispositions_by_path)):
        entry = entries_by_path[path]
        disposition = dispositions_by_path[path]
        expected_references = tuple(
            sorted(f"{reference.source_path}:{reference.line_number}" for reference in disposition.references)
        )
        if disposition.size_bytes != entry.size_bytes:
            diagnostics.append(f"curation plan size differs from manifest: {path}")
        if disposition.keep != (entry.disposition == "KEEP"):
            diagnostics.append(f"curation plan disposition differs from manifest: {path}")
        if expected_references != entry.reference_sources:
            diagnostics.append(f"curation plan references differ from manifest: {path}")
        try:
            _verify_present_figure(project_root, entry)
        except FigureEvidenceError as error:
            diagnostics.append(str(error))
    _raise_diagnostics(diagnostics)
