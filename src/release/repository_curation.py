"""Build a fail-closed, deterministic curation plan for tracked figures."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.release.markdown_links import RepositoryLinkError, collect_local_destinations

GENERATED_REFERENCE_INPUTS = frozenset({"reports/repo_slimming_plan.md"})
FIGURE_ROOT = "reports/figures/"


@dataclass(frozen=True)
class FigureReference:
    """One Markdown source line that links to a tracked figure."""

    source_path: str
    line_number: int


@dataclass(frozen=True)
class FigureDisposition:
    """The safe keep-or-drop decision for one tracked file under the figure root."""

    path: str
    size_bytes: int
    keep: bool
    references: tuple[FigureReference, ...]


def tracked_files(root: Path) -> list[str]:
    """Return the sorted inventory of paths tracked by Git at ``root``."""

    result = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def plan_figure_curation(root: Path, files: Sequence[str]) -> tuple[FigureDisposition, ...]:
    """Classify tracked figure-root files using exact resolved Markdown links."""

    inventory = frozenset(files)
    markdown_inputs = sorted(
        path
        for path in inventory
        if path.endswith(".md") and path not in GENERATED_REFERENCE_INPUTS
    )
    references_by_figure: defaultdict[str, list[FigureReference]] = defaultdict(list)

    for destination in collect_local_destinations(root, markdown_inputs):
        resolved = destination.resolved_path
        if resolved is None:
            continue
        if resolved not in inventory and not (root / resolved).is_dir():
            raise RepositoryLinkError(
                "local Markdown destination is neither tracked nor an existing directory: "
                f"{destination.source_path}:{destination.line_number}: {resolved}"
            )
        if not resolved.startswith(FIGURE_ROOT) or (root / resolved).is_dir():
            continue
        references_by_figure[resolved].append(
            FigureReference(destination.source_path, destination.line_number)
        )

    dispositions: list[FigureDisposition] = []
    for path in sorted(item for item in inventory if item.startswith(FIGURE_ROOT)):
        references = tuple(
            sorted(references_by_figure[path], key=lambda item: (item.source_path, item.line_number))
        )
        dispositions.append(
            FigureDisposition(
                path=path,
                size_bytes=(root / path).stat().st_size,
                keep=bool(references),
                references=references,
            )
        )
    return tuple(dispositions)
