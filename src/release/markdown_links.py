"""Extract and safely resolve local Markdown destinations."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


class RepositoryLinkError(ValueError):
    """A local Markdown destination cannot be resolved safely."""


@dataclass(frozen=True)
class MarkdownDestination:
    """A Markdown destination and its repository-relative resolution."""

    source_path: str
    line_number: int
    raw_target: str
    resolved_path: str | None


_FENCE_START = re.compile(r"^\s*(`{3,}|~{3,})")
_FENCE_CLOSE = re.compile(r"^\s*(`{3,}|~{3,})\s*$")
_INLINE_CODE = re.compile(r"`+[^`]*`+")
_INLINE_DESTINATION = re.compile(
    r"!?\[[^\]]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
_INLINE_OPEN = re.compile(r"!?\[[^\]]*\]\(")
_REFERENCE_DESTINATION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>\n]+>|\S+)")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _without_inline_code(line: str) -> str:
    return _INLINE_CODE.sub("", line)


def extract_markdown_destinations(text: str, source_path: str) -> tuple[tuple[int, str], ...]:
    """Return real Markdown destinations, excluding fenced and inline code."""

    del source_path
    destinations: list[tuple[int, str]] = []
    fence: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if fence is not None:
            fence_close = _FENCE_CLOSE.match(line)
            if (
                fence_close
                and fence_close.group(1)[0] == fence[0]
                and len(fence_close.group(1)) >= len(fence)
            ):
                fence = None
            continue
        fence_match = _FENCE_START.match(line)
        if fence_match:
            fence = fence_match.group(1)
            continue

        visible = _without_inline_code(line)
        reference_match = _REFERENCE_DESTINATION.match(visible)
        if reference_match:
            destinations.append((line_number, reference_match.group(1)))
            continue
        inline_matches = tuple(_INLINE_DESTINATION.finditer(visible))
        for opener in _INLINE_OPEN.finditer(visible):
            if not any(match.start() <= opener.start() < match.end() for match in inline_matches):
                raise RepositoryLinkError(f"malformed Markdown destination on line {line_number}")
        destinations.extend((line_number, match.group(1)) for match in inline_matches)
    return tuple(destinations)


def _normalized_repository_path(path: str, *, raw_target: str) -> str:
    posix_path = path.replace("\\", "/")
    if PurePosixPath(posix_path).is_absolute():
        raise RepositoryLinkError(f"absolute repository path is forbidden: {raw_target}")
    normalized = posixpath.normpath(posix_path)
    if normalized in {".", ".."} or normalized.startswith("../"):
        raise RepositoryLinkError(f"link escapes repository: {raw_target}")
    return normalized


def resolve_local_target(source_path: str, raw_target: str) -> str | None:
    """Resolve one local Markdown target relative to its document directory."""

    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if _WINDOWS_ABSOLUTE_PATH.match(target):
        raise RepositoryLinkError(f"absolute local path is forbidden: {raw_target}")
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith("#"):
        return None
    if not parsed.path:
        return None
    local_path = unquote(parsed.path).replace("\\", "/")
    if _WINDOWS_ABSOLUTE_PATH.match(local_path) or PurePosixPath(local_path).is_absolute():
        raise RepositoryLinkError(f"absolute local path is forbidden: {raw_target}")
    base = PurePosixPath(source_path).parent.as_posix()
    return _normalized_repository_path(posixpath.join(base, local_path), raw_target=raw_target)


def collect_local_destinations(root: Path, markdown_paths: Sequence[str]) -> tuple[MarkdownDestination, ...]:
    """Collect Markdown destinations from repository-relative Markdown files."""

    destinations: list[MarkdownDestination] = []
    for markdown_path in markdown_paths:
        source_path = _normalized_repository_path(markdown_path, raw_target=markdown_path)
        document = root / Path(*PurePosixPath(source_path).parts)
        for line_number, raw_target in extract_markdown_destinations(
            document.read_text(encoding="utf-8"), source_path
        ):
            destinations.append(
                MarkdownDestination(
                    source_path=source_path,
                    line_number=line_number,
                    raw_target=raw_target,
                    resolved_path=resolve_local_target(source_path, raw_target),
                )
            )
    return tuple(destinations)
