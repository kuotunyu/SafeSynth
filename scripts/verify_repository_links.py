"""Fail closed when a tracked Markdown document has a broken local link."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.markdown_links import (
    MarkdownDestination,
    RepositoryLinkError,
    collect_local_destinations,
    extract_markdown_destinations,
    resolve_local_target,
)
from src.release.repository_curation import tracked_files


@dataclass(frozen=True)
class LinkFailure:
    """One tracked Markdown destination that is unsafe or cannot be followed."""

    source_path: str
    line_number: int
    target: str
    reason: str

    def render(self) -> str:
        return f"FAIL {self.source_path}:{self.line_number}: {self.target!r}: {self.reason}"


_ERROR_LINE = re.compile(r"\bline (\d+)\b")


def _filesystem_path(root: Path, relative_path: str) -> Path:
    return root.joinpath(*PurePosixPath(relative_path).parts)


def _is_within_root(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _is_explicit_directory(raw_target: str) -> bool:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return unquote(urlsplit(target).path).replace("\\", "/").endswith("/")


def _error_failure(source_path: str, error: RepositoryLinkError) -> LinkFailure:
    match = _ERROR_LINE.search(str(error))
    return LinkFailure(
        source_path=source_path,
        line_number=int(match.group(1)) if match is not None else 0,
        target="<unresolved>",
        reason=str(error),
    )


def _source_failure(source_path: str, reason: str) -> LinkFailure:
    return LinkFailure(source_path, 0, "<source>", reason)


def _read_trusted_source(
    root: Path, canonical_root: Path, source_path: str
) -> tuple[str | None, LinkFailure | None]:
    source = _filesystem_path(root, source_path)
    try:
        canonical_source = source.resolve(strict=True)
    except FileNotFoundError:
        return None, _source_failure(source_path, "tracked Markdown source does not exist")
    except OSError as error:
        return None, _source_failure(
            source_path, f"cannot resolve tracked Markdown source: {type(error).__name__}"
        )
    if not _is_within_root(canonical_root, canonical_source):
        return None, _source_failure(source_path, "tracked Markdown source resolves outside repository")
    if not canonical_source.is_file():
        return None, _source_failure(source_path, "tracked Markdown source is not a regular file")
    try:
        return canonical_source.read_text(encoding="utf-8"), None
    except UnicodeDecodeError:
        return None, _source_failure(source_path, "tracked Markdown source is not valid UTF-8")
    except OSError as error:
        return None, _source_failure(
            source_path, f"cannot read tracked Markdown source: {type(error).__name__}"
        )


def _collect_one_document(
    root: Path, canonical_root: Path, source_path: str
) -> tuple[tuple[MarkdownDestination, ...], tuple[LinkFailure, ...]]:
    """Collect one document, retaining separate diagnostics after a bad target."""

    source_text, source_failure = _read_trusted_source(root, canonical_root, source_path)
    if source_failure is not None:
        return (), (source_failure,)
    assert source_text is not None
    try:
        return collect_local_destinations(root, [source_path]), ()
    except RepositoryLinkError:
        pass
    except UnicodeDecodeError:
        return (), (_source_failure(source_path, "tracked Markdown source is not valid UTF-8"),)
    except OSError as error:
        return (), (
            _source_failure(source_path, f"cannot read tracked Markdown source: {type(error).__name__}"),
        )

    try:
        raw_destinations = extract_markdown_destinations(source_text, source_path)
    except RepositoryLinkError as error:
        return (), (_error_failure(source_path, error),)

    destinations: list[MarkdownDestination] = []
    failures: list[LinkFailure] = []
    for line_number, raw_target in raw_destinations:
        try:
            resolved_path = resolve_local_target(source_path, raw_target)
        except RepositoryLinkError as error:
            failures.append(
                LinkFailure(source_path, line_number, raw_target, str(error))
            )
            continue
        destinations.append(
            MarkdownDestination(source_path, line_number, raw_target, resolved_path)
        )
    return tuple(destinations), tuple(failures)


def verify_repository_links(root: Path, files: Sequence[str]) -> tuple[LinkFailure, ...]:
    """Return every broken or unsafe local destination in tracked Markdown files.

    External URLs and same-document anchors are intentionally outside this
    offline check. A directory is valid only when the Markdown target explicitly
    ends in a slash, so a typo cannot silently turn a file link into a directory
    reference.
    """

    canonical_root = root.resolve(strict=True)
    failures: list[LinkFailure] = []
    markdown_paths = sorted(path for path in files if path.endswith(".md"))
    for source_path in markdown_paths:
        destinations, document_failures = _collect_one_document(root, canonical_root, source_path)
        failures.extend(document_failures)
        for destination in destinations:
            if destination.resolved_path is None:
                continue
            target_path = _filesystem_path(root, destination.resolved_path)
            try:
                canonical_target = target_path.resolve(strict=True)
            except FileNotFoundError:
                canonical_target = None
                reason = "local target does not exist"
            except OSError as error:
                canonical_target = None
                reason = f"cannot resolve local target: {type(error).__name__}"
            else:
                reason = ""
            if canonical_target is None:
                failures.append(
                    LinkFailure(
                        source_path=destination.source_path,
                        line_number=destination.line_number,
                        target=destination.raw_target,
                        reason=reason,
                    )
                )
                continue
            if not _is_within_root(canonical_root, canonical_target):
                failures.append(
                    LinkFailure(
                        source_path=destination.source_path,
                        line_number=destination.line_number,
                        target=destination.raw_target,
                        reason="local target resolves outside repository",
                    )
                )
                continue
            if canonical_target.is_file():
                continue
            if canonical_target.is_dir() and _is_explicit_directory(destination.raw_target):
                continue
            reason = (
                "directory target must end with '/'"
                if canonical_target.is_dir()
                else "local target does not exist"
            )
            failures.append(
                LinkFailure(
                    source_path=destination.source_path,
                    line_number=destination.line_number,
                    target=destination.raw_target,
                    reason=reason,
                )
            )
    return tuple(sorted(failures, key=lambda item: (item.source_path, item.line_number, item.target)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify every tracked Markdown document's local links without network access."
    )
    parser.add_argument(
        "--project-root",
        default=PROJECT_ROOT,
        type=Path,
        help="repository to verify (default: the repository containing this script)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a deterministic repository-link verdict and return its exit status."""

    args = parse_args(argv)
    root = args.project_root.resolve()
    failures = verify_repository_links(root, tracked_files(root))
    print(f"verify_repository_links: {root}")
    for failure in failures:
        print(failure.render())
    if failures:
        print(f"FAILED: {len(failures)} broken or unsafe local link(s)")
        return 1
    print("PASS: every tracked Markdown local link resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
