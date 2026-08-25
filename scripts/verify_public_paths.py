"""Fail when tracked UTF-8 text contains non-portable public paths.

The gate enumerates files from Git rather than walking selected directories, so
reports, notebooks, workflow files, source, scripts, and future tracked
locations are all in scope. Search values are assembled at runtime so the
scanner can scan its own source without embedding the private values.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_ACCOUNT_FRAGMENTS = ("3", "Hml")
_LOGIN_FRAGMENTS = ("tun", "2404")
_CI_RUNNER_ROOT_FRAGMENTS = ("c", ":/", "users/", "runneradmin/", "work/")
_DRIVE_ROOT_PATTERN = re.compile(r"(?<![a-z0-9])[a-z]:/", re.IGNORECASE)
_BINARY_EXTENSIONS = frozenset(
    {
        ".7z",
        ".bin",
        ".bmp",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".npy",
        ".npz",
        ".onnx",
        ".parquet",
        ".pdf",
        ".pkl",
        ".png",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
        ".tiff",
        ".webp",
        ".woff",
        ".woff2",
        ".zip",
    }
)


class PublicPathScanError(RuntimeError):
    """Raised when the tracked tree cannot be enumerated or read completely."""


@dataclass(frozen=True)
class ScanResult:
    findings: tuple[str, ...]
    files_scanned: int
    skipped_binary: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.findings


def _marker_patterns() -> tuple[tuple[str, tuple[str, ...]], ...]:
    account = "".join(_ACCOUNT_FRAGMENTS).casefold()
    login = "".join(_LOGIN_FRAGMENTS).casefold()
    windows_users = "".join(("c", ":/", "users/"))  # noqa: FLY002
    return (
        ("private-windows-home", (f"{windows_users}{account}",)),
        (
            "private-posix-home",
            (
                f"/users/{account}",
                f"/home/{account}",
                f"/mnt/c/users/{account}",
                f"/c/users/{account}",
            ),
        ),
        ("private-user-marker", (f"user={login}",)),
    )


def _normalized(text: str) -> str:
    """Normalize raw, JSON-escaped, and repeatedly escaped path separators."""

    return re.sub(r"\\+", "/", text).casefold()


def _is_narrow_ci_runner_path(
    relative_path: str, normalized_line: str, match_start: int
) -> bool:
    normalized_path = relative_path.replace("\\", "/").casefold()
    runner_root = "".join(_CI_RUNNER_ROOT_FRAGMENTS)
    return (
        normalized_path.startswith(".github/workflows/")
        and normalized_line.startswith(runner_root, match_start)
    )


def scan_text(relative_path: str, text: str) -> list[str]:
    """Return sanitized path/line/kind findings for one UTF-8 text file."""

    findings: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        normalized = _normalized(line)
        specific_marker_found = False
        for marker, patterns in _marker_patterns():
            if any(pattern in normalized for pattern in patterns):
                findings.append(f"{relative_path}:{line_number}: {marker}")
                specific_marker_found = True
                break
        if specific_marker_found:
            continue
        for drive_match in _DRIVE_ROOT_PATTERN.finditer(normalized):
            if _is_narrow_ci_runner_path(
                relative_path, normalized, drive_match.start()
            ):
                continue
            findings.append(f"{relative_path}:{line_number}: absolute-windows-drive-path")
            break
    return findings


def _tracked_paths(project_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "-z", "--cached"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise PublicPathScanError("git ls-files failed; tracked-tree coverage is unknown")
    try:
        decoded = completed.stdout.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PublicPathScanError("git ls-files returned a non-UTF-8 tracked path") from error
    return tuple(path for path in decoded.split("\0") if path)


def scan_repository(project_root: Path | str) -> ScanResult:
    """Scan every tracked text file; only known binary extensions may be skipped."""

    root = Path(project_root).resolve()
    findings: list[str] = []
    skipped_binary: list[str] = []
    files_scanned = 0
    for relative_path in _tracked_paths(root):
        path = root / relative_path
        try:
            content = path.read_bytes()
        except OSError as error:
            raise PublicPathScanError(
                f"tracked file could not be read: {relative_path}"
            ) from error
        if path.suffix.casefold() in _BINARY_EXTENSIONS:
            skipped_binary.append(relative_path)
            continue
        if b"\0" in content:
            raise PublicPathScanError(
                f"tracked public text contains NUL: {relative_path}"
            )
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PublicPathScanError(
                f"tracked public text is not UTF-8: {relative_path}"
            ) from error
        files_scanned += 1
        findings.extend(scan_text(relative_path, text))
    return ScanResult(
        findings=tuple(findings),
        files_scanned=files_scanned,
        skipped_binary=tuple(skipped_binary),
    )


def format_scan_lines(result: ScanResult) -> list[str]:
    lines = [
        f"tracked UTF-8 files scanned : {result.files_scanned}",
        f"known binary files skipped : {len(result.skipped_binary)}",
        f"non-portable path findings  : {len(result.findings)}",
    ]
    lines.extend(f"  {finding}" for finding in result.findings)
    if result.clean:
        lines.append("PASS: tracked public text contains no non-portable paths.")
    else:
        lines.append("FAIL: replace every finding with a portable public value.")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Git-tracked UTF-8 text for non-portable public paths."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Git worktree to scan (default: this repository)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        result = scan_repository(arguments.project_root)
    except PublicPathScanError as error:
        print(f"FAIL: {error}")
        return 2
    for line in format_scan_lines(result):
        print(line)
    return 0 if result.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
