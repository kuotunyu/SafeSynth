"""Create and verify fail-closed recovery archives for repository curation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from src.release.repository_curation import FigureDisposition

MANIFEST_NAME = "figure_manifest.json"
BUNDLE_NAME = "repository.bundle"
SCHEMA_VERSION = 1
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_ENTRY_KEYS = frozenset(
    {"path", "size_bytes", "sha256", "disposition", "reference_sources"}
)


class ArchiveError(RuntimeError):
    """A recovery artifact is incomplete, invalid, or unverifiable."""


@dataclass(frozen=True)
class ArchiveEntry:
    """One archived repository file and the evidence needed to verify it."""

    path: str
    size_bytes: int
    sha256: str
    disposition: str
    reference_sources: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveReceipt:
    """Verified paths and digests for a complete recovery package."""

    source_commit: str
    manifest_path: Path
    manifest_sha256: str
    bundle_path: Path
    bundle_sha256: str
    entries: tuple[ArchiveEntry, ...]


@dataclass(frozen=True)
class _VerifiedManifest:
    source_commit: str
    entries: tuple[ArchiveEntry, ...]
    manifest_bytes: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file's contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _is_windows_native() -> bool:
    return sys.platform == "win32"


def _require_windows_native(operation: str) -> None:
    if not _is_windows_native():
        raise ArchiveError(f"{operation} requires Windows no-replace rename semantics")


def _is_path_alias(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _validated_relative_path(raw_path: object) -> PurePosixPath:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path or "\0" in raw_path:
        raise ArchiveError(f"unsafe repository path: {raw_path!r}")
    posix_path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in raw_path.split("/"))
        or str(posix_path) != raw_path
    ):
        raise ArchiveError(f"unsafe repository path: {raw_path!r}")
    return posix_path


def _inside_root(root: Path, relative_path: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative_path.parts)
    current = root
    if _path_exists(current) and _is_path_alias(current):
        raise ArchiveError(f"path alias is not allowed: {current}")
    for part in relative_path.parts:
        current = current / part
        if _path_exists(current) and _is_path_alias(current):
            raise ArchiveError(f"path alias is not allowed: {current}")
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ArchiveError(f"unsafe repository path: {str(relative_path)!r}")
    return candidate


def _source_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise ArchiveError(f"cannot inspect source commit: {error}") from error
    commit = completed.stdout.strip()
    if completed.returncode != 0 or _COMMIT_PATTERN.fullmatch(commit) is None:
        detail = completed.stderr.strip() or "HEAD is not a 40-character Git commit"
        raise ArchiveError(f"cannot inspect source commit: {detail}")
    return commit


def _canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _entry_dict(entry: ArchiveEntry) -> dict[str, Any]:
    serialized = asdict(entry)
    serialized["reference_sources"] = list(entry.reference_sources)
    return serialized


def _preflight_dispositions(
    project_root: Path, dispositions: Sequence[FigureDisposition]
) -> tuple[tuple[FigureDisposition, PurePosixPath, Path], ...]:
    planned: list[tuple[FigureDisposition, PurePosixPath, Path]] = []
    seen: set[str] = set()
    seen_resolved: set[str] = set()
    for disposition in dispositions:
        relative_path = _validated_relative_path(disposition.path)
        if disposition.path in seen:
            raise ArchiveError(f"duplicate archive path: {disposition.path}")
        seen.add(disposition.path)
        if type(disposition.size_bytes) is not int or disposition.size_bytes < 0:
            raise ArchiveError(f"invalid source size: {disposition.path}")
        if type(disposition.keep) is not bool:
            raise ArchiveError(f"invalid disposition: {disposition.path}")
        source = _inside_root(project_root, relative_path)
        if not source.is_file():
            raise ArchiveError(f"missing source or unsupported source type: {disposition.path}")
        resolved_key = os.path.normcase(str(source.resolve()))
        if resolved_key in seen_resolved:
            raise ArchiveError(f"duplicate resolved source target: {disposition.path}")
        seen_resolved.add(resolved_key)
        actual_size = source.stat().st_size
        if actual_size != disposition.size_bytes:
            raise ArchiveError(
                f"source size mismatch for {disposition.path}: "
                f"expected {disposition.size_bytes}, found {actual_size}"
            )
        for reference in disposition.references:
            if (
                not isinstance(reference.source_path, str)
                or not reference.source_path
                or type(reference.line_number) is not int
                or reference.line_number < 1
            ):
                raise ArchiveError(f"invalid reference source for {disposition.path}")
        planned.append((disposition, relative_path, source))
    return tuple(sorted(planned, key=lambda item: item[0].path))


def _build_verified_archive(
    project_root: Path, archive_root: Path, dispositions: Sequence[FigureDisposition]
) -> _VerifiedManifest:
    source_commit = _source_commit(project_root)
    planned = _preflight_dispositions(project_root, dispositions)
    figure_root = archive_root / "figures"
    figure_root.mkdir()
    entries: list[ArchiveEntry] = []

    for disposition, relative_path, source in planned:
        source_size_before = source.stat().st_size
        source_digest = sha256_file(source)
        target = figure_root.joinpath(*relative_path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_size = target.stat().st_size
        copied_digest = sha256_file(target)
        if copied_size != disposition.size_bytes:
            raise ArchiveError(f"archived size mismatch for {disposition.path}")
        if copied_digest != source_digest:
            raise ArchiveError(f"archived SHA-256 mismatch for {disposition.path}")
        if source.stat().st_size != source_size_before or sha256_file(source) != source_digest:
            raise ArchiveError(f"source changed while archiving: {disposition.path}")

        reference_sources = tuple(
            sorted(
                f"{reference.source_path}:{reference.line_number}"
                for reference in disposition.references
            )
        )
        entries.append(
            ArchiveEntry(
                path=disposition.path,
                size_bytes=copied_size,
                sha256=copied_digest,
                disposition="KEEP" if disposition.keep else "DROP",
                reference_sources=reference_sources,
            )
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "entries": [_entry_dict(entry) for entry in entries],
    }
    (archive_root / MANIFEST_NAME).write_bytes(_canonical_manifest_bytes(manifest))
    verified = _load_verified_manifest(archive_root)
    if verified.entries != tuple(entries):
        raise ArchiveError("verified archive entries differ from source plan")
    return verified


def _publish_directory(staging: Path, destination: Path, artifact: str) -> None:
    if _path_exists(destination):
        raise FileExistsError(destination)
    try:
        os.rename(staging, destination)
    except OSError as error:
        if _path_exists(destination):
            raise FileExistsError(destination) from error
        raise ArchiveError(f"cannot publish verified {artifact}: {error}") from error


def create_verified_archive(
    project_root: Path,
    destination: Path,
    dispositions: Sequence[FigureDisposition],
) -> Path:
    """Publish a verified archive using Windows no-replace rename semantics."""

    _require_windows_native("archive publication")
    project_root = Path(project_root)
    destination = Path(destination)
    if _path_exists(destination):
        raise FileExistsError(destination)
    _preflight_dispositions(project_root, dispositions)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging-", dir=destination.parent
    ) as temporary_directory:
        staging = Path(temporary_directory)
        _build_verified_archive(project_root, staging, dispositions)
        _publish_directory(staging, destination, "archive")

    return destination / MANIFEST_NAME


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError(f"manifest schema contains duplicate key: {key}")
        result[key] = value
    return result


def _parse_entry(raw_entry: object) -> ArchiveEntry:
    if not isinstance(raw_entry, dict) or set(raw_entry) != _ENTRY_KEYS:
        raise ArchiveError("manifest schema has invalid entry fields")
    relative_path = _validated_relative_path(raw_entry["path"])
    path = str(relative_path)
    size_bytes = raw_entry["size_bytes"]
    digest = raw_entry["sha256"]
    disposition = raw_entry["disposition"]
    references = raw_entry["reference_sources"]
    if type(size_bytes) is not int or size_bytes < 0:
        raise ArchiveError(f"manifest schema has invalid size for {path}")
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise ArchiveError(f"manifest schema has invalid SHA-256 for {path}")
    if disposition not in {"KEEP", "DROP"}:
        raise ArchiveError(f"manifest schema has invalid disposition for {path}")
    if (
        not isinstance(references, list)
        or any(not isinstance(item, str) or not item for item in references)
        or references != sorted(set(references))
    ):
        raise ArchiveError(f"manifest schema has invalid reference sources for {path}")
    return ArchiveEntry(path, size_bytes, digest, disposition, tuple(references))


def _load_verified_manifest(archive_root: Path) -> _VerifiedManifest:
    archive_root = Path(archive_root)
    manifest_path = archive_root / MANIFEST_NAME
    if _is_path_alias(manifest_path) or not manifest_path.is_file():
        raise ArchiveError(f"missing manifest: {manifest_path}")
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(
            manifest_bytes.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArchiveError(f"manifest schema is not valid UTF-8 JSON: {error}") from error
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "source_commit",
        "entries",
    }:
        raise ArchiveError("manifest schema has invalid top-level fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ArchiveError("manifest schema version is unsupported")
    source_commit = manifest["source_commit"]
    if not isinstance(source_commit, str) or _COMMIT_PATTERN.fullmatch(source_commit) is None:
        raise ArchiveError("manifest schema has invalid source commit")
    raw_entries = manifest["entries"]
    if not isinstance(raw_entries, list):
        raise ArchiveError("manifest schema entries must be a list")
    entries = tuple(_parse_entry(raw_entry) for raw_entry in raw_entries)
    paths = [entry.path for entry in entries]
    if len(paths) != len(set(paths)):
        raise ArchiveError("manifest schema contains duplicate archive path")
    if paths != sorted(paths):
        raise ArchiveError("manifest schema entries are not sorted")
    if manifest_bytes != _canonical_manifest_bytes(manifest):
        raise ArchiveError("manifest schema is not canonical JSON")

    figure_root = archive_root / "figures"
    if _is_path_alias(figure_root) or not figure_root.is_dir():
        raise ArchiveError("archived file set does not match manifest")
    actual_paths: set[str] = set()
    for candidate in figure_root.rglob("*"):
        if candidate.is_dir() and not _is_path_alias(candidate):
            continue
        if _is_path_alias(candidate) or not candidate.is_file():
            raise ArchiveError("archived file set contains unsupported paths")
        actual_paths.add(candidate.relative_to(figure_root).as_posix())
    if actual_paths != set(paths):
        raise ArchiveError("archived file set does not match manifest")

    for entry in entries:
        relative_path = _validated_relative_path(entry.path)
        archived_file = _inside_root(figure_root, relative_path)
        if sha256_file(archived_file) != entry.sha256:
            raise ArchiveError(f"SHA-256 mismatch for archived file: {entry.path}")
        if archived_file.stat().st_size != entry.size_bytes:
            raise ArchiveError(f"size mismatch for archived file: {entry.path}")
    return _VerifiedManifest(source_commit, entries, manifest_bytes)


def load_and_verify_manifest(archive_root: Path) -> tuple[ArchiveEntry, ...]:
    """Load a canonical manifest and verify the exact archived file set and bytes."""

    return _load_verified_manifest(archive_root).entries


def create_verified_git_bundle(project_root: Path, bundle_path: Path) -> str:
    """Verify and publish a bundle using Windows no-replace rename semantics."""

    _require_windows_native("Git bundle publication")
    project_root = Path(project_root)
    bundle_path = Path(bundle_path)
    if _path_exists(bundle_path):
        raise FileExistsError(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    published_stat: os.stat_result | None = None

    with tempfile.TemporaryDirectory(
        prefix=f".{bundle_path.name}.staging-", dir=bundle_path.parent
    ) as temporary_directory:
        staged_bundle = Path(temporary_directory) / bundle_path.name
        try:
            created = subprocess.run(
                ["git", "bundle", "create", str(staged_bundle), "--all"],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise ArchiveError(f"Git bundle creation failed: {error}") from error
        if created.returncode != 0 or not staged_bundle.is_file():
            detail = created.stderr.strip() or "Git did not create a bundle"
            raise ArchiveError(f"Git bundle creation failed: {detail}")
        verified = subprocess.run(
            ["git", "bundle", "verify", str(staged_bundle)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if verified.returncode != 0:
            detail = verified.stderr.strip() or verified.stdout.strip()
            raise ArchiveError(f"Git bundle verification failed: {detail}")
        digest = sha256_file(staged_bundle)
        try:
            os.rename(staged_bundle, bundle_path)
            published_stat = bundle_path.stat()
        except FileExistsError:
            raise FileExistsError(bundle_path) from None
        except OSError as error:
            if _path_exists(bundle_path):
                raise FileExistsError(bundle_path) from error
            raise ArchiveError(f"cannot publish verified Git bundle: {error}") from error
    if sha256_file(bundle_path) != digest:
        try:
            current_stat = bundle_path.stat()
            if published_stat is not None and os.path.samestat(published_stat, current_stat):
                bundle_path.unlink()
        except FileNotFoundError:
            pass
        raise ArchiveError("published Git bundle SHA-256 mismatch")
    return digest


def _verify_bundle_contains_commit(bundle_path: Path, source_commit: str) -> None:
    with tempfile.TemporaryDirectory(
        prefix=".bundle-contents-", dir=bundle_path.parent
    ) as temporary_directory:
        bare_repository = Path(temporary_directory) / "verify.git"
        commands = (
            ["git", "init", "--bare", "-q", str(bare_repository)],
            ["git", "-C", str(bare_repository), "bundle", "unbundle", str(bundle_path)],
            [
                "git",
                "-C",
                str(bare_repository),
                "cat-file",
                "-e",
                f"{source_commit}^{{commit}}",
            ],
        )
        for command in commands:
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as error:
                raise ArchiveError(f"cannot inspect Git bundle contents: {error}") from error
            if completed.returncode != 0:
                if command == commands[-1]:
                    raise ArchiveError(
                        f"Git bundle does not contain source commit: {source_commit}"
                    )
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise ArchiveError(f"cannot inspect Git bundle contents: {detail}")


def create_recovery_package(
    project_root: Path,
    destination: Path,
    dispositions: Sequence[FigureDisposition],
) -> ArchiveReceipt:
    """Build, verify, and publish one complete Windows-native recovery package."""

    _require_windows_native("recovery package publication")
    project_root = Path(project_root)
    destination = Path(destination)
    if _path_exists(destination):
        raise FileExistsError(destination)
    _preflight_dispositions(project_root, dispositions)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging-", dir=destination.parent
    ) as temporary_directory:
        staging = Path(temporary_directory)
        verified_manifest = _build_verified_archive(project_root, staging, dispositions)
        staged_bundle = staging / BUNDLE_NAME
        bundle_digest = create_verified_git_bundle(project_root, staged_bundle)
        _verify_bundle_contains_commit(staged_bundle, verified_manifest.source_commit)
        _publish_directory(staging, destination, "recovery package")

    manifest_path = destination / MANIFEST_NAME
    bundle_path = destination / BUNDLE_NAME
    return ArchiveReceipt(
        source_commit=verified_manifest.source_commit,
        manifest_path=manifest_path,
        manifest_sha256=verified_manifest.sha256,
        bundle_path=bundle_path,
        bundle_sha256=bundle_digest,
        entries=verified_manifest.entries,
    )


def restore_keep_files(project_root: Path, archive_root: Path) -> tuple[str, ...]:
    """Stage and atomically publish the complete verified KEEP figure tree on Windows."""

    _require_windows_native("KEEP restoration")
    project_root = Path(project_root)
    verified_manifest = _load_verified_manifest(archive_root)
    keep_entries = tuple(
        entry for entry in verified_manifest.entries if entry.disposition == "KEEP"
    )
    if _path_exists(project_root) and (_is_path_alias(project_root) or not project_root.is_dir()):
        raise ArchiveError(f"restore root is not a directory: {project_root}")
    if not keep_entries:
        return ()

    reports_root = project_root / "reports"
    final_figures = reports_root / "figures"
    _inside_root(project_root, PurePosixPath("reports/figures"))
    if _path_exists(final_figures):
        raise FileExistsError(final_figures)

    sources: list[tuple[ArchiveEntry, Path, PurePosixPath]] = []
    for entry in keep_entries:
        relative_path = _validated_relative_path(entry.path)
        if relative_path.parts[:2] != ("reports", "figures") or len(relative_path.parts) < 3:
            raise ArchiveError(f"KEEP path is outside reports/figures: {entry.path}")
        source = _inside_root(Path(archive_root) / "figures", relative_path)
        within_figures = PurePosixPath(*relative_path.parts[2:])
        sources.append((entry, source, within_figures))

    project_root.mkdir(parents=True, exist_ok=True)
    if _path_exists(reports_root) and (_is_path_alias(reports_root) or not reports_root.is_dir()):
        raise ArchiveError(f"restore reports path is not a directory: {reports_root}")
    reports_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".figures.staging-", dir=reports_root
    ) as temporary_directory:
        staging = Path(temporary_directory)
        for entry, source, within_figures in sources:
            target = staging.joinpath(*within_figures.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as source_handle, target.open("xb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle)
            if target.stat().st_size != entry.size_bytes or sha256_file(target) != entry.sha256:
                raise ArchiveError(f"restored file verification failed: {entry.path}")
        _publish_directory(staging, final_figures, "KEEP figure tree")
    return tuple(entry.path for entry in keep_entries)
