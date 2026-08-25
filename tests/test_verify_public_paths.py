"""Regression tests for the tracked-tree personal-path release gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import verify_public_paths as scanner
from src.release.public_paths import LOCAL_ONLY_NOT_PUBLISHED, public_artifact_path


def _account_name() -> str:
    return "lmH3"[::-1]


def _login_name() -> str:
    return "4042nut"[::-1]


def _windows_home(*, slash: str = "\\", account: str | None = None) -> str:
    name = _account_name() if account is None else account
    return f"C:{slash}Users{slash}{name}"


def _external_drive_path(*, slash: str = "\\", drive: str = "D") -> str:
    return f"{drive}:{slash}sdg-data{slash}02-safesynth{slash}artifact.json"


@pytest.mark.parametrize(
    ("text", "marker"),
    [
        (_windows_home(), "private-windows-home"),
        (_windows_home(slash="\\\\"), "private-windows-home"),
        (_windows_home(slash="\\\\\\\\"), "private-windows-home"),
        (_windows_home(slash="/"), "private-windows-home"),
        (_windows_home(account=_account_name().swapcase()), "private-windows-home"),
        (f"/Users/{_account_name()}/archive", "private-posix-home"),
        (f"/home/{_account_name()}/archive", "private-posix-home"),
        (f"/mnt/c/Users/{_account_name()}/archive", "private-posix-home"),
        (f"USER={_login_name()}", "private-user-marker"),
        (f"user={_login_name().upper()}", "private-user-marker"),
    ],
)
def test_raw_escaped_notebook_and_case_variants_are_reported(
    text: str, marker: str
) -> None:
    findings = scanner.scan_text("reports/evidence.json", f"safe\n{text}\n")

    assert findings == [f"reports/evidence.json:2: {marker}"]


@pytest.mark.parametrize(
    "text",
    [
        "reports/figures/evidence.png",
        "docs/reproduction.md",
        "local_only_not_published",
        "USER=runner",
    ],
)
def test_portable_and_ci_paths_do_not_trigger_the_gate(text: str) -> None:
    assert scanner.scan_text("reports/evidence.json", text) == []


@pytest.mark.parametrize(
    "text",
    [
        _external_drive_path(),
        _external_drive_path(slash="\\\\"),
        _external_drive_path(slash="\\\\\\\\"),
        _external_drive_path(slash="/"),
        _external_drive_path(drive="d"),
    ],
)
def test_public_surfaces_reject_generic_windows_drive_paths(text: str) -> None:
    findings = scanner.scan_text("reports/evidence.json", f"safe\n{text}\n")

    assert findings == [
        "reports/evidence.json:2: absolute-windows-drive-path"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/releases/report.json",
        "doi:10.5281/zenodo.123456",
        "reports/figures/evidence.png",
        "${SAFESYNTH_DATA_ROOT}/runs/checkpoint",
        "local_only_not_published",
    ],
)
def test_urls_relative_paths_and_portable_placeholders_are_not_drive_paths(
    text: str,
) -> None:
    assert scanner.scan_text("reports/evidence.json", text) == []


@pytest.mark.parametrize("path", ["scripts/generator.py", "src/release/writer.py"])
def test_generic_drive_rule_includes_scripts_and_source(path: str) -> None:
    assert scanner.scan_text(path, _external_drive_path()) == [
        f"{path}:1: absolute-windows-drive-path"
    ]


def test_ci_runner_drive_path_is_not_a_public_artifact_finding() -> None:
    runner_path = f"C:{'/'}Users{'/'}runneradmin{'/'}work{'/'}project"

    assert scanner.scan_text(".github/workflows/ci.yml", runner_path) == []


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def test_repository_scan_reads_only_tracked_utf8_text(tmp_path: Path) -> None:
    _run_git(tmp_path, "init", "--quiet")
    tracked = tmp_path / "tracked.json"
    untracked = tmp_path / "scratch.json"
    binary = tmp_path / "image.bin"
    tracked.write_text(_windows_home(slash="\\\\"), encoding="utf-8")
    untracked.write_text(_windows_home(), encoding="utf-8")
    binary.write_bytes(b"\xff\xfe" + _windows_home().encode("utf-8"))
    _run_git(tmp_path, "add", "tracked.json", "image.bin")

    result = scanner.scan_repository(tmp_path)

    assert result.findings == ("tracked.json:1: private-windows-home",)
    assert result.files_scanned == 1
    assert result.skipped_binary == ("image.bin",)
    assert result.clean is False


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("broken.json", b"\xff\xfe", "tracked public text is not UTF-8"),
        ("contains_nul.py", b"safe\x00text", "tracked public text contains NUL"),
    ],
)
def test_repository_scan_fails_closed_for_invalid_public_text(
    tmp_path: Path, name: str, content: bytes, message: str
) -> None:
    _run_git(tmp_path, "init", "--quiet")
    path = tmp_path / name
    path.write_bytes(content)
    _run_git(tmp_path, "add", name)

    with pytest.raises(scanner.PublicPathScanError, match=message):
        scanner.scan_repository(tmp_path)


def test_scan_fails_closed_when_tracked_files_cannot_be_enumerated(tmp_path: Path) -> None:
    with pytest.raises(scanner.PublicPathScanError, match="git ls-files"):
        scanner.scan_repository(tmp_path)


def test_console_output_reports_location_and_kind_without_echoing_private_values() -> None:
    private = _windows_home()
    result = scanner.ScanResult(
        findings=("reports/evidence.json:7: private-windows-home",),
        files_scanned=4,
        skipped_binary=(),
    )

    rendered = "\n".join(scanner.format_scan_lines(result))

    assert "reports/evidence.json:7" in rendered
    assert "private-windows-home" in rendered
    assert private.casefold() not in rendered.casefold()
    assert "FAIL" in rendered


def test_scanner_source_does_not_contain_the_private_values_it_searches_for() -> None:
    source = (scanner.PROJECT_ROOT / "scripts" / "verify_public_paths.py").read_text(
        encoding="utf-8"
    )

    assert _account_name().casefold() not in source.casefold()
    assert _login_name().casefold() not in source.casefold()


def test_scanner_source_does_not_embed_a_generic_drive_fixture() -> None:
    source = (scanner.PROJECT_ROOT / "scripts" / "verify_public_paths.py").read_text(
        encoding="utf-8"
    )

    assert _external_drive_path().casefold() not in source.casefold()


def test_repository_itself_has_no_tracked_personal_paths() -> None:
    result = scanner.scan_repository(scanner.PROJECT_ROOT)

    assert result.clean, "\n".join(result.findings)
    assert result.files_scanned > 0


def test_ci_runs_the_whole_tree_public_path_gate() -> None:
    workflow = (scanner.PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "uv run python -m scripts.verify_public_paths" in workflow


def test_public_artifact_path_is_relative_inside_the_repository(tmp_path: Path) -> None:
    artifact = tmp_path / "reports" / "figures" / "evidence.png"

    assert public_artifact_path(artifact, project_root=tmp_path) == (
        "reports/figures/evidence.png"
    )


def test_public_artifact_path_marks_external_files_as_local_only(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    external = tmp_path / "downloads" / "results.zip"

    assert public_artifact_path(external, project_root=repository) == (
        LOCAL_ONLY_NOT_PUBLISHED
    )
