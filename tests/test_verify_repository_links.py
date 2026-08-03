"""Regression tests for the repository-wide Markdown link verifier."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_repository_links import main, verify_repository_links


def _write(root: Path, path: str, text: str) -> None:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _commit_fixture(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )


def _make_junction(link: Path, target: Path) -> None:
    """Create a Windows junction without requiring Developer Mode or elevation."""

    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_all_tracked_markdown_local_links_are_checked(tmp_path: Path) -> None:
    """A missing target in any tracked Markdown document must fail the audit."""

    _write(tmp_path, "README.md", "[ok](docs/ok.md)\n")
    _write(tmp_path, "docs/ok.md", "![missing](../reports/figures/missing.png)\n")

    failures = verify_repository_links(tmp_path, ["README.md", "docs/ok.md"])

    assert [(item.source_path, item.line_number, item.target) for item in failures] == [
        ("docs/ok.md", 1, "../reports/figures/missing.png")
    ]
    assert failures[0].reason == "local target does not exist"


def test_external_links_are_out_of_scope(tmp_path: Path) -> None:
    """A network URL must never be fetched or reported by the local audit."""

    _write(tmp_path, "README.md", "[site](https://example.com)\n")

    assert verify_repository_links(tmp_path, ["README.md"]) == ()


def test_existing_file_and_explicit_directory_links_pass(tmp_path: Path) -> None:
    """A real file and a slash-terminated directory are both valid local links."""

    _write(tmp_path, "README.md", "[file](docs/guide.md)\n[directory](docs/)\n")
    _write(tmp_path, "docs/guide.md", "# Guide\n")

    assert verify_repository_links(tmp_path, ["README.md", "docs/guide.md"]) == ()


def test_unsafe_destination_is_returned_as_a_sorted_failure(tmp_path: Path) -> None:
    """An escaping target must fail closed without aborting later diagnostics."""

    _write(
        tmp_path,
        "README.md",
        "[later](missing.md)\n[unsafe](../outside.md)\n",
    )

    failures = verify_repository_links(tmp_path, ["README.md"])

    assert [(item.line_number, item.target) for item in failures] == [
        (1, "missing.md"),
        (2, "../outside.md"),
    ]
    assert "escapes repository" in failures[1].reason


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction regression")
def test_junction_target_outside_repository_fails_closed(tmp_path: Path) -> None:
    """An apparently local path must not follow a junction outside the repository."""

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.md").write_text("# outside\n", encoding="utf-8")
    junction = tmp_path / "docs" / "outside"
    junction.parent.mkdir()
    _make_junction(junction, outside)
    _write(tmp_path, "README.md", "[escape](docs/outside/secret.md)\n")

    try:
        failures = verify_repository_links(tmp_path, ["README.md"])
    finally:
        junction.rmdir()
        shutil.rmtree(outside)

    assert len(failures) == 1
    assert failures[0].reason == "local target resolves outside repository"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction regression")
def test_junction_backed_markdown_source_fails_closed(tmp_path: Path) -> None:
    """A tracked Markdown path cannot read content through a junction outside root."""

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "source.md").write_text("# outside\n", encoding="utf-8")
    junction = tmp_path / "docs" / "outside"
    junction.parent.mkdir()
    _make_junction(junction, outside)

    try:
        failures = verify_repository_links(tmp_path, ["docs/outside/source.md"])
    finally:
        junction.rmdir()
        shutil.rmtree(outside)

    assert len(failures) == 1
    assert failures[0].reason == "tracked Markdown source resolves outside repository"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction regression")
def test_junction_target_inside_repository_remains_valid(tmp_path: Path) -> None:
    """Canonicalization permits a junction only when its final target stays in root."""

    target = tmp_path / "canonical" / "guide.md"
    target.parent.mkdir()
    target.write_text("# Guide\n", encoding="utf-8")
    junction = tmp_path / "docs" / "inside"
    junction.parent.mkdir()
    _make_junction(junction, target.parent)
    _write(tmp_path, "README.md", "[guide](docs/inside/guide.md)\n")

    try:
        failures = verify_repository_links(tmp_path, ["README.md"])
    finally:
        junction.rmdir()

    assert failures == ()


def test_missing_tracked_markdown_source_is_a_failure(tmp_path: Path) -> None:
    """A missing Git-tracked document must become a deterministic diagnostic."""

    failures = verify_repository_links(tmp_path, ["README.md"])

    assert len(failures) == 1
    assert failures[0].source_path == "README.md"
    assert failures[0].line_number == 0
    assert failures[0].reason == "tracked Markdown source does not exist"


def test_non_utf8_tracked_markdown_source_is_a_failure(tmp_path: Path) -> None:
    """A non-UTF-8 tracked document must fail instead of raising a decode error."""

    destination = tmp_path / "README.md"
    destination.write_bytes(b"\xff")

    failures = verify_repository_links(tmp_path, ["README.md"])

    assert len(failures) == 1
    assert failures[0].reason == "tracked Markdown source is not valid UTF-8"


def test_cli_returns_zero_for_valid_repository(tmp_path: Path, capsys) -> None:
    """The CLI must report PASS and return zero when all tracked links resolve."""

    _write(tmp_path, "README.md", "[guide](docs/guide.md)\n")
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _commit_fixture(tmp_path)

    assert main(["--project-root", str(tmp_path)]) == 0

    assert "PASS:" in capsys.readouterr().out


def test_cli_returns_one_and_sorted_diagnostics_for_broken_repository(
    tmp_path: Path, capsys
) -> None:
    """The CLI must return one and show every failure in deterministic order."""

    _write(tmp_path, "README.md", "[z](z.md)\n[a](a.md)\n")
    _commit_fixture(tmp_path)

    assert main(["--project-root", str(tmp_path)]) == 1

    output = capsys.readouterr().out
    assert output.index("README.md:1") < output.index("README.md:2")
    assert "FAILED: 2 broken or unsafe local link(s)" in output


def test_cli_returns_one_for_deleted_tracked_markdown_source(tmp_path: Path, capsys) -> None:
    """A Git-tracked document deleted from disk must make the CLI return one."""

    _write(tmp_path, "README.md", "# Fixture\n")
    _commit_fixture(tmp_path)
    (tmp_path / "README.md").unlink()

    assert main(["--project-root", str(tmp_path)]) == 1

    output = capsys.readouterr().out
    assert "README.md:0" in output
    assert "tracked Markdown source does not exist" in output
