"""Regression tests for the repository-wide Markdown link verifier."""

from __future__ import annotations

import subprocess
from pathlib import Path

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
