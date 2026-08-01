"""Tests for the ADR-005 / PLAN_PHASE2.md M20 forbidden-package checker.

The thing this file exists to prevent is a checker that can only say "clean". The
predecessor of `scripts/check_forbidden_licences.py` was a string constant in the
benchmark script asserting that the scan found nothing; adding a real import of
the forbidden package under `src/` and regenerating the report left the assertion
printed word for word. A scanner that cannot report a hit is the same fabrication
wearing a function signature, so BOTH directions are proved here: a tree that
contains the forbidden import must produce a hit with the right file and line, and
a tree that does not must produce none.

The forbidden package is written BACKWARDS in this file. That keeps the literal out
of the repository - the property the scanner relies on to avoid exempting itself -
and it also means the expected term is not the same expression as the code under
test, which assembles it from fragments. Comparing an implementation against itself
is the first of the four K-19 failure shapes.

CPU only, no model, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.check_forbidden_licences as checker
from src.data.paths import PROJECT_ROOT

FORBIDDEN_PACKAGE = "scitylartlu"[::-1]

SCANNER_SOURCES = (
    PROJECT_ROOT / "scripts" / "check_forbidden_licences.py",
    PROJECT_ROOT / "scripts" / "benchmark_latency.py",
)


def _tree(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# The search term
# --------------------------------------------------------------------------


def test_the_checker_searches_for_the_package_adr_005_actually_forbids() -> None:
    assert checker.forbidden_package_name() == FORBIDDEN_PACKAGE


@pytest.mark.parametrize("path", SCANNER_SOURCES, ids=lambda p: p.name)
def test_no_scanner_source_contains_the_literal_it_searches_for(path: Path) -> None:
    """Runtime assembly is what makes an exemption unnecessary, so it is asserted.

    The moment one of these files spells the term out it starts matching itself,
    and the only way back to a green scan is an exemption - which is exactly the
    hole a genuine import added to that same file would later escape through.
    """

    assert FORBIDDEN_PACKAGE not in path.read_text(encoding="utf-8").lower()


# --------------------------------------------------------------------------
# Direction 1: a tree that contains the forbidden import reports a hit
# --------------------------------------------------------------------------


def test_a_tree_containing_the_forbidden_import_reports_a_hit(tmp_path) -> None:
    _tree(
        tmp_path,
        {
            "src/clean.py": "import torch\n",
            "src/dirty.py": f"import torch\n\nfrom {FORBIDDEN_PACKAGE} import YOLO\n",
            "scripts/shouting.py": f"# {FORBIDDEN_PACKAGE.upper()} was considered\n",
            "notebooks/nb.ipynb": '{"cells": []}\n',
        },
    )
    scan = checker.scan_for_forbidden_package(tmp_path)

    assert scan["clean"] is False
    # Roots are visited in declared order, files within a root in sorted order,
    # and the line number is the line the import is really on.
    assert scan["matches"] == ["src/dirty.py:3", "scripts/shouting.py:1"]
    assert scan["files_scanned"] == 4
    assert scan["missing_roots"] == []


def test_the_checker_does_not_exempt_itself(tmp_path) -> None:
    """A path exemption for the scanner would be the hole, not the fix."""

    _tree(
        tmp_path,
        {"scripts/check_forbidden_licences.py": f"import {FORBIDDEN_PACKAGE}\n"},
    )
    assert checker.scan_for_forbidden_package(tmp_path)["matches"] == [
        "scripts/check_forbidden_licences.py:1"
    ]


def test_a_match_outside_the_requested_roots_is_not_reported(tmp_path) -> None:
    """`reports/` and `docs/` discuss the decision in prose and are not code."""

    _tree(
        tmp_path,
        {
            "src/clean.py": "import torch\n",
            "reports/adr.md": f"ADR-005 rejects {FORBIDDEN_PACKAGE} (AGPL-3.0).\n",
        },
    )
    scan = checker.scan_for_forbidden_package(tmp_path, ["src"])
    assert scan["clean"] is True
    assert scan["files_scanned"] == 1


# --------------------------------------------------------------------------
# Direction 2: a clean tree reports none
# --------------------------------------------------------------------------


def test_a_clean_tree_reports_no_matches_and_skips_build_artefacts(tmp_path) -> None:
    _tree(
        tmp_path,
        {
            "src/clean.py": "import torch\n",
            "src/nested/also_clean.py": "from transformers import AutoModel\n",
            "scripts/__pycache__/stale.py": f"import {FORBIDDEN_PACKAGE}\n",
        },
    )
    scan = checker.scan_for_forbidden_package(tmp_path)

    assert scan["clean"] is True
    assert scan["matches"] == []
    assert scan["files_scanned"] == 2, "a .pyc is a build artefact of a .py already read"
    assert scan["missing_roots"] == ["notebooks"]


def test_an_absent_root_is_reported_rather_than_counted_as_scanned(tmp_path) -> None:
    """A scan of a directory that is not there proves nothing about it."""

    scan = checker.scan_for_forbidden_package(tmp_path)
    assert scan["missing_roots"] == ["src", "scripts", "notebooks"]
    assert scan["files_scanned"] == 0
    # `clean` still reflects the matches found; the missing roots are what stop a
    # reader treating an empty scan as evidence, so they must reach the output.
    assert any("NOT SCANNED" in line for line in checker.format_scan_lines(scan))


def test_an_undecodable_file_cannot_make_the_scan_pass_by_throwing(tmp_path) -> None:
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
    (tmp_path / "src" / "dirty.py").write_bytes(f"import {FORBIDDEN_PACKAGE}\n".encode("utf-16"))

    scan = checker.scan_for_forbidden_package(tmp_path)
    assert scan["files_scanned"] == 2
    assert scan["clean"] is True, "utf-16 is not searched, but it must not crash either"


# --------------------------------------------------------------------------
# Rendering and the CI contract
# --------------------------------------------------------------------------


def test_the_console_rendering_states_the_outcome_it_was_given() -> None:
    clean = checker.format_scan_lines(
        {"roots": ["src"], "missing_roots": [], "files_scanned": 12, "matches": [], "clean": True}
    )
    dirty = checker.format_scan_lines(
        {
            "roots": ["src"],
            "missing_roots": [],
            "files_scanned": 12,
            "matches": ["src/dirty.py:2"],
            "clean": False,
        }
    )
    assert any(line.startswith("PASS") for line in clean)
    assert not any("FAIL" in line for line in clean)
    assert any(line.startswith("FAIL") for line in dirty)
    assert any("src/dirty.py:2" in line for line in dirty)
    assert not any("PASS" in line for line in dirty)


def test_main_exits_non_zero_on_a_hit_and_zero_on_a_clean_tree(tmp_path, capsys) -> None:
    """The exit status is the whole point of a CI gate, so both values are asserted."""

    dirty_root = tmp_path / "dirty"
    clean_root = tmp_path / "clean"
    _tree(dirty_root, {"src/detect.py": f"import {FORBIDDEN_PACKAGE}\n"})
    _tree(clean_root, {"src/detect.py": "import torch\n"})

    assert checker.main(["--project-root", str(dirty_root), "--roots", "src"]) == 1
    dirty_out = capsys.readouterr().out
    assert "src/detect.py:1" in dirty_out
    assert "FAIL" in dirty_out

    assert checker.main(["--project-root", str(clean_root), "--roots", "src"]) == 0
    clean_out = capsys.readouterr().out
    assert "PASS" in clean_out
    assert "FAIL" not in clean_out


def test_main_defaults_to_this_repository_and_finds_it_clean(capsys) -> None:
    """M20's acceptance criterion, executed by the suite instead of quoted in prose.

    No `--project-root`, so this is the same call CI makes. It reads the real
    working tree; if anyone imports the forbidden package, this goes red.
    """

    assert checker.PROJECT_ROOT == PROJECT_ROOT
    assert checker.main([]) == 0
    out = capsys.readouterr().out
    assert "matches         : 0" in out
    assert "NOT SCANNED" not in out, "all three M20 roots must exist and be read"


def test_the_repository_itself_is_free_of_the_forbidden_package() -> None:
    scan = checker.scan_for_forbidden_package(PROJECT_ROOT)
    assert scan["missing_roots"] == []
    assert scan["files_scanned"] > 0
    assert scan["matches"] == []
