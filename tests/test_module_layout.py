"""Nothing may be defined after the __main__ guard (K-16).

Python executes module level top to bottom, so a `def` placed below
`if __name__ == "__main__": main()` does not exist when main() runs. The failure
mode is nasty: it survives every unit test (tests never execute the guard) and
only fires at the end of a long script run, after the primary artifact has
already been written. It cost a 20-minute H4 run, and the same latent break was
sitting in five other scripts patched the same way.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(
    path
    for directory in ("scripts", "src")
    for path in (ROOT / directory).rglob("*.py")
)


def _main_guard_line(tree: ast.Module) -> int | None:
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("__name__"):
            return node.lineno
    return None


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_nothing_is_defined_after_the_main_guard(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guard = _main_guard_line(tree)
    if guard is None:
        pytest.skip("no __main__ guard")

    late = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.lineno > guard
    ]

    assert not late, (
        f"{path.relative_to(ROOT)} defines {late} after the __main__ guard, "
        "so they are unbound when main() runs"
    )


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_duplicate_top_level_definitions(path: Path) -> None:
    """The same patch was applied twice to one file, leaving two copies."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert not duplicates, f"{path.relative_to(ROOT)} defines {duplicates} more than once"
