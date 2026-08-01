"""PLAN_PHASE2.md M20 / ADR-005: no importable file here may name the forbidden
AGPL-3.0 detector stack.

WHY THIS IS A SCRIPT AND NOT A SENTENCE. `reports/speed_baseline_probe.md` used to
state that this scan "returns NO matches" from a string CONSTANT held in the
benchmark script. Adding a real import of the forbidden package anywhere under
`src/` and regenerating the report left the claim standing word for word, because
no scan had ever run. A verification that cannot fail is not a verification, it is
a decoration. This module performs the scan, returns what it found, and exits
non-zero when it found anything, so CI and the report can both rest on an outcome.

WHY THE SCANNER DOES NOT MATCH ITSELF. The obvious objection to a scanner that
searches for a package name is that its own source contains that name. The answer
is NOT to exempt this file: an exemption is precisely the hole through which a
genuine import added to this file would later escape. The search term is instead
assembled from fragments at run time, so the literal appears in NO source file in
the repository and this scanner is scanned exactly like every other file.

WHY IT MATTERS. ADR-005 rejected the AGPL-3.0 detector stack as the speed baseline
because importing it makes the importing file a derivative work, which is
incompatible with this repository's MIT licence. The rule holds only while it is
enforced against the working tree instead of against memory.

Reports and docs legitimately discuss the forbidden package by name, so only
importable code is scanned. CPU only: this reads text files and loads no model.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The AGPL-3.0 detector package ADR-005 forbids, held as fragments so that the
# literal name never appears in this file. `"".join(...)` runs at call time; the
# file text never contains the term, which is what makes an exemption unnecessary.
_FORBIDDEN_PACKAGE_FRAGMENTS = ("ultra", "lytics")

# PLAN_PHASE2.md M20 states the scan roots: the directories whose contents can be
# imported. `reports/` and `docs/` are prose about the decision, not code bound by
# it, so naming the package there is legitimate and is not scanned.
SCAN_ROOTS = ("src", "scripts", "notebooks")

# A `.pyc` is a build artefact of a `.py` that is itself scanned. Nothing else is
# excluded - not this file, not any other.
SKIP_DIRECTORY_NAMES = ("__pycache__",)


# spec: DEMO-05
def forbidden_package_name() -> str:
    """The AGPL-3.0 package name ADR-005 forbids, assembled at run time."""

    return "".join(_FORBIDDEN_PACKAGE_FRAGMENTS)


# spec: DEMO-05
def scan_for_forbidden_package(
    project_root: Path | str, roots: Sequence[str] = SCAN_ROOTS
) -> dict[str, Any]:
    """Run the scan and return what it actually found.

    The result is structured rather than a verdict string: the roots that were
    visited, the roots that were ASKED FOR but absent (a scan of a directory that
    is not there proves nothing and must not read as a pass), how many files were
    read, and every match as `relative/path.py:LINE`.

    Files are read as bytes and decoded leniently, so an unreadable encoding
    cannot make the check pass by throwing.
    """

    term = forbidden_package_name()
    project_root = Path(project_root)
    matches: list[str] = []
    missing_roots: list[str] = []
    files_scanned = 0

    for name in roots:
        root = project_root / name
        if not root.is_dir():
            missing_roots.append(name)
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRECTORY_NAMES for part in path.parts):
                continue
            files_scanned += 1
            text = path.read_bytes().decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if term in line.lower():
                    matches.append(f"{path.relative_to(project_root).as_posix()}:{number}")

    return {
        "roots": list(roots),
        "missing_roots": missing_roots,
        "files_scanned": files_scanned,
        "matches": matches,
        "clean": not matches,
    }


# spec: DEMO-05
def format_scan_lines(scan: dict[str, Any]) -> list[str]:
    """Plain-text rendering for the console. There is no wording here for a pass
    that the `matches` list does not support."""

    lines = [
        f"roots requested : {', '.join(scan['roots'])}",
        f"files read      : {scan['files_scanned']}",
        f"matches         : {len(scan['matches'])}",
    ]
    if scan["missing_roots"]:
        lines.append(
            f"NOT SCANNED     : {', '.join(scan['missing_roots'])} "
            "(absent from the working tree)"
        )
    lines.extend(f"  {match}" for match in scan["matches"])
    if scan["matches"]:
        lines.append(
            f"FAIL: ADR-005 forbids {forbidden_package_name()!r} in importable code; "
            "importing it would make the importing file a derivative work of an "
            "AGPL-3.0 project and break this repository's MIT licence."
        )
    else:
        lines.append("PASS: no file under those roots mentions the forbidden package.")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ADR-005 / PLAN_PHASE2.md M20: fail when importable code names the "
            f"forbidden AGPL-3.0 package {forbidden_package_name()!r}."
        )
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="tree to scan (default: the repository this script lives in)",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=list(SCAN_ROOTS),
        help="directories under the project root to scan",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Return 0 when the tree is clean and 1 when it is not, so CI can gate on it."""

    args = parse_args(argv)
    root = PROJECT_ROOT if args.project_root is None else Path(args.project_root)
    scan = scan_for_forbidden_package(root, args.roots)
    print(f"forbidden-package scan of {root}")
    for line in format_scan_lines(scan):
        print(line)
    return 0 if scan["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
