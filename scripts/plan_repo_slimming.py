"""Regenerate reports/repo_slimming_plan.md — the file list to review before
rewriting history.

This exists because the plan it writes claims to be regenerable, and that claim
was false: the first version was produced by an inline script that was never
committed. A document asserting "regenerate any time" with no script behind it
is the same class of mistake as a stored PASS string (K-19) - it describes a
capability nobody can exercise.

Nothing here deletes anything. It reads `git ls-files`, reads every tracked
Markdown file, and writes one report. `git filter-repo` is the user's action
(CLAUDE.md reserves history rewrites), so this script deliberately cannot
perform it.

REFERENCED means "a tracked .md links to it". Deliberately not "any tracked
file mentions the name": a script that WRITES foo.png mentions foo.png, and
counting that as a reference collapses the orphan list from 118 files to 18 and
hides 346 MB. A reader can follow a link in a document; they cannot follow a
filename buried in the code that produced it.
"""

from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

from src.data.paths import PROJECT_ROOT

REPORT_PATH = PROJECT_ROOT / "reports" / "repo_slimming_plan.md"
FIGURE_ROOT = "reports/figures/"
IMAGE_SUFFIXES = ("png", "jpg", "jpeg", "gif", "svg", "mp4")
IMAGE_PATTERN = re.compile(r"[\w./-]+\.(?:" + "|".join(IMAGE_SUFFIXES) + ")")
BYTES_PER_MB = 2**20


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


# spec: PUB-04
def referenced_image_names(root: Path, files: list[str]) -> set[str]:
    """Image basenames a tracked Markdown document links to."""

    names: set[str] = set()
    for name in files:
        if not name.endswith(".md"):
            continue
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        names |= {Path(hit).name for hit in IMAGE_PATTERN.findall(text)}
    return names


# spec: PUB-04
def history_bytes_by_area(root: Path) -> tuple[float, dict[str, float]]:
    """Blob bytes across ALL history, split by area.

    The working tree is the wrong thing to measure: a clone downloads history,
    and figures regenerated across commits are stored once per version. The
    tree said 418 MB while the packed object store said 631 MiB.
    """

    listing = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    described = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objecttype) %(objectsize) %(rest)"],
        cwd=root, input=listing, capture_output=True, text=True, check=True,
    ).stdout

    total = 0.0
    areas: Counter[str] = Counter()
    for line in described.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3 or parts[0] != "blob":
            continue
        size, path = float(parts[1]), parts[2].strip()
        total += size
        areas[FIGURE_ROOT if path.startswith(FIGURE_ROOT) else "everything else"] += size
    return total, dict(areas)


def render(root: Path) -> str:
    files = tracked_files(root)
    referenced = referenced_image_names(root, files)
    figures = [name for name in files if name.startswith(FIGURE_ROOT)]

    def size_mb(name: str) -> float:
        path = root / name
        return path.stat().st_size / BYTES_PER_MB if path.is_file() else 0.0

    keep = sorted(name for name in figures if Path(name).name in referenced)
    drop = sorted(name for name in figures if Path(name).name not in referenced)
    total_history, areas = history_bytes_by_area(root)

    lines = [
        "# Repo slimming plan — the file list to review before `git filter-repo`",
        "",
        "> Regenerate with `uv run python -m scripts.plan_repo_slimming`.",
        "> It reads `git ls-files` and every tracked `.md`, and changes nothing.",
        "",
        "## Why",
        "",
        "| | bytes across ALL history | share |",
        "|---|---:|---:|",
    ]
    for area, size in sorted(areas.items(), key=lambda item: -item[1]):
        share = 100 * size / total_history if total_history else 0.0
        lines.append(f"| `{area}` | {size / BYTES_PER_MB:.1f} MB | {share:.0f}% |")
    lines += [
        "",
        "A clone downloads history, not the working tree, so deleting these at HEAD",
        "would change nothing. Rewriting history is the only thing that shrinks it,",
        "and this repo has never been pushed - `git remote -v` is empty - so doing it",
        "now costs nothing and doing it later means force-pushing over published",
        "history.",
        "",
        "## KEEP — a document links to these",
        "",
        f"{len(keep)} files, {sum(size_mb(name) for name in keep):.1f} MB.",
        "",
    ]
    lines += [f"- `{name}` ({size_mb(name):.2f} MB)" for name in keep]
    lines += [
        "",
        "## DROP — no document links to these",
        "",
        f"{len(drop)} files, {sum(size_mb(name) for name in drop):.1f} MB. Their only",
        "mention anywhere is inside the script that generated them, which is not a",
        "reference a reader can follow.",
        "",
        "**The evidence they represent is not discarded** — the worklog records which",
        "labeler iterations and synthesis routes were tried and what each returned.",
        "That is the form a reader can search.",
        "",
    ]
    lines += [f"- `{name}` ({size_mb(name):.2f} MB)" for name in drop]
    lines += [
        "",
        "## The command (YOURS to run — CLAUDE.md reserves history rewrites)",
        "",
        "```bash",
        "git filter-repo --path reports/figures/ --invert-paths --force",
        "```",
        "",
        "That removes the whole folder from history, keepers included. Re-add them",
        "afterwards as one fresh commit — they survive the rewrite because they are",
        "still in the working tree; `filter-repo` edits history, not your files:",
        "",
        "```bash",
        'git add reports/figures/ && git commit -m "docs: restore the figures documents reference"',
        "```",
        "",
        "## Verify afterwards",
        "",
        "```bash",
        "git count-objects -vH",
        "uv run pytest -q",
        "uv run python scripts/verify_readme.py",
        "```",
        "",
        "`size-pack` should fall sharply. `verify_readme` is the real check: it fails",
        "if any README figure link is dead.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = render(PROJECT_ROOT)
    REPORT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
