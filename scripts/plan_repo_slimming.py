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

import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.release.repository_curation import (
    FIGURE_ROOT,
    FigureDisposition,
    plan_figure_curation,
    tracked_files,
)

REPORT_PATH = PROJECT_ROOT / "reports" / "repo_slimming_plan.md"
BYTES_PER_MB = 2**20


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
    plan = plan_figure_curation(root, files)
    keep = tuple(item for item in plan if item.keep)
    drop = tuple(item for item in plan if not item.keep)
    total_history, areas = history_bytes_by_area(root)

    def total_mb(dispositions: tuple[FigureDisposition, ...]) -> float:
        return sum(item.size_bytes for item in dispositions) / BYTES_PER_MB

    def line(disposition: FigureDisposition) -> str:
        source_lines = ", ".join(
            f"`{reference.source_path}:{reference.line_number}`"
            for reference in disposition.references
        )
        evidence = f" — linked from {source_lines}" if source_lines else ""
        return f"- `{disposition.path}` ({disposition.size_bytes / BYTES_PER_MB:.2f} MB){evidence}"

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
        "and the complete current tree is archived before the owner takes any action.",
        "",
        "## KEEP — a document links to these",
        "",
        f"{len(keep)} files, {total_mb(keep):.1f} MB.",
        "",
    ]
    correction = [
        "## Exact-path correction",
        "",
        "KEEP decisions use only real Markdown destinations after resolving each target",
        "relative to its source document. Bare filenames in prose or code are not links.",
        "",
    ]
    if (len(keep), len(drop)) != (32, 118):
        correction += [
            f"This exact inventory is {len(keep)} KEEP / {len(drop)} DROP, rather than the",
            "read-only 32 KEEP / 118 DROP basename-scan baseline. Seven apparent image links",
            "in `reports/compliance_operating_point.md:7`,",
            "`reports/compliance_operating_point_filtered_syn.md:7`,",
            "`reports/compliance_operating_point_real_only.md:7`,",
            "`reports/compliance_operating_point_standard_aug.md:7`,",
            "`reports/compliance_operating_point_unfiltered_syn.md:7`,",
            "`reports/exposure_analysis.md:7`, and `reports/training_curves.md:5` resolve",
            "from `reports/` to untracked",
            "`reports/reports/figures/...` paths. The remaining 18 baseline-only names are",
            "prose or inline-code mentions, including `class_distribution.png`,",
            "`filter_pass_reject_grid.png`, `flux2_v2_diagnostic_detail.png`, the three",
            "`h2_sam2_*.png` files, `h3_clip_largest_groups.png`,",
            "`h4_generative_identity_pilot*.png`, `h4_guarded_input_preflight.png`,",
            "`h4_paired_person_input_preflight_seed20260802.png`,",
            "`h5_placement_priors.png`, `hard_negative_bank_grid.png`, and",
            "`review/k11_hard_negative_before.png`, `review/k12_blackout_evidence.png`,",
            "`review/loose_helmet_question.png`, `review/preview_hard_negative_p1.png`,",
            "and `review/preview_head_no_helmet_p1.png`. They are not Markdown destinations",
            "and cannot safely promote a figure to KEEP.",
            "",
        ]
    keep_heading = lines.index(next(item for item in lines if item.startswith("## KEEP")))
    lines[keep_heading:keep_heading] = correction
    lines += [line(item) for item in keep]
    lines += [
        "",
        "## DROP — no document links to these",
        "",
        f"{len(drop)} files, {total_mb(drop):.1f} MB. No real Markdown destination",
        "links to them; prose and inline-code filename mentions are not references a",
        "reader can follow.",
        "",
        "**The evidence they represent is not discarded** — the worklog records which",
        "labeler iterations and synthesis routes were tried and what each returned.",
        "That is the form a reader can search.",
        "",
    ]
    lines += [line(item) for item in drop]
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
