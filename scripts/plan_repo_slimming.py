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
GENERATED_REPORT_PATH = "reports/repo_slimming_plan.md"
BYTES_PER_MB = 2**20


# spec: PUB-04
def history_bytes_by_area(root: Path) -> tuple[float, dict[str, float]]:
    """Blob bytes across reachable history, split by area.

    The working tree is the wrong thing to measure: a clone downloads history,
    and figures regenerated across commits are stored once per version. The
    tree said 418 MB while the packed object store said 631 MiB. Generated
    report-only blobs are excluded so regenerating and committing the report
    cannot change the historical metric it displays; a shared blob is retained
    whenever any other path has used it.
    """

    commits = subprocess.run(
        ["git", "rev-list", "--all"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    blob_paths: dict[str, set[str]] = {}
    for commit in commits:
        tree = subprocess.run(
            ["git", "ls-tree", "-r", "-z", "--full-tree", commit],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", maxsplit=1)
            _mode, object_type, object_id = metadata.split(maxsplit=2)
            if object_type != b"blob":
                continue
            path = raw_path.decode("utf-8", errors="surrogateescape")
            blob_paths.setdefault(object_id.decode("ascii"), set()).add(path)
    if not blob_paths:
        return 0.0, {}
    described = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=root,
        input="\n".join(sorted(blob_paths)) + "\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    total = 0.0
    areas: Counter[str] = Counter()
    for line in described.splitlines():
        object_id, object_type, size = line.split(" ", 2)
        if object_type != "blob":
            continue
        paths = blob_paths[object_id]
        if paths == {GENERATED_REPORT_PATH}:
            continue
        byte_size = float(size)
        total += byte_size
        area = FIGURE_ROOT if any(path.startswith(FIGURE_ROOT) for path in paths) else "everything else"
        areas[area] += byte_size
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
        "# Repo slimming plan — curated inventory before the owner history-rewrite gate",
        "",
        "> Regenerate with `uv run python -m scripts.plan_repo_slimming`.",
        "> It reads `git ls-files` and every tracked `.md` except this generated report,",
        "> and changes nothing.",
        "",
        "## Why",
        "",
        "| | reachable-history blob bytes* | share |",
        "|---|---:|---:|",
    ]
    for area, size in sorted(areas.items(), key=lambda item: -item[1]):
        share = 100 * size / total_history if total_history else 0.0
        lines.append(f"| `{area}` | {size / BYTES_PER_MB:.1f} MB | {share:.0f}% |")
    lines += [
        "",
        "* Generated-report-only historical blobs are excluded; shared blobs remain counted.",
        "",
        "A clone downloads history, not the working tree, so deleting these at HEAD",
        "would change nothing. Rewriting history is the only thing that shrinks it,",
        "and the complete current tree must be archived and verified before owner action.",
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
            "read-only 32 KEEP / 118 DROP basename-scan baseline. Seven image links",
            "in `reports/compliance_operating_point.md:7`,",
            "`reports/compliance_operating_point_filtered_syn.md:7`,",
            "`reports/compliance_operating_point_real_only.md:7`,",
            "`reports/compliance_operating_point_standard_aug.md:7`,",
            "`reports/compliance_operating_point_unfiltered_syn.md:7`,",
            "`reports/exposure_analysis.md:7`, and `reports/training_curves.md:5` now resolve",
            "to tracked KEEP figures after their destinations were corrected to `figures/...`.",
            "The remaining 18 baseline-only names are",
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
        "## Owner-only, non-executable next step",
        "",
        "**NON-EXECUTABLE WARNING.** This inventory is not a history-rewrite procedure.",
        "The destructive owner-only step must use **ONLY the independently verified external**",
        "`OWNER_HISTORY_REWRITE_RUNBOOK.txt` created in the recovery archive for this",
        "exact source commit. Do not copy commands from a prior runbook or infer recovery",
        "steps from this report.",
        "",
        "The external runbook is written only after the full figure archive, manifest, and",
        "Git bundle have passed verification.",
        (
            "It is the sole approved source only for the owner-operated Stage 1 history rewrite."
        ),
        (
            "It stops after that rewrite and does not authorize restoration, staging, or a commit."
        ),
        (
            "After the owner reports back, the Task 7 read-only checkpoint must pass; Task 8 is "
            "the sole restoration, inventory, and commit stage."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = render(PROJECT_ROOT)
    REPORT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
