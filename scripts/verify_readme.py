"""PUB-01..PUB-05 / PUB-10: check README.md against the repository, loudly.

A README is the one artefact a reader trusts without running anything, so every
claim in it has to be mechanically checkable or it is decoration. Six checks:

  1. Every number in a README *metric* table is recomputable from
     results/detection_metrics.csv. A cell that cannot be traced to a row of
     that file is a FAILURE, not a warning - an untraceable number is exactly
     what PUB-04 exists to catch, and downgrading it to a warning turns the
     whole check into a comment.
  2. Four disclosures are present in the BODY, not only under Limitations: the
     annotation defect, that every claim is relative and never absolute, why the
     design has four arms and not five, and the H4 outcome.
  3. No shields.io badges (static badges assert a state nothing verifies).
  4. No Chinese "unfinished" placeholders in README.md, PLAN*.md or docs/*.md.
  5. Every relative markdown link in README.md resolves to a real file.
  6. No leaked identifiers: this machine's user name, the git email local part,
     or a long provider-token-shaped string.

WHY THE MISSING-RESULTS CASE IS NOT A PASS. Before scripts/eval.py has ever run,
results/detection_metrics.csv does not exist and check 1 has nothing to compare
against. Exiting 0 there would report "README verified" about a README whose
numbers were never looked at. It exits EXIT_NOTHING_TO_VERIFY instead, which a
caller can tell apart from both success and failure.

CPU only, no model loads, no network. Reads text files and one CSV.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# PLAN_PHASE2.md M23 invokes this as `uv run python scripts/verify_readme.py`,
# which puts scripts/ - not the repository root - on sys.path, so `import src`
# would fail. Prepending the root keeps BOTH that invocation and
# `python -m scripts.verify_readme` working.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imported after the sys.path fix above, which is why it is not at the top.
from src.evaluation.detection import (
    DETECTION_METRICS_FILENAME,
    MetricRow,
    read_detection_metrics_csv,
)

# --------------------------------------------------------------------------
# Exit codes. 0 and 1 are the usual pass/fail; 2 is reserved because argparse
# already uses it for a usage error, so "nothing to verify yet" takes 3.
# --------------------------------------------------------------------------
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOTHING_TO_VERIFY = 3

# Tolerance for "equal after rounding to the precision the README shows".
# A displayed 0.3564 stands for anything in [0.35635, 0.35645), i.e. half a unit
# in the last displayed place, so the comparison is half-a-ULP wide rather than
# tied to whichever rounding convention the author's tool used. The extra
# relative slack only absorbs float round-trip noise.
_ROUNDING_SLACK = 1e-9

# `hf_transfer` and `sk-learn` are ordinary words; a real credential is long.
# The 20-character floor is the one publish-repo gate 1 already uses for the
# same reason, kept identical so the two scans agree on what a token looks like.
_TOKEN_BODY_MIN_LENGTH = 20

# A local identifier shorter than this cannot be searched for without matching
# ordinary prose ("root", "usr"), and CI runners are called "runner", which
# would otherwise flag every README that says "the runner".
_IDENTIFIER_MIN_LENGTH = 4
_GENERIC_ACCOUNT_NAMES = frozenset(
    {"runner", "runneradmin", "root", "user", "users", "admin", "administrator", "github"}
)

# A GitHub noreply address exists to be published - it is the identity every
# commit in this repository is signed with, and PUB-10 requires it to appear.
# Searching the docs for its local part would flag the repository owner's own
# public name and make the check permanently red for the one address that is
# supposed to be there.
_PUBLISHED_EMAIL_DOMAINS = frozenset({"users.noreply.github.com"})

# The exact list already in use in this repository: PLAN.md M0 requires a scan
# for "words expressing an unfinished state", and the publish-repo skill spells
# that scan out as this grep. Matched case-sensitively, exactly as that grep
# runs, so an English word like "todo" inside a URL is not a false positive.
FORBIDDEN_WORDS = ("下一步", "尚未", "待驗收", "待確認", "TODO", "待補")

# docs/worklog.md keeps a standing "what the user still has to do" bullet. That
# bullet is a handover queue, not an unfinished document, so it is exempt - and
# only it: the exemption starts at this marker and ends at the next top-level
# bullet, heading or rule, so a placeholder elsewhere in the worklog is caught.
WORKLOG_RELATIVE_PATH = "docs/worklog.md"
WORKLOG_EXEMPT_MARKER = "等使用者做的事"

FORBIDDEN_BADGE_HOST = "shields.io"

LIMITATIONS_HEADING = re.compile(r"limitation|限制", re.IGNORECASE)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")

# A leading sign is only a sign when it does not follow a digit, so the ASCII
# range "0.9013-0.9090" reads as two positive numbers rather than one positive
# and one negative.
_NUMBER = re.compile(r"(?<![\d.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?")
_PLUS_MINUS = re.compile(r"±|\+/-|\\pm")

# <!--metric: primary_map_small--> style overrides, so a README can keep a
# readable column title and still name its source unambiguously.
_ANNOTATION = re.compile(r"<!--\s*([a-z_]+)\s*:\s*([^>]*?)\s*-->")
_SKIP_ANNOTATION = re.compile(r"<!--\s*(no-source|skip)\s*-->")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# Inline links and images. Reference-style definitions are not used in this
# README; if that changes the check has to grow rather than silently skip them.
_INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(<[^>]*>|[^()\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)")
_ABSOLUTE_TARGET = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)

_TOKEN_SHAPED = re.compile(
    rf"(?:gho_|hf_|sk-|AIza)[A-Za-z0-9_\-]{{{_TOKEN_BODY_MIN_LENGTH},}}"
)


@dataclass(frozen=True)
class Failure:
    """One reason the README does not verify. Every one of these is printed."""

    check: str
    location: str
    message: str

    def render(self) -> str:
        return f"FAIL [{self.check}] {self.location}: {self.message}"


@dataclass(frozen=True)
class Number:
    """A numeric literal lifted out of a table cell, with how it was written.

    `precision` is the count of digits after the decimal point AS SHOWN, which
    is what decides how close the source value has to be. `scale` is 100 for a
    percentage, because the CSV stores fractions.
    """

    text: str
    value: float
    precision: int
    scale: float


@dataclass(frozen=True)
class MarkdownTable:
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    header_line: int
    row_lines: tuple[int, ...]


@dataclass(frozen=True)
class DisclosureTopic:
    """A required disclosure, expressed as several patterns that must ALL hit.

    One substring is too brittle to stand for a topic - rewording the sentence
    would silently drop the requirement - so each topic is a conjunction of
    small alternations. `groups` holds the conjuncts; each conjunct is an
    alternation of acceptable wordings.
    """

    key: str
    description: str
    groups: tuple[tuple[str, ...], ...]


# The literals below are not tunable parameters; they are the exact strings the
# disclosure has to contain, taken from CLAUDE.md's project table (the SHEL5K
# re-annotation counts) and from docs/release_spec.md PUB-02/PUB-03. Writing
# them anywhere but here would mean checking a different claim than the one the
# specification names.
DISCLOSURE_TOPICS: tuple[DisclosureTopic, ...] = (
    DisclosureTopic(
        key="annotation_defect",
        description=(
            "PUB-02: SHEL5K re-annotated the same images and found 75,570 labels "
            "against the original 25,502"
        ),
        groups=(
            (r"shel5k",),
            (r"75[,\s]?570",),
            (r"25[,\s]?502",),
        ),
    ),
    DisclosureTopic(
        key="relative_claims",
        description="PUB-02: every claim is relative (arm vs arm), never an absolute AP",
        groups=(
            (r"relativ",),
            (r"\b(every|all|each)\s+claim", r"claims?\s+(in|here|are)"),
            (r"never\s+absolute", r"not\s+absolute", r"absolute\s+ap"),
        ),
    ),
    DisclosureTopic(
        key="four_arm_design",
        description=(
            "PUB-03: why the design is four arms and not five - Real-only already "
            "trains on all real Train data, so there is no higher real-data ceiling"
        ),
        groups=(
            (r"four[\s-]arm", r"four\s+arms", r"4\s+arms", r"四組"),
            (r"fifth", r"five[\s-]arm", r"five\s+arms", r"5\s+arms", r"第五組"),
            (
                r"all\s+(the\s+)?real",
                r"every\s+real",
                r"full[\s-]real",
                r"real[\s-]data\s+ceiling",
                r"no\s+higher",
            ),
        ),
    ),
    DisclosureTopic(
        key="h4_result",
        description="ADR-011: the H4 gate, its measured AUC, and that it did not pass",
        groups=(
            (r"\bh4\b",),
            (r"auc[^0-9]{0,24}0\.\d+",),
            (r"did\s+not\s+pass", r"does\s+not\s+claim", r"not\s+pass", r"fail"),
            (r"pre[\s-]?regist",),
        ),
    ),
)


# --------------------------------------------------------------------------
# Markdown structure
# --------------------------------------------------------------------------
def lines_outside_code_fences(text: str) -> list[tuple[int, str]]:
    """(1-based line number, text) with fenced code replaced by blank lines.

    Blanked rather than dropped so that line numbers stay true AND a fenced
    block still breaks a table in two; dropping the lines would let a table
    above a code block merge with one below it.
    """

    result: list[tuple[int, str]] = []
    inside = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            inside = not inside
            result.append((number, ""))
            continue
        result.append((number, "" if inside else line))
    return result


def split_cells(row: str) -> tuple[str, ...]:
    """Split a markdown table row on unescaped pipes, dropping the edge pipes."""

    stripped = row.strip()
    parts = _UNESCAPED_PIPE.split(stripped)
    if stripped.startswith("|"):
        parts = parts[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|") and parts:
        parts = parts[:-1]
    return tuple(part.strip().replace("\\|", "|") for part in parts)


def parse_markdown_tables(text: str) -> list[MarkdownTable]:
    """Every pipe table in the document, outside code fences.

    A table is a header row followed by a delimiter row; anything else that
    happens to contain a pipe is not one.
    """

    numbered = lines_outside_code_fences(text)
    tables: list[MarkdownTable] = []
    index = 0
    while index < len(numbered) - 1:
        number, line = numbered[index]
        following = numbered[index + 1][1]
        if "|" not in line or not _TABLE_DELIMITER.match(following) or "|" not in following:
            index += 1
            continue
        header = split_cells(line)
        rows: list[tuple[str, ...]] = []
        row_lines: list[int] = []
        cursor = index + 2
        while cursor < len(numbered) and "|" in numbered[cursor][1]:
            rows.append(split_cells(numbered[cursor][1]))
            row_lines.append(numbered[cursor][0])
            cursor += 1
        tables.append(
            MarkdownTable(
                header=header,
                rows=tuple(rows),
                header_line=number,
                row_lines=tuple(row_lines),
            )
        )
        index = cursor
    return tables


def body_excluding_limitations(text: str) -> str:
    """The README with every Limitations section removed.

    PUB-02 says the disclosures must be in the body, so the check has to read a
    document that no longer contains the section where they would otherwise be
    parked. A section runs until the next heading at the same or shallower
    level, so a nested subsection stays excluded with its parent.
    """

    kept: list[str] = []
    skip_level: int | None = None
    for _, line in lines_outside_code_fences(text):
        heading = _HEADING.match(line)
        if heading is not None:
            level = len(heading.group(1))
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and LIMITATIONS_HEADING.search(heading.group(2)):
                skip_level = level
                continue
        if skip_level is None:
            kept.append(line)
    return "\n".join(kept)


# --------------------------------------------------------------------------
# Numbers and annotations
# --------------------------------------------------------------------------
def annotation(cell: str, key: str) -> str | None:
    """Read `<!-- key: value -->` out of a cell, if present."""

    for match in _ANNOTATION.finditer(cell):
        if match.group(1) == key:
            return match.group(2)
    return None


def strip_annotations(cell: str) -> str:
    return _COMMENT.sub(" ", cell)


def extract_numbers(cell: str) -> list[Number]:
    """Numeric literals in a cell, in order, with their displayed precision."""

    text = strip_annotations(cell)
    numbers: list[Number] = []
    for match in _NUMBER.finditer(text):
        literal = match.group(0)
        digits = literal.split(".")
        precision = len(digits[1]) if len(digits) > 1 else 0
        rest = text[match.end() :].lstrip()
        numbers.append(
            Number(
                text=literal,
                value=float(literal.replace(",", "")),
                precision=precision,
                scale=100.0 if rest.startswith("%") else 1.0,
            )
        )
    return numbers


def matches_at_shown_precision(source_value: float, shown: Number) -> bool:
    """Does `source_value` round to what the README printed?

    Half a unit in the last displayed place, so 0.35635 and 0.35644 both round
    to a displayed 0.3564 whichever way the author's formatter breaks ties.
    """

    scaled = source_value * shown.scale
    tolerance = 0.5 * 10.0 ** (-shown.precision) + _ROUNDING_SLACK * max(1.0, abs(scaled))
    return abs(scaled - shown.value) <= tolerance


def normalise_label(text: str) -> str:
    """Fold a human column title down to an identifier: '+ Standard Aug' -> 'standard_aug'."""

    cleaned = strip_annotations(text).replace("*", " ").replace("`", " ")
    return re.sub(r"[^0-9a-z]+", "_", cleaned.lower()).strip("_")


def token_prefix_match(candidate: str, label: str) -> bool:
    """Same token count, each token a prefix of the other's.

    Lets a README write "Unfiltered Synthetic" for the arm the CSV calls
    `unfiltered_syn` without a hand-maintained alias table, while still refusing
    "filtered" for "unfiltered_syn" - abbreviation is a prefix relation, and a
    different word is not.
    """

    left = candidate.split("_")
    right = label.split("_")
    if len(left) != len(right) or not label or not candidate:
        return False
    return all(a.startswith(b) or b.startswith(a) for a, b in zip(left, right, strict=True))


def resolve_arm(label: str, arms: Sequence[str]) -> tuple[str | None, list[str]]:
    """Which CSV arm a table label names. Returns (arm, all candidates)."""

    normalised = normalise_label(label)
    if not normalised:
        return None, []
    exact = [arm for arm in arms if normalise_label(arm) == normalised]
    if len(exact) == 1:
        return exact[0], exact
    candidates = [arm for arm in arms if token_prefix_match(normalise_label(arm), normalised)]
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def resolve_metric(label: str, metrics: Mapping[str, str]) -> str | None:
    """Which CSV metric a table label names, by exact normalised name only.

    Deliberately NOT fuzzy. "AP_small" is ambiguous between `map_small` (all
    three classes) and `primary_map_small` (helmet+head, the headline), and
    picking one silently is precisely the mutation K-19 was written about. An
    unresolvable label is reported so the author annotates the cell.
    """

    return metrics.get(normalise_label(label))


# --------------------------------------------------------------------------
# Check 1 - table numbers against results/detection_metrics.csv
# --------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _spreads(values: Sequence[float], rows: Sequence[MetricRow]) -> list[float]:
    """Every quantity a "value ± x" cell could plausibly be showing."""

    spreads: list[float] = []
    if len(values) > 1:
        mean = _mean(values)
        squares = sum((value - mean) ** 2 for value in values)
        spreads.append(math.sqrt(squares / len(values)))
        spreads.append(math.sqrt(squares / (len(values) - 1)))
    spreads.extend(
        (row.ci_high - row.ci_low) / 2.0
        for row in rows
        if row.ci_low is not None and row.ci_high is not None
    )
    return spreads


def _points(values: Sequence[float]) -> list[float]:
    return [*values, _mean(values)] if len(values) > 1 else list(values)


def _describe(rows: Sequence[MetricRow]) -> str:
    return ", ".join(f"seed={row.seed} value={row.value!r}" for row in rows)


def _check_cell(
    *,
    cell: str,
    arm: str,
    metric: str,
    split: str | None,
    rows: Sequence[MetricRow],
    location: str,
) -> list[Failure]:
    """Compare one cell's numbers against the rows that source it."""

    candidates = [row for row in rows if row.arm == arm and row.metric == metric]
    if split is not None:
        candidates = [row for row in candidates if row.split == split]
    if not candidates:
        return [
            Failure(
                "table-numbers",
                location,
                f"no row in {DETECTION_METRICS_FILENAME} has arm={arm!r} metric={metric!r}"
                + ("" if split is None else f" split={split!r}"),
            )
        ]
    splits = sorted({row.split for row in candidates})
    if len(splits) > 1:
        return [
            Failure(
                "table-numbers",
                location,
                f"arm={arm!r} metric={metric!r} exists on splits {splits}; add a "
                "<!--split: test--> annotation so the cell names one of them",
            )
        ]

    numbers = extract_numbers(cell)
    values = [row.value for row in candidates]
    points = _points(values)
    matched = [
        row for row in candidates if matches_at_shown_precision(row.value, numbers[0])
    ]
    if not any(matches_at_shown_precision(point, numbers[0]) for point in points):
        return [
            Failure(
                "table-numbers",
                location,
                f"shows {numbers[0].text!r} for arm={arm!r} metric={metric!r}, but "
                f"{DETECTION_METRICS_FILENAME} has {_describe(candidates)}"
                + (f" (mean {_mean(values)!r})" if len(values) > 1 else ""),
            )
        ]

    if len(numbers) == 1:
        return []
    if len(numbers) == 2 and _PLUS_MINUS.search(strip_annotations(cell)):
        spreads = _spreads(values, candidates)
        if any(matches_at_shown_precision(spread, numbers[1]) for spread in spreads):
            return []
        return [
            Failure(
                "table-numbers",
                location,
                f"shows a spread of {numbers[1].text!r} for arm={arm!r} metric={metric!r}, "
                f"but the source supports {spreads or 'no spread at all (one seed, no CI)'}",
            )
        ]
    if len(numbers) == 3:
        for row in matched:
            if row.ci_low is None or row.ci_high is None:
                continue
            if matches_at_shown_precision(row.ci_low, numbers[1]) and (
                matches_at_shown_precision(row.ci_high, numbers[2])
            ):
                return []
        return [
            Failure(
                "table-numbers",
                location,
                f"shows the interval ({numbers[1].text}, {numbers[2].text}) for arm={arm!r} "
                f"metric={metric!r}, which no matching row's ci_low/ci_high reproduces",
            )
        ]
    return [
        Failure(
            "table-numbers",
            location,
            f"has {len(numbers)} numbers and no interpretation; a metric cell may hold a "
            "value, a value ± spread, or a value with a two-number interval",
        )
    ]


def check_table_numbers(
    readme_text: str, rows: Sequence[MetricRow], source_name: str = "README.md"
) -> list[Failure]:
    """PUB-04: every number in a metric table traces back to the CSV.

    A table counts as a metric table when one of its row labels or column
    headers names an arm; that keeps a dataset-composition table out of scope
    without a hand-maintained allow-list. Inside one, EVERY numeric cell must
    resolve - that is the half of the check that catches a number nobody
    computed.
    """

    failures: list[Failure] = []
    arms = sorted({row.arm for row in rows})
    metrics = {normalise_label(row.metric): row.metric for row in rows}

    for table in parse_markdown_tables(readme_text):
        header_arms = [resolve_arm(cell, arms)[0] for cell in table.header]
        row_arms = [resolve_arm(row[0], arms)[0] if row else None for row in table.rows]
        arms_in_rows = any(arm is not None for arm in row_arms)
        arms_in_columns = any(arm is not None for arm in header_arms)
        if not arms_in_rows and not arms_in_columns:
            continue

        for row_index, row in enumerate(table.rows):
            line = table.row_lines[row_index]
            for column, cell in enumerate(row):
                if column == 0 or not cell:
                    continue
                location = f"{source_name}:{line} column {column + 1}"
                if _SKIP_ANNOTATION.search(cell):
                    continue
                if not extract_numbers(cell):
                    continue
                header = table.header[column] if column < len(table.header) else ""

                if arms_in_rows:
                    arm = annotation(cell, "arm") or row_arms[row_index]
                    metric_label = annotation(cell, "metric") or annotation(header, "metric")
                    metric = metrics.get(metric_label) if metric_label else None
                    if metric is None and metric_label is None:
                        metric = resolve_metric(header, metrics)
                    unresolved_label, kind = header, "column header"
                else:
                    # A row may be wider than the header - markdown does not
                    # forbid it - and indexing past the end would crash the
                    # verifier instead of reporting the ragged table.
                    column_arm = header_arms[column] if column < len(header_arms) else None
                    arm = annotation(cell, "arm") or column_arm
                    metric_label = annotation(cell, "metric") or annotation(row[0], "metric")
                    metric = metrics.get(metric_label) if metric_label else None
                    if metric is None and metric_label is None:
                        metric = resolve_metric(row[0], metrics)
                    unresolved_label, kind = row[0], "row label"

                if arm is None:
                    failures.append(
                        Failure(
                            "table-numbers",
                            location,
                            f"holds a number but no arm applies to it (known arms: {arms})",
                        )
                    )
                    continue
                if metric is None:
                    failures.append(
                        Failure(
                            "table-numbers",
                            location,
                            f"{kind} {unresolved_label.strip()!r} does not name a metric in "
                            f"{DETECTION_METRICS_FILENAME}; use the CSV name or annotate the "
                            f"cell with <!--metric: ...-->. Known metrics: "
                            f"{sorted(metrics.values())}",
                        )
                    )
                    continue

                split = (
                    annotation(cell, "split")
                    or annotation(header, "split")
                    or annotation(row[0], "split")
                )
                failures.extend(
                    _check_cell(
                        cell=cell,
                        arm=arm,
                        metric=metric,
                        split=split,
                        rows=rows,
                        location=location,
                    )
                )
    return failures


# --------------------------------------------------------------------------
# Check 2 - required disclosures
# --------------------------------------------------------------------------
def check_required_disclosures(
    readme_text: str,
    topics: Sequence[DisclosureTopic] = DISCLOSURE_TOPICS,
    source_name: str = "README.md",
) -> list[Failure]:
    """PUB-02 / PUB-03: the four disclosures, in the body and not only in Limitations."""

    body = body_excluding_limitations(readme_text)
    failures: list[Failure] = []
    for topic in topics:
        for group in topic.groups:
            if any(re.search(pattern, body, re.IGNORECASE) for pattern in group):
                continue
            failures.append(
                Failure(
                    "disclosure",
                    f"{source_name} (body, excluding Limitations)",
                    f"topic {topic.key!r} is incomplete - {topic.description}. Nothing "
                    f"outside Limitations matches any of {list(group)}",
                )
            )
    return failures


# --------------------------------------------------------------------------
# Check 3 - no static badges
# --------------------------------------------------------------------------
def check_no_static_badges(readme_text: str, source_name: str = "README.md") -> list[Failure]:
    """PUB-04: a shields.io badge asserts a state that nothing in this repo measures."""

    return [
        Failure(
            "static-badge",
            f"{source_name}:{number}",
            f"{FORBIDDEN_BADGE_HOST} badge - a static badge claims a state no check verifies",
        )
        for number, line in enumerate(readme_text.splitlines(), start=1)
        if FORBIDDEN_BADGE_HOST in line
    ]


# --------------------------------------------------------------------------
# Check 4 - forbidden placeholder words
# --------------------------------------------------------------------------
def worklog_exempt_lines(lines: Sequence[str]) -> set[int]:
    """1-based line numbers of the worklog's "waiting on the user" bullet.

    The block is the marker bullet plus its indented continuation lines, and
    ends at the next top-level bullet, heading or horizontal rule. Exempting the
    whole file would let a real placeholder hide two sections away.
    """

    exempt: set[int] = set()
    inside = False
    for number, line in enumerate(lines, start=1):
        if WORKLOG_EXEMPT_MARKER in line:
            inside = True
            exempt.add(number)
            continue
        if not inside:
            continue
        stripped = line.strip()
        starts_block = (
            line[:1] in {"-", "*", "#"} or stripped.startswith("---") or not stripped
        )
        if starts_block:
            inside = False
            continue
        exempt.add(number)
    return exempt


def check_forbidden_words(
    documents: Mapping[str, str], words: Sequence[str] = FORBIDDEN_WORDS
) -> list[Failure]:
    """PLAN.md M0: no "unfinished" placeholder survives into published prose."""

    failures: list[Failure] = []
    for name in sorted(documents):
        lines = documents[name].splitlines()
        exempt = worklog_exempt_lines(lines) if name == WORKLOG_RELATIVE_PATH else set()
        for number, line in enumerate(lines, start=1):
            if number in exempt:
                continue
            for word in words:
                if word in line:
                    failures.append(
                        Failure(
                            "forbidden-word",
                            f"{name}:{number}",
                            f"{word!r} states an unfinished status; express status with "
                            "[ ] / [~] / [x] instead",
                        )
                    )
    return failures


# --------------------------------------------------------------------------
# Check 5 - relative links resolve
# --------------------------------------------------------------------------
def relative_link_targets(text: str) -> list[tuple[int, str]]:
    """(line number, target) for every inline link that points inside the repo."""

    targets: list[tuple[int, str]] = []
    for number, line in lines_outside_code_fences(text):
        for match in _INLINE_LINK.finditer(line):
            target = match.group(1).strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            if not target or target.startswith("#"):
                continue
            if _ABSOLUTE_TARGET.match(target) or target.startswith("//"):
                continue
            targets.append((number, target))
    return targets


def check_relative_links(
    readme_text: str, base_directory: Path, source_name: str = "README.md"
) -> list[Failure]:
    """PUB-01: a broken relative link is a claim that a document exists when it does not."""

    failures: list[Failure] = []
    for number, target in relative_link_targets(readme_text):
        path_part = target.split("#", 1)[0].split("?", 1)[0]
        if not path_part:
            continue
        resolved = base_directory / path_part.replace("%20", " ")
        if not resolved.exists():
            failures.append(
                Failure(
                    "broken-link",
                    f"{source_name}:{number}",
                    f"{target!r} does not resolve ({resolved})",
                )
            )
    return failures


# --------------------------------------------------------------------------
# Check 6 - leaked identifiers
# --------------------------------------------------------------------------
def git_config_emails(project_root: Path) -> list[str]:
    """The repo-local AND global commit emails, or [] when git cannot answer.

    Read at run time rather than written into this file: hard-coding the address
    a scanner looks for would leak it into the very repository the scan protects.
    Both scopes are read because the local one is usually the deliberate public
    identity while the global one is the operator's real address - and it is the
    real address that must never appear in a document.
    """

    emails: list[str] = []
    for scope in ([], ["--global"]):
        try:
            completed = subprocess.run(
                ["git", "config", *scope, "user.email"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
        except OSError:
            continue
        value = (completed.stdout or "").strip()
        if value:
            emails.append(value)
    return emails


def collect_local_identifiers(
    environment: Mapping[str, str], git_emails: Sequence[str], home_name: str = ""
) -> list[str]:
    """Strings that identify this machine's operator and must not reach the docs."""

    raw = [environment.get("USERNAME", ""), environment.get("USER", ""), home_name]
    for email in git_emails:
        local, separator, domain = email.strip().partition("@")
        if not separator or domain.lower() in _PUBLISHED_EMAIL_DOMAINS:
            continue
        raw.append(local)
    identifiers = {
        value
        for value in (item.strip() for item in raw)
        if len(value) > _IDENTIFIER_MIN_LENGTH and value.lower() not in _GENERIC_ACCOUNT_NAMES
    }
    return sorted(identifiers)


def check_no_leaked_identifiers(
    documents: Mapping[str, str], identifiers: Sequence[str]
) -> list[Failure]:
    """PUB-10: no operator identity and no provider-token-shaped string in the docs."""

    failures: list[Failure] = []
    for name in sorted(documents):
        for number, line in enumerate(documents[name].splitlines(), start=1):
            lowered = line.lower()
            for identifier in identifiers:
                if identifier.lower() in lowered:
                    failures.append(
                        Failure(
                            "leaked-identifier",
                            f"{name}:{number}",
                            f"contains the local account identifier {identifier!r}",
                        )
                    )
            for match in _TOKEN_SHAPED.finditer(line):
                failures.append(
                    Failure(
                        "leaked-identifier",
                        f"{name}:{number}",
                        f"contains a credential-shaped string starting {match.group(0)[:6]!r}",
                    )
                )
    return failures


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def documents_to_scan(project_root: Path) -> dict[str, str]:
    """README.md, PLAN*.md and docs/*.md, keyed by repo-relative posix path."""

    paths: list[Path] = []
    readme = project_root / "README.md"
    if readme.is_file():
        paths.append(readme)
    paths.extend(sorted(project_root.glob("PLAN*.md")))
    paths.extend(sorted((project_root / "docs").glob("*.md")))
    return {
        path.relative_to(project_root).as_posix(): path.read_text(encoding="utf-8")
        for path in paths
    }


@dataclass(frozen=True)
class VerificationResult:
    failures: tuple[Failure, ...]
    notes: tuple[str, ...]
    metrics_present: bool

    def exit_code(self) -> int:
        if self.failures:
            return EXIT_FAILED
        return EXIT_OK if self.metrics_present else EXIT_NOTHING_TO_VERIFY


def verify(
    project_root: Path,
    *,
    metrics_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    git_emails: Sequence[str] | None = None,
) -> VerificationResult:
    """Run every check and collect every failure. Nothing short-circuits."""

    project_root = Path(project_root)
    readme_path = project_root / "README.md"
    if not readme_path.is_file():
        return VerificationResult(
            failures=(Failure("readme", str(readme_path), "README.md does not exist"),),
            notes=(),
            metrics_present=False,
        )
    readme_text = readme_path.read_text(encoding="utf-8")
    documents = documents_to_scan(project_root)

    if metrics_path is None:
        metrics_path = project_root / "results" / DETECTION_METRICS_FILENAME

    failures: list[Failure] = []
    notes: list[str] = []

    metrics_present = metrics_path.is_file()
    if metrics_present:
        rows = read_detection_metrics_csv(metrics_path)
        failures.extend(check_table_numbers(readme_text, rows))
        notes.append(
            f"{metrics_path.name}: {len(rows)} rows, "
            f"arms {sorted({row.arm for row in rows})}"
        )
    else:
        notes.append(
            f"{metrics_path} is absent, so no README number was checked against a "
            "source. This is NOT a pass - see the exit code."
        )

    failures.extend(check_required_disclosures(readme_text))
    failures.extend(check_no_static_badges(readme_text))
    failures.extend(check_forbidden_words(documents))
    failures.extend(check_relative_links(readme_text, project_root))

    if environment is None:
        environment = os.environ
    if git_emails is None:
        git_emails = git_config_emails(project_root)
    identifiers = collect_local_identifiers(environment, git_emails, Path.home().name)
    notes.append(f"local identifiers searched for: {len(identifiers)}")
    failures.extend(check_no_leaked_identifiers(documents, identifiers))

    notes.append(f"documents scanned: {len(documents)}")
    return VerificationResult(
        failures=tuple(failures), notes=tuple(notes), metrics_present=metrics_present
    )


def format_report(result: VerificationResult) -> list[str]:
    """Every failure, then a verdict whose wording the failures actually support."""

    lines = [f"  note: {note}" for note in result.notes]
    lines.extend(failure.render() for failure in result.failures)
    if result.failures:
        lines.append(f"FAILED: {len(result.failures)} problem(s) in the README and docs")
    elif not result.metrics_present:
        lines.append(
            "NOTHING TO VERIFY: every other check passed, but the results file does "
            "not exist yet, so no README number has a source"
        )
    else:
        lines.append("PASS: every README number has a source and every disclosure is present")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PUB-01..PUB-05 / PUB-10: verify README.md against the repository."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="repository to verify (default: the one this script lives in)",
    )
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help=f"long-format metrics file (default: results/{DETECTION_METRICS_FILENAME})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = PROJECT_ROOT if args.project_root is None else Path(args.project_root)
    metrics = None if args.metrics_csv is None else Path(args.metrics_csv)
    result = verify(root, metrics_path=metrics)
    print(f"verify_readme: {root}")
    for line in format_report(result):
        print(line)
    return result.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
