"""README verifier: a check that cannot fail is a decoration (K-19).

Every branch below is driven by an input that makes the WRONG answer visibly
wrong, and the rounding arithmetic is worked out by hand in the comments rather
than imported from the module under test. The two failure modes this file exists
to prevent are (a) a verifier that passes a README containing a number nobody
computed, and (b) a verifier that resolves an ambiguous metric name to the wrong
CSV row - the same class of silent substitution that once turned AP_small into
AP_medium in this repository.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_readme
from scripts.verify_readme import (
    DISCLOSURE_TOPICS,
    EXIT_FAILED,
    EXIT_NOTHING_TO_VERIFY,
    EXIT_OK,
    FORBIDDEN_WORDS,
    DisclosureTopic,
    body_excluding_limitations,
    check_forbidden_words,
    check_no_leaked_identifiers,
    check_no_static_badges,
    check_relative_links,
    check_required_disclosures,
    check_table_numbers,
    collect_local_identifiers,
    extract_numbers,
    format_report,
    lines_outside_code_fences,
    main,
    matches_at_shown_precision,
    normalise_label,
    parse_markdown_tables,
    relative_link_targets,
    resolve_arm,
    resolve_metric,
    split_cells,
    token_prefix_match,
    verify,
    worklog_exempt_lines,
)
from src.evaluation.detection import MetricRow, write_detection_metrics_csv

# Numbers copied from the four-arm Colab run so the fixtures read like the real
# table. Nothing in the module under test knows them.
REAL_ONLY_MAP_SMALL = 0.3564
FILTERED_MAP_SMALL = 0.3200


def _row(
    arm: str,
    metric: str,
    value: float,
    *,
    seed: int = 1337,
    split: str = "test",
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> MetricRow:
    return MetricRow(
        arm=arm,
        seed=seed,
        split=split,
        metric=metric,
        value=value,
        n_instances=3817,
        n_images=744,
        ci_low=ci_low,
        ci_high=ci_high,
    )


ROWS = [
    _row("real_only", "primary_map_small", REAL_ONLY_MAP_SMALL),
    _row("real_only", "bare_head_recall", 0.6120),
    _row("filtered_syn", "primary_map_small", FILTERED_MAP_SMALL),
    _row("filtered_syn", "bare_head_recall", 0.5880),
    _row("unfiltered_syn", "primary_map_small", 0.2988),
    _row("unfiltered_syn", "bare_head_recall", 0.5510),
]


def _doc(*lines: str) -> str:
    return "\n".join(lines)


def _table(header: str, *rows: str) -> str:
    columns = header.count("|") - 1
    delimiter = "|" + "|".join(["---"] * columns) + "|"
    return _doc(header, delimiter, *rows)


# ==========================================================================
# Markdown structure
# ==========================================================================
def test_code_fences_are_blanked_but_line_numbers_survive() -> None:
    text = "a\n```\nb\n```\nc"

    assert lines_outside_code_fences(text) == [
        (1, "a"),
        (2, ""),
        (3, ""),
        (4, ""),
        (5, "c"),
    ]


def test_a_table_inside_a_code_fence_is_not_a_table() -> None:
    text = "```\n| arm | primary_map_small |\n| --- | --- |\n| real_only | 0.9999 |\n```"

    assert parse_markdown_tables(text) == []


def test_two_tables_separated_by_a_fence_do_not_merge() -> None:
    text = "\n".join(
        [
            _table("| arm | primary_map_small |", "| real_only | 0.3564 |"),
            "```",
            "noise",
            "```",
            _table("| arm | bare_head_recall |", "| real_only | 0.6120 |"),
        ]
    )

    tables = parse_markdown_tables(text)

    assert len(tables) == 2
    assert [table.rows for table in tables] == [
        (("real_only", "0.3564"),),
        (("real_only", "0.6120"),),
    ]


def test_pipes_without_a_delimiter_row_are_not_a_table() -> None:
    assert parse_markdown_tables("| a | b |\nplain prose | with a pipe") == []


def test_a_horizontal_rule_under_prose_is_not_a_delimiter_row() -> None:
    # "---" matches the delimiter shape, so only the pipe requirement keeps a
    # setext-style rule from turning the paragraph above it into a table.
    assert parse_markdown_tables("| a | b |\n---\ntext") == []


def test_table_row_line_numbers_point_at_the_offending_row() -> None:
    text = "intro\n\n" + _table("| arm | primary_map_small |", "| real_only | 0.3564 |")

    table = parse_markdown_tables(text)[0]

    assert table.header_line == 3
    assert table.row_lines == (5,)


def test_split_cells_drops_edge_pipes_and_keeps_escaped_ones() -> None:
    assert split_cells("| a | b\\|c |") == ("a", "b|c")
    assert split_cells("a | b") == ("a", "b")


def test_limitations_section_is_removed_together_with_its_subsections() -> None:
    text = _doc(
        "## Results",
        "kept one",
        "## Limitations",
        "hidden one",
        "### Nested",
        "hidden two",
        "## License",
        "kept two",
    )

    body = body_excluding_limitations(text)

    assert "kept one" in body
    assert "kept two" in body
    assert "hidden one" not in body
    assert "hidden two" not in body


def test_a_subsection_after_limitations_at_the_same_level_is_kept() -> None:
    text = "# Top\n## Limitations\nhidden\n## Reproduce\nkept"

    body = body_excluding_limitations(text)

    assert "kept" in body
    assert "hidden" not in body


# ==========================================================================
# Numbers
# ==========================================================================
def test_precision_is_read_off_the_literal_not_the_value() -> None:
    (number,) = extract_numbers("0.3560")

    assert number.value == 0.356
    assert number.precision == 4


def test_a_percentage_cell_is_scaled_because_the_csv_stores_fractions() -> None:
    (number,) = extract_numbers("35.6%")

    assert number.scale == 100.0
    # 0.3564 * 100 = 35.64; half a unit in the last shown place is 0.05, and
    # |35.64 - 35.6| = 0.04 <= 0.05.
    assert matches_at_shown_precision(0.3564, number)
    # 0.3550 * 100 = 35.50; |35.50 - 35.6| = 0.10 > 0.05.
    assert not matches_at_shown_precision(0.3550, number)


def test_thousands_separators_parse_and_carry_zero_precision() -> None:
    (number,) = extract_numbers("3,500")

    assert number.value == 3500.0
    assert number.precision == 0


@pytest.mark.parametrize("dash", ["-", "\u2013", "\u2014"])
def test_a_range_reads_as_two_positive_numbers(dash: str) -> None:
    first, second = extract_numbers(f"0.9013{dash}0.9090")

    assert (first.value, second.value) == (0.9013, 0.9090)


def test_a_genuine_minus_sign_still_parses() -> None:
    (number,) = extract_numbers("change -0.0576")

    assert number.value == -0.0576


def test_rounding_tolerance_is_exactly_half_of_the_last_shown_place() -> None:
    (shown,) = extract_numbers("0.3564")

    # Half a unit in the fourth decimal place is 0.00005.
    # 0.35644 differs by 0.00004 -> inside.  0.35646 differs by 0.00006 -> out.
    assert matches_at_shown_precision(0.35644, shown)
    assert not matches_at_shown_precision(0.35646, shown)


# ==========================================================================
# Label resolution
# ==========================================================================
def test_normalise_label_folds_markdown_decoration_away() -> None:
    assert normalise_label(" + **Standard Aug** ") == "standard_aug"


def test_token_prefix_match_expands_abbreviations_but_not_different_words() -> None:
    assert token_prefix_match("unfiltered_syn", "unfiltered_synthetic")
    assert not token_prefix_match("filtered_syn", "unfiltered_synthetic")
    assert not token_prefix_match("real_only", "real")


def test_an_arm_written_out_in_full_resolves_to_its_csv_name() -> None:
    arm, _ = resolve_arm("+ Unfiltered Synthetic", ["real_only", "unfiltered_syn"])

    assert arm == "unfiltered_syn"


def test_an_ambiguous_arm_label_resolves_to_nothing_and_names_both() -> None:
    arm, candidates = resolve_arm("Filtered Synthetic", ["filtered_syn", "filtered_synth"])

    assert arm is None
    assert candidates == ["filtered_syn", "filtered_synth"]


def test_ap_small_is_refused_because_two_csv_metrics_could_mean_it() -> None:
    """The AP_small -> AP_medium substitution this repository already survived.

    `map_small` averages all three classes and `primary_map_small` averages only
    helmet+head. Guessing between them from the string "AP_small" would report
    one metric under the other's name, so the resolver refuses.
    """

    metrics = {"map_small": "map_small", "primary_map_small": "primary_map_small"}

    assert resolve_metric("AP_small", metrics) is None
    assert resolve_metric("primary_map_small", metrics) == "primary_map_small"


def test_a_per_class_metric_name_survives_the_dot() -> None:
    assert resolve_metric("AP (helmet)", {"ap_helmet": "ap.helmet"}) == "ap.helmet"


# ==========================================================================
# Check 1 - table numbers
# ==========================================================================
def test_a_correct_table_produces_no_failures() -> None:
    readme = _table(
        "| Arm | primary_map_small | bare_head_recall |",
        "| Real-only | 0.3564 | 0.6120 |",
        "| + Filtered Synthetic | 0.3200 | 0.5880 |",
    )

    assert check_table_numbers(readme, ROWS) == []


def test_a_fabricated_number_fails_and_the_message_shows_the_source() -> None:
    readme = _table("| Arm | primary_map_small |", "| Real-only | 0.4100 |")

    (failure,) = check_table_numbers(readme, ROWS)

    assert failure.check == "table-numbers"
    assert "0.4100" in failure.message
    assert "0.3564" in failure.message


def test_a_number_whose_arm_has_no_row_fails() -> None:
    readme = _table("| Arm | primary_map_small |", "| Real-only | 0.3564 |", "| Total | 0.9 |")

    (failure,) = check_table_numbers(readme, ROWS)

    assert "no arm applies to it" in failure.message
    assert failure.location.endswith("column 2")


def test_an_unresolvable_metric_column_fails_and_lists_the_known_metrics() -> None:
    readme = _table("| Arm | AP_small |", "| Real-only | 0.3564 |")

    (failure,) = check_table_numbers(readme, ROWS)

    assert "does not name a metric" in failure.message
    assert "primary_map_small" in failure.message


def test_a_known_arm_with_no_row_for_that_metric_fails() -> None:
    """The arm and the metric both exist in the CSV, but not together.

    Distinct from an unknown arm: this is the shape of a table that quietly
    carries a number for a run that was never evaluated on that metric.
    """

    rows = [
        _row("real_only", "primary_map_small", REAL_ONLY_MAP_SMALL),
        _row("filtered_syn", "bare_head_recall", 0.5880),
    ]
    readme = _table(
        "| Arm | primary_map_small |",
        "| real_only | 0.3564 |",
        "| filtered_syn | 0.3200 |",
    )

    (failure,) = check_table_numbers(readme, rows)

    assert "no row in" in failure.message
    assert "arm='filtered_syn'" in failure.message
    assert "metric='primary_map_small'" in failure.message


def test_a_table_with_no_arm_anywhere_is_not_a_metric_table() -> None:
    readme = _table(
        "| Class | Instances | Images |",
        "| helmet | 18,966 | 4,581 |",
        "| head | 5,785 | 920 |",
    )

    assert check_table_numbers(readme, ROWS) == []


def test_an_annotated_column_header_resolves_a_pretty_title() -> None:
    readme = _table(
        "| Arm | AP_small (helmet+head)<!--metric: primary_map_small--> |",
        "| Real-only | 0.3564 |",
    )

    assert check_table_numbers(readme, ROWS) == []


def test_an_annotation_that_names_the_wrong_metric_still_has_to_match() -> None:
    readme = _table(
        "| Arm | AP_small<!--metric: bare_head_recall--> |",
        "| Real-only | 0.3564 |",
    )

    (failure,) = check_table_numbers(readme, ROWS)

    assert "bare_head_recall" in failure.message
    assert "0.612" in failure.message


def test_digits_inside_an_annotation_are_not_read_as_part_of_the_value() -> None:
    """`map_50` carries a number, and the cell has one value, not two."""

    rows = [_row("real_only", "map_50", 0.6801)]
    readme = _table("| Arm | AP50 |", "| real_only | 0.6801<!--metric: map_50--> |")

    assert check_table_numbers(readme, rows) == []


def test_a_cell_arm_annotation_overrides_the_row_label() -> None:
    """<!--arm: ...--> has to win in the arms-in-rows orientation too.

    Both rows below are unreadable without the annotation being honoured: the
    first has a label that resolves to no arm at all, and the second carries the
    FILTERED value under a row labelled Real-only. If the annotation were
    ignored the first cell would be "no arm applies" and the second would be
    compared against real_only's 0.3564, so an annotation silently dropped here
    cannot look like a pass.
    """

    unlabelled = _table(
        "| Arm | primary_map_small |",
        "| Real-only | 0.3564 |",
        "| Best synthetic run | 0.3200<!--arm: filtered_syn--> |",
    )
    overriding = _table(
        "| Arm | primary_map_small |",
        "| Real-only | 0.3564 |",
        "| Real-only | 0.3200<!--arm: filtered_syn--> |",
    )

    assert check_table_numbers(unlabelled, ROWS) == []
    assert check_table_numbers(overriding, ROWS) == []


def test_an_explicitly_skipped_cell_is_left_alone() -> None:
    readme = _table(
        "| Arm | primary_map_small | Rank |",
        "| Real-only | 0.3564 | 1<!--skip--> |",
    )

    assert check_table_numbers(readme, ROWS) == []


def test_arms_in_columns_is_the_other_orientation_and_also_checked() -> None:
    readme = _table(
        "| Metric | Real-only | + Filtered Synthetic |",
        "| primary_map_small | 0.3564 | 0.3200 |",
    )

    assert check_table_numbers(readme, ROWS) == []

    wrong = _table(
        "| Metric | Real-only | + Filtered Synthetic |",
        "| primary_map_small | 0.3564 | 0.9999 |",
    )

    (failure,) = check_table_numbers(wrong, ROWS)

    assert "filtered_syn" in failure.message


# Three seeds chosen so the mean equals NONE of them: a mean that coincides with
# a member would be accepted by the "any individual seed" branch as well, and the
# test would then pass with the averaging removed.
THREE_SEEDS = [
    _row("real_only", "primary_map_small", 0.3560, seed=1),
    _row("real_only", "primary_map_small", 0.3610, seed=2),
    _row("real_only", "primary_map_small", 0.3540, seed=3),
]


def test_mean_over_seeds_is_accepted_and_a_wrong_mean_is_not() -> None:
    # (0.3560 + 0.3610 + 0.3540) / 3 = 1.0710 / 3 = 0.3570
    assert check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3570 |"), THREE_SEEDS
    ) == []

    (failure,) = check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3580 |"), THREE_SEEDS
    )
    assert "0.3580" in failure.message


def test_a_mean_plus_minus_std_cell_checks_both_halves() -> None:
    # mean 0.3570; deviations -0.0010, +0.0040, -0.0030
    # sum of squares = 1e-6 + 1.6e-5 + 9e-6 = 2.6e-5
    # population sd = sqrt(2.6e-5 / 3) = sqrt(8.6667e-6) = 0.0029439 -> 0.0029
    # sample sd     = sqrt(2.6e-5 / 2) = sqrt(1.3e-5)    = 0.0036056 -> 0.0036
    assert check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3570 ± 0.0029 |"), THREE_SEEDS
    ) == []
    assert check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3570 ± 0.0036 |"), THREE_SEEDS
    ) == []

    (failure,) = check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3570 ± 0.0100 |"), THREE_SEEDS
    )
    assert "spread" in failure.message


def test_a_spread_on_one_seed_comes_from_half_the_bootstrap_interval() -> None:
    rows = [
        _row(
            "real_only",
            "primary_map_small",
            REAL_ONLY_MAP_SMALL,
            ci_low=0.3013,
            ci_high=0.4090,
        )
    ]
    # (0.4090 - 0.3013) / 2 = 0.1077 / 2 = 0.05385 -> 0.0539 at four decimals
    assert check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3564 ± 0.0539 |"), rows
    ) == []

    (failure,) = check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3564 ± 0.1077 |"), rows
    )
    assert "spread" in failure.message


def test_a_spread_on_a_single_seed_with_no_interval_has_nothing_to_come_from() -> None:
    (failure,) = check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3564 ± 0.0100 |"), ROWS
    )

    assert "no spread at all" in failure.message


def test_two_numbers_with_no_plus_minus_sign_have_no_interpretation() -> None:
    """A pair of numbers only reads as "value ± spread" when the sign is there.

    The second number is exactly half the bootstrap interval, i.e. a spread the
    module WOULD accept after a ±. So a check that entered the spread branch on
    the count alone - without also requiring the sign - would call this cell
    verified while the README never said what the second number is.
    """

    rows = [
        _row(
            "real_only",
            "primary_map_small",
            REAL_ONLY_MAP_SMALL,
            ci_low=0.3000,
            ci_high=0.4000,
        )
    ]
    # (0.4000 - 0.3000) / 2 = 0.0500.
    (failure,) = check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3564 0.0500 |"), rows
    )

    assert "no interpretation" in failure.message
    # With the sign written the same pair is accepted, so what the test pins
    # down is the sign and not the two numbers.
    assert check_table_numbers(
        _table("| Arm | primary_map_small |", "| real_only | 0.3564 ± 0.0500 |"), rows
    ) == []


def test_a_bootstrap_interval_cell_is_checked_against_ci_low_and_ci_high() -> None:
    rows = [
        _row(
            "real_only",
            "primary_map_small",
            REAL_ONLY_MAP_SMALL,
            ci_low=0.3013,
            ci_high=0.4090,
        )
    ]
    good = _table("| Arm | primary_map_small |", "| real_only | 0.3564 (0.3013-0.4090) |")
    bad = _table("| Arm | primary_map_small |", "| real_only | 0.3564 (0.3013-0.9999) |")

    assert check_table_numbers(good, rows) == []
    (failure,) = check_table_numbers(bad, rows)
    assert "ci_low/ci_high" in failure.message


def test_a_cell_with_four_numbers_has_no_interpretation() -> None:
    readme = _table("| Arm | primary_map_small |", "| real_only | 0.3564 1 2 3 |")

    (failure,) = check_table_numbers(readme, ROWS)

    assert "no interpretation" in failure.message


def test_a_metric_present_on_two_splits_demands_an_explicit_split() -> None:
    rows = [
        _row("real_only", "primary_map_small", 0.3564, split="test"),
        _row("real_only", "primary_map_small", 0.3701, split="val"),
    ]
    ambiguous = _table("| Arm | primary_map_small |", "| real_only | 0.3564 |")
    explicit = _table(
        "| Arm | primary_map_small<!--split: test--> |", "| real_only | 0.3564 |"
    )

    (failure,) = check_table_numbers(ambiguous, rows)
    assert "add a <!--split: test--> annotation" in failure.message
    assert check_table_numbers(explicit, rows) == []


def test_the_split_annotation_is_read_from_the_cell_and_from_the_row_label() -> None:
    """The header-level form is not the only one; all three places are offered.

    README.md currently annotates two whole columns, so the header form is
    load-bearing - but the cell and row-label forms are advertised by the same
    expression and would otherwise never be executed. Each table below names a
    split that picks the value shown; without the annotation being read the
    metric is ambiguous across two splits and the cell fails.
    """

    rows = [
        _row("real_only", "primary_map_small", 0.3564, split="test"),
        _row("real_only", "primary_map_small", 0.3701, split="val"),
    ]
    on_the_cell = _table(
        "| Arm | primary_map_small |",
        "| real_only | 0.3564<!--split: test--> |",
    )
    on_the_row_label = _table(
        "| Metric | Real-only |",
        "| primary_map_small<!--split: val--> | 0.3701 |",
    )

    assert check_table_numbers(on_the_cell, rows) == []
    assert check_table_numbers(on_the_row_label, rows) == []


def test_a_row_wider_than_its_header_is_reported_not_crashed_on() -> None:
    readme = _table(
        "| Metric | Real-only |",
        "| primary_map_small | 0.3564 | 0.3200 |",
    )

    (failure,) = check_table_numbers(readme, ROWS)

    assert "no arm applies to it" in failure.message
    assert failure.location.endswith("column 3")


def test_a_non_numeric_cell_in_a_metric_table_is_not_a_failure() -> None:
    readme = _table(
        "| Arm | primary_map_small | Note |",
        "| real_only | 0.3564 | single seed |",
    )

    assert check_table_numbers(readme, ROWS) == []


# ==========================================================================
# Check 2 - disclosures
# ==========================================================================
COMPLETE_DISCLOSURES = _doc(
    "# SafeSynth",
    "SHEL5K re-annotated the same 5,000 images and produced 75,570 labels",
    "against the original 25,502, so every claim here is relative and never",
    "absolute. The design is four arms rather than five because Real-only",
    "already trains on all the real Train data, so a fifth full-real ceiling",
    "arm would be the same arm twice.",
    "The pre-registered H4 gate reached AUC 0.9053 and did not pass.",
)


def test_a_complete_readme_body_satisfies_every_topic() -> None:
    assert check_required_disclosures(COMPLETE_DISCLOSURES) == []


def test_every_shipped_topic_can_actually_fail() -> None:
    """A topic whose patterns are all satisfied by an empty document is vacuous."""

    keys = {
        failure.message.split("'")[1] for failure in check_required_disclosures("# Empty\n")
    }

    assert keys == {topic.key for topic in DISCLOSURE_TOPICS}


def test_a_disclosure_hidden_under_limitations_does_not_count() -> None:
    readme = _doc(
        "# SafeSynth",
        "The design is four arms, and a fifth full-real arm does not apply",
        "because Real-only already trains on all the real Train data.",
        "Claims are relative and never absolute.",
        "The pre-registered H4 gate reached AUC 0.9053 and did not pass.",
        "## Limitations",
        "SHEL5K found 75,570 labels against 25,502.",
    )

    failures = check_required_disclosures(readme)

    assert {failure.message.split("'")[1] for failure in failures} == {"annotation_defect"}
    assert len(failures) == 3


def test_one_missing_phrase_names_the_topic_and_the_alternatives() -> None:
    readme = COMPLETE_DISCLOSURES.replace("SHEL5K", "another study")

    (failure,) = check_required_disclosures(readme)

    assert "annotation_defect" in failure.message
    assert "shel5k" in failure.message


def test_a_custom_topic_is_reported_against_its_own_description() -> None:
    topic = DisclosureTopic(key="k", description="d", groups=((r"absent-phrase",),))

    (failure,) = check_required_disclosures("nothing here", topics=(topic,))

    assert failure.check == "disclosure"
    assert "'k'" in failure.message
    assert "d." in failure.message


# ==========================================================================
# Check 3 - badges
# ==========================================================================
def test_a_shields_badge_is_reported_with_its_line_number() -> None:
    readme = "# Title\n\n![build](https://img.shields.io/badge/build-passing-green)\n"

    (failure,) = check_no_static_badges(readme)

    assert failure.location == "README.md:3"
    assert check_no_static_badges("# Title\n") == []


# ==========================================================================
# Check 4 - forbidden placeholder words
# ==========================================================================
@pytest.mark.parametrize("word", FORBIDDEN_WORDS)
def test_each_forbidden_word_is_caught(word: str) -> None:
    (failure,) = check_forbidden_words({"PLAN.md": f"line one\n{word} here\n"})

    assert failure.location == "PLAN.md:2"
    assert word in failure.message


def test_the_scan_is_case_sensitive_like_the_grep_it_replaces() -> None:
    assert check_forbidden_words({"README.md": "https://example.com/todo/list\n"}) == []
    assert len(check_forbidden_words({"README.md": "TODO\n"})) == 1


def test_the_worklog_waiting_bullet_is_exempt_and_only_that_bullet() -> None:
    worklog = _doc(
        "## 現況快照",
        "- **等使用者做的事**：待確認 A",
        "  第二行 待補 B",
        "- **卡住的事**：尚未 C",
    )

    failures = check_forbidden_words({"docs/worklog.md": worklog})

    assert [failure.location for failure in failures] == ["docs/worklog.md:4"]


def test_the_exemption_ends_at_a_blank_line() -> None:
    worklog = _doc("- **等使用者做的事**：待確認 A", "", "待補 B")

    failures = check_forbidden_words({"docs/worklog.md": worklog})

    assert [failure.location for failure in failures] == ["docs/worklog.md:3"]


def test_the_exemption_does_not_apply_to_any_other_document() -> None:
    text = "- **等使用者做的事**：待確認 A\n"

    assert len(check_forbidden_words({"docs/worklog_archive.md": text})) == 1


def test_worklog_exempt_lines_covers_the_marker_and_its_continuations() -> None:
    lines = ["intro", "- 等使用者做的事：x", "  cont", "- next", "  tail"]

    assert worklog_exempt_lines(lines) == {2, 3}


# ==========================================================================
# Check 5 - relative links
# ==========================================================================
def test_a_missing_relative_link_fails_and_an_existing_one_does_not(tmp_path: Path) -> None:
    (tmp_path / "PLAN.md").write_text("x", encoding="utf-8", newline="\n")
    readme = "[plan](PLAN.md) and [gone](docs/missing.md)\n"

    (failure,) = check_relative_links(readme, tmp_path)

    assert "docs/missing.md" in failure.message
    assert failure.location == "README.md:1"


def test_absolute_and_anchor_targets_are_not_filesystem_paths() -> None:
    readme = "[a](https://example.com) [b](#section) [c](mailto:x@example.com)\n"

    assert relative_link_targets(readme) == []


def test_a_link_with_an_anchor_resolves_the_file_part(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "decisions.md").write_text("x", encoding="utf-8", newline="\n")

    assert check_relative_links("[adr](docs/decisions.md#adr-011)\n", tmp_path) == []


def test_a_query_string_is_stripped_before_the_path_is_resolved(tmp_path: Path) -> None:
    """GitHub's own ?plain=1 links point at a file that exists.

    Leaving the query attached asks the filesystem for `decisions.md?plain=1`,
    which resolves nowhere, so the verifier would report a broken link for a
    document sitting right there - and the module's docstring would be claiming
    a strip that never happens.
    """

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "decisions.md").write_text("x", encoding="utf-8", newline="\n")

    assert check_relative_links("[adr](docs/decisions.md?plain=1#adr-011)\n", tmp_path) == []

    # Stripping the query must not stop a genuinely missing file being caught.
    (failure,) = check_relative_links("[gone](docs/missing.md?plain=1)\n", tmp_path)
    assert "docs/missing.md?plain=1" in failure.message


def test_a_protocol_relative_target_is_a_url_and_not_a_repo_path() -> None:
    """`//host/path` inherits the page's scheme; it is not a path in this repo.

    Asserted on the target list rather than through check_relative_links,
    because resolving `//host/path` against a Windows root produces a UNC path
    and the failure would be a network timeout rather than a red assertion.
    """

    assert relative_link_targets("[cdn](//example.com/logo.png)\n") == []
    # The same shape one slash shorter IS a repo path, so the test distinguishes
    # the protocol-relative prefix from "starts with a slash".
    assert relative_link_targets("[doc](/docs/decisions.md)\n") == [(1, "/docs/decisions.md")]


def test_a_directory_target_counts_as_resolving(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()

    assert check_relative_links("see [docs](docs/)\n", tmp_path) == []


def test_a_link_inside_a_code_fence_is_not_followed(tmp_path: Path) -> None:
    readme = "```\n[gone](docs/missing.md)\n```\n"

    assert check_relative_links(readme, tmp_path) == []


def test_an_image_link_is_followed_too(tmp_path: Path) -> None:
    (failure,) = check_relative_links("![grid](reports/figures/x.png)\n", tmp_path)

    assert "reports/figures/x.png" in failure.message


# ==========================================================================
# Check 6 - leaked identifiers
# ==========================================================================
def test_the_windows_user_name_is_searched_for_but_generic_ci_names_are_not() -> None:
    assert collect_local_identifiers({"USERNAME": "jdoe1"}, []) == ["jdoe1"]
    assert collect_local_identifiers({"USERNAME": "runner"}, []) == []
    # Below _IDENTIFIER_MIN_LENGTH: "abc" would match ordinary prose.
    assert collect_local_identifiers({"USERNAME": "abc"}, []) == []


def test_an_identifier_exactly_at_the_minimum_length_is_still_searched_for() -> None:
    """The floor is inclusive, and the boundary is where this machine sits.

    _IDENTIFIER_MIN_LENGTH is 4 and the account name this repository is written
    on is four characters long, so a `>=` quietly turned into `>` would drop the
    one identifier PUB-10 most needs to look for while every other test stayed
    green. Four characters must be kept, three must not.
    """

    assert collect_local_identifiers({"USERNAME": "ab12"}, []) == ["ab12"]
    assert collect_local_identifiers({"USERNAME": "ab1"}, []) == []


def test_a_github_noreply_address_is_the_public_identity_and_is_not_searched() -> None:
    published = "61350295+someone@users.noreply.github.com"
    private = "student99@example.edu"

    assert collect_local_identifiers({}, [published]) == []
    assert collect_local_identifiers({}, [published, private]) == ["student99"]


def test_a_value_with_no_at_sign_is_not_treated_as_an_email() -> None:
    assert collect_local_identifiers({}, ["not-an-email"]) == []


def test_git_config_emails_survives_git_being_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine without git must not turn the whole verifier into a traceback."""

    def explode(*_args: object, **_kwargs: object) -> object:
        raise OSError("git not found")

    monkeypatch.setattr(verify_readme.subprocess, "run", explode)

    assert verify_readme.git_config_emails(tmp_path) == []


def test_git_config_emails_reads_both_scopes_and_drops_empty_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers = iter(["  local@example.com \n", "\n"])

    def fake_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=next(answers))

    monkeypatch.setattr(verify_readme.subprocess, "run", fake_run)

    assert verify_readme.git_config_emails(tmp_path) == ["local@example.com"]


def test_the_home_directory_name_is_also_an_identifier() -> None:
    assert collect_local_identifiers({}, [], "operator7") == ["operator7"]


def test_an_identifier_is_matched_regardless_of_case() -> None:
    (failure,) = check_no_leaked_identifiers({"README.md": "path C:/Users/JDoe1/x\n"}, ["jdoe1"])

    assert "jdoe1" in failure.message


@pytest.mark.parametrize("prefix", ["gho_", "hf_", "sk-", "AIza"])
def test_a_long_token_shaped_string_is_reported(prefix: str) -> None:
    (failure,) = check_no_leaked_identifiers({"docs/a.md": prefix + "A" * 20}, [])

    assert "credential-shaped" in failure.message


def test_a_short_prefix_match_is_an_ordinary_word_not_a_token() -> None:
    # 19 characters of body is one below the floor; `hf_transfer` is shorter still.
    assert check_no_leaked_identifiers({"docs/a.md": "hf_" + "A" * 19}, []) == []
    assert check_no_leaked_identifiers({"docs/a.md": "use hf_transfer for speed"}, []) == []


# ==========================================================================
# Driver and exit codes
# ==========================================================================
def _repository(tmp_path: Path, *, readme: str) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    return tmp_path


def _pretend_home(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make Path.home() answer with a known directory name.

    verify() reads the real home directory, so without this the driver tests
    would depend on whose machine they run on - and the identifier the test
    plants could collide with the operator's actual account name.
    """

    home = Path("/home") / name
    monkeypatch.setattr(verify_readme.Path, "home", classmethod(lambda cls: home))


def test_a_clean_repository_with_results_exits_zero(tmp_path: Path) -> None:
    root = _repository(
        tmp_path,
        readme=COMPLETE_DISCLOSURES
        + "\n\n"
        + _table("| Arm | primary_map_small |", "| Real-only | 0.3564 |"),
    )
    (root / "results").mkdir()
    write_detection_metrics_csv(ROWS, root / "results" / "detection_metrics.csv")

    result = verify(root, environment={}, git_emails=[])

    assert result.failures == ()
    assert result.exit_code() == EXIT_OK
    assert "PASS:" in format_report(result)[-1]


def test_tables_can_select_a_second_metrics_csv_with_the_same_arm_names(
    tmp_path: Path,
) -> None:
    """A model-family replication must not silently reuse the first CSV.

    Both detector families use the same four arm names and metric names.  The
    source annotation is therefore the only thing that prevents a plausible
    RT-DETR value from being accepted as an RF-DETR result (or vice versa).
    """

    root = _repository(
        tmp_path,
        readme=(
            COMPLETE_DISCLOSURES
            + "\n\n"
            + _table("| Arm | primary_map_small |", "| Real-only | 0.3564 |")
            + "\n\n<!--metrics-source: rfdetr_detection_metrics.csv-->\n"
            + _table("| Arm | primary_map_small |", "| Real-only | 0.4841 |")
        ),
    )
    (root / "results").mkdir()
    write_detection_metrics_csv(ROWS, root / "results" / "detection_metrics.csv")
    write_detection_metrics_csv(
        [_row("real_only", "primary_map_small", 0.4841)],
        root / "results" / "rfdetr_detection_metrics.csv",
    )

    result = verify(root, environment={}, git_emails=[])

    assert result.failures == ()
    assert any("rfdetr_detection_metrics.csv" in note for note in result.notes)


def test_a_second_metrics_table_cannot_borrow_a_number_from_the_default_csv(
    tmp_path: Path,
) -> None:
    root = _repository(
        tmp_path,
        readme=(
            COMPLETE_DISCLOSURES
            + "\n\n"
            + _table("| Arm | primary_map_small |", "| Real-only | 0.3564 |")
            + "\n\n<!--metrics-source: rfdetr_detection_metrics.csv-->\n"
            + _table("| Arm | primary_map_small |", "| Real-only | 0.3564 |")
        ),
    )
    (root / "results").mkdir()
    write_detection_metrics_csv(ROWS, root / "results" / "detection_metrics.csv")
    write_detection_metrics_csv(
        [_row("real_only", "primary_map_small", 0.4841)],
        root / "results" / "rfdetr_detection_metrics.csv",
    )

    result = verify(root, environment={}, git_emails=[])

    assert [failure.check for failure in result.failures] == ["table-numbers"]
    assert "0.3564" in result.failures[0].message


def test_a_clean_repository_without_results_is_not_a_pass(tmp_path: Path) -> None:
    root = _repository(tmp_path, readme=COMPLETE_DISCLOSURES + "\n")

    result = verify(root, environment={}, git_emails=[])

    assert result.failures == ()
    assert result.metrics_present is False
    assert result.exit_code() == EXIT_NOTHING_TO_VERIFY
    assert "NOTHING TO VERIFY" in format_report(result)[-1]


def test_a_failure_outranks_the_missing_results_file(tmp_path: Path) -> None:
    root = _repository(tmp_path, readme="# Nothing disclosed\n")

    result = verify(root, environment={}, git_emails=[])

    assert result.exit_code() == EXIT_FAILED


def test_every_failure_is_printed_not_just_the_first(tmp_path: Path) -> None:
    root = _repository(
        tmp_path,
        readme="# T\n![b](https://shields.io/x)\n[gone](nope.md)\nTODO\n",
    )

    result = verify(root, environment={}, git_emails=[])
    checks = {failure.check for failure in result.failures}

    assert {"static-badge", "broken-link", "forbidden-word", "disclosure"} <= checks
    rendered = format_report(result)
    assert sum(line.startswith("FAIL [") for line in rendered) == len(result.failures)
    # The verdict has to say so too: a report that lists failures and then calls
    # the run a pass is worse than no report.
    assert rendered[-1].startswith("FAILED:")


def test_verify_actually_runs_the_table_check_when_results_exist(tmp_path: Path) -> None:
    """The end-to-end wiring, not just the function in isolation.

    Reading the CSV and then not comparing anything to it leaves every other
    check green, so this is the only test that notices the call going missing.
    """

    root = _repository(
        tmp_path,
        readme=COMPLETE_DISCLOSURES
        + "\n\n"
        + _table("| Arm | primary_map_small |", "| Real-only | 0.9999 |"),
    )
    (root / "results").mkdir()
    write_detection_metrics_csv(ROWS, root / "results" / "detection_metrics.csv")

    result = verify(root, environment={}, git_emails=[])

    assert [failure.check for failure in result.failures] == ["table-numbers"]
    assert result.exit_code() == EXIT_FAILED


def test_verify_actually_runs_the_identifier_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PUB-10 has to run from the driver, not only from its own unit test.

    This repository has already shipped a report that asserted the outcome of a
    scan the driver never called, and that is exactly what deleting these lines
    would reproduce: every other check stays green and the run says PASS about
    documents nobody searched. The README below leaks the account name given in
    the environment, so a driver that skips the check cannot come back clean.
    """

    _pretend_home(monkeypatch, "nobodyhome")
    root = _repository(
        tmp_path,
        readme=COMPLETE_DISCLOSURES + "\n\nScratch files live under C:/Users/jdoe1/tmp.\n",
    )

    result = verify(root, environment={"USERNAME": "jdoe1"}, git_emails=[])

    assert [failure.check for failure in result.failures] == ["leaked-identifier"]
    assert "jdoe1" in result.failures[0].message
    assert result.exit_code() == EXIT_FAILED


def test_verify_feeds_the_home_directory_name_into_the_identifier_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The home directory is one of the three identifier sources, and the only
    one the driver has to supply itself.

    USERNAME/USER arrive in the environment mapping and the email arrives from
    git, so both survive a driver that passes an empty home name; this is the
    source that disappears silently. The environment is empty and there is no
    git email here, which leaves the home directory as the sole way the planted
    string can be found.
    """

    _pretend_home(monkeypatch, "operator7")
    root = _repository(
        tmp_path,
        readme=COMPLETE_DISCLOSURES + "\n\nThe cutout bank lives in /home/operator7/data.\n",
    )

    result = verify(root, environment={}, git_emails=[])

    assert [failure.check for failure in result.failures] == ["leaked-identifier"]
    assert "operator7" in result.failures[0].message
    assert result.exit_code() == EXIT_FAILED


def test_main_returns_non_zero_when_the_readme_does_not_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository(tmp_path, readme="# Nothing disclosed\n")

    code = main(["--project-root", str(root)])

    assert code == EXIT_FAILED
    assert "FAILED:" in capsys.readouterr().out


def test_plan_and_docs_are_scanned_for_placeholders_but_readme_links_are_not(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path, readme=COMPLETE_DISCLOSURES + "\n")
    (root / "PLAN_PHASE2.md").write_text("待補\n", encoding="utf-8", newline="\n")
    (root / "docs" / "spec.md").write_text("尚未\n", encoding="utf-8", newline="\n")

    result = verify(root, environment={}, git_emails=[])
    locations = {failure.location for failure in result.failures}

    assert locations == {"PLAN_PHASE2.md:1", "docs/spec.md:1"}


def test_a_missing_readme_is_a_failure_and_not_an_exception(tmp_path: Path) -> None:
    result = verify(tmp_path, environment={}, git_emails=[])

    assert result.exit_code() == EXIT_FAILED
    assert result.failures[0].check == "readme"


def test_main_returns_the_exit_code_and_accepts_an_explicit_metrics_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repository(
        tmp_path,
        readme=COMPLETE_DISCLOSURES
        + "\n\n"
        + _table("| Arm | primary_map_small |", "| Real-only | 0.3564 |"),
    )
    csv_path = tmp_path / "elsewhere.csv"
    write_detection_metrics_csv(ROWS, csv_path)

    code = main(["--project-root", str(root), "--metrics-csv", str(csv_path)])

    assert code == EXIT_OK
    assert "PASS:" in capsys.readouterr().out
