"""The headline figure's data selection, which is where it can lie quietly.

The picture itself is checked by eye (CLAUDE.md). What is pinned here is which
ROWS reach it. detection_metrics.csv carries the same metric names on four
splits - `test` plus three scenario slices - so selecting by metric name alone
plots a slice while the caption says Test, and nothing raises.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.plot_headline import HeadlineError, read_test_metrics
from src.training.arms import ARMS

FIELDS = (
    "arm",
    "seed",
    "split",
    "metric",
    "value",
    "n_instances",
    "n_images",
    "ci_low",
    "ci_high",
    "notes",
)


def _row(arm, split, metric, value):
    return {
        "arm": arm,
        "seed": 1337,
        "split": split,
        "metric": metric,
        "value": value,
        "n_instances": "",
        "n_images": "",
        "ci_low": "",
        "ci_high": "",
        "notes": "",
    }


def _csv(path: Path, rows) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _complete(**overrides):
    """Every arm on `test`, plus decoy slice rows with DIFFERENT values."""

    rows = []
    for index, arm in enumerate(ARMS):
        rows.append(_row(arm, "test", "primary_map_small", 0.40 + index / 100))
        rows.append(_row(arm, "test", "primary_map", 0.50 + index / 100))
        # Same metric names, different split, deliberately different values.
        rows.append(_row(arm, "test/small_object", "primary_map_small", 0.90))
        rows.append(_row(arm, "test/crowded", "primary_map", 0.95))
    rows.extend(overrides.get("extra", []))
    return rows


def test_only_whole_test_rows_are_read(tmp_path: Path) -> None:
    """The slice rows carry 0.90 / 0.95; picking them up would be silent."""

    values = read_test_metrics(_csv(tmp_path / "m.csv", _complete()))

    assert values["real_only"]["primary_map_small"] == pytest.approx(0.40)
    assert values["real_only"]["primary_map"] == pytest.approx(0.50)
    assert all(value != pytest.approx(0.90) for value in values["real_only"].values())


def test_every_arm_is_returned(tmp_path: Path) -> None:
    values = read_test_metrics(_csv(tmp_path / "m.csv", _complete()))

    assert set(values) == set(ARMS)


def test_a_missing_arm_is_refused_rather_than_plotted_short(tmp_path: Path) -> None:
    """Three bars under a four-arm caption is a wrong figure, not a partial one."""

    rows = [row for row in _complete() if row["arm"] != "filtered_syn"]

    with pytest.raises(HeadlineError, match="filtered_syn"):
        read_test_metrics(_csv(tmp_path / "m.csv", rows))


def test_an_absent_csv_is_refused_with_the_command_that_makes_it(tmp_path: Path) -> None:
    with pytest.raises(HeadlineError, match="scripts/eval.py"):
        read_test_metrics(tmp_path / "nope.csv")


def test_a_row_with_an_empty_value_is_skipped_not_crashed_on(tmp_path: Path) -> None:
    """`op_usable` rows exist for arms with no operating point and carry no value."""

    rows = _complete()
    rows.append(_row("real_only", "val", "operating_point", ""))

    values = read_test_metrics(_csv(tmp_path / "m.csv", rows))

    assert "operating_point" not in values["real_only"]


def test_validation_rows_do_not_reach_the_test_panel(tmp_path: Path) -> None:
    rows = _complete()
    rows.append(_row("real_only", "val", "primary_map_small", 0.99))

    values = read_test_metrics(_csv(tmp_path / "m.csv", rows))

    assert values["real_only"]["primary_map_small"] == pytest.approx(0.40)
