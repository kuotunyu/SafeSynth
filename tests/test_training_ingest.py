"""M16 audit: every check gets a fixture that passes and one that fails.

K-19 is the standard this file is written to. A test that cannot fail is worse
than no test, because it stops anyone looking again. So each case below mutates
exactly ONE thing away from a known-good four-arm result set and asserts that
exactly the corresponding check fires — if a test asserted only "some finding
appeared", a bug in any other check would satisfy it.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pytest

from scripts.audit_colab_results import UnsafeArchiveError, main, safe_extract
from src.training.arms import ARMS
from src.training.ingest import (
    REQUIRED_PHOTOMETRIC_KEYS,
    AuditReport,
    ColabResultsError,
    Finding,
    audit_colab_results,
    latest_checkpoint,
    load_run_records,
    package_arm_outputs,
    render_audit_markdown,
    training_curve,
)

DIGEST = "a" * 64
STEPS = 10_900
SYNTHETIC_POOL = 3_477

# Mirrors configs/training.yaml augmentation.standard_aug. Only the KEYS matter
# to the audit; the values are placeholders and are never read.
GOOD_AUGMENTATION = {
    "standard_aug": {key: 1 for key in REQUIRED_PHOTOMETRIC_KEYS}
    | {"horizontal_flip": 0.5, "perspective": 0.3},
}


def _record(arm: str, *, n_synthetic: int, profile: str) -> dict:
    return {
        "arm": arm,
        "seed": 1337,
        "total_steps": STEPS,
        "eval_metrics": {"eval_map": 0.35, "eval_map_small": 0.30},
        "composition": {
            "arm": arm,
            "n_real_train": 3_500,
            "n_real_val": 756,
            "n_synthetic": n_synthetic,
            "n_train_total": 3_500 + n_synthetic,
            "augmentation_profile": profile,
            "real_train_digest": DIGEST,
        },
    }


def _good_records() -> dict[str, dict]:
    return {
        "real_only": _record("real_only", n_synthetic=0, profile="real_only"),
        "standard_aug": _record("standard_aug", n_synthetic=0, profile="standard_aug"),
        "unfiltered_syn": _record(
            "unfiltered_syn", n_synthetic=SYNTHETIC_POOL, profile="standard_aug"
        ),
        "filtered_syn": _record(
            "filtered_syn", n_synthetic=SYNTHETIC_POOL, profile="standard_aug"
        ),
    }


def _write(root: Path, records: dict[str, dict], *, with_logs: bool = True) -> Path:
    for arm, record in records.items():
        directory = root / arm
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run_record.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8", newline="\n"
        )
        if with_logs:
            (directory / "trainer_state.json").write_text(
                json.dumps({"log_history": []}), encoding="utf-8", newline="\n"
            )
    return root


def _audit(root: Path, augmentation=None) -> AuditReport:
    return audit_colab_results(
        root, augmentation_config=augmentation if augmentation is not None else GOOD_AUGMENTATION
    )


def _checks(report: AuditReport) -> list[str]:
    return sorted(finding.check for finding in report.findings)


# --------------------------------------------------------------------------
# the baseline: a complete, comparable result set raises nothing at all
# --------------------------------------------------------------------------


def test_a_complete_consistent_result_set_produces_no_findings(tmp_path: Path) -> None:
    report = _audit(_write(tmp_path, _good_records()))

    assert report.findings == ()
    assert report.ok
    assert report.arms_present == ARMS
    assert report.real_train_digest == DIGEST
    assert report.total_steps == STEPS
    assert report.synthetic_counts["filtered_syn"] == SYNTHETIC_POOL


# --------------------------------------------------------------------------
# TRAIN-16 — missing arms are listed, not raised on one at a time
# --------------------------------------------------------------------------


def test_a_missing_arm_is_named_in_the_finding(tmp_path: Path) -> None:
    records = _good_records()
    del records["filtered_syn"]

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["arms_present"]
    assert "filtered_syn" in report.findings[0].message
    assert report.arms_present == ("real_only", "standard_aug", "unfiltered_syn")
    assert not report.ok


def test_every_missing_arm_appears_in_one_finding(tmp_path: Path) -> None:
    """One round trip per missing file would be a bad way to learn all of this.

    Asserted on the exact joined list, not with `in`. "unfiltered_syn" CONTAINS
    the substring "filtered_syn", so a containment check passes even when only
    the first missing arm is reported - a mutation that survived until this
    test was rewritten.
    """

    records = _good_records()
    del records["unfiltered_syn"]
    del records["filtered_syn"]

    report = _audit(_write(tmp_path, records))

    message = report.findings[0].message
    assert "for: unfiltered_syn, filtered_syn." in message
    assert len([f for f in report.findings if f.check == "arms_present"]) == 1
    assert report.arms_present == ("real_only", "standard_aug")


def test_an_empty_directory_reports_the_missing_arms_and_nothing_else(tmp_path: Path) -> None:
    report = _audit(tmp_path)

    assert _checks(report) == ["arms_present"]
    assert report.arms_present == ()
    assert report.real_train_digest is None


# --------------------------------------------------------------------------
# TRAIN-17 — the arms must have trained on the same real images
# --------------------------------------------------------------------------


def test_differing_real_train_digests_are_fatal(tmp_path: Path) -> None:
    records = _good_records()
    records["filtered_syn"]["composition"]["real_train_digest"] = "b" * 64

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["identical_real_images"]
    assert report.findings[0].severity == "fatal"
    assert report.real_train_digest is None


def test_a_blank_digest_is_fatal_because_sameness_cannot_be_proven(tmp_path: Path) -> None:
    """Absent evidence is not evidence of sameness; it must not pass quietly.

    ALL FOUR are blanked on purpose. With only one blanked the digest set is
    {"", DIGEST}, which trips the *inequality* branch instead, so the blank
    guard is never the thing being observed - a mutation deleting it survived
    until this fixture was changed. Blanking all four makes the set {""},
    which the inequality branch cannot fire on.
    """

    records = _good_records()
    for record in records.values():
        record["composition"]["real_train_digest"] = ""

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["identical_real_images"]
    assert "cannot be proven" in report.findings[0].message
    assert report.real_train_digest is None


def test_one_blank_digest_among_valid_ones_is_still_fatal(tmp_path: Path) -> None:
    """The partial case routes through the blank guard, not the inequality one."""

    records = _good_records()
    records["standard_aug"]["composition"]["real_train_digest"] = ""

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["identical_real_images"]
    assert "['standard_aug']" in report.findings[0].message
    assert "cannot be proven" in report.findings[0].message


# --------------------------------------------------------------------------
# TRAIN-07 — equal optimizer-step budget
# --------------------------------------------------------------------------


def test_unequal_step_budgets_are_fatal(tmp_path: Path) -> None:
    records = _good_records()
    records["filtered_syn"]["total_steps"] = STEPS + 1

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["equal_steps"]
    assert report.total_steps is None


# --------------------------------------------------------------------------
# TRAIN-18 — synthetic composition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["real_only", "standard_aug"])
def test_a_real_only_arm_carrying_synthetic_images_is_fatal(tmp_path: Path, arm: str) -> None:
    records = _good_records()
    records[arm]["composition"]["n_synthetic"] = 1

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["synthetic_counts"]
    assert arm in report.findings[0].message


def test_unequal_filtered_and_unfiltered_counts_are_fatal(tmp_path: Path) -> None:
    """Both non-zero and both large, so only the EQUALITY check can be firing."""

    records = _good_records()
    records["unfiltered_syn"]["composition"]["n_synthetic"] = SYNTHETIC_POOL - 100

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["synthetic_counts"]
    assert "more data" in report.findings[0].message


def test_two_empty_synthetic_arms_are_fatal(tmp_path: Path) -> None:
    """Equal but both zero passes the equality test and is still worthless."""

    records = _good_records()
    for arm in ("unfiltered_syn", "filtered_syn"):
        records[arm]["composition"]["n_synthetic"] = 0

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["synthetic_counts"]
    assert "zero synthetic" in report.findings[0].message


# --------------------------------------------------------------------------
# augmentation profile and EXP-01
# --------------------------------------------------------------------------


def test_a_wrong_augmentation_profile_is_fatal(tmp_path: Path) -> None:
    records = _good_records()
    records["standard_aug"]["composition"]["augmentation_profile"] = "real_only"

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["augmentation_profile"]
    assert "'real_only'" in report.findings[0].message


@pytest.mark.parametrize("dropped", REQUIRED_PHOTOMETRIC_KEYS)
def test_each_missing_photometric_key_is_caught_individually(
    tmp_path: Path, dropped: str
) -> None:
    """EXP-01 names several; dropping any ONE must be enough to fail."""

    augmentation = {
        "standard_aug": {
            key: 1 for key in REQUIRED_PHOTOMETRIC_KEYS if key != dropped
        }
    }

    report = _audit(_write(tmp_path, _good_records()), augmentation)

    assert _checks(report) == ["photometric_augmentation"]
    assert dropped in report.findings[0].message


def test_a_missing_standard_aug_block_is_fatal(tmp_path: Path) -> None:
    report = _audit(_write(tmp_path, _good_records()), {"real_only": {}})

    assert _checks(report) == ["photometric_augmentation"]


# --------------------------------------------------------------------------
# K-18 scar tissue: an arm can finish and have evaluated nothing
# --------------------------------------------------------------------------


def test_an_arm_with_no_eval_metrics_is_fatal(tmp_path: Path) -> None:
    records = _good_records()
    del records["real_only"]["eval_metrics"]

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["evaluation_ran"]
    assert "K-18" in report.findings[0].message


def test_an_empty_eval_metrics_dict_is_also_fatal(tmp_path: Path) -> None:
    """`{}` is the shape a crashed compute_metrics leaves behind."""

    records = _good_records()
    records["real_only"]["eval_metrics"] = {}

    report = _audit(_write(tmp_path, records))

    assert _checks(report) == ["evaluation_ran"]


# --------------------------------------------------------------------------
# EVAL-12 — curves come from the raw log, so the log has to be there
# --------------------------------------------------------------------------


def test_a_missing_trainer_state_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    """The comparison is still valid; only the curve cannot be re-aggregated."""

    report = _audit(_write(tmp_path, _good_records(), with_logs=False))

    assert _checks(report) == ["training_log"] * len(ARMS)
    assert all(finding.severity == "warning" for finding in report.findings)
    assert report.ok


def test_training_curve_keeps_only_eval_entries_in_step_order() -> None:
    state = {
        "log_history": [
            {"loss": 7.1, "step": 200},
            {"eval_map": 0.30, "eval_map_small": 0.25, "step": 400},
            {"loss": 6.4, "step": 300},
            {"eval_map": 0.10, "eval_map_small": 0.05, "step": 100},
        ]
    }

    curve = training_curve(state)

    assert [point["step"] for point in curve] == [100, 400]
    assert [point["eval_map"] for point in curve] == [0.10, 0.30]


def test_training_curve_drops_booleans_rather_than_counting_them_as_numbers() -> None:
    """bool is a subclass of int in Python; an unguarded isinstance lets it through."""

    curve = training_curve(
        {"log_history": [{"eval_map": 0.5, "eval_is_best": True, "step": 1}]}
    )

    assert "eval_is_best" not in curve[0]
    assert curve[0]["eval_map"] == 0.5


def test_training_curve_of_an_untouched_state_is_empty() -> None:
    assert training_curve({"log_history": []}) == []
    assert training_curve({}) == []


# --------------------------------------------------------------------------
# reading and rendering
# --------------------------------------------------------------------------


def test_unreadable_json_raises_rather_than_being_silently_skipped(tmp_path: Path) -> None:
    """A corrupt file must not look identical to an absent one."""

    (tmp_path / "real_only").mkdir(parents=True)
    (tmp_path / "real_only" / "run_record.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ColabResultsError, match="not valid JSON"):
        load_run_records(tmp_path)


def test_an_unknown_severity_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="Unknown severity"):
        Finding("x", "catastrophic", "...")


def test_markdown_says_pass_only_when_there_is_nothing_fatal(tmp_path: Path) -> None:
    passing = render_audit_markdown(_audit(_write(tmp_path, _good_records())))

    records = _good_records()
    records["filtered_syn"]["total_steps"] = 1
    failing = render_audit_markdown(_audit(_write(tmp_path, records)))

    assert "**Verdict: PASS**" in passing
    assert "0 fatal" in passing
    assert "**Verdict: FAIL**" in failing
    assert "1 fatal" in failing
    assert "equal_steps" in failing


def test_markdown_escapes_pipes_so_a_finding_cannot_break_the_table() -> None:
    report = AuditReport(findings=(Finding("c", "fatal", "a | b"),))

    rendered = render_audit_markdown(report)

    body = next(line for line in rendered.splitlines() if line.startswith("| **fatal**"))
    # Split on pipes that are NOT escaped: a correctly escaped row still parses
    # as exactly three cells, which is the property that matters. Counting raw
    # pipes would pass just as happily on an unescaped row with one fewer cell.
    cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", body)[1:-1]]
    assert cells == ["**fatal**", "`c`", "a \\| b"]


def test_a_warning_only_report_still_renders_as_pass() -> None:
    report = AuditReport(findings=(Finding("c", "warning", "minor"),))

    assert "**Verdict: PASS**" in render_audit_markdown(report)
    assert "1 warning" in render_audit_markdown(report)


# --------------------------------------------------------------------------
# packaging on the Colab side
#
# This is the code that silently shipped an archive with no trainer_state.json
# in it, on all four arms, with no error. The fixtures below put the file where
# HF actually puts it - inside checkpoint-N/ - because a fixture that put it at
# the top of the seed directory would have passed against the broken version.
# --------------------------------------------------------------------------


def _runs_tree(root: Path, *, arms=("real_only", "filtered_syn"), steps=(500, 1000)) -> Path:
    for arm in arms:
        seed = root / arm / "seed_1337"
        seed.mkdir(parents=True, exist_ok=True)
        (seed / "run_record.json").write_text(json.dumps({"arm": arm}), encoding="utf-8")
        for step in steps:
            checkpoint = seed / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "global_step": step,
                        "best_model_checkpoint": f"/content/runs/{arm}/seed_1337/checkpoint-{steps[0]}",
                        "log_history": [{"eval_map": 0.1 * step, "step": step}],
                    }
                ),
                encoding="utf-8",
            )
    return root


def test_packaging_finds_trainer_state_inside_the_checkpoint_directory(
    tmp_path: Path,
) -> None:
    runs = _runs_tree(tmp_path / "runs")
    out = tmp_path / "out"

    packaged, missing = package_arm_outputs(runs, out)

    assert missing == []
    assert (out / "real_only" / "trainer_state.json").is_file()
    assert (out / "filtered_syn" / "trainer_state.json").is_file()
    assert "real_only/trainer_state.json" in packaged


def test_packaging_takes_the_highest_numbered_checkpoint_not_the_first(
    tmp_path: Path,
) -> None:
    """The last checkpoint is the one carrying the full log_history."""

    runs = _runs_tree(tmp_path / "runs", arms=("real_only",), steps=(500, 1000, 250))
    out = tmp_path / "out"

    package_arm_outputs(runs, out)

    state = json.loads((out / "real_only" / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 1000


def test_packaging_records_which_checkpoint_holds_the_best_weights(
    tmp_path: Path,
) -> None:
    """Without this the local evaluation has to guess at a step number."""

    runs = _runs_tree(tmp_path / "runs", arms=("real_only",), steps=(500, 1000))
    out = tmp_path / "out"

    package_arm_outputs(runs, out)

    manifest = json.loads((out / "real_only" / "checkpoints.json").read_text(encoding="utf-8"))
    assert manifest["latest_checkpoint"] == "checkpoint-1000"
    assert manifest["best_model_checkpoint"].endswith("checkpoint-500")
    assert manifest["available"] == ["checkpoint-500", "checkpoint-1000"]


def test_packaging_reports_an_arm_with_no_checkpoints_instead_of_skipping_it(
    tmp_path: Path,
) -> None:
    """Silence is what made the original bug invisible."""

    runs = tmp_path / "runs"
    (runs / "real_only" / "seed_1337").mkdir(parents=True)
    (runs / "real_only" / "seed_1337" / "run_record.json").write_text("{}", encoding="utf-8")

    packaged, missing = package_arm_outputs(runs, tmp_path / "out")

    assert missing == ["real_only/trainer_state.json"]
    assert packaged == ["real_only/run_record.json"]


def test_a_non_numeric_checkpoint_name_does_not_crash_the_scan(tmp_path: Path) -> None:
    seed = tmp_path / "runs" / "real_only" / "seed_1337"
    (seed / "checkpoint-final").mkdir(parents=True)
    (seed / "checkpoint-700").mkdir(parents=True)

    assert latest_checkpoint(seed).name == "checkpoint-700"


def test_latest_checkpoint_is_none_when_there_are_none(tmp_path: Path) -> None:
    tmp_path.joinpath("seed_1337").mkdir()

    assert latest_checkpoint(tmp_path / "seed_1337") is None


# --------------------------------------------------------------------------
# the driver — exit codes and the archive guard
#
# main() is the path a real invocation takes and the one most often left
# untested; every branch below is reachable from the command line.
# --------------------------------------------------------------------------


def _zip_of(tmp_path: Path, members: dict[str, str]) -> Path:
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, body in members.items():
            bundle.writestr(name, body)
    return archive


@pytest.mark.parametrize("escaping", ["../escaped.txt", "a/../../escaped.txt"])
def test_extraction_refuses_a_member_that_escapes_the_target(
    tmp_path: Path, escaping: str
) -> None:
    archive = _zip_of(tmp_path, {escaping: "x"})
    destination = tmp_path / "out"

    with pytest.raises(UnsafeArchiveError, match="outside"):
        safe_extract(archive, destination)

    assert not (tmp_path / "escaped.txt").exists()


def test_extraction_places_normal_members_under_the_target(tmp_path: Path) -> None:
    archive = _zip_of(tmp_path, {"real_only/run_record.json": "{}"})
    destination = tmp_path / "out"

    members = safe_extract(archive, destination)

    assert members == ["real_only/run_record.json"]
    assert (destination / "real_only" / "run_record.json").read_text(encoding="utf-8") == "{}"


def test_driver_returns_zero_and_writes_a_report_on_a_clean_result_set(
    tmp_path: Path,
) -> None:
    results = _write(tmp_path / "colab", _good_records())
    report_path = tmp_path / "reports" / "m16.md"

    code = main(["--results-dir", str(results), "--report", str(report_path)])

    assert code == 0
    assert "**Verdict: PASS**" in report_path.read_text(encoding="utf-8")


def test_driver_returns_one_when_something_is_fatal(tmp_path: Path) -> None:
    records = _good_records()
    records["filtered_syn"]["total_steps"] = 1
    results = _write(tmp_path / "colab", records)
    report_path = tmp_path / "reports" / "m16.md"

    code = main(["--results-dir", str(results), "--report", str(report_path)])

    assert code == 1
    # The report is written even on failure - that list is the whole point.
    assert "equal_steps" in report_path.read_text(encoding="utf-8")


def test_driver_returns_two_when_there_is_nothing_to_audit(tmp_path: Path) -> None:
    """Distinct from 1: nothing was checked, rather than checks having failed."""

    code = main(["--results-dir", str(tmp_path / "absent")])

    assert code == 2


def test_driver_returns_two_when_the_named_archive_does_not_exist(tmp_path: Path) -> None:
    code = main(["--archive", str(tmp_path / "nope.zip")])

    assert code == 2


def test_driver_extracts_the_archive_before_auditing(tmp_path: Path) -> None:
    records = _good_records()
    members = {
        f"{arm}/run_record.json": json.dumps(record) for arm, record in records.items()
    }
    members |= {f"{arm}/trainer_state.json": '{"log_history": []}' for arm in records}
    archive = _zip_of(tmp_path, members)
    results = tmp_path / "colab"
    report_path = tmp_path / "reports" / "m16.md"

    code = main(
        [
            "--archive", str(archive),
            "--results-dir", str(results),
            "--report", str(report_path),
        ]
    )

    assert code == 0
    assert (results / "filtered_syn" / "run_record.json").is_file()
