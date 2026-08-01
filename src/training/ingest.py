"""M16: audit what came back from Colab before any of it reaches a report.

The four-arm comparison is only meaningful if the four arms differ in exactly
one thing. TRAIN-15..TRAIN-18 list the ways that can quietly stop being true —
a differing real-image set, unequal step budgets, unequal synthetic counts, a
baseline that never got the photometric augmentation EXP-01 requires — and every
one of them is invisible in the training curves. So they are checked mechanically
here rather than eyeballed.

Two design choices worth stating:

1. This returns a LIST of findings rather than raising on the first problem.
   PLAN_PHASE2 M16 says "缺檔就列清單停下來問使用者", and a caller who has to
   re-run Colab wants the whole list in one go, not one item per round trip.

2. Training curves are re-aggregated from `trainer_state.json`, never copied
   from what the notebook printed (EVAL-12). The displayed numbers came from
   whatever `metric_for_best_model` and threshold were live at the time; the
   report's numbers must come from the raw record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.training.arms import ARMS, AUGMENTATION_PROFILE

# EXP-01: the +Standard Aug arm must carry photometric augmentation, because the
# synthetic arms' images went through low-light and motion-blur post-fx. Without
# it, part of any synthetic win is just "that arm saw one more kind of
# augmentation", and the claim collapses under the first sharp question.
# These are KEY NAMES, not thresholds — the values stay in configs/training.yaml.
REQUIRED_PHOTOMETRIC_KEYS = (
    "gamma_range",
    "gauss_noise_sigma",
    "jpeg_quality",
    "motion_blur",
    "random_brightness_contrast",
)

# Arms whose training set must contain zero synthetic images.
REAL_ONLY_ARMS = ("real_only", "standard_aug")
# Arms that must be EQUAL in size, so "more data" cannot be confused with
# "better data" (the filtered/unfiltered comparison).
EQUAL_SIZE_SYNTHETIC_ARMS = ("unfiltered_syn", "filtered_syn")

SEVERITY_ORDER = ("fatal", "warning")


class ColabResultsError(RuntimeError):
    """Raised when the returned artefacts cannot be read at all."""


@dataclass(frozen=True)
class Finding:
    """One thing that is wrong. `fatal` means the comparison is invalid."""

    check: str
    severity: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"Unknown severity {self.severity!r}")


@dataclass(frozen=True)
class AuditReport:
    findings: tuple[Finding, ...] = ()
    arms_present: tuple[str, ...] = ()
    real_train_digest: str | None = None
    total_steps: int | None = None
    synthetic_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def fatal(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "fatal")

    @property
    def ok(self) -> bool:
        return not self.fatal


# spec: TRAIN-15
def load_run_records(root: Path, *, arms: Sequence[str] = ARMS) -> dict[str, dict[str, Any]]:
    """Read `<root>/<arm>/run_record.json` for each arm that produced one.

    A missing arm is not an error here — it becomes a finding in the audit, so
    that one call reports every gap instead of stopping at the first.
    """

    records: dict[str, dict[str, Any]] = {}
    for arm in arms:
        path = Path(root) / arm / "run_record.json"
        if not path.is_file():
            continue
        try:
            records[arm] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ColabResultsError(f"{path} is not valid JSON: {error}") from error
    return records


# spec: EVAL-12
def training_curve(trainer_state: Mapping[str, Any]) -> list[dict[str, float]]:
    """Re-aggregate the eval curve from the raw HF trainer state.

    `log_history` interleaves train-loss entries and eval entries; only the
    latter carry metrics. Returning them in step order lets a report plot the
    curve from the record rather than from whatever the notebook happened to
    print.
    """

    curve: list[dict[str, float]] = []
    for entry in trainer_state.get("log_history", []):
        if not any(key.startswith("eval_") for key in entry):
            continue
        point = {
            key: float(value)
            for key, value in entry.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        curve.append(point)
    curve.sort(key=lambda point: point.get("step", 0.0))
    return curve


def _composition(record: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    composition = record.get("composition")
    if not isinstance(composition, Mapping):
        raise ColabResultsError(f"{arm}: run_record.json has no `composition` block")
    return composition


# spec: TRAIN-16
def _check_arms_present(records: Mapping[str, Any], arms: Sequence[str]) -> list[Finding]:
    missing = [arm for arm in arms if arm not in records]
    if not missing:
        return []
    return [
        Finding(
            "arms_present",
            "fatal",
            f"No run_record.json for: {', '.join(missing)}. Re-run those arms; the "
            f"finished ones resume from their Drive checkpoints in minutes.",
        )
    ]


# spec: TRAIN-17
def _check_identical_real_images(records: Mapping[str, Any]) -> tuple[list[Finding], str | None]:
    digests = {
        arm: str(_composition(record, arm).get("real_train_digest", ""))
        for arm, record in records.items()
    }
    distinct = set(digests.values())
    if "" in distinct:
        blank = sorted(arm for arm, value in digests.items() if not value)
        return (
            [
                Finding(
                    "identical_real_images",
                    "fatal",
                    f"Arms {blank} recorded no real_train_digest, so it cannot be "
                    f"proven they trained on the same real images.",
                )
            ],
            None,
        )
    if len(distinct) > 1:
        return (
            [
                Finding(
                    "identical_real_images",
                    "fatal",
                    f"Arms trained on DIFFERENT real images: {digests}. The whole "
                    f"four-arm comparison is invalid; nothing downstream is salvageable.",
                )
            ],
            None,
        )
    return [], distinct.pop() if distinct else None


# spec: TRAIN-07
def _check_equal_steps(records: Mapping[str, Any]) -> tuple[list[Finding], int | None]:
    steps = {arm: int(record.get("total_steps", -1)) for arm, record in records.items()}
    distinct = set(steps.values())
    if len(distinct) > 1:
        return (
            [
                Finding(
                    "equal_steps",
                    "fatal",
                    f"Arms did not get the same optimizer-step budget: {steps}. "
                    f"A win could just be 'it trained longer'.",
                )
            ],
            None,
        )
    return [], distinct.pop() if distinct else None


# spec: TRAIN-18
def _check_synthetic_counts(
    records: Mapping[str, Any],
) -> tuple[list[Finding], dict[str, int]]:
    counts = {
        arm: int(_composition(record, arm).get("n_synthetic", 0))
        for arm, record in records.items()
    }
    findings: list[Finding] = []

    for arm in REAL_ONLY_ARMS:
        if counts.get(arm, 0):
            findings.append(
                Finding(
                    "synthetic_counts",
                    "fatal",
                    f"{arm} must be a real-only arm but carries {counts[arm]} synthetic "
                    f"images.",
                )
            )

    present = [arm for arm in EQUAL_SIZE_SYNTHETIC_ARMS if arm in counts]
    if len(present) == len(EQUAL_SIZE_SYNTHETIC_ARMS):
        sizes = {arm: counts[arm] for arm in present}
        if len(set(sizes.values())) != 1:
            findings.append(
                Finding(
                    "synthetic_counts",
                    "fatal",
                    f"filtered and unfiltered are NOT equal in size: {sizes}. This "
                    f"confounds 'more data' with 'better data' — the one comparison "
                    f"the filtering arm exists to make.",
                )
            )
        elif not any(sizes.values()):
            findings.append(
                Finding(
                    "synthetic_counts",
                    "fatal",
                    "Both synthetic arms trained on zero synthetic images.",
                )
            )
    return findings, counts


def _check_augmentation_profiles(records: Mapping[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for arm, record in records.items():
        expected = AUGMENTATION_PROFILE.get(arm)
        actual = _composition(record, arm).get("augmentation_profile")
        if expected is not None and actual != expected:
            findings.append(
                Finding(
                    "augmentation_profile",
                    "fatal",
                    f"{arm} ran with augmentation profile {actual!r}, expected "
                    f"{expected!r}.",
                )
            )
    return findings


# spec: EXP-01
def _check_photometric_augmentation(augmentation_config: Mapping[str, Any]) -> list[Finding]:
    block = augmentation_config.get("standard_aug")
    if not isinstance(block, Mapping):
        return [
            Finding(
                "photometric_augmentation",
                "fatal",
                "configs/training.yaml has no augmentation.standard_aug block, so the "
                "baseline arm's augmentation cannot be verified.",
            )
        ]
    missing = [key for key in REQUIRED_PHOTOMETRIC_KEYS if key not in block]
    if missing:
        return [
            Finding(
                "photometric_augmentation",
                "fatal",
                f"The +Standard Aug arm is missing photometric augmentation {missing}. "
                f"EXP-01 requires it, because the synthetic images carry low-light and "
                f"motion-blur post-fx; without it a synthetic win is partly just "
                f"'that arm saw one more kind of augmentation'.",
            )
        ]
    return []


# spec: TRAIN-15
def _check_evaluation_actually_ran(records: Mapping[str, Any]) -> list[Finding]:
    """K-18 left a scar: a run can look complete and have evaluated nothing."""

    findings: list[Finding] = []
    for arm, record in records.items():
        metrics = record.get("eval_metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            findings.append(
                Finding(
                    "evaluation_ran",
                    "fatal",
                    f"{arm} carries no eval_metrics. The arm trained but never produced "
                    f"a validated number — exactly the K-18 failure.",
                )
            )
    return findings


def _check_training_logs(root: Path, records: Mapping[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for arm in records:
        state = Path(root) / arm / "trainer_state.json"
        if not state.is_file():
            findings.append(
                Finding(
                    "training_log",
                    "warning",
                    f"{arm} has no trainer_state.json, so its training curve cannot be "
                    f"re-aggregated from the raw log (EVAL-12) and would have to be "
                    f"copied off the notebook display, which is not allowed.",
                )
            )
    return findings


# spec: TRAIN-15
def audit_colab_results(
    root: Path,
    *,
    augmentation_config: Mapping[str, Any],
    arms: Sequence[str] = ARMS,
) -> AuditReport:
    """Run every M16 completeness and comparability check and collect findings."""

    records = load_run_records(root, arms=arms)

    findings: list[Finding] = []
    findings += _check_arms_present(records, arms)

    digest = steps = None
    counts: dict[str, int] = {}
    if records:
        digest_findings, digest = _check_identical_real_images(records)
        step_findings, steps = _check_equal_steps(records)
        count_findings, counts = _check_synthetic_counts(records)
        findings += digest_findings
        findings += step_findings
        findings += count_findings
        findings += _check_augmentation_profiles(records)
        findings += _check_evaluation_actually_ran(records)
        findings += _check_training_logs(Path(root), records)
    findings += _check_photometric_augmentation(augmentation_config)

    return AuditReport(
        findings=tuple(findings),
        arms_present=tuple(arm for arm in arms if arm in records),
        real_train_digest=digest,
        total_steps=steps,
        synthetic_counts=counts,
    )


def render_audit_markdown(report: AuditReport) -> str:
    """A report a human can act on: what is missing, and what it costs."""

    lines = ["# M16 — Colab results audit", ""]
    verdict = "PASS" if report.ok else "FAIL"
    lines.append(f"**Verdict: {verdict}** — {len(report.fatal)} fatal, "
                 f"{len(report.findings) - len(report.fatal)} warning.")
    lines.append("")
    lines.append(f"- Arms returned: {', '.join(report.arms_present) or 'none'}")
    lines.append(f"- Shared real-train digest: `{report.real_train_digest or 'n/a'}`")
    lines.append(f"- Optimizer steps per arm: {report.total_steps or 'n/a'}")
    if report.synthetic_counts:
        rendered = ", ".join(
            f"{arm} {count}" for arm, count in sorted(report.synthetic_counts.items())
        )
        lines.append(f"- Synthetic images per arm: {rendered}")
    lines.append("")

    if not report.findings:
        lines.append("No findings. Every arm is present, comparable and evaluated.")
        return "\n".join(lines) + "\n"

    lines.append("## Findings")
    lines.append("")
    lines.append("| Severity | Check | Detail |")
    lines.append("|---|---|---|")
    for severity in SEVERITY_ORDER:
        for finding in report.findings:
            if finding.severity == severity:
                detail = finding.message.replace("|", "\\|")
                lines.append(f"| **{severity}** | `{finding.check}` | {detail} |")
    return "\n".join(lines) + "\n"
