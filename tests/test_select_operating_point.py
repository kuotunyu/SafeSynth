"""Operating-point driver: checkpoint resolution and the report's two branches.

Inference is not exercised here - it needs weights and several CPU-minutes. What
is exercised is everything around it, and in particular the path rewrite, which
is the piece most likely to be wrong in a way that still runs: the recorded
`best_model_checkpoint` is an absolute /content/... path from Colab, so trusting
it verbatim fails on every arm, and falling back silently would evaluate the
LAST checkpoint while the report claims the best one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from scripts.select_operating_point import (
    OperatingPointError,
    best_checkpoint,
    render,
)


@dataclass(frozen=True)
class FakePaths:
    runs: Path


@dataclass(frozen=True)
class FakePoint:
    score_threshold: float
    bare_head_recall: float
    compliance_precision: float
    n_predicted_compliant: int = 10
    n_predicted_non_compliant: int = 5
    n_ground_truth_bare_heads: int = 42


def _arm(root: Path, *, checkpoints, best: str | None, arm="real_only", seed=1337):
    seed_dir = root / arm / f"seed_{seed}"
    for name in checkpoints:
        (seed_dir / name).mkdir(parents=True, exist_ok=True)
    last = max(checkpoints, key=lambda name: int(name.split("-")[-1]))
    state = {"log_history": []}
    if best is not None:
        # Exactly how Colab writes it: an absolute POSIX path into /content.
        state["best_model_checkpoint"] = f"/content/runs/{arm}/seed_{seed}/{best}"
    (seed_dir / last / "trainer_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    return seed_dir


# --------------------------------------------------------------------------
# checkpoint resolution
# --------------------------------------------------------------------------


def test_the_colab_absolute_path_is_resolved_against_the_local_directory(
    tmp_path: Path,
) -> None:
    """/content/... does not exist here; only the basename is portable."""

    _arm(tmp_path, checkpoints=["checkpoint-1752", "checkpoint-10900"], best="checkpoint-1752")

    chosen = best_checkpoint(FakePaths(tmp_path), "real_only", 1337)

    assert chosen.name == "checkpoint-1752"
    assert chosen.is_dir()


def test_the_best_is_preferred_over_the_last_even_though_the_last_is_higher(
    tmp_path: Path,
) -> None:
    """If this picked the last, every reported number would be the overfit one."""

    _arm(tmp_path, checkpoints=["checkpoint-1752", "checkpoint-10900"], best="checkpoint-1752")

    assert best_checkpoint(FakePaths(tmp_path), "real_only", 1337).name != "checkpoint-10900"


def test_a_recorded_best_that_no_longer_exists_falls_back_to_the_last(
    tmp_path: Path,
) -> None:
    """Rotation can delete it; evaluating nothing is worse than evaluating the last."""

    _arm(tmp_path, checkpoints=["checkpoint-10900"], best="checkpoint-1752")

    assert best_checkpoint(FakePaths(tmp_path), "real_only", 1337).name == "checkpoint-10900"


def test_a_state_with_no_recorded_best_uses_the_last(tmp_path: Path) -> None:
    _arm(tmp_path, checkpoints=["checkpoint-500", "checkpoint-9000"], best=None)

    assert best_checkpoint(FakePaths(tmp_path), "real_only", 1337).name == "checkpoint-9000"


def test_an_arm_with_no_checkpoints_raises_rather_than_returning_none(
    tmp_path: Path,
) -> None:
    with pytest.raises(OperatingPointError, match="no checkpoint"):
        best_checkpoint(FakePaths(tmp_path), "filtered_syn", 1337)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------


def _render(points, chosen):
    return render(
        points,
        chosen,
        arm="real_only",
        checkpoint="checkpoint-1752",
        n_images=756,
        floor=0.80,
        figure_path="reports/figures/compliance_sweep.png",
    )


def test_the_chosen_row_is_marked_and_only_that_row() -> None:
    points = [
        FakePoint(0.30, 0.90, 0.70),
        FakePoint(0.50, 0.80, 0.85),
        FakePoint(0.70, 0.60, 0.95),
    ]

    rendered = _render(points, points[1])

    marked = [line for line in rendered.splitlines() if "**←**" in line]
    assert len(marked) == 1
    assert marked[0].startswith("| 0.50")


def test_failing_the_floor_is_reported_not_papered_over() -> None:
    """Lowering the floor after seeing the curve would make it meaningless."""

    points = [FakePoint(0.30, 0.90, 0.10), FakePoint(0.50, 0.80, 0.20)]

    rendered = _render(points, None)

    assert "No point clears the floor" in rendered
    assert "0.80" in rendered
    assert "Selected" not in rendered
    assert "**←**" not in rendered


def test_a_successful_selection_states_the_config_key_to_freeze() -> None:
    points = [FakePoint(0.50, 0.80, 0.85)]

    rendered = _render(points, points[0])

    assert "compliance.score_threshold" in rendered
    assert "0.50" in rendered
    assert "No point clears the floor" not in rendered


def test_the_report_states_that_test_was_never_used() -> None:
    """The one claim a reader most needs to be able to check."""

    rendered = _render([FakePoint(0.50, 0.80, 0.85)], None)

    assert "validation split" in rendered
    assert "Never swept on Test" in rendered
