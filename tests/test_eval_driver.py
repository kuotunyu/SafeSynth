"""M18 driver: the parts that can be wrong while still running.

Inference is not exercised — it needs weights and CPU-minutes, and the driver
keeps model loading behind one injectable seam so everything else is testable
without it. What is exercised here is the wiring that decides WHICH weights get
loaded, WHETHER the leak check can fail, and whether coordinates are converted
per image. Each of those produces a plausible number when wrong, which is the
failure mode K-19 is about.

Several tests read the real frozen split. That is deliberate: `training_image_names`
proves a candidate list by SHA-256 against the manifest, so a made-up digest
cannot exercise the success path at all — and the success path is where a
too-permissive check would hide.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import eval as eval_driver
from scripts.eval import (
    CheckpointSource,
    EvalDriverError,
    TrainingListUnavailableError,
    checkpoint_name_from_recorded_path,
    detections_for_evaluation,
    leak_self_check,
    processor_evaluated_size,
    resolve_checkpoint,
    training_image_names,
    verify_step_budget,
)
from src.data.paths import load_project_paths
from src.evaluation.detection import SplitLeakageError
from src.training.arms import ARMS, split_real_images


@pytest.fixture(scope="module")
def frozen_split():
    return split_real_images(load_project_paths().splits / "split_manifest.json")


@pytest.fixture(scope="module")
def real_digest(frozen_split):
    from src.training.arms import digest_names

    return digest_names(frozen_split["train"])


def test_eval_resolves_the_inherited_rf_training_config() -> None:
    """Plain YAML loading drops inherited eval batch settings and fails after training."""

    resolved = eval_driver.load_driver_training_config(
        Path("configs/training_rfdetr.yaml")
    )

    assert resolved["run"]["per_device_eval_batch_size"] == 8
    assert resolved["model"]["checkpoint"] == "Roboflow/rf-detr-nano"


def test_evaluate_arm_persists_the_same_detections_it_scores(
    tmp_path: Path, monkeypatch
) -> None:
    """Removing the write must fail without rerunning expensive Test inference."""

    checkpoint = tmp_path / "checkpoint-10900"
    checkpoint.mkdir()
    weights = eval_driver.ArmWeights(
        arm="real_only",
        seed=1337,
        seed_dir=tmp_path,
        choice=eval_driver.CheckpointChoice(
            checkpoint, eval_driver.CheckpointSource.HIGHEST_STEP
        ),
    )
    expected = [
        {
            "image_id": 7,
            "category_id": 1,
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "score": 0.75,
        }
    ]
    monkeypatch.setattr(eval_driver, "run_inference", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        eval_driver,
        "detections_for_evaluation",
        lambda *args, **kwargs: expected,
    )
    metrics = SimpleNamespace(n_images=1, n_detections=1)
    monkeypatch.setattr(
        eval_driver, "evaluate_detection_metrics", lambda *args, **kwargs: metrics
    )
    monkeypatch.setattr(eval_driver, "detection_metric_rows", lambda *args, **kwargs: ())
    destination = tmp_path / "predictions" / "real_only_test_seed1337.json"

    eval_driver.evaluate_arm(
        weights,
        samples=(SimpleNamespace(image_id=7, width=100, height=100),),
        ground_truth={},
        slices={},
        hard_negative_ids=(),
        config={},
        load_model=lambda *args, **kwargs: (
            object(),
            SimpleNamespace(size={"height": 100, "width": 100}),
        ),
        processor_source="Roboflow/rf-detr-nano",
        device="cpu",
        dtype_name="float32",
        batch_size=1,
        bootstrap_resamples=0,
        bootstrap_workers=1,
        bootstrap_seed=42,
        exposures=50.0,
        total_steps=10_900,
        predictions_path=destination,
    )

    assert json.loads(destination.read_text(encoding="utf-8")) == expected


# --------------------------------------------------------------------------
# which checkpoint gets loaded
#
# best_model_checkpoint is written on Colab as an absolute /content/... path. It
# does not exist locally, so trusting it fails on every arm; falling back
# silently would score the LAST checkpoint - the heavily overfit one - while the
# report claimed the best.
# --------------------------------------------------------------------------


def test_a_colab_absolute_path_yields_only_its_basename() -> None:
    recorded = "/content/runs/real_only/seed_1337/checkpoint-1752"

    assert checkpoint_name_from_recorded_path(recorded) == "checkpoint-1752"


def test_a_windows_recorded_path_also_yields_its_basename() -> None:
    """A rerun on this machine would record backslashes instead."""

    recorded = r"D:\sdg-data\02-safesynth\runs\real_only\seed_1337\checkpoint-1752"

    assert checkpoint_name_from_recorded_path(recorded) == "checkpoint-1752"


@pytest.mark.parametrize("recorded", [None, "", 17, [], {}])
def test_a_missing_or_non_path_recorded_value_is_none_not_a_crash(recorded) -> None:
    assert checkpoint_name_from_recorded_path(recorded) is None


def _seed_dir(root: Path, *, checkpoints, best=None, arm="real_only") -> Path:
    seed_dir = root / arm / "seed_1337"
    for name in checkpoints:
        (seed_dir / name).mkdir(parents=True, exist_ok=True)
    if checkpoints:
        last = max(checkpoints, key=lambda name: int(name.split("-")[-1]))
        state = {"log_history": []}
        if best:
            state["best_model_checkpoint"] = f"/content/runs/{arm}/seed_1337/{best}"
        (seed_dir / last / "trainer_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
    return seed_dir


def test_the_recorded_best_wins_over_the_higher_numbered_last(tmp_path: Path) -> None:
    seed_dir = _seed_dir(
        tmp_path,
        checkpoints=["checkpoint-1752", "checkpoint-10900"],
        best="checkpoint-1752",
    )

    choice = resolve_checkpoint(seed_dir.parent, seed_dir)

    assert choice.path.name == "checkpoint-1752"
    assert choice.source == CheckpointSource.TRAINER_STATE


def test_falling_back_to_the_highest_step_is_recorded_not_silent(
    tmp_path: Path,
) -> None:
    """A silent fallback is a between-arm difference nobody would see."""

    seed_dir = _seed_dir(tmp_path, checkpoints=["checkpoint-10900"], best=None)

    choice = resolve_checkpoint(seed_dir.parent, seed_dir)

    assert choice.path.name == "checkpoint-10900"
    assert choice.source == CheckpointSource.HIGHEST_STEP
    assert any("NOT necessarily the best epoch" in note for note in choice.notes)


def test_a_recorded_best_that_is_absent_locally_falls_back_and_says_so(
    tmp_path: Path,
) -> None:
    seed_dir = _seed_dir(
        tmp_path, checkpoints=["checkpoint-10900"], best="checkpoint-1752"
    )

    choice = resolve_checkpoint(seed_dir.parent, seed_dir)

    assert choice.path.name == "checkpoint-10900"
    assert choice.source == CheckpointSource.HIGHEST_STEP
    assert choice.notes


def test_a_checkpoints_json_manifest_is_preferred_over_reading_the_state(
    tmp_path: Path,
) -> None:
    """Written by src/training/ingest.py; the state file is the older fallback."""

    seed_dir = _seed_dir(
        tmp_path,
        checkpoints=["checkpoint-500", "checkpoint-10900"],
        best="checkpoint-10900",
    )
    (seed_dir.parent / "checkpoints.json").write_text(
        json.dumps({"best_model_checkpoint": "/content/x/checkpoint-500"}),
        encoding="utf-8",
    )

    choice = resolve_checkpoint(seed_dir.parent, seed_dir)

    assert choice.path.name == "checkpoint-500"
    assert choice.source == CheckpointSource.CHECKPOINTS_JSON


def test_an_arm_with_no_checkpoints_returns_no_path_and_explains(
    tmp_path: Path,
) -> None:
    """One missing arm must not abort the other three."""

    (tmp_path / "real_only" / "seed_1337").mkdir(parents=True)

    choice = resolve_checkpoint(tmp_path / "real_only", tmp_path / "real_only" / "seed_1337")

    assert choice.path is None
    assert any("no checkpoint-*" in note for note in choice.notes)


# --------------------------------------------------------------------------
# EVAL-14 — a leak check that cannot fail is worse than none
# --------------------------------------------------------------------------


def _records(digest: str):
    return {
        arm: {
            "composition": {
                "arm": arm,
                "n_real_train": 3500,
                "n_synthetic": 0 if "syn" not in arm else 3500,
                "real_train_digest": digest,
            }
        }
        for arm in ARMS
    }


def test_the_real_frozen_split_passes_and_returns_its_evidence(
    frozen_split, real_digest, monkeypatch: pytest.MonkeyPatch
) -> None:
    integrity_checks: list[str] = []
    monkeypatch.setattr(
        eval_driver,
        "assert_test_untouched",
        lambda: integrity_checks.append("checked"),
    )

    evidence = leak_self_check(
        _records(real_digest),
        real_train_names=frozen_split["train"],
        test_names=frozen_split["test"],
        arms=ARMS,
    )

    assert evidence
    assert integrity_checks == ["checked"]
    assert any("assert_test_untouched" in line for line in evidence)


def test_a_planted_overlap_raises_rather_than_reporting_pass(
    frozen_split, real_digest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One train image smuggled into the test list must be fatal."""

    monkeypatch.setattr(eval_driver, "assert_test_untouched", lambda: None)
    poisoned = list(frozen_split["test"]) + [frozen_split["train"][0]]

    with pytest.raises(SplitLeakageError):
        leak_self_check(
            _records(real_digest),
            real_train_names=frozen_split["train"],
            test_names=poisoned,
            arms=ARMS,
)


def test_evaluation_json_is_flushed_before_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsync_calls: list[int] = []
    monkeypatch.setattr(
        eval_driver.os, "fsync", lambda descriptor: fsync_calls.append(descriptor)
    )

    eval_driver.atomic_write_json_value(tmp_path / "evaluation.json", {"ok": True})

    assert fsync_calls


def test_a_digest_that_does_not_match_the_frozen_split_is_refused(
    frozen_split,
) -> None:
    """An unverified training list would let the check print PASS on the wrong data."""

    with pytest.raises(TrainingListUnavailableError, match="DIFFERENT set"):
        training_image_names(
            {"composition": {"real_train_digest": "d" * 64}},
            "real_only",
            real_train_names=frozen_split["train"],
        )


def test_a_composition_with_no_digest_is_refused(frozen_split) -> None:
    with pytest.raises(TrainingListUnavailableError, match="UNCHECKED"):
        training_image_names(
            {"composition": {"n_real_train": 3500}},
            "real_only",
            real_train_names=frozen_split["train"],
        )


def test_the_matching_digest_returns_exactly_the_frozen_train_list(
    frozen_split, real_digest
) -> None:
    names = training_image_names(
        {"composition": {"real_train_digest": real_digest}},
        "real_only",
        real_train_names=frozen_split["train"],
    )

    assert set(names) == set(frozen_split["train"])


# --------------------------------------------------------------------------
# EVAL-07 — coordinates. The split is NOT one resolution (DATA-25).
# `original_sizes` and `evaluated_size` are both (WIDTH, HEIGHT).
# --------------------------------------------------------------------------


def _detection(image_id, bbox, score=0.9, category_id=0):
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "score": score,
    }


def test_boxes_are_rescaled_from_the_evaluated_size_to_each_image_size() -> None:
    # 640 -> 416 is 0.65 on both axes: 64x64 at (64, 64) becomes 41.6x41.6.
    converted = detections_for_evaluation(
        [_detection(1, (64, 64, 64, 64))],
        evaluated_size=(640.0, 640.0),
        original_sizes={1: (416.0, 416.0)},
    )

    assert converted[0]["bbox"] == pytest.approx([41.6, 41.6, 41.6, 41.6])


def test_a_non_square_image_scales_its_axes_independently() -> None:
    """416x415 is the PLURALITY of this dataset, so this is the common case."""

    converted = detections_for_evaluation(
        [_detection(1, (100, 100, 100, 100))],
        evaluated_size=(640.0, 640.0),
        original_sizes={1: (416.0, 415.0)},  # width 416, height 415
    )

    x, y, w, h = converted[0]["bbox"]
    # x and width scale by 416/640 = 0.65; y and height by 415/640 = 0.6484375.
    assert x == pytest.approx(65.0)
    assert w == pytest.approx(65.0)
    assert y == pytest.approx(64.84375)
    assert h == pytest.approx(64.84375)
    # A transposed implementation would make these equal.
    assert y != pytest.approx(x)


def test_each_image_uses_its_own_size_not_the_first_one() -> None:
    """One global factor would be right for image 1 and wrong for image 2."""

    converted = detections_for_evaluation(
        [_detection(1, (64, 0, 64, 64)), _detection(2, (64, 0, 64, 64))],
        evaluated_size=(640.0, 640.0),
        original_sizes={1: (416.0, 416.0), 2: (208.0, 208.0)},
    )

    assert converted[0]["bbox"][0] == pytest.approx(41.6)
    assert converted[1]["bbox"][0] == pytest.approx(20.8)


def test_a_detection_for_an_unknown_image_refuses_to_guess() -> None:
    with pytest.raises(KeyError, match="refusing to guess"):
        detections_for_evaluation(
            [_detection(99, (1, 1, 1, 1))],
            evaluated_size=(640.0, 640.0),
            original_sizes={1: (416.0, 416.0)},
        )


def test_the_processor_size_is_returned_as_width_then_height() -> None:
    assert processor_evaluated_size({"height": 640, "width": 480}) == (480.0, 640.0)


def test_an_aspect_preserving_processor_is_refused_not_approximated() -> None:
    """shortest_edge resizes every image differently; one factor pair cannot undo it."""

    with pytest.raises(EvalDriverError, match="shortest_edge"):
        processor_evaluated_size({"shortest_edge": 800})


@pytest.mark.parametrize("size", [{"height": 0, "width": 640}, {"height": 640, "width": -1}])
def test_a_non_positive_processor_size_is_refused(size) -> None:
    with pytest.raises(EvalDriverError, match="not positive"):
        processor_evaluated_size(size)


# --------------------------------------------------------------------------
# TRAIN-07 — the arms must have shared a step budget
# --------------------------------------------------------------------------


def _plan(steps=10900):
    return {
        arm: {"total_steps": steps, "real_image_exposures": 49.8, "epochs": 50.0}
        for arm in ARMS
    }


def test_matching_step_budgets_pass() -> None:
    records = {arm: {"total_steps": 10900} for arm in ARMS}

    assert verify_step_budget(_plan(), records) == ()


def test_one_arm_off_by_a_single_step_is_reported() -> None:
    records = {arm: {"total_steps": 10900} for arm in ARMS}
    records["filtered_syn"] = {"total_steps": 10899}

    problems = verify_step_budget(_plan(), records)

    assert problems
    assert any("filtered_syn" in problem for problem in problems)


def test_an_arm_recording_no_step_count_is_reported_not_skipped() -> None:
    records = {arm: {"total_steps": 10900} for arm in ARMS}
    records["filtered_syn"] = {}

    problems = verify_step_budget(_plan(), records)

    assert any("no total_steps" in problem for problem in problems)


def test_rf_inputs_require_explicit_isolated_evaluation_outputs() -> None:
    args = eval_driver.parse_args(
        [
            "--runs-root",
            "D:/runs_rfdetr",
            "--training-config",
            "configs/training_rfdetr.yaml",
        ]
    )

    with pytest.raises(EvalDriverError, match="--metrics-csv.*--report.*--predictions-root"):
        eval_driver.validate_output_isolation(
            args,
            default_runs_root=Path("D:/runs"),
            default_training_config=eval_driver.TRAINING_CONFIG,
            default_metrics_csv=Path(args.metrics_csv),
            default_report=Path(args.report),
            default_predictions_root=Path("D:/runs/predictions"),
            default_predictions_index=Path("results/predictions_index.json"),
        )


def test_rf_inputs_reject_aliases_of_primary_evaluation_outputs(
    tmp_path: Path,
) -> None:
    primary_metrics = tmp_path / "metrics.csv"
    primary_report = tmp_path / "report.md"
    primary_predictions = tmp_path / "predictions"
    primary_index = tmp_path / "predictions_index.json"
    alias = lambda path: path.parent / "unused" / ".." / path.name
    args = eval_driver.parse_args(
        [
            "--runs-root",
            str(tmp_path / "runs_rfdetr"),
            "--training-config",
            "configs/training_rfdetr.yaml",
            "--metrics-csv",
            str(alias(primary_metrics)),
            "--report",
            str(alias(primary_report)),
            "--predictions-root",
            str(primary_predictions / "unused" / ".."),
            "--predictions-index",
            str(alias(primary_index)),
        ]
    )

    with pytest.raises(
        EvalDriverError,
        match=(
            "--metrics-csv.*--report.*--predictions-root.*--predictions-index"
        ),
    ):
        eval_driver.validate_output_isolation(
            args,
            default_runs_root=tmp_path / "runs",
            default_training_config=eval_driver.TRAINING_CONFIG,
            default_metrics_csv=primary_metrics,
            default_report=primary_report,
            default_predictions_root=primary_predictions,
            default_predictions_index=primary_index,
        )


def test_corrupt_evaluation_prediction_index_is_not_silently_replaced(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.json"
    index.write_text("{broken", encoding="utf-8")

    with pytest.raises(EvalDriverError, match="prediction index"):
        eval_driver.read_prediction_index_strict(index)
