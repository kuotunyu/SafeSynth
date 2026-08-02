"""Production orchestration must preserve the frozen experiment while resuming safely."""

from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.training.arms import ArmComposition, digest_names

APPROVED = ("real_only", "filtered_syn", "standard_aug", "unfiltered_syn")


def _module():
    return importlib.import_module("scripts.train_arms")


def _composition(arm: str, *, synthetic: tuple[str, ...] = ()) -> ArmComposition:
    return ArmComposition(
        arm=arm,
        real_train=("train-a.png", "train-b.png"),
        real_val=("val-a.png",),
        synthetic=synthetic,
        augmentation_profile="real_only" if arm == "real_only" else "standard_aug",
        real_train_digest=digest_names(("train-a.png", "train-b.png")),
    )


@pytest.fixture
def job_inputs(tmp_path: Path):
    pool = tmp_path / "synthetic" / "m13_pool_1x"
    paths = SimpleNamespace(
        project_root=tmp_path,
        data_root=tmp_path,
        hardhat_raw=tmp_path / "hardhat",
        interim=tmp_path / "interim",
        synthetic=tmp_path / "synthetic",
        splits=tmp_path / "splits",
        reports=tmp_path / "reports",
        runs=tmp_path / "runs",
    )
    filtered = tuple(f"filtered-{index:04d}.png" for index in range(3_500))
    unfiltered = tuple(f"unfiltered-{index:04d}.png" for index in range(3_500))
    compositions = {
        "real_only": _composition("real_only"),
        "filtered_syn": _composition("filtered_syn", synthetic=filtered),
        "standard_aug": _composition("standard_aug"),
        "unfiltered_syn": _composition("unfiltered_syn", synthetic=unfiltered),
    }
    config = {
        "arms": list(APPROVED),
        "model": {"checkpoint": "Roboflow/rf-detr-nano"},
        "run": {"seed": 1337, "total_steps": 10_900},
    }
    return SimpleNamespace(
        config=config,
        paths=paths,
        pool=pool,
        compositions=compositions,
        runs_root=tmp_path / "runs_rfdetr",
    )


def test_default_jobs_follow_the_approved_order_and_dedicated_layout(job_inputs) -> None:
    """Sorting alphabetically or reusing the RT root would change/rewrite the run."""

    jobs = _module().build_jobs(
        job_inputs.config,
        job_inputs.paths,
        job_inputs.compositions,
        runs_root=job_inputs.runs_root,
        pool_tag="m13_pool_1x",
        sealed_test_names=("test-a.png",),
    )

    assert tuple(job.arm for job in jobs) == APPROVED
    assert all(
        job.paths.output_dir == job_inputs.runs_root / job.arm / "seed_1337"
        for job in jobs
    )
    assert jobs[0].paths.synthetic_coco is None
    assert jobs[1].paths.synthetic_coco == job_inputs.pool / "annotations_filtered_1x.json"
    assert jobs[2].paths.synthetic_coco is None
    assert jobs[3].paths.synthetic_coco == job_inputs.pool / "annotations_unfiltered_1x.json"


def test_a_test_name_in_training_or_validation_is_rejected(job_inputs) -> None:
    """Dropping this guard would let a clean-looking run train on sealed Test data."""

    leaked = dict(job_inputs.compositions)
    leaked["real_only"] = replace(
        leaked["real_only"], real_train=("train-a.png", "test-a.png")
    )

    with pytest.raises(Exception, match="Test"):
        _module().build_jobs(
            job_inputs.config,
            job_inputs.paths,
            leaked,
            runs_root=job_inputs.runs_root,
            pool_tag="m13_pool_1x",
            sealed_test_names=("test-a.png",),
        )


def _jobs(job_inputs):
    return _module().build_jobs(
        job_inputs.config,
        job_inputs.paths,
        job_inputs.compositions,
        runs_root=job_inputs.runs_root,
        pool_tag="m13_pool_1x",
        sealed_test_names=("test-a.png",),
    )


def _materialize_required_inputs(job_inputs) -> Path:
    manifest = job_inputs.paths.interim.parent / "splits" / "split_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    (job_inputs.paths.hardhat_raw / "images").mkdir(parents=True)
    job_inputs.paths.interim.mkdir(parents=True)
    (job_inputs.paths.interim / "coco_all.json").write_text("{}", encoding="utf-8")
    (job_inputs.pool / "images").mkdir(parents=True)
    for subset in ("filtered", "unfiltered"):
        (job_inputs.pool / f"annotations_{subset}_1x.json").write_text(
            "{}", encoding="utf-8"
        )
    job_inputs.runs_root.parent.mkdir(parents=True, exist_ok=True)
    return manifest


def test_preflight_reports_frozen_counts_digest_and_disk(job_inputs) -> None:
    """A changed subset size or low disk must be visible before model loading."""

    manifest = _materialize_required_inputs(job_inputs)
    usage = shutil._ntuple_diskusage(total=200 << 30, used=100 << 30, free=100 << 30)

    report = _module().preflight(
        _jobs(job_inputs),
        manifest_path=manifest,
        required_free_gib=50,
        expected_synthetic_count=3_500,
        disk_usage=lambda _: usage,
    )

    assert report.arms == APPROVED
    assert report.free_disk_gib == pytest.approx(100.0)
    assert report.required_free_gib == 50
    assert report.synthetic_counts == {
        "real_only": 0,
        "filtered_syn": 3_500,
        "standard_aug": 0,
        "unfiltered_syn": 3_500,
    }
    assert report.real_train_digest == digest_names(("train-a.png", "train-b.png"))


def test_preflight_names_every_missing_required_input(job_inputs) -> None:
    manifest = _materialize_required_inputs(job_inputs)
    (job_inputs.paths.interim / "coco_all.json").unlink()
    (job_inputs.pool / "annotations_filtered_1x.json").unlink()

    with pytest.raises(Exception) as raised:
        _module().preflight(
            _jobs(job_inputs),
            manifest_path=manifest,
            required_free_gib=1,
            expected_synthetic_count=3_500,
        )

    message = str(raised.value)
    assert "coco_all.json" in message
    assert "annotations_filtered_1x.json" in message


def test_preflight_rejects_disk_below_the_reserve(job_inputs) -> None:
    manifest = _materialize_required_inputs(job_inputs)
    usage = shutil._ntuple_diskusage(total=200 << 30, used=191 << 30, free=9 << 30)

    with pytest.raises(Exception, match="9.0 GiB.*10.0 GiB"):
        _module().preflight(
            _jobs(job_inputs),
            manifest_path=manifest,
            required_free_gib=10,
            expected_synthetic_count=3_500,
            disk_usage=lambda _: usage,
        )


def _write_checkpoint(job, step: int) -> Path:
    checkpoint = job.paths.output_dir / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    return checkpoint


def _complete_record(job, *, checkpoint: str = "checkpoint-10900") -> dict:
    return {
        "arm": job.arm,
        "seed": 1337,
        "total_steps": 10_900,
        "model_checkpoint": "Roboflow/rf-detr-nano",
        "config_sha256": job.config_sha256,
        "composition": job.composition.summary(),
        "train_loss": 1.25,
        "eval_metrics": {"eval_map": 0.1},
        "started_at_utc": "2026-08-02T00:00:00+00:00",
        "finished_at_utc": "2026-08-02T01:00:00+00:00",
        "latest_checkpoint": checkpoint,
    }


def test_run_inspection_distinguishes_absent_resumable_and_complete(job_inputs) -> None:
    """Directory existence alone must never be accepted as a completed arm."""

    jobs = _jobs(job_inputs)
    assert _module().inspect_run(jobs[0]) == "absent"

    _write_checkpoint(jobs[1], 500)
    assert _module().inspect_run(jobs[1]) == "resumable"

    _write_checkpoint(jobs[2], 10_900)
    _module().atomic_write_json(
        jobs[2].paths.output_dir / "run_record.json", _complete_record(jobs[2])
    )
    assert _module().inspect_run(jobs[2]) == "complete"


def test_a_conflicting_completed_record_is_an_unsafe_collision(job_inputs) -> None:
    job = _jobs(job_inputs)[0]
    _write_checkpoint(job, 10_900)
    record = _complete_record(job)
    record["model_checkpoint"] = "wrong/model"
    _module().atomic_write_json(job.paths.output_dir / "run_record.json", record)

    with pytest.raises(Exception, match="model_checkpoint"):
        _module().inspect_run(job)


def test_non_finite_completed_loss_is_not_accepted(job_inputs) -> None:
    job = _jobs(job_inputs)[0]
    _write_checkpoint(job, 10_900)
    record = _complete_record(job)
    record["train_loss"] = float("nan")
    _module().atomic_write_json(job.paths.output_dir / "run_record.json", record)

    with pytest.raises(Exception, match="train_loss"):
        _module().inspect_run(job)


def test_completed_runs_skip_and_checkpoint_only_runs_resume(job_inputs) -> None:
    jobs = _jobs(job_inputs)
    _write_checkpoint(jobs[0], 10_900)
    _module().atomic_write_json(
        jobs[0].paths.output_dir / "run_record.json", _complete_record(jobs[0])
    )
    _write_checkpoint(jobs[1], 500)
    calls: list[str] = []

    def train_one(job, config):
        calls.append(job.arm)
        _write_checkpoint(job, 10_900)
        return {
            "arm": job.arm,
            "seed": job.seed,
            "total_steps": job.total_steps,
            "train_loss": 1.0,
            "eval_metrics": {"eval_map": 0.2},
            "composition": job.composition.summary(),
        }

    summary = job_inputs.runs_root.parent / "reports" / "orchestration.json"
    records = job_inputs.runs_root.parent / "run_records"
    code = _module().run_jobs(
        jobs,
        config=job_inputs.config,
        train_one=train_one,
        summary_path=summary,
        run_records_root=records,
    )

    assert code == 0
    assert calls == ["filtered_syn", "standard_aug", "unfiltered_syn"]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["arms"]["real_only"]["status"] == "skipped_complete"
    assert payload["arms"]["filtered_syn"]["status"] == "completed"
    for job in jobs:
        assert (records / job.arm / "run_record.json").is_file()
        assert _module().inspect_run(job) == "complete"


def test_first_training_failure_stops_later_arms_and_is_recorded(job_inputs) -> None:
    jobs = _jobs(job_inputs)
    calls: list[str] = []

    def fail_filtered(job, config):
        calls.append(job.arm)
        if job.arm == "filtered_syn":
            raise RuntimeError("CUDA out of memory")
        _write_checkpoint(job, 10_900)
        return {
            "arm": job.arm,
            "seed": job.seed,
            "total_steps": job.total_steps,
            "train_loss": 1.0,
            "eval_metrics": {},
            "composition": job.composition.summary(),
        }

    summary = job_inputs.runs_root.parent / "reports" / "orchestration.json"
    code = _module().run_jobs(
        jobs,
        config=job_inputs.config,
        train_one=fail_filtered,
        summary_path=summary,
        run_records_root=job_inputs.runs_root.parent / "run_records",
    )

    assert code == 1
    assert calls == ["real_only", "filtered_syn"]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["arms"]["filtered_syn"]["status"] == "failed"
    assert payload["arms"]["filtered_syn"]["error"] == "CUDA out of memory"
    assert payload["arms"]["standard_aug"]["status"] == "pending"


def test_explicit_arm_subset_preserves_config_order(job_inputs) -> None:
    """CLI argument order must not silently reorder the approved experiment."""

    selected = _module().select_arms(
        job_inputs.config["arms"], ("unfiltered_syn", "real_only")
    )

    assert selected == ("real_only", "unfiltered_syn")


def test_unknown_requested_arm_is_rejected(job_inputs) -> None:
    with pytest.raises(Exception, match="unknown arm"):
        _module().select_arms(job_inputs.config["arms"], ("invented",))


def test_dry_run_calls_no_training_and_writes_no_summary(
    job_inputs, monkeypatch, capsys
) -> None:
    """Removing the dry-run return would allocate the model during preflight."""

    module = _module()
    manifest = _materialize_required_inputs(job_inputs)
    monkeypatch.setattr(module, "load_project_paths", lambda: job_inputs.paths)
    monkeypatch.setattr(module, "load_training_config", lambda path: job_inputs.config)
    monkeypatch.setattr(
        module,
        "build_all_arms",
        lambda **kwargs: job_inputs.compositions,
    )
    monkeypatch.setattr(
        module,
        "split_real_images",
        lambda path: {
            "train": ("train-a.png", "train-b.png"),
            "val": ("val-a.png",),
            "test": ("test-a.png",),
        },
    )
    calls = []
    summary = job_inputs.paths.reports / "orchestration.json"

    code = module.main(
        [
            "--config",
            "configs/training_rfdetr.yaml",
            "--runs-root",
            str(job_inputs.runs_root),
            "--run-records-root",
            str(job_inputs.runs_root.parent / "records"),
            "--summary",
            str(summary),
            "--manifest",
            str(manifest),
            "--min-free-gib",
            "0.01",
            "--dry-run",
        ],
        train_one=lambda job, config: calls.append(job.arm),
    )

    assert code == 0
    assert calls == []
    assert not summary.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["arms"] == list(APPROVED)
    assert printed["synthetic_counts"]["filtered_syn"] == 3_500
