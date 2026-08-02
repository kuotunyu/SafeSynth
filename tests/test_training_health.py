"""Unattended detector runs must stop on measurable safety violations."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from src.training.config import load_training_config
from src.training.health import (
    GpuProcess,
    HealthSnapshot,
    SafetyPolicyError,
    TrainerHealthCallback,
    UnattendedSafetyPolicy,
    UnattendedWatchdog,
)
from src.training.trainer import build_training_arguments

NOW = datetime(2026, 8, 2, 14, 0, tzinfo=UTC)


def _snapshot(**overrides) -> HealthSnapshot:
    values = {
        "observed_at_utc": NOW,
        "cuda_available": True,
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_temperature_c": 55.0,
        "gpu_memory_used_mib": 1_000,
        "gpu_memory_total_mib": 24_564,
        "gpu_processes": (),
        "disk_free_gib": 500.0,
    }
    values.update(overrides)
    return HealthSnapshot(**values)


@pytest.mark.parametrize(
    ("snapshot", "deadline", "message"),
    [
        (_snapshot(cuda_available=False), NOW + timedelta(hours=1), "CUDA"),
        (_snapshot(gpu_name="NVIDIA T4"), NOW + timedelta(hours=1), "RTX 4090"),
        (_snapshot(gpu_memory_total_mib=16_384), NOW + timedelta(hours=1), "VRAM"),
        (_snapshot(disk_free_gib=49.9), NOW + timedelta(hours=1), "disk"),
        (_snapshot(gpu_temperature_c=86.0), NOW + timedelta(hours=1), "temperature"),
        (
            _snapshot(
                gpu_processes=(GpuProcess(pid=999, process_name="python.exe"),)
            ),
            NOW + timedelta(hours=1),
            "competing GPU process",
        ),
        (_snapshot(), NOW - timedelta(seconds=1), "deadline"),
    ],
)
def test_policy_rejects_each_unattended_safety_violation(
    tmp_path: Path,
    snapshot: HealthSnapshot,
    deadline: datetime,
    message: str,
) -> None:
    """Removing any gate would let a known unsafe run continue overnight."""

    policy = UnattendedSafetyPolicy(
        output_root=tmp_path,
        health_log=tmp_path / "health.jsonl",
        deadline_utc=deadline,
        min_free_gib=50.0,
        max_gpu_temperature_c=85.0,
        own_pid=123,
        snapshot_reader=lambda _: snapshot,
    )

    with pytest.raises(SafetyPolicyError, match=message):
        policy.check(stage="before_arm", arm="real_only")

    event = json.loads((tmp_path / "health.jsonl").read_text(encoding="utf-8"))
    assert event["status"] == "rejected"
    assert event["violations"]


def test_policy_records_a_passing_snapshot_and_ignores_its_own_gpu_pid(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        gpu_memory_used_mib=13_000,
        gpu_processes=(GpuProcess(pid=123, process_name="python.exe"),),
    )
    policy = UnattendedSafetyPolicy(
        output_root=tmp_path,
        health_log=tmp_path / "health.jsonl",
        deadline_utc=NOW + timedelta(hours=16),
        min_free_gib=50.0,
        max_gpu_temperature_c=85.0,
        own_pid=123,
        snapshot_reader=lambda _: snapshot,
    )

    returned = policy.check(stage="training_log", arm="real_only", step=50)

    assert returned == snapshot
    event = json.loads((tmp_path / "health.jsonl").read_text(encoding="utf-8"))
    assert event["status"] == "passed"
    assert event["step"] == 50


def test_policy_rejects_unknown_foreign_compute_even_without_python_in_its_name(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        gpu_processes=(GpuProcess(pid=999, process_name="render.exe"),)
    )
    policy = UnattendedSafetyPolicy(
        output_root=tmp_path,
        health_log=tmp_path / "health.jsonl",
        deadline_utc=NOW + timedelta(hours=1),
        min_free_gib=50.0,
        max_gpu_temperature_c=85.0,
        own_pid=123,
        snapshot_reader=lambda _: snapshot,
    )

    with pytest.raises(SafetyPolicyError, match="render.exe"):
        policy.check(stage="training_log", arm="real_only", step=50)


def test_policy_allows_known_windows_desktop_gpu_clients(tmp_path: Path) -> None:
    snapshot = _snapshot(
        gpu_processes=(
            GpuProcess(pid=999, process_name=r"C:\Windows\explorer.exe"),
            GpuProcess(pid=998, process_name=r"C:\Program Files\Google\Chrome\chrome.exe"),
            GpuProcess(pid=997, process_name=r"C:\Windows\System32\ShellHost.exe"),
        )
    )
    policy = UnattendedSafetyPolicy(
        output_root=tmp_path,
        health_log=tmp_path / "health.jsonl",
        deadline_utc=NOW + timedelta(hours=1),
        min_free_gib=50.0,
        max_gpu_temperature_c=85.0,
        own_pid=123,
        snapshot_reader=lambda _: snapshot,
    )

    assert policy.check(stage="training_log", arm="real_only", step=50) == snapshot


def test_unidentified_windows_client_requires_memory_or_ownership_evidence(
    tmp_path: Path,
) -> None:
    unidentified = GpuProcess(pid=999, process_name="[Insufficient Permissions]")
    safe = _snapshot(gpu_memory_used_mib=1_000, gpu_processes=(unidentified,))
    unsafe = _snapshot(gpu_memory_used_mib=8_000, gpu_processes=(unidentified,))

    def policy(snapshot: HealthSnapshot, name: str) -> UnattendedSafetyPolicy:
        return UnattendedSafetyPolicy(
            output_root=tmp_path,
            health_log=tmp_path / f"{name}.jsonl",
            deadline_utc=NOW + timedelta(hours=1),
            own_pid=123,
            snapshot_reader=lambda _: snapshot,
        )

    assert policy(safe, "safe").check(stage="startup") == safe
    with pytest.raises(SafetyPolicyError, match="Insufficient Permissions"):
        policy(unsafe, "unsafe").check(stage="startup")


def test_watchdog_hard_exits_on_deadline_without_trainer_logs(tmp_path: Path) -> None:
    expired = _snapshot(observed_at_utc=NOW + timedelta(hours=2))
    policy = UnattendedSafetyPolicy(
        output_root=tmp_path,
        health_log=tmp_path / "health.jsonl",
        deadline_utc=NOW + timedelta(hours=1),
        own_pid=123,
        snapshot_reader=lambda _: expired,
    )
    terminated = Event()
    exit_codes: list[int] = []

    def hard_exit(code: int) -> None:
        exit_codes.append(code)
        terminated.set()

    watchdog = UnattendedWatchdog(
        policy=policy,
        arm="real_only",
        output_dir=tmp_path / "run",
        poll_interval_seconds=0.01,
        max_progress_stall_seconds=60.0,
        hard_exit=hard_exit,
    )

    with watchdog:
        assert terminated.wait(timeout=1.0)

    assert exit_codes == [70]
    events = [
        json.loads(line)
        for line in (tmp_path / "health.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["stage"] == "watchdog"
    assert events[-1]["status"] == "rejected"


def test_watchdog_rejects_stale_checkpoint_progress(tmp_path: Path) -> None:
    ticks = iter((0.0, 3_601.0))
    policy = UnattendedSafetyPolicy(
        output_root=tmp_path,
        health_log=tmp_path / "health.jsonl",
        deadline_utc=NOW + timedelta(hours=2),
        own_pid=123,
        snapshot_reader=lambda _: _snapshot(),
    )
    watchdog = UnattendedWatchdog(
        policy=policy,
        arm="filtered_syn",
        output_dir=tmp_path / "run",
        poll_interval_seconds=60.0,
        max_progress_stall_seconds=3_600.0,
        monotonic=lambda: next(ticks),
        hard_exit=lambda _: None,
    )

    with pytest.raises(SafetyPolicyError, match="checkpoint progress"):
        watchdog.poll_once()

    event = json.loads((tmp_path / "health.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert event["stage"] == "watchdog_progress"
    assert event["status"] == "rejected"


def test_trainer_callback_rejects_non_finite_logs_before_next_step(tmp_path: Path) -> None:
    policy = UnattendedSafetyPolicy(
        output_root=tmp_path,
        health_log=tmp_path / "health.jsonl",
        deadline_utc=NOW + timedelta(hours=1),
        min_free_gib=50.0,
        max_gpu_temperature_c=85.0,
        own_pid=123,
        snapshot_reader=lambda _: _snapshot(),
    )
    callback = TrainerHealthCallback(policy=policy, arm="real_only")

    with pytest.raises(SafetyPolicyError, match="non-finite.*loss"):
        callback.on_log(
            SimpleNamespace(),
            SimpleNamespace(global_step=50),
            SimpleNamespace(),
            logs={"loss": float("nan")},
        )


def test_trainer_callback_checks_system_health_on_logs(tmp_path: Path) -> None:
    observed: list[tuple[str, str, int | None]] = []

    class RecordingPolicy:
        def check(self, *, stage: str, arm: str, step: int | None = None):
            observed.append((stage, arm, step))

    callback = TrainerHealthCallback(policy=RecordingPolicy(), arm="filtered_syn")
    callback.on_log(
        SimpleNamespace(),
        SimpleNamespace(global_step=100),
        SimpleNamespace(),
        logs={"loss": 1.25, "grad_norm": 0.5},
    )

    assert observed == [("training_log", "filtered_syn", 100)]


def test_training_arguments_expose_nan_and_inf_to_the_health_callback(
    tmp_path: Path,
) -> None:
    config = load_training_config("configs/training_rfdetr.yaml")

    arguments = build_training_arguments(
        output_dir=str(tmp_path),
        config=config,
        total_steps=4,
        seed=1337,
        use_bf16=True,
        dataloader_num_workers=0,
    )

    assert arguments.logging_nan_inf_filter is False
