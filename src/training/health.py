"""Measured safety gates for long, unattended detector training."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback


class SafetyPolicyError(RuntimeError):
    """A measured condition makes unattended training unsafe to continue."""


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    process_name: str


@dataclass(frozen=True)
class HealthSnapshot:
    observed_at_utc: datetime
    cuda_available: bool
    gpu_name: str
    gpu_temperature_c: float
    gpu_memory_used_mib: int
    gpu_memory_total_mib: int
    gpu_processes: tuple[GpuProcess, ...]
    disk_free_gib: float


def _existing_ancestor(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _run_nvidia_smi(*arguments: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def _parse_gpu_processes(output: str) -> tuple[GpuProcess, ...]:
    processes: list[GpuProcess] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        pid_text, separator, process_name = line.partition(",")
        if not separator:
            continue
        try:
            pid = int(pid_text.strip())
        except ValueError:
            continue
        processes.append(GpuProcess(pid=pid, process_name=process_name.strip()))
    return tuple(processes)


def read_health_snapshot(output_root: Path) -> HealthSnapshot:
    """Read CUDA, GPU and disk evidence without allocating model memory."""

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    temperature = float("nan")
    memory_used = 0
    memory_total = 0
    processes: tuple[GpuProcess, ...] = ()
    if cuda_available:
        gpu_line = _run_nvidia_smi(
            "--query-gpu=temperature.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ).strip()
        fields = [field.strip() for field in gpu_line.split(",")]
        if len(fields) != 3:
            raise SafetyPolicyError(f"unexpected nvidia-smi GPU output: {gpu_line!r}")
        temperature = float(fields[0])
        memory_used = int(fields[1])
        memory_total = int(fields[2])
        process_output = _run_nvidia_smi(
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        )
        processes = _parse_gpu_processes(process_output)

    disk = shutil.disk_usage(_existing_ancestor(output_root))
    return HealthSnapshot(
        observed_at_utc=datetime.now(UTC),
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_temperature_c=temperature,
        gpu_memory_used_mib=memory_used,
        gpu_memory_total_mib=memory_total,
        gpu_processes=processes,
        disk_free_gib=float(disk.free) / (1024**3),
    )


_WINDOWS_DESKTOP_GPU_CLIENTS = {
    "[insufficient permissions]",
    "applicationframehost.exe",
    "armourydevice.exe",
    "asus_framework.exe",
    "chatgpt.exe",
    "chrome.exe",
    "crossdeviceresume.exe",
    "explorer.exe",
    "firefox.exe",
    "lockapp.exe",
    "msedgewebview2.exe",
    "nvidia share.exe",
    "phoneexperiencehost.exe",
    "searchhost.exe",
    "shellhost.exe",
    "shellexperiencehost.exe",
    "startmenuexperiencehost.exe",
    "steamwebhelper.exe",
    "systemsettings.exe",
    "texinputhost.exe",
    "textinputhost.exe",
}


def _is_allowlisted_desktop_gpu_client(process_name: str) -> bool:
    lowered = process_name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
    return lowered in _WINDOWS_DESKTOP_GPU_CLIENTS


@dataclass
class UnattendedSafetyPolicy:
    output_root: Path
    health_log: Path
    deadline_utc: datetime
    min_free_gib: float = 50.0
    max_gpu_temperature_c: float = 85.0
    expected_gpu_name: str = "NVIDIA GeForce RTX 4090"
    min_gpu_memory_total_mib: int = 23_000
    own_pid: int = field(default_factory=os.getpid)
    snapshot_reader: Callable[[Path], HealthSnapshot] = read_health_snapshot

    def _append_event(self, event: Mapping[str, Any]) -> None:
        path = Path(self.health_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def check(
        self, *, stage: str, arm: str | None = None, step: int | None = None
    ) -> HealthSnapshot:
        snapshot = self.snapshot_reader(Path(self.output_root))
        competitors = tuple(
            process
            for process in snapshot.gpu_processes
            if process.pid != self.own_pid
            and not _is_allowlisted_desktop_gpu_client(process.process_name)
        )
        violations: list[str] = []
        if not snapshot.cuda_available:
            violations.append("CUDA is unavailable")
        elif self.expected_gpu_name not in snapshot.gpu_name:
            violations.append(
                f"expected RTX 4090 GPU ({self.expected_gpu_name!r}), "
                f"found {snapshot.gpu_name!r}"
            )
        if (
            snapshot.cuda_available
            and snapshot.gpu_memory_total_mib < self.min_gpu_memory_total_mib
        ):
            violations.append(
                f"GPU VRAM {snapshot.gpu_memory_total_mib} MiB is below "
                f"{self.min_gpu_memory_total_mib} MiB"
            )
        if snapshot.disk_free_gib < self.min_free_gib:
            violations.append(
                f"disk free {snapshot.disk_free_gib:.1f} GiB is below "
                f"{self.min_free_gib:.1f} GiB"
            )
        if not math.isfinite(snapshot.gpu_temperature_c) or (
            snapshot.gpu_temperature_c > self.max_gpu_temperature_c
        ):
            violations.append(
                f"GPU temperature {snapshot.gpu_temperature_c!r} exceeds "
                f"{self.max_gpu_temperature_c:.1f} C"
            )
        if competitors:
            rendered = ", ".join(
                f"{process.pid}:{process.process_name}" for process in competitors
            )
            violations.append(f"competing GPU process detected: {rendered}")
        own_gpu_process = any(
            process.pid == self.own_pid for process in snapshot.gpu_processes
        )
        if (
            stage in {"startup", "before_arm"}
            and not own_gpu_process
            and snapshot.gpu_memory_used_mib > 4_096
        ):
            violations.append(
                f"GPU already uses {snapshot.gpu_memory_used_mib} MiB before allocation"
            )
        if snapshot.observed_at_utc > self.deadline_utc:
            violations.append(
                f"unattended deadline {self.deadline_utc.isoformat()} has passed"
            )

        event = {
            **asdict(snapshot),
            "observed_at_utc": snapshot.observed_at_utc.isoformat(),
            "gpu_processes": [asdict(process) for process in snapshot.gpu_processes],
            "stage": stage,
            "arm": arm,
            "step": step,
            "deadline_utc": self.deadline_utc.isoformat(),
            "status": "rejected" if violations else "passed",
            "violations": violations,
        }
        self._append_event(event)
        if violations:
            raise SafetyPolicyError("; ".join(violations))
        return snapshot


def validate_finite_logs(logs: Mapping[str, Any] | None) -> None:
    for name, value in (logs or {}).items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SafetyPolicyError(f"non-finite training log {name}: {value!r}")


class TrainerHealthCallback(TrainerCallback):
    """Run the policy at every Trainer logging boundary during an arm."""

    def __init__(self, *, policy: UnattendedSafetyPolicy, arm: str):
        self.policy = policy
        self.arm = arm

    def on_log(self, args, state, control, logs=None, **kwargs):
        validate_finite_logs(logs)
        self.policy.check(
            stage="training_log",
            arm=self.arm,
            step=int(state.global_step),
        )
        return control
