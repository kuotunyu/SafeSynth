"""Run frozen detector arms sequentially with resumable, auditable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from src.data.paths import load_project_paths
from src.training.arms import ArmComposition, build_all_arms, split_real_images
from src.training.config import load_training_config
from src.training.health import (
    TrainerHealthCallback,
    UnattendedSafetyPolicy,
    UnattendedWatchdog,
)
from src.training.ingest import latest_checkpoint
from src.training.run import RunPaths, run_arm
from src.training.trainer import find_resumable_checkpoint


class TrainingOrchestrationError(RuntimeError):
    """The requested run would violate the frozen experiment or its recovery rules."""


@dataclass(frozen=True)
class ArmJob:
    arm: str
    seed: int
    total_steps: int
    composition: ArmComposition
    paths: RunPaths
    config_sha256: str
    model_checkpoint: str


@dataclass(frozen=True)
class PreflightReport:
    arms: tuple[str, ...]
    free_disk_gib: float
    required_free_gib: float
    real_train_digest: str
    synthetic_counts: Mapping[str, int]


@contextmanager
def orchestration_lock(
    path: Path, *, owner: Mapping[str, Any] | None = None
):
    """Hold a non-blocking OS lock; stale metadata never blocks crash recovery."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            path.open("xb").close()
        except FileExistsError:
            pass
    try:
        stream = path.open("r+b")
    except OSError as error:
        raise TrainingOrchestrationError(
            f"orchestration is already owned; cannot open {path}: {error}"
        ) from error
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                try:
                    current = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    current = "<owner metadata is locked>"
                raise TrainingOrchestrationError(
                    f"orchestration is already owned: {current}"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                try:
                    current = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    current = "<owner metadata is locked>"
                raise TrainingOrchestrationError(
                    f"orchestration is already owned: {current}"
                ) from error
        locked = True
        metadata = dict(
            owner
            or {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "acquired_at_utc": _utc_now(),
            }
        )
        encoded = (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
        stream.seek(0)
        stream.truncate(0)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        yield metadata
    finally:
        if locked:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def resolved_config_digest(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace a JSON object atomically so an interruption leaves valid evidence."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def build_jobs(
    config: Mapping[str, Any],
    project_paths: Any,
    compositions: Mapping[str, ArmComposition],
    *,
    runs_root: Path,
    pool_tag: str,
    sealed_test_names: Sequence[str],
    selected_arms: Sequence[str] | None = None,
) -> tuple[ArmJob, ...]:
    """Resolve approved arms into isolated run paths without touching CUDA."""

    arms = tuple(
        str(arm)
        for arm in (config["arms"] if selected_arms is None else selected_arms)
    )
    unknown = [arm for arm in arms if arm not in compositions]
    if unknown:
        raise TrainingOrchestrationError(f"unknown arm(s): {unknown}")

    sealed = set(sealed_test_names)
    for arm in arms:
        composition = compositions[arm]
        leaked = sealed.intersection(
            (*composition.real_train, *composition.real_val)
        )
        if leaked:
            raise TrainingOrchestrationError(
                f"Arm {arm!r} would read {len(leaked)} sealed Test image(s)"
            )

    pool = Path(project_paths.synthetic) / pool_tag
    seed = int(config["run"]["seed"])
    steps = int(config["run"]["total_steps"])
    digest = resolved_config_digest(config)
    checkpoint = str(config["model"]["checkpoint"])
    jobs: list[ArmJob] = []
    for arm in arms:
        subset = (
            "filtered"
            if arm == "filtered_syn"
            else "unfiltered" if arm == "unfiltered_syn" else None
        )
        jobs.append(
            ArmJob(
                arm=arm,
                seed=seed,
                total_steps=steps,
                composition=compositions[arm],
                paths=RunPaths(
                    real_images=Path(project_paths.hardhat_raw) / "images",
                    real_coco=Path(project_paths.interim) / "coco_all.json",
                    synthetic_images=pool / "images",
                    synthetic_coco=(
                        pool / f"annotations_{subset}_1x.json" if subset else None
                    ),
                    output_dir=Path(runs_root) / arm / f"seed_{seed}",
                ),
                config_sha256=digest,
                model_checkpoint=checkpoint,
            )
        )
    return tuple(jobs)


def _existing_ancestor(path: Path) -> Path:
    candidate = Path(path)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def preflight(
    jobs: Sequence[ArmJob],
    *,
    manifest_path: Path,
    required_free_gib: float,
    expected_synthetic_count: int,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> PreflightReport:
    """Validate all frozen inputs and disk reserve without allocating a model."""

    if not jobs:
        raise TrainingOrchestrationError("no arms selected")

    first = jobs[0]
    required: list[tuple[Path, str]] = [
        (Path(manifest_path), "file"),
        (first.paths.real_images, "directory"),
        (first.paths.real_coco, "file"),
        (first.paths.synthetic_images, "directory"),
    ]
    required.extend(
        (job.paths.synthetic_coco, "file")
        for job in jobs
        if job.paths.synthetic_coco is not None
    )
    missing = [
        str(path)
        for path, kind in dict.fromkeys(required)
        if (kind == "file" and not path.is_file())
        or (kind == "directory" and not path.is_dir())
    ]
    if missing:
        raise TrainingOrchestrationError(
            "missing required input(s): " + ", ".join(missing)
        )

    counts = {job.arm: len(job.composition.synthetic) for job in jobs}
    for arm, count in counts.items():
        expected = expected_synthetic_count if arm.endswith("_syn") else 0
        if count != expected:
            raise TrainingOrchestrationError(
                f"{arm} has {count} synthetic images; expected {expected}"
            )

    digests = {job.composition.real_train_digest for job in jobs}
    if len(digests) != 1:
        raise TrainingOrchestrationError(
            f"arms disagree on the frozen real-train digest: {sorted(digests)}"
        )

    usage = disk_usage(_existing_ancestor(first.paths.output_dir))
    free_gib = float(usage.free) / (1024**3)
    if free_gib < required_free_gib:
        raise TrainingOrchestrationError(
            f"only {free_gib:.1f} GiB free; require {required_free_gib:.1f} GiB"
        )

    return PreflightReport(
        arms=tuple(job.arm for job in jobs),
        free_disk_gib=free_gib,
        required_free_gib=float(required_free_gib),
        real_train_digest=next(iter(digests)),
        synthetic_counts=counts,
    )


def _read_record(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingOrchestrationError(
            f"cannot read run record {path}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise TrainingOrchestrationError(f"run record {path} is not a JSON object")
    return payload


def _finite_record_values(record: Mapping[str, Any]) -> None:
    values: dict[str, Any] = {"train_loss": record.get("train_loss")}
    metrics = record.get("eval_metrics", {})
    if isinstance(metrics, Mapping):
        values.update({f"eval_metrics.{key}": value for key, value in metrics.items()})
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TrainingOrchestrationError(f"{name} is not finite: {value!r}")


def _validate_final_checkpoint(checkpoint: Path, *, expected_step: int) -> None:
    expected_name = f"checkpoint-{expected_step}"
    if checkpoint.name != expected_name:
        raise TrainingOrchestrationError(
            f"completed run names {checkpoint.name}; expected {expected_name}"
        )

    state_path = checkpoint / "trainer_state.json"
    state = _read_record(state_path)
    if state.get("global_step") != expected_step:
        raise TrainingOrchestrationError(
            f"{state_path} global_step={state.get('global_step')!r}; "
            f"expected {expected_step}"
        )

    safetensors_path = checkpoint / "model.safetensors"
    safetensors_index = checkpoint / "model.safetensors.index.json"
    bin_path = checkpoint / "pytorch_model.bin"
    bin_index = checkpoint / "pytorch_model.bin.index.json"
    if safetensors_path.is_file():
        _validate_safetensors_file(safetensors_path)
    elif safetensors_index.is_file():
        _validate_sharded_weights(safetensors_index, safetensors=True)
    elif bin_path.is_file():
        _validate_torch_weights_file(bin_path)
    elif bin_index.is_file():
        _validate_sharded_weights(bin_index, safetensors=False)
    else:
        raise TrainingOrchestrationError(
            f"completed checkpoint {checkpoint} has no readable model weights"
        )


def _validate_safetensors_file(path: Path) -> None:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            keys = tuple(stream.keys())
            if not keys:
                raise TrainingOrchestrationError(f"safetensors file {path} is empty")
            for key in keys:
                stream.get_slice(key)
    except TrainingOrchestrationError:
        raise
    except Exception as error:
        raise TrainingOrchestrationError(
            f"cannot read safetensors file {path}: {type(error).__name__}: {error}"
        ) from error


def _validate_torch_weights_file(path: Path) -> None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as error:
        raise TrainingOrchestrationError(
            f"cannot read PyTorch weights {path}: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, Mapping) or not payload:
        raise TrainingOrchestrationError(f"PyTorch weights {path} are empty")


def _validate_sharded_weights(index_path: Path, *, safetensors: bool) -> None:
    index = _read_record(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise TrainingOrchestrationError(
            f"sharded weight index {index_path} has no weight_map"
        )
    shards = sorted({str(name) for name in weight_map.values()})
    for shard_name in shards:
        shard = index_path.parent / shard_name
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise TrainingOrchestrationError(
                f"sharded weight index {index_path} names missing shard {shard}"
            )
        if safetensors:
            _validate_safetensors_file(shard)
        else:
            _validate_torch_weights_file(shard)


def inspect_run(job: ArmJob) -> str:
    """Return absent/resumable/complete; reject any conflicting completion claim."""

    record_path = job.paths.output_dir / "run_record.json"
    resumable = find_resumable_checkpoint(job.paths.output_dir)
    if not record_path.is_file():
        return "resumable" if resumable else "absent"

    record = _read_record(record_path)
    orchestrator_keys = {
        "model_checkpoint",
        "config_sha256",
        "started_at_utc",
        "finished_at_utc",
        "latest_checkpoint",
    }
    if orchestrator_keys.isdisjoint(record):
        raw_expected = {
            "arm": job.arm,
            "seed": job.seed,
            "total_steps": job.total_steps,
            "composition": job.composition.summary(),
        }
        mismatched = {
            name: (record.get(name), value)
            for name, value in raw_expected.items()
            if record.get(name) != value
        }
        if mismatched:
            raise TrainingOrchestrationError(
                f"conflicting raw run record {record_path}: {mismatched!r}"
            )
        _finite_record_values(record)
        if resumable is None:
            raise TrainingOrchestrationError(
                f"raw run record {record_path} has no resumable checkpoint"
            )
        return "resumable"

    expected = {
        "arm": job.arm,
        "seed": job.seed,
        "total_steps": job.total_steps,
        "model_checkpoint": job.model_checkpoint,
        "config_sha256": job.config_sha256,
        "composition": job.composition.summary(),
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise TrainingOrchestrationError(
                f"conflicting run record {record_path}: {name}="
                f"{record.get(name)!r}, expected {value!r}"
            )
    _finite_record_values(record)

    checkpoint_name = record.get("latest_checkpoint")
    if not isinstance(checkpoint_name, str) or not checkpoint_name:
        raise TrainingOrchestrationError(
            f"completed run record {record_path} has no latest_checkpoint"
        )
    checkpoint = job.paths.output_dir / checkpoint_name
    if not checkpoint.is_dir():
        raise TrainingOrchestrationError(
            f"completed run record names unreadable checkpoint {checkpoint}"
        )
    newest = latest_checkpoint(job.paths.output_dir)
    if newest is None or newest.name != checkpoint_name:
        raise TrainingOrchestrationError(
            f"run record names {checkpoint_name}, newest checkpoint is {newest}"
        )
    _validate_final_checkpoint(checkpoint, expected_step=job.total_steps)
    return "complete"


def execute_job(
    job: ArmJob,
    config: Mapping[str, Any],
    *,
    callbacks: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """The only production seam that enters single-arm model training."""

    return run_arm(
        job.composition,
        job.paths,
        config=config,
        total_steps=job.total_steps,
        seed=job.seed,
        resume=True,
        callbacks=callbacks,
    )


def build_production_trainer(
    safety_policy: UnattendedSafetyPolicy,
) -> Callable[[ArmJob, Mapping[str, Any]], Mapping[str, Any]]:
    """Bind the measured unattended policy to every production arm."""

    def train_one(job: ArmJob, config: Mapping[str, Any]) -> Mapping[str, Any]:
        safety_policy.check(stage="before_arm", arm=job.arm)
        callback = TrainerHealthCallback(policy=safety_policy, arm=job.arm)
        with UnattendedWatchdog(
            policy=safety_policy,
            arm=job.arm,
            output_dir=job.paths.output_dir,
        ):
            return execute_job(job, config, callbacks=(callback,))

    return train_one


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def run_jobs(
    jobs: Sequence[ArmJob],
    *,
    config: Mapping[str, Any],
    train_one: Callable[[ArmJob, Mapping[str, Any]], Mapping[str, Any]],
    summary_path: Path,
    run_records_root: Path,
    provenance: Mapping[str, Any] | None = None,
) -> int:
    """Skip complete arms, resume incomplete ones, and stop on the first failure."""

    if not jobs:
        raise TrainingOrchestrationError("no arms selected")
    attempts: list[dict[str, Any]] = []
    if Path(summary_path).is_file():
        previous = _read_record(summary_path)
        previous_attempts = previous.get("attempts", [])
        if isinstance(previous_attempts, list):
            attempts.extend(
                item for item in previous_attempts if isinstance(item, dict)
            )
        attempts.append({key: value for key, value in previous.items() if key != "attempts"})

    started_at = _utc_now()
    summary: dict[str, Any] = {
        "config_sha256": jobs[0].config_sha256,
        "model_checkpoint": jobs[0].model_checkpoint,
        "seed": jobs[0].seed,
        "total_steps": jobs[0].total_steps,
        "arm_order": [job.arm for job in jobs],
        "arms": {job.arm: {"status": "pending"} for job in jobs},
        "provenance": dict(provenance or {}),
        "attempts": attempts,
        "started_at_utc": started_at,
        "updated_at_utc": started_at,
    }
    atomic_write_json(summary_path, summary)

    for job in jobs:
        state = inspect_run(job)
        if state == "complete":
            record = _read_record(job.paths.output_dir / "run_record.json")
            atomic_write_json(
                Path(run_records_root) / job.arm / "run_record.json", record
            )
            summary["arms"][job.arm] = {
                "status": "skipped_complete",
                "run_record": str(job.paths.output_dir / "run_record.json"),
            }
            summary["updated_at_utc"] = _utc_now()
            atomic_write_json(summary_path, summary)
            continue

        started = _utc_now()
        summary["arms"][job.arm] = {"status": "running", "started_at_utc": started}
        summary["updated_at_utc"] = _utc_now()
        atomic_write_json(summary_path, summary)
        try:
            raw_record = dict(train_one(job, config))
            returned = {
                "arm": raw_record.get("arm"),
                "seed": raw_record.get("seed"),
                "total_steps": raw_record.get("total_steps"),
                "composition": raw_record.get("composition"),
            }
            expected = {
                "arm": job.arm,
                "seed": job.seed,
                "total_steps": job.total_steps,
                "composition": job.composition.summary(),
            }
            if returned != expected:
                raise TrainingOrchestrationError(
                    f"training returned mismatched provenance: {returned!r}"
                )
            newest = find_resumable_checkpoint(job.paths.output_dir)
            if newest is None:
                raise TrainingOrchestrationError(
                    f"{job.arm} returned without writing a checkpoint"
                )
            record = {
                **raw_record,
                "model_checkpoint": job.model_checkpoint,
                "config_sha256": job.config_sha256,
                "started_at_utc": started,
                "finished_at_utc": _utc_now(),
                "latest_checkpoint": Path(newest).name,
            }
            _finite_record_values(record)
            record_path = job.paths.output_dir / "run_record.json"
            atomic_write_json(record_path, record)
            atomic_write_json(
                Path(run_records_root) / job.arm / "run_record.json", record
            )
            if inspect_run(job) != "complete":
                raise TrainingOrchestrationError(f"{job.arm} did not verify complete")
        except (RuntimeError, OSError, ValueError, TypeError, KeyError) as error:
            summary["arms"][job.arm] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "finished_at_utc": _utc_now(),
            }
            summary["updated_at_utc"] = _utc_now()
            summary["finished_at_utc"] = summary["updated_at_utc"]
            atomic_write_json(summary_path, summary)
            return 1

        summary["arms"][job.arm] = {
            "status": "completed",
            "run_record": str(record_path),
            "finished_at_utc": record["finished_at_utc"],
        }
        summary["updated_at_utc"] = _utc_now()
        atomic_write_json(summary_path, summary)
    summary["finished_at_utc"] = _utc_now()
    summary["updated_at_utc"] = summary["finished_at_utc"]
    atomic_write_json(summary_path, summary)
    return 0


def select_arms(
    configured: Sequence[str], requested: Sequence[str] | None
) -> tuple[str, ...]:
    """Validate a subset while retaining the frozen configuration order."""

    configured_order = tuple(str(arm) for arm in configured)
    if requested is None or not requested:
        return configured_order
    requested_names = tuple(str(arm) for arm in requested)
    unknown = sorted(set(requested_names) - set(configured_order))
    if unknown:
        raise TrainingOrchestrationError(f"unknown arm(s): {unknown}")
    if len(set(requested_names)) != len(requested_names):
        raise TrainingOrchestrationError("requested arms contain duplicates")
    return tuple(arm for arm in configured_order if arm in requested_names)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training_rfdetr.yaml")
    )
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--run-records-root", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--health-log", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--pool-tag", default="m13_pool_1x")
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--min-free-gib", type=float, default=50.0)
    parser.add_argument("--max-runtime-hours", type=float, default=16.0)
    parser.add_argument("--max-gpu-temperature-c", type=float, default=85.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    *,
    train_one: Callable[[ArmJob, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> int:
    args = parse_args(argv)
    paths = load_project_paths()
    config = load_training_config(args.config)
    manifest = args.manifest or (paths.splits / "split_manifest.json")
    real_splits = split_real_images(manifest)
    if "test" not in real_splits:
        raise TrainingOrchestrationError(f"{manifest} declares no sealed Test split")

    pool = paths.synthetic / args.pool_tag
    compositions = build_all_arms(
        manifest_path=manifest,
        synthetic_annotations={
            "filtered": pool / "annotations_filtered_1x.json",
            "unfiltered": pool / "annotations_unfiltered_1x.json",
        },
    )
    selected = select_arms(config["arms"], args.arms)
    runs_root = args.runs_root or (paths.data_root / "runs_rfdetr")
    run_records_root = args.run_records_root or (
        paths.data_root / "runs_rfdetr_records"
    )
    summary_path = args.summary or (paths.reports / "rfdetr_orchestration.json")
    jobs = build_jobs(
        config,
        paths,
        compositions,
        runs_root=runs_root,
        pool_tag=args.pool_tag,
        sealed_test_names=real_splits["test"],
        selected_arms=selected,
    )
    report = preflight(
        jobs,
        manifest_path=manifest,
        required_free_gib=args.min_free_gib,
        expected_synthetic_count=3_500,
    )

    if args.dry_run:
        payload = {
            **asdict(report),
            "model_checkpoint": str(config["model"]["checkpoint"]),
            "seed": int(config["run"]["seed"]),
            "total_steps": int(config["run"]["total_steps"]),
            "config_sha256": jobs[0].config_sha256,
            "runs_root": str(runs_root),
            "run_records_root": str(run_records_root),
            "summary": str(summary_path),
            "run_status": {job.arm: inspect_run(job) for job in jobs},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.max_runtime_hours <= 0:
        raise TrainingOrchestrationError("--max-runtime-hours must be positive")
    if args.max_gpu_temperature_c <= 0:
        raise TrainingOrchestrationError("--max-gpu-temperature-c must be positive")

    selected_train = train_one
    if selected_train is None:
        health_log = args.health_log or (paths.reports / "rfdetr_health.jsonl")
        policy = UnattendedSafetyPolicy(
            output_root=runs_root,
            health_log=health_log,
            deadline_utc=datetime.now(UTC)
            + timedelta(hours=float(args.max_runtime_hours)),
            min_free_gib=float(args.min_free_gib),
            max_gpu_temperature_c=float(args.max_gpu_temperature_c),
        )
        policy.check(stage="startup")
        selected_train = build_production_trainer(policy)

    with orchestration_lock(Path(runs_root) / ".rfdetr-orchestration.lock"):
        return run_jobs(
            jobs,
            config=config,
            train_one=selected_train,
            summary_path=summary_path,
            run_records_root=run_records_root,
            provenance={
                "config_path": str(args.config),
                "manifest_path": str(manifest),
                "pool_tag": str(args.pool_tag),
                "real_train_digest": report.real_train_digest,
                "synthetic_counts": dict(report.synthetic_counts),
                "required_free_gib": report.required_free_gib,
                "free_disk_gib_at_preflight": report.free_disk_gib,
                "runs_root": str(runs_root),
                "run_records_root": str(run_records_root),
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
