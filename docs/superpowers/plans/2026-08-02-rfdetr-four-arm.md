# RF-DETR Four-Arm Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train, evaluate, and report the frozen four-arm RF-DETR-Nano replication on the local RTX 4090 without waiting at non-blocking checkpoints during the owner's 16-hour unattended window.

**Architecture:** Keep `run_arm()` as the single-arm training implementation and add a tested orchestration CLI that resolves the frozen data once, validates disk and provenance without touching CUDA in dry-run mode, then executes isolated resumable arms sequentially. Reuse the existing evaluation and benchmark implementations, but make RF-DETR config inheritance and model-specific output locations explicit so no RT-DETRv2 source-of-truth file is overwritten.

**Tech Stack:** Python 3.12, PyTorch 2.13 + CUDA 13.0, Transformers 5.14.1, Hugging Face Trainer, PyYAML, pytest, Ruff, PowerShell 5.1, NVIDIA RTX 4090.

## Global Constraints

- Execute locally on the NVIDIA GeForce RTX 4090; use Colab only if the local GPU becomes unavailable.
- Run arms sequentially in this order: `real_only`, `filtered_syn`, `standard_aug`, `unfiltered_syn`.
- Use seed `1337` and exactly `10,900` optimizer steps for every arm.
- Train and validate only from the frozen manifest; Test is forbidden until post-training evaluation.
- Use the existing M13 1x subsets: `3,500` filtered and `3,500` unfiltered synthetic images.
- Do not change RF-DETR preprocessing, optimizer, schedule, arm composition, thresholds, or hyperparameters after results are observed.
- A 16-hour unattended window permits continuous execution, not a smaller experiment; preserve resumable checkpoints if the window ends.
- Stop only for CUDA OOM, NaN/Inf, corrupt input/checkpoint, insufficient disk, failed provenance, or material GPU contention from another project.
- Never stop or modify another project's process and never recursively delete a production run directory.
- Keep RF-DETR runs, predictions, metrics, reports, and latency output separate from RT-DETRv2 source-of-truth artifacts.
- Commit author and committer must remain `kuotunyu`; no commit may contain a `Co-Authored-By:` trailer.

## File Structure

- Modify `configs/training_rfdetr.yaml`: freeze the approved four-arm scope.
- Modify `tests/test_training_rfdetr_config.py` and `tests/test_training_config.py`: enforce the four-arm order through raw and inheritance-aware loading.
- Modify `scripts/probe_train_speed.py` and `tests/test_probe_train_speed.py`: reject incoherent or non-finite speed probes before production starts.
- Create `scripts/train_arms.py`: dry-run preflight, resumable sequential orchestration, run-record mirroring, and atomic summary updates.
- Create `tests/test_train_arms.py`: all orchestration behavior with injected training and GPU-free fixtures.
- Modify `scripts/smoke_train.py` and create `tests/test_smoke_train.py`: allow the real RF-DETR child config to drive the resume smoke.
- Modify `scripts/eval.py` and `tests/test_eval_driver.py`: resolve inherited RF config, accept local run records, and persist Test predictions under an RF-specific root.
- Modify `scripts/dump_predictions.py` and create `tests/test_dump_predictions.py`: persist Validation predictions from a selectable runs root, processor, and index.
- Modify `scripts/benchmark_latency.py` and `tests/test_benchmark.py`: accept an explicit report path for a same-session fine-tuned RT-versus-RF comparison.
- Generate `reports/rfdetr_train_speed.{md,json}`, `reports/rfdetr_orchestration.json`, `results/rfdetr_predictions_index.json`, `results/rfdetr_detection_metrics.csv`, `reports/rfdetr_detection_main_table.md`, and `reports/rfdetr_speed_baseline_probe.md`.
- Modify `README.md`, `PLAN_PHASE2.md`, `docs/worklog.md`, and `docs/decisions.md` only after tracked RF result sources pass their numerical checks.

---

### Task 1: Freeze the Four-Arm RF-DETR Configuration

**Files:**
- Modify: `tests/test_training_rfdetr_config.py`
- Modify: `tests/test_training_config.py`
- Modify: `configs/training_rfdetr.yaml`

**Interfaces:**
- Consumes: `src.training.config.load_training_config(path)`.
- Produces: `config["arms"] == ["real_only", "filtered_syn", "standard_aug", "unfiltered_syn"]` for the production CLI.

- [ ] **Step 1: Replace the two-arm assertions with the approved execution order**

```python
def test_the_four_arm_scope_and_execution_order_are_explicit() -> None:
    assert _load(RFDETR)["arms"] == [
        "real_only",
        "filtered_syn",
        "standard_aug",
        "unfiltered_syn",
    ]


def test_the_rfdetr_config_resolves_to_the_approved_four_arms() -> None:
    resolved = load_training_config("configs/training_rfdetr.yaml")
    assert resolved["arms"] == [
        "real_only",
        "filtered_syn",
        "standard_aug",
        "unfiltered_syn",
    ]
```

- [ ] **Step 2: Run the focused tests and verify the old two-arm config fails**

Run: `uv run pytest tests/test_training_rfdetr_config.py tests/test_training_config.py -q`

Expected: FAIL because the YAML still resolves to `real_only, filtered_syn` only.

- [ ] **Step 3: Change only the RF child config's scope block**

```yaml
# Scope: approved four-arm architecture replication
arms:
  - "real_only"       # source: fixed
  - "filtered_syn"    # source: fixed
  - "standard_aug"    # source: fixed
  - "unfiltered_syn"  # source: fixed
```

Preserve `extends`, all verified preprocessing values, every optimizer/schedule value, `total_steps: 10900`, and `seed: 1337` unchanged.

- [ ] **Step 4: Run the focused tests and config lint**

Run: `uv run pytest tests/test_training_rfdetr_config.py tests/test_training_config.py -q`

Expected: PASS.

Run: `uv run ruff check tests/test_training_rfdetr_config.py tests/test_training_config.py`

Expected: PASS.

- [ ] **Step 5: Commit the frozen scope**

```powershell
git add -- configs/training_rfdetr.yaml tests/test_training_rfdetr_config.py tests/test_training_config.py
git commit -m "config(m20): freeze four RF-DETR arms"
```

### Task 2: Make the Speed Probe a Hard Safety Gate

**Files:**
- Modify: `tests/test_probe_train_speed.py`
- Modify: `scripts/probe_train_speed.py`

**Interfaces:**
- Consumes: `SpeedProbe` measurements and `run_arm()` records.
- Produces: `validate_probe(probe) -> None` and `validate_finite_run_record(record) -> None`; both raise `SpeedProbeError` on invalid evidence.

- [ ] **Step 1: Add failing tests for invalid slopes and non-finite training evidence**

```python
def test_a_probe_with_a_negative_intercept_is_rejected() -> None:
    with pytest.raises(SpeedProbeError, match="negative fixed overhead"):
        validate_probe(_probe(40, 140, 10.0, 100.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_training_loss_is_rejected(value: float) -> None:
    with pytest.raises(SpeedProbeError, match="train_loss"):
        validate_finite_run_record({"train_loss": value, "eval_metrics": {}})


def test_a_valid_probe_and_finite_record_pass() -> None:
    validate_probe(_probe(40, 140, 60 + 0.25 * 40, 60 + 0.25 * 140))
    validate_finite_run_record({"train_loss": 1.25, "eval_metrics": {"eval_map": 0.1}})
```

- [ ] **Step 2: Run the new tests and verify the missing guards fail**

Run: `uv run pytest tests/test_probe_train_speed.py -q`

Expected: FAIL on missing `SpeedProbeError`, `validate_probe`, and `validate_finite_run_record`.

- [ ] **Step 3: Implement finite-value and slope validation**

```python
class SpeedProbeError(RuntimeError):
    """The measured probe cannot support a production-time projection."""


def validate_finite_run_record(record: Mapping[str, Any]) -> None:
    values = {"train_loss": record.get("train_loss")}
    values.update(dict(record.get("eval_metrics", {})))
    for name, value in values.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SpeedProbeError(f"non-finite {name}: {value}")


def validate_probe(probe: SpeedProbe) -> None:
    if probe.long_seconds <= probe.short_seconds:
        raise SpeedProbeError("long probe did not take longer than short probe")
    if not math.isfinite(probe.seconds_per_step) or probe.seconds_per_step <= 0:
        raise SpeedProbeError("seconds per step must be finite and positive")
    if not math.isfinite(probe.fixed_overhead_seconds):
        raise SpeedProbeError("fixed overhead must be finite")
    if probe.fixed_overhead_seconds < 0:
        raise SpeedProbeError("negative fixed overhead makes the slope unreliable")
```

Capture `record = run_arm(...)` in `timed_run()`, call `validate_finite_run_record(record)`, and call `validate_probe()` before rendering or writing each result. Let `main()` return non-zero by propagating `SpeedProbeError`; do not write a projection from invalid measurements.

- [ ] **Step 4: Run probe tests and static checks**

Run: `uv run pytest tests/test_probe_train_speed.py -q`

Expected: PASS.

Run: `uv run ruff check scripts/probe_train_speed.py tests/test_probe_train_speed.py`

Expected: PASS.

- [ ] **Step 5: Commit the safety gate**

```powershell
git add -- scripts/probe_train_speed.py tests/test_probe_train_speed.py
git commit -m "test(m20): reject invalid speed probes"
```

### Task 3: Add the Resumable Four-Arm Training CLI

**Files:**
- Create: `tests/test_train_arms.py`
- Create: `scripts/train_arms.py`

**Interfaces:**
- Consumes: `load_training_config()`, `build_all_arms()`, `RunPaths`, and `run_arm()`.
- Produces: `ArmJob`, `PreflightReport`, `build_jobs()`, `inspect_run()`, `preflight()`, `run_jobs()`, `execute_job()`, and `main(argv=None, *, train_one=None) -> int`.
- Writes: atomic `reports/rfdetr_orchestration.json` and mirrored `<run-records-root>/<arm>/run_record.json` files.

- [ ] **Step 1: Write failing pure tests for job order, dedicated paths, and Test exclusion**

```python
APPROVED = ("real_only", "filtered_syn", "standard_aug", "unfiltered_syn")


def test_default_jobs_follow_the_approved_order(fake_inputs) -> None:
    jobs = build_jobs(fake_inputs.config, fake_inputs.paths, fake_inputs.compositions)
    assert tuple(job.arm for job in jobs) == APPROVED
    assert all(job.output_dir == fake_inputs.runs_root / job.arm / "seed_1337" for job in jobs)


def test_jobs_never_include_test_names(fake_inputs) -> None:
    jobs = build_jobs(fake_inputs.config, fake_inputs.paths, fake_inputs.compositions)
    sealed = set(fake_inputs.test_names)
    for job in jobs:
        assert sealed.isdisjoint(job.composition.real_train)
        assert sealed.isdisjoint(job.composition.real_val)
```

- [ ] **Step 2: Write failing tests for dry-run, unknown arms, disk, completion, resume, and fail-fast behavior**

```python
def test_dry_run_never_calls_training(fake_inputs) -> None:
    called = []
    code = main(fake_inputs.argv("--dry-run"), train_one=lambda job, config: called.append(job))
    assert code == 0
    assert called == []


def test_an_unknown_arm_is_rejected_before_training(fake_inputs) -> None:
    with pytest.raises(TrainingOrchestrationError, match="unknown arm"):
        main(fake_inputs.argv("--arms", "invented"), train_one=pytest.fail)


def test_completed_runs_are_skipped_but_incomplete_runs_resume(fake_inputs) -> None:
    calls = []
    fake_inputs.write_complete("real_only")
    fake_inputs.write_checkpoint("filtered_syn", 500)
    assert main(
        fake_inputs.argv(),
        train_one=lambda job, config: calls.append(job.arm) or job.record(),
    ) == 0
    assert calls == ["filtered_syn", "standard_aug", "unfiltered_syn"]


def test_first_real_failure_stops_later_arms(fake_inputs) -> None:
    calls = []
    def fail_filtered(job, config):
        calls.append(job.arm)
        if job.arm == "filtered_syn":
            raise RuntimeError("CUDA out of memory")
        return job.record()
    assert main(fake_inputs.argv(), train_one=fail_filtered) == 1
    assert calls == ["real_only", "filtered_syn"]
```

Also assert that preflight fails below `--min-free-gib`, every required file is named in the error, a non-matching completed record is treated as an unsafe collision, and the JSON summary is parseable after each arm rather than only at process exit.

- [ ] **Step 3: Run the CLI tests and verify they fail because the module is absent**

Run: `uv run pytest tests/test_train_arms.py -q`

Expected: FAIL with `ModuleNotFoundError: scripts.train_arms`.

- [ ] **Step 4: Implement the orchestration dataclasses and pure validation helpers**

```python
@dataclass(frozen=True)
class ArmJob:
    arm: str
    seed: int
    total_steps: int
    composition: ArmComposition
    paths: RunPaths
    config_sha256: str


@dataclass(frozen=True)
class PreflightReport:
    arms: tuple[str, ...]
    free_disk_gib: float
    required_free_gib: float
    real_train_digest: str
    synthetic_counts: Mapping[str, int]


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
```

Use `hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())` for the resolved-config digest. Require the manifest, real COCO, real image directory, both M13 annotation files, M13 image directory, output parent, and at least `--min-free-gib 50` by default. Verify synthetic counts are exactly `3,500` for both synthetic arms and that all four real-train digests match.

- [ ] **Step 5: Implement run classification and sequential execution**

`inspect_run(job)` returns only `"absent"`, `"resumable"`, `"complete"`, or raises `TrainingOrchestrationError` for a conflicting record. Completion requires exact arm, seed, total steps, model checkpoint, resolved-config SHA-256, composition summary, a finite loss, and a readable `checkpoint-*` directory. A checkpoint without a complete matching record is resumable.

Define `execute_job(job, config)` to call `run_arm(job.composition, job.paths, config=config, total_steps=job.total_steps, seed=job.seed, resume=True)`. `main(..., train_one=None)` selects `execute_job`; tests inject a callable with the same `(ArmJob, config) -> record` signature. Augment its record with `model_checkpoint`, `config_sha256`, UTC start/end timestamps, and the newest checkpoint name; atomically rewrite the run record, copy it to `<run-records-root>/<arm>/run_record.json`, then atomically update the orchestration summary. Catch a genuine training exception only to record `failed` and return `1`; never start the next arm.

- [ ] **Step 6: Implement CLI arguments and the dry-run output**

```text
--config configs/training_rfdetr.yaml
--runs-root D:/sdg-data/02-safesynth/runs_rfdetr
--run-records-root D:/sdg-data/02-safesynth/runs_rfdetr_records
--summary reports/rfdetr_orchestration.json
--pool-tag m13_pool_1x
--arms real_only filtered_syn standard_aug unfiltered_syn
--min-free-gib 50
--dry-run
```

An omitted `--arms` uses the config order. An explicit subset preserves config order, not command-line order. Dry-run prints the resolved model, seed, steps, arm counts, digests, disk evidence, and output paths and exits before importing Torch model classes or calling `run_arm()`.

- [ ] **Step 7: Run focused tests and lint**

Run: `uv run pytest tests/test_train_arms.py tests/test_training_arms.py -q`

Expected: PASS.

Run: `uv run ruff check scripts/train_arms.py tests/test_train_arms.py`

Expected: PASS.

- [ ] **Step 8: Commit the production CLI**

```powershell
git add -- scripts/train_arms.py tests/test_train_arms.py
git commit -m "feat(m20): orchestrate resumable RF-DETR arms"
```

### Task 4: Exercise RF-DETR Through the Existing Resume Smoke

**Files:**
- Create: `tests/test_smoke_train.py`
- Modify: `scripts/smoke_train.py`

**Interfaces:**
- Consumes: `--config` and `load_training_config()`.
- Produces: the same cold-start/warm-resume smoke path for either RT-DETRv2 or RF-DETR.

- [ ] **Step 1: Add failing argument/config tests without loading a model**

```python
def test_smoke_accepts_an_inherited_training_config() -> None:
    args = parse_args(["--config", "configs/training_rfdetr.yaml", "--arm", "real_only"])
    assert args.config.as_posix().endswith("configs/training_rfdetr.yaml")


def test_smoke_output_is_namespaced_by_config(tmp_path) -> None:
    path = smoke_output_dir(tmp_path, Path("configs/training_rfdetr.yaml"), "real_only", 1337)
    assert path == tmp_path / "smoke" / "training_rfdetr" / "real_only_seed_1337"
```

- [ ] **Step 2: Run and observe the missing interfaces**

Run: `uv run pytest tests/test_smoke_train.py -q`

Expected: FAIL because `parse_args(argv)` and `smoke_output_dir()` do not exist.

- [ ] **Step 3: Load the child config through the inheritance-aware loader**

Change `parse_args()` to accept `argv`, add `--config` as a `Path`, replace `yaml.safe_load` with `load_training_config(args.config)`, and namespace the smoke directory by `args.config.stem`. Preserve the 16-image evaluation, cold start, checkpoint assertion, warm resume, and cleanup behavior.

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_smoke_train.py tests/test_training_notebook.py -q`

Expected: PASS.

Run: `uv run ruff check scripts/smoke_train.py tests/test_smoke_train.py`

Expected: PASS.

- [ ] **Step 5: Commit the RF smoke path**

```powershell
git add -- scripts/smoke_train.py tests/test_smoke_train.py
git commit -m "feat(m20): smoke inherited detector configs"
```

### Task 5: Make Evaluation Consume RF Outputs Without Overwriting RT Results

**Files:**
- Modify: `tests/test_eval_driver.py`
- Modify: `scripts/eval.py`
- Create: `tests/test_dump_predictions.py`
- Modify: `scripts/dump_predictions.py`

**Interfaces:**
- Consumes: local mirrored run records, inherited RF config, RF run root, and best Validation checkpoints.
- Produces: RF-specific Val/Test predictions, prediction index, metrics CSV, and Markdown report.

- [ ] **Step 1: Add failing tests for inherited config and RF-specific prediction persistence**

```python
def test_eval_uses_the_inheritance_aware_training_config() -> None:
    resolved = load_driver_training_config(PROJECT_ROOT / "configs" / "training_rfdetr.yaml")
    assert resolved["run"]["per_device_eval_batch_size"] == 8
    assert resolved["model"]["checkpoint"] == "Roboflow/rf-detr-nano"


def test_evaluate_arm_persists_original_coordinate_detections(tmp_path, fake_eval_inputs) -> None:
    destination = tmp_path / "real_only_test_seed1337.json"
    evaluate_arm(**fake_eval_inputs, predictions_path=destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload and {"image_id", "category_id", "bbox", "score"} <= payload[0].keys()
```

- [ ] **Step 2: Add failing tests for selectable prediction roots and processor identity**

```python
def test_dump_predictions_accepts_rf_roots() -> None:
    args = parse_args([
        "--runs-root", "D:/runs_rfdetr",
        "--processor", "Roboflow/rf-detr-nano",
        "--index", "results/rfdetr_predictions_index.json",
    ])
    assert args.runs_root == Path("D:/runs_rfdetr")
    assert args.processor == "Roboflow/rf-detr-nano"
    assert args.index.name == "rfdetr_predictions_index.json"
```

- [ ] **Step 3: Run the focused tests and verify the current hard-coded behavior fails**

Run: `uv run pytest tests/test_eval_driver.py tests/test_dump_predictions.py -q`

Expected: FAIL on the missing loader wrapper, prediction path, and RF CLI options.

- [ ] **Step 4: Replace plain YAML loading and add local record naming**

Import `src.training.config.load_training_config` under the name `load_resolved_training_config`. Define `load_driver_training_config(path)` as `load_resolved_training_config(path)`. Rename `--colab-results` to `--run-records` while retaining `--colab-results` as a hidden compatibility alias; all messages must say `run records`, not imply the RF runs came from Colab.

- [ ] **Step 5: Persist Test predictions atomically from the same inference used for metrics**

Add `--predictions-root` and pass `<root>/<arm>_test_seed<seed>.json` into `evaluate_arm()`. Immediately after `detections_for_evaluation()`, write compact JSON to a temporary sibling and replace the destination. Record checkpoint, split, image count, path, coordinate convention, and threshold `0.0` in the RF prediction index. This reuses the inference already required by evaluation and never performs Test inference twice.

- [ ] **Step 6: Parameterize Validation prediction export**

Add `--runs-root`, `--processor`, and `--index` to `dump_predictions.py`; use them instead of `paths.runs`, `PROCESSOR_ID`, and `INDEX_PATH`. Keep per-image original-coordinate mapping and best-checkpoint resolution unchanged. Production will call this script with `--splits val` only because Test is persisted by `eval.py`.

- [ ] **Step 7: Run focused and related tests**

Run: `uv run pytest tests/test_eval_driver.py tests/test_dump_predictions.py tests/test_select_operating_point.py -q`

Expected: PASS.

Run: `uv run ruff check scripts/eval.py scripts/dump_predictions.py tests/test_eval_driver.py tests/test_dump_predictions.py`

Expected: PASS.

- [ ] **Step 8: Commit evaluation isolation**

```powershell
git add -- scripts/eval.py scripts/dump_predictions.py tests/test_eval_driver.py tests/test_dump_predictions.py
git commit -m "feat(m20): isolate RF-DETR evaluation artifacts"
```

### Task 6: Parameterize the Fine-Tuned Latency Report

**Files:**
- Modify: `tests/test_benchmark.py`
- Modify: `scripts/benchmark_latency.py`

**Interfaces:**
- Consumes: fine-tuned `rtdetrv2_r18` and `rf_detr_nano` checkpoint directories.
- Produces: an explicit `--report` path while retaining `reports/speed_baseline_probe.md` as the default.

- [ ] **Step 1: Add a failing parser/output test**

```python
def test_latency_report_path_can_be_isolated(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "rfdetr_speed.md"
    args = parse_args(["--models", "rf_detr_nano", "--report", str(destination)])
    assert args.report == destination
```

Refactor `parse_args(argv=None)` so this test cannot read the pytest process arguments.

- [ ] **Step 2: Run the parser test and verify `--report` is absent**

Run: `uv run pytest tests/test_benchmark.py -q -k "report_path_can_be_isolated"`

Expected: FAIL.

- [ ] **Step 3: Add `--report` and route the final write through it**

```python
parser.add_argument(
    "--report",
    type=Path,
    default=PROJECT_ROOT / "reports" / REPORT_NAME,
)
```

Create `args.report.parent`, write Markdown there, and print that exact path. Do not change model specs, clock sampling, contention checks, iteration counts, dtype, or license evidence.

- [ ] **Step 4: Run the entire benchmark test module and lint**

Run: `uv run pytest tests/test_benchmark.py -q`

Expected: PASS.

Run: `uv run ruff check scripts/benchmark_latency.py tests/test_benchmark.py`

Expected: PASS.

- [ ] **Step 5: Commit report isolation**

```powershell
git add -- scripts/benchmark_latency.py tests/test_benchmark.py
git commit -m "feat(m20): isolate RF latency report"
```

### Task 7: Complete CPU Verification and Production Preflight

**Files:**
- Modify only if a verification failure identifies a real defect in Tasks 1-6.

**Interfaces:**
- Consumes: all implementation commits.
- Produces: a green CPU gate and a machine-readable dry-run before CUDA allocation.

- [ ] **Step 1: Run the focused M20 suite**

Run: `uv run pytest tests/test_training_rfdetr_config.py tests/test_training_config.py tests/test_probe_train_speed.py tests/test_train_arms.py tests/test_smoke_train.py tests/test_eval_driver.py tests/test_dump_predictions.py tests/test_benchmark.py -q`

Expected: PASS.

- [ ] **Step 2: Run Ruff and the full suite**

Run: `uv run ruff check .`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS with only the project's declared skips.

- [ ] **Step 3: Verify locks, README, license, and contributor invariant**

Run: `uv lock --check`

Expected: PASS.

Run: `uv run python scripts/verify_readme.py`

Expected: PASS.

Run: `uv run python scripts/license_scan.py`

Expected: PASS.

Run: `git log --format="%an%n%cn" | Sort-Object -Unique`

Expected: the only non-empty line is `kuotunyu`.

Run: `git log --format="%B" | Select-String -Pattern "Co-Authored-By:"`

Expected: no output.

- [ ] **Step 4: Inspect live GPU, disk, and competing processes**

Run: `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,pstate --format=csv,noheader`

Expected: RTX 4090, enough free VRAM for RF-DETR, and no sustained unrelated load.

Run: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`

Expected: no material competing compute process. Do not terminate anything found.

Run: `Get-PSDrive D | Select-Object Name,Free,Used`

Expected: more than 50 GiB free.

- [ ] **Step 5: Run the no-CUDA preflight**

Run: `uv run python -m scripts.train_arms --config configs/training_rfdetr.yaml --runs-root D:/sdg-data/02-safesynth/runs_rfdetr --run-records-root D:/sdg-data/02-safesynth/runs_rfdetr_records --summary reports/rfdetr_orchestration.json --dry-run`

Expected: four arms in approved order, seed 1337, 10,900 steps each, synthetic counts 0/3,500/0/3,500, one shared real-train digest, sufficient disk, and no model/GPU allocation.

- [ ] **Step 6: Commit any verification-driven correction separately**

If no correction was needed, do not create an empty commit. If one was needed, stage only its exact files and use `git commit -m "fix(m20): correct production preflight"` after re-running Steps 1-5.

### Task 8: Run GPU Smoke, Measure Speed, and Continue Automatically

**Files:**
- Generate: `reports/train_smoke.json`
- Generate: `reports/rfdetr_train_speed.md`
- Generate: `reports/rfdetr_train_speed.json`

**Interfaces:**
- Consumes: the same RF child config and `run_arm()` path as production.
- Produces: verified cold start/resume and a valid 40/140-step slope.

- [ ] **Step 1: Recheck contention immediately before CUDA allocation**

Run the two `nvidia-smi` commands from Task 7 Step 4. If another project materially owns the GPU, leave all processes untouched and defer only the CUDA steps.

- [ ] **Step 2: Run the RF cold/warm resume smoke with 16 Validation images**

Run: `uv run python -m scripts.smoke_train --config configs/training_rfdetr.yaml --arm real_only --steps 2 --resume-steps 4 --val-images 16`

Expected: a checkpoint after the cold start, a non-null `resumed_from` on the warm start, finite loss/metrics, and `reports/train_smoke.json` with `resume_verified: true`.

- [ ] **Step 3: Run the measured RF-only speed probe**

Run: `uv run python -m scripts.probe_train_speed --configs configs/training_rfdetr.yaml --arm real_only --short 40 --long 140 --seed 1337 --val-images 16 --out reports/rfdetr_train_speed.md`

Expected: positive seconds/step, long elapsed > short elapsed, non-negative fixed overhead, finite losses, and both Markdown and JSON reports.

- [ ] **Step 4: Record the measured ETA and continue without waiting**

Compute `4 * production_hours` from `reports/rfdetr_train_speed.json`, report it in commentary, and proceed to Task 9 unless the probe safety gate failed. The owner has already authorized this continuation.

### Task 9: Train All Four RF-DETR Arms Sequentially

**Files:**
- Generate: `D:/sdg-data/02-safesynth/runs_rfdetr/<arm>/seed_1337/`
- Generate: `D:/sdg-data/02-safesynth/runs_rfdetr_records/<arm>/run_record.json`
- Update: `reports/rfdetr_orchestration.json`

**Interfaces:**
- Consumes: valid Task 8 evidence.
- Produces: four isolated resumable 10,900-step runs with exact provenance.

- [ ] **Step 1: Start production orchestration**

Run: `uv run python -m scripts.train_arms --config configs/training_rfdetr.yaml --runs-root D:/sdg-data/02-safesynth/runs_rfdetr --run-records-root D:/sdg-data/02-safesynth/runs_rfdetr_records --summary reports/rfdetr_orchestration.json`

Expected order: `real_only`, `filtered_syn`, `standard_aug`, `unfiltered_syn`.

- [ ] **Step 2: Monitor bounded safety evidence without altering the experiment**

At arm boundaries and roughly every 30 minutes, inspect `nvidia-smi`, D: free space, latest checkpoint timestamps, summary status, and Trainer loss output. Ordinary warnings do not stop the run. OOM, NaN/Inf, corrupt checkpoint/input, less than 50 GiB free, provenance mismatch, or material unrelated GPU contention stops the current orchestration safely.

- [ ] **Step 3: Resume after recoverable interruption**

Re-run the exact Step 1 command. Expected: matching completed arms skip; an incomplete arm resumes from its newest checkpoint; subsequent arms remain untouched until it completes. Never delete a production run directory.

- [ ] **Step 4: Verify all completion records**

Run: `uv run python -m scripts.train_arms --config configs/training_rfdetr.yaml --runs-root D:/sdg-data/02-safesynth/runs_rfdetr --run-records-root D:/sdg-data/02-safesynth/runs_rfdetr_records --summary reports/rfdetr_orchestration.json --dry-run`

Expected: all four status values are `complete`, step counts are 10,900, config/model/real-data digests match, and every newest/best checkpoint is readable.

### Task 10: Persist Predictions, Evaluate, Benchmark, and Freeze M20

**Files:**
- Generate: `D:/sdg-data/02-safesynth/runs_rfdetr_predictions/*.json`
- Generate: `results/rfdetr_predictions_index.json`
- Generate: `results/rfdetr_detection_metrics.csv`
- Generate: `reports/rfdetr_detection_main_table.md`
- Generate: `reports/rfdetr_speed_baseline_probe.md`
- Modify: `README.md`
- Modify: `PLAN_PHASE2.md`
- Modify: `docs/worklog.md`
- Modify: `docs/decisions.md`

**Interfaces:**
- Consumes: four successful run records and best Validation checkpoints.
- Produces: frozen RF replication evidence and release-facing documentation.

- [ ] **Step 1: Persist Validation predictions once**

Run: `uv run python -m scripts.dump_predictions --splits val --arms real_only filtered_syn standard_aug unfiltered_syn --seed 1337 --runs-root D:/sdg-data/02-safesynth/runs_rfdetr --processor Roboflow/rf-detr-nano --out-root D:/sdg-data/02-safesynth/runs_rfdetr_predictions --index results/rfdetr_predictions_index.json`

Expected: four Val JSON files in original per-image coordinates and four index entries.

- [ ] **Step 2: Evaluate frozen Test, persist its predictions, and bootstrap 1,000 times**

Run: `uv run python -m scripts.eval --runs-root D:/sdg-data/02-safesynth/runs_rfdetr --run-records D:/sdg-data/02-safesynth/runs_rfdetr_records --training-config configs/training_rfdetr.yaml --predictions-root D:/sdg-data/02-safesynth/runs_rfdetr_predictions --predictions-index results/rfdetr_predictions_index.json --metrics-csv results/rfdetr_detection_metrics.csv --report reports/rfdetr_detection_main_table.md --bootstrap-resamples 1000`

Expected: exit 0; four Test prediction files; metrics for all four arms; 1,000-resample intervals; EVAL-14 leak evidence; no write to `results/detection_metrics.csv` or `reports/detection_main_table.md`.

- [ ] **Step 3: Resolve both fine-tuned real-only checkpoints**

Use `scripts.eval.resolve_checkpoint()` evidence or the respective `trainer_state.json` files to identify the best RT-DETRv2 real-only checkpoint under `D:/sdg-data/02-safesynth/runs/real_only/seed_1337` and RF-DETR real-only checkpoint under `D:/sdg-data/02-safesynth/runs_rfdetr/real_only/seed_1337`. Do not benchmark an in-memory final model or highest-step fallback without an explicit report warning.

- [ ] **Step 4: Benchmark both fine-tuned architectures in the same session**

Run:

```powershell
$rtCheckpoint = uv run python -c "from pathlib import Path; from scripts.eval import resolve_checkpoint; print(resolve_checkpoint(Path('D:/sdg-data/02-safesynth/runs/real_only'), Path('D:/sdg-data/02-safesynth/runs/real_only/seed_1337')).path)"
$rfCheckpoint = uv run python -c "from pathlib import Path; from scripts.eval import resolve_checkpoint; print(resolve_checkpoint(Path('D:/sdg-data/02-safesynth/runs_rfdetr/real_only'), Path('D:/sdg-data/02-safesynth/runs_rfdetr/real_only/seed_1337')).path)"
uv run python -m scripts.benchmark_latency --models rtdetrv2_r18 rf_detr_nano --weights "rtdetrv2_r18=$rtCheckpoint" "rf_detr_nano=$rfCheckpoint" --report reports/rfdetr_speed_baseline_probe.md
```

Expected: both rows marked fine-tuned, contention PASS, SM-clock spread PASS, and no provisional-pretrained banner. If clock spread fails, preserve the failed report and defer publication until a same-clock rerun; do not invent a corrected value.

- [ ] **Step 5: Update tracked claims only from generated sources**

Add an RF-DETR replication subsection to README that names all four arms, seed, step budget, AP_small, mAP, bare-head recall, exposure confound, 95% intervals, latency, H4 gate failure, and whether the synthetic result replicated. Update M20 status/evidence in `PLAN_PHASE2.md`, append commands and artifact paths to `docs/worklog.md`, and add an ADR only if the RF result changes a project conclusion. Do not hand-copy unverified display values.

- [ ] **Step 6: Verify numerical aggregation and the entire repository**

Run: `uv run python scripts/verify_readme.py`

Expected: PASS against the new tracked RF result source.

Run: `uv run python scripts/license_scan.py`

Expected: PASS.

Run: `uv run ruff check .`

Expected: PASS.

Run: `uv run pytest -q`

Expected: PASS with declared skips.

Run: `git diff --check`

Expected: PASS.

- [ ] **Step 7: Commit the frozen RF replication**

```powershell
git add -- results/rfdetr_predictions_index.json results/rfdetr_detection_metrics.csv reports/rfdetr_train_speed.md reports/rfdetr_train_speed.json reports/rfdetr_orchestration.json reports/rfdetr_detection_main_table.md reports/rfdetr_speed_baseline_probe.md README.md PLAN_PHASE2.md docs/worklog.md docs/decisions.md
git commit -m "results(m20): freeze RF-DETR replication"
```

Stage only files that exist and were intentionally generated. Prediction JSONs and model checkpoints remain on D: and are not committed.

### Task 11: Audit the Boundary to the Separate Release Subproject

**Files:**
- Inspect and modify only the release files required by `docs/release_spec.md` and the repository's existing publication checks.

**Interfaces:**
- Consumes: frozen M20 result commit.
- Produces: exact repository-size evidence and a bounded input set for the separate release design/plan; this RF experiment does not rewrite history or publish externally.

- [ ] **Step 1: Re-audit repository size and tracked artifacts**

Run: `git count-objects -vH`

Run: `git ls-files | ForEach-Object { Get-Item -LiteralPath $_ -ErrorAction SilentlyContinue } | Sort-Object Length -Descending | Select-Object -First 30 FullName,Length`

Run: `uv run python scripts/plan_repo_slimming.py`

Expected: a non-destructive slimming plan that names tracked history targets without changing them.

Run: `uv run python scripts/audit_phase1_handoff.py`

Expected: a machine-readable audit; any post-M20 failure is copied exactly into the release input notes rather than silently waived.

- [ ] **Step 2: Re-run existing publication-facing local checks**

Run: `uv run python scripts/verify_readme.py`

Expected: PASS.

Run: `uv run python scripts/license_scan.py`

Expected: PASS.

Run: `git status --short`

Expected: clean after the M20 result commit.

- [ ] **Step 3: Start the release subproject without performing destructive or external actions**

Use the output of Steps 1-2 to create a separate release design and implementation plan covering repository history slimming, GitHub creation/push, model and dataset cards, and Hugging Face upload. Do not run `git filter-repo`, delete repository history, create/push a GitHub repository, publish a Hugging Face dataset/model, or expose credentials under this RF plan. The separate plan must preserve the GitHub invariant that the Contributors page lists only `kuotunyu` and must present exact owner actions for credentials and repository creation.

## Self-Review

- Spec coverage: Tasks 1-4 cover frozen config, preflight, resume, no-Test training, and stop-on-first-failure; Tasks 5-6 cover isolated predictions/evaluation/latency; Tasks 7-10 cover CPU/GPU gates, all four runs, bootstrap, reports, and documentation; Task 11 continues safely toward publication.
- Placeholder scan: the plan contains no undefined implementation marker; commands, paths, signatures, failure conditions, and expected outcomes are explicit.
- Type consistency: `ArmJob`, `PreflightReport`, `validate_probe`, `validate_finite_run_record`, `build_jobs`, `inspect_run`, `execute_job`, and `main(argv=None, *, train_one=None)` retain the same names and roles throughout all dependent tasks.
- Execution choice: the owner already selected the complete four-arm version and authorized continuous inline execution, so use `superpowers:executing-plans` rather than pausing to request the standard handoff choice.
