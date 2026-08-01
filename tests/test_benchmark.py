"""Tests for the DEMO-03 latency harness.

K-18 is why this file exists; K-19 is why it was rewritten. The first version was
green, ruff-clean and at 100% branch coverage, and eight one-token mutations of
the module survived it. Coverage was never the problem: coverage.py does not
count a ternary or an `or`-fallback as a branch, so four completely unexecuted
default paths - the only paths production takes - were reported as covered.

Three rules follow, and every test here obeys them.

1. ORDER IS A PROPERTY, NOT A COUNT. `sync()` called twice is not the same as
   `sync()` called twice in the right place. The fakes therefore append to a
   shared event log, and the assertions are on the SEQUENCE.
2. THE FAKE MUST HAVE THE SEMANTICS THAT MAKE THE BUG VISIBLE. The fake
   `synchronize` costs time on the fake clock, because on a real GPU that is
   where the time is; the fake `reset_peak_memory_stats` sets the peak back to
   what is currently allocated, because that is what makes a late reset report
   zero. A misplaced call then changes a NUMBER the test asserts.
3. A DEFAULT ARGUMENT NO TEST EXERCISES IS UNTESTED CODE, whatever coverage
   says. Every `cuda_api=None` default is executed here on purpose.

Nothing in this file touches a real model or a GPU; it runs on CPU in seconds.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import scripts.benchmark_latency as bl
import scripts.check_forbidden_licences as checker
from scripts.benchmark_latency import (
    ResolutionProbe,
    build_end_to_end_callable,
    build_model_only_callable,
    dispatch_bound_verdict,
    fetch_licence_evidence,
    format_forbidden_scan,
    landing_check,
    measure_at_resolution,
    processor_native_size,
    read_score_threshold,
    render_report,
    resolution_sensitivity_note,
    select_probe_image,
)
from src.data.paths import PROJECT_ROOT
from src.evaluation import benchmark as bench

EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"

# The AGPL-3.0 package ADR-005 forbids, spelled BACKWARDS, so this file cannot be
# mistaken for an import of it. The scanner itself is tested in
# tests/test_forbidden_licences.py; what is tested here is that this script calls
# that scanner rather than carrying a second copy or a stored verdict.
FORBIDDEN_PACKAGE = "scitylartlu"[::-1]


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class RecordingClock:
    """A deterministic clock that LOGS every read.

    Logging the reads is what makes `sync()` placement observable: a synchronize
    between the two reads of a timed region is inside the measurement, and one
    after the second read is not.
    """

    def __init__(self, log: list[str]) -> None:
        self.now = 0.0
        self.log = log

    def __call__(self) -> float:
        self.log.append("clock")
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeClock(RecordingClock):
    """A clock for tests that do not care about the event order."""

    def __init__(self) -> None:
        super().__init__([])


class ClockDrivenWorkload:
    """A fake 'model' whose cost depends on whether it is still warming up."""

    def __init__(self, clock, *, warmup_iterations, warmup_cost_s, timed_costs_s, log=None):
        self.clock = clock
        self.warmup_iterations = warmup_iterations
        self.warmup_cost_s = warmup_cost_s
        self.timed_costs_s = list(timed_costs_s)
        self.log = log
        self.calls = 0

    def __call__(self):
        index = self.calls
        self.calls += 1
        if self.log is not None:
            self.log.append("call")
        if index < self.warmup_iterations:
            self.clock.advance(self.warmup_cost_s)
        else:
            self.clock.advance(self.timed_costs_s[index - self.warmup_iterations])
        return index


class FakeCuda:
    """Duck-typed stand-in for `torch.cuda` that models what a counter cannot see.

    * Every call is appended to a shared event log, so ORDER is assertable.
    * `synchronize()` charges the clock, because on a real GPU it is where the
      queued kernels are actually waited on. A synchronize outside the timed
      region therefore removes real time from the measurement, and the resulting
      duration is numerically different - not merely differently ordered.
    * `reset_peak_memory_stats()` sets the peak back to what is CURRENTLY
      allocated, exactly like the real one. A reset issued after the timed region
      therefore reports a peak of roughly zero.
    """

    def __init__(self, *, available=True, clock=None, sync_cost_s=0.0, log=None):
        self._available = available
        self._clock = clock
        self._sync_cost_s = sync_cost_s
        self.events: list[str] = [] if log is None else log
        self.current_bytes = 0
        self.peak_bytes = 0

    # -- the four calls the harness makes -----------------------------------
    def is_available(self) -> bool:
        self.events.append("is_available")
        return self._available

    def synchronize(self) -> None:
        self.events.append("synchronize")
        if self._clock is not None:
            self._clock.advance(self._sync_cost_s)

    def reset_peak_memory_stats(self) -> None:
        self.events.append("reset_peak_memory_stats")
        self.peak_bytes = self.current_bytes

    def max_memory_allocated(self) -> int:
        self.events.append("max_memory_allocated")
        return self.peak_bytes

    # -- what a workload does to the device ---------------------------------
    def allocate(self, nbytes: int) -> None:
        self.current_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def free(self, nbytes: int) -> None:
        self.current_bytes -= nbytes

    @property
    def syncs(self) -> int:
        return self.events.count("synchronize")

    @property
    def resets(self) -> int:
        return self.events.count("reset_peak_memory_stats")


class AllocatingWorkload:
    """A fake forward pass that allocates activations and frees them again."""

    def __init__(self, clock, cuda, *, cost_s, activation_bytes, log=None):
        self.clock = clock
        self.cuda = cuda
        self.cost_s = cost_s
        self.activation_bytes = activation_bytes
        self.log = log

    def __call__(self):
        if self.log is not None:
            self.log.append("call")
        self.cuda.allocate(self.activation_bytes)
        self.clock.advance(self.cost_s)
        self.cuda.free(self.activation_bytes)


def make_settings(**overrides) -> bench.BenchmarkSettings:
    """Test fixture settings. Deliberately NOT the project config values."""

    base = {
        "warmup_iterations": 3,
        "timed_iterations": 5,
        "report_statistic": "median",
        "also_report_p95": True,
        "separate_model_and_e2e": True,
        "batch_size": 1,
        "dtype": "float16",
        "input_size": 64,
        "max_p95_to_statistic_ratio": 1.5,
    }
    base.update(overrides)
    return bench.BenchmarkSettings(**base)


def make_result(**overrides) -> bench.LatencyResult:
    base = {
        "label": "fake",
        "batch_size": 1,
        "input_size": 64,
        "dtype": "float16",
        "device": "cuda",
        "warmup_iterations": 3,
        "timed_iterations": 5,
        "statistic": "median",
        "latency_ms": 10.0,
        "p95_ms": 12.0,
        "fps": 100.0,
    }
    base.update(overrides)
    return bench.LatencyResult(**base)


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------


def test_settings_are_read_from_the_project_config_and_nothing_else() -> None:
    """The harness must not carry a private copy of any benchmark number."""

    with EVALUATION_CONFIG.open("r", encoding="utf-8") as stream:
        block = yaml.safe_load(stream)["benchmark"]
    settings = bench.load_benchmark_settings()

    assert settings.warmup_iterations == block["warmup_iterations"]
    assert settings.timed_iterations == block["timed_iterations"]
    assert settings.report_statistic == block["report_statistic"]
    assert settings.also_report_p95 == block["also_report_p95"]
    assert settings.separate_model_and_e2e == block["separate_model_and_e2e"]
    assert settings.batch_size == block["batch_size"]
    assert settings.dtype == block["dtype"]
    assert settings.input_size == block["input_size"]
    assert settings.max_p95_to_statistic_ratio == block["max_p95_to_statistic_ratio"]
    # Every key the dataclass needs must be declared, or a missing one degrades
    # from a named config error into an AttributeError somewhere else.
    assert set(bench.BENCHMARK_KEYS) == {
        field.name for field in dataclasses.fields(bench.BenchmarkSettings)
    }


def test_the_contention_threshold_is_config_not_code() -> None:
    """A1: the publish/repeat acceptance criterion is a number, so it lives in yaml."""

    source = (PROJECT_ROOT / "scripts" / "benchmark_latency.py").read_text(encoding="utf-8")
    with EVALUATION_CONFIG.open("r", encoding="utf-8") as stream:
        configured = yaml.safe_load(stream)["benchmark"]["max_p95_to_statistic_ratio"]
    assert str(configured) not in source, (
        "the contention threshold must not be spelled out in the script; it is read "
        "from benchmark.max_p95_to_statistic_ratio"
    )


def test_load_benchmark_settings_rejects_a_config_without_a_benchmark_block(tmp_path) -> None:
    path = tmp_path / "evaluation.yaml"
    path.write_text("metrics:\n  bootstrap_ci: 0.95\n", encoding="utf-8", newline="\n")
    with pytest.raises(bench.BenchmarkConfigError, match="No `benchmark` block"):
        bench.load_benchmark_settings(path)


def test_an_empty_benchmark_block_is_a_typed_error_not_a_typeerror(tmp_path) -> None:
    """A3: `benchmark:` with nothing under it parses as None."""

    path = tmp_path / "evaluation.yaml"
    path.write_text("benchmark:\n", encoding="utf-8", newline="\n")
    with pytest.raises(bench.BenchmarkConfigError, match="must be a mapping, got NoneType"):
        bench.load_benchmark_settings(path)


def test_a_missing_benchmark_key_names_the_key_and_the_file(tmp_path) -> None:
    """A3: a bare KeyError tells the reader neither which key nor which file."""

    with EVALUATION_CONFIG.open("r", encoding="utf-8") as stream:
        block = yaml.safe_load(stream)["benchmark"]
    del block["input_size"]
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump({"benchmark": block}), encoding="utf-8", newline="\n")

    with pytest.raises(bench.BenchmarkConfigError) as caught:
        bench.load_benchmark_settings(path)
    assert "input_size" in str(caught.value)
    assert str(path) in str(caught.value)


def test_an_unusable_benchmark_value_is_a_typed_error(tmp_path) -> None:
    """A3: `int("many")` raising a raw ValueError names nothing useful."""

    with EVALUATION_CONFIG.open("r", encoding="utf-8") as stream:
        block = yaml.safe_load(stream)["benchmark"]
    block["timed_iterations"] = "many"
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump({"benchmark": block}), encoding="utf-8", newline="\n")

    with pytest.raises(bench.BenchmarkConfigError, match="cannot use"):
        bench.load_benchmark_settings(path)


def test_a_validation_failure_keeps_its_own_message_when_loaded_from_a_file(tmp_path) -> None:
    """A3: the wrapper must not bury the specific complaint under a generic one."""

    with EVALUATION_CONFIG.open("r", encoding="utf-8") as stream:
        block = yaml.safe_load(stream)["benchmark"]
    block["timed_iterations"] = 0  # a valid int, an impossible setting
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump({"benchmark": block}), encoding="utf-8", newline="\n")

    with pytest.raises(bench.BenchmarkConfigError) as caught:
        bench.load_benchmark_settings(path)
    assert "timed_iterations must be >= 1" in str(caught.value)
    assert "cannot use" not in str(caught.value)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"warmup_iterations": -1}, "warmup_iterations"),
        ({"timed_iterations": 0}, "timed_iterations"),
        ({"report_statistic": "p50"}, "report_statistic"),
        ({"batch_size": 0}, "batch_size"),
        ({"input_size": 0}, "input_size"),
        ({"dtype": "int8"}, "dtype"),
        # p95 can never be below the median, so a threshold of 1.0 would condemn
        # every run that ever completes.
        ({"max_p95_to_statistic_ratio": 1.0}, "max_p95_to_statistic_ratio"),
    ],
)
def test_settings_validation_rejects_unusable_values(overrides, message) -> None:
    with pytest.raises(bench.BenchmarkConfigError, match=message):
        make_settings(**overrides)


# --------------------------------------------------------------------------
# Device / dtype resolution
# --------------------------------------------------------------------------


def test_resolve_device_falls_back_to_cpu_so_the_suite_runs_anywhere() -> None:
    assert bench.resolve_device(FakeCuda(available=True)) == "cuda"
    assert bench.resolve_device(FakeCuda(available=False)) == "cpu"


@pytest.mark.parametrize(
    ("requested", "device", "expected"),
    [
        ("float16", "cuda", "float16"),
        ("bfloat16", "cuda", "bfloat16"),
        ("float32", "cuda", "float32"),
        # Half precision on CPU is emulated; the result must record what really ran.
        ("float16", "cpu", "float32"),
        ("bfloat16", "cpu", "float32"),
        ("float32", "cpu", "float32"),
    ],
)
def test_resolve_dtype_name_reports_the_dtype_that_actually_ran(requested, device, expected):
    assert bench.resolve_dtype_name(requested, device) == expected


def test_resolve_dtype_name_rejects_an_unknown_dtype() -> None:
    with pytest.raises(bench.BenchmarkConfigError, match="Unsupported dtype"):
        bench.resolve_dtype_name("int4", "cuda")


def test_torch_dtype_maps_every_supported_name_and_rejects_others() -> None:
    import torch

    assert bench.torch_dtype("float32") is torch.float32
    assert bench.torch_dtype("float16") is torch.float16
    assert bench.torch_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(bench.BenchmarkConfigError):
        bench.torch_dtype("int4")


def test_default_cuda_api_is_torch_cuda() -> None:
    import torch

    assert bench.default_cuda_api() is torch.cuda


def test_every_cuda_default_path_is_exercised_because_production_takes_only_those(
    monkeypatch,
) -> None:
    """M2: `main()` never injects a cuda_api, so `cuda_api=None` IS production.

    Four call sites resolve their CUDA api through `default_cuda_api()`. Written
    as ternaries, the default half of each is invisible to coverage.py, so all
    four were reported covered while no test had ever executed them.
    """

    fake = FakeCuda()
    monkeypatch.setattr(bench, "default_cuda_api", lambda: fake)

    assert bench.resolve_device() == "cuda"
    bench.make_synchronizer("cuda")()
    bench.reset_peak_vram("cuda")
    fake.allocate(2 * 1024 * 1024)
    assert bench.measure_peak_vram_mb("cuda") == pytest.approx(2.0)

    assert fake.events == [
        "is_available",
        "synchronize",
        "reset_peak_memory_stats",
        "max_memory_allocated",
    ], "all four defaults must reach the api returned by default_cuda_api()"


def test_the_cpu_paths_never_touch_the_cuda_api_at_all() -> None:
    fake = FakeCuda(available=False)
    assert bench.make_synchronizer("cpu", fake)() is None
    bench.reset_peak_vram("cpu", fake)
    assert bench.measure_peak_vram_mb("cpu", fake) is None
    assert fake.events == []


# --------------------------------------------------------------------------
# Synchronization and VRAM
# --------------------------------------------------------------------------


def test_synchronizer_is_a_noop_on_cpu_and_the_real_call_on_cuda() -> None:
    cuda = FakeCuda()
    assert bench.make_synchronizer("cpu", cuda)() is None
    assert cuda.syncs == 0

    bench.make_synchronizer("cuda", cuda)()
    assert cuda.syncs == 1


def test_peak_vram_is_none_on_cpu_and_mib_on_cuda() -> None:
    cuda = FakeCuda()

    bench.reset_peak_vram("cpu", cuda)
    assert bench.measure_peak_vram_mb("cpu", cuda) is None
    assert cuda.resets == 0

    bench.reset_peak_vram("cuda", cuda)
    cuda.allocate(3 * 1024 * 1024)
    assert cuda.resets == 1
    assert bench.measure_peak_vram_mb("cuda", cuda) == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def test_warmup_iterations_are_executed_but_excluded_from_the_timings() -> None:
    """The warmup calls are made 1000x more expensive than the timed ones.

    If any of them leaked into the returned durations the median would be nowhere
    near 1 ms.
    """

    clock = FakeClock()
    workload = ClockDrivenWorkload(
        clock, warmup_iterations=3, warmup_cost_s=1.0, timed_costs_s=[0.001] * 5
    )
    durations = bench.time_callable(
        workload, warmup_iterations=3, timed_iterations=5, timer=clock
    )

    assert workload.calls == 8, "warmup iterations must still be executed"
    assert len(durations) == 5, "only the timed iterations may be returned"
    assert durations == pytest.approx([1.0] * 5)


def test_the_synchronize_is_inside_the_timed_region_not_after_it() -> None:
    """M1: the single most important correctness property of a CUDA benchmark.

    CUDA is asynchronous, so the work is not finished when the call returns - it
    is finished when the synchronize returns. Moving `sync()` to after the clock
    is read leaves the timer measuring how fast Python enqueues kernels, which is
    fiction, and a test that counts sync CALLS cannot see the difference: the
    count is identical either way.

    Two independent assertions catch it. The event log pins the ORDER, and the
    fake synchronize charges the clock so a misplaced one also changes the
    DURATION - 1 ms of enqueueing instead of the 5 ms the work really took.
    """

    log: list[str] = []
    clock = RecordingClock(log)
    cuda = FakeCuda(clock=clock, sync_cost_s=0.004, log=log)
    workload = ClockDrivenWorkload(
        clock,
        warmup_iterations=1,
        warmup_cost_s=0.5,
        timed_costs_s=[0.001] * 2,
        log=log,
    )

    durations = bench.time_callable(
        workload,
        warmup_iterations=1,
        timed_iterations=2,
        synchronize=bench.make_synchronizer("cuda", cuda),
        timer=clock,
    )

    assert log == [
        "call",  # warmup, never timed
        "synchronize",  # drain what the warmup queued
        "clock", "call", "synchronize", "clock",  # timed iteration 1
        "clock", "call", "synchronize", "clock",  # timed iteration 2
    ]
    assert durations == pytest.approx([5.0, 5.0]), (
        "each timed duration must include the synchronize the work is waited on in; "
        "1.0 ms here would mean the timer measured kernel launches only"
    )


def test_time_callable_works_with_zero_warmup() -> None:
    clock = FakeClock()
    workload = ClockDrivenWorkload(
        clock, warmup_iterations=0, warmup_cost_s=0.0, timed_costs_s=[0.003, 0.003]
    )
    durations = bench.time_callable(
        workload, warmup_iterations=0, timed_iterations=2, timer=clock
    )
    assert workload.calls == 2
    assert durations == pytest.approx([3.0, 3.0])


def test_time_callable_uses_a_real_clock_by_default() -> None:
    durations = bench.time_callable(lambda: sum(range(1000)), warmup_iterations=1,
                                    timed_iterations=3)
    assert len(durations) == 3
    assert all(d >= 0.0 for d in durations)


@pytest.mark.parametrize(
    ("warmup", "timed"),
    [(-1, 5), (0, 0)],
)
def test_time_callable_rejects_impossible_iteration_counts(warmup, timed) -> None:
    with pytest.raises(ValueError):
        bench.time_callable(lambda: None, warmup_iterations=warmup, timed_iterations=timed)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_median_and_p95_are_computed_as_specified() -> None:
    durations = [float(v) for v in range(1, 101)]
    result = bench.LatencyResult.from_durations(
        durations, label="x", settings=make_settings(), device="cuda", dtype="float16"
    )
    # median of 1..100 = (50+51)/2; p95 by linear interpolation = 95 + 0.05*(96-95).
    assert result.latency_ms == pytest.approx(50.5)
    assert result.p95_ms == pytest.approx(95.05)
    assert result.statistic == "median"


def test_the_mean_is_available_but_is_not_the_default() -> None:
    """A single 1000 ms outlier moves the mean by 90 ms and the median not at all."""

    durations = [10.0] * 10 + [1000.0]
    median = bench.LatencyResult.from_durations(
        durations, label="x", settings=make_settings(report_statistic="median"),
        device="cuda", dtype="float16",
    )
    mean = bench.LatencyResult.from_durations(
        durations, label="x", settings=make_settings(report_statistic="mean"),
        device="cuda", dtype="float16",
    )
    assert median.latency_ms == pytest.approx(10.0)
    assert mean.latency_ms == pytest.approx(100.0)


def test_p95_is_omitted_when_the_config_turns_it_off() -> None:
    result = bench.LatencyResult.from_durations(
        [1.0, 2.0, 3.0],
        label="x",
        settings=make_settings(also_report_p95=False),
        device="cuda",
        dtype="float16",
    )
    assert result.p95_ms is None
    assert result.p95_ratio() is None


def test_fps_follows_batch_size_and_the_reported_latency() -> None:
    result = bench.LatencyResult.from_durations(
        [10.0] * 4,
        label="x",
        settings=make_settings(batch_size=4),
        device="cuda",
        dtype="float16",
    )
    assert result.fps == pytest.approx(400.0)
    assert result.batch_size == 4


def test_input_size_can_be_overridden_for_a_native_preset_row() -> None:
    settings = make_settings()
    result = bench.LatencyResult.from_durations(
        [5.0], label="x", settings=settings, device="cuda", dtype="float16", input_size=384
    )
    assert result.input_size == 384
    assert bench.LatencyResult.from_durations(
        [5.0], label="x", settings=settings, device="cuda", dtype="float16"
    ).input_size == settings.input_size


def test_no_timed_iterations_is_an_error_not_a_zero() -> None:
    with pytest.raises(ValueError, match="no timed iterations"):
        bench.LatencyResult.from_durations(
            [], label="x", settings=make_settings(), device="cpu", dtype="float32"
        )


def test_a_zero_latency_is_rejected_rather_than_turned_into_infinite_fps() -> None:
    with pytest.raises(ValueError, match="timer had no resolution"):
        bench.LatencyResult.from_durations(
            [0.0, 0.0], label="x", settings=make_settings(), device="cpu", dtype="float32"
        )


# --------------------------------------------------------------------------
# Contention check
# --------------------------------------------------------------------------


def test_p95_ratio_is_the_measured_tail_not_a_flag() -> None:
    assert make_result(latency_ms=10.0, p95_ms=13.0).p95_ratio() == pytest.approx(1.3)


def test_contention_names_the_rows_that_exceed_the_configured_ratio() -> None:
    """The threshold is an acceptance criterion, so the check is performed here."""

    clean = make_result(label="clean", latency_ms=10.0, p95_ms=12.0)  # 1.20
    contended = make_result(label="contended", latency_ms=10.0, p95_ms=14.0)  # 1.40
    no_p95 = make_result(label="no-p95", p95_ms=None)

    check = bench.evaluate_contention([clean, contended, no_p95], max_ratio=1.3)

    assert check["offenders"] == ["contended"]
    assert check["unchecked"] == ["no-p95"]
    assert check["passed"] is False
    assert dict(check["ratios"])["clean"] == pytest.approx(1.2)

    assert bench.evaluate_contention([clean], max_ratio=1.3)["passed"] is True


def test_a_row_exactly_on_the_threshold_is_accepted() -> None:
    """The boundary decides something, so it is asserted (K-19 shape 4)."""

    on_the_line = make_result(latency_ms=10.0, p95_ms=13.0)
    assert bench.evaluate_contention([on_the_line], max_ratio=1.3)["passed"] is True
    just_over = make_result(latency_ms=10.0, p95_ms=13.001)
    assert bench.evaluate_contention([just_over], max_ratio=1.3)["passed"] is False


# --------------------------------------------------------------------------
# The three required context fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["batch_size", "input_size", "dtype"])
def test_a_result_cannot_be_constructed_without_batch_input_size_and_dtype(missing) -> None:
    """An FPS number without all three is meaningless, so it must be unconstructable."""

    fields = {
        "label": "fake",
        "batch_size": 1,
        "input_size": 64,
        "dtype": "float16",
        "device": "cuda",
        "warmup_iterations": 3,
        "timed_iterations": 5,
        "statistic": "median",
        "latency_ms": 10.0,
        "p95_ms": 12.0,
        "fps": 100.0,
    }
    del fields[missing]
    with pytest.raises(TypeError, match=missing):
        bench.LatencyResult(**fields)


def test_a_result_is_frozen_and_serialises_with_every_context_field() -> None:
    result = make_result()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.batch_size = 8  # type: ignore[misc]
    payload = result.as_dict()
    for key in ("batch_size", "input_size", "dtype", "device", "latency_ms", "p95_ms", "fps"):
        assert key in payload


# --------------------------------------------------------------------------
# run_benchmark
# --------------------------------------------------------------------------


def test_run_benchmark_measures_model_only_and_end_to_end_separately() -> None:
    clock = FakeClock()
    settings = make_settings(warmup_iterations=2, timed_iterations=3)
    model_only = ClockDrivenWorkload(
        clock, warmup_iterations=2, warmup_cost_s=0.9, timed_costs_s=[0.004] * 3
    )
    end_to_end = ClockDrivenWorkload(
        clock, warmup_iterations=2, warmup_cost_s=0.9, timed_costs_s=[0.010] * 3
    )
    cuda = FakeCuda()

    report = bench.run_benchmark(
        label="Fake-Net",
        end_to_end=end_to_end,
        model_only=model_only,
        settings=settings,
        device="cuda",
        cuda_api=cuda,
        timer=clock,
    )

    assert report.model_only is not None
    assert report.model_only.latency_ms == pytest.approx(4.0)
    assert report.end_to_end.latency_ms == pytest.approx(10.0)
    assert report.model_only.label.endswith("[model-only]")
    assert report.end_to_end.label.endswith("[end-to-end]")
    assert [r.dtype for r in report.results()] == ["float16", "float16"]
    assert len(report.results()) == 2


def test_peak_vram_is_reset_before_the_timed_region_not_after_it() -> None:
    """M3: a reset after the timed region reports a peak of nearly zero.

    The real `reset_peak_memory_stats()` sets the peak back to what is currently
    allocated, and activations are freed by the time the run ends - so a late
    reset silently publishes ~0 MiB while the CALL COUNT stays at exactly one.
    The fake reproduces that semantics, which turns the misplacement into a wrong
    NUMBER, and the event log pins the order independently.
    """

    log: list[str] = []
    clock = FakeClock()
    cuda = FakeCuda(log=log)
    workload = AllocatingWorkload(
        clock, cuda, cost_s=0.004, activation_bytes=512 * 1024 * 1024, log=log
    )

    report = bench.run_benchmark(
        label="Fake-Net",
        end_to_end=workload,
        model_only=workload,
        settings=make_settings(warmup_iterations=1, timed_iterations=2),
        device="cuda",
        cuda_api=cuda,
        timer=clock,
    )

    assert report.peak_vram_mb == pytest.approx(512.0), (
        "the peak must cover the timed region; ~0 means the reset landed after it"
    )
    assert cuda.resets == 1
    assert log.index("reset_peak_memory_stats") < log.index("call"), (
        "the reset must precede every workload call"
    )
    assert log[-1] == "max_memory_allocated"


def test_run_benchmark_refuses_to_report_end_to_end_alone_when_config_wants_both() -> None:
    with pytest.raises(ValueError, match="separate_model_and_e2e"):
        bench.run_benchmark(
            label="Fake-Net",
            end_to_end=lambda: None,
            settings=make_settings(separate_model_and_e2e=True),
            device="cpu",
        )


def test_run_benchmark_skips_the_model_only_pass_when_the_config_turns_it_off() -> None:
    clock = FakeClock()
    model_only = ClockDrivenWorkload(
        clock, warmup_iterations=0, warmup_cost_s=0.0, timed_costs_s=[0.004] * 3
    )
    end_to_end = ClockDrivenWorkload(
        clock, warmup_iterations=1, warmup_cost_s=0.5, timed_costs_s=[0.010] * 3
    )
    report = bench.run_benchmark(
        label="Fake-Net",
        end_to_end=end_to_end,
        model_only=model_only,
        settings=make_settings(
            separate_model_and_e2e=False, warmup_iterations=1, timed_iterations=3
        ),
        device="cpu",
        timer=clock,
    )
    assert report.model_only is None
    assert model_only.calls == 0
    assert len(report.results()) == 1
    assert report.peak_vram_mb is None
    # CPU degradation: float16 was requested, float32 is what ran.
    assert report.dtype == "float32"
    assert report.end_to_end.dtype == "float32"


def test_the_model_only_row_is_selected_by_role_never_by_index() -> None:
    """M5: `results()[0]` is the model-only row only while the config says so."""

    both = bench.BenchmarkReport(
        label="Fake-Net",
        device="cuda",
        dtype="float16",
        model_only=make_result(label="Fake-Net [model-only]", latency_ms=4.0),
        end_to_end=make_result(label="Fake-Net [end-to-end]", latency_ms=10.0),
        peak_vram_mb=None,
    )
    chosen = both.model_only_result()
    assert chosen.latency_ms == pytest.approx(4.0)
    assert chosen is not both.end_to_end

    e2e_only = dataclasses.replace(both, model_only=None)
    assert e2e_only.results() == [e2e_only.end_to_end]
    with pytest.raises(ValueError, match="no model-only measurement"):
        # Indexing would have silently returned the end-to-end row here.
        e2e_only.model_only_result()


def test_run_benchmark_accepts_an_explicit_dtype_and_input_size_override() -> None:
    clock = FakeClock()

    def workload():
        return clock.advance(0.002)

    report = bench.run_benchmark(
        label="Fake-Net",
        end_to_end=workload,
        model_only=workload,
        settings=make_settings(warmup_iterations=0, timed_iterations=2),
        device="cuda",
        dtype="bfloat16",
        input_size=384,
        cuda_api=FakeCuda(),
        timer=clock,
    )
    assert report.dtype == "bfloat16"
    assert {r.input_size for r in report.results()} == {384}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_the_table_always_carries_batch_input_size_and_dtype() -> None:
    table = bench.format_results_table([make_result(), make_result(p95_ms=None)])
    header = table.splitlines()[0]
    for column in ("Batch", "Input", "dtype", "Median (ms)", "p95 (ms)", "FPS"):
        assert column in header
    assert "n/a" in table.splitlines()[-1], "a missing p95 must render, not crash"
    assert "float16" in table
    assert "3+5" in table


def test_the_latency_column_is_named_after_the_statistic_that_produced_it() -> None:
    """M4: `report_statistic: mean` under a column headed "Median" is a lie.

    The mean is not hypothetical - the harness validates it, computes it and is
    tested on it, so this rendering is reachable from a legal config.
    """

    median_header = bench.format_results_table([make_result(statistic="median")]).splitlines()[0]
    mean_header = bench.format_results_table([make_result(statistic="mean")]).splitlines()[0]

    assert "Median (ms)" in median_header
    assert "Mean (ms)" in mean_header
    assert "Median" not in mean_header, "means must not be published under 'Median'"
    assert bench.statistic_column_header([]) == "Latency (ms)"


def test_one_column_cannot_label_two_different_statistics() -> None:
    with pytest.raises(ValueError, match="different"):
        bench.format_results_table([make_result(statistic="median"), make_result(statistic="mean")])


# --------------------------------------------------------------------------
# Forbidden-package scan (ADR-005 / PLAN_PHASE2.md M20)
# --------------------------------------------------------------------------


def _tree(root: Path, files: dict[str, str]) -> None:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def test_the_report_calls_the_real_checker_and_holds_no_verdict_of_its_own() -> None:
    """F1: section 5 used to be rendered from a CONSTANT that said the scan passed.

    Two things have to stay true for that not to come back. The name this script
    calls must BE the standalone checker, not a second copy that could drift from
    it, and no source line here may state a scan outcome - the outcome comes from
    the returned dict or it does not exist.
    """

    assert bl.scan_for_forbidden_package is checker.scan_for_forbidden_package

    source = (PROJECT_ROOT / "scripts" / "benchmark_latency.py").read_text(encoding="utf-8")
    assert FORBIDDEN_PACKAGE not in source.lower()
    assert "def scan_for_forbidden_package" not in source, (
        "the scan must live in scripts/check_forbidden_licences.py and be imported, "
        "so the report and CI cannot disagree about what was scanned"
    )
    for fabricated in ("ripgrep exit status", "returns NO matches", "no matches"):
        assert fabricated not in source.lower(), (
            f"{fabricated!r} is a stored verdict; section 5 must render the scan result"
        )


def test_the_rendered_scan_section_states_the_outcome_it_was_given() -> None:
    clean = format_forbidden_scan(
        {"roots": ["src"], "missing_roots": [], "files_scanned": 12, "matches": [], "clean": True}
    )
    dirty = format_forbidden_scan(
        {
            "roots": ["src"],
            "missing_roots": ["notebooks"],
            "files_scanned": 12,
            "matches": ["src/dirty.py:2"],
            "clean": False,
        }
    )
    assert any("**PASS**" in line for line in clean)
    assert not any("FAIL" in line for line in clean)
    assert any("FAIL" in line for line in dirty)
    assert any("src/dirty.py:2" in line for line in dirty)
    assert any("ABSENT" in line for line in dirty)


# --------------------------------------------------------------------------
# Script helpers
# --------------------------------------------------------------------------


def test_select_probe_image_is_deterministic_and_split_scoped(tmp_path) -> None:
    manifest = {
        "images": [
            {"file_name": "images/b.png", "split": "val"},
            {"file_name": "images/a.png", "split": "val"},
            {"file_name": "images/aaa.png", "split": "test"},
        ]
    }
    chosen = select_probe_image(manifest, tmp_path, "val")
    assert chosen == tmp_path / "images/a.png"


def test_select_probe_image_fails_loudly_on_an_empty_split(tmp_path) -> None:
    with pytest.raises(SystemExit):
        select_probe_image({"images": []}, tmp_path, "val")


def test_licence_evidence_is_read_from_the_hub() -> None:
    class FakeApi:
        def model_info(self, model_id):
            return SimpleNamespace(
                cardData={"license": "apache-2.0"},
                sha="deadbeef",
                tags=["vision", "license:apache-2.0"],
            )

    evidence = fetch_licence_evidence("Roboflow/rf-detr-nano", api=FakeApi())
    assert evidence["licence"] == "apache-2.0"
    assert evidence["revision"] == "deadbeef"
    assert evidence["tags"] == ["license:apache-2.0"]
    assert evidence["error"] is None


def test_licence_evidence_survives_a_repo_with_no_card_metadata() -> None:
    """M8: both fallbacks are real Hub states and neither had ever been executed."""

    class BareApi:
        def model_info(self, model_id):
            return SimpleNamespace(sha="cafe")  # no cardData, no tags

    evidence = fetch_licence_evidence("Fake/net", api=BareApi())
    assert evidence["licence"] is None
    assert evidence["tags"] == []
    assert evidence["revision"] == "cafe"
    assert evidence["error"] is None


def test_licence_evidence_records_the_failure_instead_of_inventing_a_licence() -> None:
    class BrokenApi:
        def model_info(self, model_id):
            raise OSError("offline")

    evidence = fetch_licence_evidence("Roboflow/rf-detr-nano", api=BrokenApi())
    assert evidence["licence"] is None
    assert "OSError" in evidence["error"]


class FakeShaped:
    def __init__(self, shape):
        self.shape = shape


class FakeParameter:
    def __init__(self, count):
        self._count = count

    def numel(self):
        return self._count


class FakeForObjectDetection:
    """The MODEL half of the landing check. Deliberately a different class from
    the processor, so `type(model).__name__` cannot be confused with the
    processor's - passing one object as both made the two indistinguishable."""

    def parameters(self):
        return [FakeParameter(1_000_000), FakeParameter(500_000)]


class FakeImageProcessor:
    """The PROCESSOR half of the landing check."""

    def __init__(self, size):
        self.size = size


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        # What the RT-DETR / RF-DETR processors carry, as a dict and as a SizeDict.
        ({"height": 384, "width": 384}, 384),
        (SimpleNamespace(height=384, width=384), 384),
        # A2: the wider DETR family carries these instead, and a SizeDict reads
        # the absent key back as None rather than raising - so `int(size.height)`
        # would die with TypeError on a perfectly valid processor.
        ({"shortest_edge": 800, "longest_edge": 1333}, 800),
        (SimpleNamespace(height=None, width=None, shortest_edge=800, longest_edge=1333), 800),
    ],
)
def test_processor_native_size_handles_both_shapes_in_the_wild(size, expected) -> None:
    assert processor_native_size(size) == expected


def test_processor_native_size_refuses_to_guess() -> None:
    with pytest.raises(ValueError, match="neither"):
        processor_native_size({"max_pixels": 100})


@pytest.mark.parametrize(
    "size",
    [{"height": 384, "width": 384}, SimpleNamespace(height=384, width=384)],
)
def test_landing_check_reads_shapes_off_the_real_outputs(size) -> None:
    """M7: model and processor are DISTINCT objects, so their classes are distinct."""

    model = FakeForObjectDetection()
    processor = FakeImageProcessor(size)
    outputs = SimpleNamespace(
        logits=FakeShaped((1, 300, 91)), pred_boxes=FakeShaped((1, 300, 4))
    )
    landing = landing_check(
        model,
        processor,
        model_only=lambda: outputs,
        end_to_end=lambda: [{"boxes": FakeShaped((7, 4))}],
    )
    assert landing["logits_shape"] == [1, 300, 91]
    assert landing["boxes_shape"] == [1, 300, 4]
    assert landing["n_detections"] == 7
    assert landing["native_input_size"] == 384
    assert landing["parameters_m"] == pytest.approx(1.5)
    assert landing["model_class"] == "FakeForObjectDetection"
    assert landing["processor_class"] == "FakeImageProcessor"


def test_score_threshold_comes_from_the_project_config() -> None:
    with EVALUATION_CONFIG.open("r", encoding="utf-8") as stream:
        expected = yaml.safe_load(stream)["compliance"]["score_threshold"]
    assert read_score_threshold(EVALUATION_CONFIG) == pytest.approx(expected)


def test_model_only_callable_feeds_a_device_resident_tensor_of_the_right_shape() -> None:
    seen = []

    def fake_model(*, pixel_values):
        seen.append((tuple(pixel_values.shape), str(pixel_values.dtype)))

    run = build_model_only_callable(
        fake_model, batch_size=2, input_size=32, device="cpu", dtype_name="float32"
    )
    run()
    run()
    assert seen == [((2, 3, 32, 32), "torch.float32")] * 2


def test_end_to_end_callable_keeps_height_and_width_in_that_order() -> None:
    """M6: a square fixture cannot see a transposition, and the split is not square.

    The frozen Test split holds 416x416, 416x415, 415x416 and 415x415 images, so
    `target_sizes` written as (width, height) would corrupt real coordinates on
    three quarters of the shapes present. The fixture is therefore 415 wide by
    416 tall, and the two numbers are different.
    """

    import torch

    calls = {"preprocess": 0, "postprocess": 0, "threshold": None, "size": None}

    class FakeProcessor:
        def __call__(self, *, images, return_tensors, size):
            calls["preprocess"] += 1
            calls["size"] = size
            assert len(images) == 2, "batch_size copies of the probe image"
            return {"pixel_values": torch.zeros(2, 3, 32, 32)}

        def post_process_object_detection(self, outputs, *, threshold, target_sizes):
            calls["postprocess"] += 1
            calls["threshold"] = threshold
            return [{"boxes": target_sizes}]

    run = build_end_to_end_callable(
        lambda *, pixel_values: SimpleNamespace(logits=pixel_values),
        FakeProcessor(),
        object(),
        batch_size=2,
        input_size=32,
        device="cpu",
        dtype_name="float32",
        score_threshold=0.5,
        source_size=(416, 415),  # (height, width) of a 415x416 image
    )
    result = run()
    assert calls == {
        "preprocess": 1,
        "postprocess": 1,
        "threshold": 0.5,
        "size": {"height": 32, "width": 32},
    }
    assert result[0]["boxes"].tolist() == [[416, 415], [416, 415]], (
        "target_sizes must be (height, width) per image, as transformers expects"
    )


def test_measure_at_resolution_records_the_size_it_actually_ran() -> None:
    seen = []

    def fake_model(*, pixel_values):
        seen.append(tuple(pixel_values.shape))

    result = measure_at_resolution(
        fake_model,
        size=32,
        settings=make_settings(warmup_iterations=1, timed_iterations=2),
        device="cpu",
        dtype_name="float32",
        label="Fake-Net [model-only @ 32]",
    )
    assert result.input_size == 32
    assert seen == [(1, 3, 32, 32)] * 3, "one warmup call plus two timed calls"


def test_resolution_sensitivity_note_reports_the_change_not_a_verdict() -> None:
    """The note states the measured change; it does not silently classify it."""

    flat = resolution_sensitivity_note(
        "Fake-Net", full_size=640, full_ms=12.0, probe_size=320, probe_ms=12.6
    )
    assert "+5.0%" in flat
    assert "4x fewer input pixels" in flat

    scaling = resolution_sensitivity_note(
        "Fake-Net", full_size=640, full_ms=12.0, probe_size=320, probe_ms=3.0
    )
    assert "-75.0%" in scaling

    upward = resolution_sensitivity_note(
        "Fake-Net", full_size=640, full_ms=12.0, probe_size=1280, probe_ms=40.0
    )
    assert "+233.3%" in upward
    assert "4x more input pixels" in upward


def test_a_resolution_probe_at_the_configured_size_compares_nothing() -> None:
    with pytest.raises(ValueError, match="says nothing"):
        resolution_sensitivity_note(
            "Fake-Net", full_size=640, full_ms=12.0, probe_size=640, probe_ms=12.0
        )


def test_the_dispatch_verdict_is_computed_from_the_probes_not_written_down() -> None:
    """Whether this machine is dispatch-bound is a property of the run.

    A fixed sentence would go on claiming yesterday's answer, so the verdict has
    to move when the measurements move - including all the way to the opposite
    reading.
    """

    # The two moves are deliberately of DIFFERENT magnitude, or "the largest" is
    # not a choice and the assertion below would not discriminate.
    flat = [
        ResolutionProbe(title="A", full_size=640, full_ms=12.0, probe_size=320, probe_ms=11.4),
        ResolutionProbe(title="B", full_size=640, full_ms=12.0, probe_size=1280, probe_ms=13.2),
    ]
    verdict = dispatch_bound_verdict(flat)
    assert "16x span" in verdict, "320 to 1280 is a 16x pixel-count span"
    assert "320x320 to 1280x1280" in verdict
    assert "`+10.0%` (B at 1280x1280)" in verdict, "the larger move must be the one quoted"

    compute_bound = [
        ResolutionProbe(title="A", full_size=640, full_ms=12.0, probe_size=320, probe_ms=3.0),
        ResolutionProbe(title="A", full_size=640, full_ms=12.0, probe_size=1280, probe_ms=48.0),
    ]
    assert "`+300.0%`" in dispatch_bound_verdict(compute_bound)

    assert "No resolution probes ran" in dispatch_bound_verdict([])


CLEAN_SCAN = {
    "roots": ["src", "scripts", "notebooks"],
    "missing_roots": [],
    "files_scanned": 42,
    "matches": [],
    "clean": True,
}


def _render(entries, resolution_results, resolution_probes=(), scan=None, settings=None):
    return render_report(
        generated_at="2026-08-01T00:00:00Z",
        device_name="cuda (Fake GPU)",
        settings=make_settings() if settings is None else settings,
        dtype_name="float16",
        probe_image="probe.png",
        source_size=(416, 415),
        score_threshold=0.5,
        entries=entries,
        resolution_results=resolution_results,
        resolution_probes=resolution_probes,
        forbidden_scan=CLEAN_SCAN if scan is None else scan,
    )


def _fake_entry(**overrides):
    entry = {
        "spec": SimpleNamespace(
            title="Fake-Net", checkpoint="Fake/net", role="speed baseline"
        ),
        "landing": {
            "model_class": "FakeForObjectDetection",
            "processor_class": "FakeImageProcessor",
            "parameters_m": 20.17,
            "native_input_size": 640,
            "logits_shape": [1, 300, 91],
            "boxes_shape": [1, 300, 4],
            "n_detections": 1,
        },
        "report": bench.BenchmarkReport(
            label="Fake-Net",
            device="cuda",
            dtype="float16",
            model_only=make_result(label="Fake-Net [model-only]"),
            end_to_end=make_result(label="Fake-Net [end-to-end]"),
            peak_vram_mb=123.4,
        ),
        "licence": {
            "model_id": "Fake/net",
            "licence": "apache-2.0",
            "revision": "deadbeef",
            "tags": ["license:apache-2.0"],
            "error": None,
        },
    }
    entry.update(overrides)
    return entry


def test_the_report_labels_the_numbers_provisional_and_states_the_context() -> None:
    markdown = _render(
        [_fake_entry()],
        [make_result(label="Fake-Net [model-only @ 32]", input_size=32)],
        [ResolutionProbe(
            title="Fake-Net", full_size=64, full_ms=10.0, probe_size=32, probe_ms=9.0
        )],
    )
    assert "PROVISIONAL" in markdown
    assert "fine-tuned 3-class weights" in markdown
    assert "not comparable to section 1" in markdown
    assert "dispatch-bound" in markdown
    assert "-10.0%" in markdown, "the sensitivity notes must reach the report"
    assert "apache-2.0" in markdown
    assert "PML-1.0" in markdown
    assert "123.4" in markdown
    assert "VALIDATION split, never Test" in markdown
    assert "`float16`" in markdown
    assert "(415x416)" in markdown, "the probe image is described as width x height"


def test_the_report_prints_the_contention_ratio_instead_of_asking_the_reader() -> None:
    settings = make_settings(max_p95_to_statistic_ratio=1.15)
    entry = _fake_entry()  # p95 12.0 over median 10.0 -> 1.20, above 1.15
    markdown = _render([entry], [], scan=CLEAN_SCAN, settings=settings)

    assert "| Fake-Net [model-only] | 1.20 | CONTENDED" in markdown
    assert "**FAIL**" in markdown
    assert "1.15" in markdown

    relaxed = _render([entry], [], scan=CLEAN_SCAN, settings=make_settings())
    assert "| Fake-Net [model-only] | 1.20 | ok |" in relaxed
    assert "**PASS**" in relaxed


def test_a_contended_resolution_probe_fails_the_check_too() -> None:
    """Section 4 carries the dispatch-bound reading, so its rows are checked as well.

    Checking only the headline table would let the conclusion rest on a row that
    was measured while the machine was busy.
    """

    noisy_probe = make_result(
        label="Fake-Net [model-only @ 1280]", input_size=1280, latency_ms=10.0, p95_ms=19.0
    )
    markdown = _render([_fake_entry()], [noisy_probe])
    assert "| Fake-Net [model-only @ 1280] | 1.90 | CONTENDED" in markdown
    assert "**FAIL** - 1 of 3 measured rows exceeded" in markdown


def test_the_contention_table_says_so_when_p95_reporting_is_switched_off() -> None:
    """`also_report_p95: false` is a legal config, and then the check cannot run."""

    entry = _fake_entry()
    entry["report"] = dataclasses.replace(
        entry["report"],
        model_only=make_result(label="Fake-Net [model-only]", p95_ms=None),
        end_to_end=make_result(label="Fake-Net [end-to-end]", p95_ms=None),
    )
    markdown = _render([entry], [], settings=make_settings(also_report_p95=False))
    assert "| Fake-Net [model-only] | n/a | not checked (`also_report_p95` is off) |" in markdown
    assert "0 of 0 measured rows exceeded" in markdown


def test_the_report_renders_a_cpu_run_that_has_no_peak_vram() -> None:
    """M8: `n/a (CPU)` is what every CPU-only run renders, and no test took it."""

    entry = _fake_entry()
    entry["report"] = dataclasses.replace(entry["report"], peak_vram_mb=None)
    markdown = _render([entry], [])
    assert "| Fake-Net | n/a (CPU) |" in markdown


def test_the_report_says_so_when_no_resolution_probe_ran() -> None:
    markdown = _render([_fake_entry()], [])
    assert "No resolution probes were recorded" in markdown


def test_the_report_does_not_invent_a_licence_when_the_hub_is_unreachable() -> None:
    entry = _fake_entry()
    entry["licence"] = {
        "model_id": "Fake/net",
        "licence": None,
        "revision": None,
        "tags": [],
        "error": "OSError: offline",
    }
    markdown = _render([entry], [])
    assert "UNAVAILABLE (OSError: offline)" in markdown
    assert "| `UNAVAILABLE (OSError: offline)` | - |" in markdown, "empty tags render as -"


def test_the_report_section_5_reflects_the_scan_it_was_handed() -> None:
    """F1: the fabricated constant claimed a pass no matter what was in the tree."""

    clean = _render([_fake_entry()], [])
    assert "**PASS** - no file under those roots mentions" in clean

    dirty = _render(
        [_fake_entry()],
        [],
        scan={
            "roots": ["src", "scripts", "notebooks"],
            "missing_roots": [],
            "files_scanned": 42,
            "matches": ["src/detect.py:3"],
            "clean": False,
        },
    )
    assert "ADR-005 IS VIOLATED" in dirty
    assert "src/detect.py:3" in dirty
    assert "**PASS**" not in dirty.split("## 5.")[1]


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------


class MainFakeModel:
    """A 'model' cheap enough for a unit test but expensive enough to time."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *, pixel_values):
        import torch

        self.calls += 1
        sum(range(4_000))  # ~50 us, so perf_counter resolves it
        batch = pixel_values.shape[0]
        return SimpleNamespace(
            logits=torch.zeros(batch, 300, 91), pred_boxes=torch.zeros(batch, 300, 4)
        )

    def parameters(self):
        return [FakeParameter(20_170_000)]

    def to(self, device):
        return self

    def eval(self):
        return self


class MainFakeProcessor:
    """Preprocessing is made ~20x the model cost, so the two are distinguishable."""

    def __init__(self, native=48):
        self.size = {"height": native, "width": native}
        self.target_sizes = []

    def __call__(self, *, images, return_tensors, size):
        import torch

        sum(range(120_000))
        return {"pixel_values": torch.zeros(len(images), 3, size["height"], size["width"])}

    def post_process_object_detection(self, outputs, *, threshold, target_sizes):
        self.target_sizes.append(target_sizes.tolist())
        return [{"boxes": target_sizes}]


CLEAN_TREE = {
    "src/clean.py": "import torch\n",
    "scripts/also_clean.py": "from transformers import AutoModel\n",
    "notebooks/nb.ipynb": '{"cells": []}\n',
}


def _install_main_fakes(
    tmp_path, monkeypatch, *, scan_tree=None, native=48, models="rtdetrv2_r18", **setting_overrides
):
    """Wire main() to fakes and a tmp filesystem; return the pieces to assert on.

    The forbidden-package scan is NOT faked. `bl.PROJECT_ROOT` is pointed at a
    real tmp tree and the real checker walks it, so what section 5 of the written
    report says is the outcome of a scan that actually ran over files that
    actually exist - which is the whole difference between this and the constant
    it replaced.
    """

    from PIL import Image

    reports = tmp_path / "reports"
    splits = tmp_path / "splits"
    images = tmp_path / "images"
    for directory in (reports, splits, images):
        directory.mkdir(parents=True)

    scan_root = tmp_path / "scanned"
    _tree(scan_root, CLEAN_TREE if scan_tree is None else scan_tree)
    monkeypatch.setattr(bl, "PROJECT_ROOT", scan_root)

    # 415 wide by 416 tall: a shape that really occurs in the frozen split, and
    # one where a transposed target_sizes would be visible.
    Image.new("RGB", (415, 416), color=(10, 20, 30)).save(images / "probe.png")
    (splits / "split_manifest.json").write_text(
        json.dumps({"images": [{"file_name": "probe.png", "split": "val"}]}),
        encoding="utf-8",
        newline="\n",
    )

    model, processor = MainFakeModel(), MainFakeProcessor(native=native)
    settings = make_settings(
        warmup_iterations=1, timed_iterations=2, input_size=64, **setting_overrides
    )

    monkeypatch.setattr(sys, "argv", ["benchmark_latency.py", "--models", models])
    monkeypatch.setattr(
        bl,
        "load_project_paths",
        lambda: SimpleNamespace(reports=reports, splits=splits, hardhat_raw=images),
    )
    monkeypatch.setattr(bl, "load_benchmark_settings", lambda: settings)
    monkeypatch.setattr(bl, "read_score_threshold", lambda path: 0.5)
    monkeypatch.setattr(bl, "resolve_device", lambda: "cpu")
    monkeypatch.setattr(bl, "load_detector", lambda checkpoint, **kw: (model, processor))

    class FakeHfApi:
        def model_info(self, model_id):
            return SimpleNamespace(
                cardData={"license": "apache-2.0"}, sha="deadbeef", tags=["license:apache-2.0"]
            )

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
    return SimpleNamespace(
        reports=reports,
        model=model,
        processor=processor,
        settings=settings,
        scan_root=scan_root,
    )


def _row(markdown: str, label: str) -> list[str]:
    for line in markdown.splitlines():
        if line.startswith(f"| {label} |"):
            return [cell.strip() for cell in line.split("|")]
    raise AssertionError(f"no table row for {label!r} in the report")


def test_main_writes_a_report_whose_baseline_is_the_model_only_row(tmp_path, monkeypatch) -> None:
    """M5 / M6 end to end: nothing used to execute main(), so nothing checked either.

    The end-to-end pass is made roughly twenty times the cost of the model-only
    pass, so the resolution notes quote a visibly different number depending on
    which row they were built from. `results()[0]` and `results()[-1]` are then
    distinguishable, which they were not while every row cost the same.
    """

    fakes = _install_main_fakes(tmp_path, monkeypatch)
    bl.main()

    markdown = (fakes.reports / "speed_baseline_probe.md").read_text(encoding="utf-8")
    model_only_ms = _row(markdown, "RT-DETRv2-R18 [model-only]")[6]
    end_to_end_ms = _row(markdown, "RT-DETRv2-R18 [end-to-end]")[6]
    assert float(model_only_ms) < float(end_to_end_ms), (
        "fixture degenerated: the two rows must differ or this test discriminates nothing"
    )

    quoted = re.findall(r"against `([0-9.]+)` ms", markdown)
    assert len(quoted) == 2, "one sensitivity note per control probe"
    assert set(quoted) == {model_only_ms}, (
        "the resolution probes must be compared against the MODEL-ONLY baseline"
    )

    # Both control directions ran, plus the processor's native preset.
    assert "[model-only @ 32]" in markdown
    assert "[model-only @ 128]" in markdown
    assert "[model-only @ 48, native preset]" in markdown

    # M6: (height, width), from a 415x416 probe image.
    assert fakes.processor.target_sizes[0] == [[416, 415]]
    assert "(415x416)" in markdown

    # A CPU run has no peak VRAM, and the contention check really ran.
    assert "| RT-DETRv2-R18 | n/a (CPU) |" in markdown
    assert "### Contention check (performed, not asserted)" in markdown

    # Section 5 reports a scan that really walked the tmp tree: three files, and
    # the count is the checker's, not a sentence.
    assert "**PASS** - no file under those roots mentions" in markdown
    assert f"- Files read: `{len(CLEAN_TREE)}`" in markdown
    assert "- Matches: `0`" in markdown


def test_main_writes_the_evidence_and_then_fails_when_the_scan_is_dirty(
    tmp_path, monkeypatch
) -> None:
    """An ADR-005 violation must not be something a green exit status hides.

    F1 end to end, with NO fake scan anywhere: a real import of the forbidden
    package is planted in a real file, the real checker walks the real tree, and
    the report has to say so. The predecessor of this section rendered a constant
    and would have printed PASS over exactly this tree.
    """

    dirty_tree = dict(CLEAN_TREE)
    dirty_tree["src/detect.py"] = (
        f"import torch\n\n\nfrom {FORBIDDEN_PACKAGE} import YOLO  # ADR-005 violation\n"
    )
    # native == input_size here, so the extra "native preset" row is correctly
    # skipped rather than duplicating a measurement section 1 already holds.
    fakes = _install_main_fakes(tmp_path, monkeypatch, scan_tree=dirty_tree, native=64)

    with pytest.raises(SystemExit, match="ADR-005 violated"):
        bl.main()

    markdown = (fakes.reports / "speed_baseline_probe.md").read_text(encoding="utf-8")
    assert "ADR-005 IS VIOLATED" in markdown
    assert "src/detect.py:4" in markdown, "the real line number of the planted import"
    assert "- Matches: `1`" in markdown
    assert "**PASS**" not in markdown.split("### Forbidden-package scan")[1]
    assert "native preset" not in markdown


def test_main_skips_the_control_probes_when_there_is_no_model_only_baseline(
    tmp_path, monkeypatch
) -> None:
    """`separate_model_and_e2e: false` is legal, and then there is nothing to compare to.

    Comparing a model-only probe against the end-to-end row would silently mix
    preprocessing into the baseline, which is the mistake selecting by index made.
    """

    fakes = _install_main_fakes(tmp_path, monkeypatch, separate_model_and_e2e=False)
    bl.main()

    markdown = (fakes.reports / "speed_baseline_probe.md").read_text(encoding="utf-8")
    assert "RT-DETRv2-R18 [model-only] |" not in markdown
    assert not re.search(r"against `[0-9.]+` ms", markdown), (
        "no baseline means no sensitivity note may be invented"
    )
    assert "No resolution probes ran" in markdown


def test_main_refuses_an_unknown_model_key_instead_of_writing_an_empty_report(
    tmp_path, monkeypatch
) -> None:
    fakes = _install_main_fakes(tmp_path, monkeypatch, models="yolo_something")
    with pytest.raises(SystemExit, match="no known model keys"):
        bl.main()
    assert not (fakes.reports / "speed_baseline_probe.md").exists()
