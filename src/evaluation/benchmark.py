"""Latency / throughput / peak-VRAM benchmark harness (DEMO-03).

Every knob is read from configs/evaluation.yaml `benchmark`; this module contains
no tunable number of its own. The design follows four rules that are easy to get
wrong and impossible to notice afterwards:

1. The first inference is never timed. It pays for CUDA context creation, kernel
   autotuning and lazy module init, and it is several times slower than steady
   state. `warmup_iterations` are executed and discarded.
2. The reported number is the MEDIAN (config `report_statistic`), plus p95. A
   mean over a few hundred iterations is dragged around by GC pauses and OS
   scheduling; the median is what the machine actually does and p95 is what the
   tail actually costs.
3. CUDA is asynchronous. Without a synchronize inside the timed region the timer
   measures how fast Python can enqueue work, which is fiction. The synchronize
   is placed after the call and before the clock is read.
4. Model-only and end-to-end are separate measurements. Model-only is a property
   of the architecture; end-to-end includes preprocessing, the host-to-device
   copy and post-processing, and is what a user actually feels.

A latency number without batch size, input resolution and dtype is unusable, so
those three are REQUIRED fields of `LatencyResult` rather than optional metadata:
constructing a result without them raises TypeError.

The harness degrades to CPU when no GPU is present, and every CUDA interaction
goes through an injectable `cuda_api` object (defaulting to `torch.cuda`) so the
CUDA branches are covered by tests on machines that have no GPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.data.paths import PROJECT_ROOT

EVALUATION_CONFIG = PROJECT_ROOT / "configs" / "evaluation.yaml"

# Unit conversions, not tunables.
_MS_PER_SECOND = 1000.0
_BYTES_PER_MIB = 1024.0 * 1024.0

# The percentile level is fixed by the name of the config key `also_report_p95`;
# there is no separate number to configure.
P95_LEVEL = 95.0

SUPPORTED_DTYPES = ("float32", "float16", "bfloat16")

STATISTIC_FUNCTIONS: dict[str, Callable[[Sequence[float]], float]] = {
    "median": lambda values: float(np.median(values)),
    "mean": lambda values: float(np.mean(values)),
}

# Every key `BenchmarkSettings` needs. Checked up front so a missing one is a
# typed BenchmarkConfigError naming the file, not a bare KeyError from a
# subscript twenty lines lower.
BENCHMARK_KEYS = (
    "warmup_iterations",
    "timed_iterations",
    "report_statistic",
    "also_report_p95",
    "separate_model_and_e2e",
    "batch_size",
    "dtype",
    "input_size",
    "max_p95_to_statistic_ratio",
)


class BenchmarkConfigError(ValueError):
    """Raised when configs/evaluation.yaml `benchmark` is not usable as given."""


@dataclass(frozen=True, kw_only=True)
class BenchmarkSettings:
    """The `benchmark` block of configs/evaluation.yaml, validated once."""

    warmup_iterations: int
    timed_iterations: int
    report_statistic: str
    also_report_p95: bool
    separate_model_and_e2e: bool
    batch_size: int
    dtype: str
    input_size: int
    max_p95_to_statistic_ratio: float

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise BenchmarkConfigError(
                f"benchmark.warmup_iterations must be >= 0, got {self.warmup_iterations}"
            )
        if self.timed_iterations < 1:
            raise BenchmarkConfigError(
                f"benchmark.timed_iterations must be >= 1, got {self.timed_iterations}"
            )
        if self.report_statistic not in STATISTIC_FUNCTIONS:
            raise BenchmarkConfigError(
                f"benchmark.report_statistic must be one of "
                f"{sorted(STATISTIC_FUNCTIONS)}, got {self.report_statistic!r}"
            )
        if self.batch_size < 1:
            raise BenchmarkConfigError(
                f"benchmark.batch_size must be >= 1, got {self.batch_size}"
            )
        if self.input_size < 1:
            raise BenchmarkConfigError(
                f"benchmark.input_size must be >= 1, got {self.input_size}"
            )
        if self.dtype not in SUPPORTED_DTYPES:
            raise BenchmarkConfigError(
                f"benchmark.dtype must be one of {list(SUPPORTED_DTYPES)}, got {self.dtype!r}"
            )
        if self.max_p95_to_statistic_ratio <= 1.0:
            # p95 can never be below the median, so a threshold at or under 1.0
            # would condemn every run that ever completes.
            raise BenchmarkConfigError(
                "benchmark.max_p95_to_statistic_ratio must be > 1.0, got "
                f"{self.max_p95_to_statistic_ratio}"
            )


# spec: DEMO-03
def load_benchmark_settings(config_path: Path | None = None) -> BenchmarkSettings:
    """Read the `benchmark` block. UTF-8 is explicit because this machine is cp950 (K-10).

    Every failure mode here raises BenchmarkConfigError naming the offending file.
    A bare KeyError, or the TypeError that `benchmark:` with an empty body used to
    produce, tells the reader nothing about which config is wrong.
    """

    path = EVALUATION_CONFIG if config_path is None else Path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "benchmark" not in config:
        raise BenchmarkConfigError(f"No `benchmark` block in {path}")
    block = config["benchmark"]
    if not isinstance(block, dict):
        raise BenchmarkConfigError(
            f"`benchmark` in {path} must be a mapping, got {type(block).__name__}"
        )
    missing = [key for key in BENCHMARK_KEYS if key not in block]
    if missing:
        raise BenchmarkConfigError(f"`benchmark` in {path} is missing key(s): {missing}")
    try:
        return BenchmarkSettings(
            warmup_iterations=int(block["warmup_iterations"]),
            timed_iterations=int(block["timed_iterations"]),
            report_statistic=str(block["report_statistic"]),
            also_report_p95=bool(block["also_report_p95"]),
            separate_model_and_e2e=bool(block["separate_model_and_e2e"]),
            batch_size=int(block["batch_size"]),
            dtype=str(block["dtype"]),
            input_size=int(block["input_size"]),
            max_p95_to_statistic_ratio=float(block["max_p95_to_statistic_ratio"]),
        )
    except BenchmarkConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise BenchmarkConfigError(
            f"`benchmark` in {path} holds a value this harness cannot use: {error}"
        ) from error


def default_cuda_api() -> Any:
    """`torch.cuda` already exposes exactly the four calls this harness needs."""

    import torch

    return torch.cuda


# spec: DEMO-03
def resolve_device(cuda_api: Any | None = None) -> str:
    """Return `cuda` when a GPU is visible, else `cpu` so the suite runs anywhere."""

    api = default_cuda_api() if cuda_api is None else cuda_api
    return "cuda" if api.is_available() else "cpu"


# spec: DEMO-03
def resolve_dtype_name(requested: str, device: str) -> str:
    """The dtype actually used, which is not always the dtype requested.

    Half precision on CPU is emulated and would produce a latency number that
    describes nothing. The harness silently falls back to float32 there and
    reports the fallback, because the result records the dtype it really ran.
    """

    if requested not in SUPPORTED_DTYPES:
        raise BenchmarkConfigError(
            f"Unsupported dtype {requested!r}; expected one of {list(SUPPORTED_DTYPES)}"
        )
    if device != "cuda" and requested != "float32":
        return "float32"
    return requested


def torch_dtype(name: str):
    """Map the config dtype name onto a torch dtype."""

    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise BenchmarkConfigError(f"Unsupported dtype {name!r}")
    return mapping[name]


# spec: DEMO-03
def make_synchronizer(device: str, cuda_api: Any | None = None) -> Callable[[], None]:
    """A no-op on CPU; `torch.cuda.synchronize` on GPU.

    Without this, a timed region measures kernel-launch throughput, not compute.
    """

    if device != "cuda":
        return lambda: None
    api = default_cuda_api() if cuda_api is None else cuda_api
    return api.synchronize


# spec: DEMO-03
def reset_peak_vram(device: str, cuda_api: Any | None = None) -> None:
    """Peak VRAM is only meaningful relative to a reset immediately before."""

    if device != "cuda":
        return
    api = default_cuda_api() if cuda_api is None else cuda_api
    api.reset_peak_memory_stats()


# spec: DEMO-03
def measure_peak_vram_mb(device: str, cuda_api: Any | None = None) -> float | None:
    """Peak allocated VRAM in MiB, or None on CPU where the number does not exist."""

    if device != "cuda":
        return None
    api = default_cuda_api() if cuda_api is None else cuda_api
    return float(api.max_memory_allocated()) / _BYTES_PER_MIB


# spec: DEMO-03
def time_callable(
    fn: Callable[[], Any],
    *,
    warmup_iterations: int,
    timed_iterations: int,
    synchronize: Callable[[], None] | None = None,
    timer: Callable[[], float] | None = None,
) -> list[float]:
    """Run warmup then timed iterations; return ONLY the timed durations, in ms.

    `timer` is injectable so tests can drive a deterministic clock and assert
    that the warmup iterations really are absent from the returned list.
    """

    if warmup_iterations < 0:
        raise ValueError(f"warmup_iterations must be >= 0, got {warmup_iterations}")
    if timed_iterations < 1:
        raise ValueError(f"timed_iterations must be >= 1, got {timed_iterations}")

    clock = time.perf_counter if timer is None else timer
    sync = (lambda: None) if synchronize is None else synchronize

    for _ in range(warmup_iterations):
        fn()
    # Drain everything the warmup queued, so the first timed iteration does not
    # inherit it.
    sync()

    durations_ms: list[float] = []
    for _ in range(timed_iterations):
        start = clock()
        fn()
        sync()
        durations_ms.append((clock() - start) * _MS_PER_SECOND)
    return durations_ms


@dataclass(frozen=True, kw_only=True)
class LatencyResult:
    """One measurement.

    batch_size, input_size and dtype have no defaults on purpose: an FPS number
    without all three is meaningless, so the harness makes it impossible to
    construct a result that omits them.
    """

    label: str
    batch_size: int
    input_size: int
    dtype: str
    device: str
    warmup_iterations: int
    timed_iterations: int
    statistic: str
    latency_ms: float
    p95_ms: float | None
    fps: float

    # spec: DEMO-03
    @classmethod
    def from_durations(
        cls,
        durations_ms: Sequence[float],
        *,
        label: str,
        settings: BenchmarkSettings,
        device: str,
        dtype: str,
        input_size: int | None = None,
    ) -> LatencyResult:
        """Reduce raw per-iteration durations to the statistics DEMO-03 asks for."""

        if len(durations_ms) == 0:
            raise ValueError(f"{label}: no timed iterations were recorded")
        statistic = STATISTIC_FUNCTIONS[settings.report_statistic](durations_ms)
        if statistic <= 0.0:
            raise ValueError(
                f"{label}: {settings.report_statistic} latency is {statistic} ms, which "
                "means the timer had no resolution here; the number would be fiction"
            )
        p95 = float(np.percentile(durations_ms, P95_LEVEL)) if settings.also_report_p95 else None
        return cls(
            label=label,
            batch_size=settings.batch_size,
            input_size=settings.input_size if input_size is None else int(input_size),
            dtype=dtype,
            device=device,
            warmup_iterations=settings.warmup_iterations,
            timed_iterations=len(durations_ms),
            statistic=settings.report_statistic,
            latency_ms=statistic,
            p95_ms=p95,
            fps=settings.batch_size * _MS_PER_SECOND / statistic,
        )

    # spec: DEMO-03
    def p95_ratio(self) -> float | None:
        """p95 divided by the reported statistic, or None when p95 is switched off.

        This is the contention indicator: a machine that was busy during the run
        produces a long tail, so the ratio climbs while the median may look
        unremarkable. `BenchmarkSettings.max_p95_to_statistic_ratio` is the line.
        """

        if self.p95_ms is None:
            return None
        return self.p95_ms / self.latency_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "dtype": self.dtype,
            "device": self.device,
            "warmup_iterations": self.warmup_iterations,
            "timed_iterations": self.timed_iterations,
            "statistic": self.statistic,
            "latency_ms": self.latency_ms,
            "p95_ms": self.p95_ms,
            "fps": self.fps,
        }


@dataclass(frozen=True, kw_only=True)
class BenchmarkReport:
    """Model-only and end-to-end for one model, plus the peak VRAM they cost."""

    label: str
    device: str
    dtype: str
    model_only: LatencyResult | None
    end_to_end: LatencyResult
    peak_vram_mb: float | None

    def results(self) -> list[LatencyResult]:
        return [r for r in (self.model_only, self.end_to_end) if r is not None]

    # spec: DEMO-03
    def model_only_result(self) -> LatencyResult:
        """The model-only measurement, selected by ROLE.

        `results()[0]` is NOT equivalent. It is the model-only row only while
        `benchmark.separate_model_and_e2e` is true; the moment that is turned off
        the same index silently returns the end-to-end row, and any comparison
        built on it starts comparing two different things without saying so.
        """

        if self.model_only is None:
            raise ValueError(
                f"{self.label}: no model-only measurement exists in this report "
                "(benchmark.separate_model_and_e2e is off), so nothing may be "
                "compared against it"
            )
        return self.model_only


# spec: DEMO-03
def run_benchmark(
    *,
    label: str,
    end_to_end: Callable[[], Any],
    settings: BenchmarkSettings,
    device: str,
    model_only: Callable[[], Any] | None = None,
    dtype: str | None = None,
    input_size: int | None = None,
    cuda_api: Any | None = None,
    timer: Callable[[], float] | None = None,
) -> BenchmarkReport:
    """Measure one model both ways and read peak VRAM across the whole run."""

    if settings.separate_model_and_e2e and model_only is None:
        raise ValueError(
            "configs/evaluation.yaml sets benchmark.separate_model_and_e2e, so a "
            f"model-only callable is required for {label!r}. Reporting end-to-end "
            "alone hides which half of the time the model is responsible for."
        )

    dtype_name = resolve_dtype_name(settings.dtype, device) if dtype is None else dtype
    synchronize = make_synchronizer(device, cuda_api)
    reset_peak_vram(device, cuda_api)

    model_result: LatencyResult | None = None
    if settings.separate_model_and_e2e:
        assert model_only is not None  # guarded above; narrows the type
        model_result = LatencyResult.from_durations(
            time_callable(
                model_only,
                warmup_iterations=settings.warmup_iterations,
                timed_iterations=settings.timed_iterations,
                synchronize=synchronize,
                timer=timer,
            ),
            label=f"{label} [model-only]",
            settings=settings,
            device=device,
            dtype=dtype_name,
            input_size=input_size,
        )

    e2e_result = LatencyResult.from_durations(
        time_callable(
            end_to_end,
            warmup_iterations=settings.warmup_iterations,
            timed_iterations=settings.timed_iterations,
            synchronize=synchronize,
            timer=timer,
        ),
        label=f"{label} [end-to-end]",
        settings=settings,
        device=device,
        dtype=dtype_name,
        input_size=input_size,
    )

    return BenchmarkReport(
        label=label,
        device=device,
        dtype=dtype_name,
        model_only=model_result,
        end_to_end=e2e_result,
        peak_vram_mb=measure_peak_vram_mb(device, cuda_api),
    )


def _format_optional(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


# spec: DEMO-03
def statistic_column_header(results: Sequence[LatencyResult]) -> str:
    """Name the latency column after the statistic the rows actually hold.

    `benchmark.report_statistic` is a real switch: the harness validates, computes
    and tests `mean` as well as `median`. A hardcoded "Median (ms)" header would
    therefore publish means under the word Median - a mislabelling no reader could
    detect from the table.
    """

    names = sorted({result.statistic for result in results})
    if len(names) > 1:
        raise ValueError(
            "one latency column cannot honestly label rows computed with different "
            f"statistics: {names}"
        )
    if not names:
        return "Latency (ms)"
    return f"{names[0].capitalize()} (ms)"


# spec: DEMO-03
def evaluate_contention(
    results: Sequence[LatencyResult], *, max_ratio: float
) -> dict[str, Any]:
    """Perform the p95 / statistic contention check instead of describing it.

    A run taken while the machine was busy shows a long tail. The threshold is
    `benchmark.max_p95_to_statistic_ratio`; this function reports the measured
    ratio of every row and names the rows that exceed it, so the report can state
    an outcome rather than an instruction to the reader.
    """

    ratios: list[tuple[str, float]] = []
    unchecked: list[str] = []
    for result in results:
        ratio = result.p95_ratio()
        if ratio is None:
            unchecked.append(result.label)
        else:
            ratios.append((result.label, ratio))
    offenders = [label for label, ratio in ratios if ratio > max_ratio]
    return {
        "max_ratio": max_ratio,
        "ratios": ratios,
        "offenders": offenders,
        "unchecked": unchecked,
        "passed": not offenders,
    }


# spec: DEMO-03
def format_results_table(results: Sequence[LatencyResult]) -> str:
    """Markdown table. Batch, input size and dtype are columns, never a footnote."""

    header = (
        "| Measurement | Device | Batch | Input | dtype | "
        f"{statistic_column_header(results)} | p95 (ms) | FPS | Iters (warmup+timed) |"
    )
    divider = "|---|---|---:|---:|---|---:|---:|---:|---|"
    lines = [header, divider]
    for result in results:
        lines.append(
            f"| {result.label} | {result.device} | {result.batch_size} | "
            f"{result.input_size} | {result.dtype} | {result.latency_ms:.2f} | "
            f"{_format_optional(result.p95_ms, 2)} | {result.fps:.1f} | "
            f"{result.warmup_iterations}+{result.timed_iterations} |"
        )
    return "\n".join(lines)
