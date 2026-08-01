"""DEMO-03 / DEMO-05: latency, FPS and peak VRAM for the primary detector and the
permissively-licensed speed baseline, written to reports/speed_baseline_probe.md.

Three honesty constraints are baked into this script.

THE PROVISIONAL LABEL IS DERIVED, NOT STORED. A model benchmarked from its public
pretrained COCO head is not measuring the weights this project ships, and the
report has to say so. It used to say so via a hard-coded banner - the same shape
of mistake as the licence-scan constant described below, and with the same failure
mode: the sentence would have kept printing after the condition it describes
stopped being true. The banner is now computed from the label set of each model
that actually loaded, so passing `--weights key=path` for every model is the only
way to make it disappear, and pointing that flag at a COCO checkpoint does not.

LICENCE. ADR-005 selected `Roboflow/rf-detr-nano` as the speed baseline because
its code and weights are Apache-2.0 and therefore compatible with an MIT repo.
AGPL-3.0 detector stacks are forbidden here: importing one would make the
importing file a derivative work. Only the nano / small / medium / base / large
RF-DETR variants are Apache-2.0; XL and 2XL are PML-1.0 and must not be used. The
licence is re-verified against the Hub at run time rather than trusted from the
ADR text.

THE AGPL SCAN IS PERFORMED, NOT QUOTED. The repository-wide check (PLAN_PHASE2.md
M20) is "no file under src/, scripts/ or notebooks/ mentions the forbidden
package". It lives in `scripts/check_forbidden_licences.py`, which is a standalone
CI-usable checker; this script CALLS it and renders whatever it returns - files
read, matches found, and the path of every match. There is no stored sentence to
render instead, which is what an earlier version of section 5 did: a constant
asserting the scan passed, printed whether or not any scan had run.

The probe image comes from the VALIDATION split, never Test. Nothing here trains
or tunes anything, but keeping Test untouched by default costs nothing.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.check_forbidden_licences import scan_for_forbidden_package
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.benchmark import (
    BenchmarkReport,
    BenchmarkSettings,
    LatencyResult,
    evaluate_clock_spread,
    evaluate_contention,
    format_results_table,
    load_benchmark_settings,
    make_clock_sampler,
    make_synchronizer,
    resolve_device,
    resolve_dtype_name,
    run_benchmark,
    time_callable_with_clock,
    torch_dtype,
)

REPORT_NAME = "speed_baseline_probe.md"
PROBE_SPLIT = "val"
PROJECT_CLASSES = frozenset({"helmet", "head", "person"})


@dataclass(frozen=True)
class ModelSpec:
    key: str
    checkpoint: str
    title: str
    role: str


@dataclass(frozen=True)
class WeightsProvenance:
    """Which weights produced the numbers, decided by reading the loaded model.

    `fine_tuned` is NOT "did the caller pass --weights". It is "does this head
    predict this project's three classes", read off `config.id2label` after the
    checkpoint is on the device. A caller who points the flag at a COCO
    checkpoint still gets a PROVISIONAL report, which is the point: the label
    has to track the thing it describes, not the intent of whoever ran it.
    """

    source: str
    labels: tuple[str, ...]

    @property
    def fine_tuned(self) -> bool:
        return set(self.labels) == PROJECT_CLASSES

    def describe(self) -> str:
        kind = "fine-tuned" if self.fine_tuned else "pretrained"
        return f"{kind}, {len(self.labels)} classes"


MODEL_SPECS = (
    ModelSpec(
        key="rtdetrv2_r18",
        checkpoint="PekingU/rtdetr_v2_r18vd",
        title="RT-DETRv2-R18",
        role="primary detector (Apache-2.0)",
    ),
    ModelSpec(
        key="rf_detr_nano",
        checkpoint="Roboflow/rf-detr-nano",
        title="RF-DETR-Nano",
        role="speed baseline (Apache-2.0, ADR-005)",
    ),
)


# spec: DEMO-03
def select_probe_image(manifest: Mapping[str, Any], images_root: Path, split: str) -> Path:
    """First image of `split` in file-name order, so the probe is reproducible."""

    names = sorted(
        str(entry["file_name"]) for entry in manifest["images"] if entry["split"] == split
    )
    if not names:
        raise SystemExit(f"split manifest contains no images in split {split!r}")
    return Path(images_root) / names[0]


# spec: DEMO-05
def fetch_licence_evidence(model_id: str, *, api: Any) -> dict[str, Any]:
    """Read the licence straight off the Hub instead of trusting the ADR text."""

    try:
        info = api.model_info(model_id)
    except Exception as error:  # noqa: BLE001 - being offline is a legitimate outcome
        return {
            "model_id": model_id,
            "licence": None,
            "revision": None,
            "tags": [],
            "error": f"{type(error).__name__}: {error}",
        }
    # Both fallbacks below are real Hub states, not paranoia: a repo can carry no
    # card metadata at all, and the tag list is absent on some responses. Written
    # as statements rather than `or` expressions because coverage.py does not
    # count the untaken side of an `or` as a missed branch, which is how two
    # untested paths hid inside a "100% branch coverage" module.
    card = getattr(info, "cardData", None)
    if card is None:
        card = {}
    tags = getattr(info, "tags", None)
    if tags is None:
        tags = []
    return {
        "model_id": model_id,
        "licence": card.get("license"),
        "revision": getattr(info, "sha", None),
        "tags": [tag for tag in tags if tag.startswith("license")],
        "error": None,
    }


def load_detector(checkpoint: str, *, weights: str | None = None, device: str, dtype_name: str):
    """Auto classes only (ADR-014); dtype is explicit because v5 defaults it to "auto".

    The processor always comes from the Hub id, never from `weights`: this
    project's fine-tuned checkpoints are Trainer output directories and carry no
    `preprocessor_config.json`, which is also why `app.py` names the base
    processor separately. Preprocessing is therefore identical across both, and
    the only thing `weights` changes is the network.
    """

    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(checkpoint)
    # No num_labels override: whatever head the checkpoint carries is the head
    # that gets timed, and the report reads the class list back off it.
    source = checkpoint if weights is None else weights
    model = AutoModelForObjectDetection.from_pretrained(source, dtype=torch_dtype(dtype_name))
    model.to(device)
    model.eval()
    labels = tuple(str(name) for name in model.config.id2label.values())
    return model, processor, WeightsProvenance(source=str(source), labels=labels)


# spec: DEMO-03
def build_model_only_callable(
    model: Any, *, batch_size: int, input_size: int, device: str, dtype_name: str
):
    """Pure forward pass on a tensor that already lives on the device."""

    import torch

    pixel_values = torch.randn(
        batch_size, 3, input_size, input_size, device=device, dtype=torch_dtype(dtype_name)
    )

    def run() -> Any:
        with torch.inference_mode():
            return model(pixel_values=pixel_values)

    return run


# spec: DEMO-03
def build_end_to_end_callable(
    model: Any,
    processor: Any,
    image: Any,
    *,
    batch_size: int,
    input_size: int,
    device: str,
    dtype_name: str,
    score_threshold: float,
    source_size: tuple[int, int],
):
    """Preprocess, transfer, forward, post-process - what a user actually waits for."""

    import torch

    images = [image] * batch_size
    height, width = source_size
    target_sizes = torch.tensor([[height, width]] * batch_size)

    def run() -> Any:
        with torch.inference_mode():
            encoded = processor(
                images=images,
                return_tensors="pt",
                size={"height": input_size, "width": input_size},
            )
            pixel_values = encoded["pixel_values"].to(device=device, dtype=torch_dtype(dtype_name))
            outputs = model(pixel_values=pixel_values)
            return processor.post_process_object_detection(
                outputs, threshold=score_threshold, target_sizes=target_sizes
            )

    return run


# spec: DEMO-05
def processor_native_size(size: Any) -> int:
    """The processor's native input edge, for either `size` shape found in the wild.

    The RT-DETR / RF-DETR processors carry `{"height": H, "width": W}`. The wider
    DETR family carries `{"shortest_edge": S, "longest_edge": L}` instead, and on
    a SizeDict the absent key reads back as None rather than raising - so
    `int(size.height)` would die with a TypeError on a perfectly valid processor.
    """

    if isinstance(size, Mapping):
        height = size.get("height")
        shortest_edge = size.get("shortest_edge")
    else:
        height = getattr(size, "height", None)
        shortest_edge = getattr(size, "shortest_edge", None)

    if height is not None:
        return int(height)
    if shortest_edge is not None:
        return int(shortest_edge)
    raise ValueError(
        f"processor size {size!r} carries neither `height` nor `shortest_edge`, so "
        "the native input size cannot be read off it"
    )


# spec: DEMO-05
def landing_check(model: Any, processor: Any, *, model_only: Any, end_to_end: Any) -> dict[str, Any]:
    """One real forward pass each way, so "it loads" is a measurement, not a claim."""

    outputs = model_only()
    detections = end_to_end()[0]
    native = processor_native_size(processor.size)
    return {
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "parameters_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "native_input_size": native,
        "logits_shape": list(outputs.logits.shape),
        "boxes_shape": list(outputs.pred_boxes.shape),
        "n_detections": int(detections["boxes"].shape[0]),
    }


# spec: DEMO-03
def measure_at_resolution(
    model: Any,
    *,
    size: int,
    settings: BenchmarkSettings,
    device: str,
    dtype_name: str,
    label: str,
    clock_sampler: Any | None = None,
) -> LatencyResult:
    """A model-only measurement at an input size other than the configured one."""

    run = build_model_only_callable(
        model,
        batch_size=settings.batch_size,
        input_size=size,
        device=device,
        dtype_name=dtype_name,
    )
    durations, sm_clock = time_callable_with_clock(
        run,
        warmup_iterations=settings.warmup_iterations,
        timed_iterations=settings.timed_iterations,
        synchronize=make_synchronizer(device),
        clock_sampler=clock_sampler,
    )
    return LatencyResult.from_durations(
        durations,
        label=label,
        settings=settings,
        device=device,
        dtype=dtype_name,
        input_size=size,
        sm_clock_mhz=sm_clock,
    )


@dataclass(frozen=True)
class ResolutionProbe:
    """One model-only measurement at an input size other than the configured one."""

    title: str
    full_size: int
    full_ms: float
    probe_size: int
    probe_ms: float

    def change_percent(self) -> float:
        return (self.probe_ms - self.full_ms) / self.full_ms * 100.0


# spec: DEMO-03
def resolution_sensitivity_note(
    title: str, *, full_size: int, full_ms: float, probe_size: int, probe_ms: float
) -> str:
    """State how much latency actually moved when the pixel count changed.

    This is the difference between "the architecture costs 12 ms" and "our
    dispatch path costs 12 ms and the architecture is hiding underneath it". The
    probe runs in BOTH directions: a flat line downwards alone is weak evidence,
    because it is also what a probe that changed nothing would produce.
    """

    if probe_size == full_size:
        raise ValueError(
            f"{title}: a resolution probe at the configured input size {full_size} "
            "compares a measurement with itself and says nothing"
        )
    change = (probe_ms - full_ms) / full_ms * 100.0
    pixel_ratio = (probe_size / full_size) ** 2
    if pixel_ratio < 1.0:
        factor, direction = 1.0 / pixel_ratio, "fewer"
    else:
        factor, direction = pixel_ratio, "more"
    return (
        f"- **{title}**: {probe_size}x{probe_size} costs `{probe_ms:.2f}` ms against "
        f"`{full_ms:.2f}` ms at {full_size}x{full_size} (`{change:+.1f}%`), on "
        f"{factor:.0f}x {direction} input pixels."
    )


# spec: DEMO-03
def dispatch_bound_verdict(probes: Sequence[ResolutionProbe]) -> str:
    """Report what the probes measured, with the number the reading rests on.

    The verdict has to be computed, not written down: whether this machine is
    dispatch-bound is a property of the run, and a sentence that assumes an answer
    would keep claiming it after the answer changed.
    """

    if not probes:
        return "No resolution probes ran, so nothing here characterises the bottleneck."
    sizes = {probe.probe_size for probe in probes} | {probe.full_size for probe in probes}
    span = (max(sizes) / min(sizes)) ** 2
    largest = max(probes, key=lambda probe: abs(probe.change_percent()))
    return (
        f"MEASURED: across a {span:.0f}x span of input pixel counts "
        f"({min(sizes)}x{min(sizes)} to {max(sizes)}x{max(sizes)}), the largest move any "
        f"model-only measurement made was `{largest.change_percent():+.1f}%` "
        f"({largest.title} at {largest.probe_size}x{largest.probe_size}). Compute-bound "
        "behaviour would be roughly -75% at the small end and +300% at the large end."
    )


def read_score_threshold(config_path: Path) -> float:
    """The post-processing threshold is a config value like every other number."""

    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return float(config["compliance"]["score_threshold"])


# spec: DEMO-05
def format_forbidden_scan(scan: Mapping[str, Any]) -> list[str]:
    """Render what the scan FOUND. There is no wording here for "it passed"."""

    roots = ", ".join(f"`{name}/`" for name in scan["roots"])
    roots_line = f"- Roots scanned: {roots}"
    if scan["missing_roots"]:
        absent = ", ".join(f"`{name}/`" for name in scan["missing_roots"])
        roots_line += f" - **ABSENT from the working tree and therefore NOT scanned: {absent}**"
    lines = [
        (
            "ADR-005 forbids the AGPL-3.0 detector stack because importing it would "
            "make the importing file a derivative work. The counts below are the "
            "return value of `scripts/check_forbidden_licences.py`, a standalone "
            "checker that exits non-zero when it finds a match - they are not a "
            "stored sentence. The search term is assembled from fragments at run "
            "time so the literal appears in no source file, which is why NO file is "
            "exempt from the scan, the checker included."
        ),
        "",
        roots_line,
        f"- Files read: `{scan['files_scanned']}`",
        f"- Matches: `{len(scan['matches'])}`",
        "",
    ]
    if scan["matches"]:
        lines.append(
            "**FAIL - ADR-005 IS VIOLATED.** The forbidden package is named in the "
            "files below. Until they are cleaned up this repository cannot be "
            "released under MIT."
        )
        lines.extend(f"- `{match}`" for match in scan["matches"])
    else:
        lines.append(
            "**PASS** - no file under those roots mentions the forbidden package."
        )
    return lines


# spec: DEMO-03
def weights_banner(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """The PROVISIONAL header, or its absence, decided by what actually loaded."""

    stale = [entry for entry in entries if not entry["weights"].fine_tuned]
    if not stale:
        return [
            "> **Measured on this project's fine-tuned 3-class weights.**",
            "> Every model below predicts `helmet` / `head` / `person`; no row is a public",
            "> COCO checkpoint standing in for one. The weights are named in section 3.",
        ]
    named = ", ".join(f"{entry['spec'].title} (`{entry['weights'].source}`)" for entry in stale)
    return [
        "> **PROVISIONAL - NOT A FINAL RESULT.**",
        "> Still measured from a PUBLIC PRETRAINED COCO checkpoint rather than this",
        f"> project's fine-tuned 3-class weights ({len(stale)} of {len(entries)}): {named}.",
        "> Fine-tuning changes one linear layer at the end of the network - 80 or 91 class",
        "> logits per query become 3 - and nothing else, so those numbers should carry over",
        "> closely. They MUST still be re-measured before any of them reaches the README.",
    ]


# spec: DEMO-03
def format_clock_check(clocks: Mapping[str, Any]) -> list[str]:
    """Render the between-row GPU clock check (K-22).

    Section 1's p95 check is within-row and cannot see the failure this
    addresses: a whole run taken at a low power state, where every number is
    twice what it should be and every ratio still looks fine.
    """

    lines = [
        "### GPU clock check (the failure the p95 ratio cannot see)",
        "",
        (
            "The p95 / statistic ratio is a WITHIN-row test. On 2026-08-01 two runs of "
            "this harness, minutes apart and with no code change, reported `11.81` ms "
            "and `26.74` ms for the same model - a 2.26x move in which every row "
            "scaled together, so the ratios above stayed unremarkable. The SM clocks "
            "during those runs were 2520 MHz and 1215 MHz (2.07x). The clock is "
            "therefore recorded per row above, and the spread between rows is checked "
            "here: rows timed at clocks this far apart are comparing power states "
            "rather than networks."
        ),
        "",
    ]
    if not clocks["observed"]:
        lines += ["No SM clock was readable on this device, so the rows carry `n/a`.", ""]
        return lines
    lines += [
        "| Lowest (MHz) | Highest (MHz) | Spread | Limit | Verdict |",
        "|---:|---:|---:|---:|---|",
        (
            f"| {clocks['lowest']} | {clocks['highest']} | {clocks['spread']:.2f} | "
            f"{clocks['max_ratio']} | {'PASS' if clocks['passed'] else 'FAIL - re-measure'} |"
        ),
        "",
    ]
    if clocks["unchecked"]:
        lines += [
            f"{len(clocks['unchecked'])} row(s) had no readable clock: "
            + ", ".join(f"`{label}`" for label in clocks["unchecked"]),
            "",
        ]
    return lines


# spec: DEMO-03
def pretrained_caveat(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """The section-6 bullet, present only while some row still needs re-measuring."""

    stale = [entry["spec"].title for entry in entries if not entry["weights"].fine_tuned]
    if not stale:
        return []
    return [
        (
            f"- These are pretrained-checkpoint numbers for {', '.join(stale)}. Those "
            "rows must be re-measured on fine-tuned 3-class weights, and until they "
            "are, no number from them may appear in the README without the PROVISIONAL "
            "label."
        )
    ]


# spec: DEMO-03
def weights_asymmetry_note(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    """Say it plainly when the section-1 rows are not on equal footing."""

    kinds = {entry["weights"].fine_tuned for entry in entries}
    if len(kinds) < 2:
        return []
    fine = [e["spec"].title for e in entries if e["weights"].fine_tuned]
    pre = [e["spec"].title for e in entries if not e["weights"].fine_tuned]
    return [
        (
            f"**The rows above are not a like-for-like comparison.** "
            f"{', '.join(fine)} ran on fine-tuned 3-class weights while "
            f"{', '.join(pre)} ran on a public pretrained head. The difference is one "
            "linear layer over 300 queries, so it is small - but it is not zero, and "
            "section 4 already withdraws any claim that these two models can be "
            "separated on this measurement at all."
        ),
        "",
    ]


# spec: DEMO-03
def render_report(
    *,
    generated_at: str,
    device_name: str,
    settings: BenchmarkSettings,
    dtype_name: str,
    probe_image: str,
    source_size: tuple[int, int],
    score_threshold: float,
    entries: Sequence[Mapping[str, Any]],
    resolution_results: Sequence[LatencyResult],
    resolution_probes: Sequence[ResolutionProbe],
    forbidden_scan: Mapping[str, Any],
) -> str:
    """Compose the markdown. Every number in it comes from a measurement above."""

    all_results: list[LatencyResult] = []
    for entry in entries:
        all_results.extend(entry["report"].results())
    # EVERY measured row, not just section 1. The section-4 probes are what the
    # dispatch-bound reading rests on, so a contended probe there corrupts the
    # conclusion just as surely as a contended headline number.
    contention = evaluate_contention(
        [*all_results, *resolution_results], max_ratio=settings.max_p95_to_statistic_ratio
    )
    # Section 1 only, deliberately: the resolution probes in section 4 are meant
    # to run at other input sizes and a clock difference there is not a defect
    # in the head-to-head comparison this check protects.
    clocks = evaluate_clock_spread(all_results, max_ratio=settings.max_clock_spread_ratio)

    lines = [
        "# Speed baseline probe (DEMO-03 / DEMO-05)",
        "",
        *weights_banner(entries),
        "",
        f"- Generated: `{generated_at}`",
        f"- Device: `{device_name}`",
        f"- Requested dtype: `{settings.dtype}` -> effective dtype: `{dtype_name}`",
        (
            f"- Warmup / timed iterations: `{settings.warmup_iterations}` / "
            f"`{settings.timed_iterations}` (warmup discarded, never timed)"
        ),
        (
            f"- Reported statistic: `{settings.report_statistic}`; p95 also reported: "
            f"`{settings.also_report_p95}`"
        ),
        (
            f"- Probe image (VALIDATION split, never Test): `{probe_image}` "
            f"({source_size[1]}x{source_size[0]})"
        ),
        (
            f"- End-to-end post-processing threshold: `{score_threshold}` "
            "(`compliance.score_threshold`)"
        ),
        "",
        (
            "Every setting above is read from `configs/evaluation.yaml`; the harness in "
            "`src/evaluation/benchmark.py` holds no tunable number of its own."
        ),
        "",
        "## 1. Latency",
        "",
        format_results_table(all_results),
        "",
        (
            "`model-only` is a forward pass on a tensor already resident on the device. "
            "`end-to-end` additionally includes image preprocessing, the host-to-device "
            "copy and `post_process_object_detection`; that is what a user feels. Both "
            "are wrapped in `torch.cuda.synchronize()` inside the timed region - without "
            "it the timer would measure how fast Python enqueues work."
        ),
        "",
        *weights_asymmetry_note(entries),
        "### Contention check (performed, not asserted)",
        "",
        (
            "A benchmark taken while the machine was busy shows a long tail: the "
            f"reported {settings.report_statistic} can look ordinary while p95 blows "
            "out. EVERY measured row in this report - the resolution probes in "
            "section 4 as well as the headline numbers above - is therefore checked "
            "against `benchmark.max_p95_to_statistic_ratio` = "
            f"`{settings.max_p95_to_statistic_ratio}`, and the measured ratio is "
            "printed rather than left for the reader to compute."
        ),
        "",
        f"| Measurement | p95 / {settings.report_statistic} | Verdict |",
        "|---|---:|---|",
    ]
    for label, ratio in contention["ratios"]:
        verdict = "CONTENDED - repeat, do not publish" if label in contention["offenders"] else "ok"
        lines.append(f"| {label} | {ratio:.2f} | {verdict} |")
    for label in contention["unchecked"]:
        lines.append(f"| {label} | n/a | not checked (`also_report_p95` is off) |")

    lines += [
        "",
        (
            f"**{'PASS' if contention['passed'] else 'FAIL'}** - "
            f"{len(contention['offenders'])} of {len(contention['ratios'])} measured "
            "rows exceeded the threshold."
        ),
        "",
        *format_clock_check(clocks),
        "## 2. Peak VRAM",
        "",
        "| Model | Peak allocated VRAM (MiB) |",
        "|---|---:|",
    ]
    for entry in entries:
        peak = entry["report"].peak_vram_mb
        if peak is None:
            # The only rendering a CPU-only run ever takes, which is why it is a
            # statement: coverage.py does not flag the untaken side of a ternary.
            peak_text = "n/a (CPU)"
        else:
            peak_text = f"{peak:.1f}"
        lines.append(f"| {entry['spec'].title} | {peak_text} |")

    lines += [
        "",
        (
            "Measured with `torch.cuda.max_memory_allocated()` after "
            "`reset_peak_memory_stats()` immediately before each model's timed region, "
            "so the figure covers weights plus activations for that model alone."
        ),
        "",
        "## 3. Landing check",
        "",
        (
            "| Model | Role | Weights measured | Head | Model class | Params (M) | "
            "Native input | Logits | Boxes | Detections |"
        ),
        "|---|---|---|---|---|---:|---:|---|---|---:|",
    ]
    for entry in entries:
        spec, landing, weights = entry["spec"], entry["landing"], entry["weights"]
        lines.append(
            f"| {spec.title} | {spec.role} | `{weights.source}` | "
            f"{weights.describe()} | "
            f"`{landing['model_class']}` | {landing['parameters_m']:.2f} | "
            f"{landing['native_input_size']} | `{landing['logits_shape']}` | "
            f"`{landing['boxes_shape']}` | {landing['n_detections']} |"
        )

    lines += [
        "",
        (
            "Every model above completed a real forward pass on a real project image; "
            "the shapes are read off the returned tensors, not assumed. `Head` is the "
            "class list read back off `config.id2label` after loading, and it is what "
            "decides the banner at the top of this report. `Detections` counts what "
            "survives the post-processing threshold, and carries meaning for this "
            "project's classes only on a fine-tuned row."
        ),
        "",
        "## 4. Resolution sensitivity: compute-bound or dispatch-bound?",
        "",
    ]
    if resolution_results:
        lines += [
            (
                "Section 1 runs BOTH models at the config `input_size`, so that "
                "comparison is apples-to-apples. This section re-measures model-only "
                "latency at other input sizes, which is the only way to tell whether "
                "the section-1 numbers characterise the architecture or the dispatch "
                "path. The probe runs in BOTH directions - half the configured size "
                "and double it - because a flat line downwards on its own is also "
                "what a probe that changed nothing would produce. "
                "**These rows are not comparable to section 1.**"
            ),
            "",
            format_results_table(resolution_results),
            "",
            *[
                resolution_sensitivity_note(
                    probe.title,
                    full_size=probe.full_size,
                    full_ms=probe.full_ms,
                    probe_size=probe.probe_size,
                    probe_ms=probe.probe_ms,
                )
                for probe in resolution_probes
            ],
            "",
            dispatch_bound_verdict(resolution_probes),
            "",
            (
                "The criterion, stated before the numbers above are read: a "
                "compute-bound batch-1 measurement falls towards a quarter of its cost "
                "when both input dimensions are halved and rises towards four times it "
                "when they are doubled. To the extent a measurement does NOT follow "
                "the pixel count, its wall clock is dominated by per-operator Python "
                "and CUDA launch overhead in eager mode - it characterises OUR "
                "INFERENCE PATH rather than the network, and two models cannot be "
                "separated on the strength of it. Read section 1 accordingly: it is "
                "the latency a user of this eager-PyTorch demo would feel, not a claim "
                "about which architecture is faster."
            ),
            "",
        ]
    else:
        lines += ["No resolution probes were recorded for this run.", ""]

    lines += [
        "## 5. Licence evidence",
        "",
        (
            "ADR-005 forbids AGPL-3.0 detector stacks in this repository: importing one "
            "would make the importing file a derivative work, which is incompatible "
            "with an MIT release. The licence is therefore re-verified against the Hub "
            "at benchmark time rather than trusted from the ADR text."
        ),
        "",
        "| Model | Hub licence | Licence tags | Pinned revision |",
        "|---|---|---|---|",
    ]
    for entry in entries:
        evidence = entry["licence"]
        tags = ", ".join(f"`{tag}`" for tag in evidence["tags"])
        if not tags:
            tags = "-"
        licence = evidence["licence"]
        if licence is None:
            licence = f"UNAVAILABLE ({evidence['error']})"
        lines.append(
            f"| {entry['spec'].title} | `{licence}` | {tags} | `{evidence['revision']}` |"
        )

    lines += [
        "",
        (
            "Only the nano / small / medium / base / large RF-DETR variants are "
            "Apache-2.0. XL and 2XL are PML-1.0 and must not be used here."
        ),
        "",
        "### Forbidden-package scan (PLAN_PHASE2.md M20)",
        "",
        *format_forbidden_scan(forbidden_scan),
        "",
        "## 6. What this probe does NOT establish",
        "",
        *pretrained_caveat(entries),
        (
            "- The accuracy half of DEMO-05 (RF-DETR-Nano trained on the same four "
            "arms) is out of scope: this is a latency-only probe."
        ),
        (
            "- Batch-1 latency only, because latency is a batch-1 question "
            "(`benchmark.batch_size`). Throughput at larger batches is a different "
            "measurement and is not reported here."
        ),
        (
            "- A single run on a desktop GPU that also drives the display, so it is "
            "exposed to whatever else the machine was doing. That exposure is not "
            "argued away here, it is measured: the contention check in section 1 "
            f"prints the p95 / {settings.report_statistic} ratio of every measured "
            "row in the report against "
            "`benchmark.max_p95_to_statistic_ratio`. Run-to-run variation is NOT "
            "characterised - that would need repeated runs recorded as artefacts, "
            "and this report contains one run."
        ),
        (
            "- Eager PyTorch only. Read the measured resolution response in section 4 "
            "before treating any row in section 1 as a statement about the "
            "architectures: to the extent latency does not track the pixel count, "
            "section 1 is measuring this inference path and not the networks. An "
            "export path (`torch.compile`, ONNX or TensorRT) would move the bottleneck "
            "and change the ordering; nothing here has been exported."
        ),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DEMO-03 latency benchmark")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[spec.key for spec in MODEL_SPECS],
        help="subset of model keys to benchmark",
    )
    parser.add_argument("--device", default=None, help="override the auto-detected device")
    parser.add_argument(
        "--weights",
        nargs="*",
        default=[],
        metavar="KEY=PATH",
        help=(
            "benchmark a model from local fine-tuned weights instead of its Hub "
            "checkpoint, e.g. rtdetrv2_r18=D:/.../checkpoint-1752. The processor and "
            "the licence evidence still come from the Hub id."
        ),
    )
    return parser.parse_args()


def parse_weight_overrides(pairs: Sequence[str]) -> dict[str, str]:
    """`KEY=PATH` into a mapping, refusing anything this run cannot honour.

    An unknown key is a hard error rather than a silent no-op: the whole purpose
    of the flag is to remove the PROVISIONAL banner, and a typo that quietly did
    nothing would produce a report labelled provisional for a reason the operator
    believes they just fixed.
    """

    known = {spec.key for spec in MODEL_SPECS}
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, separator, path = pair.partition("=")
        if not separator or not key or not path:
            raise SystemExit(f"--weights expects KEY=PATH, got {pair!r}")
        if key not in known:
            raise SystemExit(f"--weights key {key!r} is not one of {sorted(known)}")
        if not Path(path).is_dir():
            raise SystemExit(f"--weights path for {key} is not a directory: {path}")
        overrides[key] = path
    return overrides


def main() -> None:
    from PIL import Image

    args = parse_args()
    paths = load_project_paths()
    settings = load_benchmark_settings()
    score_threshold = read_score_threshold(PROJECT_ROOT / "configs" / "evaluation.yaml")

    device = resolve_device() if args.device is None else args.device
    dtype_name = resolve_dtype_name(settings.dtype, device)
    device_name = device
    if device == "cuda":
        import torch

        device_name = f"cuda ({torch.cuda.get_device_name(0)})"

    manifest = json.loads((paths.splits / "split_manifest.json").read_text(encoding="utf-8"))
    probe_path = select_probe_image(manifest, paths.hardhat_raw, PROBE_SPLIT)
    image = Image.open(probe_path).convert("RGB")
    source_size = (image.height, image.width)
    print(f"probe image: {probe_path} ({image.width}x{image.height}, split={PROBE_SPLIT})")
    print(f"device={device}  requested dtype={settings.dtype}  effective dtype={dtype_name}")

    from huggingface_hub import HfApi

    hub_api = HfApi()

    wanted = set(args.models)
    selected = [spec for spec in MODEL_SPECS if spec.key in wanted]
    if not selected:
        raise SystemExit(f"no known model keys in {args.models}")
    overrides = parse_weight_overrides(args.weights)
    clock_sampler = make_clock_sampler(device)

    entries: list[dict[str, Any]] = []
    resolution_results: list[LatencyResult] = []
    resolution_probes: list[ResolutionProbe] = []
    for spec in selected:
        print(f"\n=== {spec.title} ({spec.checkpoint}) ===")
        model, processor, weights = load_detector(
            spec.checkpoint,
            weights=overrides.get(spec.key),
            device=device,
            dtype_name=dtype_name,
        )
        print(f"    weights: {weights.source} ({weights.describe()})")

        end_to_end = build_end_to_end_callable(
            model,
            processor,
            image,
            batch_size=settings.batch_size,
            input_size=settings.input_size,
            device=device,
            dtype_name=dtype_name,
            score_threshold=score_threshold,
            source_size=source_size,
        )
        model_only = build_model_only_callable(
            model,
            batch_size=settings.batch_size,
            input_size=settings.input_size,
            device=device,
            dtype_name=dtype_name,
        )
        landing = landing_check(
            model, processor, model_only=model_only, end_to_end=end_to_end
        )
        print(
            f"    loaded {landing['model_class']} / {landing['processor_class']}, "
            f"{landing['parameters_m']:.2f} M params, native input "
            f"{landing['native_input_size']}, logits {landing['logits_shape']}, "
            f"{landing['n_detections']} detections"
        )

        report: BenchmarkReport = run_benchmark(
            label=spec.title,
            end_to_end=end_to_end,
            model_only=model_only,
            settings=settings,
            device=device,
            dtype=dtype_name,
            clock_sampler=clock_sampler,
        )
        for result in report.results():
            p95 = "n/a" if result.p95_ms is None else f"{result.p95_ms:.2f}"
            print(
                f"    {result.label}: {result.statistic}={result.latency_ms:.2f} ms, "
                f"p95={p95} ms, {result.fps:.1f} FPS"
            )
        if report.peak_vram_mb is not None:
            print(f"    peak VRAM: {report.peak_vram_mb:.1f} MiB")

        # Control probes in BOTH directions. Halving both input dimensions cuts
        # the pixel count 4x and doubling them multiplies it by 4. If latency does
        # not follow downwards, the measurement is bound by kernel dispatch rather
        # than by the network; the upward probe is what shows the harness would
        # have noticed if it did - without it, a flat line proves nothing.
        control_sizes = (settings.input_size // 2, settings.input_size * 2)
        if report.model_only is None:
            # benchmark.separate_model_and_e2e is off, so there is no
            # full-resolution model-only row to compare a probe against.
            print("    skipping resolution control probes: no model-only baseline")
            control_sizes = ()
        for probe_size in control_sizes:
            probe = measure_at_resolution(
                model,
                size=probe_size,
                settings=settings,
                device=device,
                dtype_name=dtype_name,
                label=f"{spec.title} [model-only @ {probe_size}]",
                clock_sampler=clock_sampler,
            )
            resolution_results.append(probe)
            resolution_probes.append(
                ResolutionProbe(
                    title=spec.title,
                    full_size=settings.input_size,
                    # By ROLE, never by index: results()[0] stops being the
                    # model-only row the moment separate_model_and_e2e is off.
                    full_ms=report.model_only_result().latency_ms,
                    probe_size=probe_size,
                    probe_ms=probe.latency_ms,
                )
            )
            print(f"    control probe {probe_size}: {probe.latency_ms:.2f} ms")

        native = landing["native_input_size"]
        if native != settings.input_size and native not in control_sizes:
            resolution_results.append(
                measure_at_resolution(
                    model,
                    size=native,
                    settings=settings,
                    device=device,
                    dtype_name=dtype_name,
                    label=f"{spec.title} [model-only @ {native}, native preset]",
                    clock_sampler=clock_sampler,
                )
            )
            print(
                f"    native preset {native}: "
                f"{resolution_results[-1].latency_ms:.2f} ms, "
                f"{resolution_results[-1].fps:.1f} FPS"
            )

        entries.append(
            {
                "spec": spec,
                "landing": landing,
                "report": report,
                "weights": weights,
                "licence": fetch_licence_evidence(spec.checkpoint, api=hub_api),
            }
        )

        del model
        if device == "cuda":
            import torch

            torch.cuda.empty_cache()

    forbidden_scan = scan_for_forbidden_package(PROJECT_ROOT)
    print(
        f"\nforbidden-package scan: {forbidden_scan['files_scanned']} files read, "
        f"{len(forbidden_scan['matches'])} matches"
    )

    markdown = render_report(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        device_name=device_name,
        settings=settings,
        dtype_name=dtype_name,
        probe_image=probe_path.name,
        source_size=source_size,
        score_threshold=score_threshold,
        entries=entries,
        resolution_results=resolution_results,
        resolution_probes=resolution_probes,
        forbidden_scan=forbidden_scan,
    )
    out_path = paths.reports / REPORT_NAME
    out_path.write_text(markdown, encoding="utf-8", newline="\n")
    print(f"wrote {out_path}")

    # The report is written first so the evidence survives, then the run fails:
    # an ADR-005 violation must not be something a green exit status hides.
    if not forbidden_scan["clean"]:
        raise SystemExit(
            f"ADR-005 violated: the forbidden AGPL-3.0 package is named in "
            f"{len(forbidden_scan['matches'])} place(s); see {out_path}"
        )


if __name__ == "__main__":
    main()
