"""Write reports/error_analysis.md + .json and the EVAL-15 comparison grids.

Reads only files. No model is loaded, nothing runs on the GPU: the per-arm
detections are read from `results/predictions/<arm>_test.json`, which the eval
driver writes, and everything else is re-derived from the frozen Test split and
`configs/*.yaml`.

The pieces, and what happens when one is missing:

  EVAL-15  four-way comparison + grids   needs predictions for the two arms
  EVAL-16  hard-negative subset          needs predictions; subset is derived
  EVAL-17  scenario slices + targeting   needs the Test images (luminance) and
                                         the synthesis records for the delivered
                                         scenario mix
  EVAL-18  negative-result path          needs the headline number per arm

A missing input downgrades a section to an explicit "this input was not
supplied" line in the report. Nothing disappears quietly, because a section that
vanishes reads as a section that did not apply.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Same bootstrap as the other entry points in scripts/: the documented way to run
# these is `uv run python scripts/<name>.py`, which puts scripts/ on sys.path and
# not the repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.error_analysis import (
    DEFAULT_FIGURE_DIR,
    DEFAULT_PAYLOAD_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_SLICE_METRIC,
    SLICE_METRICS,
    ErrorAnalysisConfigError,
    ExposureConfound,
    build_analysis,
    default_predictions_path,
    exposure_confound,
    load_error_analysis_config,
    load_predictions,
    placeholder_compositions,
    read_records_jsonl,
    scenario_shares_from_records,
    write_report,
)
from src.evaluation.slices import image_mean_luminances, scenario_slices
from src.training.arms import ARMS, split_real_images
from src.training.data import CLASS_NAMES, load_coco_samples
from src.training.ingest import latest_checkpoint, training_curve
from src.training.metrics import build_coco_ground_truth

DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "results" / "colab" / "training_summary.json"
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"


class DriverError(RuntimeError):
    """Raised when an input this script needs is absent or unusable."""


def load_yaml(path: Path) -> dict[str, Any]:
    """UTF-8 forced everywhere (K-10)."""

    import yaml

    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise DriverError(f"Expected a mapping in {path}")
    return payload


def budget_inputs(training_config_path: Path) -> tuple[str, int, int]:
    """(reference arm, reference epochs, batch size) for the exposure arithmetic.

    Refuses to compute an exposure confound when `budget_alignment` is not
    `equal_steps`: under equal EPOCHS the arms see each real image the same
    number of times and the confound this reports would not exist, so producing
    the table anyway would be describing a run that did not happen.
    """

    config = load_yaml(training_config_path)
    alignment = str(config["budget_alignment"])
    if alignment != "equal_steps":
        raise DriverError(
            f"configs/training.yaml budget_alignment is {alignment!r}, not 'equal_steps'. "
            "The EVAL-18 exposure confound is a consequence of the equal-STEP budget; "
            "reporting it for another schedule would misdescribe the run."
        )
    run = config["run"]
    return (
        "real_only",
        int(run["num_train_epochs_real_only"]),
        int(run["per_device_train_batch_size"]),
    )


def read_exposure(summary_path: Path, training_config_path: Path) -> ExposureConfound:
    """Recompute real-image exposures, then check them against the run log.

    `equal_step_budget` is the source of truth (EVAL-18 says compute it, do not
    hardcode it), but the Colab run already recorded its own plan. If the two
    disagree, one of them is describing a different run and the report must not
    pick a side silently.
    """

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    counts = {
        arm: {
            "n_real_train": int(entry["n_real_train"]),
            "n_synthetic": int(entry.get("n_synthetic", 0)),
        }
        for arm, entry in summary["arms"].items()
    }
    reference_arm, reference_epochs, batch_size = budget_inputs(training_config_path)
    confound = exposure_confound(
        placeholder_compositions(counts),
        reference_arm=reference_arm,
        reference_epochs=reference_epochs,
        batch_size=batch_size,
    )

    recorded = summary.get("plan") or {}
    disagreements = {
        arm: (confound.exposures[arm], float(entry["real_image_exposures"]))
        for arm, entry in recorded.items()
        if arm in confound.exposures
        and abs(confound.exposures[arm] - float(entry["real_image_exposures"])) > 1e-6
    }
    if disagreements:
        raise DriverError(
            "Recomputed real-image exposures disagree with the ones the run recorded "
            f"{disagreements} (recomputed, recorded). One of them describes a different "
            "run; refusing to publish either."
        )
    return confound


def read_best_checkpoint_values(
    runs_root: Path, seed: int, metric: str
) -> dict[str, float]:
    """Each arm's BEST value of `metric` over its whole validation curve.

    Read from `trainer_state.json`, not from the run summary. Every arm here
    peaks early and decays for the rest of its schedule, and the summary records
    the eval that happened to be last; using it would report each arm at a point
    it was never selected at. `load_best_model_at_end` picks the peak, so the
    peak is what the arms are actually compared at, and EVAL-12 says re-derive it
    from the raw state rather than copy a printed number.
    """

    best: dict[str, float] = {}
    for arm in ARMS:
        checkpoint = latest_checkpoint(Path(runs_root) / arm / f"seed_{seed}")
        if checkpoint is None:
            continue
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        values = [
            float(point[metric]) for point in training_curve(state) if metric in point
        ]
        if values:
            best[arm] = max(values)
    return best


def read_headline_values(summary_path: Path, metric: str) -> dict[str, float]:
    """Fallback: the eval each arm recorded in the run summary.

    Not the peak — see `read_best_checkpoint_values`. Used only when the
    checkpoints are not on this machine, and the caller says which source it got.
    """

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    best: dict[str, float] = {}
    for record in summary.get("records", []):
        arm = str(record["arm"])
        value = record.get("eval_metrics", {}).get(metric)
        if value is None:
            continue
        best[arm] = max(best.get(arm, float("-inf")), float(value))
    if not best:
        raise DriverError(f"No arm in {summary_path} records a {metric!r} value")
    return best


def read_arm_predictions(
    directory: Path, arms: list[str]
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Load whatever prediction files exist. Missing arms are reported, not faked."""

    loaded: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for arm in arms:
        path = default_predictions_path(arm, directory)
        if not path.is_file():
            missing.append(f"{arm}: no {path}")
            continue
        loaded[arm] = load_predictions(path)
    return loaded, missing


def load_test_samples(
    annotations: Path, images_root: Path, manifest: Path
) -> list[Any]:
    """The frozen Test split: coco_all.json restricted to the manifest's `test`.

    There is no standalone test.json in this repo — the split is a manifest of
    file names over one COCO file (DATA-*), and materialising it here rather
    than trusting a derived copy is what keeps the Test set frozen. If the
    manifest names images the COCO does not carry, that is a broken split and
    this raises rather than analysing whatever survived the intersection.
    """

    names = split_real_images(Path(manifest)).get("test", ())
    if not names:
        raise DriverError(f"{manifest} declares no `test` split")
    samples = load_coco_samples(
        Path(annotations), Path(images_root), keep_names=list(names)
    )
    if len(samples) != len(names):
        missing = sorted(set(names) - {sample.image_path.name for sample in samples})
        raise DriverError(
            f"{manifest} names {len(names)} Test images but only {len(samples)} are in "
            f"{annotations}; missing e.g. {missing[:5]}"
        )
    # `load_coco_samples` strips the directory off `file_name`, so `images_root`
    # has to be the folder the PNGs are actually in. Getting that wrong produces
    # a FileNotFoundError several hundred images into the luminance pass, after
    # a couple of minutes of work.
    if not samples[0].image_path.is_file():
        raise DriverError(
            f"{samples[0].image_path} does not exist. --images-root must point at the "
            "directory holding the PNGs themselves, not its parent."
        )
    return samples


def build_slices(samples: list[Any]) -> dict[str, frozenset[int]]:
    """The three EVAL-17 slices. Reads every Test image once, for luminance.

    Thresholds come from `configs/evaluation.yaml` via `slices.default_slice_config`,
    so this function passes none of its own.
    """

    luminance = image_mean_luminances(samples)
    return scenario_slices(samples, luminance_by_image=luminance)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-annotations",
        type=Path,
        default=None,
        help="COCO file the split manifest indexes (default: interim/coco_all.json)",
    )
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="frozen split (default: splits/split_manifest.json)",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="directory holding <arm>_test.json (default: results/predictions)",
    )
    parser.add_argument("--records", type=Path, default=None, help="synthesis records.jsonl")
    parser.add_argument(
        "--synthetic-annotations",
        type=Path,
        default=None,
        help="the delivered synthetic COCO, used to restrict the scenario mix",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="where <arm>/seed_<n>/checkpoint-* live (default: paths.yaml runs)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--headline-json",
        type=Path,
        default=None,
        help="JSON mapping arm -> headline value; overrides the run summary",
    )
    parser.add_argument("--headline-metric", type=str, default="eval_map")
    parser.add_argument(
        "--slice-metric", type=str, default=DEFAULT_SLICE_METRIC, choices=SLICE_METRICS
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD_PATH)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="skip the grids; the counts table and the JSON spine are still written",
    )
    parser.add_argument("--no-slices", action="store_true", help="skip EVAL-17")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = load_project_paths()
    settings = load_error_analysis_config()

    annotations = args.test_annotations or (paths.interim / "coco_all.json")
    images_root = args.images_root or (paths.hardhat_raw / "images")
    manifest = args.split_manifest or (paths.splits / "split_manifest.json")
    for required in (annotations, manifest):
        if not Path(required).is_file():
            print(f"FATAL: missing {required}")
            return 2

    samples = load_test_samples(Path(annotations), Path(images_root), Path(manifest))
    ground_truth = build_coco_ground_truth(samples, CLASS_NAMES)
    print(
        f"Test split: {len(samples)} images, "
        f"{len(ground_truth['annotations'])} instances"
    )

    predictions, missing = read_arm_predictions(
        args.predictions_dir or (PROJECT_ROOT / "results" / "predictions"), list(ARMS)
    )
    for line in missing:
        print("MISSING:", line)
    for arm in (settings.baseline_arm, settings.best_arm):
        if arm not in predictions:
            print(
                f"FATAL: error_analysis.compare_arms needs {arm!r}; run the eval driver "
                "first or pass --predictions-dir"
            )
            return 2

    image_paths = None
    if not args.no_figures:
        image_paths = {int(sample.image_id): Path(sample.image_path) for sample in samples}

    slices = None
    budget_shares = None
    if not args.no_slices:
        slices = build_slices(samples)
        records_path = args.records
        if records_path is not None and Path(records_path).is_file():
            records = read_records_jsonl(Path(records_path))
            keep = None
            if args.synthetic_annotations and Path(args.synthetic_annotations).is_file():
                payload = json.loads(
                    Path(args.synthetic_annotations).read_text(encoding="utf-8")
                )
                keep = [image["file_name"] for image in payload["images"]]
            budget_shares = scenario_shares_from_records(records, keep_names=keep)
        else:
            print(
                "MISSING: no --records file, so the EVAL-17 targeting verdict is skipped"
            )

    headline_values = None
    if args.headline_json and Path(args.headline_json).is_file():
        headline_values = {
            str(arm): float(value)
            for arm, value in json.loads(
                Path(args.headline_json).read_text(encoding="utf-8")
            ).items()
        }
        print(f"headline: {args.headline_metric} from {args.headline_json}")
    else:
        runs_root = args.runs_root or paths.runs
        headline_values = read_best_checkpoint_values(
            runs_root, args.seed, args.headline_metric
        ) or None
        if headline_values:
            print(
                f"headline: best {args.headline_metric} per arm, re-read from "
                f"trainer_state.json under {runs_root}"
            )
        elif Path(args.summary).is_file():
            headline_values = read_headline_values(
                Path(args.summary), args.headline_metric
            )
            print(
                f"headline: no checkpoints under {runs_root}; falling back to the "
                f"{args.summary} records, which are NOT each arm's peak"
            )
        else:
            print(
                f"MISSING: no checkpoints and no run summary at {args.summary}; "
                "EVAL-18 has no headline table"
            )

    exposure = None
    if Path(args.summary).is_file():
        try:
            exposure = read_exposure(Path(args.summary), Path(args.training_config))
        except (DriverError, ErrorAnalysisConfigError) as error:
            print("MISSING:", error)

    analysis = build_analysis(
        ground_truth,
        predictions,
        analysis_config=settings,
        slices=slices,
        slice_metric=args.slice_metric,
        budget_shares=budget_shares,
        headline_values=headline_values,
        headline_metric=args.headline_metric,
        exposure=exposure,
        image_paths=image_paths,
        figure_dir=args.figure_dir,
    )
    report_path, payload_path = write_report(
        analysis, report_path=args.report, payload_path=args.payload
    )

    print(f"wrote {report_path}")
    print(f"wrote {payload_path}")
    for category, count in analysis.counts.items():
        figure = analysis.figures.get(category)
        size = f"  {Path(PROJECT_ROOT / figure).stat().st_size / 1024:.0f} KiB" if figure else ""
        print(f"  {category:22s} {count:6d}{size}")
    if analysis.targeting is not None:
        print(f"  targeting verdict: {analysis.targeting.verdict}")
    if analysis.negative_result is not None:
        print(f"  synthetic beats baseline: {not analysis.negative_result.is_negative}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
