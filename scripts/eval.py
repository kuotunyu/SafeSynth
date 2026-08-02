"""M18 driver: score every trained arm on the FROZEN Test split (EVAL-05..EVAL-14).

Usage:

    uv run python -m scripts.eval
    uv run python -m scripts.eval --runs-root D:/sdg-data/02-safesynth/runs

What this file is and is not. Every metric, bucket, bootstrap, CSV column and
leak assertion already exists in `src/evaluation/detection.py` and
`src/evaluation/slices.py`, both of which are committed and covered. This driver
imports them and adds only the four things they cannot know about:

  1. WHICH weights to score. The layout is `<runs-root>/<arm>/seed_<n>/
     checkpoint-<step>/`, and picking the wrong checkpoint for one arm is a
     difference between arms that no table would show. See `resolve_checkpoint`.
  2. THAT the weights never saw Test (EVAL-14), asserted before anything loads.
  3. The coordinate mapping (EVAL-07). Predictions come back at the resolution
     the image processor resized to; the Test split is 416x416, 416x415, 415x416
     AND 415x415 (DATA-25), so `scale_x` and `scale_y` differ per image and a
     single scalar is wrong for three of those four shapes without raising.
  4. `real_image_exposures` (TRAIN-07). The arms share an optimizer-step budget,
     so the synthetic arms saw each real image about half as often as real_only
     (~25 versus ~50 times). That confound belongs in the main table, not a
     footnote, and it is re-derived here from `src.training.arms.equal_step_budget`
     and cross-checked against each arm's recorded `total_steps`.

Everything except one call - `load_model`, the seam that reads weights - runs on
CPU against plain arguments, so the whole driver is exercisable without a GPU.

Numbers come from configs/evaluation.yaml and configs/training.yaml. This file
contains none.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from scripts.profile_test_set import load_test_samples
from src.data.integrity import assert_test_untouched
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.benchmark import resolve_device, resolve_dtype_name, torch_dtype
from src.evaluation.detection import (
    UNDEFINED,
    DetectionMetrics,
    HardNegativeReport,
    MetricRow,
    assert_no_test_images_in_training_list,
    attach_bootstrap_cis,
    default_detection_metrics_path,
    detection_metric_rows,
    detections_for_evaluation,
    evaluate_detection_metrics,
    hard_negative_false_positives_per_image,
    load_evaluation_config,
    write_detection_metrics_csv,
)
from src.evaluation.slices import (
    EVALUATION_CONFIG,
    image_mean_luminances,
    load_slice_config,
    scenario_slices,
)
from src.training.arms import (
    ARMS,
    ArmComposition,
    digest_names,
    equal_step_budget,
    split_real_images,
)
from src.training.config import load_training_config as load_resolved_training_config
from src.training.data import CLASS_NAMES, Sample
from src.training.ingest import ColabResultsError, latest_checkpoint, load_run_records
from src.training.metrics import build_coco_ground_truth, predictions_to_coco

SPLIT_NAME = "test"
REPORT_NAME = "detection_main_table.md"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"
COLAB_RESULTS = PROJECT_ROOT / "results" / "colab"
H4_GATE_PATH = PROJECT_ROOT / "reports" / "h4_artifact_gate_m13.json"

SEED_DIR_PREFIX = "seed_"
CHECKPOINTS_JSON = "checkpoints.json"
TRAINER_STATE_JSON = "trainer_state.json"
BEST_CHECKPOINT_KEY = "best_model_checkpoint"

# The arm whose epoch count defines the shared step budget (TRAIN-07). It is the
# smallest training set, so every other arm gets FEWER epochs, never more.
REFERENCE_ARM = "real_only"

# Evaluation runs in fp32 by default. A headline metric must not depend on which
# reduced precision happened to be available on the machine that produced it;
# the latency numbers are where dtype is a variable (configs/evaluation.yaml
# benchmark.dtype), and this is not that measurement.
DEFAULT_DTYPE = "float32"

# mAP integrates over every confidence, so nothing is filtered out before
# COCOeval. The compliance operating point is a separate number and
# `hard_negative_false_positives_per_image` reads it from the config itself.
MAP_SCORE_THRESHOLD = 0.0


class EvalDriverError(RuntimeError):
    """Raised when the driver cannot produce a trustworthy number."""


class TrainingListUnavailableError(EvalDriverError):
    """Raised when EVAL-14 cannot be checked, which is not the same as passing."""


class CheckpointSource(StrEnum):
    """How a checkpoint was chosen, in the resolution order that produced it."""

    CHECKPOINTS_JSON = "checkpoints.json:best_model_checkpoint"
    TRAINER_STATE = "trainer_state.json:best_model_checkpoint"
    HIGHEST_STEP = "highest-step FALLBACK"


@dataclass(frozen=True)
class CheckpointChoice:
    """The weights picked for one (arm, seed), and the trail that picked them."""

    path: Path | None
    source: CheckpointSource | None
    notes: tuple[str, ...] = ()

    @property
    def is_fallback(self) -> bool:
        return self.source is CheckpointSource.HIGHEST_STEP


@dataclass(frozen=True)
class ArmWeights:
    arm: str
    seed: int
    seed_dir: Path
    choice: CheckpointChoice


@dataclass(frozen=True)
class ArmResult:
    """One scored arm: everything the CSV and the report need, already computed."""

    arm: str
    seed: int
    checkpoint: Path
    source: CheckpointSource
    resolution_notes: tuple[str, ...]
    metrics: DetectionMetrics
    rows: tuple[MetricRow, ...]
    hard_negative: HardNegativeReport | None
    slice_metrics: Mapping[str, DetectionMetrics]
    exposures: float | None
    total_steps: int | None


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def read_json_mapping(path: Path) -> dict[str, Any] | None:
    """Parse a JSON object, or None when the file is absent or not a JSON object.

    None rather than an exception because every caller here is deciding whether
    to fall through to the next resolution step, and an unreadable
    `checkpoints.json` on one arm must not abort the other three.
    """

    if not Path(path).is_file():
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def checkpoint_name_from_recorded_path(recorded: Any) -> str | None:
    """The last component of a `best_model_checkpoint` recorded on another machine.

    Colab wrote an ABSOLUTE POSIX path - `/content/safesynth/runs/real_only/
    seed_1337/checkpoint-9156` - which does not exist on this machine and never
    will. Trusting it is the obvious bug here and it would fail on every arm at
    once, so the stored string is used for its BASENAME only and re-rooted
    against the local seed directory by the caller.

    Backslashes are normalised first: `Path` on Windows would treat a POSIX
    string as one long file name, and `PurePosixPath` would treat a Windows
    string the same way, so neither type alone reads both spellings.
    """

    if not isinstance(recorded, str):
        return None
    text = recorded.strip().replace("\\", "/")
    if not text:
        return None
    name = PurePosixPath(text).name
    return name or None


def _checkpoint_from_record(
    payload: Mapping[str, Any], seed_dir: Path, origin: str
) -> tuple[Path | None, str | None]:
    """Resolve one record's `best_model_checkpoint` against the local seed dir.

    Returns `(path, note)`. Exactly one is ever set: a note explains why this
    resolution step could not be used, so the caller can fall through and the
    report can say what happened instead of showing an unexplained fallback.
    """

    name = checkpoint_name_from_recorded_path(payload.get(BEST_CHECKPOINT_KEY))
    if name is None:
        return None, f"{origin} records no {BEST_CHECKPOINT_KEY}"
    candidate = Path(seed_dir) / name
    if candidate.is_dir():
        return candidate, None
    return None, f"{origin} names {name!r}, which is not present under {Path(seed_dir).name}/"


# spec: EVAL-11
def resolve_checkpoint(arm_dir: Path, seed_dir: Path) -> CheckpointChoice:
    """Pick the weights to score for one (arm, seed), in a fixed, stated order.

    a. `<arm>/checkpoints.json` -> `best_model_checkpoint`. Written by
       `src.training.ingest.package_arm_outputs`, which post-dates the archive
       already downloaded, so it is frequently absent.
    b. otherwise the newest `checkpoint-*/trainer_state.json`, which HF Trainer
       writes on every save and which also records `best_model_checkpoint`. The
       "newest" is `latest_checkpoint()` from `src.training.ingest` - highest
       step, not newest mtime, because copying a tree back from Drive rewrites
       timestamps.
    c. otherwise the highest-numbered checkpoint, WHICH IS SAID OUT LOUD. It is
       not the same weights as (a) or (b) whenever the best epoch was not the
       last, and an arm silently scored on its last checkpoint while the others
       used their best is a between-arm difference nobody would see.
    d. nothing found -> `path is None`; the caller reports it and carries on with
       the other arms.
    """

    arm_dir, seed_dir = Path(arm_dir), Path(seed_dir)
    notes: list[str] = []

    manifest_path = arm_dir / CHECKPOINTS_JSON
    manifest = read_json_mapping(manifest_path)
    if manifest is None:
        notes.append(f"no readable {CHECKPOINTS_JSON} at the arm level")
    else:
        path, note = _checkpoint_from_record(manifest, seed_dir, CHECKPOINTS_JSON)
        if path is not None:
            return CheckpointChoice(path, CheckpointSource.CHECKPOINTS_JSON, tuple(notes))
        notes.append(str(note))

    newest = latest_checkpoint(seed_dir)
    if newest is None:
        notes.append(f"no checkpoint-* directory under {seed_dir}")
        return CheckpointChoice(None, None, tuple(notes))

    state = read_json_mapping(newest / TRAINER_STATE_JSON)
    if state is None:
        notes.append(f"{newest.name} carries no readable {TRAINER_STATE_JSON}")
    else:
        origin = f"{newest.name}/{TRAINER_STATE_JSON}"
        path, note = _checkpoint_from_record(state, seed_dir, origin)
        if path is not None:
            return CheckpointChoice(path, CheckpointSource.TRAINER_STATE, tuple(notes))
        notes.append(str(note))

    notes.append(
        f"fell back to {newest.name}, the highest step - this is NOT necessarily the "
        f"best epoch, and the other arms may have been resolved differently"
    )
    return CheckpointChoice(newest, CheckpointSource.HIGHEST_STEP, tuple(notes))


def seed_directories(arm_dir: Path) -> tuple[tuple[tuple[int, Path], ...], tuple[str, ...]]:
    """`(seed, directory)` pairs under one arm, plus the names that were not seeds.

    Returned rather than skipped: `seed_best` or `seed_1337_old` sitting beside
    `seed_1337` means somebody has a second copy of a run, and quietly ignoring
    it would let the report claim a coverage it does not have.
    """

    pairs: list[tuple[int, Path]] = []
    skipped: list[str] = []
    for path in sorted(Path(arm_dir).glob(f"{SEED_DIR_PREFIX}*")):
        if not path.is_dir():
            skipped.append(path.name)
            continue
        suffix = path.name[len(SEED_DIR_PREFIX) :]
        if not suffix.isdigit():
            skipped.append(path.name)
            continue
        pairs.append((int(suffix), path))
    pairs.sort()
    return tuple(pairs), tuple(skipped)


def discover_arm_weights(
    runs_root: Path, *, arms: Sequence[str] = ARMS
) -> tuple[tuple[ArmWeights, ...], tuple[str, ...]]:
    """Locate weights for every arm, collecting a problem line for each gap.

    A missing arm is a problem, not an abort (M18): the three arms that did come
    back are still worth scoring, and the caller exits non-zero afterwards.
    """

    runs_root = Path(runs_root)
    found: list[ArmWeights] = []
    problems: list[str] = []
    for arm in arms:
        arm_dir = runs_root / arm
        if not arm_dir.is_dir():
            problems.append(f"{arm}: no directory at {arm_dir}")
            continue
        pairs, skipped = seed_directories(arm_dir)
        for name in skipped:
            problems.append(
                f"{arm}: ignoring {name!r} - a {SEED_DIR_PREFIX}<n> directory needs an "
                f"integer seed"
            )
        if not pairs:
            problems.append(f"{arm}: no {SEED_DIR_PREFIX}<n> directory under {arm_dir}")
            continue
        for seed, seed_dir in pairs:
            choice = resolve_checkpoint(arm_dir, seed_dir)
            if choice.path is None:
                trail = "; ".join(choice.notes)
                problems.append(f"{arm} seed {seed}: no checkpoint found ({trail})")
                continue
            found.append(ArmWeights(arm=arm, seed=seed, seed_dir=seed_dir, choice=choice))
    return tuple(found), tuple(problems)


# ---------------------------------------------------------------------------
# EVAL-14 - leak self-check, before any weights are read
# ---------------------------------------------------------------------------


def training_image_names(
    record: Mapping[str, Any], arm: str, *, real_train_names: Sequence[str]
) -> tuple[str, ...]:
    """The real training list one arm used - reconstructed, and PROVEN by digest.

    `run_record.json` stores `composition.real_train_digest` and the counts, not
    the names, so the list cannot simply be read out of it. What it can do is
    prove a candidate list: `digest_names` is an order-independent SHA-256 of the
    sorted names, so a digest equal to the frozen split's train digest means the
    arm trained on exactly those images and nothing else.

    When the digest is absent or disagrees this raises instead of returning an
    unverified list. A leak check that quietly runs on the wrong list is worse
    than no leak check, because it prints PASS.
    """

    composition = record.get("composition")
    if not isinstance(composition, Mapping):
        raise TrainingListUnavailableError(
            f"{arm}: run_record.json has no `composition` block, so its training image "
            f"list cannot be established and EVAL-14 cannot be checked for this arm"
        )
    recorded = str(composition.get("real_train_digest", ""))
    if not recorded:
        raise TrainingListUnavailableError(
            f"{arm}: composition records no real_train_digest, so there is nothing to "
            f"prove its training list with. EVAL-14 is UNCHECKED for this arm"
        )
    expected = digest_names(real_train_names)
    if recorded != expected:
        raise TrainingListUnavailableError(
            f"{arm}: real_train_digest {recorded[:12]}... does not match the frozen "
            f"train split digest {expected[:12]}... The arm trained on a DIFFERENT set "
            f"of real images than splits/split_manifest.json declares, so its training "
            f"list is unknown and may contain Test images"
        )
    return tuple(real_train_names)


# spec: EVAL-14
def leak_self_check(
    records: Mapping[str, Mapping[str, Any]],
    *,
    real_train_names: Sequence[str],
    test_names: Sequence[str],
    arms: Sequence[str],
) -> tuple[str, ...]:
    """Prove no Test image reached training, and return the evidence as text.

    Three steps, in this order:

      1. `assert_test_untouched()` - every Test image still hashes to what
         splits/test_blocklist.json froze (DATA-20).
      2. the frozen manifest's own train and test name lists are disjoint. This
         is the fact everything below is measured against, so it is asserted
         rather than assumed.
      3. each arm's training list, reconstructed and digest-proven above, is
         checked against the Test names with detection.py's own comparison,
         which normalises separators and case before comparing (a leak hidden by
         `data\\test\\A.PNG` versus `data/test/a.png` would otherwise report
         PASS).

    Raises rather than returning a verdict. Scoring a checkpoint that saw Test
    produces a number that looks fine and means nothing.
    """

    assert_test_untouched()
    evidence = [
        f"`assert_test_untouched()` passed over {len(test_names)} frozen Test images."
    ]

    assert_no_test_images_in_training_list(real_train_names, test_names)
    evidence.append(
        f"splits/split_manifest.json: {len(real_train_names)} train names and "
        f"{len(test_names)} test names are disjoint."
    )

    missing = [arm for arm in arms if arm not in records]
    if missing:
        raise TrainingListUnavailableError(
            f"No run_record.json for {missing}, so what those arms trained on is "
            f"unknown. EVAL-14 cannot be checked for them and this driver will not "
            f"score weights whose training data it cannot account for"
        )

    for arm in arms:
        names = training_image_names(records[arm], arm, real_train_names=real_train_names)
        assert_no_test_images_in_training_list(names, test_names)
        digest = digest_names(names)
        evidence.append(
            f"`{arm}`: {len(names)} real training images (digest `{digest[:12]}...` "
            f"matches the frozen train split), 0 of them in Test."
        )
    evidence.append(
        "Not covered by this check: whether a SYNTHETIC image was derived from a Test "
        "image. run_record.json carries only the synthetic COUNT, so that guarantee "
        "rests on Phase 1 - the generator read the train split only, and every sample's "
        "source image is recorded in the synthetic `records.jsonl`."
    )
    return tuple(evidence)


# ---------------------------------------------------------------------------
# TRAIN-07 - the shared step budget, and what it cost in real-image exposure
# ---------------------------------------------------------------------------


def reconstruct_compositions(
    records: Mapping[str, Mapping[str, Any]],
    *,
    real_train_names: Sequence[str],
    real_val_names: Sequence[str],
) -> dict[str, ArmComposition]:
    """Rebuild each arm's `ArmComposition` from its run record.

    `equal_step_budget` reads exactly two things off a composition:
    `n_train_images` and `len(real_train)`. The run record stores how MANY
    synthetic images an arm trained on but not which, so the synthetic tuple is
    filled with placeholder identifiers of the recorded length - their cardinality
    is the whole of their contribution, and `verify_step_budget` then proves the
    reconstruction by reproducing each arm's recorded `total_steps` exactly.

    Re-deriving the arithmetic here instead would put a second copy of the
    step-budget formula in the repository, free to drift from the one that ran.
    """

    compositions: dict[str, ArmComposition] = {}
    for arm, record in records.items():
        composition = record.get("composition")
        if not isinstance(composition, Mapping):
            raise EvalDriverError(f"{arm}: run_record.json has no `composition` block")
        n_synthetic = int(composition.get("n_synthetic", 0))
        compositions[arm] = ArmComposition(
            arm=arm,
            real_train=tuple(real_train_names),
            real_val=tuple(real_val_names),
            synthetic=tuple(f"{arm}-synthetic-{index}" for index in range(n_synthetic)),
            augmentation_profile=str(composition.get("augmentation_profile", "")),
            real_train_digest=str(composition.get("real_train_digest", "")),
        )
    return compositions


# spec: TRAIN-07
def step_budget_plan(
    compositions: Mapping[str, ArmComposition], *, training_config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """`equal_step_budget` driven by configs/training.yaml, reference arm included."""

    if REFERENCE_ARM not in compositions:
        raise EvalDriverError(
            f"The step budget is defined relative to {REFERENCE_ARM!r}, which has no run "
            f"record; real_image_exposures cannot be derived for any arm without it"
        )
    run = training_config["run"]
    return equal_step_budget(
        compositions,
        reference_arm=REFERENCE_ARM,
        reference_epochs=int(run["num_train_epochs_real_only"]),
        batch_size=int(run["per_device_train_batch_size"]),
    )


# spec: TRAIN-07
def verify_step_budget(
    plan: Mapping[str, Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]]
) -> tuple[str, ...]:
    """Prove the reconstructed plan is the plan that actually ran.

    Each arm recorded the `total_steps` it was given. If the reconstruction
    reproduces that number, the `real_image_exposures` beside it describes the
    real run; if it does not, the exposure figure is fiction and the report has
    to say so rather than print it.
    """

    problems: list[str] = []
    for arm, entry in sorted(plan.items()):
        recorded = records.get(arm, {}).get("total_steps")
        if recorded is None:
            problems.append(f"{arm}: run_record.json records no total_steps to check against")
            continue
        if int(recorded) != int(entry["total_steps"]):
            problems.append(
                f"{arm}: reconstructed step budget {entry['total_steps']} does not match "
                f"the recorded {int(recorded)}, so real_image_exposures "
                f"{entry['real_image_exposures']} does not describe the run that happened"
            )
    return tuple(problems)


# ---------------------------------------------------------------------------
# Inference - the only part that needs weights
# ---------------------------------------------------------------------------


def _size_field(size: Any, name: str) -> float | None:
    """Read one field off a processor `size`, which is a dict OR a SizeDict."""

    if isinstance(size, Mapping):
        value = size.get(name)
    else:
        value = getattr(size, name, None)
    return None if value is None else float(value)


# spec: EVAL-07
def processor_evaluated_size(size: Any) -> tuple[float, float]:
    """`(width, height)` the image processor resizes every image to.

    Read off the processor rather than taken from a config key, because the
    number that has to be divided out again is whatever the processor actually
    did; a constant can disagree with it and nothing would raise.

    Width and height are returned separately - `rescale_detections_to_original`
    scales the two axes independently, and collapsing them to one number would
    hide a non-square processor entirely.

    A `shortest_edge` processor is REFUSED rather than approximated: that mode
    preserves aspect ratio, so each image leaves preprocessing at its own size
    and there is no single `evaluated_size` to undo.
    """

    height = _size_field(size, "height")
    width = _size_field(size, "width")
    if height is None or width is None:
        raise EvalDriverError(
            f"Image processor size {size!r} does not declare both `height` and `width`. "
            f"An aspect-preserving (shortest_edge) processor resizes every image "
            f"differently, so predictions cannot be mapped back with one factor pair"
        )
    if width <= 0 or height <= 0:
        raise EvalDriverError(f"Image processor size {size!r} is not positive")
    return width, height


def run_inference(
    samples: Sequence[Sample],
    *,
    model: Any,
    processor: Any,
    evaluated_size: tuple[float, float],
    device: str,
    batch_size: int,
    score_threshold: float = MAP_SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Detections for every sample, in the coordinate space the model ran in.

    `target_sizes` is `(height, width)` per image, which is what transformers
    expects, and every row is the SAME pair here on purpose: the model saw one
    fixed resolution, so post-processing scales into that space and the per-image
    mapping back to annotation coordinates happens afterwards, in
    `detections_for_evaluation`. Doing it in one step would mean passing each
    image's own size here, which reads identically and skips the EVAL-07 config
    switch entirely.
    """

    import torch

    if batch_size <= 0:
        raise EvalDriverError(f"batch_size must be positive, got {batch_size}")
    evaluated_width, evaluated_height = (float(value) for value in evaluated_size)

    detections: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        chunk = list(samples[start : start + batch_size])
        images = []
        for sample in chunk:
            with Image.open(sample.image_path) as handle:
                images.append(handle.convert("RGB"))
        encoded = processor(images=images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values)
        target_sizes = torch.tensor(
            [[evaluated_height, evaluated_width]] * len(chunk), dtype=torch.float32
        )
        processed = processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )
        detections.extend(
            predictions_to_coco(processed, [int(sample.image_id) for sample in chunk])
        )
    return detections


def default_load_model(
    checkpoint: Path, *, processor_source: str, device: str, dtype_name: str
) -> tuple[Any, Any]:
    """Weights from the fine-tuned checkpoint, image processor from the base repo.

    THE ONLY FUNCTION IN THIS FILE THAT NEEDS A GPU OR A DOWNLOAD, which is why
    it is a parameter of `main` and never called by the tests.

    It deliberately does not call `src.training.run.load_model_and_processor`,
    which reads BOTH halves from one path. `src/training/run.py` constructs the
    Trainer without `processing_class`, so HF never wrote a
    preprocessor_config.json into any checkpoint directory and
    `AutoImageProcessor.from_pretrained(<checkpoint>)` would raise on every arm.
    The processor therefore comes from `processor_source` - configs/training.yaml
    `model.checkpoint`, the repo these runs fine-tuned - and only the weights
    come from the local directory. `num_labels` / `ignore_mismatched_sizes` are
    likewise absent: those exist to rebuild an 80-class head as a 3-class one,
    and this checkpoint already has the 3-class head.
    """

    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(processor_source)
    model = AutoModelForObjectDetection.from_pretrained(
        str(checkpoint), dtype=torch_dtype(dtype_name)
    )
    model.to(device)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Scoring one arm
# ---------------------------------------------------------------------------


def subset_by_images(
    ground_truth: Mapping[str, Any],
    detections: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int] | frozenset[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Restrict a COCO payload and its detections to `image_ids`, ids unchanged.

    Ids are preserved (unlike the bootstrap resampler, which must renumber
    because it repeats images) so a slice row in the CSV can be joined back to
    the full-split row for the same image.
    """

    keep = {int(value) for value in image_ids}
    subset = {
        "images": [image for image in ground_truth["images"] if int(image["id"]) in keep],
        "annotations": [
            annotation
            for annotation in ground_truth["annotations"]
            if int(annotation["image_id"]) in keep
        ],
        "categories": list(ground_truth["categories"]),
    }
    kept_detections = [
        dict(detection) for detection in detections if int(detection["image_id"]) in keep
    ]
    return subset, kept_detections


# spec: EVAL-05, EVAL-16
def hard_negative_image_ids(
    ground_truth: Mapping[str, Any], *, config: Mapping[str, Any]
) -> tuple[int, ...]:
    """Test images carrying no primary-class ground truth at all.

    These are the only images on which "false positives per image" is a
    meaningful number: a detection there cannot be a missed match, so every box
    above the operating point is a false alarm. `person`-only images count as
    hard negatives for this purpose because `person` is not a primary class -
    the same reason it carries no load anywhere else (ADR-003).
    """

    primary = set(config["metrics"]["primary_classes"])
    names = {int(c["id"]): str(c["name"]) for c in ground_truth["categories"]}
    with_primary = {
        int(annotation["image_id"])
        for annotation in ground_truth["annotations"]
        if names[int(annotation["category_id"])] in primary
        and int(annotation.get("iscrowd", 0)) == 0
    }
    return tuple(
        sorted(
            int(image["id"])
            for image in ground_truth["images"]
            if int(image["id"]) not in with_primary
        )
    )


def driver_metric_rows(
    *,
    arm: str,
    seed: int,
    n_images: int,
    exposures: float | None,
    total_steps: int | None,
    hard_negative: HardNegativeReport | None,
) -> list[MetricRow]:
    """The rows detection.py cannot produce because they are not detection metrics.

    `real_image_exposures` is here rather than in a footnote because the arms
    share a step budget and therefore do NOT see each real image equally often
    (TRAIN-07); any table that omits it invites the reader to attribute a gap to
    the data when it may be the exposure.
    """

    rows: list[MetricRow] = []
    if exposures is not None:
        rows.append(
            MetricRow(
                arm=arm,
                seed=seed,
                split=SPLIT_NAME,
                metric="real_image_exposures",
                value=float(exposures),
                n_images=n_images,
                notes=(
                    "TRAIN-07 equal-step budget: times each real training image was "
                    "seen. Not a Test-set quantity"
                ),
            )
        )
    if total_steps is not None:
        rows.append(
            MetricRow(
                arm=arm,
                seed=seed,
                split=SPLIT_NAME,
                metric="total_steps",
                value=float(total_steps),
                n_images=n_images,
                notes="optimizer steps, from run_record.json",
            )
        )
    if hard_negative is not None:
        rows.append(
            MetricRow(
                arm=arm,
                seed=seed,
                split=SPLIT_NAME,
                metric="hard_negative_fp_per_image",
                value=float(hard_negative.false_positives_per_image),
                n_instances=hard_negative.n_false_positives,
                n_images=hard_negative.n_images,
                notes=f"score_threshold={hard_negative.score_threshold:g}",
            )
        )
    return rows


def evaluate_arm(
    weights: ArmWeights,
    *,
    samples: Sequence[Sample],
    ground_truth: Mapping[str, Any],
    slices: Mapping[str, frozenset[int]],
    hard_negative_ids: Sequence[int],
    config: Mapping[str, Any],
    load_model: Callable[..., tuple[Any, Any]],
    processor_source: str,
    device: str,
    dtype_name: str,
    batch_size: int,
    bootstrap_resamples: int,
    bootstrap_workers: int | None,
    bootstrap_seed: int,
    exposures: float | None,
    total_steps: int | None,
    predictions_path: Path | None = None,
) -> ArmResult:
    """Score one arm on the frozen Test split, in annotation coordinates.

    The order matters. Predictions come out of the model at the processor's own
    resolution; they are mapped back to each image's OWN width and height first,
    and only then handed to the metrics. Evaluating the other way round raises
    nothing at all - IoU simply collapses and every area is inflated by about
    2.37x, which moves most `small` objects into `medium` and makes AP_small a
    measurement of something else (EVAL-07).
    """

    if weights.choice.path is None or weights.choice.source is None:
        raise EvalDriverError(f"{weights.arm}: cannot evaluate without a checkpoint")

    model, processor = load_model(
        weights.choice.path,
        processor_source=processor_source,
        device=device,
        dtype_name=dtype_name,
    )
    evaluated_size = processor_evaluated_size(getattr(processor, "size", None))
    raw = run_inference(
        samples,
        model=model,
        processor=processor,
        evaluated_size=evaluated_size,
        device=device,
        batch_size=batch_size,
    )
    detections = detections_for_evaluation(
        raw,
        evaluated_size=evaluated_size,
        original_sizes={
            int(sample.image_id): (float(sample.width), float(sample.height))
            for sample in samples
        },
        config=config,
    )
    if predictions_path is not None:
        atomic_write_json_value(predictions_path, detections, compact=True)

    metrics = evaluate_detection_metrics(ground_truth, detections, config=config)
    rows = detection_metric_rows(
        metrics, arm=weights.arm, seed=weights.seed, split=SPLIT_NAME
    )
    if bootstrap_resamples > 0:
        rows = attach_bootstrap_cis(
            rows,
            ground_truth,
            detections,
            seed=bootstrap_seed,
            config=config,
            resamples=bootstrap_resamples,
            workers=bootstrap_workers,
        )

    hard_negative = None
    if hard_negative_ids:
        hard_negative = hard_negative_false_positives_per_image(
            ground_truth, detections, hard_negative_ids, config=config
        )
    rows = list(rows) + driver_metric_rows(
        arm=weights.arm,
        seed=weights.seed,
        n_images=metrics.n_images,
        exposures=exposures,
        total_steps=total_steps,
        hard_negative=hard_negative,
    )

    slice_metrics: dict[str, DetectionMetrics] = {}
    for name in sorted(slices):
        image_ids = slices[name]
        if not image_ids:
            continue
        slice_gt, slice_detections = subset_by_images(ground_truth, detections, image_ids)
        sliced = evaluate_detection_metrics(slice_gt, slice_detections, config=config)
        slice_metrics[name] = sliced
        # The slice travels in the `split` column, so every slice table in the
        # report re-aggregates out of the same long-format file (EVAL-12).
        rows += detection_metric_rows(
            sliced, arm=weights.arm, seed=weights.seed, split=f"{SPLIT_NAME}/{name}"
        )

    return ArmResult(
        arm=weights.arm,
        seed=weights.seed,
        checkpoint=weights.choice.path,
        source=weights.choice.source,
        resolution_notes=weights.choice.notes,
        metrics=metrics,
        rows=tuple(rows),
        hard_negative=hard_negative,
        slice_metrics=slice_metrics,
        exposures=exposures,
        total_steps=total_steps,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def row_by_metric(
    rows: Sequence[MetricRow], metric: str, *, split: str = SPLIT_NAME
) -> MetricRow | None:
    for row in rows:
        if row.metric == metric and row.split == split:
            return row
    return None


def format_metric(row: MetricRow | None) -> str:
    """A number with its interval, or a word that is not a number.

    UNDEFINED (-1.0) is COCO's "there was nothing of this class to find", and it
    must never be rendered as `-1.0000` next to real APs where a reader would
    average it in.
    """

    if row is None:
        return "n/a"
    if row.value <= UNDEFINED:
        return "undefined"
    text = f"{row.value:.4f}"
    if row.ci_low is not None and row.ci_high is not None:
        text = f"{text} [{row.ci_low:.4f}, {row.ci_high:.4f}]"
    return text


def format_count(row: MetricRow | None) -> str:
    if row is None or row.n_instances is None:
        return "n/a"
    return f"{int(row.n_instances):,}"


def h4_gate_note(path: Path = H4_GATE_PATH) -> str:
    """The H4 artifact-detectability AUC, which CLAUDE.md requires beside results.

    Read from `reports/h4_artifact_gate_m13.json` rather than transcribed, so it
    cannot go stale relative to the gate that produced it.
    """

    payload = read_json_mapping(path)
    required = ("auc", "auc_ci95", "max_auc_for_scaleup", "n_examples")
    if payload is None or any(key not in payload for key in required):
        return (
            f"**H4 artifact-detectability AUC unavailable** - `{path.name}` is missing or "
            f"incomplete, and CLAUDE.md requires that number beside every result table. "
            f"Do not publish this table until it can be read."
        )
    low, high = (float(value) for value in payload["auc_ci95"])
    return (
        f"**H4 gate: FAILED, and the results below inherit it.** A logistic probe "
        f"separates synthetic patches from real ones at AUC "
        f"**{float(payload['auc']):.4f}** (95% CI {low:.4f}-{high:.4f}, "
        f"{int(payload['n_examples']):,} patches) against a pre-registered ceiling of "
        f"{float(payload['max_auc_for_scaleup']):.2f}. The synthetic images are trivially "
        f"distinguishable from real ones, so any gain an arm shows here may be a "
        f"detectable-domain effect rather than a useful one. ADR-011 allows the 1x pool "
        f"anyway and forbids 2x; it does not allow calling the gate passed."
    )


def render_main_table(
    results: Sequence[ArmResult],
    *,
    leak_evidence: Sequence[str],
    problems: Sequence[str],
    n_test_images: int,
    slice_sizes: Mapping[str, int],
    hard_negative_ids: Sequence[int],
    bootstrap_resamples: int,
    runs_root: Path,
    metrics_csv: Path,
) -> str:
    """reports/detection_main_table.md. Regenerate, never hand-edit."""

    lines = [
        "# M18 - four-arm detection results on the frozen Test split",
        "",
        (
            "Generated by `scripts/eval.py`. Every number here is re-aggregatable from "
            f"`{metrics_csv.name}`, which holds one row per arm x seed x split x metric "
            "(EVAL-11, EVAL-12). Nothing was copied from a training log: the validation "
            "numbers printed during training used a different split, a different "
            "threshold and a different coordinate space."
        ),
        "",
        f"- Weights root: `{runs_root}`",
        f"- Frozen Test images: **{n_test_images:,}** (real only, never Validation)",
        (
            "- Coordinates: predictions are mapped back to **each image's own** width and "
            "height before scoring, per EVAL-07. The split is not one resolution "
            "(DATA-25), so `scale_x` and `scale_y` differ."
        ),
        "",
        "## 1. Leak self-check (EVAL-14)",
        "",
    ]
    lines += [f"- {line}" for line in leak_evidence]

    lines += [
        "",
        "## 2. Which checkpoint each arm was scored on",
        "",
        (
            "Resolution order: `<arm>/checkpoints.json` -> the newest "
            "`checkpoint-*/trainer_state.json` -> the highest-numbered checkpoint. The "
            "third is a **fallback**: it is the last epoch, not necessarily the best "
            "one, and an arm resolved that way is not strictly comparable with an arm "
            "resolved by either of the first two."
        ),
        "",
        "| Arm | Seed | Checkpoint | Resolved by | Trail |",
        "|---|---:|---|---|---|",
    ]
    for result in results:
        marker = " **(FALLBACK)**" if result.source is CheckpointSource.HIGHEST_STEP else ""
        trail = "; ".join(result.resolution_notes) or "-"
        lines.append(
            f"| `{result.arm}` | {result.seed} | `{result.checkpoint.name}` | "
            f"{result.source.value}{marker} | {trail} |"
        )

    lines += [
        "",
        "## 3. Main table (EVAL-05)",
        "",
        (
            "`real image exposures` is a column and not a footnote on purpose: the arms "
            "share an optimizer-step budget (TRAIN-07), so the synthetic arms saw each "
            "real image roughly half as often as `real_only`. A gap in either direction "
            "is confounded with that difference."
        ),
        "",
        (
            "Intervals in brackets are percentile bootstraps over Test **images** "
            f"({bootstrap_resamples} resamples, EVAL-09)."
            if bootstrap_resamples > 0
            else (
                "**No bootstrap intervals were computed for this run "
                "(`--bootstrap-resamples 0`), so EVAL-09 is NOT satisfied by it.**"
            )
        ),
        "",
        (
            "| Arm | Seed | Real-image exposures | AP_small (primary) | n small | "
            "bare-head recall | mAP50-95 (primary) | HN FP/image | Detections |"
        ),
        "|---|---:|---:|---|---:|---|---|---:|---:|",
    ]
    for result in results:
        small = row_by_metric(result.rows, "primary_map_small")
        overall = row_by_metric(result.rows, "primary_map")
        recall = row_by_metric(result.rows, "bare_head_recall")
        hard_negative = row_by_metric(result.rows, "hard_negative_fp_per_image")
        exposures = "n/a" if result.exposures is None else f"{result.exposures:.2f}"
        hn_text = "n/a" if hard_negative is None else f"{hard_negative.value:.4f}"
        lines.append(
            f"| `{result.arm}` | {result.seed} | {exposures} | {format_metric(small)} | "
            f"{format_count(small)} | {format_metric(recall)} | {format_metric(overall)} | "
            f"{hn_text} | {result.metrics.n_detections:,} |"
        )

    lines += [
        "",
        (
            f"Hard-negative subset: **{len(hard_negative_ids):,}** Test images with no "
            "`helmet`/`head` ground truth. A false positive there can only move "
            "precision, so it gets its own column instead of being diluted into mAP."
        ),
        "",
        "Every arm above is a **single seed**. EVAL-10 forbids reading a fraction of a",
        "point as a win; only a 3-seed mean +/- std or a bootstrap interval that clears",
        "the gap supports a directional claim.",
        "",
        "## 4. Size buckets (EVAL-08)",
        "",
        "| Arm | " + " | ".join(f"AP_{name} | n {name}" for name in ("small", "medium", "large"))
        + " |",
        "|---|" + "---:|" * 6,
    ]
    for result in results:
        cells: list[str] = []
        for bucket in ("small", "medium", "large"):
            row = row_by_metric(result.rows, f"map_{bucket}")
            cells.append(format_metric(row))
            cells.append(format_count(row))
        lines.append(f"| `{result.arm}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 5. Scenario slices (EVAL-17)",
        "",
        (
            "Slices are cut from ground-truth statistics, not by hand, and they overlap. "
            "This is the check on whether targeted synthesis hit its target: if the "
            "small-object slice improves no more than the others, the effect is `more "
            "data`, not `targeted data`, and that has to be written as such."
        ),
        "",
        "| Arm | " + " | ".join(
            f"{name} AP_small ({slice_sizes.get(name, 0)} img)" for name in sorted(slice_sizes)
        )
        + " |",
        "|---|" + "---:|" * max(len(slice_sizes), 1),
    ]
    for result in results:
        cells = [
            format_metric(
                row_by_metric(result.rows, "primary_map_small", split=f"{SPLIT_NAME}/{name}")
            )
            for name in sorted(slice_sizes)
        ]
        lines.append(f"| `{result.arm}` | " + " | ".join(cells) + " |")

    lines += ["", "## 6. Caveats", "", h4_gate_note(), ""]
    lines += [
        (
            "**Every claim above is relative.** Roughly two thirds of the real objects in "
            "this dataset are unannotated (ADR-003), so an absolute AP measures the "
            "annotation as much as the detector. Only arm-versus-arm differences on this "
            "same frozen Test split mean anything."
        ),
        "",
    ]

    if problems:
        lines += ["## 7. Problems", "", "These are reported, not swallowed:", ""]
        lines += [f"- {problem}" for problem in problems]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def load_driver_training_config(path: Path = TRAINING_CONFIG) -> dict[str, Any]:
    """Resolve child configs; plain YAML loading silently drops inherited run keys."""

    return load_resolved_training_config(path)


def atomic_write_json_value(path: Path, payload: Any, *, compact: bool = False) -> None:
    """Persist predictions/index evidence without leaving a partial JSON file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = (
        json.dumps(payload, separators=(",", ":"))
        if compact
        else json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_prediction_index_strict(path: Path) -> dict[str, dict[str, Any]]:
    if not Path(path).is_file():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvalDriverError(f"cannot read prediction index {path}: {error}") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, dict)
        for key, value in payload.items()
    ):
        raise EvalDriverError(f"prediction index {path} is not an object of objects")
    return payload


def validate_output_isolation(
    args: argparse.Namespace,
    *,
    default_runs_root: Path,
    default_training_config: Path,
    default_metrics_csv: Path,
    default_report: Path,
    default_predictions_root: Path,
    default_predictions_index: Path,
) -> None:
    canonical = lambda path: Path(path).resolve(strict=False)
    nondefault_model = canonical(args.runs_root) != canonical(
        default_runs_root
    ) or canonical(args.training_config) != canonical(default_training_config)
    if not nondefault_model:
        return
    missing: list[str] = []
    if canonical(args.metrics_csv) == canonical(default_metrics_csv):
        missing.append("--metrics-csv")
    if canonical(args.report) == canonical(default_report):
        missing.append("--report")
    if args.predictions_root is None or canonical(args.predictions_root) == canonical(
        default_predictions_root
    ):
        missing.append("--predictions-root")
    if canonical(args.predictions_index) == canonical(default_predictions_index):
        missing.append("--predictions-index")
    if missing:
        raise EvalDriverError(
            "nondefault detector inputs require isolated " + " and ".join(missing)
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    paths = load_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=paths.runs)
    parser.add_argument(
        "--run-records",
        "--colab-results",
        dest="run_records",
        type=Path,
        default=COLAB_RESULTS,
    )
    parser.add_argument("--manifest", type=Path, default=paths.splits / "split_manifest.json")
    parser.add_argument("--annotations", type=Path, default=paths.interim / "coco_all.json")
    parser.add_argument("--images-root", type=Path, default=paths.hardhat_raw / "images")
    parser.add_argument("--config", type=Path, default=EVALUATION_CONFIG)
    parser.add_argument("--training-config", type=Path, default=TRAINING_CONFIG)
    parser.add_argument("--metrics-csv", type=Path, default=default_detection_metrics_path())
    parser.add_argument("--report", type=Path, default=paths.reports / REPORT_NAME)
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument(
        "--predictions-index",
        type=Path,
        default=PROJECT_ROOT / "results" / "rfdetr_predictions_index.json",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto-detect)")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=None,
        help=(
            "override metrics.bootstrap_resamples. 0 skips EVAL-09 entirely and the "
            "report says so; a full run is one COCOeval per resample per metric per arm"
        ),
    )
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=None,
        help=(
            "processes for the resample loop (default: metrics.bootstrap_workers). "
            "Wall clock only - the interval is identical at any worker count "
            "because each resample draws from its own SeedSequence child"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, load_model: Callable[..., Any] | None = None) -> int:
    """Exit 0 clean, 1 ran with gaps, 2 could not run at all."""

    args = parse_args(argv)
    if load_model is None:
        load_model = default_load_model

    project_paths = load_project_paths()
    validate_output_isolation(
        args,
        default_runs_root=project_paths.runs,
        default_training_config=TRAINING_CONFIG,
        default_metrics_csv=default_detection_metrics_path(),
        default_report=project_paths.reports / REPORT_NAME,
        default_predictions_root=project_paths.runs / "predictions",
        default_predictions_index=PROJECT_ROOT / "results" / "predictions_index.json",
    )
    config = load_evaluation_config(args.config)
    training_config = load_driver_training_config(args.training_config)

    # The four-arm archive is still being downloaded, so every input below can
    # legitimately be absent today. Each one says which file to produce instead
    # of raising a traceback out of a helper three modules down.
    try:
        splits = split_real_images(args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot read {args.manifest}: {type(error).__name__}: {error}")
        return 2
    for name in ("train", "val", SPLIT_NAME):
        if name not in splits:
            print(f"{args.manifest} declares no {name!r} split")
            return 2

    if not args.run_records.is_dir():
        print(
            f"{args.run_records} does not exist, so no arm's training composition can "
            f"be read and EVAL-14 cannot be checked. Produce or mirror run records "
            f"before evaluation."
        )
        return 2
    try:
        records = load_run_records(args.run_records)
    except ColabResultsError as error:
        print(f"Cannot read the run records: {error}")
        return 2

    weights, problems = discover_arm_weights(args.runs_root)
    if not weights:
        print(f"No weights under {args.runs_root}:")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    scored_arms = sorted({item.arm for item in weights})
    try:
        leak_evidence = leak_self_check(
            records,
            real_train_names=splits["train"],
            test_names=splits[SPLIT_NAME],
            arms=scored_arms,
        )
    except (TrainingListUnavailableError, EvalDriverError) as error:
        print(f"EVAL-14 self-check failed: {error}")
        return 2
    for line in leak_evidence:
        print(f"EVAL-14: {line}")

    compositions = reconstruct_compositions(
        records, real_train_names=splits["train"], real_val_names=splits["val"]
    )
    plan: dict[str, dict[str, Any]] = {}
    if REFERENCE_ARM in compositions:
        plan = step_budget_plan(compositions, training_config=training_config)
        problems = tuple(problems) + verify_step_budget(plan, records)
    else:
        problems = tuple(problems) + (
            (
                f"no run record for {REFERENCE_ARM}, so real_image_exposures is "
                f"unavailable for every arm"
            ),
        )

    samples = load_test_samples(args.manifest, args.annotations, args.images_root)
    ground_truth = build_coco_ground_truth(samples, CLASS_NAMES)
    slice_config = load_slice_config(args.config)
    slices = scenario_slices(
        samples,
        luminance_by_image=image_mean_luminances(samples),
        config=slice_config,
    )
    hard_negative_ids = hard_negative_image_ids(ground_truth, config=config)

    resamples = (
        int(config["metrics"]["bootstrap_resamples"])
        if args.bootstrap_resamples is None
        else int(args.bootstrap_resamples)
    )
    workers = (
        int(config["metrics"].get("bootstrap_workers", 1))
        if args.bootstrap_workers is None
        else int(args.bootstrap_workers)
    )
    batch_size = (
        int(training_config["run"]["per_device_eval_batch_size"])
        if args.batch_size is None
        else int(args.batch_size)
    )
    device = resolve_device() if args.device is None else str(args.device)
    dtype_name = resolve_dtype_name(str(args.dtype), device)

    results: list[ArmResult] = []
    prediction_index = (
        read_prediction_index_strict(args.predictions_index)
        if args.predictions_root is not None
        else {}
    )
    for item in weights:
        entry = plan.get(item.arm, {})
        print(f"scoring {item.arm} seed {item.seed} from {item.choice.path}")
        predictions_path = (
            args.predictions_root / f"{item.arm}_{SPLIT_NAME}_seed{item.seed}.json"
            if args.predictions_root is not None
            else None
        )
        result = evaluate_arm(
                item,
                samples=samples,
                ground_truth=ground_truth,
                slices=slices,
                hard_negative_ids=hard_negative_ids,
                config=config,
                load_model=load_model,
                processor_source=str(training_config["model"]["checkpoint"]),
                device=device,
                dtype_name=dtype_name,
                batch_size=batch_size,
                bootstrap_resamples=resamples,
                bootstrap_workers=workers,
                bootstrap_seed=project_paths.seed,
                exposures=entry.get("real_image_exposures"),
                total_steps=records.get(item.arm, {}).get("total_steps"),
                predictions_path=predictions_path,
            )
        results.append(result)
        if predictions_path is not None:
            prediction_index[f"{item.arm}/{SPLIT_NAME}/seed_{item.seed}"] = {
                "arm": item.arm,
                "split": SPLIT_NAME,
                "seed": item.seed,
                "checkpoint": result.checkpoint.name,
                "n_images": result.metrics.n_images,
                "n_detections": result.metrics.n_detections,
                "path": predictions_path.as_posix(),
                "coordinates": "original per-image annotation space (DATA-25)",
                "score_threshold": MAP_SCORE_THRESHOLD,
            }

    if args.predictions_root is not None:
        atomic_write_json_value(args.predictions_index, prediction_index)
        print(f"wrote {args.predictions_index}")

    rows = [row for result in results for row in result.rows]
    csv_path = write_detection_metrics_csv(rows, args.metrics_csv)

    report = render_main_table(
        results,
        leak_evidence=leak_evidence,
        problems=problems,
        n_test_images=len(samples),
        slice_sizes={name: len(members) for name, members in slices.items()},
        hard_negative_ids=hard_negative_ids,
        bootstrap_resamples=resamples,
        runs_root=args.runs_root,
        metrics_csv=csv_path,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")

    print(f"wrote {csv_path} ({len(rows)} rows)")
    print(f"wrote {args.report}")
    for result in results:
        if result.source is CheckpointSource.HIGHEST_STEP:
            print(
                f"WARNING {result.arm} seed {result.seed} fell back to "
                f"{result.checkpoint.name}, the highest step, not a recorded best"
            )
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
