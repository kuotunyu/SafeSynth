"""Compliance derivation from detections (EVAL-01..EVAL-04).

Two semantics for what a `helmet` box means are implemented here and selected by
`compliance.mode` in configs/evaluation.yaml. Spike H1 / ADR-007 already decided
that `class_direct` is the live one, but keeping the fallback in code - rather
than only in a worklog sentence - is what makes "we considered the other
semantics" a verifiable claim instead of a memory.

`person` is never an input to a verdict. EVAL-02 permits it as an OPTIONAL
grouping hint; this module implements no grouping path at all, because the only
robust way to stop a hint from quietly becoming a requirement is not to have
one. Person detections are dropped before either mode function can see them, and
tests/test_compliance.py -k person_not_load_bearing is the mechanical proof that
adding them back changes nothing.

Every numeric threshold is read from configs/evaluation.yaml (compliance.*,
metrics.*) or, for the FILT-07 geometry, from configs/filtering.yaml. None are
written down here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import load_project_paths

# FILT-07's inequalities AND its calibrated constants live in the filter. The
# `geometric_pairing` mode has to ask exactly the question the compositor asked,
# so the predicate is imported rather than re-derived; a second copy of the
# geometry would drift from the first without anything failing.
from src.filtering.rules import _filt_07 as filt_07_reject_reasons
from src.filtering.rules import _iou as box_iou
from src.training.data import CLASS_NAMES, Sample

# Not a threshold, a name: the class that must never carry load (ADR-003).
PERSON_CLASS = "person"
HELMET_CLASS = "helmet"
HEAD_CLASS = "head"

# The split whose detections may never be used to choose an operating point.
FORBIDDEN_SWEEP_SPLIT = "test"


class ComplianceStatus(StrEnum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"


class ComplianceConfigError(RuntimeError):
    """Raised when the config asks for behaviour this module refuses to fake."""


class SplitLeakageError(RuntimeError):
    """Raised when an operating point is about to be chosen on Test."""


@dataclass(frozen=True)
class Detection:
    """One detected box. `detection_id` is optional caller-side provenance."""

    class_name: str
    bbox_xywh: tuple[float, float, float, float]
    score: float
    detection_id: str | None = None

    def __post_init__(self) -> None:
        if self.class_name not in CLASS_NAMES:
            raise ValueError(
                f"Unknown class {self.class_name!r}; expected one of {list(CLASS_NAMES)}"
            )
        box = tuple(float(value) for value in self.bbox_xywh)
        if len(box) != 4:
            raise ValueError(f"bbox_xywh must be [x, y, w, h], got {self.bbox_xywh!r}")
        object.__setattr__(self, "bbox_xywh", box)
        object.__setattr__(self, "score", float(self.score))


@dataclass(frozen=True)
class Verdict:
    """A compliance judgement about ONE detection.

    Deliberately identified by the detection's own content, never by its index
    in the input list: removing the `person` detections would shift every later
    index and make the EVAL-03 bit-identity check impossible to satisfy for even
    a correct implementation.
    """

    detection_id: str | None
    class_name: str
    bbox_xywh: tuple[float, float, float, float]
    score: float
    status: ComplianceStatus


@dataclass(frozen=True)
class ComplianceResult:
    mode: str
    score_threshold: float
    verdicts: tuple[Verdict, ...]
    n_compliant: int
    n_non_compliant: int
    # None, not NaN and not 0.0. NaN propagates silently through any later mean
    # and 0.0 would assert "nobody was compliant" from zero evidence. None makes
    # an unconsidered aggregation raise at the point of the mistake.
    compliance_rate: float | None


@dataclass(frozen=True)
class OperatingPoint:
    score_threshold: float
    bare_head_recall: float | None
    compliance_precision: float | None
    n_predicted_compliant: int
    n_predicted_non_compliant: int
    n_ground_truth_bare_heads: int


def _config_path(name: str) -> Path:
    """Resolve a config file through configs/paths.yaml, never a literal root."""

    return load_project_paths().project_root / "configs" / name


def _load_yaml(path: Path) -> dict[str, Any]:
    # K-10: this machine's default text encoding is cp950, so every text open
    # states utf-8 or dies on the first non-ASCII comment in the config.
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return payload


def load_evaluation_config(path: Path | None = None) -> dict[str, Any]:
    return _load_yaml(Path(path) if path is not None else _config_path("evaluation.yaml"))


def load_filtering_config(path: Path | None = None) -> dict[str, Any]:
    return _load_yaml(Path(path) if path is not None else _config_path("filtering.yaml"))


# One memoized parse of the DEFAULT configs/filtering.yaml. One slot, no path
# parameter, no key.
#
# There used to be a path-keyed dict here. No caller ever passed a path, so the
# key expression never ran in its non-trivial form and replacing it outright with
# `None` changed nothing observable - K-19's exact shape, in the cache instead of
# in a test. It is deleted rather than wired up because pointing compliance at
# different FILT-07 thresholds is ALREADY expressible: derive_compliance and
# sweep_operating_points take the parsed mapping as `filtering_config=`, which is
# checked by tests/test_compliance.py::
# test_an_explicit_filtering_config_is_used_instead_of_the_memo. A path argument
# would be a second spelling of that request, and keying on the resolved DEFAULT
# path is worse than useless: resolving it is itself a YAML parse
# (_config_path -> load_project_paths reads configs/paths.yaml), so computing the
# key would leave a per-call disk read behind and defeat the cache it keys.
#
# The memo exists because derive_compliance falls back to reading the file when
# no filtering_config is passed, and sweep_operating_points calls
# derive_compliance once per (image x candidate threshold). Measured on
# 20 images x 200 thresholds in geometric_pairing mode: 0.06 s memoized against
# 30-38 s re-parsing on every call, i.e. 600x. The real Validation split is 756
# images, so the un-memoized version is hours.
#
# The parsed mapping is shared by every caller and must never be mutated in
# place; callers who need to edit a config take their own copy.
_FILTERING_CONFIG: dict[str, Any] | None = None


def _cached_filtering_config() -> dict[str, Any]:
    """`load_filtering_config()` with the disk read paid at most once per process."""

    global _FILTERING_CONFIG
    if _FILTERING_CONFIG is None:
        _FILTERING_CONFIG = load_filtering_config()
    return _FILTERING_CONFIG


def detections_from_coco(records: Sequence[Mapping[str, Any]]) -> dict[int, list[Detection]]:
    """Group `predictions_to_coco` output into per-image detection lists.

    Per-image grouping is structural: nothing downstream can accidentally pair a
    helmet in one frame with a head in another.
    """

    grouped: dict[int, list[Detection]] = {}
    for record in records:
        category = int(record["category_id"])
        if not 0 <= category < len(CLASS_NAMES):
            raise ValueError(
                f"category_id {category} is outside the contiguous range for {list(CLASS_NAMES)}"
            )
        grouped.setdefault(int(record["image_id"]), []).append(
            Detection(
                class_name=CLASS_NAMES[category],
                bbox_xywh=tuple(float(value) for value in record["bbox"]),
                score=float(record["score"]),
            )
        )
    return grouped


# spec: EVAL-01
def build_pair_descriptor(
    helmet_bbox: Sequence[float], head_bbox: Sequence[float]
) -> dict[str, float] | None:
    """The five FILT-07 quantities, normalized by the HEAD box.

    Definitions are docs/filtering_spec.md FILT-07 verbatim. Returns None for a
    degenerate head box: the normalization divides by head width and height, so
    a zero-sized head offers nothing to normalize against and the pair simply
    cannot be judged.
    """

    head_x, head_y, head_w, head_h = (float(value) for value in head_bbox)
    helmet_x, helmet_y, helmet_w, helmet_h = (float(value) for value in helmet_bbox)
    if head_w <= 0 or head_h <= 0:
        return None
    return {
        "dx": ((helmet_x + helmet_w / 2) - (head_x + head_w / 2)) / head_w,
        "dy": (head_y - (helmet_y + helmet_h / 2)) / head_h,
        "overlap_y": ((helmet_y + helmet_h) - head_y) / head_h,
        "r_w": helmet_w / head_w,
        "r_h": helmet_h / head_h,
        "iou": box_iou(helmet_bbox, head_bbox),
    }


# spec: EVAL-01
def is_worn_helmet_pair(
    helmet_bbox: Sequence[float],
    head_bbox: Sequence[float],
    filtering_config: Mapping[str, Any] | None,
) -> bool:
    """True when FILT-07 accepts this (helmet, head) as a worn helmet.

    The decision is delegated wholesale to the filter's own predicate, so the
    thresholds used here are exactly the ones in configs/filtering.yaml under
    rules.helmet_above_head.
    """

    if filtering_config is None:
        raise ComplianceConfigError(
            "geometric_pairing needs configs/filtering.yaml for the FILT-07 thresholds"
        )
    descriptor = build_pair_descriptor(helmet_bbox, head_bbox)
    if descriptor is None:
        return False
    return not filt_07_reject_reasons({"pairs": [descriptor]}, filtering_config)


# The class-to-status table IS the class_direct semantics (EVAL-01), not a
# tunable, so it belongs in code. `person` is absent by design.
_CLASS_DIRECT_STATUS = {
    HELMET_CLASS: ComplianceStatus.COMPLIANT,
    HEAD_CLASS: ComplianceStatus.NON_COMPLIANT,
}


# spec: EVAL-01
def mode_class_direct(
    detections: Sequence[Detection], filtering_config: Mapping[str, Any] | None = None
) -> tuple[tuple[Detection, ComplianceStatus], ...]:
    """The detected class IS the status. Zero spatial reasoning.

    `filtering_config` is accepted and ignored so that both modes share one
    signature and the registry dispatch stays honest.
    """

    del filtering_config
    return tuple(
        (detection, _CLASS_DIRECT_STATUS[detection.class_name])
        for detection in detections
        if detection.class_name in _CLASS_DIRECT_STATUS
    )


# spec: EVAL-01
def mode_geometric_pairing(
    detections: Sequence[Detection], filtering_config: Mapping[str, Any] | None = None
) -> tuple[tuple[Detection, ComplianceStatus], ...]:
    """A `head` is COMPLIANT iff some `helmet` passes the FILT-07 pairing.

    Under this (fallback) semantics the helmet box bounds only the shell, so a
    helmet is EVIDENCE about a head rather than a judgeable unit of its own: an
    unpaired helmet yields no verdict, because the head it belongs to was simply
    not detected.
    """

    helmets = [item for item in detections if item.class_name == HELMET_CLASS]
    verdicts: list[tuple[Detection, ComplianceStatus]] = []
    for detection in detections:
        if detection.class_name != HEAD_CLASS:
            continue
        worn = any(
            is_worn_helmet_pair(helmet.bbox_xywh, detection.bbox_xywh, filtering_config)
            for helmet in helmets
        )
        verdicts.append(
            (
                detection,
                ComplianceStatus.COMPLIANT if worn else ComplianceStatus.NON_COMPLIANT,
            )
        )
    return tuple(verdicts)


ModeFunction = Callable[
    [Sequence[Detection], Mapping[str, Any] | None],
    tuple[tuple[Detection, ComplianceStatus], ...],
]

COMPLIANCE_MODES: dict[str, ModeFunction] = {
    "class_direct": mode_class_direct,
    "geometric_pairing": mode_geometric_pairing,
}

_MODES_NEEDING_FILTERING_CONFIG = frozenset({"geometric_pairing"})


# spec: EVAL-03
def _drop_person(detections: Sequence[Detection]) -> tuple[Detection, ...]:
    """`person` is removed before any mode function can observe it.

    EVAL-02 allows person only as an optional grouping hint. We implement no
    grouping at all: a hint that does not exist cannot silently become a
    required input during a later refactor.
    """

    return tuple(item for item in detections if item.class_name != PERSON_CLASS)


# spec: EVAL-02
def _build_result(
    mode: str,
    score_threshold: float,
    judged: Sequence[tuple[Detection, ComplianceStatus]],
) -> ComplianceResult:
    verdicts = tuple(
        Verdict(
            detection_id=detection.detection_id,
            class_name=detection.class_name,
            bbox_xywh=detection.bbox_xywh,
            score=detection.score,
            status=status,
        )
        for detection, status in judged
    )
    n_compliant = sum(1 for item in verdicts if item.status is ComplianceStatus.COMPLIANT)
    n_non_compliant = len(verdicts) - n_compliant
    total = n_compliant + n_non_compliant
    return ComplianceResult(
        mode=mode,
        score_threshold=score_threshold,
        verdicts=verdicts,
        n_compliant=n_compliant,
        n_non_compliant=n_non_compliant,
        compliance_rate=(n_compliant / total) if total else None,
    )


# spec: EVAL-01, EVAL-02, EVAL-04
def derive_compliance(
    detections: Sequence[Detection],
    *,
    config: Mapping[str, Any],
    filtering_config: Mapping[str, Any] | None = None,
    mode: str | None = None,
    score_threshold: float | None = None,
) -> ComplianceResult:
    """Frame-level compliance for one image's detections.

    `score_threshold` (default: compliance.score_threshold) is the operating
    point of the detector for THIS task only. It is deliberately NOT the mAP
    threshold: mAP integrates over every confidence, whereas a compliance
    verdict needs one frozen decision point chosen on Validation (EVAL-04).
    Anyone tempted to "unify the two thresholds" would be silently changing what
    mAP measures. Passing an explicit value is how the sweep evaluates
    candidates; it never mutates the config.
    """

    settings = config["compliance"]
    if bool(settings["use_person_as_grouping_hint"]):
        # Refusing beats pretending. The flag is false in config; if someone
        # turns it on they must implement grouping AND re-prove EVAL-03, rather
        # than get a silent no-op that looks like the feature works.
        raise ComplianceConfigError(
            "compliance.use_person_as_grouping_hint is true, but no grouping path is "
            "implemented (EVAL-03 keeps `person` non-load-bearing by construction)"
        )

    selected_mode = str(settings["mode"]) if mode is None else str(mode)
    if selected_mode not in COMPLIANCE_MODES:
        raise ComplianceConfigError(
            f"Unknown compliance.mode {selected_mode!r}; expected one of "
            f"{sorted(COMPLIANCE_MODES)}"
        )

    threshold = (
        float(settings["score_threshold"]) if score_threshold is None else float(score_threshold)
    )
    if selected_mode in _MODES_NEEDING_FILTERING_CONFIG and filtering_config is None:
        filtering_config = _cached_filtering_config()

    kept = tuple(item for item in _drop_person(detections) if item.score >= threshold)
    judged = COMPLIANCE_MODES[selected_mode](kept, filtering_config)
    return _build_result(selected_mode, threshold, judged)


# Ground truth carries the ADR-007 reading of the labels: a `helmet` annotation
# is a compliant head and a `head` annotation is a bare one. This is fixed by the
# dataset, so a fallback mode that disagrees with it simply scores worse - which
# is the honest outcome, not a bug in the scoring.
_GROUND_TRUTH_STATUS = dict(_CLASS_DIRECT_STATUS)


def _ground_truth_statuses(sample: Sample) -> list[tuple[tuple[float, ...], ComplianceStatus]]:
    boxes: list[tuple[tuple[float, ...], ComplianceStatus]] = []
    for box, index in zip(sample.boxes_xywh, sample.class_indices, strict=True):
        name = CLASS_NAMES[int(index)]
        if name in _GROUND_TRUTH_STATUS:
            boxes.append((tuple(float(value) for value in box), _GROUND_TRUTH_STATUS[name]))
    return boxes


def _match_one_to_one(
    verdicts: Sequence[Verdict],
    ground_truth: Sequence[tuple[tuple[float, ...], ComplianceStatus]],
    status: ComplianceStatus,
    iou_threshold: float,
) -> int:
    """Greedy one-to-one match of same-status verdicts to same-status GT boxes.

    Returns the number of matched pairs, which is simultaneously the true
    positives for precision and the covered ground truth for recall. Verdicts
    are consumed in descending score with a deterministic tie-break, so the
    result never depends on input ordering.
    """

    candidates = sorted(
        (item for item in verdicts if item.status is status),
        key=lambda item: (-item.score, item.class_name, item.bbox_xywh),
    )
    targets = [box for box, box_status in ground_truth if box_status is status]
    taken: set[int] = set()
    matched = 0
    for candidate in candidates:
        best_index, best_iou = -1, iou_threshold
        for index, target in enumerate(targets):
            if index in taken:
                continue
            overlap = box_iou(candidate.bbox_xywh, target)
            if overlap >= best_iou:
                best_index, best_iou = index, overlap
        if best_index >= 0:
            taken.add(best_index)
            matched += 1
    return matched


# spec: EVAL-04
def sweep_operating_points(
    detections_by_image: Mapping[int, Sequence[Detection]],
    ground_truth: Sequence[Sample],
    *,
    split: str,
    config: Mapping[str, Any],
    filtering_config: Mapping[str, Any] | None = None,
    mode: str | None = None,
    thresholds: Sequence[float] | None = None,
) -> tuple[OperatingPoint, ...]:
    """Bare-head recall and compliance precision at each candidate threshold.

    `split` is required and rejected when it names Test, so that pointing this at
    Test is a structural impossibility rather than a discipline problem: choosing
    an operating point on Test is tuning on the test set.

    Candidate thresholds default to the distinct detection scores - every
    achievable operating point and no invented grid, which also keeps this
    function free of hard-coded numbers. IoU for matching comes from
    metrics.bare_head_recall_iou.
    """

    if split.strip().lower() == FORBIDDEN_SWEEP_SPLIT:
        raise SplitLeakageError(
            "Operating points are selected on Validation only; sweeping on Test would "
            "tune the decision threshold on the frozen test set (EVAL-04)"
        )

    iou_threshold = float(config["metrics"]["bare_head_recall_iou"])
    truth_by_image = {
        int(sample.image_id): _ground_truth_statuses(sample) for sample in ground_truth
    }
    # The sweep iterates the GROUND TRUTH, so a detection file from another
    # split cannot be scored here by accident; it fails loudly instead.
    orphans = sorted(set(detections_by_image) - set(truth_by_image))
    if orphans:
        raise SplitLeakageError(
            f"Detections reference image ids with no ground truth in split {split!r}: {orphans}"
        )

    if thresholds is None:
        scores = sorted({item.score for items in detections_by_image.values() for item in items})
        # With nothing detected anywhere there is no achievable operating point
        # to report, so fall back to the configured one and report the (empty)
        # truth about it rather than returning no rows at all.
        candidates: list[float] = scores or [float(config["compliance"]["score_threshold"])]
    else:
        candidates = [float(value) for value in thresholds]

    points: list[OperatingPoint] = []
    for threshold in candidates:
        n_compliant = n_non_compliant = 0
        true_compliant = 0
        matched_bare_heads = 0
        total_bare_heads = 0
        for image_id, truth in truth_by_image.items():
            result = derive_compliance(
                detections_by_image.get(image_id, ()),
                config=config,
                filtering_config=filtering_config,
                mode=mode,
                score_threshold=threshold,
            )
            n_compliant += result.n_compliant
            n_non_compliant += result.n_non_compliant
            true_compliant += _match_one_to_one(
                result.verdicts, truth, ComplianceStatus.COMPLIANT, iou_threshold
            )
            matched_bare_heads += _match_one_to_one(
                result.verdicts, truth, ComplianceStatus.NON_COMPLIANT, iou_threshold
            )
            total_bare_heads += sum(
                1 for _, status in truth if status is ComplianceStatus.NON_COMPLIANT
            )
        points.append(
            OperatingPoint(
                score_threshold=threshold,
                # Same None-not-NaN policy as compliance_rate: an undefined
                # ratio must not be mistaken for a measured zero.
                bare_head_recall=(
                    matched_bare_heads / total_bare_heads if total_bare_heads else None
                ),
                compliance_precision=(true_compliant / n_compliant if n_compliant else None),
                n_predicted_compliant=n_compliant,
                n_predicted_non_compliant=n_non_compliant,
                n_ground_truth_bare_heads=total_bare_heads,
            )
        )
    return tuple(points)


# spec: EVAL-04
def select_operating_point(
    points: Sequence[OperatingPoint], *, config: Mapping[str, Any]
) -> OperatingPoint | None:
    """Highest bare-head recall subject to compliance.min_compliance_precision.

    Returns None when no candidate clears the precision floor - the caller then
    has to decide something explicitly instead of receiving a silently bad point.

    A point with ZERO bare-head recall is never eligible, however good its
    precision looks. Precision is undefined-shaped near the top of the sweep: a
    threshold that fires on almost nothing scores 1.0 on the handful of calls it
    still makes, so "maximise recall subject to precision" would happily return
    a detector that never reports anyone non-compliant. Measured on this
    project's `unfiltered_syn` arm, which cannot reach the 0.80 floor at any
    threshold where it detects anything, the unguarded rule selected threshold
    0.20 with recall 0.0000 and precision 1.0000 - a perfect score for a safety
    check that never fires. The honest answer there is "no usable operating
    point", and that is what None now means.
    """

    minimum = float(config["compliance"]["min_compliance_precision"])
    eligible = [
        point
        for point in points
        if point.bare_head_recall is not None
        and point.bare_head_recall > 0.0
        and point.compliance_precision is not None
        and point.compliance_precision >= minimum
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda point: (
            point.bare_head_recall,
            point.compliance_precision,
            point.score_threshold,
        ),
    )
