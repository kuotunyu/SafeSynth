"""EVAL-16: false positives per image on hard-negative regions of the Test split.

WHY THIS IS A SEPARATE MODULE FROM src/synthetic/hard_negatives.py. That one
mines distractors to paste into training images, and it reads the TRAIN split
and nothing else - `train_context(paths)` is the only door in. That is a safety
property, not an implementation detail: the iron rule is that the generator
never touches Test. Parameterising it by split to reuse it here would replace a
guarantee held by construction with one held by a caller passing the right
argument.

So the guard PREDICATES are imported and the mining loop is rewritten around
them. The generator's entry point still cannot reach Test.

WHY THIS RUNS ON TEST AT ALL. EVAL-16 asks for false positives per image on
images containing hard negatives. The frozen Test split has none by
construction: all 744 images contain a helmet or a head, so the subset the
metric wants is empty. The spec's own fallback is to mine candidate regions on
Test FOR ANALYSIS ONLY, which is what this does. Nothing here feeds training,
and nothing here may inform an operating point - EVAL-04 selects that on
Validation, and this module has no path to it.

WHY THE MINED REGIONS CAN BE TRUSTED. The same guards, on the same kind of
images, were purity-checked in Phase 1: spike H6 put 64 mined cells in front of
a human and recorded **0 real helmets** against a maximum tolerated 10%
(reports/h6_hard_negative_spike.md). That was measured on Train. The guards are
content-based rather than split-dependent, so the expectation carries, but it
is an expectation - `render_region_sheet` exists so the Test regions can be
checked the same way rather than assumed.

WHAT A COUNT HERE DOES AND DOES NOT MEAN. This dataset leaves roughly two
thirds of real objects unannotated (SHEL5K: 75,570 against 25,502), so
"detection with no matching ground truth" is NOT a false positive in general -
it is very often a correct detection of an unannotated object. That is exactly
why this metric is defined over MINED REGIONS rather than over unmatched
detections: a mined region has passed the worn-helmet guards, so a box on it is
a genuine error rather than an annotation gap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# Private on purpose over there; imported here rather than duplicated, because
# two copies of a purity guard drift and only one of them gets the fix.
from src.synthetic.hard_negatives import (
    _has_head_or_person_below,
    _outside_helmet_typical_range,
    _skin_like_below,
    _xywh_iou,
)


class HardNegativeAnalysisError(RuntimeError):
    """Raised when the analysis cannot be run as specified."""


@dataclass(frozen=True)
class Region:
    """One mined distractor region on one image, with the guards it passed."""

    image_id: int
    file_name: str
    bbox: tuple[float, float, float, float]
    circularity: float
    max_iou_with_annotation: float

    def contains(self, detection_bbox: Sequence[float]) -> bool:
        """Does a detection's CENTRE fall inside this region?

        Centre-inside rather than IoU: the question is "did the model fire on
        this distractor", and a detector that boxes the object loosely or
        tightly is equally wrong about it. An IoU threshold would make the
        answer depend on how well it framed something that should not have been
        detected at all.
        """

        x, y, width, height = self.bbox
        cx = float(detection_bbox[0]) + float(detection_bbox[2]) / 2.0
        cy = float(detection_bbox[1]) + float(detection_bbox[3]) / 2.0
        return x <= cx <= x + width and y <= cy <= y + height


@dataclass(frozen=True)
class ArmFalsePositives:
    """EVAL-16's number for one arm, plus what it was computed over."""

    arm: str
    n_false_positives: int
    n_regions: int
    n_images_with_regions: int
    score_threshold: float

    @property
    def per_image(self) -> float:
        if self.n_images_with_regions == 0:
            raise HardNegativeAnalysisError(
                "no image carries a mined region, so a per-image rate has no denominator"
            )
        return self.n_false_positives / self.n_images_with_regions


# spec: EVAL-16
def mine_regions(
    image_records: Mapping[int, Mapping[str, Any]],
    annotations: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    category_names: Mapping[int, str],
    helmet_geometry: Mapping[str, Any],
    mining: Mapping[str, Any],
    load_hsv,
) -> tuple[list[Region], dict[str, int]]:
    """Distractor regions and the guard-rejection tally, on any set of images.

    `load_hsv` is injected so this is testable without image files, and so the
    caller decides which split's pixels are read - this module never resolves a
    split itself.
    """

    hue_low, hue_high = (float(value) / 2 for value in mining["hue_deg"])
    saturation_low = round(float(mining["min_saturation"]) * 255)
    value_low = round(float(mining["min_value"]) * 255)
    area_low, area_high = (float(v) for v in mining["contour_area_px"])
    circ_low, circ_high = (float(v) for v in mining["circularity"])
    max_iou_allowed = float(mining["max_iou_with_any_annotation"])

    regions: list[Region] = []
    rejects: dict[str, int] = {
        "annotation_overlap": 0,
        "head_like_region_below": 0,
        "inside_helmet_typical_range": 0,
    }

    for image_id in sorted(image_records):
        hsv = load_hsv(image_records[image_id]["file_name"])
        mask = cv2.inRange(
            hsv,
            np.array([hue_low, saturation_low, value_low], dtype=np.uint8),
            np.array([hue_high, 255, 255], dtype=np.uint8),
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            circularity = (
                4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0
            )
            if not area_low <= area <= area_high:
                continue
            if not circ_low <= circularity <= circ_high:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            bbox = [float(x), float(y), float(width), float(height)]
            image_annotations = annotations.get(image_id, [])
            max_iou = max(
                (_xywh_iou(bbox, a["bbox"]) for a in image_annotations), default=0.0
            )
            if max_iou > max_iou_allowed:
                rejects["annotation_overlap"] += 1
                continue
            if _has_head_or_person_below(bbox, image_annotations, category_names) or (
                _skin_like_below(hsv, bbox)
            ):
                rejects["head_like_region_below"] += 1
                continue
            if not _outside_helmet_typical_range(bbox, helmet_geometry):
                rejects["inside_helmet_typical_range"] += 1
                continue
            regions.append(
                Region(
                    image_id=int(image_id),
                    file_name=str(image_records[image_id]["file_name"]),
                    bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                    circularity=circularity,
                    max_iou_with_annotation=max_iou,
                )
            )
    return regions, rejects


# spec: EVAL-16
def count_false_positives(
    detections: Sequence[Mapping[str, Any]],
    regions: Sequence[Region],
    *,
    arm: str,
    score_threshold: float,
) -> ArmFalsePositives:
    """Detections at or above the operating point whose centre lands on a region.

    `>=` matches detection.py and demo.py; a boundary that disagreed by one box
    between the metric and the demo would be a quiet inconsistency.
    """

    by_image: dict[int, list[Region]] = {}
    for region in regions:
        by_image.setdefault(region.image_id, []).append(region)

    hits = 0
    for detection in detections:
        if float(detection["score"]) < score_threshold:
            continue
        candidates = by_image.get(int(detection["image_id"]), ())
        if any(region.contains(detection["bbox"]) for region in candidates):
            hits += 1

    return ArmFalsePositives(
        arm=arm,
        n_false_positives=hits,
        n_regions=len(regions),
        n_images_with_regions=len(by_image),
        score_threshold=score_threshold,
    )


# spec: EVAL-16
def discriminates(results: Sequence[ArmFalsePositives], *, min_spread: int = 2) -> bool:
    """Can this metric tell the arms apart at all?

    A metric that returns 0 or 1 for every arm has not ranked anything; it has
    reported that nothing happened. Printing such a table next to the others
    invites a reader to compare 0.005 against 0.000 as though the difference
    meant something, so the caller is expected to check this and say so.
    """

    counts = [result.n_false_positives for result in results]
    return bool(counts) and (max(counts) - min(counts)) >= min_spread


# spec: EVAL-16
def render_table(results: Sequence[ArmFalsePositives]) -> str:
    """Markdown for the report. Denominator in the header, never implied."""

    if not results:
        return "No arms were analysed."
    first = results[0]
    lines = [
        (
            f"Mined regions: **{first.n_regions}** across "
            f"**{first.n_images_with_regions}** Test images, at score threshold "
            f"`{first.score_threshold}`."
        ),
        "",
        "| Arm | false positives | per image |",
        "|---|---:|---:|",
    ]
    for result in sorted(results, key=lambda r: r.per_image):
        lines.append(
            f"| `{result.arm}` | {result.n_false_positives} | {result.per_image:.3f} |"
        )
    return "\n".join(lines)
