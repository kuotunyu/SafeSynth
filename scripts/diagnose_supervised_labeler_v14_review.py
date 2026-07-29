"""Diagnose the four owner-reported v14 model-review failures.

Only the already revealed independent-audit cells named in the frozen owner
review are read. The frozen v14 checkpoint is run at a low score floor to
separate score-threshold, geometry-filter, and localization failures.
"""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.train_supervised_labeler import (
    _build_datasets,
    _evaluation_collate,
    _predict,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import box_iou_xyxy
from src.synthetic.supervised_labeler import filter_prediction_geometry

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v14.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v14_split.json"
REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_model_human_review.json"
)
TRAINING_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v14_training.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v14_review_diagnosis.json"
)
LOW_SCORE_FLOOR = 0.001
EXPECTED_FALSE_POSITIVE_CELLS = [10]
EXPECTED_MISSED_CELLS = [7, 40, 43]
EXPECTED_CELLS = sorted(
    EXPECTED_FALSE_POSITIVE_CELLS + EXPECTED_MISSED_CELLS
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _passes_geometry(
    row: tuple[float, Sequence[float]],
    *,
    width: int,
    height: int,
    geometry: dict[str, Any],
) -> bool:
    return bool(
        filter_prediction_geometry(
            [row],
            image_width=width,
            image_height=height,
            max_relative_area=float(geometry["max_relative_area"]),
            max_relative_height=float(geometry["max_relative_height"]),
            min_aspect_ratio=float(geometry["min_aspect_ratio"]),
            max_aspect_ratio=float(geometry["max_aspect_ratio"]),
        )
    )


def _greedy_unmatched(
    truth_boxes: Sequence[Sequence[float]],
    predictions: Sequence[tuple[float, Sequence[float]]],
    *,
    score_threshold: float,
    match_iou: float,
) -> tuple[list[int], list[tuple[float, Sequence[float]]]]:
    remaining_truths = set(range(len(truth_boxes)))
    false_positives = []
    for score, prediction in sorted(predictions, key=lambda item: -item[0]):
        if float(score) < score_threshold:
            continue
        if not remaining_truths:
            false_positives.append((score, prediction))
            continue
        best_index, best_iou = max(
            (
                (index, box_iou_xyxy(prediction, truth_boxes[index]))
                for index in remaining_truths
            ),
            key=lambda item: item[1],
        )
        if best_iou >= match_iou:
            remaining_truths.remove(best_index)
        else:
            false_positives.append((score, prediction))
    return sorted(remaining_truths), false_positives


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Diagnosis already exists: {OUTPUT_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("v14 review diagnosis requires an available CUDA GPU")

    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    if (
        review["categories"]["model_false_positive_cells"]
        != EXPECTED_FALSE_POSITIVE_CELLS
        or review["categories"]["model_missed_helmeted_head_cells"]
        != EXPECTED_MISSED_CELLS
        or review["problem_cells"] != EXPECTED_CELLS
    ):
        raise RuntimeError("Frozen v14 owner review changed")

    config, _, _, _, _, _, _, audit = _build_datasets(
        config_path=CONFIG_PATH,
        split_path=SPLIT_PATH,
    )
    expected_image_ids = {
        int(row["cell"]): int(row["image_id"])
        for row in review["problem_cases"]
    }
    selected_rows = []
    for cell in EXPECTED_CELLS:
        dataset_item = audit[cell - 1]
        if int(dataset_item["image_id"]) != expected_image_ids[cell]:
            raise RuntimeError(f"Frozen v14 audit cell {cell:02d} changed")
        selected_rows.append(dataset_item)

    checkpoint_dir = Path(training["checkpoint_path"])
    checkpoint_path = checkpoint_dir / "model.safetensors"
    if _sha256(checkpoint_path) != training["checkpoint_sha256"]:
        raise RuntimeError("Frozen v14 checkpoint hash changed")

    processor = AutoImageProcessor.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    ).to("cuda")
    loader = DataLoader(
        selected_rows,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: _evaluation_collate(processor, batch),
    )
    image_ids, truth, raw_predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=LOW_SCORE_FLOOR,
        geometry_filter=None,
    )

    score_threshold = float(training["best_calibration"]["threshold"])
    match_iou = float(config["calibration"]["match_iou"])
    geometry = dict(config["postprocessing"])
    cases = []
    cause_counts: dict[str, int] = {}
    for cell, image_id, dataset_item in zip(
        EXPECTED_CELLS,
        image_ids,
        selected_rows,
        strict=True,
    ):
        width = int(dataset_item["image"].width)
        height = int(dataset_item["image"].height)
        raw = raw_predictions[image_id]
        geometry_rows = [
            row
            for row in raw
            if _passes_geometry(
                row,
                width=width,
                height=height,
                geometry=geometry,
            )
        ]
        unmatched_truths, false_positives = _greedy_unmatched(
            truth[image_id],
            geometry_rows,
            score_threshold=score_threshold,
            match_iou=match_iou,
        )
        misses = []
        for truth_index in unmatched_truths:
            truth_box = truth[image_id][truth_index]
            ranked = sorted(
                (
                    (
                        float(box_iou_xyxy(box, truth_box)),
                        float(score),
                        [float(value) for value in box],
                    )
                    for score, box in raw
                ),
                reverse=True,
            )
            best_iou, best_score, best_box = ranked[0]
            passes_geometry = _passes_geometry(
                (best_score, best_box),
                width=width,
                height=height,
                geometry=geometry,
            )
            blockers = []
            if best_iou < match_iou:
                cause = "no_localized_candidate_at_iou_0_50"
            else:
                if best_score < score_threshold:
                    blockers.append("score_threshold")
                if not passes_geometry:
                    blockers.append("geometry_filter")
                if blockers == ["score_threshold", "geometry_filter"]:
                    cause = "candidate_below_score_and_rejected_by_geometry"
                elif blockers == ["score_threshold"]:
                    cause = "candidate_below_score_threshold"
                elif blockers == ["geometry_filter"]:
                    cause = "candidate_rejected_by_geometry_filter"
                else:
                    cause = "candidate_lost_during_greedy_matching"
            cause_counts[cause] = cause_counts.get(cause, 0) + 1
            misses.append(
                {
                    "truth_index": truth_index,
                    "truth_box": [float(value) for value in truth_box],
                    "best_raw_candidate": {
                        "box": best_box,
                        "score": best_score,
                        "iou": best_iou,
                        "passes_current_geometry": passes_geometry,
                        "passes_current_score": best_score >= score_threshold,
                    },
                    "blocking_stages": blockers,
                    "diagnosed_cause": cause,
                }
            )
        false_positive_rows = [
            {
                "box": [float(value) for value in box],
                "score": float(score),
                "minimum_exclusive_threshold": float(score) + 1e-9,
            }
            for score, box in false_positives
        ]
        if false_positive_rows:
            cause_counts["unmatched_prediction_above_threshold"] = (
                cause_counts.get("unmatched_prediction_above_threshold", 0)
                + len(false_positive_rows)
            )
        cases.append(
            {
                "cell": cell,
                "image_id": image_id,
                "owner_category": (
                    "model_false_positive"
                    if cell in EXPECTED_FALSE_POSITIVE_CELLS
                    else "model_missed_helmeted_head"
                ),
                "image_size": [width, height],
                "truth_box_count": len(truth[image_id]),
                "raw_candidate_count_at_0_001": len(raw),
                "geometry_candidate_count_at_0_001": len(geometry_rows),
                "unmatched_truth_count_at_frozen_settings": len(
                    unmatched_truths
                ),
                "false_positive_count_at_frozen_settings": len(
                    false_positive_rows
                ),
                "misses": misses,
                "false_positives": false_positive_rows,
            }
        )

    output = {
        "schema_version": 1,
        "status": "v14_owner_failure_diagnosis_complete",
        "scope": {
            "already_revealed_audit_cells_only": EXPECTED_CELLS,
            "images_read": len(EXPECTED_CELLS),
            "validation_images_read": 0,
            "test_images_read": 0,
            "whole_image_generation_run": False,
        },
        "frozen_inputs": {
            "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "config_sha256": _sha256(CONFIG_PATH),
            "split_path": str(SPLIT_PATH.relative_to(PROJECT_ROOT)),
            "split_sha256": _sha256(SPLIT_PATH),
            "review_path": str(REVIEW_PATH.relative_to(PROJECT_ROOT)),
            "review_sha256": _sha256(REVIEW_PATH),
            "training_path": str(TRAINING_PATH.relative_to(PROJECT_ROOT)),
            "training_sha256": _sha256(TRAINING_PATH),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "score_threshold": score_threshold,
            "match_iou": match_iou,
            "low_score_floor": LOW_SCORE_FLOOR,
            "postprocessing": geometry,
        },
        "cause_counts": cause_counts,
        "cases": cases,
        "generation_allowed": False,
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
