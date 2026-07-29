"""Diagnose the three owner-reported v13 worn-helmet misses.

This script re-reads only the already revealed independent-audit images named
in the frozen owner review. It runs the frozen v13 checkpoint at a deliberately
low score floor, both before and after the preregistered geometry filter, so a
future intervention can distinguish score, geometry, and localization failures.
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

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v13.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v13_split.json"
REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_model_human_review.json"
)
TRAINING_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v13_training.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v13_review_diagnosis.json"
)
LOW_SCORE_FLOOR = 0.001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _greedy_unmatched_truths(
    truth_boxes: Sequence[Sequence[float]],
    predictions: Sequence[tuple[float, Sequence[float]]],
    *,
    score_threshold: float,
    match_iou: float,
) -> list[int]:
    remaining = set(range(len(truth_boxes)))
    for score, prediction in sorted(predictions, key=lambda item: -item[0]):
        if float(score) < score_threshold or not remaining:
            continue
        best_index, best_iou = max(
            (
                (index, box_iou_xyxy(prediction, truth_boxes[index]))
                for index in remaining
            ),
            key=lambda item: item[1],
        )
        if best_iou >= match_iou:
            remaining.remove(best_index)
    return sorted(remaining)


def _passes_geometry(
    row: tuple[float, Sequence[float]],
    *,
    width: int,
    height: int,
    geometry: dict[str, Any],
) -> bool:
    kept = filter_prediction_geometry(
        [row],
        image_width=width,
        image_height=height,
        max_relative_area=float(geometry["max_relative_area"]),
        max_relative_height=float(geometry["max_relative_height"]),
        min_aspect_ratio=float(geometry["min_aspect_ratio"]),
        max_aspect_ratio=float(geometry["max_aspect_ratio"]),
    )
    return bool(kept)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("v13 review diagnosis requires an available CUDA GPU")

    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    expected_cells = [3, 22, 34]
    if review["categories"]["model_missed_helmeted_head_cells"] != expected_cells:
        raise RuntimeError("Frozen v13 owner review no longer names cells 03, 22, 34")

    config, split, _, _, _, _, _, audit = _build_datasets(
        config_path=CONFIG_PATH,
        split_path=SPLIT_PATH,
    )
    selected_rows = []
    reviewed_image_ids = {
        int(case["cell"]): int(case["image_id"])
        for case in review["problem_cases"]
    }
    for cell in expected_cells:
        dataset_item = audit[cell - 1]
        image_id = int(dataset_item["image_id"])
        expected_image_id = reviewed_image_ids[cell]
        if image_id != expected_image_id:
            raise RuntimeError(
                f"Cell {cell:02d} changed from image {expected_image_id} to {image_id}"
            )
        selected_rows.append(dataset_item)

    checkpoint_dir = Path(training["checkpoint_path"])
    checkpoint_path = checkpoint_dir / "model.safetensors"
    if _sha256(checkpoint_path) != training["checkpoint_sha256"]:
        raise RuntimeError("Frozen v13 checkpoint hash changed")

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
        batch_size=3,
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
        expected_cells,
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
        unmatched = _greedy_unmatched_truths(
            truth[image_id],
            geometry_rows,
            score_threshold=score_threshold,
            match_iou=match_iou,
        )
        misses = []
        for truth_index in unmatched:
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
        cases.append(
            {
                "cell": cell,
                "image_id": image_id,
                "image_size": [width, height],
                "truth_box_count": len(truth[image_id]),
                "raw_candidate_count_at_0_001": len(raw),
                "geometry_candidate_count_at_0_001": len(geometry_rows),
                "unmatched_truth_count_at_frozen_settings": len(unmatched),
                "misses": misses,
            }
        )

    output = {
        "schema_version": 1,
        "status": "v13_owner_miss_diagnosis_complete",
        "scope": {
            "already_revealed_audit_cells_only": expected_cells,
            "images_read": len(expected_cells),
            "validation_images_read": 0,
            "test_images_read": 0,
            "whole_image_generation_run": False,
        },
        "frozen_inputs": {
            "config_path": str(CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "config_sha256": _sha256(CONFIG_PATH),
            "split_path": str(SPLIT_PATH.relative_to(PROJECT_ROOT)),
            "split_manifest_sha256": split["manifest_sha256"],
            "owner_review_path": str(REVIEW_PATH.relative_to(PROJECT_ROOT)),
            "owner_review_sha256": review["review_sha256"],
            "checkpoint_path": str(checkpoint_dir),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "score_threshold": score_threshold,
            "low_score_floor": LOW_SCORE_FLOOR,
            "match_iou": match_iou,
            "geometry_filter": geometry,
        },
        "cause_counts": cause_counts,
        "cases": cases,
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
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
