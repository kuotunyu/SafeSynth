"""Diagnose owner-reported v8 failures on revealed Train-only evidence."""

from __future__ import annotations

import gc
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.train_supervised_labeler import (
    _aggregate,
    _build_datasets,
    _evaluation_collate,
    _predict,
    _sha256,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import box_iou_xyxy
from src.synthetic.supervised_labeler import filter_prediction_geometry

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v8.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v8_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v8_training.json"
REVIEW_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v8_human_review.json"
AUDIT_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v8_audit_evidence.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v8_review_diagnosis.json"
)
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v8_review_diagnosis.md"
)
SCORE_FLOOR = 0.001
OWNER_CATEGORIES = {
    1: "background_or_other_false_positive",
    6: "background_or_other_false_positive",
    10: "background_or_other_false_positive",
    16: "missed_helmet",
    41: "severe_localization_failure",
    42: "missed_helmet",
}
THRESHOLDS = (
    0.035,
    0.040,
    0.045,
    0.050,
    0.060,
    0.070,
    0.080,
    0.083,
    0.090,
    0.100,
)


def _geometry_kept(
    prediction: tuple[float, Sequence[float]],
    *,
    width: int,
    height: int,
    settings: dict[str, Any],
) -> bool:
    return bool(
        filter_prediction_geometry(
            [prediction],
            image_width=width,
            image_height=height,
            max_relative_area=float(settings["max_relative_area"]),
            max_relative_height=float(settings["max_relative_height"]),
            min_aspect_ratio=float(settings["min_aspect_ratio"]),
            max_aspect_ratio=float(settings["max_aspect_ratio"]),
        )
    )


def _greedy_partition(
    truth: Sequence[Sequence[float]],
    predictions: Sequence[tuple[float, Sequence[float]]],
    *,
    threshold: float,
    match_iou: float,
) -> dict[str, Any]:
    remaining = set(range(len(truth)))
    matches = []
    false_positives = []
    filtered = sorted(
        (
            (float(score), [float(value) for value in box])
            for score, box in predictions
            if float(score) >= threshold
        ),
        key=lambda row: -row[0],
    )
    for score, box in filtered:
        if not remaining:
            false_positives.append(
                {"score": score, "box": box, "best_truth_iou": 0.0}
            )
            continue
        best_index, best_iou = max(
            (
                (index, box_iou_xyxy(box, truth[index]))
                for index in remaining
            ),
            key=lambda row: row[1],
        )
        if best_iou >= match_iou:
            remaining.remove(best_index)
            matches.append(
                {
                    "truth_index": int(best_index),
                    "score": score,
                    "box": box,
                    "iou": float(best_iou),
                }
            )
        else:
            false_positives.append(
                {
                    "score": score,
                    "box": box,
                    "best_truth_iou": float(best_iou),
                }
            )
    return {
        "matches": matches,
        "false_positives": false_positives,
        "unmatched_truth_indices": sorted(remaining),
    }


def _best_raw_candidate(
    truth_box: Sequence[float],
    raw_predictions: Sequence[tuple[float, Sequence[float]]],
    *,
    width: int,
    height: int,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        {
            "iou": float(box_iou_xyxy(truth_box, box)),
            "score": float(score),
            "box": [float(value) for value in box],
            "geometry_kept": _geometry_kept(
                (score, box),
                width=width,
                height=height,
                settings=geometry,
            ),
        }
        for score, box in raw_predictions
    ]
    return max(
        candidates,
        key=lambda row: (float(row["iou"]), float(row["score"])),
        default={"iou": 0.0, "score": 0.0, "box": [], "geometry_kept": False},
    )


def _miss_reason(
    candidate: dict[str, Any],
    *,
    threshold: float,
    match_iou: float,
) -> str:
    if float(candidate["iou"]) < match_iou:
        return "no_matching_localization"
    below_threshold = float(candidate["score"]) < threshold
    removed_by_geometry = not bool(candidate["geometry_kept"])
    if below_threshold and removed_by_geometry:
        return "below_score_threshold_and_removed_by_geometry_filter"
    if below_threshold:
        return "matching_box_below_frozen_score_threshold"
    if removed_by_geometry:
        return "removed_by_frozen_geometry_filter"
    return "matching_box_consumed_by_another_truth"


def main() -> None:
    if OUTPUT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("v8 owner-review diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("v8 owner-review diagnosis requires CUDA")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    sidecar = json.loads(AUDIT_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        report["status"] != "supervised_labeler_audit_passed"
        or review["status"] != "rejected_by_kuotunyu"
        or review["problem_cells"] != sorted(OWNER_CATEGORIES)
        or int(report["validation_images_read"]) != 0
        or int(report["test_images_read"]) != 0
    ):
        raise RuntimeError("Expected the frozen v8 owner rejection")

    (
        config,
        split,
        _,
        train_images,
        _,
        _,
        _,
        consumed_audit,
    ) = _build_datasets(config_path=CONFIG_PATH, split_path=SPLIT_PATH)
    checkpoint = Path(report["checkpoint_path"])
    if _sha256(checkpoint / "model.safetensors") != report["checkpoint_sha256"]:
        raise RuntimeError("Passed v8 checkpoint changed")

    processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to("cuda")
    loader = DataLoader(
        consumed_audit,
        batch_size=int(config["optimization"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: _evaluation_collate(processor, batch),
    )
    image_ids, truth, raw_predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=SCORE_FLOOR,
        geometry_filter=None,
    )
    geometry = config["postprocessing"]
    filtered_predictions = {}
    for image_id in image_ids:
        image = train_images[int(image_id)]
        filtered_predictions[int(image_id)] = filter_prediction_geometry(
            raw_predictions[int(image_id)],
            image_width=int(image["width"]),
            image_height=int(image["height"]),
            max_relative_area=float(geometry["max_relative_area"]),
            max_relative_height=float(geometry["max_relative_height"]),
            min_aspect_ratio=float(geometry["min_aspect_ratio"]),
            max_aspect_ratio=float(geometry["max_aspect_ratio"]),
        )

    frozen_threshold = float(report["best_calibration"]["threshold"])
    match_iou = float(config["calibration"]["match_iou"])
    reproduced = _aggregate(
        image_ids=image_ids,
        truth=truth,
        predictions=filtered_predictions,
        threshold=frozen_threshold,
        match_iou=match_iou,
    )
    for metric in ("precision", "recall", "f1", "median_matched_iou"):
        if abs(
            float(reproduced[metric]) - float(report["audit_metrics"][metric])
        ) > 1e-12:
            raise RuntimeError(f"v8 audit reproduction changed: {metric}")

    cases_by_cell = {int(case["cell"]): case for case in sidecar["cases"]}
    problem_cases = []
    miss_reason_counts: dict[str, int] = {}
    owner_false_positive_scores = []
    for cell in review["problem_cells"]:
        case = cases_by_cell[int(cell)]
        image_id = int(case["image_id"])
        image = train_images[image_id]
        accepted = [
            (float(row["score"]), [float(v) for v in row["box"]])
            for row in case["model_predictions"]
        ]
        partition = _greedy_partition(
            truth[image_id],
            accepted,
            threshold=frozen_threshold,
            match_iou=match_iou,
        )
        misses = []
        for truth_index in partition["unmatched_truth_indices"]:
            truth_box = truth[image_id][int(truth_index)]
            candidate = _best_raw_candidate(
                truth_box,
                raw_predictions[image_id],
                width=int(image["width"]),
                height=int(image["height"]),
                geometry=geometry,
            )
            reason = _miss_reason(
                candidate,
                threshold=frozen_threshold,
                match_iou=match_iou,
            )
            miss_reason_counts[reason] = miss_reason_counts.get(reason, 0) + 1
            misses.append(
                {
                    "truth_index": int(truth_index),
                    "truth_box": [float(value) for value in truth_box],
                    "best_raw_candidate": candidate,
                    "reason": reason,
                }
            )
        if OWNER_CATEGORIES[int(cell)] == "background_or_other_false_positive":
            owner_false_positive_scores.extend(
                float(row["score"]) for row in partition["false_positives"]
            )
        problem_cases.append(
            {
                "cell": int(cell),
                "image_id": image_id,
                "owner_category": OWNER_CATEGORIES[int(cell)],
                "accepted_matches": partition["matches"],
                "accepted_false_positives": partition["false_positives"],
                "unmatched_truth_count": len(
                    partition["unmatched_truth_indices"]
                ),
                "misses": misses,
            }
        )

    threshold_grid = [
        {
            "threshold": threshold,
            **_aggregate(
                image_ids=image_ids,
                truth=truth,
                predictions=filtered_predictions,
                threshold=threshold,
                match_iou=match_iou,
            ),
        }
        for threshold in THRESHOLDS
    ]
    payload = {
        "schema_version": 1,
        "status": "diagnostic_on_revealed_v8_train_only_audit",
        "eligible_for_generation_gate": False,
        "reason": (
            "The 48 audit images were revealed to kuotunyu. This diagnosis "
            "may motivate a new preregistered experiment but cannot pass a gate."
        ),
        "experiment_id": "supervised_labeler_v8",
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_split_manifest_sha256": split["manifest_sha256"],
        "review_evidence_sha256": review["evidence_sha256"],
        "problem_cells": review["problem_cells"],
        "owner_category_counts": {
            category: list(OWNER_CATEGORIES.values()).count(category)
            for category in sorted(set(OWNER_CATEGORIES.values()))
        },
        "automatically_unmatched_truth_instances": sum(
            int(case["unmatched_truth_count"]) for case in problem_cases
        ),
        "automatic_miss_reason_counts": miss_reason_counts,
        "owner_false_positive_scores": sorted(
            owner_false_positive_scores,
            reverse=True,
        ),
        "max_owner_false_positive_score": max(
            owner_false_positive_scores,
            default=0.0,
        ),
        "problem_cases": problem_cases,
        "frozen_threshold": frozen_threshold,
        "score_floor": SCORE_FLOOR,
        "threshold_grid": threshold_grid,
        "reproduced_frozen_audit_metrics": reproduced,
        "revealed_audit_images_read": len(image_ids),
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    lines = [
        "# Supervised labeler v8 owner-review diagnosis",
        "",
        "- Evidence class: **revealed Train-only audit; not gate-eligible**",
        f"- Owner problem cells: **{', '.join(str(v) for v in review['problem_cells'])}**",
        "- Cell 41 category: **severe localization failure**",
        (
            "- Automatically unmatched truths in the six reported cells: "
            f"**{payload['automatically_unmatched_truth_instances']}**"
        ),
        (
            "- Miss/localization cause counts: "
            f"**{json.dumps(miss_reason_counts, sort_keys=True)}**"
        ),
        (
            "- Highest owner-reported false-positive score: "
            f"**{payload['max_owner_false_positive_score']:.4f}**"
        ),
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image generations: **0**",
        "",
        "| Cell | Owner category | Numeric FP | Numeric FN | Best raw IoU / score / cause |",
        "|---:|---|---:|---:|---|",
    ]
    for case in problem_cases:
        details = "; ".join(
            (
                f"{float(miss['best_raw_candidate']['iou']):.3f} / "
                f"{float(miss['best_raw_candidate']['score']):.4f} / "
                f"{miss['reason']}"
            )
            for miss in case["misses"]
        )
        lines.append(
            f"| {case['cell']} | {case['owner_category']} | "
            f"{len(case['accepted_false_positives'])} | "
            f"{case['unmatched_truth_count']} | {details or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Revealed audit threshold sensitivity",
            "",
            "| Threshold | TP | FP | FN | Precision | Recall | F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_grid:
        lines.append(
            f"| {float(row['threshold']):.3f} | "
            f"{int(row['true_positives'])} | "
            f"{int(row['false_positives'])} | "
            f"{int(row['false_negatives'])} | "
            f"{float(row['precision']):.4f} | "
            f"{float(row['recall']):.4f} | "
            f"{float(row['f1']):.4f} |"
        )
    MARKDOWN_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
