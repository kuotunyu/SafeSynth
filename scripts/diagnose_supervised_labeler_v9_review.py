"""Diagnose owner-reported v9 failures on revealed Train-only evidence."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.diagnose_supervised_labeler_v8_review import (
    _best_raw_candidate,
    _greedy_partition,
    _miss_reason,
)
from scripts.train_supervised_labeler import (
    _aggregate,
    _build_datasets,
    _evaluation_collate,
    _predict,
    _sha256,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.supervised_labeler import filter_prediction_geometry

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v9.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v9_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v9_training.json"
REVIEW_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v9_human_review.json"
AUDIT_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v9_audit_evidence.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v9_review_diagnosis.json"
)
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v9_review_diagnosis.md"
)
SCORE_FLOOR = 0.001
OWNER_CATEGORIES = {
    6: "background_or_other_false_positive",
    11: "missed_helmet",
    12: "background_or_other_false_positive",
    37: "missed_helmet",
}
THRESHOLDS = (
    0.010,
    0.015,
    0.020,
    0.030,
    0.040,
    0.050,
    0.056,
    0.060,
    0.070,
    0.080,
    0.100,
)


def main() -> None:
    if OUTPUT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("v9 owner-review diagnosis evidence already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("v9 owner-review diagnosis requires CUDA")

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
        raise RuntimeError("Expected the frozen v9 owner rejection")

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
        raise RuntimeError("Passed v9 checkpoint changed")

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
            raise RuntimeError(f"v9 audit reproduction changed: {metric}")

    cases_by_cell = {int(case["cell"]): case for case in sidecar["cases"]}
    problem_cases = []
    miss_reason_counts: dict[str, int] = {}
    owner_false_positive_scores = []
    for cell in review["problem_cells"]:
        case = cases_by_cell[int(cell)]
        image_id = int(case["image_id"])
        image = train_images[image_id]
        accepted = [
            (float(row["score"]), [float(value) for value in row["box"]])
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
        "status": "diagnostic_on_revealed_v9_train_only_audit",
        "eligible_for_generation_gate": False,
        "reason": (
            "The 48 audit images were revealed to kuotunyu. This diagnosis "
            "may motivate a new preregistered experiment but cannot pass a gate."
        ),
        "experiment_id": "supervised_labeler_v9",
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
        "# Supervised labeler v9 owner-review diagnosis",
        "",
        "- Evidence class: **revealed Train-only audit; not gate-eligible**",
        f"- Owner problem cells: **{', '.join(str(v) for v in review['problem_cells'])}**",
        (
            "- Automatically unmatched truths in the four reported cells: "
            f"**{payload['automatically_unmatched_truth_instances']}**"
        ),
        (
            "- Miss cause counts: "
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
