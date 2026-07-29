"""Diagnose the final owner-reported v15 model-review outcome on CPU.

Only frozen sidecar evidence is read. No model inference is run. Cell 11 is
the sole confirmed model failure; cells 29 and 38 are ambiguous GT samples
that must be quarantined rather than counted against the model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import box_iou_xyxy

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v15.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v15_split.json"
REVIEW_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_model_human_review.json"
)
TRAINING_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v15_training.json"
EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_audit_evidence.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v15_review_diagnosis.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Diagnosis already exists: {OUTPUT_PATH}")
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        review["categories"]["model_false_positive_cells"] != [11]
        or review["categories"]["model_missed_helmeted_head_cells"] != []
        or review["categories"]["ambiguous_dataset_gt_quarantine_cells"]
        != [29, 38]
        or review["problem_cells"] != [11, 29, 38]
        or training["audit_evidence_sha256"] != _sha256(EVIDENCE_PATH)
    ):
        raise RuntimeError("Frozen v15 owner outcome changed")

    evidence_by_cell = {
        int(row["cell"]): row for row in evidence["cases"]
    }
    cell_11 = evidence_by_cell[11]
    if int(cell_11["image_id"]) != 4100 or cell_11["truth_boxes"]:
        raise RuntimeError("Frozen v15 cell 11 changed")
    false_positives = [
        {
            "box": [float(value) for value in row["box"]],
            "score": float(row["score"]),
            "minimum_exclusive_threshold": float(row["score"]) + 1e-9,
        }
        for row in cell_11["model_predictions"]
    ]
    if [row["score"] for row in false_positives] != [
        0.05029296875,
        0.048095703125,
    ]:
        raise RuntimeError("Frozen v15 cell 11 predictions changed")

    ambiguous_cases: list[dict[str, Any]] = []
    expected = {
        29: {
            "image_id": 787,
            "group_id": 785,
            "matched_truth_index": 1,
            "reason": "occlusion_makes_boxing_optional",
        },
        38: {
            "image_id": 243,
            "group_id": 242,
            "matched_truth_index": 0,
            "reason": "viewing_angle_makes_helmet_identity_uncertain",
        },
    }
    for cell, expected_row in expected.items():
        row = evidence_by_cell[cell]
        if int(row["image_id"]) != expected_row["image_id"]:
            raise RuntimeError(f"Frozen v15 cell {cell} changed")
        matched_truth_index = int(expected_row["matched_truth_index"])
        matched_iou = box_iou_xyxy(
            row["model_predictions"][0]["box"],
            row["truth_boxes"][matched_truth_index],
        )
        ambiguous_cases.append(
            {
                "cell": cell,
                **expected_row,
                "matched_iou": float(matched_iou),
                "action": "quarantine_complete_image_and_group",
                "count_as_model_failure": False,
            }
        )

    output = {
        "schema_version": 1,
        "status": "v15_owner_outcome_diagnosed_without_new_inference",
        "scope": {
            "already_revealed_audit_cells_only": [11, 29, 38],
            "sidecar_images_read": 3,
            "source_image_pixels_read": 0,
            "model_inference_run": False,
            "validation_images_read": 0,
            "test_images_read": 0,
            "whole_image_generation_run": False,
        },
        "confirmed_model_failure": {
            "cell": 11,
            "image_id": 4100,
            "group_id": 3971,
            "truth_box_count": 0,
            "false_positive_count": 2,
            "false_positives": false_positives,
            "diagnosed_cause": "unmatched_predictions_on_empty_gt",
        },
        "ambiguous_gt_quarantines": ambiguous_cases,
        "cause_counts": {
            "unmatched_prediction_above_threshold": 2,
            "ambiguous_dataset_gt_complete_image_quarantine": 2,
        },
        "v16_intervention": {
            "hard_negative_error_replay_image_ids": [210, 361, 4100],
            "hard_negative_error_replay_weight": 12.0,
            "positive_error_replay_image_ids": [361, 2534, 3605],
            "positive_error_replay_weight": 12.0,
            "overlap_policy": "maximum_weight",
            "new_quarantined_image_ids": [243, 787],
            "new_quarantined_group_ids": [242, 785],
            "score_threshold_grid_change": False,
            "reason": (
                "A threshold above 0.05029296975 would be required to remove "
                "both confirmed cell-11 false positives. Do not tune the "
                "threshold on revealed audit outcomes; use exact revealed "
                "hard-negative replay and fresh independent audit data."
            ),
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
            "evidence_path": str(EVIDENCE_PATH.relative_to(PROJECT_ROOT)),
            "evidence_sha256": _sha256(EVIDENCE_PATH),
            "checkpoint_sha256": training["checkpoint_sha256"],
            "score_threshold": training["best_calibration"]["threshold"],
        },
        "generation_allowed": False,
    }
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
