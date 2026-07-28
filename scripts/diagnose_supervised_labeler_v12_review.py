"""Diagnose the exact owner-rejected v12 model-review cells."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from scripts.prepare_supervised_labeler_v12_gt_review import CONFIG_PATH
from scripts.record_supervised_labeler_v12_gt_review import AUDIT_PATH
from scripts.record_supervised_labeler_v12_model_review import (
    OUTPUT_PATH as REVIEW_PATH,
)
from scripts.run_supervised_labeler_v12_model_audit import (
    EVIDENCE_PATH as MODEL_EVIDENCE_PATH,
)
from scripts.train_supervised_labeler import (
    HelmetDataset,
    _evaluation_collate,
    _predict,
)
from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.grounded_labeler import box_iou_xyxy
from src.synthetic.supervised_labeler import filter_prediction_geometry
from src.synthetic.whole_image import canonical_mapping_sha256

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v12_review_diagnosis.json"
)
RAW_SCORE_FLOOR = 0.001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_json(
    path: Path,
    *,
    hash_field: str,
    expected_status: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = dict(payload)
    embedded_sha = str(canonical.pop(hash_field, ""))
    if (
        canonical_mapping_sha256(canonical) != embedded_sha
        or payload.get("status") != expected_status
    ):
        raise RuntimeError(f"Frozen evidence changed: {path}")
    return payload


def _match_details(
    truth: Sequence[Sequence[float]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    match_iou: float,
) -> list[dict[str, Any]]:
    remaining = set(range(len(truth)))
    details = []
    for prediction in sorted(
        predictions,
        key=lambda row: -float(row["score"]),
    ):
        box = [float(value) for value in prediction["box"]]
        if not truth:
            best_index = None
            best_iou = 0.0
        else:
            best_index, best_iou = max(
                (
                    (index, box_iou_xyxy(box, truth_box))
                    for index, truth_box in enumerate(truth)
                ),
                key=lambda row: row[1],
            )
        matched = (
            best_index is not None
            and best_index in remaining
            and best_iou >= match_iou
        )
        if matched:
            remaining.remove(best_index)
        details.append(
            {
                "score": float(prediction["score"]),
                "box": box,
                "best_truth_index": best_index,
                "best_iou": float(best_iou),
                "numerically_matched": matched,
            }
        )
    return details


def _best_raw_candidate(
    *,
    truth_box: Sequence[float],
    predictions: Sequence[tuple[float, Sequence[float]]],
    image_width: int,
    image_height: int,
    score_threshold: float,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = [
        {
            "score": float(score),
            "box": [float(value) for value in box],
            "iou": float(box_iou_xyxy(box, truth_box)),
        }
        for score, box in predictions
    ]
    candidates.sort(key=lambda row: (-row["iou"], -row["score"]))
    best = candidates[0] if candidates else None
    if best is None:
        return {
            "raw_candidate_count": 0,
            "best_candidate": None,
            "cause": "no_candidate_at_raw_score_floor",
        }
    geometry_kept = filter_prediction_geometry(
        [(best["score"], best["box"])],
        image_width=image_width,
        image_height=image_height,
        max_relative_area=float(geometry["max_relative_area"]),
        max_relative_height=float(geometry["max_relative_height"]),
        min_aspect_ratio=float(geometry["min_aspect_ratio"]),
        max_aspect_ratio=float(geometry["max_aspect_ratio"]),
    )
    passes_geometry = bool(geometry_kept)
    passes_score = float(best["score"]) >= score_threshold
    localizes_truth = float(best["iou"]) >= 0.5
    failure_reasons = []
    if not localizes_truth:
        cause = "model_did_not_localize_truth"
    else:
        if not passes_score:
            failure_reasons.append("below_score_threshold")
        if not passes_geometry:
            failure_reasons.append("removed_by_geometry")
        cause = (
            "candidate_should_have_survived_investigate_pipeline"
            if not failure_reasons
            else "candidate_" + "_and_".join(failure_reasons)
        )
    x1, y1, x2, y2 = (float(value) for value in truth_box)
    return {
        "raw_candidate_count": len(candidates),
        "best_candidate": {
            **best,
            "localizes_truth_at_iou_0_50": localizes_truth,
            "passes_score_threshold": passes_score,
            "passes_geometry": passes_geometry,
        },
        "truth_relative_area": (
            max(x2 - x1, 0.0)
            * max(y2 - y1, 0.0)
            / max(image_width * image_height, 1)
        ),
        "truth_relative_height": max(y2 - y1, 0.0) / max(image_height, 1),
        "failure_reasons": failure_reasons,
        "cause": cause,
    }


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"v12 diagnosis already exists: {OUTPUT_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("v12 miss diagnosis requires CUDA")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registration = config["model_audit_registration"]
    review = _verified_json(
        REVIEW_PATH,
        hash_field="review_sha256",
        expected_status="rejected_by_kuotunyu",
    )
    evidence = _verified_json(
        MODEL_EVIDENCE_PATH,
        hash_field="evidence_sha256",
        expected_status="v12_frozen_model_audit_evidence",
    )
    audit = _verified_json(
        AUDIT_PATH,
        hash_field="manifest_sha256",
        expected_status="v12_adjudicated_audit_frozen_before_model_inference",
    )
    if (
        review["model_evidence_sha256"] != evidence["evidence_sha256"]
        or review["audit_manifest_sha256"] != audit["manifest_sha256"]
    ):
        raise RuntimeError("v12 diagnosis inputs disagree")
    by_cell = {int(row["cell"]): row for row in evidence["cases"]}
    false_positive_cells = review["categories"][
        "model_false_positive_cells"
    ]
    threshold = float(registration["score_threshold"])
    match_iou = float(registration["match_iou"])
    false_positive_diagnoses = []
    for cell in false_positive_cells:
        case = by_cell[int(cell)]
        details = _match_details(
            case["truth_boxes"],
            case["model_predictions"],
            match_iou=match_iou,
        )
        unmatched = sum(not row["numerically_matched"] for row in details)
        false_positive_diagnoses.append(
            {
                "cell": int(cell),
                "image_id": int(case["image_id"]),
                "stratum": str(case["stratum"]),
                "truth_box_count": len(case["truth_boxes"]),
                "model_box_count": len(case["model_predictions"]),
                "numerically_unmatched_model_boxes": unmatched,
                "numerically_matched_model_boxes": len(details) - unmatched,
                "owner_gt_semantics_conflict": (
                    bool(case["model_predictions"]) and unmatched == 0
                ),
                "prediction_details": details,
            }
        )

    missed_cells = review["categories"]["model_missed_helmeted_head_cells"]
    missed_cases = [by_cell[int(cell)] for cell in missed_cells]
    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    missed_ids = [int(row["image_id"]) for row in missed_cases]
    if test_ids & set(missed_ids):
        raise RuntimeError("Val/Test leakage entered v12 diagnosis")
    helmet_category_id = next(
        int(row["id"])
        for row in coco["categories"]
        if str(row["name"]) == "helmet"
    )
    dataset = HelmetDataset(
        image_ids=missed_ids,
        images=train_images,
        annotations=annotations,
        image_root=paths.hardhat_raw,
        helmet_category_id=helmet_category_id,
        input_normalization=config["input_normalization"],
    )
    checkpoint_dir = Path(str(registration["checkpoint_path"]))
    checkpoint_path = checkpoint_dir / "model.safetensors"
    if _sha256(checkpoint_path) != registration["checkpoint_sha256"]:
        raise RuntimeError("Registered checkpoint changed before diagnosis")
    processor = AutoImageProcessor.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    )
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
    ).to("cuda")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: _evaluation_collate(processor, batch),
    )
    predicted_ids, raw_truth, raw_predictions = _predict(
        model=model,
        processor=processor,
        loader=loader,
        device="cuda",
        score_floor=RAW_SCORE_FLOOR,
        geometry_filter=None,
    )
    if predicted_ids != missed_ids:
        raise RuntimeError("v12 miss diagnosis order changed")
    missed_diagnoses = []
    for cell, case in zip(missed_cells, missed_cases, strict=True):
        image_id = int(case["image_id"])
        image = train_images[image_id]
        if raw_truth[image_id] != case["truth_boxes"]:
            raise RuntimeError("v12 truth changed during miss diagnosis")
        missed_diagnoses.append(
            {
                "cell": int(cell),
                "image_id": image_id,
                "group_id": int(frozen[image_id]["group_id"]),
                "truth_box_diagnoses": [
                    _best_raw_candidate(
                        truth_box=truth_box,
                        predictions=raw_predictions[image_id],
                        image_width=int(image["width"]),
                        image_height=int(image["height"]),
                        score_threshold=threshold,
                        geometry=registration["postprocessing"],
                    )
                    for truth_box in case["truth_boxes"]
                ],
            }
        )

    conflicts = [
        int(row["cell"])
        for row in false_positive_diagnoses
        if row["owner_gt_semantics_conflict"]
    ]
    diagnosis = {
        "schema_version": 1,
        "status": "v12_owner_review_diagnosed",
        "experiment_id": "supervised_labeler_v12",
        "owner_review_sha256": str(review["review_sha256"]),
        "model_evidence_sha256": str(evidence["evidence_sha256"]),
        "audit_manifest_sha256": str(audit["manifest_sha256"]),
        "checkpoint_sha256": str(registration["checkpoint_sha256"]),
        "score_threshold": threshold,
        "raw_diagnostic_score_floor": RAW_SCORE_FLOOR,
        "match_iou": match_iou,
        "postprocessing": registration["postprocessing"],
        "false_positive_diagnoses": false_positive_diagnoses,
        "missed_helmet_diagnoses": missed_diagnoses,
        "owner_gt_semantics_conflict_cells": conflicts,
        "numeric_audit_disposition": (
            "diagnostic_only_due_owner_gt_semantics_conflicts"
            if conflicts
            else "owner_rejected"
        ),
        "revealed_problem_images_read_for_gpu_diagnosis": len(missed_ids),
        "new_audit_images_read": 0,
        "sealed_reserve_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    diagnosis["diagnosis_sha256"] = canonical_mapping_sha256(diagnosis)
    OUTPUT_PATH.write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(diagnosis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
