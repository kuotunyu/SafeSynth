"""Record kuotunyu's final review of the frozen v18 model audit pages."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v18.yaml"
MODEL_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v18_training.json"
)
MODEL_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v18_audit_evidence.json"
)
MODEL_REVIEW_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v18_model_review_manifest.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v18_model_human_review.json"
)
DIAGNOSIS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v18_review_diagnosis.json"
)
FALSE_POSITIVE_CELL = 36
FALSE_POSITIVE_IMAGE_ID = 4618
ACCEPTED_OCCLUDED_MISS_CELL = 29
ACCEPTED_OCCLUDED_MISS_IMAGE_ID = 3981
IMAGE_WIDTH = 416.0
IMAGE_HEIGHT = 416.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_manifest() -> dict[str, Any]:
    manifest = json.loads(
        MODEL_REVIEW_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    canonical = dict(manifest)
    embedded_sha = str(canonical.pop("manifest_sha256", ""))
    if (
        canonical_mapping_sha256(canonical) != embedded_sha
        or manifest.get("status")
        != "supervised_labeler_v18_model_review_pages_frozen"
        or int(manifest.get("reviewed_images", -1)) != 48
        or manifest.get("render_model_inference_run") is not False
    ):
        raise RuntimeError("Frozen v18 model-review manifest changed")
    return manifest


def build_review(
    *,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    reviewed_on: str,
) -> dict[str, Any]:
    """Build the exact one-FP owner rejection and accepted exception."""

    date.fromisoformat(reviewed_on)
    if (
        len(evidence.get("cases", [])) != 48
        or manifest["source_audit_evidence_sha256"]
        != _sha256(MODEL_EVIDENCE_PATH)
    ):
        raise RuntimeError("Frozen v18 model evidence changed")
    evidence_by_cell = {
        int(row["cell"]): row for row in evidence["cases"]
    }
    if (
        int(evidence_by_cell[FALSE_POSITIVE_CELL]["image_id"])
        != FALSE_POSITIVE_IMAGE_ID
        or int(
            evidence_by_cell[ACCEPTED_OCCLUDED_MISS_CELL]["image_id"]
        )
        != ACCEPTED_OCCLUDED_MISS_IMAGE_ID
    ):
        raise RuntimeError("Frozen v18 reviewed cells changed")

    review = {
        "schema_version": 1,
        "status": "rejected_by_kuotunyu",
        "experiment_id": "supervised_labeler_v18",
        "reviewed_by": "kuotunyu",
        "reviewed_on": reviewed_on,
        "decision": "reject",
        "label_semantics": str(manifest["label_semantics"]),
        "problem_count": 1,
        "problem_cells": [FALSE_POSITIVE_CELL],
        "categories": {
            "model_false_positive_cells": [FALSE_POSITIVE_CELL],
            "model_missed_helmeted_head_cells": [],
            "ambiguous_dataset_gt_quarantine_cells": [],
        },
        "problem_cases": [
            {
                "cell": FALSE_POSITIVE_CELL,
                "image_id": FALSE_POSITIVE_IMAGE_ID,
                "category": "model_false_positive",
            }
        ],
        "accepted_exceptions": [
            {
                "cell": ACCEPTED_OCCLUDED_MISS_CELL,
                "image_id": ACCEPTED_OCCLUDED_MISS_IMAGE_ID,
                "observation": "partially_occluded_helmeted_head_not_predicted",
                "decision": "acceptable_due_to_occlusion",
                "counts_as_problem": False,
            }
        ],
        "review_note": (
            "Cell 29 contains one partially occluded missed helmeted head and "
            "is explicitly accepted. Cell 36 contains one large magenta model "
            "box on the lower-right ground/background and fails the required "
            "zero-problem visual gate. All other reviewed cells were accepted."
        ),
        "numeric_audit_status": "passed_but_owner_visual_gate_failed",
        "numeric_audit_report_path": str(
            MODEL_REPORT_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "numeric_audit_report_file_sha256": _sha256(MODEL_REPORT_PATH),
        "model_evidence_path": str(
            MODEL_EVIDENCE_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "model_evidence_file_sha256": _sha256(MODEL_EVIDENCE_PATH),
        "model_review_manifest_path": str(
            MODEL_REVIEW_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        ).replace("\\", "/"),
        "model_review_manifest_file_sha256": _sha256(
            MODEL_REVIEW_MANIFEST_PATH
        ),
        "model_review_manifest_sha256": str(manifest["manifest_sha256"]),
        "checkpoint_sha256": str(manifest["checkpoint_sha256"]),
        "score_threshold": float(manifest["score_threshold"]),
        "pages": manifest["pages"],
        "generation_allowed": False,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    review["review_sha256"] = canonical_mapping_sha256(review)
    return review


def build_diagnosis(
    *,
    evidence: dict[str, Any],
    review: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Explain cell 36 using frozen coordinates without new inference."""

    cell = next(
        row
        for row in evidence["cases"]
        if int(row["cell"]) == FALSE_POSITIVE_CELL
    )
    if (
        int(cell["image_id"]) != FALSE_POSITIVE_IMAGE_ID
        or len(cell["truth_boxes"]) != 1
        or len(cell["model_predictions"]) != 2
    ):
        raise RuntimeError("Cell 36 frozen evidence changed")

    false_positive = min(
        cell["model_predictions"],
        key=lambda row: float(row["score"]),
    )
    x1, y1, x2, y2 = (
        float(value) for value in false_positive["box"]
    )
    width = x2 - x1
    height = y2 - y1
    area = width * height
    clipped_width = max(0.0, min(x2, IMAGE_WIDTH) - max(x1, 0.0))
    clipped_height = max(0.0, min(y2, IMAGE_HEIGHT) - max(y1, 0.0))
    clipped_area = clipped_width * clipped_height
    relative_area = area / (IMAGE_WIDTH * IMAGE_HEIGHT)
    relative_height = height / IMAGE_HEIGHT
    aspect_ratio = width / height
    outside_fraction = 1.0 - clipped_area / area
    threshold = float(evidence["score_threshold"])
    limits = config["postprocessing"]
    passes_frozen_geometry_filter = (
        relative_area <= float(limits["max_relative_area"])
        and relative_height <= float(limits["max_relative_height"])
        and float(limits["min_aspect_ratio"])
        <= aspect_ratio
        <= float(limits["max_aspect_ratio"])
        and outside_fraction <= float(limits["max_outside_fraction"])
    )
    if (
        false_positive["score"] <= threshold
        or not passes_frozen_geometry_filter
    ):
        raise RuntimeError("Cell 36 false-positive diagnosis changed")

    diagnosis = {
        "schema_version": 1,
        "status": "v18_owner_review_diagnosed_without_new_inference",
        "experiment_id": "supervised_labeler_v18",
        "source_owner_review_sha256": str(review["review_sha256"]),
        "scope": "frozen_audit_evidence_only",
        "cell_36": {
            "image_id": FALSE_POSITIVE_IMAGE_ID,
            "image_size": [416, 416],
            "truth_box_count": len(cell["truth_boxes"]),
            "model_prediction_count": len(cell["model_predictions"]),
            "background_false_positive": {
                "box": [x1, y1, x2, y2],
                "score": float(false_positive["score"]),
                "score_threshold": threshold,
                "score_margin_above_threshold": (
                    float(false_positive["score"]) - threshold
                ),
                "relative_area": relative_area,
                "relative_height": relative_height,
                "aspect_ratio": aspect_ratio,
                "outside_fraction": outside_fraction,
                "extends_outside_image": (
                    x1 < 0
                    or y1 < 0
                    or x2 > IMAGE_WIDTH
                    or y2 > IMAGE_HEIGHT
                ),
                "passes_frozen_geometry_filter": (
                    passes_frozen_geometry_filter
                ),
            },
            "root_cause": (
                "The model emitted a low-confidence ground/background box "
                "only slightly above the frozen score threshold. The box is "
                "almost entirely inside the image and its size, height, "
                "aspect ratio, and outside fraction all pass the "
                "preregistered v18 geometry limits."
            ),
            "rendering_bug": False,
        },
        "accepted_cell_29": {
            "image_id": ACCEPTED_OCCLUDED_MISS_IMAGE_ID,
            "decision": "acceptable_due_to_occlusion",
            "counts_as_problem": False,
        },
        "future_intervention_constraint": (
            "Do not change the v18 threshold or geometry after audit. Any "
            "background suppression, harder calibration precision floor, or "
            "replay intervention must be preregistered for a fresh model and "
            "independent audit."
        ),
        "diagnosis_model_inference_run": False,
        "source_image_pixels_read": 0,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    diagnosis["report_sha256"] = canonical_mapping_sha256(diagnosis)
    return diagnosis


def main() -> None:
    if OUTPUT_PATH.exists() or DIAGNOSIS_PATH.exists():
        raise RuntimeError("v18 owner review evidence already exists")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registration = config["model_review_registration"]
    manifest = _verified_manifest()
    report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(MODEL_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        config["status"] != "owner_model_review_pending"
        or registration["owner_review_status"] != "pending_kuotunyu"
        or _sha256(MODEL_REVIEW_MANIFEST_PATH)
        != registration["manifest_file_sha256"]
        or manifest["manifest_sha256"]
        != registration["manifest_sha256"]
        or report.get("status") != "supervised_labeler_audit_passed"
        or not all(report["checks"].values())
        or report["audit_evidence_sha256"]
        != _sha256(MODEL_EVIDENCE_PATH)
    ):
        raise RuntimeError("Configured v18 model outcome changed")

    review = build_review(
        manifest=manifest,
        evidence=evidence,
        reviewed_on="2026-07-30",
    )
    diagnosis = build_diagnosis(
        evidence=evidence,
        review=review,
        config=config,
    )
    OUTPUT_PATH.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    DIAGNOSIS_PATH.write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": review["status"],
                "problem_cells": review["problem_cells"],
                "accepted_exception_cells": [
                    row["cell"] for row in review["accepted_exceptions"]
                ],
                "review_sha256": review["review_sha256"],
                "diagnosis_sha256": diagnosis["report_sha256"],
                "cell_36_score_margin": diagnosis["cell_36"][
                    "background_false_positive"
                ]["score_margin_above_threshold"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
