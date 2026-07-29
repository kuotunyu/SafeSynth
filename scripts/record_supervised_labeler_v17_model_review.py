"""Record kuotunyu's final rejection of the frozen v17 model audit pages."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT
from src.synthetic.whole_image import canonical_mapping_sha256

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v17.yaml"
MODEL_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v17_training.json"
)
MODEL_EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v17_audit_evidence.json"
)
MODEL_REVIEW_MANIFEST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v17_model_review_manifest.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v17_model_human_review.json"
)
DIAGNOSIS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised_labeler_v17_review_diagnosis.json"
)
FALSE_POSITIVE_CELLS = [5, 21, 31]
MISSED_HELMETED_HEAD_CELLS = [4, 12]
PROBLEM_CELLS = sorted(
    FALSE_POSITIVE_CELLS + MISSED_HELMETED_HEAD_CELLS
)
EXPECTED_IMAGE_IDS = {
    4: 857,
    5: 2580,
    12: 4187,
    21: 2941,
    31: 2262,
}


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
        != "supervised_labeler_v17_model_review_pages_frozen"
        or int(manifest.get("reviewed_images", -1)) != 48
        or manifest.get("render_model_inference_run") is not False
    ):
        raise RuntimeError("Frozen v17 model-review manifest changed")
    return manifest


def build_review(
    *,
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    reviewed_on: str,
) -> dict[str, Any]:
    """Build the exact three-FP, two-miss owner rejection."""

    date.fromisoformat(reviewed_on)
    if (
        len(evidence.get("cases", [])) != 48
        or manifest["source_audit_evidence_sha256"]
        != _sha256(MODEL_EVIDENCE_PATH)
    ):
        raise RuntimeError("Frozen v17 model evidence changed")
    evidence_by_cell = {
        int(row["cell"]): row for row in evidence["cases"]
    }
    if {
        cell: int(evidence_by_cell[cell]["image_id"])
        for cell in PROBLEM_CELLS
    } != EXPECTED_IMAGE_IDS:
        raise RuntimeError("Frozen v17 problem cells changed")

    problem_cases = [
        {
            "cell": cell,
            "image_id": EXPECTED_IMAGE_IDS[cell],
            "category": (
                "model_false_positive"
                if cell in FALSE_POSITIVE_CELLS
                else "model_missed_helmeted_head"
            ),
        }
        for cell in PROBLEM_CELLS
    ]
    review = {
        "schema_version": 1,
        "status": "rejected_by_kuotunyu",
        "experiment_id": "supervised_labeler_v17",
        "reviewed_by": "kuotunyu",
        "reviewed_on": reviewed_on,
        "decision": "reject",
        "label_semantics": str(manifest["label_semantics"]),
        "problem_count": len(PROBLEM_CELLS),
        "problem_cells": PROBLEM_CELLS,
        "categories": {
            "model_false_positive_cells": FALSE_POSITIVE_CELLS,
            "model_missed_helmeted_head_cells": (
                MISSED_HELMETED_HEAD_CELLS
            ),
            "ambiguous_dataset_gt_quarantine_cells": [],
        },
        "problem_cases": problem_cases,
        "review_note": (
            "Cells 04 and 12 miss genuine worn helmeted heads. Cells 05, 21, "
            "and 31 contain model boxes on background, ordinary faces, logos, "
            "or unworn objects. Cell 31 includes four oversized background "
            "predictions clipped by the image boundary."
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
) -> dict[str, Any]:
    """Explain cell 31 using frozen coordinates without new inference."""

    cell = next(
        row for row in evidence["cases"] if int(row["cell"]) == 31
    )
    image_width = 416.0
    image_height = 416.0
    predictions = []
    for row in cell["model_predictions"]:
        x1, y1, x2, y2 = (float(value) for value in row["box"])
        width = x2 - x1
        height = y2 - y1
        relative_area = width * height / (image_width * image_height)
        if relative_area <= 0.20:
            continue
        predictions.append(
            {
                "box": [x1, y1, x2, y2],
                "score": float(row["score"]),
                "relative_area": relative_area,
                "relative_height": height / image_height,
                "aspect_ratio": width / height,
                "extends_outside_image": (
                    x1 < 0
                    or y1 < 0
                    or x2 > image_width
                    or y2 > image_height
                ),
                "passes_frozen_geometry_filter": (
                    relative_area <= 0.70
                    and height / image_height <= 0.90
                    and 0.20 <= width / height <= 4.0
                ),
            }
        )
    if len(predictions) != 4 or not all(
        row["passes_frozen_geometry_filter"] for row in predictions
    ):
        raise RuntimeError("Cell 31 oversized predictions changed")

    diagnosis = {
        "schema_version": 1,
        "status": "v17_owner_review_diagnosed_without_new_inference",
        "experiment_id": "supervised_labeler_v17",
        "source_owner_review_sha256": str(review["review_sha256"]),
        "scope": "frozen_audit_evidence_only",
        "cell_31": {
            "image_id": int(cell["image_id"]),
            "image_size": [416, 416],
            "truth_box_count": len(cell["truth_boxes"]),
            "model_prediction_count": len(cell["model_predictions"]),
            "oversized_background_prediction_count": len(predictions),
            "oversized_background_predictions": predictions,
            "rendering_explanation": (
                "Coordinates outside the source image are clipped to the "
                "panel boundary, so complete predicted rectangles appear to "
                "grow from the bottom or side edges."
            ),
            "root_cause": (
                "The model emitted high-enough-confidence background boxes. "
                "Their raw relative area, relative height, and aspect ratio "
                "still pass the preregistered v17 geometry limits."
            ),
            "rendering_bug": False,
        },
        "future_intervention_constraint": (
            "Do not change v17 threshold or geometry after audit. Any tighter "
            "geometry or background suppression must be preregistered for a "
            "fresh model and independent audit."
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
        raise RuntimeError("v17 owner review evidence already exists")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registration = config["model_review_registration"]
    manifest = _verified_manifest()
    report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(MODEL_EVIDENCE_PATH.read_text(encoding="utf-8"))
    if (
        config["status"] != "numeric_audit_passed_owner_model_review_pending"
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
        raise RuntimeError("Configured v17 model outcome changed")
    review = build_review(
        manifest=manifest,
        evidence=evidence,
        reviewed_on="2026-07-29",
    )
    diagnosis = build_diagnosis(evidence=evidence, review=review)
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
                "review_sha256": review["review_sha256"],
                "diagnosis_sha256": diagnosis["report_sha256"],
                "cell_31_oversized_predictions": diagnosis["cell_31"][
                    "oversized_background_prediction_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
