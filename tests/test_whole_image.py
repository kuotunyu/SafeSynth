from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.record_whole_image_v10_review import (
    build_output_review_evidence,
    parse_problem_cases,
)
from src.synthetic.grounded_labeler import load_whole_image_config
from src.synthetic.whole_image import (
    diagnostic_manifest,
    human_review_evidence_sha256,
    require_generation_approval,
    require_scaleup_approval,
)


def _passed_v6_report(config: dict) -> dict:
    registered = config["supervised_labeler"]
    return {
        "status": "supervised_labeler_audit_passed",
        "checks": {
            "audit_precision": True,
            "audit_recall": True,
            "audit_median_matched_iou": True,
        },
        "audit_metrics": {
            "precision": registered["audit_precision"],
            "recall": registered["audit_recall"],
            "median_matched_iou": registered[
                "audit_median_matched_iou"
            ],
        },
        "best_calibration": {
            "threshold": registered["score_threshold"],
        },
        "postprocessing": {
            "max_relative_area": registered["max_relative_area"],
            "max_relative_height": registered["max_relative_height"],
        },
        "checkpoint_sha256": registered["checkpoint_sha256"],
        "split_manifest_sha256": registered["split_manifest_sha256"],
        "untouched_audit_images_read": registered["audit_images"],
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }


def _approved_human_review(config: dict) -> dict:
    registered = config["supervised_labeler"]
    review = registered["human_review"]
    evidence = {
        "schema_version": 1,
        "status": "approved_by_kuotunyu",
        "reviewed_by": "kuotunyu",
        "reviewed_on": "2026-07-28",
        "review_note": "",
        "experiment_id": registered["experiment_id"],
        "checkpoint_sha256": registered["checkpoint_sha256"],
        "split_manifest_sha256": registered["split_manifest_sha256"],
        "score_threshold": registered["score_threshold"],
        "audit_images": registered["audit_images"],
        "figure": review["figure"],
        "figure_sha256": review["figure_sha256"],
        "pages": review["pages"],
        "separated_pages": review["separated_pages"],
        "problem_count": 0,
        "problem_cells": [],
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    evidence["evidence_sha256"] = human_review_evidence_sha256(evidence)
    return evidence


def _pending_v10_report(config: dict) -> dict:
    manifest = diagnostic_manifest(config)
    registered = config["supervised_labeler"]
    return {
        "status": "pending_kuotunyu_visual_review",
        "manifest_sha256": manifest["manifest_sha256"],
        "labeler_checkpoint_sha256": registered["checkpoint_sha256"],
        "labeler_split_manifest_sha256": registered[
            "split_manifest_sha256"
        ],
        "figure": "reports/figures/whole_image_v10_diagnostic.png",
        "figure_sha256": "f" * 64,
        "cases": [
            {"case_index": case_index, "image_sha256": str(case_index) * 64}
            for case_index in range(1, 5)
        ],
        "validation_images_read": 0,
        "test_images_read": 0,
        "expanded_to_64": False,
    }


def test_diagnostic_manifest_is_frozen_and_train_independent() -> None:
    config = load_whole_image_config()

    first = diagnostic_manifest(config)
    second = diagnostic_manifest(config)

    assert first == second
    assert first["manifest_sha256"] == (
        "843b3cfcd333e2f965389a5dac76d72f5811af3a900d6cf6ef7f3ec6f404bc51"
    )
    assert [case["case_index"] for case in first["cases"]] == [1, 2, 3, 4]
    assert len({case["seed"] for case in first["cases"]}) == 4
    assert first["validation_images_read"] == 0
    assert first["test_images_read"] == 0


def test_generation_gate_requires_labeler_and_exact_owner_approval() -> None:
    config = load_whole_image_config()
    manifest = diagnostic_manifest(config)
    labeler_report = _passed_v6_report(config)

    with pytest.raises(RuntimeError, match="GPU gate locked"):
        require_generation_approval(
            config=config,
            labeler_report=labeler_report,
            human_review_report=_approved_human_review(config),
            manifest=manifest,
        )

    approved = deepcopy(config)
    approved["generation_gate"]["allowed"] = True
    approved["supervised_labeler"]["human_review"][
        "status"
    ] = "approved_by_kuotunyu"
    require_generation_approval(
        config=approved,
        labeler_report=labeler_report,
        human_review_report=_approved_human_review(config),
        manifest=manifest,
    )


def test_changed_prompt_invalidates_owner_approval() -> None:
    config = load_whole_image_config()
    approved = deepcopy(config)
    approved["generation_gate"]["allowed"] = True
    approved["supervised_labeler"]["human_review"][
        "status"
    ] = "approved_by_kuotunyu"
    approved["diagnostic"]["cases"][0]["prompt"] += " Changed after review."

    with pytest.raises(RuntimeError, match="GPU gate locked"):
        require_generation_approval(
            config=approved,
            labeler_report=_passed_v6_report(config),
            human_review_report=_approved_human_review(config),
            manifest=diagnostic_manifest(approved),
        )


def test_old_zero_shot_labeler_report_cannot_open_v10_gate() -> None:
    config = load_whole_image_config()
    approved = deepcopy(config)
    approved["generation_gate"]["allowed"] = True
    approved["supervised_labeler"]["human_review"][
        "status"
    ] = "approved_by_kuotunyu"

    with pytest.raises(RuntimeError, match="GPU gate locked"):
        require_generation_approval(
            config=approved,
            labeler_report={
                "status": "labeler_audit_passed",
                "validation_images_read": 0,
                "test_images_read": 0,
            },
            human_review_report=_approved_human_review(config),
            manifest=diagnostic_manifest(config),
        )


def test_changed_v6_checkpoint_evidence_cannot_open_gate() -> None:
    config = load_whole_image_config()
    approved = deepcopy(config)
    approved["generation_gate"]["allowed"] = True
    approved["supervised_labeler"]["human_review"][
        "status"
    ] = "approved_by_kuotunyu"
    changed_report = _passed_v6_report(config)
    changed_report["checkpoint_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="GPU gate locked"):
        require_generation_approval(
            config=approved,
            labeler_report=changed_report,
            human_review_report=_approved_human_review(config),
            manifest=diagnostic_manifest(config),
        )


def test_tampered_human_review_evidence_cannot_open_gate() -> None:
    config = load_whole_image_config()
    approved = deepcopy(config)
    approved["generation_gate"]["allowed"] = True
    approved["supervised_labeler"]["human_review"][
        "status"
    ] = "approved_by_kuotunyu"
    evidence = _approved_human_review(config)
    evidence["problem_cells"] = [7]

    with pytest.raises(RuntimeError, match="GPU gate locked"):
        require_generation_approval(
            config=approved,
            labeler_report=_passed_v6_report(config),
            human_review_report=evidence,
            manifest=diagnostic_manifest(config),
        )


def test_v10_scaleup_requires_exact_owner_output_review() -> None:
    config = load_whole_image_config()
    manifest = diagnostic_manifest(config)
    diagnostic_report = _pending_v10_report(config)
    output_review = build_output_review_evidence(
        diagnostic_report=diagnostic_report,
        decision="approve",
        reviewed_on="2026-07-28",
        problem_cases=parse_problem_cases(""),
        note="All four outputs reviewed.",
    )

    with pytest.raises(RuntimeError, match="scale-up gate locked"):
        require_scaleup_approval(
            config=config,
            diagnostic_report=diagnostic_report,
            output_review_report=output_review,
            manifest=manifest,
        )

    approved = deepcopy(config)
    approved["scaleup_gate"]["allowed"] = True
    approved["diagnostic"]["output_review"][
        "status"
    ] = "approved_by_kuotunyu"
    require_scaleup_approval(
        config=approved,
        diagnostic_report=diagnostic_report,
        output_review_report=output_review,
        manifest=manifest,
    )


def test_v10_output_review_rejects_inconsistent_or_tampered_decisions() -> None:
    config = load_whole_image_config()
    diagnostic_report = _pending_v10_report(config)

    with pytest.raises(ValueError, match="approved review"):
        build_output_review_evidence(
            diagnostic_report=diagnostic_report,
            decision="approve",
            reviewed_on="2026-07-28",
            problem_cases=[3],
            note="",
        )
    with pytest.raises(ValueError, match="rejected review"):
        build_output_review_evidence(
            diagnostic_report=diagnostic_report,
            decision="reject",
            reviewed_on="2026-07-28",
            problem_cases=[],
            note="",
        )

    approved = deepcopy(config)
    approved["scaleup_gate"]["allowed"] = True
    approved["diagnostic"]["output_review"][
        "status"
    ] = "approved_by_kuotunyu"
    output_review = build_output_review_evidence(
        diagnostic_report=diagnostic_report,
        decision="approve",
        reviewed_on="2026-07-28",
        problem_cases=[],
        note="",
    )
    output_review["figure_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="scale-up gate locked"):
        require_scaleup_approval(
            config=approved,
            diagnostic_report=diagnostic_report,
            output_review_report=output_review,
            manifest=diagnostic_manifest(config),
        )
