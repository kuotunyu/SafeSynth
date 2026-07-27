from __future__ import annotations

from copy import deepcopy

import pytest

from src.synthetic.grounded_labeler import load_whole_image_config
from src.synthetic.whole_image import (
    diagnostic_manifest,
    require_generation_approval,
)


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
    labeler_report = {
        "status": "labeler_audit_passed",
        "validation_images_read": 0,
        "test_images_read": 0,
    }

    with pytest.raises(RuntimeError, match="GPU gate locked"):
        require_generation_approval(
            config=config,
            labeler_report=labeler_report,
            manifest=manifest,
        )

    approved = deepcopy(config)
    approved["generation_gate"]["allowed"] = True
    approved["diagnostic"]["input_review"] = {
        "required_reviewer": "kuotunyu",
        "status": "approved_by_kuotunyu",
        "approved_manifest_sha256": manifest["manifest_sha256"],
    }
    require_generation_approval(
        config=approved,
        labeler_report=labeler_report,
        manifest=manifest,
    )


def test_changed_prompt_invalidates_owner_approval() -> None:
    config = load_whole_image_config()
    original = diagnostic_manifest(config)
    approved = deepcopy(config)
    approved["generation_gate"]["allowed"] = True
    approved["diagnostic"]["input_review"] = {
        "required_reviewer": "kuotunyu",
        "status": "approved_by_kuotunyu",
        "approved_manifest_sha256": original["manifest_sha256"],
    }
    approved["diagnostic"]["cases"][0]["prompt"] += " Changed after review."

    with pytest.raises(RuntimeError, match="GPU gate locked"):
        require_generation_approval(
            config=approved,
            labeler_report={
                "status": "labeler_audit_passed",
                "validation_images_read": 0,
                "test_images_read": 0,
            },
            manifest=diagnostic_manifest(approved),
        )
