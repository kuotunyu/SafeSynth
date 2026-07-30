"""Register the preregistered v23 model intervention before split freeze."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT

V22_CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v22.yaml"
V23_GT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "supervised_labeler_v23_gt_review.yaml"
)
OUTPUT_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v23.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _union(values: list[int], added: list[int]) -> list[int]:
    return sorted({int(value) for value in [*values, *added]})


def build_config(
    *,
    v22: dict[str, Any],
    gt: dict[str, Any],
) -> dict[str, Any]:
    """Build v23 from frozen v22 settings plus the five exact replay cases."""

    intervention = gt["future_v23_intervention"]
    evidence = intervention["revealed_development_evidence"]
    changes = intervention["model_facing_changes"]
    adjudication = gt["owner_adjudication_outcome"]
    if (
        v22["status"] != "owner_model_review_rejected"
        or gt["status"]
        != "gt_only_primary_adjudicated_three_images_quarantined"
        or intervention["status"]
        != "preregistered_before_v23_pool_pixels_or_training"
        or intervention["initialization"] != "pinned_base_checkpoint_only"
        or not changes["inherit_v22_sampling_and_replay_configuration"]
        or _sha256(PROJECT_ROOT / evidence["owner_review"])
        != evidence["owner_review_file_sha256"]
        or _sha256(PROJECT_ROOT / evidence["diagnosis"])
        != evidence["diagnosis_file_sha256"]
        or _sha256(PROJECT_ROOT / adjudication["owner_review_path"])
        != adjudication["owner_review_file_sha256"]
        or _sha256(PROJECT_ROOT / adjudication["adjudicated_audit_path"])
        != adjudication["adjudicated_audit_file_sha256"]
    ):
        raise RuntimeError("Frozen v22/v23 preregistration changed")

    sampling = deepcopy(v22["sampling"])
    sampling["owner_miss_replay_image_ids"] = _union(
        sampling["owner_miss_replay_image_ids"],
        changes["add_owner_miss_replay_image_ids"],
    )
    sampling["hard_negative_error_replay_image_ids"] = _union(
        sampling["hard_negative_error_replay_image_ids"],
        changes["add_hard_negative_error_replay_image_ids"],
    )
    sampling["owner_miss_replay_weight"] = float(
        changes["owner_miss_replay_weight"]
    )
    sampling["hard_negative_error_replay_weight"] = float(
        changes["hard_negative_error_replay_weight"]
    )
    sampling["overlap_policy"] = str(changes["overlap_policy"])
    sampling["selection_basis"] = (
        "Preregistered in configs/supervised_labeler_v23_gt_review.yaml "
        "before any v23 pool pixels or training were opened. Inherit every "
        "v22 sampling setting; add exact owner-miss images 487 and 93 at "
        "weight 40 and exact semantic-false-positive images 2969, 972, and "
        "3405 at hard-negative weight 28. No global weight changes."
    )

    data = deepcopy(v22["data"])
    data.update(
        {
            "independent_audit_manifest": (
                "splits/supervised_labeler_v23_adjudicated_audit.json"
            ),
            "include_owner_approved_v22_audit_groups_in_training": True,
            "exclude_v23_primary_and_sealed_reserve_groups_from_model_data": (
                True
            ),
            "new_untouched_audit_images": 48,
        }
    )
    data.pop(
        "include_owner_approved_v21_audit_groups_in_training",
        None,
    )
    data["quarantined_gt_defect_or_ambiguous_image_ids"] = _union(
        data["quarantined_gt_defect_or_ambiguous_image_ids"],
        [4052, 233, 2302],
    )

    config: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "supervised_labeler_v23",
        "status": "model_intervention_preregistered_split_pending",
        "architecture": str(v22["architecture"]),
        "split_seed": int(gt["split_seed"]),
        "training_seed": int(v22["training_seed"]),
        "split_manifest_sha256": "pending_split_freeze",
        "model": deepcopy(v22["model"]),
        "data": data,
        "input_normalization": deepcopy(v22["input_normalization"]),
        "sampling": sampling,
        "optimization": deepcopy(v22["optimization"]),
        "postprocessing": deepcopy(v22["postprocessing"]),
        "calibration": deepcopy(v22["calibration"]),
        "audit_gate": deepcopy(v22["audit_gate"]),
        "human_review_gate": deepcopy(v22["human_review_gate"]),
        "independence_registration": {
            "status": "frozen_before_v23_split_or_training",
            "gt_review_config": (
                "configs/supervised_labeler_v23_gt_review.yaml"
            ),
            "gt_owner_review": str(adjudication["owner_review_path"]),
            "gt_owner_review_file_sha256": str(
                adjudication["owner_review_file_sha256"]
            ),
            "gt_owner_review_sha256": str(
                adjudication["owner_review_sha256"]
            ),
            "audit_manifest": str(adjudication["adjudicated_audit_path"]),
            "audit_manifest_file_sha256": str(
                adjudication["adjudicated_audit_file_sha256"]
            ),
            "audit_manifest_sha256": str(
                adjudication["adjudicated_audit_sha256"]
            ),
            "v23_reserved_groups": 96,
            "v22_owner_approved_audit_groups_allowed_as_revealed_training_data": (
                True
            ),
            "prior_sealed_development_groups_excluded": True,
            "base_checkpoint_only": True,
            "v23_training_started": False,
            "audit_model_inference_run": False,
            "sealed_reserve_pixels_read": 0,
            "validation_images_read": 0,
            "test_images_read": 0,
        },
        "generation_gate": {
            "allowed": False,
            "reason": (
                "The v23 five-image replay intervention and independent "
                "audit are frozen. A leakage-free split, CPU preflight, GPU "
                "smoke, base-only formal training, numeric audit, and owner "
                "model review are still required."
            ),
        },
    }
    if (
        config["optimization"]["initialization"]
        != "pinned_base_checkpoint_only"
        or config["sampling"]["owner_miss_replay_weight"] != 40.0
        or config["sampling"]["hard_negative_error_replay_weight"] != 28.0
        or not {487, 93}
        <= set(config["sampling"]["owner_miss_replay_image_ids"])
        or not {2969, 972, 3405}
        <= set(config["sampling"]["hard_negative_error_replay_image_ids"])
    ):
        raise RuntimeError("v23 model intervention was not preserved")
    return config


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"v23 model config already exists: {OUTPUT_PATH}")
    config = build_config(
        v22=_load_yaml(V22_CONFIG_PATH),
        gt=_load_yaml(V23_GT_CONFIG_PATH),
    )
    OUTPUT_PATH.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
            allow_unicode=True,
            width=88,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        {
            "status": config["status"],
            "owner_miss_replay_images": len(
                config["sampling"]["owner_miss_replay_image_ids"]
            ),
            "hard_negative_replay_images": len(
                config["sampling"]["hard_negative_error_replay_image_ids"]
            ),
            "validation_images_read": 0,
            "test_images_read": 0,
        }
    )


if __name__ == "__main__":
    main()
