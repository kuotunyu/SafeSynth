"""Freeze the v11 Train-only audit with known GT defects quarantined."""

from __future__ import annotations

import hashlib
import json

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import (
    freeze_supervised_split,
    load_supervised_labeler_config,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "supervised_labeler_v11.yaml"
SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v11_split.json"
V10_SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v10_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v11_split.md"
QUARANTINED_GT_DEFECT_IMAGE_IDS = {3060, 4155, 4364}


def _canonical_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def main() -> None:
    if SPLIT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Supervised labeler v11 split is already frozen")
    config = load_supervised_labeler_config(CONFIG_PATH)
    if config["experiment_id"] != "supervised_labeler_v11":
        raise RuntimeError("Supervised v11 registration changed")
    if set(config["data"]["quarantined_gt_defect_image_ids"]) != (
        QUARANTINED_GT_DEFECT_IMAGE_IDS
    ):
        raise RuntimeError("Supervised v11 GT quarantine changed")

    v10_split = json.loads(V10_SPLIT_PATH.read_text(encoding="utf-8"))
    prior_calibration_ids = [
        int(value) for value in v10_split["calibration_image_ids"]
    ]
    prior_revealed_audit_ids = [
        int(value) for value in v10_split["untouched_audit_image_ids"]
    ]
    complete_prior_history = set(prior_calibration_ids) | set(
        prior_revealed_audit_ids
    )
    if not QUARANTINED_GT_DEFECT_IMAGE_IDS <= complete_prior_history:
        raise RuntimeError("A quarantined GT defect is not in revealed history")

    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    helmet_category_id = next(
        int(category["id"])
        for category in coco["categories"]
        if str(category["name"]) == "helmet"
    )
    payload = freeze_supervised_split(
        config=config,
        train_images=train_images,
        annotations=annotations,
        frozen=frozen,
        helmet_category_id=helmet_category_id,
        zero_shot_calibration_ids=prior_calibration_ids,
        zero_shot_audit_ids=prior_revealed_audit_ids,
    )

    payload["calibration_image_ids"] = sorted(
        complete_prior_history - QUARANTINED_GT_DEFECT_IMAGE_IDS
    )
    payload["calibration_images"] = len(payload["calibration_image_ids"])
    payload["quarantined_gt_defect_image_ids"] = sorted(
        QUARANTINED_GT_DEFECT_IMAGE_IDS
    )
    payload["quarantined_gt_defect_images"] = len(
        QUARANTINED_GT_DEFECT_IMAGE_IDS
    )
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = _canonical_sha256(payload)

    if (
        set(payload["calibration_image_ids"])
        | set(payload["quarantined_gt_defect_image_ids"])
    ) != complete_prior_history:
        raise AssertionError("v11 did not account for all revealed history")
    if set(payload["untouched_audit_image_ids"]) & complete_prior_history:
        raise AssertionError("A revealed image entered the v11 audit")
    selected = (
        set(payload["training_image_ids"])
        | set(payload["calibration_image_ids"])
        | set(payload["quarantined_gt_defect_image_ids"])
        | set(payload["untouched_audit_image_ids"])
    )
    if test_ids & selected:
        raise AssertionError("Validation/Test leakage entered v11 split")

    SPLIT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler v11 Train-only split",
        "",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        f"- Split seed: **{payload['split_seed']}**",
        f"- Training seed: **{config['training_seed']}**",
        (
            f"- Training: **{payload['training_images']} images**, "
            f"**{payload['training_groups']} groups**, "
            f"**{payload['training_helmet_annotations']} helmet boxes**"
        ),
        (
            f"- Calibration: **{payload['calibration_images']} images** "
            "(revealed Train-only history after quarantine)"
        ),
        (
            "- Quarantined owner-confirmed GT defects: "
            f"**{payload['quarantined_gt_defect_images']} images**"
        ),
        f"- New sealed audit: **{payload['untouched_audit_images']} images**",
        "- Group overlap: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- New audit pixels/metrics inspected before v11 training: **no**",
        "",
    ]
    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "manifest_sha256": payload["manifest_sha256"],
                "training_images": payload["training_images"],
                "calibration_images": payload["calibration_images"],
                "quarantined_gt_defect_images": payload[
                    "quarantined_gt_defect_images"
                ],
                "sealed_audit_images": payload["untouched_audit_images"],
                "validation_images_read": 0,
                "test_images_read": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
