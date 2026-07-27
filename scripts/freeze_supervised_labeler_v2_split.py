"""Freeze the v2 Train-only audit after the v1 audit was consumed."""

from __future__ import annotations

import json

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import (
    SPLIT_PATH,
    freeze_supervised_split,
    load_supervised_labeler_config,
)

V1_SPLIT_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_split.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v2_split.md"


def main() -> None:
    if SPLIT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Supervised labeler v2 split is already frozen")
    config = load_supervised_labeler_config()
    v1_split = json.loads(V1_SPLIT_PATH.read_text(encoding="utf-8"))
    prior_calibration_ids = [
        int(value) for value in v1_split["calibration_image_ids"]
    ]
    prior_failed_audit_ids = [
        int(value) for value in v1_split["untouched_audit_image_ids"]
    ]
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
        zero_shot_audit_ids=prior_failed_audit_ids,
    )
    expected_calibration = set(prior_calibration_ids) | set(
        prior_failed_audit_ids
    )
    if set(payload["calibration_image_ids"]) != expected_calibration:
        raise AssertionError("v2 calibration did not consume the full v1 history")
    selected = (
        set(payload["training_image_ids"])
        | set(payload["calibration_image_ids"])
        | set(payload["untouched_audit_image_ids"])
    )
    if test_ids & selected:
        raise AssertionError("Validation/Test leakage entered v2 split")
    SPLIT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler v2 Train-only split",
        "",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        (
            f"- Training: **{payload['training_images']} images**, "
            f"**{payload['training_groups']} groups**, "
            f"**{payload['training_helmet_annotations']} helmet boxes**"
        ),
        (
            f"- Calibration: **{payload['calibration_images']} images** "
            "(all previously consumed v1 calibration/audit images)"
        ),
        f"- New untouched audit: **{payload['untouched_audit_images']} images**",
        "- Group overlap: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- New audit pixels/metrics inspected before v2 training: **no**",
        "",
    ]
    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
