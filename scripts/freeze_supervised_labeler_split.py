"""Freeze a new Train-only supervised-labeler audit before model training."""

from __future__ import annotations

import json

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context
from src.synthetic.supervised_labeler import (
    SPLIT_PATH,
    freeze_supervised_split,
    load_supervised_labeler_config,
)

ZERO_SHOT_REPORT = PROJECT_ROOT / "reports" / "grounded_labeler_audit.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_split.md"


def main() -> None:
    if SPLIT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Supervised labeler split is already frozen")
    config = load_supervised_labeler_config()
    paths = load_project_paths()
    coco, _, train_images, annotations, frozen, test_ids = _load_context(paths)
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco["categories"]
    }
    helmet_category_id = next(
        category_id
        for category_id, name in categories.items()
        if name == "helmet"
    )
    zero_shot = json.loads(ZERO_SHOT_REPORT.read_text(encoding="utf-8"))
    payload = freeze_supervised_split(
        config=config,
        train_images=train_images,
        annotations=annotations,
        frozen=frozen,
        helmet_category_id=helmet_category_id,
        zero_shot_calibration_ids=zero_shot["calibration_image_ids"],
        zero_shot_audit_ids=zero_shot["untouched_audit_image_ids"],
    )
    selected = (
        set(payload["training_image_ids"])
        | set(payload["calibration_image_ids"])
        | set(payload["untouched_audit_image_ids"])
    )
    if test_ids & selected:
        raise AssertionError("Validation/Test leakage entered supervised split")
    SPLIT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler Train-only split",
        "",
        f"- Manifest SHA-256: `{payload['manifest_sha256']}`",
        (
            f"- Training: **{payload['training_images']} images**, "
            f"**{payload['training_groups']} groups**, "
            f"**{payload['training_helmet_annotations']} helmet boxes**"
        ),
        (
            f"- Calibration: **{payload['calibration_images']} images** "
            "(the already-inspected zero-shot subsets)"
        ),
        f"- New untouched audit: **{payload['untouched_audit_images']} images**",
        "- Group overlap: **0**",
        "- Validation/Test images read: **0 / 0**",
        "- Audit pixels/metrics inspected before training: **no**",
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
