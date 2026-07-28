"""Freeze exact v10 owner-review files before presenting them."""

from __future__ import annotations

import json

from scripts.record_supervised_labeler_v10_review import (
    REVIEW_MANIFEST_PATH,
    build_v10_registration,
)
from scripts.train_supervised_labeler import REPORT_PATH
from src.synthetic.supervised_labeler import (
    CONFIG_PATH,
    SPLIT_PATH,
    load_supervised_labeler_config,
)
from src.synthetic.whole_image import canonical_mapping_sha256


def main() -> None:
    if REVIEW_MANIFEST_PATH.exists():
        raise RuntimeError("v10 owner-review manifest is already frozen")
    config = load_supervised_labeler_config(CONFIG_PATH)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    registration = build_v10_registration(
        config=config,
        split=split,
        report=report,
    )
    payload = {
        "schema_version": 1,
        "status": "v10_owner_review_files_frozen",
        "registration": registration,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    payload["manifest_sha256"] = canonical_mapping_sha256(payload)
    REVIEW_MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
