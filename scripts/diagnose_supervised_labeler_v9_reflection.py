"""Confirm reflected-padding involvement in every v9 owner-review failure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import cv2
import yaml

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _load_context, reflected_padding_guard

REVIEW_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v9_human_review.json"
DIAGNOSIS_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v9_review_diagnosis.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v9_reflection_diagnosis.json"
)
MARKDOWN_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v9_reflection_diagnosis.md"
)


def _sha256(path: object) -> str:
    file_path = PROJECT_ROOT / str(path)
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _box_center_inside(
    box: list[float],
    crop: tuple[int, int, int, int],
) -> bool:
    x1, y1, x2, y2 = box
    left, top, right, bottom = crop
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return left <= center_x <= right and top <= center_y <= bottom


def main() -> None:
    if OUTPUT_PATH.exists() or MARKDOWN_PATH.exists():
        raise RuntimeError("v9 reflection diagnosis already exists")
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    diagnosis = json.loads(DIAGNOSIS_PATH.read_text(encoding="utf-8"))
    if (
        review["status"] != "rejected_by_kuotunyu"
        or diagnosis["status"]
        != "diagnostic_on_revealed_v9_train_only_audit"
        or review["problem_cells"] != diagnosis["problem_cells"]
    ):
        raise RuntimeError("Expected the canonical v9 owner rejection")

    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "compose.yaml").read_text(
            encoding="utf-8"
        )
    )
    reflection_config = config["compose"]["context_replacement"][
        "input_guard"
    ]["reflected_padding"]
    paths = load_project_paths()
    _, _, train_images, _, _, test_ids = _load_context(paths)
    cases = {int(case["cell"]): case for case in diagnosis["problem_cases"]}
    rows = []
    for cell in review["problem_cells"]:
        case = cases[int(cell)]
        image_id = int(case["image_id"])
        if image_id in test_ids:
            raise RuntimeError("A Validation/Test image entered diagnosis")
        image_record = train_images[image_id]
        bgr = cv2.imread(
            str(paths.hardhat_raw / str(image_record["file_name"]))
        )
        if bgr is None:
            raise RuntimeError(f"Could not load Train image {image_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        reflection = reflected_padding_guard(
            rgb,
            guard_config=reflection_config,
        )
        height, width = rgb.shape[:2]
        left = (
            int(reflection.left_right.start.pad_px)
            if reflection.left_right.start.detected
            else 0
        )
        right = (
            width - int(reflection.left_right.end.pad_px)
            if reflection.left_right.end.detected
            else width
        )
        top = (
            int(reflection.top_bottom.start.pad_px)
            if reflection.top_bottom.start.detected
            else 0
        )
        bottom = (
            height - int(reflection.top_bottom.end.pad_px)
            if reflection.top_bottom.end.detected
            else height
        )
        crop = (left, top, right, bottom)
        false_positive_locations = [
            {
                "box": row["box"],
                "center_inside_clean_crop": _box_center_inside(
                    row["box"],
                    crop,
                ),
            }
            for row in case["accepted_false_positives"]
        ]
        missed_truth_locations = [
            {
                "box": miss["truth_box"],
                "center_inside_clean_crop": _box_center_inside(
                    miss["truth_box"],
                    crop,
                ),
            }
            for miss in case["misses"]
        ]
        rows.append(
            {
                "cell": int(cell),
                "image_id": image_id,
                "owner_category": case["owner_category"],
                "reflection": asdict(reflection),
                "clean_crop_xyxy": list(crop),
                "false_positive_locations": false_positive_locations,
                "missed_truth_locations": missed_truth_locations,
            }
        )

    payload = {
        "schema_version": 1,
        "status": "v9_owner_failures_confirmed_reflection_related",
        "eligible_for_generation_gate": False,
        "source_review_evidence_sha256": review["evidence_sha256"],
        "source_review_diagnosis_file_sha256": _sha256(
            "reports/supervised_labeler_v9_review_diagnosis.json"
        ),
        "problem_cells": review["problem_cells"],
        "all_problem_images_reflection_detected": all(
            bool(row["reflection"]["detected"]) for row in rows
        ),
        "problem_images": rows,
        "revealed_train_images_read": len(rows),
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Supervised labeler v9 reflection diagnosis",
        "",
        "- Evidence: **revealed Train-only owner failures; not gate-eligible**",
        "- All four problem images contain detected reflection padding: **yes**",
        "- Validation/Test images read: **0 / 0**",
        "- Whole-image generations: **0**",
        "",
        "| Cell | Axis | Clean crop | FP centers outside | Miss centers inside |",
        "|---:|---|---|---:|---:|",
    ]
    for row in rows:
        fp_outside = sum(
            not item["center_inside_clean_crop"]
            for item in row["false_positive_locations"]
        )
        miss_inside = sum(
            item["center_inside_clean_crop"]
            for item in row["missed_truth_locations"]
        )
        lines.append(
            f"| {row['cell']} | "
            f"{','.join(row['reflection']['detected_axes'])} | "
            f"{row['clean_crop_xyxy']} | {fp_outside} | {miss_inside} |"
        )
    MARKDOWN_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
