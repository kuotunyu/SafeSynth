"""Audit whether Train person cutouts carry a paired helmet/head annotation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.paths import PROJECT_ROOT, load_project_paths


def paired_headlike_annotations(
    person_bbox_xywh: Sequence[float],
    *,
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    upper_fraction: float = 0.55,
) -> list[dict[str, Any]]:
    """Return headlike boxes centred inside the upper portion of a person box."""

    x, y, width, height = (float(value) for value in person_bbox_xywh)
    paired: list[dict[str, Any]] = []
    for source in annotations:
        if categories[int(source["category_id"])] not in {"helmet", "head"}:
            continue
        box_x, box_y, box_width, box_height = (
            float(value) for value in source["bbox"]
        )
        center_x = box_x + box_width / 2
        center_y = box_y + box_height / 2
        if (
            x <= center_x <= x + width
            and y <= center_y <= y + upper_fraction * height
        ):
            paired.append(dict(source))
    paired.sort(key=lambda item: int(item["id"]))
    return paired


def _write_reports(payload: dict[str, Any]) -> None:
    reports = PROJECT_ROOT / "reports"
    json_path = reports / "h4_paired_person_unit_feasibility.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = [
        "# H4 paired-person unit CPU feasibility audit",
        "",
        "- Status: **candidate successor feasible; not preregistered**",
        "- Data scope: **Train cutout bank and Train annotations only**",
        "- Validation/Test images read: **0 / 0**",
        "- Model inference run: **no**",
        f"- Person cutouts: **{payload['person_cutouts']}**",
        (
            "- Person cutouts with an upper-body helmet/head pair: "
            f"**{payload['paired_person_cutouts']}**"
        ),
        f"- Paired source groups: **{payload['paired_source_groups']}**",
        (
            "- Paired person cutouts at least 80 px high: "
            f"**{payload['paired_person_cutouts_height_ge_80']}**"
        ),
        (
            "- Paired headlike annotations: "
            f"**{payload['paired_headlike_annotations']}** "
            f"({payload['paired_class_counts']})"
        ),
        "",
        "## Interpretation",
        "",
        (
            "The existing Train bank can support a successor that moves one "
            "anatomically coupled person + helmet/head unit instead of an "
            "isolated helmet. This removes the structural cause of the v5 "
            "floating-hat failures without downloading data or using a GPU."
        ),
        "",
        (
            "This is only a feasibility result. Person masks and paired labels "
            "still require stricter truncation, pose, and visual gates followed "
            "by a new zero-issue CPU draft sheet before any FLUX call."
        ),
        "",
    ]
    (reports / "h4_paired_person_unit_feasibility.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    paths = load_project_paths()
    coco = json.loads((paths.interim / "coco_all.json").read_text(encoding="utf-8"))
    bank = [
        json.loads(line)
        for line in (paths.cutouts / "bank_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco["categories"]
    }
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    person_cutouts = [
        item for item in bank if str(item["class_name"]) == "person"
    ]
    if any(str(item["src_split"]) != "train" for item in person_cutouts):
        raise AssertionError("Paired-person candidate contains a non-Train source")

    paired_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for item in person_cutouts:
        pairs = paired_headlike_annotations(
            item["src_bbox_xywh"],
            annotations=annotations_by_image[int(item["src_image_id"])],
            categories=categories,
        )
        if pairs:
            paired_rows.append((item, pairs))

    pair_classes = Counter(
        categories[int(annotation["category_id"])]
        for _, pairs in paired_rows
        for annotation in pairs
    )
    payload = {
        "schema_version": 1,
        "status": "candidate_successor_feasible_not_preregistered",
        "scope": "Train cutout bank and Train annotations only",
        "validation_images_read": 0,
        "test_images_read": 0,
        "model_inference_run": False,
        "h4_auc_computed": False,
        "person_cutouts": len(person_cutouts),
        "person_source_images": len(
            {int(item["src_image_id"]) for item in person_cutouts}
        ),
        "person_source_groups": len(
            {int(item["src_group_id"]) for item in person_cutouts}
        ),
        "paired_person_cutouts": len(paired_rows),
        "paired_person_cutouts_unique_pair": sum(
            len(pairs) == 1 for _, pairs in paired_rows
        ),
        "paired_person_cutouts_preferred_tier": sum(
            bool(item["preferred_tier"]) for item, _ in paired_rows
        ),
        "paired_person_cutouts_height_ge_80": sum(
            float(item["src_bbox_xywh"][3]) >= 80 for item, _ in paired_rows
        ),
        "paired_source_groups": len(
            {int(item["src_group_id"]) for item, _ in paired_rows}
        ),
        "paired_headlike_annotations": sum(
            len(pairs) for _, pairs in paired_rows
        ),
        "paired_class_counts": dict(sorted(pair_classes.items())),
    }
    _write_reports(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
