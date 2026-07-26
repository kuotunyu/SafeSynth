import json
from pathlib import Path

import pytest

from src.data.integrity import assert_test_untouched
from src.data.paths import ProjectPaths
from src.data.split import (
    aggregate_split_stats,
    build_group_stats,
    image_split_map,
    stratified_group_split,
    verify_split_invariants,
)
from src.data.voc_to_coco import DataInvariantError, sha256_file

CLASSES = ("helmet", "head", "person")


def synthetic_coco() -> dict:
    images = [{"id": image_id, "file_name": f"images/{image_id}.png"} for image_id in range(1, 31)]
    annotations = []
    annotation_id = 1
    for image_id in range(1, 31):
        for category_id, count in (
            (1, 2),
            (2, 1 if image_id % 2 == 0 else 0),
            (3, 1 if image_id % 3 == 0 else 0),
        ):
            for _ in range(count):
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                    }
                )
                annotation_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "helmet"},
            {"id": 2, "name": "head"},
            {"id": 3, "name": "person"},
        ],
    }


def test_stratified_split_keeps_groups_whole_and_person_in_every_split() -> None:
    coco = synthetic_coco()
    image_to_group = {image_id: (image_id - 1) // 2 for image_id in range(1, 31)}
    groups = build_group_stats(coco, image_to_group, CLASSES)
    fractions = {"train": 0.70, "val": 0.15, "test": 0.15}

    assignments = stratified_group_split(
        groups,
        classes=CLASSES,
        fractions=fractions,
        seed=42,
    )
    image_splits = image_split_map(groups, assignments)
    stats = aggregate_split_stats(coco, image_splits, CLASSES)

    verify_split_invariants(
        groups=groups,
        group_assignments=assignments,
        image_splits=image_splits,
        split_stats=stats,
        fractions=fractions,
        person_min_fraction=0.10,
        ratio_tolerance=0.10,
    )
    for group in groups:
        assert len({image_splits[image_id] for image_id in group.image_ids}) == 1
    assert all(stats[split]["class_instances"]["person"] > 0 for split in fractions)


def make_paths(tmp_path: Path) -> ProjectPaths:
    data_root = tmp_path / "data"
    return ProjectPaths(
        project_root=tmp_path,
        config_path=tmp_path / "configs" / "paths.yaml",
        grouping_config_path=tmp_path / "configs" / "grouping.yaml",
        data_root=data_root,
        raw=data_root / "raw",
        hardhat_raw=data_root / "raw" / "dataset",
        interim=data_root / "interim",
        splits=tmp_path / "splits",
        reports=tmp_path / "reports",
        figures=tmp_path / "reports" / "figures",
        dotenv=tmp_path / ".env",
        kaggle_handle="owner/data",
        pinned_version=1,
        classes=CLASSES,
        seed=42,
    )


def test_assert_test_untouched_detects_byte_change(tmp_path: Path) -> None:
    paths = make_paths(tmp_path)
    image_path = paths.hardhat_raw / "images" / "sample.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"original")
    paths.splits.mkdir(parents=True)
    blocklist = {
        "images": [
            {
                "image_id": 1,
                "file_name": "images/sample.png",
                "sha256": sha256_file(image_path),
            }
        ]
    }
    blocklist_path = paths.splits / "test_blocklist.json"
    blocklist_path.write_text(json.dumps(blocklist), encoding="utf-8")

    assert_test_untouched(paths=paths)
    image_path.write_bytes(b"changed")

    with pytest.raises(DataInvariantError, match="verification failed"):
        assert_test_untouched(paths=paths)
