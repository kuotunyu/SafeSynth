from __future__ import annotations

from scripts.audit_paired_person_units import paired_headlike_annotations


def test_paired_headlike_requires_upper_person_support() -> None:
    annotations = [
        {"id": 1, "category_id": 1, "bbox": [20, 12, 20, 18]},
        {"id": 2, "category_id": 2, "bbox": [21, 78, 18, 16]},
        {"id": 3, "category_id": 1, "bbox": [80, 10, 18, 18]},
        {"id": 4, "category_id": 3, "bbox": [10, 10, 40, 100]},
    ]

    paired = paired_headlike_annotations(
        [10, 10, 40, 100],
        annotations=annotations,
        categories={1: "helmet", 2: "head", 3: "person"},
    )

    assert [annotation["id"] for annotation in paired] == [1]
