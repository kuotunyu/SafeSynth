from __future__ import annotations

from collections import Counter

from src.synthetic.survey import H2_BINS, select_h2_candidates, summarize_h2


def test_h2_selection_has_equal_bins_and_is_seeded() -> None:
    annotations = {
        image_id: [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": image_id % 3 + 1,
                "bbox": [0, 0, image_id + 1, image_id + 2],
            }
        ]
        for image_id in range(1, 91)
    }
    category_names = {1: "helmet", 2: "head", 3: "person"}

    first = select_h2_candidates(annotations, category_names, per_bin=10, seed=42)
    second = select_h2_candidates(annotations, category_names, per_bin=10, seed=42)

    assert first == second
    assert Counter(item["size_bin"] for item in first) == Counter(
        {bin_name: 10 for bin_name in H2_BINS}
    )


def test_h2_summary_aggregates_each_mode() -> None:
    records = []
    for bin_index, bin_name in enumerate(H2_BINS):
        for item_index in range(2):
            metric = float(bin_index + item_index + 1)
            records.append(
                {
                    "size_bin": bin_name,
                    "min_side_px": metric,
                    "class_name": "helmet",
                    "modes": {
                        mode: {
                            "metrics": {
                                "iou_score": metric,
                                "object_score_logit": metric,
                                "mask_to_box_coverage": metric,
                                "component_count": metric,
                                "solidity": metric,
                            }
                        }
                        for mode in ("full", "crop_1024", "crop_512")
                    },
                }
            )

    summary = summarize_h2(records)

    assert set(summary) == set(H2_BINS)
    assert summary["very_small"]["full"]["iou_score"]["p50"] == 1.5
