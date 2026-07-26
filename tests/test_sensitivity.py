from __future__ import annotations

from src.filtering.sensitivity import (
    acceptance_rate,
    analyze_threshold_sensitivity,
    numeric_rule_leaves,
)


def config() -> dict:
    return {
        "sensitivity_alarm_points": 15,
        "rules": {
            "bbox_in_bounds": {"min_box_side_px": 4, "min_inside_ratio": 0.6},
            "min_visible_area": {"min_area_px": 10},
            "visible_fraction": {"helmet": 0.3},
            "overlap": {
                "max_overlap_score_same_class": 0.8,
                "max_overlap_iou_same_class": 0.8,
            },
            "mask_to_box_coverage": {"helmet": [0.2, 0.9]},
            "helmet_above_head": {
                "k_x": 1,
                "dy_min": -1,
                "dy_max": 1,
                "overlap_y_min": 0,
                "overlap_y_max": 1,
                "r_w": [0, 2],
                "r_h": [0, 2],
                "min_iou_helmet_head": 0,
            },
            "size_ratio": {
                "containment_threshold": 1,
                "head_over_person_area": [0, 1],
                "helmet_over_person_area": [0, 1],
                "head_over_person_width": [0, 1],
                "head_top_within_person": 1,
            },
            "clipping_artifact": {
                "poke_through_range": [0.02, 0.2],
                "max_seam_energy_ratio": 10,
                "seam_band_px": 2,
            },
            "hard_negative_no_overlap": {"max_iou_with_annotation": 0.02},
            "phash_dedup": {
                "max_hamming_to_accepted_synthetic": 6,
                "max_hamming_to_any_real_image": 6,
                "min_changed_pixel_ratio": 0.005,
            },
        },
        "assertions": {
            "real_annotation_count_invariant": True,
            "z_order_matches_y_bottom": True,
            "no_zero_area_annotation": True,
            "test_blocklist_untouched": True,
        },
        "reject_reasons": [
            "OUT_OF_BOUNDS",
            "BOX_TOO_SMALL",
            "LOW_VISIBLE_FRACTION",
            "EXCESSIVE_OVERLAP",
            "BAD_MASK_COVERAGE",
            "FLOATING_HELMET",
            "HELMET_HEAD_MISALIGNED",
            "HELMET_SWALLOWS_HEAD",
            "BAD_SIZE_RATIO",
            "CLIPPING_ARTIFACT",
            "SEAM_ARTIFACT",
            "HARD_NEGATIVE_OVERLAPS_ANNOTATION",
            "NEAR_DUPLICATE_SYNTHETIC",
            "NEAR_DUPLICATE_REAL",
            "NO_CHANGE",
            "SAM2_MASK_REJECTED",
            "PLACEMENT_RETRIES_EXHAUSTED",
        ],
    }


def sample(visible_fraction: float) -> dict:
    return {
        "width": 100,
        "height": 100,
        "instances": [
            {
                "instance_id": "h",
                "class_name": "helmet",
                "kind": "pasted",
                "bbox_xywh": [10, 10, 20, 20],
                "visible_fraction": visible_fraction,
                "mask_to_box_coverage": 0.6,
                "sam2_qc_pass": True,
                "z_index": 0,
            }
        ],
        "pairs": [],
        "dedup": {
            "changed_pixel_ratio": 0.2,
            "min_hamming_to_accepted_synthetic": 20,
            "min_hamming_to_any_real_image": 20,
        },
        "invariants": {
            "n_real_ann_in": 0,
            "n_real_ann_out": 0,
            "intentional_removals": [],
            "test_blocklist_untouched": True,
        },
    }


def test_numeric_leaves_exclude_booleans() -> None:
    leaves = numeric_rule_leaves({"enabled": True, "threshold": 0.5})

    assert [(leaf.path, leaf.value) for leaf in leaves] == [
        (("threshold",), 0.5)
    ]


def test_acceptance_rate_and_sensitivity() -> None:
    settings = config()
    samples = [sample(0.4), sample(0.25)]

    assert acceptance_rate(samples, settings) == 0.5
    result = analyze_threshold_sensitivity(samples, settings)

    assert result["n_samples"] == 2
    assert result["rows"]
    assert any(
        row["path"] == "rules.visible_fraction.helmet" for row in result["rows"]
    )
