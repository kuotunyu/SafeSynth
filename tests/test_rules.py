from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.filtering.rules import filter_sample

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "filtering.yaml").read_text(encoding="utf-8"))


def instance(
    instance_id: str,
    class_name: str,
    bbox: list[float],
    *,
    kind: str = "pasted",
    z_index: int = 0,
    **extra,
) -> dict:
    return {
        "instance_id": instance_id,
        "class_name": class_name,
        "bbox_xywh": bbox,
        "kind": kind,
        "kept": True,
        "z_index": z_index,
        "visible_fraction": 1.0,
        "mask_to_box_coverage": 0.60 if class_name != "person" else 0.50,
        "sam2_qc_pass": True,
        **extra,
    }


def valid_sample(*instances: dict) -> dict:
    return {
        "width": 416,
        "height": 416,
        "instances": list(instances)
        or [instance("h", "helmet", [100, 100, 40, 30])],
        "pairs": [],
        "dedup": {
            "changed_pixel_ratio": 0.20,
            "min_hamming_to_accepted_synthetic": 20,
            "min_hamming_to_other_real_image": 20,
        },
        "invariants": {
            "n_real_ann_in": 0,
            "n_real_ann_out": 0,
            "intentional_removals": [],
            "test_blocklist_untouched": True,
        },
    }


def reasons(sample: dict, config: dict) -> set[str]:
    return set(filter_sample(sample, config).reject_reasons)


def test_valid_worn_helmet_passes(config: dict) -> None:
    helmet = instance("helmet", "helmet", [110, 87, 40, 25], z_index=0)
    head = instance("head", "head", [108, 100, 44, 52], z_index=1)
    sample = valid_sample(helmet, head)
    sample["pairs"] = [
        {"dx": 0, "dy": 0.25, "overlap_y": 0.23, "r_w": 0.91, "r_h": 0.48, "iou": 0.15}
    ]

    assert filter_sample(sample, config).passed


def test_floating_helmet(config: dict) -> None:
    sample = valid_sample()
    sample["pairs"] = [
        {"dx": 0, "dy": 0.7, "overlap_y": 0.01, "r_w": 1.0, "r_h": 0.5, "iou": 0.01}
    ]

    assert "FLOATING_HELMET" in reasons(sample, config)


def test_helmet_on_head_side(config: dict) -> None:
    sample = valid_sample()
    sample["pairs"] = [
        {"dx": 0.8, "dy": 0.2, "overlap_y": 0.3, "r_w": 1.0, "r_h": 0.5, "iou": 0.10}
    ]

    assert "HELMET_HEAD_MISALIGNED" in reasons(sample, config)


def test_helmet_swallows_face(config: dict) -> None:
    sample = valid_sample()
    sample["pairs"] = [
        {"dx": 0, "dy": 0, "overlap_y": 0.95, "r_w": 1.2, "r_h": 1.0, "iou": 0.50}
    ]

    assert "HELMET_SWALLOWS_HEAD" in reasons(sample, config)


def test_out_of_bounds_box(config: dict) -> None:
    item = instance(
        "h",
        "helmet",
        [0, 20, 20, 20],
        bbox_xywh_preclip=[-30, 20, 50, 20],
    )

    assert "OUT_OF_BOUNDS" in reasons(valid_sample(item), config)


def test_only_five_percent_visible(config: dict) -> None:
    item = instance("h", "helmet", [100, 100, 20, 20], visible_fraction=0.05)

    assert "LOW_VISIBLE_FRACTION" in reasons(valid_sample(item), config)


def test_duplicate_synthetic(config: dict) -> None:
    sample = valid_sample()
    sample["dedup"]["min_hamming_to_accepted_synthetic"] = 2

    assert "NEAR_DUPLICATE_SYNTHETIC" in reasons(sample, config)


def test_hard_negative_on_annotation(config: dict) -> None:
    negative = instance(
        "n",
        "hard_negative",
        [200, 200, 50, 50],
        kind="hard_negative",
        max_iou_with_annotation=0.20,
    )

    assert "HARD_NEGATIVE_OVERLAPS_ANNOTATION" in reasons(
        valid_sample(negative), config
    )


def test_no_change(config: dict) -> None:
    sample = valid_sample()
    sample["dedup"]["changed_pixel_ratio"] = 0

    assert "NO_CHANGE" in reasons(sample, config)


def test_seam_artifact(config: dict) -> None:
    item = instance("h", "helmet", [100, 100, 40, 30], seam_energy_ratio=100)

    assert "SEAM_ARTIFACT" in reasons(valid_sample(item), config)


def test_depth_inconsistent_poke_through(config: dict) -> None:
    item = instance(
        "h",
        "helmet",
        [100, 100, 40, 30],
        is_behind_other=True,
        poke_through_fraction=0.10,
    )

    assert "CLIPPING_ARTIFACT" in reasons(valid_sample(item), config)


def test_missing_real_annotation_crashes(config: dict) -> None:
    sample = valid_sample()
    sample["invariants"].update({"n_real_ann_in": 2, "n_real_ann_out": 1})

    with pytest.raises(AssertionError, match="real annotation count"):
        filter_sample(sample, config)


def test_bad_size_ratio(config: dict) -> None:
    person = instance("p", "person", [100, 100, 100, 200], z_index=1)
    head = instance("h", "head", [105, 105, 90, 100], z_index=0)

    assert "BAD_SIZE_RATIO" in reasons(valid_sample(head, person), config)


def test_excessive_same_class_overlap(config: dict) -> None:
    first = instance("a", "helmet", [100, 100, 50, 40], z_index=0)
    second = instance("b", "helmet", [102, 102, 50, 40], z_index=1)

    assert "EXCESSIVE_OVERLAP" in reasons(valid_sample(first, second), config)


def test_z_order_bug_crashes(config: dict) -> None:
    near = instance("near", "person", [100, 200, 60, 150], z_index=0)
    far = instance("far", "person", [10, 10, 40, 80], z_index=1)

    with pytest.raises(AssertionError, match="z-order mismatch"):
        filter_sample(valid_sample(near, far), config)


def test_every_reject_reason_is_declared(config: dict) -> None:
    sample = valid_sample()
    sample["dedup"]["min_hamming_to_other_real_image"] = 0
    result = filter_sample(sample, config)

    assert set(result.reject_reasons) <= set(config["reject_reasons"])


def test_config_fixture_is_not_mutated(config: dict) -> None:
    before = copy.deepcopy(config)
    filter_sample(valid_sample(), config)
    assert config == before
