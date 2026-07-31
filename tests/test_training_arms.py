"""Arm composition rules, all of which fail silently if unchecked (TRAIN-03..07)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.arms import (
    ARMS,
    ArmCompositionError,
    assert_arm_invariants,
    build_all_arms,
    build_arm,
    digest_names,
    equal_step_budget,
    split_real_images,
)


def write_manifest(tmp_path: Path, **counts: int) -> Path:
    images = []
    for split, count in counts.items():
        images.extend(
            {"file_name": f"images/{split}_{i:04d}.png", "split": split}
            for i in range(count)
        )
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps({"images": images}), encoding="utf-8")
    return path


def write_coco(tmp_path: Path, name: str, count: int) -> Path:
    payload = {
        "images": [
            {"id": i, "file_name": f"images/{name}_{i:04d}.png"} for i in range(count)
        ],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "helmet"},
            {"id": 1, "name": "head"},
            {"id": 2, "name": "person"},
        ],
    }
    path = tmp_path / f"annotations_{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def scenario(tmp_path: Path):
    manifest = write_manifest(tmp_path, train=40, val=8, test=8)
    return {
        "manifest_path": manifest,
        "synthetic_annotations": {
            "filtered": write_coco(tmp_path, "filtered", 40),
            "unfiltered": write_coco(tmp_path, "unfiltered", 40),
        },
    }


def test_four_arms_and_only_the_synthetic_ones_get_synthetic(scenario) -> None:
    arms = build_all_arms(**scenario)

    assert set(arms) == set(ARMS)
    assert arms["real_only"].synthetic == ()
    assert arms["standard_aug"].synthetic == ()
    assert len(arms["filtered_syn"].synthetic) == 40
    assert len(arms["unfiltered_syn"].synthetic) == 40


# spec: TRAIN-04
def test_every_arm_trains_on_identical_real_images(scenario) -> None:
    arms = build_all_arms(**scenario)

    digests = {comp.real_train_digest for comp in arms.values()}
    assert len(digests) == 1


# spec: TRAIN-05
def test_synthetic_arms_reuse_the_baseline_augmentation(scenario) -> None:
    """Otherwise 'synthetic helps' is confounded with 'different augmentation'."""

    arms = build_all_arms(**scenario)

    assert arms["real_only"].augmentation_profile == "real_only"
    for arm in ("standard_aug", "unfiltered_syn", "filtered_syn"):
        assert arms[arm].augmentation_profile == "standard_aug"


# spec: TRAIN-06
def test_mismatched_synthetic_sizes_are_rejected(tmp_path: Path) -> None:
    scenario = {
        "manifest_path": write_manifest(tmp_path, train=40, val=8, test=8),
        "synthetic_annotations": {
            "filtered": write_coco(tmp_path, "filtered", 40),
            "unfiltered": write_coco(tmp_path, "unfiltered", 31),
        },
    }

    with pytest.raises(ArmCompositionError, match="not size matched"):
        build_all_arms(**scenario)


# spec: TRAIN-23
def test_test_images_may_never_enter_training_or_validation(scenario) -> None:
    arms = build_all_arms(**scenario)
    real_splits = split_real_images(scenario["manifest_path"])
    poisoned = dict(arms)
    leaked = real_splits["test"][0]
    poisoned["real_only"] = arms["real_only"].__class__(
        arm="real_only",
        real_train=(*arms["real_only"].real_train, leaked),
        real_val=arms["real_only"].real_val,
        synthetic=(),
        augmentation_profile="real_only",
        real_train_digest=digest_names((*arms["real_only"].real_train, leaked)),
    )

    with pytest.raises(ArmCompositionError, match="Test images"):
        assert_arm_invariants(poisoned, real_splits=real_splits)


def test_unknown_arm_name_is_rejected(scenario) -> None:
    real_splits = split_real_images(scenario["manifest_path"])

    with pytest.raises(ArmCompositionError, match="Unknown arm"):
        build_arm(
            "real_plus_vibes",
            real_splits=real_splits,
            synthetic_annotations=scenario["synthetic_annotations"],
        )


# spec: TRAIN-07
def test_equal_steps_gives_every_arm_the_same_optimizer_budget(scenario) -> None:
    arms = build_all_arms(**scenario)

    plan = equal_step_budget(
        arms, reference_arm="real_only", reference_epochs=50, batch_size=8
    )

    assert len({row["total_steps"] for row in plan.values()} ) == 1
    # Synthetic arms hold twice the images, so they cover half the epochs.
    assert plan["real_only"]["epochs"] == pytest.approx(50.0)
    assert plan["filtered_syn"]["epochs"] == pytest.approx(25.0)


# spec: TRAIN-07
def test_equal_steps_reports_the_real_exposure_it_costs(scenario) -> None:
    """Fixing steps halves how often synthetic arms see each real image.

    That is the price of controlling compute, and it has to be a reported number
    rather than a surprise a reviewer finds later.
    """

    arms = build_all_arms(**scenario)

    plan = equal_step_budget(
        arms, reference_arm="real_only", reference_epochs=50, batch_size=8
    )

    assert plan["real_only"]["real_image_exposures"] == pytest.approx(50.0, rel=0.02)
    assert plan["filtered_syn"]["real_image_exposures"] == pytest.approx(25.0, rel=0.02)
