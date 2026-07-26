import numpy as np

from scripts.analyze_h4_feature_families import _slice_examples
from src.filtering.artifact_gate import PatchExample


def _example() -> PatchExample:
    return PatchExample(
        feature=np.arange(30, dtype=np.float32),
        label=1,
        group_key="g",
        class_name="helmet",
        example_id="e",
    )


def test_feature_family_slices_keep_hog_and_hsv_boundaries() -> None:
    hog = _slice_examples([_example()], feature_family="hog")[0].feature
    hsv = _slice_examples([_example()], feature_family="hsv")[0].feature

    assert hog.tolist() == list(range(6))
    assert hsv.tolist() == list(range(6, 30))


def test_combined_feature_slice_is_unchanged() -> None:
    combined = _slice_examples([_example()], feature_family="hog+hsv")[0]

    assert combined.feature.tolist() == list(range(30))
