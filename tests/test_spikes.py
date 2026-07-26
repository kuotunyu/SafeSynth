import numpy as np
import pytest

from src.data.spikes import (
    ClipGroupingResult,
    GroupingResult,
    box_iou_xywh,
    choose_clip_candidate,
    choose_grouping_threshold,
    group_from_distances,
    group_with_clip_guard,
    hamming_distance_matrix,
    split_counts,
    stable_group_split,
)
from src.data.voc_to_coco import DataInvariantError


def test_box_iou_xywh() -> None:
    assert box_iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert box_iou_xywh((0, 0, 10, 10), (10, 10, 5, 5)) == 0
    assert box_iou_xywh((0, 0, 10, 10), (5, 5, 10, 10)) == pytest.approx(25 / 175)


def test_hamming_distance_matrix_exact() -> None:
    hashes = ["0000000000000000", "0000000000000001", "ffffffffffffffff"]

    distances = hamming_distance_matrix(hashes)

    assert distances.tolist() == [[0, 1, 64], [1, 0, 63], [64, 63, 0]]


def test_grouping_uses_transitive_connected_components() -> None:
    distances = np.asarray(
        [
            [0, 1, 2, 8],
            [1, 0, 1, 8],
            [2, 1, 0, 8],
            [8, 8, 8, 0],
        ],
        dtype=np.uint8,
    )

    labels = group_from_distances(distances, threshold=1)

    assert labels.tolist() == [0, 0, 0, 1]


def test_stable_group_split_keeps_groups_whole_and_balanced() -> None:
    labels = np.repeat(np.arange(100), 10)

    first = stable_group_split(labels, seed=42)
    second = stable_group_split(labels, seed=42)
    counts = split_counts(labels, first)

    assert first == second
    assert counts == {"train": 700, "val": 150, "test": 150}


def test_choose_grouping_threshold_prefers_more_merging_valid_result() -> None:
    valid_low = GroupingResult(
        threshold=4,
        labels=np.arange(100),
        group_sizes=(1,) * 100,
        split_simulations={42: {"train": 70, "val": 15, "test": 15}},
    )
    valid_high = GroupingResult(
        threshold=6,
        labels=np.repeat(np.arange(50), 2),
        group_sizes=(2,) * 50,
        split_simulations={42: {"train": 70, "val": 14, "test": 16}},
    )

    assert choose_grouping_threshold([valid_low, valid_high]).threshold == 6


def test_choose_grouping_threshold_rejects_oversized_groups() -> None:
    invalid = GroupingResult(
        threshold=10,
        labels=np.zeros(300, dtype=np.int32),
        group_sizes=(300,),
        split_simulations={42: {"train": 300, "val": 0, "test": 0}},
    )

    with pytest.raises(DataInvariantError, match="No pHash threshold"):
        choose_grouping_threshold([invalid])


def test_clip_guard_never_connects_semantic_similarity_without_phash_guard() -> None:
    distances = np.asarray([[0, 12, 40], [12, 0, 40], [40, 40, 0]], dtype=np.uint8)
    similarities = np.asarray(
        [[1.0, 0.96, 0.99], [0.96, 1.0, 0.99], [0.99, 0.99, 1.0]],
        dtype=np.float32,
    )

    labels = group_with_clip_guard(
        distances,
        similarities,
        phash_threshold=10,
        cosine_threshold=0.95,
        phash_guard=16,
    )

    assert labels.tolist() == [0, 0, 1]


def test_choose_clip_candidate_prefers_fewer_valid_groups() -> None:
    more_groups = ClipGroupingResult(
        cosine_threshold=0.95,
        phash_guard=16,
        labels=np.repeat(np.arange(50), 2),
        group_sizes=(2,) * 50,
        split_simulations={42: {"train": 70, "val": 14, "test": 16}},
    )
    fewer_groups = ClipGroupingResult(
        cosine_threshold=0.90,
        phash_guard=20,
        labels=np.repeat(np.arange(25), 4),
        group_sizes=(4,) * 25,
        split_simulations={42: {"train": 68, "val": 16, "test": 16}},
    )

    selected = choose_clip_candidate(
        [more_groups, fewer_groups],
        max_group_size=250,
        split_tolerance=0.02,
    )

    assert selected is fewer_groups
