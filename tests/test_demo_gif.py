"""DEMO-04 frame selection: the part that picked the wrong pictures twice.

Neither mistake was a crash. The first ranking chose a crowd with no helmets
because it optimised for bare-head count; the second let a frame through on 8
annotations that then drew 22 boxes. Both were only visible by opening the GIF,
so the rules that replaced them are pinned here.
"""

from __future__ import annotations

import pytest

from scripts.make_demo_gif import (
    EXCLUDED_FRAMES,
    MAX_DRAWN_BOXES,
    MIN_DRAWN_PER_VERDICT,
    DemoGifError,
    FrameChoice,
    caption_for,
    choose_frames,
    publishable,
)
from src.inference.demo import FrameSummary


def _choice(name: str, *, helmets: int, heads: int, area: float = 500.0) -> FrameChoice:
    return FrameChoice(
        file_name=name, n_helmets=helmets, n_bare_heads=heads, smallest_box_area=area
    )


# --------------------------------------------------------------------------
# ranking: balance, not bare-head count
# --------------------------------------------------------------------------


def test_a_balanced_frame_outranks_one_with_more_bare_heads() -> None:
    """The first attempt got this backwards and produced an all-red montage."""

    balanced = _choice("balanced.png", helmets=4, heads=4)
    lopsided = _choice("lopsided.png", helmets=1, heads=9)

    assert balanced.sort_key() < lopsided.sort_key()
    assert balanced.balance == 4
    assert lopsided.balance == 1


def test_between_equally_balanced_frames_the_quieter_one_wins() -> None:
    quiet = _choice("quiet.png", helmets=3, heads=3)
    busy = _choice("busy.png", helmets=3, heads=8)

    assert quiet.sort_key() < busy.sort_key()


def test_ties_break_on_file_name_so_the_montage_is_reproducible() -> None:
    first = _choice("a.png", helmets=3, heads=3)
    second = _choice("b.png", helmets=3, heads=3)

    assert first.sort_key() < second.sort_key()


@pytest.mark.parametrize(
    ("helmets", "heads", "ok"),
    [
        (2, 2, True),      # exactly at both floors
        (1, 5, False),     # one class below MIN_PER_CLASS
        (5, 1, False),     # ...in either direction
        (1, 2, False),
        (6, 6, True),      # 12 instances: exactly at the ceiling
        (7, 6, False),     # 13: over it
        (1, 1, False),     # 2 instances: under MIN_INSTANCES as well
    ],
)
def test_the_readability_bounds_are_boundaries_not_suggestions(helmets, heads, ok) -> None:
    assert _choice("f.png", helmets=helmets, heads=heads).readable() is ok


# --------------------------------------------------------------------------
# the drawn-box cut: counts what a reader sees, not what the annotations say
# --------------------------------------------------------------------------


def test_a_frame_the_model_floods_with_boxes_is_rejected() -> None:
    """8 annotations, 22 drawn boxes. Ground-truth counts could not see this."""

    assert publishable(n_drawn=22, n_compliant=17, n_non_compliant=5) == "too busy"
    assert publishable(n_drawn=MAX_DRAWN_BOXES, n_compliant=6, n_non_compliant=6) is None
    assert publishable(n_drawn=MAX_DRAWN_BOXES + 1, n_compliant=6, n_non_compliant=6) is not None


def test_an_all_one_colour_frame_is_rejected_however_few_boxes_it_has() -> None:
    """A montage of single-colour frames shows nothing the project cares about.

    BOTH directions. Testing only the all-green case let a rule that looked at
    n_non_compliant alone survive mutation - it would have passed every all-red
    frame, which is exactly the failure the first attempt produced.
    """

    assert publishable(n_drawn=3, n_compliant=3, n_non_compliant=0) is not None
    assert publishable(n_drawn=10, n_compliant=9, n_non_compliant=1) is not None
    assert publishable(n_drawn=3, n_compliant=0, n_non_compliant=3) is not None
    assert publishable(n_drawn=10, n_compliant=1, n_non_compliant=9) is not None
    assert (
        publishable(
            n_drawn=4, n_compliant=MIN_DRAWN_PER_VERDICT, n_non_compliant=MIN_DRAWN_PER_VERDICT
        )
        is None
    )


def test_the_two_rejection_reasons_are_distinguishable() -> None:
    """A caller printing the reason must not print the same string for both."""

    busy = publishable(n_drawn=99, n_compliant=50, n_non_compliant=49)
    lopsided = publishable(n_drawn=4, n_compliant=4, n_non_compliant=0)

    assert busy != lopsided
    assert busy is not None and lopsided is not None


# --------------------------------------------------------------------------
# selection over a manifest
# --------------------------------------------------------------------------


def _manifest_and_annotations(names, split="val"):
    manifest = {"images": [{"file_name": n, "split": split} for n in names]}
    images, annotations = [], []
    for index, name in enumerate(names, start=1):
        images.append({"id": index, "file_name": name})
        for _ in range(3):
            annotations.append(
                {"id": len(annotations) + 1, "image_id": index, "category_id": 1,
                 "bbox": [0, 0, 20, 20]}
            )
            annotations.append(
                {"id": len(annotations) + 1, "image_id": index, "category_id": 2,
                 "bbox": [0, 0, 20, 20]}
            )
    return manifest, {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "helmet"}, {"id": 2, "name": "head"}],
    }


def test_only_the_requested_split_is_considered() -> None:
    """A README figure drawn on Test invites a question worth not inviting."""

    manifest, annotations = _manifest_and_annotations(["a.png", "b.png"], split="test")

    with pytest.raises(DemoGifError):
        choose_frames(manifest, annotations, split="val", n_frames=1)


def test_an_excluded_frame_never_reaches_the_montage() -> None:
    excluded = next(iter(EXCLUDED_FRAMES))
    manifest, annotations = _manifest_and_annotations([excluded, "fine.png"])

    chosen = choose_frames(manifest, annotations, split="val", n_frames=1)

    assert [c.file_name for c in chosen] == ["fine.png"]


def test_every_exclusion_carries_a_written_reason() -> None:
    """The list is editorial, so an unexplained entry is an unaudited decision."""

    for name, reason in EXCLUDED_FRAMES.items():
        assert name.endswith(".png"), name
        assert len(reason) > 40, f"{name} has no real reason: {reason!r}"


def test_too_few_candidates_raises_instead_of_returning_a_short_list() -> None:
    manifest, annotations = _manifest_and_annotations(["only.png"])

    with pytest.raises(DemoGifError, match="fewer than the 5"):
        choose_frames(manifest, annotations, split="val", n_frames=5)


def test_images_missing_either_class_are_not_candidates() -> None:
    """Enforced by `balance`, not by a separate membership test (see G13)."""

    manifest = {"images": [{"file_name": "helmets_only.png", "split": "val"}]}
    annotations = {
        "images": [{"id": 1, "file_name": "helmets_only.png"}],
        "annotations": [
            {"id": i, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20]}
            for i in range(1, 6)
        ],
        "categories": [{"id": 1, "name": "helmet"}, {"id": 2, "name": "head"}],
    }

    with pytest.raises(DemoGifError):
        choose_frames(manifest, annotations, split="val", n_frames=1)


# --------------------------------------------------------------------------
# caption
# --------------------------------------------------------------------------


def test_the_caption_reports_the_rate_the_demo_computed() -> None:
    text = caption_for(FrameSummary(n_compliant=3, n_non_compliant=1, n_neutral=2), "x.png")

    assert "x.png" in text
    assert "3 / 4 compliant" in text
    assert "0.75" in text


def test_a_frame_with_no_heads_says_so_rather_than_printing_a_rate_of_zero() -> None:
    text = caption_for(FrameSummary(n_compliant=0, n_non_compliant=0, n_neutral=1), "y.png")

    assert "n/a" in text
    assert "0.00" not in text
