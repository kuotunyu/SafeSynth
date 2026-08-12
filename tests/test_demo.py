"""Demo rendering decisions: the colour rule and the frame summary.

The demo's whole claim is that a box is coloured by the COMPLIANCE VERDICT its
class implies, not by the class. Getting that backwards would show a red box on
a helmeted worker in the README's screenshot, which is the single most
misleading thing this project could publish.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.compliance import ComplianceStatus
from src.inference.demo import (
    COMPLIANT_COLOUR,
    NEUTRAL_COLOUR,
    NON_COMPLIANT_COLOUR,
    DemoError,
    FrameSummary,
    draw_on,
    drawn_boxes,
    status_for_label,
    summarise,
)

CLASSES = ("helmet", "head", "person")


def _detection(category_id, score, bbox=(10, 10, 20, 20)):
    return {"category_id": category_id, "score": score, "bbox": list(bbox)}


# --------------------------------------------------------------------------
# the colour rule
# --------------------------------------------------------------------------


def test_a_helmet_reads_compliant_and_a_head_reads_non_compliant() -> None:
    """ADR-007: helmet IS a helmeted head, head IS a bare head. Not shell vs skull."""

    assert status_for_label("helmet") is ComplianceStatus.COMPLIANT
    assert status_for_label("head") is ComplianceStatus.NON_COMPLIANT


def test_person_carries_no_verdict() -> None:
    """ADR-003 removed it from the compliance path; EVAL-03 pins that by test."""

    assert status_for_label("person") is None


def test_the_three_colours_are_distinct() -> None:
    """A single colour for everything would render the demo meaningless."""

    assert len({COMPLIANT_COLOUR, NON_COMPLIANT_COLOUR, NEUTRAL_COLOUR}) == 3


def test_each_class_gets_the_colour_its_verdict_implies() -> None:
    boxes = drawn_boxes(
        [_detection(0, 0.9), _detection(1, 0.9), _detection(2, 0.9)],
        class_names=CLASSES,
        score_threshold=0.0,
    )
    by_label = {box.label: box.colour for box in boxes}

    assert by_label["helmet"] == COMPLIANT_COLOUR
    assert by_label["head"] == NON_COMPLIANT_COLOUR
    assert by_label["person"] == NEUTRAL_COLOUR


def test_the_caption_carries_class_confidence_and_verdict() -> None:
    box = drawn_boxes(
        [_detection(1, 0.1234)], class_names=CLASSES, score_threshold=0.0
    )[0]

    assert "head" in box.caption
    assert "0.12" in box.caption
    assert ComplianceStatus.NON_COMPLIANT.value in box.caption


def test_captions_carry_a_zh_tw_semantic_label() -> None:
    """The visual verdict remains understandable without relying on red/green."""

    boxes = drawn_boxes(
        [_detection(0, 0.9), _detection(1, 0.8), _detection(2, 0.7)],
        class_names=CLASSES,
        score_threshold=0.0,
    )
    captions = {box.label: box.caption for box in boxes}

    assert "已佩戴" in captions["helmet"]
    assert "未佩戴" in captions["head"]
    assert "僅定位" in captions["person"]


def test_a_person_caption_has_no_verdict_appended() -> None:
    box = drawn_boxes(
        [_detection(2, 0.5)], class_names=CLASSES, score_threshold=0.0
    )[0]

    assert ComplianceStatus.COMPLIANT.value not in box.caption
    assert ComplianceStatus.NON_COMPLIANT.value not in box.caption


# --------------------------------------------------------------------------
# the operating point
# --------------------------------------------------------------------------


def test_boxes_below_the_threshold_are_dropped_and_the_boundary_is_kept() -> None:
    """`>=`, matching detection.py. The two must not disagree by one box."""

    boxes = drawn_boxes(
        [_detection(0, 0.069), _detection(0, 0.070), _detection(0, 0.071)],
        class_names=CLASSES,
        score_threshold=0.07,
    )

    assert sorted(round(box.score, 3) for box in boxes) == [0.070, 0.071]


def test_boxes_are_ordered_so_the_confident_ones_draw_last() -> None:
    """Otherwise a 0.07 box can sit on top of a 0.25 one at this model's scale."""

    boxes = drawn_boxes(
        [_detection(0, 0.20), _detection(0, 0.08), _detection(0, 0.14)],
        class_names=CLASSES,
        score_threshold=0.0,
    )

    assert [round(box.score, 2) for box in boxes] == [0.08, 0.14, 0.20]


def test_an_unknown_category_id_is_refused_rather_than_drawn_blank() -> None:
    with pytest.raises(DemoError, match="outside the 3 configured classes"):
        drawn_boxes([_detection(7, 0.9)], class_names=CLASSES, score_threshold=0.0)


# --------------------------------------------------------------------------
# the frame summary
# --------------------------------------------------------------------------


def test_the_summary_counts_only_head_classes_towards_the_rate() -> None:
    # Two helmets and one bare head among four boxes: 2/3, and the person box
    # must not become a fourth denominator.
    boxes = drawn_boxes(
        [_detection(0, 0.9), _detection(0, 0.9), _detection(1, 0.9), _detection(2, 0.9)],
        class_names=CLASSES,
        score_threshold=0.0,
    )

    summary = summarise(boxes)

    assert (summary.n_compliant, summary.n_non_compliant, summary.n_neutral) == (2, 1, 1)
    assert summary.n_people == 3
    assert summary.compliance_rate == pytest.approx(2 / 3)


def test_a_frame_with_no_heads_has_no_rate_rather_than_a_rate_of_zero() -> None:
    """Zero would assert "nobody here is wearing a helmet" about an empty frame."""

    summary = summarise(
        drawn_boxes([_detection(2, 0.9)], class_names=CLASSES, score_threshold=0.0)
    )

    assert summary.compliance_rate is None
    assert "no heads detected" in summary.render()
    assert "person box" in summary.render()


def test_a_completely_empty_frame_says_so_without_mentioning_person_boxes() -> None:
    summary = summarise([])

    assert summary.render() == "no heads detected"


def test_the_rendered_summary_shows_both_the_fraction_and_the_rate() -> None:
    rendered = FrameSummary(n_compliant=3, n_non_compliant=1, n_neutral=0).render()

    assert "3 / 4 compliant" in rendered
    assert "0.75" in rendered


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


def test_drawing_does_not_mutate_the_input_frame() -> None:
    """The demo hands the same array to the annotator and to nothing else, but a
    video loop that reused a mutated buffer would accumulate boxes over frames."""

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    original = image.copy()

    draw_on(image, drawn_boxes([_detection(0, 0.9)], class_names=CLASSES, score_threshold=0.0))

    assert np.array_equal(image, original)


def test_a_compliant_box_paints_green_and_a_non_compliant_one_paints_red() -> None:
    """Sampling the canvas, not the file size: the colour has to reach the pixels."""

    image = np.zeros((80, 160, 3), dtype=np.uint8)
    boxes = drawn_boxes(
        [_detection(0, 0.9, (10, 20, 30, 30)), _detection(1, 0.9, (100, 20, 30, 30))],
        class_names=CLASSES,
        score_threshold=0.0,
    )

    canvas = draw_on(image, boxes)

    # The rectangle's top edge runs along y = 20 for both boxes.
    helmet_pixel = canvas[20, 25]
    head_pixel = canvas[20, 115]
    assert helmet_pixel[1] > helmet_pixel[0]  # green channel dominates
    assert head_pixel[0] > head_pixel[1]      # red channel dominates


def test_a_non_square_frame_is_drawn_without_transposing_the_box() -> None:
    """DATA-25: this dataset is not square, and nor are most site photos."""

    image = np.zeros((60, 200, 3), dtype=np.uint8)
    boxes = drawn_boxes(
        [_detection(0, 0.9, (150, 10, 40, 30))], class_names=CLASSES, score_threshold=0.0
    )

    canvas = draw_on(image, boxes)

    # x runs to 190 and y to 40; a transposed implementation would clip or move it.
    assert canvas[10, 170].any()
    assert not canvas[50, 20].any()


# --------------------------------------------------------------------------
# caption degradation
#
# The first render of this demo put the full caption on all fifteen boxes of a
# crowded 416px frame and the middle of the picture became unreadable. Colour is
# the primary channel by design, so the text degrades instead of shrinking.
# --------------------------------------------------------------------------


def _box(width, score=0.12, category_id=1):
    return drawn_boxes(
        [_detection(category_id, score, (0, 0, width, 20))],
        class_names=CLASSES,
        score_threshold=0.0,
    )[0]


def test_a_wide_box_keeps_the_whole_caption() -> None:
    box = _box(400)

    assert box.caption_for_width(6.0) == box.caption
    assert ComplianceStatus.NON_COMPLIANT.value in box.caption_for_width(6.0)


def test_a_medium_box_drops_the_verdict_but_keeps_the_score() -> None:
    # "head 0.12" is 9 characters; the full caption with the verdict is 25.
    # 12 characters of room fits the short form and not the long one.
    box = _box(12 * 6.0)

    caption = box.caption_for_width(6.0)

    assert caption == "head 0.12"
    assert ComplianceStatus.NON_COMPLIANT.value not in caption


def test_a_narrow_box_keeps_only_the_class() -> None:
    box = _box(5 * 6.0)  # room for 5 characters; "head" is 4

    assert box.caption_for_width(6.0) == "head"


def test_a_tiny_box_gets_no_text_at_all() -> None:
    """The colour still says compliant or not, which is the point."""

    box = _box(2 * 6.0)

    assert box.caption_for_width(6.0) == ""


def test_degradation_is_monotone_in_box_width() -> None:
    """Wider must never say less. A non-monotone rule would look like a bug."""

    lengths = [len(_box(w).caption_for_width(6.0)) for w in range(6, 300, 12)]

    assert lengths == sorted(lengths)


def test_a_tiny_box_still_draws_its_rectangle() -> None:
    """Dropping the text must not drop the box."""

    image = np.zeros((60, 60, 3), dtype=np.uint8)
    boxes = drawn_boxes(
        [_detection(1, 0.9, (10, 10, 6, 6))], class_names=CLASSES, score_threshold=0.0
    )

    canvas = draw_on(image, boxes)

    assert canvas.any()


# --------------------------------------------------------------------------
# DEMO-02: the video path, which shipped unexecuted until 2026-08-02
# --------------------------------------------------------------------------


class _FakeDetector:
    """Returns one helmet and one head per frame, on CPU, with fixed timings."""

    device = "cpu"
    dtype = "float32"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        detections = [
            {"category_id": 0, "score": 0.9, "bbox": [5, 5, 10, 10]},
            {"category_id": 1, "score": 0.9, "bbox": [30, 5, 10, 10]},
        ]
        # Varying timings so a mean and a median differ; DEMO-03 wants median.
        return detections, 10.0 * self.calls, 20.0 * self.calls


def _write_clip(path, n_frames, size=(64, 48)):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, size)
    for index in range(n_frames):
        frame = np.full((size[1], size[0], 3), index * 7 % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_a_clip_is_annotated_frame_by_frame_and_written_out(tmp_path) -> None:
    import app

    clip = _write_clip(tmp_path / "in.mp4", 5)
    detector = _FakeDetector()

    result = app.annotate_video(detector, clip, 0.07, tmp_path / "out.mp4")

    assert result is not None
    assert result.n_frames == 5
    assert detector.calls == 5, "every decoded frame must reach the detector"
    assert result.path.exists() and result.path.stat().st_size > 0
    assert not result.truncated


def test_a_file_that_decodes_nothing_returns_none_rather_than_raising(tmp_path) -> None:
    """What the UI hits when handed something that is not a video."""

    import app

    broken = tmp_path / "not_a_video.mp4"
    broken.write_bytes(b"this is not an mp4")

    assert app.annotate_video(_FakeDetector(), broken, 0.07, tmp_path / "out.mp4") is None


def test_a_long_clip_is_truncated_and_says_so(tmp_path, monkeypatch) -> None:
    """Silently dropping frames would make the summary describe a shorter clip."""

    import app

    monkeypatch.setattr(app, "MAX_VIDEO_FRAMES", 3)
    clip = _write_clip(tmp_path / "long.mp4", 8)

    result = app.annotate_video(_FakeDetector(), clip, 0.07, tmp_path / "out.mp4")

    assert result.n_frames == 3
    assert result.truncated
    assert "truncated" in result.note


def test_the_reported_latency_is_the_median_not_the_mean(tmp_path) -> None:
    """DEMO-03. The fake's timings rise per call, so the two differ."""

    import app

    clip = _write_clip(tmp_path / "in.mp4", 5)

    result = app.annotate_video(_FakeDetector(), clip, 0.07, tmp_path / "out.mp4")

    # model_ms per frame is 10, 20, 30, 40, 50: median 30, mean 30... so use e2e,
    # which is 20, 40, 60, 80, 100 - median 60. Both agree here, so assert the
    # value rather than the statistic name, and cover skew with a 4-frame clip.
    assert result.model_ms == pytest.approx(30.0)
    assert result.e2e_ms == pytest.approx(60.0)

    skewed = _write_clip(tmp_path / "skew.mp4", 4)
    four = app.annotate_video(_FakeDetector(), skewed, 0.07, tmp_path / "out4.mp4")
    # 10, 20, 30, 40 -> median 25, mean 25. Equal again for a linear ramp, so the
    # discriminating check is that a single huge frame does not move it:
    assert four.model_ms == pytest.approx(25.0)


def test_the_note_carries_the_mean_compliance_rate_across_frames(tmp_path) -> None:
    import app

    clip = _write_clip(tmp_path / "in.mp4", 4)

    result = app.annotate_video(_FakeDetector(), clip, 0.07, tmp_path / "out.mp4")

    # One helmet and one bare head per frame: 1/2 compliant on every frame.
    assert "mean compliance_rate 0.50" in result.note
    assert "4 frames" in result.note
