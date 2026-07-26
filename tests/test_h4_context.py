from scripts.analyze_h4_replacement import _group_summary
from src.filtering.artifact_gate import has_person_context


def test_near_person_anchor_accepts_head_above_upper_body() -> None:
    assert has_person_context(
        [45.0, 15.0, 10.0, 10.0],
        [[35.0, 20.0, 30.0, 100.0]],
    )


def test_near_person_anchor_rejects_distant_or_lower_body_boxes() -> None:
    person = [35.0, 20.0, 30.0, 100.0]

    assert not has_person_context([100.0, 15.0, 10.0, 10.0], [person])
    assert not has_person_context([45.0, 100.0, 10.0, 10.0], [person])


def test_replacement_group_summary_uses_medians_and_fractions() -> None:
    rows = [
        {
            "score": 0.1,
            "source_min_side": 10,
            "paste_to_source_area_ratio": 0.8,
            "paste_to_anchor_area_ratio": 0.9,
            "postfx_applied": False,
            "filter_pass": True,
        },
        {
            "score": 0.9,
            "source_min_side": 20,
            "paste_to_source_area_ratio": 1.2,
            "paste_to_anchor_area_ratio": 1.1,
            "postfx_applied": True,
            "filter_pass": False,
        },
    ]

    summary = _group_summary(rows)

    assert summary["score_median"] == 0.5
    assert summary["source_min_side_median"] == 15
    assert summary["postfx_fraction"] == 0.5
    assert summary["filter_pass_fraction"] == 0.5
