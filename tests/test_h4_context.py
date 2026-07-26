from scripts.analyze_h4_context import _near_person_anchor


def test_near_person_anchor_accepts_head_above_upper_body() -> None:
    assert _near_person_anchor(
        headlike_xywh=[45.0, 15.0, 10.0, 10.0],
        person_xywh=[35.0, 20.0, 30.0, 100.0],
    )


def test_near_person_anchor_rejects_distant_or_lower_body_boxes() -> None:
    person = [35.0, 20.0, 30.0, 100.0]

    assert not _near_person_anchor([100.0, 15.0, 10.0, 10.0], person)
    assert not _near_person_anchor([45.0, 100.0, 10.0, 10.0], person)
