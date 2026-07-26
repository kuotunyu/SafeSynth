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
