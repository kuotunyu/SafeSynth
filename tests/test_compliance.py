"""EVAL-01..EVAL-04: compliance derivation, both modes, and the operating sweep.

K-18 is the reason this file is long: a branch no test executes has zero
coverage no matter how green the suite is. Every mode, every guard clause and
every undefined-ratio case below is reached by something here.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import pytest

from src.inference import compliance as compliance_module
from src.inference.compliance import (
    COMPLIANCE_MODES,
    ComplianceConfigError,
    ComplianceStatus,
    Detection,
    OperatingPoint,
    SplitLeakageError,
    build_pair_descriptor,
    derive_compliance,
    detections_from_coco,
    is_worn_helmet_pair,
    load_evaluation_config,
    load_filtering_config,
    mode_class_direct,
    mode_geometric_pairing,
    select_operating_point,
    sweep_operating_points,
)
from src.training.data import Sample

ROOT = Path(__file__).resolve().parents[1]

# The scene threshold is stated by the tests rather than taken from the config,
# so that recalibrating compliance.score_threshold cannot silently change what
# these scenes are testing.
SCENE_THRESHOLD = 0.50

HELMET_CLASS_NAME = "helmet"
HEAD_CLASS_NAME = "head"

# A (helmet, head) pair that FILT-07 accepts as WORN. Verified against
# configs/filtering.yaml in test_scene_pair_is_accepted_by_filt_07 below, so the
# geometric-mode expectations rest on a checked premise, not on a hope.
WORN_HELMET_BOX = (110.0, 87.0, 40.0, 25.0)
HEAD_UNDER_HELMET_BOX = (108.0, 100.0, 44.0, 52.0)
BARE_HEAD_BOX = (250.0, 150.0, 30.0, 34.0)
DISTANT_HELMET_BOX = (300.0, 40.0, 22.0, 18.0)
# Same shape as WORN_HELMET_BOX but shifted right, so its dx numerator is 10 px
# instead of 0. Any test that has to tell "divide by head width" apart from
# "divide by head height" needs a pair whose offset is not zero.
OFFSET_HELMET_BOX = (120.0, 87.0, 40.0, 25.0)
# Contains BOTH the worn helmet and the head beneath it: an implementation that
# grouped detections by person would produce a different answer for this scene
# once the person box disappears, which is exactly what EVAL-03 forbids.
PERSON_OVER_BOTH_BOX = (90.0, 80.0, 90.0, 240.0)
# A `person` detection sitting exactly where a worn helmet would sit over
# BARE_HEAD_BOX. FILT-07 accepts this geometry (asserted as a premise in
# test_person_not_load_bearing), so any code path that lets a `person` box reach
# the helmet list flips that bare head to COMPLIANT. It is the whole point of
# EVAL-03 that adding or removing this detection changes nothing.
PERSON_SHAPED_LIKE_A_HELMET_BOX = (252.0, 140.0, 28.0, 20.0)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_evaluation_config()


@pytest.fixture(scope="module")
def filtering_config() -> dict:
    return load_filtering_config()


def multi_object_scene(*, with_person: bool) -> list[Detection]:
    """A realistic frame: a worn helmet, a bare head, a stray helmet, people.

    Two detections sit below SCENE_THRESHOLD so the score gate is exercised by
    every test that uses this scene rather than only by the dedicated one.
    """

    scene = [
        Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.93, "helmet_worn"),
        Detection(HEAD_CLASS_NAME, HEAD_UNDER_HELMET_BOX, 0.85, "head_under_helmet"),
        Detection("person", PERSON_SHAPED_LIKE_A_HELMET_BOX, 0.78, "person_as_helmet"),
        Detection("person", PERSON_OVER_BOTH_BOX, 0.71, "person_over_both"),
        Detection(HEAD_CLASS_NAME, BARE_HEAD_BOX, 0.66, "head_bare_far"),
        Detection(HELMET_CLASS_NAME, DISTANT_HELMET_BOX, 0.55, "helmet_far"),
        Detection("person", (240.0, 140.0, 70.0, 200.0), 0.40, "person_low_score"),
        Detection(HEAD_CLASS_NAME, (350.0, 300.0, 20.0, 22.0), 0.20, "head_low_score"),
    ]
    if with_person:
        return scene
    return [item for item in scene if item.class_name != "person"]


def verdict_ids(result) -> list[tuple[str | None, str]]:
    return [(item.detection_id, str(item.status)) for item in result.verdicts]


# spec: EVAL-01
def test_scene_pair_is_accepted_by_filt_07(filtering_config: dict) -> None:
    """The geometric expectations below depend on this pair actually pairing."""

    assert is_worn_helmet_pair(WORN_HELMET_BOX, HEAD_UNDER_HELMET_BOX, filtering_config)
    assert not is_worn_helmet_pair(DISTANT_HELMET_BOX, BARE_HEAD_BOX, filtering_config)
    assert not is_worn_helmet_pair(WORN_HELMET_BOX, BARE_HEAD_BOX, filtering_config)


# spec: EVAL-01
def test_pair_descriptor_matches_the_filt_07_definitions() -> None:
    descriptor = build_pair_descriptor(WORN_HELMET_BOX, HEAD_UNDER_HELMET_BOX)

    assert descriptor is not None
    assert descriptor["dx"] == pytest.approx(0.0)
    assert descriptor["dy"] == pytest.approx((100.0 - 99.5) / 52.0)
    assert descriptor["overlap_y"] == pytest.approx((87.0 + 25.0 - 100.0) / 52.0)
    assert descriptor["r_w"] == pytest.approx(40.0 / 44.0)
    assert descriptor["r_h"] == pytest.approx(25.0 / 52.0)
    assert descriptor["iou"] == pytest.approx(480.0 / 2808.0)


# spec: EVAL-01
def test_pair_descriptor_normalizes_dx_by_head_width_not_head_height() -> None:
    """FILT-07 fixes dx = (helmet.cx - head.cx) / head.w. The head is not square.

    OFFSET_HELMET_BOX centre x = 120 + 40/2 = 140; HEAD_UNDER_HELMET_BOX centre
    x = 108 + 44/2 = 130, so the numerator is 10 px. head.w = 44 and head.h = 52,
    which are deliberately different: normalizing by width gives 10/44 = 0.2272...
    and by height 10/52 = 0.1923..., so this scene can tell the two apart. The
    other pair in this file has both centres at x = 130 and therefore cannot.
    """

    descriptor = build_pair_descriptor(OFFSET_HELMET_BOX, HEAD_UNDER_HELMET_BOX)

    assert descriptor is not None
    assert descriptor["dx"] == pytest.approx(10.0 / 44.0)
    # dy and overlap_y are the ones normalized by head HEIGHT; shifting the
    # helmet sideways must leave them exactly where the square-offset pair had
    # them, which pins the two normalizers to different axes.
    assert descriptor["dy"] == pytest.approx((100.0 - 99.5) / 52.0)
    assert descriptor["overlap_y"] == pytest.approx((87.0 + 25.0 - 100.0) / 52.0)


# spec: EVAL-01
def test_pair_descriptor_is_none_for_a_degenerate_head(filtering_config: dict) -> None:
    assert build_pair_descriptor(WORN_HELMET_BOX, (108.0, 100.0, 0.0, 52.0)) is None
    assert not is_worn_helmet_pair(WORN_HELMET_BOX, (108.0, 100.0, 44.0, 0.0), filtering_config)


# spec: EVAL-01
def test_geometric_pairing_without_a_filtering_config_refuses() -> None:
    with pytest.raises(ComplianceConfigError, match="filtering.yaml"):
        is_worn_helmet_pair(WORN_HELMET_BOX, HEAD_UNDER_HELMET_BOX, None)


# spec: EVAL-01
def test_class_direct_maps_the_class_straight_to_the_status(config: dict) -> None:
    result = derive_compliance(
        multi_object_scene(with_person=True),
        config=config,
        mode="class_direct",
        score_threshold=SCENE_THRESHOLD,
    )

    assert verdict_ids(result) == [
        ("helmet_worn", "COMPLIANT"),
        ("head_under_helmet", "NON_COMPLIANT"),
        ("head_bare_far", "NON_COMPLIANT"),
        ("helmet_far", "COMPLIANT"),
    ]
    assert result.verdicts[0].status is ComplianceStatus.COMPLIANT
    assert result.verdicts[1].status is ComplianceStatus.NON_COMPLIANT
    assert (result.n_compliant, result.n_non_compliant) == (2, 2)
    assert result.compliance_rate == 0.5


# spec: EVAL-01
def test_geometric_pairing_judges_heads_and_treats_helmets_as_evidence(
    config: dict, filtering_config: dict
) -> None:
    result = derive_compliance(
        multi_object_scene(with_person=True),
        config=config,
        filtering_config=filtering_config,
        mode="geometric_pairing",
        score_threshold=SCENE_THRESHOLD,
    )

    assert verdict_ids(result) == [
        ("head_under_helmet", "COMPLIANT"),
        ("head_bare_far", "NON_COMPLIANT"),
    ]
    assert (result.n_compliant, result.n_non_compliant) == (1, 1)


# spec: EVAL-01
def test_switching_mode_changes_the_answer_on_a_scene_built_to_disagree(
    config: dict, filtering_config: dict
) -> None:
    """A worn pair is where the two semantics genuinely conflict.

    class_direct sees two judgeable objects (a compliant helmet and a
    non-compliant bare head); geometric_pairing sees one head wearing one
    helmet. If this test ever passes for both modes with the same numbers, the
    mode switch has stopped meaning anything.
    """

    worn_pair = [
        Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.93, "helmet_worn"),
        Detection(HEAD_CLASS_NAME, HEAD_UNDER_HELMET_BOX, 0.85, "head_under_helmet"),
    ]
    direct = derive_compliance(
        worn_pair, config=config, mode="class_direct", score_threshold=SCENE_THRESHOLD
    )
    geometric = derive_compliance(
        worn_pair,
        config=config,
        filtering_config=filtering_config,
        mode="geometric_pairing",
        score_threshold=SCENE_THRESHOLD,
    )

    assert (direct.n_compliant, direct.n_non_compliant) == (1, 1)
    assert direct.compliance_rate == 0.5
    assert (geometric.n_compliant, geometric.n_non_compliant) == (1, 0)
    assert geometric.compliance_rate == 1.0
    assert direct.verdicts != geometric.verdicts


# spec: EVAL-03
@pytest.mark.parametrize("mode", sorted(COMPLIANCE_MODES))
def test_person_not_load_bearing(mode: str, config: dict, filtering_config: dict) -> None:
    """Deleting every `person` detection must change NOTHING, bit for bit.

    The scene is deliberately not a toy. It carries two person boxes that a
    sloppy implementation would trip over in different ways:

    * PERSON_OVER_BOTH_BOX contains both the worn helmet and the head under it,
      so anything that grouped heads with helmets by person answers differently
      once that box is gone.
    * PERSON_SHAPED_LIKE_A_HELMET_BOX sits over the far BARE head in exactly the
      pose FILT-07 calls a worn helmet, so anything that lets `person` leak into
      the helmet list reports that bare head as COMPLIANT.

    Equality on the frozen result is necessary but not sufficient: two runs that
    are both wrong are still equal. So the verdict for the bare head is pinned by
    hand as well, and the FILT-07 premise behind the second person box is checked
    rather than assumed.
    """

    kwargs = {
        "config": config,
        "filtering_config": filtering_config,
        "mode": mode,
        "score_threshold": SCENE_THRESHOLD,
    }
    with_person = derive_compliance(multi_object_scene(with_person=True), **kwargs)
    without_person = derive_compliance(multi_object_scene(with_person=False), **kwargs)

    assert with_person == without_person
    # And the scene really did carry person detections worth ignoring.
    assert any(item.class_name == "person" for item in multi_object_scene(with_person=True))
    assert with_person.verdicts

    # Premise: that person box IS a FILT-07-plausible helmet over the bare head.
    # Without this the equality above could hold for the boring reason that the
    # person box was harmless all along.
    assert is_worn_helmet_pair(PERSON_SHAPED_LIKE_A_HELMET_BOX, BARE_HEAD_BOX, filtering_config)

    # The far head is bare under BOTH semantics: class_direct reads its class,
    # and geometric_pairing finds no `helmet` anywhere near it. A person box may
    # not change that in either direction.
    for result in (with_person, without_person):
        statuses = {item.detection_id: str(item.status) for item in result.verdicts}
        assert statuses["head_bare_far"] == "NON_COMPLIANT"

    # The minimal counter-example, kept literal: one head plus one person box in
    # the position a worn helmet would occupy. The head is bare in both
    # semantics, so the rate is 0.0 with the person and 0.0 without it.
    lone_head = Detection(HEAD_CLASS_NAME, HEAD_UNDER_HELMET_BOX, 0.90, "lone_head")
    person_as_helmet = Detection("person", WORN_HELMET_BOX, 0.80, "person_in_helmet_pose")
    with_minimal = derive_compliance([lone_head, person_as_helmet], **kwargs)
    without_minimal = derive_compliance([lone_head], **kwargs)

    assert with_minimal == without_minimal
    assert with_minimal.compliance_rate == 0.0
    assert (with_minimal.n_compliant, with_minimal.n_non_compliant) == (0, 1)


# spec: EVAL-02
def test_person_never_appears_in_a_verdict(config: dict) -> None:
    result = derive_compliance(
        multi_object_scene(with_person=True),
        config=config,
        mode="class_direct",
        score_threshold=SCENE_THRESHOLD,
    )

    assert all(item.class_name != "person" for item in result.verdicts)


# spec: EVAL-03
def test_mode_functions_ignore_person_even_when_handed_one(filtering_config: dict) -> None:
    """The mode functions are also person-proof on their own, not only via the caller.

    `_drop_person` never runs here, so this is the layer that has to survive a
    person box arriving in-band. Statuses are asserted, not just class names: a
    helmet list built as "everything that is not a head" would still emit exactly
    one verdict per head, and only the STATUS of the bare head would betray it.
    """

    scene = [
        Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.93),
        Detection(HEAD_CLASS_NAME, HEAD_UNDER_HELMET_BOX, 0.85),
        Detection("person", PERSON_SHAPED_LIKE_A_HELMET_BOX, 0.78),
        Detection("person", PERSON_OVER_BOTH_BOX, 0.71),
        Detection(HEAD_CLASS_NAME, BARE_HEAD_BOX, 0.66),
    ]

    direct = mode_class_direct(scene)
    geometric = mode_geometric_pairing(scene, filtering_config)

    assert [(detection.class_name, str(status)) for detection, status in direct] == [
        (HELMET_CLASS_NAME, "COMPLIANT"),
        (HEAD_CLASS_NAME, "NON_COMPLIANT"),
        (HEAD_CLASS_NAME, "NON_COMPLIANT"),
    ]
    assert [(detection.bbox_xywh, str(status)) for detection, status in geometric] == [
        (HEAD_UNDER_HELMET_BOX, "COMPLIANT"),
        (BARE_HEAD_BOX, "NON_COMPLIANT"),
    ]


# spec: EVAL-03
def test_the_mode_function_never_receives_a_person_detection(
    monkeypatch: pytest.MonkeyPatch, config: dict
) -> None:
    """`_drop_person` is a layer that runs, not a sentence in a docstring.

    The test above proves each shipped mode filters `person` out for itself. That
    is exactly why deleting `_drop_person` changes no verdict anywhere, and why
    no assertion about a ComplianceResult can tell whether it ran: the two
    person-guards mask each other, which is how the K-19 two-guard mutation got
    to survive in the first place. So this looks at the hand-off instead. A spy
    mode is registered and records the class names it was actually given; if
    `_drop_person` stops dropping, the person boxes show up in that record even
    though every verdict stays identical.
    """

    seen: list[tuple[str, ...]] = []

    def spy(detections: Sequence[Detection], filtering_config: object = None) -> tuple:
        seen.append(tuple(item.class_name for item in detections))
        return ()

    monkeypatch.setitem(compliance_module.COMPLIANCE_MODES, "spy_mode", spy)

    scene = multi_object_scene(with_person=True)
    derive_compliance(scene, config=config, mode="spy_mode", score_threshold=SCENE_THRESHOLD)

    # Two of the three person boxes in this scene score 0.78 and 0.71, above
    # SCENE_THRESHOLD, so the score gate cannot be what removes them.
    assert sum(1 for item in scene if item.class_name == "person") == 3
    assert seen == [("helmet", "head", "head", "helmet")]


# spec: EVAL-02
def test_zero_detections_leaves_the_rate_undefined_rather_than_nan(config: dict) -> None:
    """No detections is not 0% compliance and not 100%; it is no evidence."""

    result = derive_compliance([], config=config, mode="class_direct")

    assert result.verdicts == ()
    assert (result.n_compliant, result.n_non_compliant) == (0, 0)
    assert result.compliance_rate is None


# spec: EVAL-04
def test_everything_below_the_threshold_is_the_zero_detection_case(config: dict) -> None:
    result = derive_compliance(
        [Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.1)],
        config=config,
        mode="class_direct",
        score_threshold=SCENE_THRESHOLD,
    )

    assert result.compliance_rate is None


# spec: EVAL-04
def test_a_detection_exactly_at_the_threshold_is_kept(config: dict) -> None:
    """Excluding what is BELOW the threshold means the boundary itself counts."""

    result = derive_compliance(
        [Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, SCENE_THRESHOLD)],
        config=config,
        mode="class_direct",
        score_threshold=SCENE_THRESHOLD,
    )

    assert result.compliance_rate == 1.0


# spec: EVAL-04
def test_the_configured_threshold_and_mode_are_the_defaults(config: dict) -> None:
    result = derive_compliance([], config=config)

    assert result.score_threshold == float(config["compliance"]["score_threshold"])
    assert result.mode == str(config["compliance"]["mode"])


# spec: EVAL-01
def test_geometric_mode_loads_the_filtering_config_when_not_given(config: dict) -> None:
    result = derive_compliance(
        [
            Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.93),
            Detection(HEAD_CLASS_NAME, HEAD_UNDER_HELMET_BOX, 0.85),
        ],
        config=config,
        mode="geometric_pairing",
        score_threshold=SCENE_THRESHOLD,
    )

    assert result.compliance_rate == 1.0


# spec: EVAL-01, EVAL-04
def test_the_fallback_filtering_config_is_parsed_once_not_once_per_call(
    monkeypatch: pytest.MonkeyPatch, config: dict
) -> None:
    """The disk read behind the geometric fallback is amortized, not per-call.

    sweep_operating_points calls derive_compliance once per
    (image x candidate threshold), so a fresh YAML parse here is paid tens of
    thousands of times on the 756-image Validation split. The loader is
    monkeypatched and counted; the answer is checked too, so satisfying the count
    by never loading at all is not an option.

    The memo holds ONE slot for the default config and takes no path argument, so
    "once" means once per process. Identity is pinned as well as the count: a memo
    that handed back a fresh copy of an equal mapping would still parse once, and
    the module comment promises every caller the SAME shared object.
    """

    parse_count = 0
    real_loader = compliance_module.load_filtering_config

    def counting_loader() -> dict:
        nonlocal parse_count
        parse_count += 1
        return real_loader()

    # Start from a cold memo so an earlier test cannot make this pass for free.
    monkeypatch.setattr(compliance_module, "_FILTERING_CONFIG", None)
    monkeypatch.setattr(compliance_module, "load_filtering_config", counting_loader)

    worn_pair = [
        Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.93),
        Detection(HEAD_CLASS_NAME, HEAD_UNDER_HELMET_BOX, 0.85),
    ]
    rates = [
        derive_compliance(
            worn_pair,
            config=config,
            mode="geometric_pairing",
            score_threshold=SCENE_THRESHOLD,
        ).compliance_rate
        for _ in range(25)
    ]

    assert rates == [1.0] * 25
    assert parse_count == 1

    first = compliance_module._cached_filtering_config()
    second = compliance_module._cached_filtering_config()
    assert first is second
    assert parse_count == 1


# spec: EVAL-01
def test_an_explicit_filtering_config_is_used_instead_of_the_memo(
    monkeypatch: pytest.MonkeyPatch, config: dict, filtering_config: dict
) -> None:
    """Different FILT-07 thresholds are asked for BY VALUE, never by path.

    This is why the memo above needs no path parameter and no cache key: a caller
    who wants other thresholds hands over the parsed mapping, and the memo is not
    consulted at all. Both halves are asserted, because either alone has a passing
    bug - the loader must never be called, AND the mapping that was passed must be
    the one that decides the verdict.

    The override is one number. WORN_HELMET_BOX over HEAD_UNDER_HELMET_BOX has
    overlap_y = (87 + 25 - 100) / 52 = 0.2308, which clears the shipped
    rules.helmet_above_head.overlap_y_min (asserted below, so this scene cannot
    quietly start out rejected). Raising that floor to 0.90 - a value no config
    file contains - turns the same pair into a floating helmet, so the head
    beneath it flips from COMPLIANT to NON_COMPLIANT.
    """

    def exploding_loader() -> dict:
        raise AssertionError("the memo must not be read when a filtering_config is passed")

    monkeypatch.setattr(compliance_module, "_FILTERING_CONFIG", None)
    monkeypatch.setattr(compliance_module, "load_filtering_config", exploding_loader)

    assert float(filtering_config["rules"]["helmet_above_head"]["overlap_y_min"]) < 12.0 / 52.0
    tightened = copy.deepcopy(filtering_config)
    tightened["rules"]["helmet_above_head"]["overlap_y_min"] = 0.90

    worn_pair = [
        Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.93, "helmet_worn"),
        Detection(HEAD_CLASS_NAME, HEAD_UNDER_HELMET_BOX, 0.85, "head_under_helmet"),
    ]
    kwargs = {
        "config": config,
        "mode": "geometric_pairing",
        "score_threshold": SCENE_THRESHOLD,
    }
    shipped_result = derive_compliance(worn_pair, filtering_config=filtering_config, **kwargs)
    tightened_result = derive_compliance(worn_pair, filtering_config=tightened, **kwargs)

    assert verdict_ids(shipped_result) == [("head_under_helmet", "COMPLIANT")]
    assert verdict_ids(tightened_result) == [("head_under_helmet", "NON_COMPLIANT")]


# spec: EVAL-01
def test_unknown_mode_is_refused(config: dict) -> None:
    with pytest.raises(ComplianceConfigError, match="Unknown compliance.mode"):
        derive_compliance([], config=config, mode="vibes")


# spec: EVAL-03
def test_turning_on_the_person_grouping_hint_is_refused(config: dict) -> None:
    """A hint we do not implement must fail loudly, not no-op convincingly."""

    mutated = copy.deepcopy(dict(config))
    mutated["compliance"]["use_person_as_grouping_hint"] = True

    with pytest.raises(ComplianceConfigError, match="use_person_as_grouping_hint"):
        derive_compliance([], config=mutated)


def test_the_shipped_config_keeps_the_grouping_hint_off(config: dict) -> None:
    assert config["compliance"]["use_person_as_grouping_hint"] is False
    assert config["compliance"]["mode"] in COMPLIANCE_MODES


def test_detection_rejects_an_unknown_class() -> None:
    with pytest.raises(ValueError, match="Unknown class"):
        Detection("hardhat", WORN_HELMET_BOX, 0.9)


def test_detection_rejects_a_malformed_box() -> None:
    with pytest.raises(ValueError, match="bbox_xywh"):
        Detection(HELMET_CLASS_NAME, (1.0, 2.0, 3.0), 0.9)


def test_detection_normalizes_boxes_so_equality_is_exact() -> None:
    from_list = Detection(HELMET_CLASS_NAME, [110, 87, 40, 25], 1)
    from_tuple = Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 1.0)

    assert from_list == from_tuple
    assert isinstance(from_list.bbox_xywh, tuple)


def test_detections_from_coco_groups_by_image() -> None:
    grouped = detections_from_coco(
        [
            {"image_id": 7, "category_id": 0, "bbox": [110, 87, 40, 25], "score": 0.9},
            {"image_id": 7, "category_id": 1, "bbox": [250, 150, 30, 34], "score": 0.7},
            {"image_id": 8, "category_id": 2, "bbox": [90, 80, 90, 240], "score": 0.6},
        ]
    )

    assert sorted(grouped) == [7, 8]
    assert [item.class_name for item in grouped[7]] == ["helmet", "head"]
    assert grouped[8][0].class_name == "person"


def test_detections_from_coco_rejects_a_non_contiguous_category() -> None:
    with pytest.raises(ValueError, match="outside the contiguous range"):
        detections_from_coco([{"image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1], "score": 1.0}])


def test_config_loader_rejects_a_yaml_file_that_is_not_a_mapping(tmp_path: Path) -> None:
    broken = tmp_path / "evaluation.yaml"
    broken.write_text("- not\n- a mapping\n", encoding="utf-8", newline="\n")

    with pytest.raises(TypeError, match="Expected a mapping"):
        load_evaluation_config(broken)


def test_config_loaders_accept_an_explicit_path() -> None:
    explicit = load_evaluation_config(ROOT / "configs" / "evaluation.yaml")
    filtering = load_filtering_config(ROOT / "configs" / "filtering.yaml")

    assert explicit["compliance"]["mode"] in COMPLIANCE_MODES
    assert "helmet_above_head" in filtering["rules"]


def sweep_ground_truth() -> list[Sample]:
    """One image: a helmet, a bare head, and a person that must be ignored."""

    return [
        Sample(
            image_path=Path("frame.png"),
            image_id=1,
            boxes_xywh=(WORN_HELMET_BOX, BARE_HEAD_BOX, PERSON_OVER_BOTH_BOX),
            class_indices=(0, 1, 2),
            is_synthetic=False,
        )
    ]


def one_image_ground_truth(
    boxes: tuple[tuple[float, float, float, float], ...],
    class_indices: tuple[int, ...],
    *,
    image_id: int = 1,
) -> Sample:
    """One GT image. Class indices follow CLASS_NAMES: 0 helmet, 1 head, 2 person."""

    return Sample(
        image_path=Path(f"frame_{image_id}.png"),
        image_id=image_id,
        boxes_xywh=boxes,
        class_indices=class_indices,
        is_synthetic=False,
    )


def sweep_detections() -> dict[int, list[Detection]]:
    return {
        1: [
            Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.9, "tp_helmet"),
            Detection(HEAD_CLASS_NAME, BARE_HEAD_BOX, 0.7, "tp_head"),
            Detection(HELMET_CLASS_NAME, (10.0, 10.0, 20.0, 20.0), 0.6, "fp_helmet"),
        ]
    }


# spec: EVAL-04
def test_sweep_refuses_the_test_split(config: dict) -> None:
    for split in ("test", "TEST", "  Test  "):
        with pytest.raises(SplitLeakageError, match="Validation only"):
            sweep_operating_points(
                sweep_detections(), sweep_ground_truth(), split=split, config=config
            )


# spec: EVAL-04
def test_sweep_reports_recall_and_precision_per_threshold(config: dict) -> None:
    points = sweep_operating_points(
        sweep_detections(), sweep_ground_truth(), split="validation", config=config
    )

    assert [point.score_threshold for point in points] == [0.6, 0.7, 0.9]
    at_060, at_070, at_090 = points
    # The stray helmet is a compliant prediction with no compliant GT to match.
    assert at_060.n_predicted_compliant == 2
    assert at_060.compliance_precision == 0.5
    assert at_060.bare_head_recall == 1.0
    assert at_070.compliance_precision == 1.0
    assert at_070.bare_head_recall == 1.0
    # Above the head's score the bare head is no longer reported at all.
    assert at_090.bare_head_recall == 0.0
    assert at_090.n_predicted_non_compliant == 0
    assert at_090.n_ground_truth_bare_heads == 1


# spec: EVAL-04
def test_sweep_matching_is_one_to_one(config: dict) -> None:
    """Two predictions on one GT box may not both count as true positives."""

    detections = {
        1: [
            Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.9),
            Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.8),
        ]
    }
    (point,) = sweep_operating_points(
        detections,
        sweep_ground_truth(),
        split="validation",
        config=config,
        thresholds=[0.5],
    )

    assert point.n_predicted_compliant == 2
    assert point.compliance_precision == 0.5


# spec: EVAL-04
def test_sweep_leaves_precision_undefined_when_nothing_is_predicted(config: dict) -> None:
    (point,) = sweep_operating_points(
        sweep_detections(),
        sweep_ground_truth(),
        split="validation",
        config=config,
        thresholds=[0.95],
    )

    assert point.compliance_precision is None
    assert point.bare_head_recall == 0.0


# spec: EVAL-04
def test_sweep_leaves_recall_undefined_when_there_are_no_bare_heads(config: dict) -> None:
    helmet_only = [
        Sample(
            image_path=Path("frame.png"),
            image_id=1,
            boxes_xywh=(WORN_HELMET_BOX,),
            class_indices=(0,),
            is_synthetic=False,
        )
    ]
    (point,) = sweep_operating_points(
        sweep_detections(), helmet_only, split="validation", config=config, thresholds=[0.5]
    )

    assert point.bare_head_recall is None
    assert point.n_ground_truth_bare_heads == 0


# spec: EVAL-04
def test_sweep_with_no_detections_falls_back_to_the_configured_threshold(config: dict) -> None:
    points = sweep_operating_points({}, sweep_ground_truth(), split="validation", config=config)

    assert len(points) == 1
    assert points[0].score_threshold == float(config["compliance"]["score_threshold"])
    assert points[0].compliance_precision is None
    assert points[0].bare_head_recall == 0.0


# spec: EVAL-04
def test_sweep_rejects_detections_from_images_outside_the_split(config: dict) -> None:
    detections = dict(sweep_detections())
    detections[999] = [Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.9)]

    with pytest.raises(SplitLeakageError, match="no ground truth"):
        sweep_operating_points(
            detections, sweep_ground_truth(), split="validation", config=config
        )


# spec: EVAL-04
def test_sweep_runs_in_geometric_mode_too(config: dict, filtering_config: dict) -> None:
    points = sweep_operating_points(
        sweep_detections(),
        sweep_ground_truth(),
        split="validation",
        config=config,
        filtering_config=filtering_config,
        mode="geometric_pairing",
        thresholds=[0.5],
    )

    # No helmet pairs with the distant bare head, so it stays non-compliant and
    # nothing is predicted compliant at all under this semantics.
    assert points[0].n_predicted_compliant == 0
    assert points[0].compliance_precision is None
    assert points[0].bare_head_recall == 1.0


# spec: EVAL-04, EVAL-05
def test_a_near_miss_box_is_not_a_recalled_bare_head(config: dict) -> None:
    """Bare-head recall is gated on metrics.bare_head_recall_iou, not on "overlaps".

    GT: one bare head, 30 x 34 = 1020 px^2, at x = 250.
    Prediction: the same 30 x 34 box shifted 15 px right, so the intersection is
    15 x 34 = 510 and the union is 1020 + 1020 - 510 = 1530, giving IoU = 1/3.
    One third is well above zero, so a matcher that applies no gate at all pairs
    them and reports full recall; one third is below the configured gate, so the
    honest answer is that this bare head was MISSED. The second sweep pins the
    other side, otherwise "recall is 0.0" could mean the matcher never matches
    anything.
    """

    gate = float(config["metrics"]["bare_head_recall_iou"])
    assert gate > 1 / 3

    (near_miss,) = sweep_operating_points(
        {1: [Detection(HEAD_CLASS_NAME, (265.0, 150.0, 30.0, 34.0), 0.9, "shifted")]},
        [one_image_ground_truth((BARE_HEAD_BOX,), (1,))],
        split="validation",
        config=config,
        thresholds=[SCENE_THRESHOLD],
    )
    (exact,) = sweep_operating_points(
        {1: [Detection(HEAD_CLASS_NAME, BARE_HEAD_BOX, 0.9, "on_the_box")]},
        [one_image_ground_truth((BARE_HEAD_BOX,), (1,))],
        split="validation",
        config=config,
        thresholds=[SCENE_THRESHOLD],
    )

    assert near_miss.n_predicted_non_compliant == 1
    assert near_miss.n_ground_truth_bare_heads == 1
    assert near_miss.bare_head_recall == 0.0
    assert exact.bare_head_recall == 1.0


# spec: EVAL-04
def test_a_helmet_predicted_over_a_bare_head_is_never_a_true_compliant(config: dict) -> None:
    """The safety-critical failure of this whole project, scored explicitly.

    `helmet` and `head` are mutually exclusive per person (ADR-007), so a helmet
    predicted on top of a bare-head ground-truth box is a FALSE compliant - the
    model saying "protected" about someone who is not.

    GT: one bare head. Prediction: a helmet on the identical box, IoU 1.0. There
    are zero compliant GT boxes, so of the single compliant prediction none can
    be true: precision = 0/1 = 0.0. Nothing was predicted non-compliant, so the
    bare head is unrecalled: 0/1 = 0.0. Matching that ignored ground-truth status
    would instead score this perfect, on IoU alone.
    """

    (point,) = sweep_operating_points(
        {1: [Detection(HELMET_CLASS_NAME, BARE_HEAD_BOX, 0.9, "false_compliant")]},
        [one_image_ground_truth((BARE_HEAD_BOX,), (1,))],
        split="validation",
        config=config,
        thresholds=[SCENE_THRESHOLD],
    )

    assert point.n_predicted_compliant == 1
    assert point.n_predicted_non_compliant == 0
    assert point.n_ground_truth_bare_heads == 1
    assert point.compliance_precision == 0.0
    assert point.bare_head_recall == 0.0


# spec: EVAL-04
def test_the_sweep_aggregates_over_every_ground_truth_image(config: dict) -> None:
    """Cross-image aggregation is the only thing the sweep exists to do.

    Three images, each contributing something different to the same totals:
      image 1 - GT one bare head, predicted as a head on the same box
                -> +1 non-compliant prediction, +1 matched bare head, +1 bare head
      image 2 - GT one helmet, predicted as a helmet on the same box
                -> +1 compliant prediction, +1 true compliant, +0 bare heads
      image 3 - GT one bare head, nothing predicted at all
                -> +0 predictions, +0 matched, +1 bare head
    Totals: precision = 1/1 = 1.0 and recall = 1/2 = 0.5, over 2 GT bare heads.

    No single image produces that row, so truncating the loop changes it; and
    image 3, iterated last, contributes zero to three of the five accumulators,
    so an assignment written where an increment belongs collapses them.
    """

    ground_truth = [
        one_image_ground_truth((BARE_HEAD_BOX,), (1,), image_id=1),
        one_image_ground_truth((WORN_HELMET_BOX,), (0,), image_id=2),
        # A second bare head, far from every other box in this scene.
        one_image_ground_truth(((40.0, 200.0, 26.0, 30.0),), (1,), image_id=3),
    ]
    detections = {
        1: [Detection(HEAD_CLASS_NAME, BARE_HEAD_BOX, 0.9, "recalled_bare_head")],
        2: [Detection(HELMET_CLASS_NAME, WORN_HELMET_BOX, 0.9, "true_compliant")],
    }

    (point,) = sweep_operating_points(
        detections,
        ground_truth,
        split="validation",
        config=config,
        thresholds=[SCENE_THRESHOLD],
    )

    assert point.n_predicted_compliant == 1
    assert point.n_predicted_non_compliant == 1
    assert point.n_ground_truth_bare_heads == 2
    assert point.compliance_precision == 1.0
    assert point.bare_head_recall == 0.5


# spec: EVAL-04
def test_greedy_matching_consumes_predictions_in_descending_score(config: dict) -> None:
    """Greedy one-to-one matching is order-sensitive, and the order is score-first.

    Two ground-truth bare heads, 40 x 40 each, at x = 100 and x = 120, same y.
    Two predicted bare heads, also 40 x 40 (every box is 1600 px^2, so an x-offset
    of d gives intersection (40 - d) * 40 and union 3200 - that):
      strong, score 0.9, at x =  94 -> IoU 1360/1840 = 0.739 with GT #1,
                                          560/2640 = 0.212 with GT #2
      weak,   score 0.7, at x = 108 -> IoU 1280/1920 = 0.667 with GT #1,
                                         1120/2080 = 0.538 with GT #2
    Highest score first: strong claims GT #1 (its best), weak is left GT #2 and
    still clears the gate -> 2 of 2 recalled. Lowest score first: weak claims
    GT #1 (its own best too), and strong is left only 0.212, below the gate ->
    1 of 2. The ordering is worth exactly half the recall on this scene.
    """

    gate = float(config["metrics"]["bare_head_recall_iou"])
    assert 560 / 2640 < gate <= 1120 / 2080

    detections = {
        1: [
            Detection(HEAD_CLASS_NAME, (94.0, 100.0, 40.0, 40.0), 0.9, "strong"),
            Detection(HEAD_CLASS_NAME, (108.0, 100.0, 40.0, 40.0), 0.7, "weak"),
        ]
    }
    (point,) = sweep_operating_points(
        detections,
        [one_image_ground_truth(((100.0, 100.0, 40.0, 40.0), (120.0, 100.0, 40.0, 40.0)), (1, 1))],
        split="validation",
        config=config,
        thresholds=[SCENE_THRESHOLD],
    )

    assert point.n_predicted_non_compliant == 2
    assert point.n_ground_truth_bare_heads == 2
    assert point.bare_head_recall == 1.0


# spec: EVAL-04
def test_select_operating_point_maximises_recall_under_the_precision_floor(
    config: dict,
) -> None:
    minimum = float(config["compliance"]["min_compliance_precision"])
    points = (
        # Highest recall of all, but too imprecise to qualify.
        OperatingPoint(0.30, 0.95, minimum - 0.1, 10, 10, 10),
        OperatingPoint(0.50, 0.60, minimum, 10, 10, 10),
        OperatingPoint(0.70, 0.80, 1.0, 10, 10, 10),
        # Undefined recall can never be "the best recall".
        OperatingPoint(0.90, None, 1.0, 10, 10, 10),
    )

    chosen = select_operating_point(points, config=config)

    assert chosen is not None
    assert chosen.score_threshold == 0.70


# spec: EVAL-04
def test_a_point_sitting_exactly_on_the_precision_floor_can_win(config: dict) -> None:
    """EVAL-04 says "not BELOW the floor", so the boundary point is eligible.

    The previous test has the boundary point losing on recall anyway, which means
    it cannot tell an inclusive floor from an exclusive one. Here the boundary
    point has the best recall of the two, so it wins iff the comparison admits
    equality; an exclusive floor would silently hand the answer to the lower-recall
    point instead - a real cost in missed bare heads, paid to no one.
    """

    minimum = float(config["compliance"]["min_compliance_precision"])
    on_the_floor = OperatingPoint(0.40, 0.90, minimum, 10, 10, 10)
    well_above = OperatingPoint(0.70, 0.50, 1.0, 10, 10, 10)

    chosen = select_operating_point((on_the_floor, well_above), config=config)

    assert chosen is on_the_floor


# spec: EVAL-04
def test_select_operating_point_returns_none_when_nothing_qualifies(config: dict) -> None:
    minimum = float(config["compliance"]["min_compliance_precision"])
    points = (
        OperatingPoint(0.30, 0.95, minimum - 0.2, 10, 10, 10),
        OperatingPoint(0.50, 0.60, None, 0, 10, 10),
    )

    assert select_operating_point(points, config=config) is None


# spec: EVAL-04
def test_select_operating_point_on_a_real_sweep(config: dict) -> None:
    points = sweep_operating_points(
        sweep_detections(), sweep_ground_truth(), split="validation", config=config
    )

    chosen = select_operating_point(points, config=config)

    assert chosen is not None
    assert chosen.score_threshold == 0.7
    assert chosen.bare_head_recall == 1.0
