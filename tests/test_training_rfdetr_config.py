"""The RF-DETR config must not drift back into being a copy of the RT-DETR one.

The failure this guards against is not a crash. RT-DETR sets `do_normalize:
false` because it only divides by 255, and training.yaml carries a loud warning
saying so. RF-DETR REQUIRES ImageNet normalization. Copying the RT-DETR value
across would train a DINOv2 backbone on unnormalized input and lose accuracy
silently, which is the same class of mistake in the opposite direction.

These tests read both configs and assert they disagree where the checkpoints
disagree. They deliberately do NOT re-derive the right answer - that came from
the checkpoint on 2026-08-02 and is recorded in the config's own comments.
"""

from __future__ import annotations

import yaml

from src.data.paths import PROJECT_ROOT

RTDETR = PROJECT_ROOT / "configs" / "training.yaml"
RFDETR = PROJECT_ROOT / "configs" / "training_rfdetr.yaml"


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_rf_detr_normalizes_and_rt_detr_does_not() -> None:
    """The one that would have been silently wrong."""

    assert _load(RFDETR)["model"]["do_normalize"] is True
    assert _load(RTDETR)["model"]["do_normalize"] is False


def test_the_two_configs_disagree_on_every_key_the_checkpoints_disagree_on() -> None:
    rt, rf = _load(RTDETR)["model"], _load(RFDETR)["model"]

    for key in ("checkpoint", "image_size", "do_normalize", "do_pad"):
        assert rt[key] != rf[key], f"{key} is the same in both configs; one of them is wrong"


def test_the_rf_detr_input_size_matches_its_processor_preset() -> None:
    """384, read off the checkpoint. Training at 640 would resize every image."""

    assert _load(RFDETR)["model"]["image_size"] == 384


def test_imagenet_statistics_are_present_and_are_the_standard_ones() -> None:
    model = _load(RFDETR)["model"]

    assert model["image_mean"] == [0.485, 0.456, 0.406]
    assert model["image_std"] == [0.229, 0.224, 0.225]


def test_the_head_is_replaced_rather_than_reused() -> None:
    """91 pretrained classes to 3 needs the flag, or from_pretrained raises."""

    model = _load(RFDETR)["model"]

    assert model["num_labels"] == 3
    assert model["ignore_mismatched_sizes"] is True


def test_the_step_budget_matches_what_the_main_run_actually_did() -> None:
    """TRAIN-07: an architecture comparison at a different budget compares budgets.

    Sourced from results/detection_metrics.csv, not from training.yaml - that
    file specifies EPOCHS, and the step count is what came out. Comparing
    against the configured epochs would compare an intention to a measurement.
    """

    import csv

    metrics = PROJECT_ROOT / "results" / "detection_metrics.csv"
    with metrics.open(encoding="utf-8", newline="") as handle:
        actual = {
            float(row["value"])
            for row in csv.DictReader(handle)
            if row["metric"] == "total_steps" and row["value"]
        }

    assert len(actual) == 1, f"the four arms did not share a step budget: {actual}"
    assert _load(RFDETR)["run"]["total_steps"] == int(actual.pop())


def test_the_four_arm_scope_and_execution_order_are_explicit() -> None:
    """A missing or reordered arm changes the approved replication experiment."""

    assert _load(RFDETR)["arms"] == [
        "real_only",
        "filtered_syn",
        "standard_aug",
        "unfiltered_syn",
    ]


def test_every_optimizer_value_is_labelled_a_guess() -> None:
    """They were inherited from a CNN recipe and never validated on a ViT.

    Asserted on the file text, because the tag lives in a comment: a `source:`
    marker that quietly became `verified` without anyone measuring anything is
    exactly the kind of claim this project has been bitten by.
    """

    text = RFDETR.read_text(encoding="utf-8")
    block = text[text.index("optimizer:") : text.index("# ==========", text.index("optimizer:"))]

    settings = [
        line for line in block.splitlines()
        if line.strip() and not line.strip().startswith("#") and ":" in line
        and not line.startswith(("optimizer:", "schedule:"))
    ]
    assert settings, "no optimizer settings found; the parser is looking in the wrong place"
    for line in settings:
        assert "source: guess" in line, f"unvalidated value not marked as a guess: {line.strip()}"


def test_the_config_says_the_training_time_is_unmeasured() -> None:
    """The last extrapolation on this project was wrong by 3x and cost a night."""

    text = RFDETR.read_text(encoding="utf-8").upper()

    assert "UNMEASURED" in text
