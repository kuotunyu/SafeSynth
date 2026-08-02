"""Config inheritance: the mechanism that decides what a training run actually does.

Written after `configs/training_rfdetr.yaml` - a deliberate differences-only
file - was loaded with plain yaml.safe_load and killed run_arm with
`KeyError: 'per_device_eval_batch_size'` eighteen lines in.

The dangerous failure here is not the crash. It is a merge that silently keeps
the BASE value where the child meant to override it: RF-DETR needs
`do_normalize: true` and RT-DETR declares `false`, and inheriting the base
there would not raise anything. It would train a model on wrongly scaled pixels
and report a mediocre number that looks like an architecture result.
"""

from __future__ import annotations

import pytest
import yaml

from src.data.paths import PROJECT_ROOT
from src.training.config import (
    TrainingConfigError,
    deep_merge,
    load_training_config,
    missing_run_keys,
)


def _write(tmp_path, name: str, payload: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8", newline="\n")
    return name


# --------------------------------------------------------------------------
# deep_merge
# --------------------------------------------------------------------------


def test_a_child_key_wins_over_the_base() -> None:
    assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_a_child_can_flip_a_boolean_off_to_on() -> None:
    """do_normalize is exactly this case, and inheriting it would not raise."""

    merged = deep_merge({"model": {"do_normalize": False}}, {"model": {"do_normalize": True}})

    assert merged["model"]["do_normalize"] is True


def test_nested_sections_merge_rather_than_replace() -> None:
    """A shallow merge would delete the sixteen run keys the child omits."""

    base = {"run": {"batch": 16, "eval_batch": 8, "precision": "bf16"}}
    child = {"run": {"batch": 32}}

    merged = deep_merge(base, child)

    assert merged["run"] == {"batch": 32, "eval_batch": 8, "precision": "bf16"}


def test_merging_is_recursive_beyond_one_level() -> None:
    base = {"a": {"b": {"c": 1, "d": 2}}}

    assert deep_merge(base, {"a": {"b": {"c": 9}}}) == {"a": {"b": {"c": 9, "d": 2}}}


def test_lists_are_replaced_not_concatenated() -> None:
    """`arms` exists so a child can NARROW the run. Appending would forbid that."""

    merged = deep_merge({"arms": ["w", "x", "y", "z"]}, {"arms": ["w", "z"]})

    assert merged["arms"] == ["w", "z"]


def test_the_base_mapping_is_not_mutated() -> None:
    base = {"run": {"batch": 16}}
    deep_merge(base, {"run": {"batch": 32}})

    assert base == {"run": {"batch": 16}}, "the caller's config was edited in place"


def test_a_child_may_introduce_a_key_the_base_never_had() -> None:
    assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


# --------------------------------------------------------------------------
# load_training_config
# --------------------------------------------------------------------------


def test_a_config_without_extends_is_returned_as_written(tmp_path) -> None:
    name = _write(tmp_path, "solo.yaml", {"run": {"batch": 4}})

    assert load_training_config(name, project_root=tmp_path) == {"run": {"batch": 4}}


def test_extends_is_resolved_and_removed_from_the_result(tmp_path) -> None:
    _write(tmp_path, "base.yaml", {"run": {"batch": 16, "eval_batch": 8}})
    name = _write(tmp_path, "child.yaml", {"extends": "base.yaml", "run": {"batch": 32}})

    resolved = load_training_config(name, project_root=tmp_path)

    assert resolved == {"run": {"batch": 32, "eval_batch": 8}}
    assert "extends" not in resolved, "the marker must not survive into the config"


def test_inheritance_chains_more_than_one_level(tmp_path) -> None:
    _write(tmp_path, "a.yaml", {"x": 1, "y": 1, "z": 1})
    _write(tmp_path, "b.yaml", {"extends": "a.yaml", "y": 2})
    name = _write(tmp_path, "c.yaml", {"extends": "b.yaml", "z": 3})

    assert load_training_config(name, project_root=tmp_path) == {"x": 1, "y": 2, "z": 3}


def test_a_cycle_raises_instead_of_recursing_forever(tmp_path) -> None:
    _write(tmp_path, "one.yaml", {"extends": "two.yaml"})
    _write(tmp_path, "two.yaml", {"extends": "one.yaml"})

    with pytest.raises(TrainingConfigError, match="cycle"):
        load_training_config("one.yaml", project_root=tmp_path)


def test_a_config_that_extends_itself_is_a_cycle_too(tmp_path) -> None:
    _write(tmp_path, "self.yaml", {"extends": "self.yaml"})

    with pytest.raises(TrainingConfigError, match="cycle"):
        load_training_config("self.yaml", project_root=tmp_path)


def test_a_missing_base_names_both_files(tmp_path) -> None:
    """"No such file: absent.yaml" alone leaves the reader hunting for who asked."""

    name = _write(tmp_path, "child.yaml", {"extends": "absent.yaml"})

    with pytest.raises(TrainingConfigError, match="child.yaml"):
        load_training_config(name, project_root=tmp_path)


def test_a_missing_config_raises_a_typed_error(tmp_path) -> None:
    with pytest.raises(TrainingConfigError, match="no training config"):
        load_training_config("nope.yaml", project_root=tmp_path)


def test_a_yaml_file_that_is_not_a_mapping_is_refused(tmp_path) -> None:
    (tmp_path / "list.yaml").write_text("- a\n- b\n", encoding="utf-8", newline="\n")

    with pytest.raises(TrainingConfigError, match="must hold a mapping"):
        load_training_config("list.yaml", project_root=tmp_path)


# --------------------------------------------------------------------------
# missing_run_keys
# --------------------------------------------------------------------------


def test_missing_run_keys_names_what_is_absent() -> None:
    required = {"run": {"a": 1, "b": 2, "c": 3}}

    assert missing_run_keys({"run": {"a": 9}}, required) == ["b", "c"]
    assert missing_run_keys({"run": {"a": 1, "b": 2, "c": 3}}, required) == []


def test_a_config_with_no_run_block_is_missing_all_of_them() -> None:
    assert missing_run_keys({}, {"run": {"a": 1, "b": 2}}) == ["a", "b"]


# --------------------------------------------------------------------------
# the real project configs
# --------------------------------------------------------------------------


def test_the_rfdetr_config_resolves_to_a_runnable_whole() -> None:
    """The regression this module exists for. Every run key must be present."""

    base = load_training_config("configs/training.yaml")
    resolved = load_training_config("configs/training_rfdetr.yaml")

    assert missing_run_keys(resolved, base) == []


def test_the_rfdetr_config_overrides_the_three_settings_that_differ() -> None:
    """Verified against both checkpoints on 2026-08-02; see the config header."""

    base = load_training_config("configs/training.yaml")
    resolved = load_training_config("configs/training_rfdetr.yaml")

    assert base["model"]["do_normalize"] is False
    assert resolved["model"]["do_normalize"] is True, "RF-DETR needs ImageNet stats"
    assert resolved["model"]["image_size"] == 384
    assert resolved["model"]["checkpoint"] == "Roboflow/rf-detr-nano"


def test_the_rfdetr_config_resolves_to_the_approved_four_arms() -> None:
    resolved = load_training_config("configs/training_rfdetr.yaml")

    assert resolved["arms"] == [
        "real_only",
        "filtered_syn",
        "standard_aug",
        "unfiltered_syn",
    ]


def test_the_rfdetr_config_keeps_the_batch_size_that_makes_steps_comparable() -> None:
    """TRAIN-07 matches optimizer steps; a different batch changes what a step is."""

    base = load_training_config("configs/training.yaml")
    resolved = load_training_config("configs/training_rfdetr.yaml")

    assert (
        resolved["run"]["per_device_train_batch_size"]
        == base["run"]["per_device_train_batch_size"]
    )


def test_the_rfdetr_config_is_genuinely_a_delta_not_a_copy() -> None:
    """If it ever becomes a full copy, the two files start drifting silently."""

    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "training_rfdetr.yaml").read_text(encoding="utf-8")
    )

    assert raw["extends"] == "configs/training.yaml"
    assert len(raw.get("run", {})) < 6, "this file should state differences, not everything"
