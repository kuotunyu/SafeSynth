"""Training configs that state DIFFERENCES, and the loader that resolves them.

`configs/training_rfdetr.yaml` is deliberately not a copy of `training.yaml`
with the checkpoint swapped - its own header says so. It records the handful of
values that genuinely differ for RF-DETR and stays silent about the rest.

That is the right way to write it and the wrong way to load it. Loading it
alone gives a `run:` block holding two keys, and `run_arm` dies on the
eighteenth line with `KeyError: 'per_device_eval_batch_size'` - which is what
happened, and which reads like a corrupt file rather than a config that was
never meant to stand alone.

So the relationship becomes data. A config may declare:

    extends: "configs/training.yaml"

and this module merges it over that base. Two properties matter:

* THE MERGE IS DEEP, per section. `run:` in the child adds to `run:` in the
  base rather than replacing it wholesale - a shallow merge would delete the
  sixteen keys the child does not mention and reintroduce the same KeyError
  with more steps.
* A CHILD KEY ALWAYS WINS. `do_normalize: true` in the RF-DETR config must
  override `false` in the base. If inheritance could not flip that value, the
  whole file would be decoration: it is the one setting whose silent
  inheritance would wreck the model rather than crash it.

Cycles raise rather than recurse forever, and a base that does not exist raises
naming both files, because "no such file" pointing at a path nobody wrote is
the least useful error this could produce.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT

EXTENDS_KEY = "extends"


class TrainingConfigError(ValueError):
    """Raised when a training config cannot be resolved into a usable mapping."""


# spec: TRAIN-01
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """`override` wins, but nested mappings merge instead of replacing.

    Lists are replaced, not concatenated. A child that says `arms: ["a"]` means
    exactly that list; appending to the base's would make it impossible to
    narrow a run, which is the only reason a child would state `arms` at all.
    """

    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


# spec: TRAIN-01
def load_training_config(
    path: Path | str, *, project_root: Path | None = None, _seen: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Read a training config, resolving `extends` into a complete mapping."""

    root = PROJECT_ROOT if project_root is None else Path(project_root)
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()

    key = config_path.as_posix()
    if key in _seen:
        chain = " -> ".join([*_seen, key])
        raise TrainingConfigError(f"`{EXTENDS_KEY}` forms a cycle: {chain}")
    if not config_path.is_file():
        origin = f" (named by {_seen[-1]})" if _seen else ""
        raise TrainingConfigError(f"no training config at {config_path}{origin}")

    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise TrainingConfigError(
            f"{config_path} must hold a mapping, got {type(loaded).__name__}"
        )

    parent = loaded.pop(EXTENDS_KEY, None)
    if parent is None:
        return loaded
    base = load_training_config(parent, project_root=root, _seen=(*_seen, key))
    return deep_merge(base, loaded)


# spec: TRAIN-01
def missing_run_keys(config: Mapping[str, Any], required: Mapping[str, Any]) -> list[str]:
    """Which `run:` keys a resolved config still lacks, compared with a reference.

    Used as a startup assertion rather than as a fixer: a training run that
    silently falls back to a default for `eval_do_concat_batches` produces
    garbage metrics, and discovering that after the run is far worse than
    refusing to start.
    """

    have = config.get("run", {})
    want = required.get("run", {})
    return sorted(set(want) - set(have))
