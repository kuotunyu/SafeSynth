"""Trainer subclass and TrainingArguments assembly.

Two things live here that Trainer does not give you:

1. Per-parameter-group learning rates. The upstream RT-DETRv2 recipe puts the
   backbone on 0.1x LR. That is strictly better than freezing it on a
   domain-shifted dataset like this one, and Trainer has no per-group LR, so
   create_optimizer has to be overridden.
2. The TrainingArguments that are not optional. Four of them are defaults that
   are wrong for detection, and getting them wrong does not raise.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments


def build_parameter_groups(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    backbone_lr_multiplier: float,
    weight_decay: float,
    no_decay_on_norm_and_bias: bool = True,
) -> list[dict[str, Any]]:
    """Three groups: backbone at 0.1x, decayable at 1x, norm/bias at 1x with no decay."""

    backbone, decay, no_decay = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "backbone" in name:
            backbone.append(parameter)
        elif no_decay_on_norm_and_bias and (
            name.endswith(".bias") or parameter.ndim == 1
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups = [
        {
            "params": backbone,
            "lr": learning_rate * backbone_lr_multiplier,
            "weight_decay": weight_decay,
            "name": "backbone",
        },
        {
            "params": decay,
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "name": "decay",
        },
        {
            "params": no_decay,
            "lr": learning_rate,
            "weight_decay": 0.0,
            "name": "no_decay",
        },
    ]
    return [group for group in groups if group["params"]]


class RTDetrTrainer(Trainer):
    """Trainer with the upstream three-group optimizer."""

    def __init__(self, *args, optimizer_config: Mapping[str, Any] | None = None, **kwargs):
        self._optimizer_config = dict(optimizer_config or {})
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        config = self._optimizer_config
        groups = build_parameter_groups(
            self.model,
            learning_rate=float(config.get("learning_rate", self.args.learning_rate)),
            backbone_lr_multiplier=float(config.get("backbone_lr_multiplier", 0.1)),
            weight_decay=float(config.get("weight_decay", self.args.weight_decay)),
            no_decay_on_norm_and_bias=bool(
                config.get("no_decay_on_norm_and_bias", True)
            ),
        )
        betas = tuple(float(b) for b in config.get("betas", (0.9, 0.999)))
        self.optimizer = torch.optim.AdamW(groups, betas=betas)
        return self.optimizer


# The four settings whose Trainer defaults are wrong for detection, and whose
# wrongness is silent rather than loud.
MANDATORY_TRAINING_ARGUMENTS = {
    # Trainer would otherwise concatenate ragged per-image label lists.
    "eval_do_concat_batches": False,
    # Trainer would strip the columns the dataset transform needs.
    "remove_unused_columns": False,
    # Upstream clip_max_norm is 0.1, not Trainer's default 1.0. The usual NaN
    # loss reports trace back to too-high LR or too-loose clipping.
    "max_grad_norm": 0.1,
    # Surface raw NaN/Inf losses to Trainer callbacks instead of replacing them
    # with the previous finite average, which would defeat unattended health gates.
    "logging_nan_inf_filter": False,
}


def build_training_arguments(
    *,
    output_dir: str,
    config: Mapping[str, Any],
    total_steps: int,
    seed: int,
    use_bf16: bool,
    dataloader_num_workers: int,
) -> TrainingArguments:
    run = config["run"]
    schedule = config["schedule"]
    optimizer = config["optimizer"]

    arguments = TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        max_steps=int(total_steps),
        per_device_train_batch_size=int(run["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(run["per_device_eval_batch_size"]),
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        lr_scheduler_type=str(schedule["lr_scheduler_type"]),
        warmup_steps=int(schedule["warmup_steps"]),
        max_grad_norm=float(schedule["max_grad_norm"]),
        bf16=bool(use_bf16),
        fp16=not bool(use_bf16),
        dataloader_num_workers=int(dataloader_num_workers),
        eval_strategy=str(run["eval_strategy"]),
        save_strategy=str(run["save_strategy"]),
        save_total_limit=int(run["save_total_limit"]),
        load_best_model_at_end=bool(run["load_best_model_at_end"]),
        metric_for_best_model=str(run["metric_for_best_model"]),
        greater_is_better=bool(run["greater_is_better"]),
        remove_unused_columns=MANDATORY_TRAINING_ARGUMENTS["remove_unused_columns"],
        eval_do_concat_batches=MANDATORY_TRAINING_ARGUMENTS["eval_do_concat_batches"],
        logging_steps=50,
        logging_nan_inf_filter=MANDATORY_TRAINING_ARGUMENTS["logging_nan_inf_filter"],
        report_to=[],
    )
    assert_mandatory_arguments(arguments)
    return arguments


def assert_mandatory_arguments(arguments: TrainingArguments) -> None:
    """Fail loudly rather than train for hours on a silently wrong setting."""

    wrong = {
        key: getattr(arguments, key)
        for key, expected in MANDATORY_TRAINING_ARGUMENTS.items()
        if getattr(arguments, key) != expected
    }
    if wrong:
        raise RuntimeError(
            f"TrainingArguments deviate from the mandatory detection settings: {wrong}. "
            f"Expected {MANDATORY_TRAINING_ARGUMENTS}."
        )


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _weight_evidence_complete(checkpoint: Path) -> bool:
    if _nonempty_file(checkpoint / "model.safetensors") or _nonempty_file(
        checkpoint / "pytorch_model.bin"
    ):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint / name
        if not _nonempty_file(index_path):
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        shard_names = tuple(weight_map.values()) if isinstance(weight_map, Mapping) else ()
        if (
            shard_names
            and all(isinstance(shard, str) for shard in shard_names)
            and all(
                _nonempty_file(checkpoint / shard) for shard in set(shard_names)
            )
        ):
            return True
    return False


def resumable_checkpoint_step(checkpoint: Path) -> int | None:
    checkpoint = Path(checkpoint)
    try:
        step = int(checkpoint.name.rsplit("-", 1)[-1])
    except ValueError:
        return None
    if step < 0 or not checkpoint.is_dir():
        return None
    state_path = checkpoint / "trainer_state.json"
    if not _nonempty_file(state_path):
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(state, Mapping) or state.get("global_step") != step:
        return None
    required = ("optimizer.pt", "scheduler.pt", "rng_state.pth")
    if not all(_nonempty_file(checkpoint / name) for name in required):
        return None
    return step if _weight_evidence_complete(checkpoint) else None


def find_resumable_checkpoint(output_dir) -> str | None:
    """Colab disconnects; resuming is not an optional feature (TRAIN-10)."""

    directory = Path(output_dir)
    if not directory.is_dir():
        return None
    checkpoints = [
        (step, path)
        for path in directory.iterdir()
        if (step := resumable_checkpoint_step(path)) is not None
    ]
    if not checkpoints:
        return None

    return str(max(checkpoints)[1])
