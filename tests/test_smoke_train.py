"""The local smoke must exercise the same inherited config as production."""

from __future__ import annotations

from pathlib import Path

from scripts import smoke_train


def test_smoke_accepts_an_inherited_training_config() -> None:
    """Ignoring --config would smoke RT-DETR while RF-DETR remains untested."""

    args = smoke_train.parse_args(
        ["--config", "configs/training_rfdetr.yaml", "--arm", "real_only"]
    )

    assert args.config == Path("configs/training_rfdetr.yaml")
    assert args.arm == "real_only"


def test_smoke_output_is_namespaced_by_config(tmp_path: Path) -> None:
    """Sharing one smoke directory can resume incompatible model weights."""

    path = smoke_train.smoke_output_dir(
        tmp_path,
        Path("configs/training_rfdetr.yaml"),
        "real_only",
        1337,
    )

    assert path == tmp_path / "smoke" / "training_rfdetr" / "real_only_seed_1337"
