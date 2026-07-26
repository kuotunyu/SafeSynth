from pathlib import Path

import pytest

from src.data.paths import load_project_paths, pin_dataset_version


def write_config(path: Path, pinned: str = "null") -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                'data_root: "D:/bulk"',
                "paths:",
                '  raw: "${data_root}/raw"',
                '  hardhat_raw: "${data_root}/raw/dataset"',
                '  interim: "${data_root}/interim"',
                '  splits: "splits"',
                '  reports: "reports"',
                '  figures: "reports/figures"',
                'dotenv: "../.env"',
                "dataset:",
                '  kaggle_handle: "owner/data"',
                f"  pinned_version: {pinned}  # retained comment",
                '  classes: ["helmet", "head", "person"]',
                "seed: 42",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def test_load_paths_expands_declared_variables(tmp_path: Path) -> None:
    config_path = tmp_path / "project" / "configs" / "paths.yaml"
    write_config(config_path)

    paths = load_project_paths(config_path)

    assert paths.data_root == Path("D:/bulk")
    assert paths.hardhat_raw == Path("D:/bulk/raw/dataset")
    assert paths.splits == tmp_path / "project" / "splits"
    assert paths.pinned_version is None


def test_pin_dataset_version_preserves_comments(tmp_path: Path) -> None:
    config_path = tmp_path / "project" / "configs" / "paths.yaml"
    write_config(config_path)

    pin_dataset_version(config_path, 7)

    text = config_path.read_text(encoding="utf-8")
    assert "pinned_version: 7  # retained comment" in text
    assert load_project_paths(config_path).pinned_version == 7


def test_pin_dataset_version_refuses_to_change_existing_pin(tmp_path: Path) -> None:
    config_path = tmp_path / "project" / "configs" / "paths.yaml"
    write_config(config_path, pinned="7")

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        pin_dataset_version(config_path, 8)
