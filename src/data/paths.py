"""Project path loading with explicit UTF-8 handling and no hard-coded roots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"
_VARIABLE_PATTERN = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths and immutable dataset settings."""

    project_root: Path
    config_path: Path
    grouping_config_path: Path
    data_root: Path
    raw: Path
    hardhat_raw: Path
    interim: Path
    cutouts: Path
    masks_pass1: Path
    synthetic: Path
    cache: Path
    runs: Path
    splits: Path
    reports: Path
    figures: Path
    dotenv: Path
    kaggle_handle: str
    pinned_version: int | None
    classes: tuple[str, ...]
    seed: int


def _expand_variables(value: str, variables: dict[str, str]) -> str:
    """Expand only variables declared by the path config."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise ValueError(f"Unknown path variable in configs/paths.yaml: {name}")
        return variables[name]

    return _VARIABLE_PATTERN.sub(replace, value)


def _resolve_path(value: str, project_root: Path, variables: dict[str, str]) -> Path:
    expanded = _expand_variables(value, variables)
    path = Path(expanded)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_raw_config(config_path: Path = PATHS_CONFIG) -> dict[str, Any]:
    """Load the path config using UTF-8 explicitly (ENV-10)."""

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {config_path}")
    return config


def load_project_paths(config_path: Path = PATHS_CONFIG) -> ProjectPaths:
    """Resolve every data-preparation path from configs/paths.yaml."""

    config_path = config_path.resolve()
    project_root = config_path.parents[1]
    config = load_raw_config(config_path)

    data_root_value = str(config["data_root"])
    data_root = _resolve_path(data_root_value, project_root, {})
    variables = {"data_root": data_root.as_posix()}
    configured_paths = config["paths"]
    dataset = config["dataset"]

    def data_path(name: str) -> Path:
        value = configured_paths.get(name, f"${{data_root}}/{name}")
        return _resolve_path(str(value), project_root, variables)

    return ProjectPaths(
        project_root=project_root,
        config_path=config_path,
        grouping_config_path=project_root / "configs" / "grouping.yaml",
        data_root=data_root,
        raw=_resolve_path(str(configured_paths["raw"]), project_root, variables),
        hardhat_raw=_resolve_path(str(configured_paths["hardhat_raw"]), project_root, variables),
        interim=_resolve_path(str(configured_paths["interim"]), project_root, variables),
        cutouts=data_path("cutouts"),
        masks_pass1=data_path("masks_pass1"),
        synthetic=data_path("synthetic"),
        cache=data_path("cache"),
        runs=data_path("runs"),
        splits=_resolve_path(str(configured_paths["splits"]), project_root, variables),
        reports=_resolve_path(str(configured_paths["reports"]), project_root, variables),
        figures=_resolve_path(str(configured_paths["figures"]), project_root, variables),
        dotenv=_resolve_path(str(config["dotenv"]), project_root, variables),
        kaggle_handle=str(dataset["kaggle_handle"]),
        pinned_version=(
            int(dataset["pinned_version"]) if dataset.get("pinned_version") is not None else None
        ),
        classes=tuple(str(item) for item in dataset["classes"]),
        seed=int(config["seed"]),
    )


def pin_dataset_version(config_path: Path, version: int) -> None:
    """Pin a discovered version while preserving the hand-written YAML comments."""

    if version <= 0:
        raise ValueError(f"Dataset version must be positive, got {version}")

    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"^(\s*pinned_version:\s*)(null|\d+)(\s*(?:#.*)?)$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not locate dataset.pinned_version in {config_path}")

    current = match.group(2)
    if current != "null" and int(current) != version:
        raise RuntimeError(
            f"Refusing to replace pinned dataset version {current} with upstream version {version}"
        )
    if current == str(version):
        return

    replacement = f"{match.group(1)}{version}{match.group(3)}"
    updated = text[: match.start()] + replacement + text[match.end() :]
    config_path.write_text(updated, encoding="utf-8", newline="\n")
