"""The generated notebook must be valid, runnable-shaped and token-free.

A notebook is the one artifact nobody runs until it costs real money, so the
checks that would normally be "it obviously works" have to be assertions.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_train_rtdetrv2.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.is_file():
        pytest.skip("notebook not generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells(notebook: dict) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_notebook_is_valid_nbformat(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], list)


def test_every_code_cell_parses(notebook: dict) -> None:
    """A syntax error would only surface after Drive is mounted on Colab."""

    for index, source in enumerate(code_cells(notebook)):
        # %pip is notebook magic, not Python; strip it before parsing.
        cleaned = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("%")
        )
        try:
            ast.parse(cleaned)
        except SyntaxError as error:
            raise AssertionError(f"code cell {index} does not parse: {error}") from error


# spec: TRAIN-12
def test_no_plaintext_token_anywhere(notebook: dict) -> None:
    """This notebook needs no token at all, so any token-shaped string is a bug."""

    text = json.dumps(notebook)
    patterns = (r"hf_[A-Za-z0-9]{20,}", r"gh[pousr]_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{20,}")
    for pattern in patterns:
        assert not re.search(pattern, text), f"token-shaped string matching {pattern}"
    # It must not even reach for one; a Secrets read implies a token exists.
    assert "userdata.get" not in text
    assert "HF_TOKEN" not in text


# spec: TRAIN-08
def test_data_is_unpacked_to_local_disk_not_read_from_drive(notebook: dict) -> None:
    """Training straight off mounted Drive leaves the GPU waiting on I/O."""

    joined = "\n".join(code_cells(notebook))
    assert "unpack_archive" in joined
    assert "/content/data" in joined


# spec: TRAIN-11
def test_each_arm_gets_a_unique_output_directory(notebook: dict) -> None:
    """runs/<arm>/seed_<n>/, so two arms can never write over each other."""

    joined = "\n".join(code_cells(notebook))
    assert re.search(r"RUNS\s*/\s*arm\s*/\s*f[\"']seed_\{SEED\}[\"']", joined)


# spec: TRAIN-10
def test_resume_is_wired_and_checkpoints_come_back_from_drive(notebook: dict) -> None:
    joined = "\n".join(code_cells(notebook))
    assert "resume=True" in joined
    assert "copytree(drive_dir, output_dir)" in joined


# spec: TRAIN-09
def test_checkpoints_sync_after_each_arm_not_only_at_the_end(notebook: dict) -> None:
    """A disconnect in hour four must not cost the three finished arms."""

    joined = "\n".join(code_cells(notebook))
    assert "synced to Drive" in joined


def test_embedded_modules_match_the_tested_sources(notebook: dict) -> None:
    """The whole point of generating the notebook: no second copy of the logic."""

    joined = "\n".join(code_cells(notebook))
    for name in ("arms", "data", "trainer", "metrics", "run"):
        source = (ROOT / "src" / "training" / f"{name}.py").read_text(encoding="utf-8")
        assert source in joined, f"{name}.py source is not embedded verbatim"


def test_gpu_assertion_comes_before_training(notebook: dict) -> None:
    cells = code_cells(notebook)
    gpu_cell = next(i for i, s in enumerate(cells) if "torch.cuda.is_available()" in s)
    train_cell = next(i for i, s in enumerate(cells) if "run_arm(" in s)
    assert gpu_cell < train_cell
