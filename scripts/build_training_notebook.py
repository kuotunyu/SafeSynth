"""Generate notebooks/01_train_rtdetrv2.ipynb from the tested src/training modules.

The notebook is generated rather than hand-written because a hand-written
notebook is a second copy of the logic that no test ever runs. Here the modules
are the single source of truth, `uv run pytest` covers them, and this script
embeds their exact source. What executes on Colab is what the tests ran against.

The repo has no GitHub remote yet, so the notebook cannot clone itself; that is
why the source is embedded rather than imported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT

MODULES = ("arms", "data", "trainer", "metrics", "run", "ingest")
DRIVE_DIR = "/content/drive/MyDrive/sdg-portfolio/02-safesynth-ppe"
ARCHIVE = "safesynth_train_data.zip"
EXPECTED_REAL_IMAGES = 4256
EXPECTED_SYNTHETIC_IMAGES = 6152


def _markdown(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def _code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def _module_bootstrap() -> str:
    """Cell that writes the tested module sources into an importable package."""

    header = '''
import pathlib
import sys

# The modules import each other as `src.training.x`, so the package layout has
# to exist before any of them is written.
ROOT = pathlib.Path("/content/safesynth")
PKG = ROOT / "src" / "training"
PKG.mkdir(parents=True, exist_ok=True)
(ROOT / "src" / "__init__.py").write_text("", encoding="utf-8")
(PKG / "__init__.py").write_text("", encoding="utf-8")

MODULE_SOURCE = {}
'''
    body = []
    for name in MODULES:
        source = (PROJECT_ROOT / "src" / "training" / f"{name}.py").read_text(
            encoding="utf-8"
        )
        body.append(f"MODULE_SOURCE[{name!r}] = r'''{source}'''\n")
    footer = '''
for _name, _source in MODULE_SOURCE.items():
    (PKG / f"{_name}.py").write_text(_source, encoding="utf-8")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("training modules written:", sorted(MODULE_SOURCE))
'''
    return header + "\n" + "".join(body) + footer


CELL_GPU = '''
import torch

assert torch.cuda.is_available(), "沒有 GPU。執行階段 → 變更執行階段類型 → L4 GPU"
_name = torch.cuda.get_device_name(0)
_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"{_name}  {_total_gb:.0f} GB  bf16={torch.cuda.is_bf16_supported()}")
if not torch.cuda.is_bf16_supported():
    print(
        "WARNING: 這張卡不支援 bf16（大概是 T4），會退回 fp16。\\n"
        "         這個模型在 fp16 下比較容易出現 NaN loss，建議改用 L4。"
    )
'''

CELL_INSTALL = '''
%pip install -q "transformers>=5.14.1" albumentations pycocotools accelerate

import transformers

print("transformers", transformers.__version__)
'''

CELL_DATA = f'''
import pathlib
import shutil
import time

from google.colab import drive

drive.mount("/content/drive")

DRIVE_DIR = {DRIVE_DIR!r}
ARCHIVE = {ARCHIVE!r}

archive = pathlib.Path(DRIVE_DIR) / ARCHIVE
assert archive.is_file(), f"找不到 {{archive}}，確認 Drive 路徑與檔名"
print(f"{{archive.stat().st_size / 1e9:.2f}} GB")

DATA = pathlib.Path("/content/data")
if not (DATA / "real" / "coco_all.json").is_file():
    DATA.mkdir(parents=True, exist_ok=True)
    _start = time.time()
    shutil.unpack_archive(str(archive), str(DATA), "zip")
    print(f"unpacked in {{time.time() - _start:.0f}}s")
else:
    print("already unpacked")

n_real = len(list((DATA / "real" / "images").glob("*.png")))
n_syn = len(list((DATA / "synthetic" / "images").glob("*.png")))
print(f"real={{n_real}}  synthetic={{n_syn}}")
assert n_real == {EXPECTED_REAL_IMAGES}, "真實影像張數不符，資料包可能不完整"
assert n_syn == {EXPECTED_SYNTHETIC_IMAGES}, "合成影像張數不符，資料包可能不完整"
'''

CELL_ARMS = '''
import json
import pathlib

import yaml

from src.training.arms import build_all_arms, equal_step_budget

DATA = pathlib.Path("/content/data")
config = yaml.safe_load(
    (DATA / "configs" / "training.yaml").read_text(encoding="utf-8")
)

arms = build_all_arms(
    manifest_path=DATA / "splits" / "split_manifest.json",
    synthetic_annotations={
        "filtered": DATA / "synthetic" / "annotations_filtered_1x.json",
        "unfiltered": DATA / "synthetic" / "annotations_unfiltered_1x.json",
    },
)
print("不變式通過：TRAIN-04 相同真實影像 / TRAIN-06 合成等量 / TRAIN-23 無 Test")
for _arm, _comp in arms.items():
    _s = _comp.summary()
    print(
        f"  {_arm:<16} real={_s['n_real_train']} + syn={_s['n_synthetic']}"
        f" = {_s['n_train_total']}  aug={_s['augmentation_profile']}"
    )

BATCH = int(config["run"]["per_device_train_batch_size"])
plan = equal_step_budget(
    arms,
    reference_arm="real_only",
    reference_epochs=int(config["run"]["num_train_epochs_real_only"]),
    batch_size=BATCH,
)
print()
for _arm, _row in plan.items():
    print(
        f"  {_arm:<16} steps={_row['total_steps']}  epochs={_row['epochs']:.1f}"
        f"  每張真實圖看 {_row['real_image_exposures']:.1f} 次"
    )
'''

CELL_TRAIN = '''
import shutil
import time
import traceback

from src.training.run import RunPaths, run_arm

SEED = int(config["seeds"]["primary"])
RUNS = pathlib.Path("/content/runs")
DRIVE_RUNS = pathlib.Path(DRIVE_DIR) / "runs"
DRIVE_RUNS.mkdir(parents=True, exist_ok=True)

SUBSET = {"unfiltered_syn": "unfiltered", "filtered_syn": "filtered"}
records = []

for arm in ("real_only", "standard_aug", "unfiltered_syn", "filtered_syn"):
    output_dir = RUNS / arm / f"seed_{SEED}"
    drive_dir = DRIVE_RUNS / arm / f"seed_{SEED}"

    # Bring a previous session's checkpoints back before deciding to resume.
    if drive_dir.is_dir() and not output_dir.is_dir():
        shutil.copytree(drive_dir, output_dir)
        print(f"[{arm}] restored checkpoints from Drive")

    subset = SUBSET.get(arm)
    run_paths = RunPaths(
        real_images=DATA / "real" / "images",
        real_coco=DATA / "real" / "coco_all.json",
        synthetic_images=DATA / "synthetic" / "images",
        synthetic_coco=(
            DATA / "synthetic" / f"annotations_{subset}_1x.json" if subset else None
        ),
        output_dir=output_dir,
    )

    print(f"\\n===== {arm} =====")
    started = time.time()
    try:
        record = run_arm(
            arms[arm],
            run_paths,
            config=config,
            total_steps=plan[arm]["total_steps"],
            seed=SEED,
            resume=True,
        )
    except Exception:  # noqa: BLE001 - one failed arm must not stop the other three
        traceback.print_exc()
        print(f"[{arm}] FAILED — 其餘組別繼續，稍後單獨重跑這一組即可")
        continue

    record["wall_clock_hours"] = round((time.time() - started) / 3600, 3)
    records.append(record)
    print(f"[{arm}] done in {record['wall_clock_hours']:.2f} h")

    # Sync after each arm, not at the end: a disconnect in hour four must not
    # cost the three arms that already finished.
    drive_dir.parent.mkdir(parents=True, exist_ok=True)
    if drive_dir.is_dir():
        shutil.rmtree(drive_dir)
    shutil.copytree(output_dir, drive_dir)
    print(f"[{arm}] synced to Drive")

print("\\n完成的組別:", [r["arm"] for r in records])
'''

CELL_RESULTS = '''
summary = {
    "records": records,
    "plan": plan,
    "arms": {a: c.summary() for a, c in arms.items()},
    "gpu": torch.cuda.get_device_name(0),
    "bf16": torch.cuda.is_bf16_supported(),
    "transformers": transformers.__version__,
}

out = pathlib.Path("/content/results_colab")
out.mkdir(exist_ok=True)
(out / "training_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
)
from src.training.ingest import package_arm_outputs

packaged, missing = package_arm_outputs(RUNS, out)
print(f"packaged {len(packaged)} files")
if missing:
    print("WARNING - not packaged:", missing)

archive_path = shutil.make_archive("/content/results_colab", "zip", str(out))
shutil.copy2(archive_path, pathlib.Path(DRIVE_DIR) / "results_colab.zip")
print("寫到 Drive:", pathlib.Path(DRIVE_DIR) / "results_colab.zip")
print(json.dumps(summary["records"], indent=2, default=str))
'''


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            "# SafeSynth — RT-DETRv2 四組對照訓練\n"
            "\n"
            "**跑之前確認兩件事：**\n"
            "\n"
            "1. Runtime 是 **L4 GPU**（執行階段 → 變更執行階段類型）\n"
            "2. Drive 有 `sdg-portfolio/02-safesynth-ppe/safesynth_train_data.zip`\n"
            "\n"
            "然後「執行階段 → 全部執行」，四組依序跑完，預估 **4–5 小時**。\n"
            "斷線就重新全部執行一次，會自動從 checkpoint 接續。\n"
            "\n"
            "**不需要任何 token。** 模型是公開的，這本 notebook 不讀 Secrets、不上傳任何東西。"
        ),
        _markdown("## 1. 確認 GPU"),
        _code(CELL_GPU),
        _markdown("## 2. 安裝相依套件"),
        _code(CELL_INSTALL),
        _markdown(
            "## 3. 掛載 Drive，解壓到本機磁碟\n"
            "\n"
            "**不直接從 Drive 讀圖訓練**（TRAIN-08）——Drive 的隨機讀取延遲會讓 GPU "
            "大部分時間在等 I/O。"
        ),
        _code(CELL_DATA),
        _markdown(
            "## 4. 寫入訓練模組\n"
            "\n"
            "這是本機 `uv run pytest` 測過的同一份程式碼，內嵌於此以免 Colab "
            "需要 clone 一個還不存在的 GitHub repo。"
        ),
        _code(_module_bootstrap()),
        _markdown("## 5. 組成四組並檢查不變式"),
        _code(CELL_ARMS),
        _markdown(
            "## 6. 依序訓練四組\n"
            "\n"
            "每組一個獨立輸出目錄（TRAIN-11），每組跑完立刻同步回 Drive（TRAIN-09）。\n"
            "**斷線後重新執行這格會自動接續**（TRAIN-10）。"
        ),
        _code(CELL_TRAIN),
        _markdown(
            "## 7. 打包結果回 Drive\n"
            "\n"
            "跑完把 Drive 上的 `results_colab.zip` 下載回本機，放進 repo 的 `results/colab/`。"
        ),
        _code(CELL_RESULTS),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "notebooks" / "01_train_rtdetrv2.ipynb")
    )
    args = parser.parse_args()

    notebook = build_notebook()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"wrote {output}  ({output.stat().st_size / 1024:.0f} KB, "
        f"{len(notebook['cells'])} cells)"
    )


if __name__ == "__main__":
    main()
