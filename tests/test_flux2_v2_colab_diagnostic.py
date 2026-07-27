from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "00_flux2_v2_diagnostic.ipynb"


def test_diagnostic_notebook_is_valid_and_has_no_embedded_secret() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "HF_TOKEN =" not in source
    assert "kaggle.json" not in source
    assert "run_artifact_gate" not in source
    assert "safetensors==0.8.0" in source
    assert "os.environ['HF_HUB_DISABLE_XET'] = '1'" in source
    assert "hf_constants.HF_HUB_DISABLE_XET = True" in source
    assert "hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = 120" in source
    assert "relative = Path(*source.name.split('\\\\'))" in source
    assert "final_h4_auc_computed': False" in source
    assert "source_split'] == 'train_only'" in source
    assert "if vram_gib >= 35:" in source
    assert "execution_mode = 'full_model_on_cuda'" in source
    assert "files.upload()" in source
    assert "files.download(archive)" in source


def test_every_python_code_cell_compiles() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")
