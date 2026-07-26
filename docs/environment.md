# Environment — Windows 11 native, no WSL

> 對應里程碑 M1。決策依據：[ADR-002](decisions.md#adr-002)。踩到的坑記在 [troubleshooting.md](troubleshooting.md)。
> 環境自檢與修復流程：呼叫 `/safesynth-env` skill。

---

## 1. 本機實測基準（2026-07-27）

| 項目 | 實測值 | 備註 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB | |
| 驅動 | **591.86** | cu130 要求 ≥580.88 → 過關 |
| 系統 Python | anaconda 3.10.9 | **不使用**；專案走 uv 管理的 3.12 |
| uv | 0.11.18 | |
| git / gh / git-lfs | 2.41.0 / 2.96.0 / 3.3.0 | |
| `huggingface-cli` | anaconda 下拋 traceback；`hf` 指令不存在 | 專案 venv 內另裝 |
| 磁碟 | C: 剩 185.8 GB／**D: 剩 1728.1 GB** | 大檔一律放 D: |
| `LongPathsEnabled` | **0** | 260 字元上限是活的，見 ENV-09 |
| `~\.kaggle\kaggle.json` | 不存在 | 憑證只在上層 `..\.env` |
| `KAGGLE_*` 環境變數 | 皆未設定 | 同上 |

---

## 2. 需求

### ENV-01 — Python 版本
**需求**：專案使用 Python **3.12**（`requires-python = ">=3.12,<3.13"`）。
**理由**：`kaggle` 2.x 要求 ≥3.11；`kagglehub` 要求 ≥3.10；`pycocotools` 2.0.11 的
`cp312-abi3-win_amd64` wheel 一顆涵蓋 3.12/3.13/3.14，在原生 Windows 上免 MSVC 即可安裝。
**驗證**：`uv run python --version` → `Python 3.12.x`

### ENV-02 — torch 必須來自 cu130 index
**需求**：`torch` 與 `torchvision` 只能從 `https://download.pytorch.org/whl/cu130` 安裝，
於 `pyproject.toml` 以 `[[tool.uv.index]]` ＋ `explicit = true` ＋ `[tool.uv.sources]` 指定。
**理由**：PyPI 的 `win_amd64` wheel 是 **CPU-only**（約 122 MB vs Linux CUDA wheel 約 527 MB）。
裸的 `pip install torch` 會安裝成功、不報錯，然後 `torch.cuda.is_available()` 回 `False`。
**這是本專案最容易安靜出錯的一步。**
**已實測存在**：`torch-2.13.0+cu130-cp312-cp312-win_amd64.whl`、
`torchvision-0.28.0+cu130-cp312-cp312-win_amd64.whl`。
**cu128 不可用**：該 index 最高只到 torch 2.11.0。
**驗證**：見 ENV-10 的驗證表第 2 列。

### ENV-03 — transformers 硬性下限
**需求**：`transformers>=5.14.1`。

**兩個獨立理由**：
1. **4.56.x 的 SAM2 `_embed_boxes` 少了 padding point**，box prompt 的 mask 品質
   從 IoU ~0.98 掉到 ~0.94，**不報錯、只是每個 cutout 都差一點**。
   修正在 4.57.0，但該版在 PyPI 被 yank，所以 4.x 的下限是 4.57.1（[ADR-001](decisions.md#adr-001)）
2. **v5 改了 image processor 的命名**：`RTDetrImageProcessor` 現在**就是**快版，
   慢版改叫 `RTDetrImageProcessorPil`，`RTDetrImageProcessorFast` 已不存在。
   把下限訂在 v5 讓 Phase 1 與 Phase 2 統一在同一個 API 世代（[ADR-006](decisions.md#adr-006)）

**v5 的連帶影響**：`from_pretrained` 的預設 dtype 從 `float32` 改為 `"auto"`，
可能造成與 v4 的靜默數值差異——**一律明確傳 `dtype=`**（`torch_dtype=` 已棄用）。

**驗證**：見 ENV-10 的驗證表第 3 列。

### ENV-04 — 不使用 `kaggle` 套件
**需求**：資料下載一律用 `kagglehub`，**程式碼中不得出現 `import kaggle`**。
**理由**：`kaggle/__init__.py` 在 import 時會自動認證並執行
`os.environ.pop("KAGGLE_API_TOKEN")`，把環境變數從行程中移除
（kaggle-cli issue #882）。任何在其後才讀該變數的程式碼都會拿到 KeyError。
**附帶好處**：`kagglehub.dataset_download("owner/name/versions/N")` 可以**釘住資料集版本**，
這對「凍結 split」是必要條件——上游若重新上傳，凍結的 manifest 就失去意義。
**驗證**：`grep -rn "import kaggle" src/ scripts/` → 零命中。

### ENV-05 — Kaggle 憑證的執行期偵測與轉換
**需求**：憑證讀取邏輯必須先判斷 `KAGGLE_API_TOKEN` 的**格式**再決定怎麼用。

Kaggle 有兩代憑證，混用是最常見的失敗：
- **新式**：`KAGGLE_API_TOKEN` = 一個**純字串** token（不是 JSON）
- **舊式**：`kaggle.json` = `{"username": "...", "key": "..."}`，
  對應環境變數 `KAGGLE_USERNAME` ＋ `KAGGLE_KEY`

`kagglehub` 的解析順序是 `login()` → `KAGGLE_API_TOKEN` → `~/.kaggle/access_token`
→ Colab secret → `~/.kaggle/kaggle.json`。

**若 `..\.env` 裡的 `KAGGLE_API_TOKEN` 存的其實是 kaggle.json 的 JSON blob**，
它會被當成 bearer token 送出而 401。處理方式：

```python
raw = os.environ.get("KAGGLE_API_TOKEN", "").strip()
if raw.startswith("{"):
    blob = json.loads(raw)
    os.environ["KAGGLE_USERNAME"] = blob["username"]
    os.environ["KAGGLE_KEY"] = blob["key"]
    os.environ.pop("KAGGLE_API_TOKEN", None)   # <-- NOT optional
```

**最後那行 `pop` 不是可選的**：`KAGGLE_API_TOKEN` 在解析順序中排在 `kaggle.json` 之前，
留著一個格式錯誤的值會 shadow 掉剛剛寫好的正確憑證。
**驗證**：M2 實際下載成功。

### ENV-06 — 密鑰位置
**需求**：密鑰檔是 repo **上一層**的 `..\.env`，用 `python-dotenv` 讀，
路徑從 `configs/paths.yaml` 的 `dotenv` 欄位取得。**絕不 print 內容、絕不進 Git。**
**驗證**：`.gitignore` 含 `.env`；全樹 grep 無 `sk-` / `hf_` / `gho_` / `AIza` 開頭的長字串。

### ENV-07 — PowerShell 5.1 語法
**需求**：所有給使用者的指令用 PowerShell 形式。
**沒有 `&&` 與 `||`**（會直接是 parser error）→ 用 `A; if ($?) { B }`。
沒有三元運算子、沒有 `??`。`head` / `tail` / `which` / `touch` 都不存在。
**驗證**：文件內指令人工檢查。

### ENV-08 — multiprocessing 是 spawn 不是 fork
**需求**：所有會用到 `multiprocessing` 或 `DataLoader(num_workers>0)` 的進入點
**必須包在 `if __name__ == "__main__":` 之內**，否則子行程重新 import 主模組會無限遞迴。
腳本預設 `num_workers=0`。
**理由**：Windows 沒有 `fork`。除了遞迴問題，spawn 的每個 worker 都要重新 import 整個模組並
重新 pickle dataset，啟動延遲以秒計。對 5,000 張圖的 pHash 這種工作，
單執行緒或 `ThreadPoolExecutor`（Pillow 解碼時會釋放 GIL）反而更快。
**驗證**：`grep -L "__main__" $(grep -rln "multiprocessing\|num_workers" src/ scripts/)` → 空。

### ENV-09 — MAX_PATH 與快取位置
**需求**：`LongPathsEnabled = 0`，260 字元上限是活的。
HF 快取先沿用 C: 預設（與兄弟專案一致；SAM2 large 權重約 898 MB，C: 吃得下）。
若 HF 快取的 blob 檔名撞到上限，把 `configs/paths.yaml` 的 `hf_home` 設成短路徑
（例如 `D:/hf`）並重跑。`kagglehub` 快取同理可用 `KAGGLEHUB_CACHE` 重導。
**驗證**：下載完成且能載入模型。

### ENV-10 — 檔案與雜湊的跨平台一致性
**需求**：
- **每一個文字檔的 `open()` 都必須明寫 `encoding="utf-8"`**（讀與寫都要）。
  ⚠️ **本機實測**：這台機器的 Python 預設編碼是 **cp950**（繁中 Windows），
  `yaml.safe_load(open("configs/paths.yaml"))` 會直接拋
  `UnicodeDecodeError: 'cp950' codec can't decode byte 0xe2`——
  因為 config 註解裡有 UTF-8 字元。
  **這會在讀 config、寫 manifest、讀寫 COCO JSON 時隨機爆炸**，而且錯誤訊息指向編碼、
  不指向真正的原因，很浪費時間。`Path.read_text()` / `write_text()` 同樣要帶 `encoding="utf-8"`
- 雜湊一律以 binary 模式開檔（`open(p, "rb")`）。文字模式會改寫行尾，SHA256 就跟 Linux 對不上
- manifest 內的路徑一律 `Path.as_posix()`，不得出現反斜線
- 寫 JSON 用 `sort_keys=True, separators=(",", ":"), ensure_ascii=True`，並以 `newline="\n"` 寫檔
- 檔案系統大小寫不敏感（`images/` 與 `Images/` 在 Windows 同一個），manifest 路徑要正規化大小寫
- 解析 XML 用 `ET.parse(path)` 走路徑，**不要** `open(path).read()` 再 `fromstring`
  （Windows 文字模式預設 cp1252，遇到雜散位元組會炸）
**驗證**：連續兩次執行 `prepare_data.py`，`splits/MANIFEST.sha256` 完全相同。

---

## 3. 安裝流程

```powershell
# 在 repo 根目錄執行

# 1) 取得 Python 3.12（本機目前只有 3.10.9 與 3.14.5）
uv python install 3.12

# 2) 建立虛擬環境並安裝（pyproject.toml 已含 cu130 index 設定）
uv sync

# 3) HF CLI（系統的 huggingface-cli 在 anaconda 下是壞的）
uv add "huggingface_hub[cli]"
```

不要用 `pip install torch`。`uv sync` 會依 `pyproject.toml` 的 `[tool.uv.sources]`
自動從 cu130 index 取 torch 與 torchvision。

---

## 4. 版本表（2026-07-27 查證）

| 套件 | 版本 | 來源／理由 |
|---|---|---|
| Python | 3.12 | ENV-01 |
| `torch` | 2.13.0+cu130 | ENV-02，wheel 已實地確認存在 |
| `torchvision` | 0.28.0+cu130 | 同上；實際交給 uv 從同一 index 解析 |
| `transformers` | **≥5.14.1** | ENV-03 / [ADR-001](decisions.md#adr-001) ＋ [ADR-006](decisions.md#adr-006) |
| `kagglehub` | ≥1.0.2 | ENV-04；1.0.2 修了 tar 解壓的安全性問題 |
| `pycocotools` | ≥2.0.11 | cp312-abi3 win_amd64 wheel，免 MSVC |
| `imagehash` | ≥4.3.2 | pHash 分群 |
| `open-clip-torch` | ≥3.3.0 | 條件性 CLIP 分群；embedding 不可跨實作混用，需釘 pretrained tag |
| `opencv-python` | 最新 | 連通元件、inpaint、blending、濾波 |
| `scipy` | 最新 | `csgraph.connected_components` |
| `faster-coco-eval` | 1.7.2（選配） | pycocotools 的 drop-in 替代，宣稱結果完全一致但快約 4 倍 |

---

## 5. 驗證指令表

每一列都要能跑出預期輸出，全過才算 M1 完成。

| # | 指令 | 預期輸出 |
|---|---|---|
| 1 | `uv run python --version` | `Python 3.12.x` |
| 2 | `uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | `2.13.0+cu130 13.0 True NVIDIA GeForce RTX 4090` |
| 3 | `uv run python -c "import transformers; print(transformers.__version__)"` | `≥ 5.14.1` |
| 4 | `uv run python -c "from transformers import Sam2Model, Sam2Processor; print('ok')"` | `ok` |
| 5 | `uv run python -c "import cv2, scipy, imagehash, pycocotools, kagglehub; print('ok')"` | `ok` |
| 6 | `uv run python -c "import numpy, cv2; print(cv2.connectedComponentsWithStats(numpy.eye(4, dtype='uint8'))[0])"` | 一個整數（確認 cv2 正常） |
| 7 | `uv run python -c "import yaml,pathlib; d=yaml.safe_load(open('configs/paths.yaml')); print(d['data_root'])"` | `D:/sdg-data/02-safesynth` |
| 8 | `Test-Path D:\sdg-data\02-safesynth` | `True`（M2 建立後） |
| 9 | `uv run python -c "import os,dotenv; dotenv.load_dotenv('../.env'); print('KAGGLE_API_TOKEN' in os.environ)"` | `True`（**只印布林，不印值**） |
| 10 | `uv lock --check` | 無輸出（lock 與 pyproject 一致） |

第 2 列若 `torch.cuda.is_available()` 是 `False`，**幾乎一定是裝到 CPU-only 的 PyPI wheel**——
處理方式見 [troubleshooting.md](troubleshooting.md)。

---

## 6. Windows 原生踩雷清單

1. **`pip install torch` 會裝到 CPU-only 版**，不報錯，`cuda.is_available()` 直接 False（ENV-02）
2. **`import kaggle` 會刪掉 `KAGGLE_API_TOKEN` 環境變數**（ENV-04）
3. **`KAGGLE_API_TOKEN` 若存的是 JSON blob 會 401**，且會 shadow 掉正確憑證（ENV-05）
4. **PowerShell 5.1 沒有 `&&`**，是 parser error 不是警告（ENV-07）
5. **spawn 不是 fork**：忘記包 `if __name__ == "__main__":` 會無限產生行程（ENV-08）
6. **DataLoader worker 啟動極慢**：每個 worker 重新 import 整個模組。5,000 張圖用單執行緒或 thread pool 反而快
7. **MAX_PATH 260 字元是活的**（`LongPathsEnabled = 0`）（ENV-09）
8. **雜湊要用 binary 模式**，否則跨平台 SHA256 對不上（ENV-10）
9. **manifest 路徑要 `as_posix()`**，反斜線在 JSON 還要跳脫，且不可攜（ENV-10）
10. **檔案系統大小寫不敏感**：`images/` 與 `Images/` 在 Windows 同一個、在 Linux 不是（ENV-10）
11. **`ET.parse(path)` 走路徑**，不要先 `open().read()`（cp1252 解碼會炸）（ENV-10）
12. **Windows Defender 即時掃描會顯著拖慢** 5,000 個小 PNG 的解壓與 glob，
    偶爾還會在解壓中暫時鎖檔造成 `PermissionError`。可對資料目錄加排除項
13. **`--unzip` 會刪掉原始 zip**，那樣就沒辦法拿 zip 的 SHA256 當來源指紋。
    自己用 `zipfile` 解壓，先算完雜湊
14. **不要用 `torch.compile`**（Windows CUDA inductor 需要從 `vcvars64.bat` 啟動 shell，
    徒增失敗面）；**不要裝 flash-attn**（無官方 Windows wheel，transformers 的 SDPA 已足夠）
