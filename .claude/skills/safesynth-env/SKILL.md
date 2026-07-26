---
name: safesynth-env
description: 02-safesynth-ppe 在原生 Windows（不使用 WSL）的環境自檢與修復 runbook。當使用者說「環境壞了」「裝不起來」「CUDA 不能用」「torch 看不到 GPU」「Kaggle 下載失敗 401」「模型載不起來」「路徑太長」，或任何一條 docs/environment.md 的驗證指令失敗時使用；也用於 M1 建立環境時逐項確認。涵蓋 uv 建環境、torch 必須從 cu130 index 安裝（PyPI 的 Windows wheel 是 CPU-only 會安靜失敗）、transformers 版本下限、Kaggle 憑證格式偵測、MAX_PATH、PowerShell 5.1 沒有 && 等本專案特有的踩雷點。
---

# SafeSynth — Windows 原生環境自檢與修復

規格見 [docs/environment.md](../../../docs/environment.md)（ENV-01 ~ ENV-10），
已知坑見 [docs/troubleshooting.md](../../../docs/troubleshooting.md)（K-01 ~ K-09）。

**這個專案不使用 WSL。** 所有指令用 PowerShell 5.1 形式，
**沒有 `&&` 與 `||`**（那是 parser error 不是執行失敗），用 `A; if ($?) { B }`。

---

## 流程

### 1. 先跑完整自檢表

依序執行 [docs/environment.md §5](../../../docs/environment.md) 的十列驗證指令。
**逐列比對預期輸出**，把失敗的列記下來，不要跑到一半就開始修。

最關鍵的是第 2 列：
```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
預期：`2.13.0+cu130 13.0 True NVIDIA GeForce RTX 4090`

### 2. 對照下表修復

### 3. 修完重跑整張表，不要只重跑修過的那列

環境問題常常互相牽連（例如重裝 torch 會動到 torchvision）。

### 4. 把新踩到的坑追加進 `docs/troubleshooting.md`

一坑一節，格式：症狀 → 根因 → 解法 → 預防。

---

## 故障對照表

| 症狀 | 根因 | 解法 |
|---|---|---|
| `torch.cuda.is_available()` 是 **False** | **裝到 CPU-only 的 PyPI wheel**。Windows 的 CUDA 版本從來不上 PyPI（wheel 約 122 MB vs Linux 約 527 MB），裸的 `pip install torch` 會成功且不報錯 | 確認 `pyproject.toml` 有 cu130 的 `[[tool.uv.index]]`（`explicit = true`）與 `[tool.uv.sources]`，然後 `uv sync --reinstall-package torch --reinstall-package torchvision`。版本字串必須含 `+cu130`（K-01） |
| torch 版本解析不到 2.12／2.13 | 用了 **cu128** index——它最高只到 2.11.0，CUDA 12.8 自 torch 2.12 起已從標準發佈矩陣移除 | 改成 cu130（ADR-002） |
| SAM2 mask 邊界糊、`iou_scores` 約 0.94 而非 0.98，**但沒有錯誤訊息** | `transformers` 4.56.x 的 `_embed_boxes` 少了 padding point，**安靜劣化** | 升級到 `>=4.57.1`（4.57.0 在 PyPI 被 yank，不要用）。K-02 / ADR-001 |
| 呼叫了 `post_process_masks(max_hole_area=...)` 但破洞還在 | 那兩個參數在 transformers 是**接受但不作用的 no-op** | 自己用 `cv2.connectedComponentsWithStats` ＋ `scipy.ndimage.binary_fill_holes`。K-03 / CUT-08 |
| Kaggle 下載 **401** | `KAGGLE_API_TOKEN` 存的是舊式 `kaggle.json` 的 **JSON blob**，被當成 bearer token 送出；或它 shadow 掉了正確的 `KAGGLE_USERNAME`/`KAGGLE_KEY` | 執行期偵測格式並轉換，**轉換後必須 `os.environ.pop("KAGGLE_API_TOKEN", None)`**。K-04 / ENV-05 |
| 讀 `KAGGLE_API_TOKEN` 拿到 KeyError | **`import kaggle` 在 import 時就把它 `os.environ.pop` 掉了**（kaggle-cli issue #882） | **不要 `import kaggle`**，改用 `kagglehub`。K-05 / ENV-04 |
| 行程無限增生、電腦卡死 | Windows 的 multiprocessing 用 **spawn** 不是 fork，子行程重新 import 主模組造成遞迴 | 進入點包在 `if __name__ == "__main__":`；腳本預設 `num_workers=0`。K-06 / ENV-08 |
| 兩次執行的 `MANIFEST.sha256` 不同 | 文字模式開檔雜湊／路徑存了反斜線／JSON 非 canonical／沒指定 `newline="\n"`／`image_id` 依賴 glob 順序 | 見 K-07。ENV-10 ＋ DATA-09 ＋ DATA-18 |
| 指令貼上去直接語法錯誤 | PowerShell 5.1 **沒有** `&&` | 改 `A; if ($?) { B }`。K-08 / ENV-07 |
| 路徑相關 IO 錯誤 | `LongPathsEnabled = 0`，260 字元上限是活的 | 把 `configs/paths.yaml` 的 `hf_home` 設成短路徑（如 `D:/hf`）；kagglehub 用 `KAGGLEHUB_CACHE`。K-09 / ENV-09 |
| `huggingface-cli` 拋 traceback、`hf` 找不到 | 系統 anaconda 裡的那份是壞的 | 在專案 venv 內 `uv add "huggingface_hub[cli]"`，並用 `uv run` 執行 |
| 解壓 5,000 個 PNG 很慢、偶爾 `PermissionError` | Windows Defender 即時掃描 | 對 `D:\sdg-data\02-safesynth` 加排除項；`PermissionError` 通常重試即可 |

---

## 從零建立環境（M1）

```powershell
# 在 repo 根目錄執行
uv python install 3.12
uv sync
uv add "huggingface_hub[cli]"
```

`uv sync` 會依 `pyproject.toml` 的 `[tool.uv.sources]` 自動從 cu130 index 取 torch 與 torchvision。
**不要用 `pip install torch`。**

完成後跑一次完整自檢表，全綠才算 M1 通過。

---

## 不該做的事

- **不要用 `torch.compile`**：Windows 的 CUDA inductor 需要從 `vcvars64.bat` 啟動 shell，
  徒增失敗面，而本專案不需要那點速度
- **不要裝 flash-attn**：沒有官方 Windows wheel，從原始碼編要 30 分鐘以上且需要 MSVC；
  社群預編 wheel 是第三方binary。transformers 內建的 SDPA 已經夠用
- **不要改用 WSL**：使用者明確排除（ADR-002）。所有問題都要在原生 Windows 解決
- **不要硬編絕對路徑**：一律讀 `configs/paths.yaml`
