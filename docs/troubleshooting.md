# Troubleshooting

> 踩到的坑寫進這裡，一坑一節。格式：症狀 → 根因 → 解法 → 預防。
> 環境類問題可直接呼叫 `/safesynth-env` skill 走一遍自檢。

本檔目前只有**預先寫入的已知風險**（來自 M0 的查證）。
實際踩到的坑會在實作過程中追加。

---

## 已知風險（查證所得，實際上還沒踩到）

### K-01 — `torch.cuda.is_available()` 回 False

**症狀**：安裝成功、`import torch` 沒問題，但 CUDA 不可用，4090 完全沒被用到。

**根因**：PyPI 上的 `win_amd64` wheel 是 **CPU-only**（約 122 MB，
對比 Linux CUDA wheel 的約 527 MB）。PyPI 的檔案大小限制使得 Windows 的 CUDA 版本
從來不上傳 PyPI。裸的 `pip install torch` 會安裝成功且不報錯。

**解法**：確認 `pyproject.toml` 有 cu130 的 `[[tool.uv.index]]` 與 `[tool.uv.sources]`，
然後重裝：
```powershell
uv sync --reinstall-package torch --reinstall-package torchvision
```
驗證 `torch.__version__` 應含 `+cu130` 後綴。

**預防**：[ENV-02](environment.md)。永遠不要用 `pip install torch`。

---

### K-02 — SAM2 mask 品質莫名偏低

**症狀**：mask 看起來「差不多對但邊界糊」，`iou_scores` 約 0.94 而不是 0.98，沒有任何錯誤訊息。

**根因**：`transformers` 4.56.x 的 `_embed_boxes` 少了原始實作會補的 padding point。
這是**安靜的劣化**。

**解法**：升級到 `transformers>=4.57.1`（4.57.0 在 PyPI 被 yank，不要用）。

**預防**：[ENV-03](environment.md) / [ADR-001](decisions.md#adr-001)。
`/safesynth-env` 的自檢表第 3 列會檢查版本。

---

### K-03 — mask 有破洞或雜點，明明呼叫了後處理

**症狀**：`post_process_masks(max_hole_area=..., max_sprinkle_area=...)` 呼叫了，
但破洞和雜點還在。

**根因**：這兩個參數在 `transformers` 是**接受但完全不作用的 no-op**——
原始碼在該處留了一行註解，說明連通元件 kernel 是預計要補、目前沒有的。

**解法**：自己用 `cv2.connectedComponentsWithStats`（保留最大連通元件）
＋ `scipy.ndimage.binary_fill_holes` 做清理。參數見
`configs/compose.yaml` 的 `sam2.cleanup`。

**預防**：[CUT-08](synthesis_spec.md) / [ADR-001](decisions.md#adr-001)。

---

### K-04 — Kaggle 下載 401

**症狀**：憑證明明設了，`kagglehub` 仍回 401。

**根因（兩種）**：
1. `KAGGLE_API_TOKEN` 存的其實是舊式 `kaggle.json` 的 **JSON blob**，
   但新式憑證預期的是一個**純字串** token，於是整個 JSON 被當成 bearer token 送出
2. 已經正確設了 `KAGGLE_USERNAME` / `KAGGLE_KEY`，
   但格式錯誤的 `KAGGLE_API_TOKEN` 在解析順序中排在前面，把它們 shadow 掉了

**解法**：見 [ENV-05](environment.md) 的偵測與轉換程式碼。
關鍵是轉換完必須 `os.environ.pop("KAGGLE_API_TOKEN", None)`。

**預防**：ENV-05。

---

### K-05 — 讀 `KAGGLE_API_TOKEN` 拿到 KeyError

**症狀**：程式一開始環境變數還在，某個 import 之後就不見了。

**根因**：`kaggle/__init__.py` 在 import 時自動認證，並執行
`os.environ.pop("KAGGLE_API_TOKEN")`（kaggle-cli issue #882）。

**解法**：**不要 `import kaggle`**，改用 `kagglehub`。

**預防**：[ENV-04](environment.md)。`grep -rn "import kaggle" src/ scripts/` 應為零命中。

---

### K-06 — 行程無限增生 / 記憶體爆掉

**症狀**：跑平行處理時電腦卡死，工作管理員看到大量 python 行程。

**根因**：Windows 的 `multiprocessing` 用 **spawn** 不是 fork。
子行程會重新 import 主模組，若模組頂層就有啟動平行的程式碼，會無限遞迴。

**解法**：所有進入點包在 `if __name__ == "__main__":` 之內。

**預防**：[ENV-08](environment.md)。腳本預設 `num_workers=0`。

---

### K-07 — 兩次執行的 manifest SHA256 不同

**症狀**：`splits/MANIFEST.sha256` 每次跑都不一樣，凍結失去意義。

**根因（可能多重）**：
- 雜湊時用文字模式開檔（Windows 會改寫行尾）
- manifest 內存了 `WindowsPath` 的反斜線而非 `as_posix()`
- JSON 沒有用 canonical 形式（`sort_keys` / `separators`）
- 寫檔時沒指定 `newline="\n"`
- `image_id` 依賴了 `os.listdir` / `Path.glob` 的順序而非排序後的檔名

**解法／預防**：[ENV-10](environment.md) ＋ [DATA-09](data_protocol.md)、[DATA-18](data_protocol.md)。

---

### K-08 — PowerShell 指令直接 parser error

**症狀**：貼上一行含 `&&` 的指令，PowerShell 立刻報語法錯誤（不是執行失敗）。

**根因**：PowerShell 5.1 **沒有** `&&` 與 `||`。

**解法**：`A; if ($?) { B }`。

**預防**：[ENV-07](environment.md)。

---

### K-09 — 路徑過長錯誤

**症狀**：下載或載入模型時出現路徑相關的 IO 錯誤。

**根因**：本機 `LongPathsEnabled = 0`，260 字元上限是活的。
HF 快取的 blob 檔名很長，加上深層目錄容易超標。

**解法**：把 `configs/paths.yaml` 的 `hf_home` 設成短路徑（例如 `D:/hf`），
`kagglehub` 則用 `KAGGLEHUB_CACHE` 重導。

**預防**：[ENV-09](environment.md)。
