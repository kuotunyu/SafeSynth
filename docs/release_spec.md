# Release Spec — Demo、README 與發佈

> 指標定義見 [evaluation_spec.md](evaluation_spec.md)。
> **發佈流程本身不在這裡**——那是個人 skill `publish-repo` 的職責，
> 本文件只規範「發佈前必須備妥什麼」。

---

## 1. Demo（`src/inference/demo_gradio.py`，本機 4090，原生 Windows）

> ⚠️ 原始計畫寫「本機 4090，WSL2」。**本專案不使用 WSL**（[ADR-002](decisions.md#adr-002)），
> demo 直接在原生 Windows 的 uv 環境跑。

### DEMO-01 — 形式：Gradio，上傳圖片或影片
不做 webcam 即時（工地場景在攝影頭前不好重現，且不該假設使用者手邊有 webcam）。
上傳一張圖或一段短片 → 輸出標註後的圖／影片。

### DEMO-02 — 畫面上要顯示的東西
- 每個偵測框：類別、信心值
- **每個人的合規狀態**用顏色區分（compliant / non-compliant），
  而不是只畫框——這才是這個專案在做的事
- 幀層級摘要：`compliant / total`、`compliance_rate`
- **效能數字**：單張 latency（ms）、FPS、峰值 VRAM

### DEMO-03 — 效能量測方法
不要拿第一次推論的時間當數字。規範：
- 先跑若干次 warm-up 不計時
- 量測若干次取**中位數**與 p95（不是平均——平均會被偶發的 GC／排程雜訊拉高）
- 分開報「純模型推論」與「含前後處理的端到端」兩個數字，
  因為前者是模型的性質、後者才是使用者體感
- 峰值 VRAM 用 `torch.cuda.max_memory_allocated()`，量測前先 `reset_peak_memory_stats()`
- 記錄當時的 batch size、輸入解析度、dtype（fp32／fp16／bf16）——
  **沒有這三個數字的 FPS 是沒有意義的**

參數與重複次數見 `configs/evaluation.yaml` 的 `benchmark`。

### DEMO-04 — README 的 demo GIF
錄一段短 GIF 放進 README。選材要能看出這個專案的價值：
**畫面裡要同時有戴帽與沒戴帽的人**，最好再有一個遠距小物件。
GIF 檔放 `assets/`（`.gitignore` 目前擋掉 `assets/*`，發佈時要對這個檔案加例外）。

### DEMO-05 — 速度對照組
使用**寬鬆授權**（Apache-2.0 級）的另一個偵測器跑同樣四組資料當速度基準。
**不用 Ultralytics YOLO**——AGPL-3.0 會傳染到 `import` 它的程式碼，
與 MIT repo 牴觸（[ADR-001](decisions.md#adr-001) 已因同一理由否決它）。
具體選型見 [training_spec.md](training_spec.md)，並在該處記錄一則 ADR。

主成果仍以 RT-DETRv2 為準；速度對照只是佐證「我知道怎麼在準確度與延遲之間選型」。

---

## 2. README

### PUB-01 — 結構
英文為主，開頭一段繁中摘要。章節順序：

1. **Problem** — 為什麼工地 PPE 偵測會在困難情境失敗、為什麼標註成本是瓶頸
2. **Method** — mermaid 流程圖：資料凍結 → SAM2 cutout → 情境化合成 → 幾何過濾 → 四組訓練 → 評測
3. **Experiments** — 四組設計說明 ＋ **防洩漏聲明** ＋ 為什麼只有四組（見 PUB-03）
4. **Results** — 主表 ＋ AP_small／bare-head recall 的對照圖 ＋ 合成量曲線 ＋ FP/FN 對照 grid
5. **Reproduce** — 從零開始的完整步驟（含 Windows 原生的注意事項）
6. **Limitations** — 資料集標註缺陷、`person` 類、任何負面結果
7. **License & Citations**

### PUB-02 — ⚠️ 資料集缺陷必須在 README 正文交代，不能只放 Limitations
SHEL5K 重標同樣 5,000 張圖得到 75,570 個標註 vs 原版 25,502，約 2/3 真實物件未標註。
因此：
- **所有主張都是相對的**（同一凍結 Test 上 A 組 vs B 組），**絕不報絕對 AP 當成果**
- `person` AP 依 [EXP-03](experiment_protocol.md) 的規範呈現

把這件事講清楚是**可信度的來源**，不是要道歉的弱點——
五組（此處四組）對照協定本質上就是相對比較，正好對症。

### PUB-03 — 說明為什麼是四組不是五組
第五組「Full-real 上限」只適用於 Real-only 是縮減子集的實驗。SafeSynth 的 Real-only
**本來就吃全部真實 Train**，沒有更高的真實資料上限可言，
所以第五組不適用。**README 要主動說明這件事**，否則讀者會以為漏做一組。

### PUB-04 — 數字可追溯
README 每一個數字都必須能追回 `results/` 或 `reports/` 底下的檔案。
寫 `scripts/verify_readme.py` 自動核對主表數字能從原始檔重算，並跑給使用者看。
**不要放靜態假 badge**（`publish-repo` gate 5 會抓 `shields.io`）。

### PUB-05 — 成本與環境揭露
- Colab compute units 實際用量
- 本機 GPU 型號與訓練/評測時數
- **Phase 1 全部在本機完成、零 API 花費**這件事值得寫出來
  （標註成本歸零是這個專案的核心賣點）

---

## 3. Hugging Face 發佈

### PUB-06 — 合成資料集（COCO 格式）
上傳 `filtered` 與 `unfiltered` 兩版，**以及 `records.jsonl` 的 provenance**。

Dataset card 必須含：
- **來源與授權鏈**：Hard Hat Workers（Kaggle `andrewmvd/hard-hat-detection`，CC0 1.0），
  溯源到 Northeastern University China / Harvard Dataverse。
  衍生的合成影像以 **CC0 1.0** 釋出以配合來源
- **生成方法**：SAM 2.1（Apache-2.0）取 cutout、情境化 copy-paste、
  幾何與品質過濾；每筆樣本的 provenance 欄位說明
- **filtered vs unfiltered 的差別**，以及兩者**等量**這件事（[COMP-26](synthesis_spec.md)）
- **限制**：標註來自原始資料集，繼承其漏標問題；
  合成影像不得用於宣稱絕對效能
- **明確聲明**：SAM2 的自動 mask 只用於合成素材，未用於任何 ground truth

### PUB-07 — 模型權重
上傳最佳組的權重。Model card 需含：訓練資料組成（真實 + 合成的張數與比例）、
四組對照的結果表、指標定義（特別是 `AP_small` 在原始 416 座標下計算，見 [EVAL-07](evaluation_spec.md)）、
基礎模型 `PekingU/rtdetr_v2_r18vd` 的授權、以及**不要拿絕對 AP 當品質保證**的警語。

### PUB-08 — 交叉連結
HF card ↔ GitHub README 互相連結（`publish-repo` gate 9 的要求）。

---

## 4. 發佈前的檢查

### PUB-09 — 呼叫個人 skill `publish-repo`
**不要在這個 repo 內重寫發佈流程**：
維護者的 publishing workflow 涵蓋洩漏掃描、體積檢查、必備檔案、CI、數字可追溯性、
stale 文件偵測、commit 切分與 tag/Release；repo 內只保留可公開驗證的產物契約。

### PUB-10 — 本 repo 自己要先備妥的
- `.github/workflows/ci.yml`：至少 `uv sync --locked` → `pytest`
- `uv.lock` 已提交
- `scripts/verify_readme.py` 可跑且通過
- **commit 歷史零個 `Co-Authored-By:` trailer**，且所有 commit 的 author 都是 repo 擁有者
  （這會讓 GitHub 首頁的 Contributors 多出第二個人）
- 全 repo 無 API key、無本機絕對路徑、無個資

### PUB-11 — 轉 public 前讓使用者過目
`publish-repo` 的鐵則：所有 git／gh／hf 的**寫入**動作由使用者親自執行。
Claude 只負責檢查、準備內容、產出逐行指令。
