# CLAUDE.md — 02-safesynth-ppe

> 每個 session 開工前必讀。與 [PLAN.md](PLAN.md)、[docs/worklog.md](docs/worklog.md)、[docs/decisions.md](docs/decisions.md) 一起構成本專案的長期記憶。
> 快速恢復脈絡：直接呼叫 `/safesynth` skill。環境出問題：`/safesynth-env`。

## 【去哪找】

| 要做什麼 | 讀哪份 |
|---|---|
| 動資料／split／COCO 轉換 | [docs/data_protocol.md](docs/data_protocol.md)（`DATA-*`） |
| 動 cutout bank／合成引擎 | [docs/synthesis_spec.md](docs/synthesis_spec.md)（`CUT-*`、`COMP-*`） |
| 動過濾規則／預覽圖 | [docs/filtering_spec.md](docs/filtering_spec.md)（`FILT-*`、`PREV-*`） |
| 訓練與 Colab 往返 | [docs/training_spec.md](docs/training_spec.md)（`TRAIN-*`） |
| 合規邏輯／指標／錯誤分析 | [docs/evaluation_spec.md](docs/evaluation_spec.md)（`EVAL-*`） |
| Demo／README／HF 發佈 | [docs/release_spec.md](docs/release_spec.md)（`DEMO-*`、`PUB-*`） |
| 實驗協定與防洩漏 | [docs/experiment_protocol.md](docs/experiment_protocol.md)（`EXP-*`） |
| 裝套件／CUDA／uv 問題 | [docs/environment.md](docs/environment.md)（`ENV-*`）或 `/safesynth-env` |
| 「當初為什麼這樣選」 | [docs/decisions.md](docs/decisions.md)（`ADR-*`） |
| 撞到看過的錯誤 | [docs/troubleshooting.md](docs/troubleshooting.md)（`K-*`） |
| 里程碑 | Phase 1 → [PLAN.md](PLAN.md)｜Phase 2 → [PLAN_PHASE2.md](PLAN_PHASE2.md) |
| 要發佈 | 個人 skill `publish-repo`（**不要在本 repo 重寫發佈流程**） |

**數值一律在 `configs/*.yaml`**；`docs/` 的規格只寫判定式與 config key，不複製數值。

---

## 【分工】本機 vs Colab

**Phase 1 全部在本機完成，不需要 Colab。** 本 phase 沒有任何超過 30 分鐘的 GPU 訓練——SAM2 只做推論，兩趟合計不到一小時；合成是純 numpy/cv2。

**本機 RTX 4090（24GB, Windows 11 native）**
- 資料處理、split 凍結、SAM2 cutout、大量離線合成、過濾、預覽圖、評測、demo
- ≤30 分鐘的訓練，以及所有 notebook 的 1-step smoke test

**Colab notebook**（Phase 2 的 RT-DETRv2 訓練才會用到）
- 資料先解壓到 `/content/data` 再訓練，**絕不**直接從掛載的 Drive 讀圖訓練
- checkpoint 定期同步回 Drive 的 `MyDrive/sdg-portfolio/02-safesynth-ppe/`
- 必須支援**斷點續跑**（偵測既有 checkpoint 自動接續）
- 每個平行 notebook 用**唯一輸出目錄**：`runs/<group>/seed_<n>/`
- token 只從 Colab Secrets 讀，絕不寫進 notebook

Colab 產出由使用者放回 `results/colab/` 後，Claude Code 才接手分析。

---

## 【實驗鐵律】

**本專案是四組**（Phase 2 執行，協定見 [docs/experiment_protocol.md](docs/experiment_protocol.md)）
1. Real-only
2. + Standard Augmentation
3. + Unfiltered Synthetic
4. + Filtered Synthetic

通用鐵律的第五組「Full-real 上限」標了「適用時」，**本專案不適用**：
SafeSynth 的 Real-only 本來就吃全部真實 Train，沒有更高的真實資料上限可言。
README 要主動說明這點，否則讀者會以為漏做一組。

**不可違反的規則**
- Validation / Test **只用真實資料**；generator、過濾器都**不得接觸 Test**
- 先凍結 split manifest（`splits/*.json` ＋ seed=42 ＋ 來源檔 SHA256）**才能**開始生成
- 近似圖片先用 pHash 分群，**同群必須同 split**
- **SAM2 的自動 mask 只用來做合成素材，絕不拿來當 Test ground truth**
- 全組合先跑 1 seed；Real-only 與最佳 Filtered 組補到 3 seeds 報 mean±std
- 合成樣本的標籤**一律由生成流程自動產生**，人工只做抽查觀察
- 每筆合成樣本記 provenance：來源圖／來源 bbox／seed／參數／filter score／拒絕原因
- **filtered 與 unfiltered 兩組必須等量**（從同一個 pool 均勻抽樣），否則會把「資料更多」和「資料更好」混在一起
- **若 synthetic 沒有提升，如實報告並分析原因**，不准挑選性隱藏實驗
- 因約 2/3 真實物件未標註，**所有主張必須是相對的**（同一凍結 Test 上 A 組 vs B 組），永遠不能是絕對 AP

---

## 【工作方式】

- 開工先把本階段拆成 [PLAN.md](PLAN.md) 里程碑，**每項附驗證方法**，做完勾掉
- **勾選誠實性**：`[x]` 必須附「驗證於 `<sha>` @ 日期」，且驗證指令要**當場跑過並貼出真實輸出**
- **繁體中文**溝通；程式碼註解與 README 用**英文**
- >2GB 下載或任何花錢動作**先問使用者**
- 套件版本、模型名稱、價格**一律先上網查證再選型**，並把來源連結寫進文件
- 自己產的圖表與樣本圖**要自己打開檢視**，不合理就修
- **數值只寫在 `configs/*.yaml`**；`docs/` 的規格只寫判定式與 config key，不複製數值
- 每階段結束：更新 `PLAN.md` ＋ 追加 [docs/worklog.md](docs/worklog.md) 一筆 ＋ `git commit` ＋ 給使用者「**換你做**」清單
- 重大選型寫成 ADR 追加到 [docs/decisions.md](docs/decisions.md)，**只追加不改寫**
- 踩到的坑寫進 [docs/troubleshooting.md](docs/troubleshooting.md)
- commit 訊息**不要帶 `Co-Authored-By:`**（會讓 repo 首頁多出一個貢獻者），且用英文
- **git 權限界線**：本機的 `init` / `add` / `commit` / `branch` Claude 可自行執行（可逆、不對外）；
  **`push`、建立 remote repo、Hugging Face 上傳、`filter-repo` 等改寫歷史的動作一律由使用者親自執行**
- **API key 絕不進 Git**

---

## 【本機環境】Windows-native，不使用 WSL

細節見 [docs/environment.md](docs/environment.md)。以下是每次都會踩的重點：

- **這個專案不使用 WSL**。所有路徑用 Windows 形式，不准出現 `/mnt/c/...` 或 `~/sdg-portfolio/...`
- Shell 是 **PowerShell 5.1**：沒有 `&&` 與 `||`，用 `A; if ($?) { B }`；沒有三元運算子；`head`/`tail`/`which`/`touch` 都不存在
- **不得硬編絕對路徑**。所有路徑一律讀 [configs/paths.yaml](configs/paths.yaml)
- 大檔（原始資料、cutout bank、合成影像）放 `D:\sdg-data\02-safesynth`；專案資料夾只留程式碼、設定、文件、`splits/`、`reports/` 小圖
- 密鑰放在上層的 `..\.env`（**不在** repo 內），用 `python-dotenv` 讀，絕不 print 內容
- PyTorch DataLoader 在 Windows 用 spawn：`num_workers>0` 時所有進入點必須包 `if __name__ == "__main__":`，寫腳本時預設 `num_workers=0`
- 檔案雜湊一律用 binary 模式開（`"rb"`）；manifest 內路徑一律 `Path.as_posix()`——否則 SHA256 跨平台對不上

**三條 SafeSynth 專屬紅線**
- **torch 必須從 cu130 index 裝**。PyPI 上的 Windows wheel 是 CPU-only，`pip install torch` 會安靜地讓 4090 閒置。cu128 index 最高只到 torch 2.11.0，不可用
- **`transformers>=5.14.1`**。兩個理由：4.56.x 的 SAM2 box prompt 有數值 bug 會安靜降低 mask 品質（[ADR-001](docs/decisions.md#adr-001)）；v5 改了 image processor 命名，抄 2025 年的教學會壞（[ADR-006](docs/decisions.md#adr-006)）。v5 的 `from_pretrained` 預設 dtype 改為 `"auto"`，**一律明確傳 `dtype=`**
- **不要 `import kaggle`**——它在 import 時會 `os.environ.pop("KAGGLE_API_TOKEN")`。一律用 `kagglehub`

---

## 【專案座標】

| 項目 | 值 |
|---|---|
| 本機資料夾 | `mySyntheticData\2_SafeSynth`（三案並列於同一個母資料夾） |
| 未來 GitHub repo | `02-safesynth-ppe` |
| 資料集 | Hard Hat Workers（Kaggle `andrewmvd/hard-hat-detection`，**CC0 1.0**），5,000 張 416×416 PNG |
| 三類實例數 | `helmet` 18,966（4,581 張圖）／`head` 5,785（**920 張圖**）／`person` 751（**僅 158 張圖**） |
| 已知標註缺陷 | SHEL5K 重標同樣 5,000 張圖得 75,570 個標註，原版僅 25,502；`person` 類標註不完整。處置見 [ADR-003](docs/decisions.md#adr-003) |
| 合規狀態定義 | 由 `helmet` vs `head` 推導，**不依賴 `person`** |
| 主敘事指標 | **AP_small** 與 **bare-head recall**；次要：per-class AP、compliance P/R、hard-negative 每圖誤報數 |
| 硬性版本 | Python 3.12｜torch 2.13.0+**cu130**｜torchvision 0.28.0+cu130｜**transformers ≥5.14.1** |
| 主模型 | RT-DETRv2-R18（`PekingU/rtdetr_v2_r18vd`，Apache-2.0，20.2M 參數）。**不做 ImageNet 正規化**，checkpoint 已是 640×640 |
| 目前階段 | **Phase 1**（資料凍結 ＋ cutout bank ＋ 合成引擎 ＋ 過濾）。**RT-DETRv2 訓練、compliance 推論、Gradio demo、發佈全部是 Phase 2**——規格已寫好（見【去哪找】），但 **Phase 1 未完成前不要動手做 Phase 2** |
| Colab 方案 | 每月 **500** compute units |
| 速度對照組 | **不得使用 Ultralytics/YOLO**（AGPL-3.0 會傳染到我們的程式碼）。改用寬鬆授權的偵測器，選型見 [ADR-005](docs/decisions.md#adr-005) |

**兩道硬閘門**（見 PLAN.md）
1. `M3`–`M5` 全綠之前，**一張合成圖都不准生**
2. `M11`（貼上痕跡可偵測度驗證）之前，**合成總量不得超過 300 張**
