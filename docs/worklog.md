# Worklog — 02-safesynth-ppe 施工日誌

<!-- 收工時做兩件事：(1) 覆寫「現況快照」 (2) 在「工作日誌」最上面插入一筆。 -->
<!-- 快照是待證偽的假設，不是真相。真相以 git log 為準——開工時務必交叉驗證。 -->
<!-- 本檔超過約 18 KB 時把舊日誌摺疊或歸檔（publish-repo gate 2 的門檻是 ~20 KB）。 -->

## 現況快照

*每次收工覆寫，只留最新一份。*

- **更新時間**：2026-07-31
- **最後驗證 commit**：`d83d4cf` test(synthesis): freeze v23 owner review pages（共 231 筆）
- **目前里程碑**：`M0`–`M8`、`M10`、`M12` 完成。
  `M9` 的 H6 已由 kuotunyu 以 0/64 簽核（保持 `[~]` 直到素材庫接入 compositor）。
  **`M11` 已於 2026-07-31 結案為 failed-and-accepted**（[ADR-011](decisions.md#adr-011)）——
  H4 未通過且不宣稱通過，但閘門後果從「無限阻擋」改為「允許到 1×、禁止 2×」。
  `M13` 起解除阻擋。
- **⚠️ 未 commit 的改動**：無。
- **已凍結不得再動**：`splits/split_manifest.json`、`test_blocklist.json`、
  `source_checksums.json`、`MANIFEST.sha256`；manifest SHA256 為
  `ce9d76ee336cfba5e6071727442f7af413a8372f28cc9882093cb784587287a3`。
- **資料與素材落地**：Kaggle version 1、Pass-1 masks、7,255 個通過素材、
  300 張 H4/M12 候選與全部診斷 run 都在 `D:\sdg-data\02-safesynth`；
  Test 洩漏為 0，cutout 重現抽查 100/100。
- **環境**：已安裝並驗證。Python 3.12.13、torch 2.13.0+cu130、
  torchvision 0.28.0+cu130、transformers 5.14.1、diffusers 0.39.0。
  `docs/environment.md §5` 驗證表十列全過（含 `cuda.is_available()==True` / RTX 4090）。
- **下一個動作（一句話、可直接動手）**：執行 `M13`——生成 1× pool
  （**上限 1×，不做 2×**）、產出等量 filtered/unfiltered 與 `0.5× ⊂ 1×` 巢狀子集，
  完成後重跑一次 H4 確認 AUC 未顯著移動。
- **卡住的事**：無阻擋。H4 AUC 0.7964 現在是**已知限制**而非 blocker，
  但它必須出現在 README 正文與每一張成果表旁。
- **⚠️ 已知待修（不阻擋 M13）**：`uv run ruff check .` 有 4 個 import 排序錯誤
  （`notebooks/00_flux2_v2_diagnostic.ipynb`、
  `scripts/diagnose_supervised_labeler_v20_numeric_failure.py`），可 `--fix` 自動修。
- **等使用者做的事**：目前無；遠端 GitHub repo 仍未建立。
- **驗證本快照的指令**：
  ```
  uv run python -m scripts.audit_phase1_handoff
  uv run ruff check .
  uv run pytest -q
  ```

---

## 工作日誌

*append-only，新的插在最上面。*

### 2026-07-31 · M11 結案為 failed-and-accepted；退回 v23 labeler

- **接手時發現快照落後 128 筆 commit**（停在 `355fbd2` / 07-28，
  HEAD 已是 `d83d4cf`），期間跑完 v6→v23 共 18 輪 labeler 迭代。
  這是證偽步驟抓到的，快照已重寫。舊日誌移到 `worklog_archive.md`
  以維持本檔在 20 KB 門檻下（20,401 → 4,978 bytes）。
- **v23 數值獨立重算驗證通過**：用 `audit_evidence.json` 的原始 box
  以 IoU≥0.5 貪婪配對重算，得 `TP=91 / FP=13 / FN=19`、
  precision 0.8750、recall 0.8273、median IoU 0.8303——與 Codex 回報**逐位相同**。
- **但 v23 退回，理由不是目視格數**：`best epoch 3`、48 圖 audit 最高信心僅 **0.1396**、
  且 **TP 與 FP 的分數分布幾乎完全重疊**（中位數 0.0583 vs 0.0481）。
  這代表**不存在能分開兩者的門檻**，precision 0.875 是壓到 0.035 門檻換來的假象。
  診斷是**嚴重欠訓練**，不是標註品質問題，再跑一輪同型迭代不會改善。
- **自我更正**：初次目視我判斷格 11／29 是「GT 框了未佩戴安全帽、模型正確略過」，
  查原始 box 後確認**錯誤**——模型有框，只是洋紅與綠框幾乎完全重合而在縮圖上分不出。
  真實模式相反：**模型一致地會框未佩戴安全帽，是 GT 自己前後不一致**
  （1629、3803 有框，4029 沒框），格 35 的 3 個 FP 由此而來。
  審查表印的「未佩戴的孤立安全帽不應框」**不是 GT 的實際行為**。
- **決策 [ADR-011](decisions.md#adr-011)**：H4 的技術判定維持不變（未通過、不宣稱通過、
  未放寬 0.60、未換分類器），改變的是**判定的後果**。
  9 條合成路線與 18 輪 labeler 迭代都無法翻轉，且 feature-family 診斷顯示
  HOG-only 0.7792／HSV-only 0.6816 兩類訊號都超標，非單一參數可修。
  因此把 H4 失敗當成本專案的一項公開發現，生成上限提高到 **1×**（禁止 2×），
  帶著這個限制推進 Phase 2。
- **whole-image v10 與 FLUX 路線一併停止**（ADR-010 標記為停止）。
  相關腳本與報告全部保留在 repo 內作為失敗路線的證據，不刪除。
- **驗證**：`uv run pytest -q` → **311 passed**；
  `uv run python -m scripts.audit_phase1_handoff` → `integrity_passed: true`、
  單一作者、零 `Co-Authored-By`、無 remote、
  唯一 blocker 是 `M11/H4 AUC 0.7964 exceeds the 0.60 scale-up maximum`（已由 ADR-011 處置）。
  `uv run ruff check .` → 4 個 import 排序錯誤，已記入快照待修。
- **刻意不做**：不修 v23、不跑 v24、不下載 FLUX 權重。

### 2026-07-28 · supervised labeler v6 人工審查拒絕

- v6 數值 audit 為 precision 0.8995、recall 0.8584、median matched IoU
  0.8430，但 kuotunyu 在固定 48 格 Train-only 審查中確認 9 個問題格：
  `04, 06, 07, 13, 23, 27, 38, 43, 45`。
- 問題包含背景誤框、04 漏掉一頂安全帽，以及 13/27 相鄰安全帽未逐一
  分離。原始綠／青頁與分離後綠／洋紅頁的 SHA256 都已綁進拒絕證據。
- 正式證據：
  `reports/supervised_labeler_v6_human_review.json`，canonical evidence SHA256
  `4f23014a5ec9eea77317a172e3c0901e61fa9b9c91b9a40470c3d6c35464e4ec`。
- `generation_gate.allowed=false`；whole-image v10 不得執行。Validation/Test
  讀取皆為 0，whole-image generation 亦為 false。
- v6 的 48 格已揭露，只能供 v7 診斷，不能再作 v7 untouched audit。
  驗證：`uv run ruff check src scripts tests` 通過，`uv run pytest -q`
  為 161 passed；提交 `355fbd2`。

### 2026-07-27 · FLUX.2 v2 A100 診斷完成

- A100 40 GB 以 `full_model_on_cuda` 完成四個 Train 案例、三個預註冊
  variant，共 12/12 輸出；總推論 86.70 秒。
- 結果 ZIP SHA256：
  `33bd82ae1625137b0a42aaf92473e94c95591eb29a1d846bf4833b060003e7c6`。
- 三個 variant 的 outside-mask changes 都是 0；移除 reference 的 masked
  RGB MAE 僅 0.2260/255，降低 strength 也沒有一致的視覺改善。
- 沒有選出替代 variant。v1 identity gate 的失敗維持有效；未計算新 H4
  AUC，沒有開啟 M13 或 Phase 2。
- 下一個方法必須先處理 rejected pilot 暴露的 invalid draft 與
  mislocalized anchor，並在新的 untouched identity pilot 前預註冊。

### 2026-07-27 · H6 簽核與 H4 Option A 預註冊

- kuotunyu 對 exact-grid SHA 簽核 0/64 真正安全帽，H6 通過。
- 選定 Apache-2.0 `FLUX.2-klein-base-4B`，revision 與 14.88 GiB
  Diffusers 檔案清單已由 Hugging Face metadata 驗證；權重未下載。
- 鎖定 reference-conditioned boundary inpainting、protected core、64 圖
  人工 identity gate 與新的 one-shot H4 fold；0.60 門檻不變。
- 新增 local-only loader、模型 manifest hard gate、像素身份不變式與 7 項測試。


---

> 更早的日誌已移到 [worklog_archive.md](worklog_archive.md)。
