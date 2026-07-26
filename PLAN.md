# PLAN.md — Phase 1 里程碑

> 規則見 [CLAUDE.md](CLAUDE.md)｜決策見 [docs/decisions.md](docs/decisions.md)｜工作紀錄見 [docs/worklog.md](docs/worklog.md)
> **Phase 1 目標**：資料與 split 凍結 → SAM2 cutout bank → 合成引擎 → 幾何與品質過濾 → 預覽圖與交接文件。
> **Phase 1 明確不做**：RT-DETRv2 訓練、compliance 推論、Gradio demo、發佈（全部是 Phase 2）。

狀態圖例：`[ ]` 未開始｜`[~]` 進行中｜`[x]` 完成

**勾選誠實性**：`[x]` 必須同時附一行 `**驗證於**：<sha> @ 日期`，且該 sha 必須存在於 `git log`。
驗證指令要**當場跑過並把真實輸出貼進 [docs/worklog.md](docs/worklog.md)**，不接受「應該會過」。

---

## M0 — 文件與規則凍結

- [x] **M0** 建立 `CLAUDE.md`、`PLAN.md`、`PLAN_PHASE2.md`、`docs/` 十一份文件、ADR-001~006、
  `configs/` 五份、兩支 skill、目錄骨架
  - **對應規格**：全部規格文件本身
  - **驗證**：`(Get-Content CLAUDE.md).Count` < 200；
    洩漏掃描（本機使用者名稱、學校信箱、`gho_`/`hf_`/`sk-`/`AIza` 開頭長字串）→ 零命中；
    WSL 路徑樣式僅允許出現在「明確聲明不使用」的句子中；
    禁用詞掃描（表示未完成狀態的字眼）在 `README.md`／`PLAN.md`／`docs/*.md` → 零命中，
    狀態一律用 `[ ]`／`[~]`／`[x]` 表達；
    `docs/*.md` 內所有相對連結的目標檔案都存在；
    Python／torch／transformers 版本在 `CLAUDE.md`、`docs/environment.md`、`pyproject.toml` 三處一致
  - **驗證於**：`6ca155e` @ 2026-07-27
    （個資／金鑰／禁用詞／AGPL 依賴掃描全 PASS；連結、ADR anchor、config 引用、
    四個版本號三處一致全 PASS；`CLAUDE.md` 127 行）

---

## M1–M2 — 環境與資料落地

- [x] **M1** 建立 uv 虛擬環境並鎖版（Python 3.12、torch 2.13.0+cu130、transformers≥5.14.1）
  - **對應規格**：ENV-01 ~ ENV-10
  - **驗證**：[docs/environment.md §5](docs/environment.md) 的十列驗證指令表**全部**通過；
    特別是第 2 列必須印出 `2.13.0+cu130 13.0 True NVIDIA GeForce RTX 4090`
    （若 `cuda.is_available()` 是 False，幾乎一定是裝到 CPU-only 的 PyPI wheel，見 K-01）；
    `uv.lock` 存在並進 git；`uv lock --check` 無輸出
  - **驗證於**：`6587a83` @ 2026-07-27
    實測輸出：Python `3.12.13`｜`2.13.0+cu130 13.0 True NVIDIA GeForce RTX 4090`｜
    `transformers 5.14.1`｜`Sam2Model` 可匯入｜`cv2/scipy/imagehash/pycocotools/kagglehub` 全可匯入｜
    `uv lock --check` 通過｜`KAGGLE_API_TOKEN` 是 opaque string（非 JSON blob，
    [ENV-05](docs/environment.md) 的轉換分支用不到）
    。第 8 列（`D:\sdg-data\02-safesynth` 存在）預期在 M2 才成立

- [x] **M2** 下載 Hard Hat Workers（約 1.2–1.5 GB，**下載前先跟使用者報備實測大小**），
  解壓到 `D:\sdg-data\02-safesynth\raw`，轉成 COCO
  - **對應規格**：DATA-01 ~ DATA-12
  - **驗證**：`uv run python scripts/prepare_data.py --verify` 全數通過——
    影像數 = 5,000；標註數 = 25,502（±1 記錄不中止）；
    per-class 實例數 = 18,966 / 5,785 / 751；per-class 圖片數 = 4,581 / 920 / 158；
    **不符就停下來報告，不要自行調整**；
    座標 offset 偵測結果印出並寫進 manifest（DATA-05）；
    未知標籤計數 = 0；所有 `iscrowd == 0`；
    **COCO 自評測試**：`pycocotools` 載入產出的 GT 與自己跑 `COCOeval` → mAP == `1.000`；
    `configs/paths.yaml` 的 `dataset.pinned_version` 已回填
  - **驗證於**：`1af595c` @ 2026-07-27
    實測來源 archive `1,314,241,385` bytes，SHA256
    `aa5c80a85f9f4bd3b27e44256f8e36f9a32c53ee423132fa6cd5ea603781be62`；
    Kaggle version 1 已釘住；座標全域最小值 `0` → offset `0`；
    5,000 張／25,502 框及三類實例與圖片數完全吻合；未知標籤 0、`iscrowd != 0` 為 0；
    COCO 自評 mAP `1.000`；11 個單元測試通過。

---

## M3–M5 — Split 凍結（**這三項全綠之前，一張合成圖都不准生**）

> 這一段的錯誤**事後無法補救**：split 一旦被生成端汙染，整份結論作廢。

- [x] **M3** Spike H1（`helmet` 框語意）／H3（近似分群結構）／H5（放置先驗品質）
  - **對應規格**：[docs/data_protocol.md §3](docs/data_protocol.md)
  - **驗證**：三張 contact sheet／熱圖產出並**自己打開檢視**後交使用者過目；
    H1 印出「同圖內 helmet×head IoU>0.1 的配對數」與 per-class 長寬比直方圖，
    並在 `docs/decisions.md` 追加一則 ADR 記錄合規定義走主定義還是 fallback；
    H3 印出四個 Hamming 門檻各自的群數、群大小直方圖、最大群大小，
    並用 5 個不同 seed 模擬 group split 印出各 split 圖片數，**選定門檻寫進 manifest**；
    H5 的位置先驗熱圖若過於發散，記錄「改以錨定放置為主」的決定
  - **驗證於**：`d29405d` @ 2026-07-27
    H1 contact sheets 與長寬比圖已打開檢視；`helmet×head` IoU>0.1 為
    95/9,603（0.99%），ADR-007 決定走 `class_direct`。
    pHash 四門檻與 5 seeds 模擬完成；因群數仍 >2,000，依規格啟用 guarded CLIP，
    選定 Hamming≤10，或 cosine≥0.85 且 Hamming≤20。
    最大 20 群 grid 已檢視，無 component collapse。
    H5 熱圖已檢視：head/helmet 可用位置先驗，person 改以錨定放置為主。

- [x] **M4** pHash（＋條件性 CLIP）近似分群 ＋ 分層 70/15/15 group split
  - **對應規格**：DATA-13 ~ DATA-17
  - **驗證**：**斷言「同 `group_id` 必定同 `split`」通過（hard fail）**；
    分群大小直方圖與最大 20 群印出，且**最大群 ≤ 250 張（5%）**——
    超過即門檻太鬆，必須調緊重跑；
    每個 split 至少拿到 10% 的 `person` 實例；
    三分割互斥、聯集 = 5,000、比例在 70/15/15 的 ±2% 內；
    CLIP 若啟用，模型與 pretrained tag 已寫進 manifest
  - **驗證於**：`d29405d` @ 2026-07-27
    4,808 群、最大群 8；`same group -> same split = PASS`。
    Train/Val/Test = 3,500/756/744（70.00%/15.12%/14.88%）；
    person 實例 = 525/113/113（各 split 均 ≥10%）；
    CLIP 模型與 pretrained tag、cosine、pHash guard 全寫入 manifest。

- [x] **M5** **凍結** split manifest ＋ SHA256 ＋ test blocklist ＋ 類別分布報告
  - **對應規格**：DATA-18 ~ DATA-21
  - **驗證**：`splits/{split_manifest.json, test_blocklist.json, source_checksums.json, MANIFEST.sha256}`
    四個檔案齊備；**連跑兩次 `MANIFEST.sha256` 完全相同**；
    manifest 含 kagglehub 版本號、座標 offset 決定、pHash/CLIP 門檻與模型 tag、seed；
    `reports/class_distribution.md` 的每個數字都能從 manifest ＋ COCO JSON 重新聚合出**完全相同**的值，
    且**解決了面積讀數衝突**（[data_protocol.md §1.2](docs/data_protocol.md)）；
    `reports/figures/class_distribution.png` **自己打開檢視**後交使用者過目
  - **驗證於**：`d29405d` @ 2026-07-27
    四個凍結檔齊備；完整重建兩次的 manifest SHA256 均為
    `ce9d76ee336cfba5e6071727442f7af413a8372f28cc9882093cb784587287a3`；
    744 張 Test 逐檔重雜湊通過；類別報告由 manifest＋COCO 重聚合，
    `class_distribution.png` 已打開檢視；全套 21 tests 與 ruff 通過。

---

## M6–M9 — 素材（**錯了事後難補救的第二段**）

- [x] **M6** 門檻校準工具（Spike H7）：算出所有幾何量的經驗百分位並回填 config
  - **對應規格**：[docs/filtering_spec.md §6](docs/filtering_spec.md)
  - **驗證**：`reports/calibration.md` 列出 per-class 的
    `mask_to_box_coverage`／框面積／最短邊／長寬比／solidity／
    helmet-head 在 person 內的包含比例與垂直位置／**真實物件邊界的 `seam_energy_ratio`**
    的 `[p1, p5, p50, p95, p99]`；
    `configs/*.yaml` 中原本標 `source: calibrated` 的欄位**全部**已填入實測值；
    仍標 `source: guess` 的欄位清單被明確印出（M13 前必須列進 `reports/filter_report.md`）
  - **驗證於**：`10718ba` @ 2026-07-27
    （凍結 Train 3,500 圖／17,815 框；分布、邊界能量與仍屬 guess 的欄位
    全列入 `reports/calibration.{json,md}`，config 校準值已回填）

- [x] **M7** Spike H2（SAM2 小框品質）＋ SAM2 Pass 1 全圖巡覽
  - **對應規格**：CUT-01 ~ CUT-05
  - **驗證**：H2 對 60 個框（依最短邊分三組）跑三種模式並產出三欄並排 grid，
    **自己打開檢視**；決定採用模式、真正的尺寸下限、
    以及校準後的 `sam_iou_min` / `min_object_score_logit`（目視良好組的 p10），寫回 config；
    Pass 1 對全部 Train 影像產出既有標註的 mask 並存到 `masks_pass1/`，
    每張記錄 QC 是否通過（`compose.py` 的 COMP-09 會用到）
  - **驗證於**：`10718ba` @ 2026-07-27
    （60 框 × 3 模式比較圖已目視；選定 effective crop-512；
    Pass 1 覆蓋 3,500 Train 圖／17,815 標註）

- [x] **M8** cutout bank Pass 2 ＋ contact sheets
  - **對應規格**：CUT-06 ~ CUT-12
  - **驗證**：**零個** `src_image_id` 落在 Val/Test（比對 `test_blocklist.json`，命中即失敗）；
    每個 RGBA PNG 有 4 通道且 alpha 非全 0 也非全 255；
    `bank_manifest.jsonl` 行數 == PNG 數；
    `reports/bank_report.md` 的漏斗（候選 → 逐閘門拒絕數 → 最終數）
    能從 `bank_rejects.jsonl` **重新聚合出完全相同的數字**；
    同 seed 對 100 張重跑，mask 一致；
    **同時列出 `n_person_cutouts` 與 `n_distinct_person_groups`**（ADR-003）；
    `reports/figures/bank_<class>_grid.png` 疊在**洋紅色**背景上產出並**自己打開檢視**——
    不能有背景滲漏、光暈、或第二個物件入鏡
  - **驗證於**：`916c6bf` @ 2026-07-27
    （7,255 accepted／10,560 rejected、Test 命中 0、manifest == PNG；
    100/100 mask 重跑一致；三類洋紅底 contact sheets 已目視）

- [~] **M9** hard negative 挖料與程序生成 ＋ Spike H6 ＋ **使用者人工簽核**
  - **對應規格**：COMP-20 ~ COMP-24
  - **驗證**：H6 對 200 張 Train 影像跑挖料器，產出 8×8 contact sheet 並**人工數出真實安全帽數量**；
    比例超過 `max_tolerated_helmet_rate` → **翻轉為程序生成為主**並記錄一則 ADR；
    三層防護全部實作（IoU 上限、通不過「像戴著的安全帽」測試、人工簽核）；
    **素材庫凍結前必須取得使用者對 contact sheet 的簽核**；
    程序生成的形狀確認是**調變真實背景紋理**而非平坦填色
  - **驗證於**：`c7514f4` @ 2026-07-27（程式與三層防護已完成）；
    H6 64 格候選圖已產出，**等待 kuotunyu 人工簽核，素材庫尚未解鎖**

---

## M10–M12 — 合成引擎與過濾（**M11 之前，合成總量不得超過 300 張**）

- [x] **M10** `src/synthetic/compose.py` ＋ COCO 自評測試
  - **對應規格**：COMP-01 ~ COMP-19、COMP-28、COMP-29
  - **驗證**：`uv run python -m src.synthetic.compose --n 32 --seed 42 --draw-boxes` 產出後
    **自己打開這 32 張畫了框的圖檢視**（沒有任何自動測試能取代這一步）；
    **COCO 自評測試** mAP == `1.000`；
    解析式 bbox 單元測試（已知 cutout ＋ 已知遮擋物，重算結果與閉式解差距 ≤ 1 px）；
    每個樣本的不變式通過：**真實標註輸出數 == 輸入數 − 蓄意移除數**（違反要 crash 不是過濾）、
    z-order == `y_bottom` 排序、無零面積標註、`assert_test_untouched()` 啟動時通過；
    同 seed 兩次產出影像 SHA256 相同；
    `reports/synthetic_stats.md` 的「情境 × 類別 × 尺寸桶」交叉表證明
    `small_distant` **真的**產出最短邊落在目標區間的框
  - **驗證於**：`e276d3e`、`dce0b85`、`e5c5bd9`、`49a51fe` @ 2026-07-27
    （最終 32 圖 review grid 已目視；COCO self-mAP 1.000；
    同 seed 兩次 32/32 SHA256 相同；`small_distant` 皆為 8–20 px）

- [~] **M11** Spike H4：貼上痕跡可偵測度（**放大生成量的硬閘門**）
  - **對應規格**：[docs/synthesis_spec.md §5](docs/synthesis_spec.md)
  - **驗證**：以當時設定生成 300 張，訓練小型二元分類器分辨「貼上的 patch」與「真實物件 patch」，
    印出 AUC；
    **AUC 高 → 先修調和與羽化並重跑，不准進入 M13**；
    AUC 接近隨機 → 記錄數值與判定，通過閘門
  - **驗證於**：`49a51fe`、`0178a62`、`63c22ba`、`889eba8`、`2cfce69` @ 2026-07-27
    （300 圖、group-disjoint 且類別/尺寸配對的 2,028 patches；
    AUC **0.7964**，95% CI 0.7481–0.8392，高於 0.60；
    context-matched 診斷因 frozen fold 缺真實對照而依預註冊規則停止；
    同類別原位替換 spike 也失敗，AUC **0.8312**（CI 0.7505–0.8984）；
    Poisson spike 因洗掉物件顏色而失敗，AUC **0.8869**（CI 0.8551–0.9170）；
    exact-source 成對控制仍達 AUC **0.9049**（CI 0.8788–0.9289），
    排除素材庫 selection bias 是主要假象；
    **硬閘門維持關閉，M13 不得開始**）

- [x] **M12** `src/filtering/rules.py` ＋ golden tests ＋ 門檻敏感度表
  - **對應規格**：FILT-01 ~ FILT-14
  - **驗證**：`uv run pytest tests/test_rules.py -v` 全綠，
    含約 12 個手工建構的 golden case（漂浮安全帽、安全帽貼在頭側面、安全帽吞掉臉、
    出界框、只剩 5% 可見、完全重複的合成圖、hard negative 壓到真實框、
    正確配戴、零改動、接縫瑕疵、深度不一致探出、真實標註遺失）；
    漏斗對帳：`n_pass + Σ n_reject_by_first_reason == n_total`，
    且能從 `records.jsonl` **重新聚合出完全相同的數字**；
    每筆被拒記錄的 `reject_reasons` 非空且全部在 enum 內；
    `reports/threshold_sensitivity.md` 產出且**沒有任何門檻的 ±20% 讓接受率變動超過警戒值**；
    12 通過 vs 12 被拒的並排圖產出並**自己打開檢視**——
    **若被拒的樣本看起來明明沒問題，就是門檻錯了，改門檻並記進 `docs/decisions.md`**
  - **驗證於**：`dce0b85`、`49a51fe` @ 2026-07-27
    （300 = 196 pass + 104 reject，七項 ledger/enum 對帳全 PASS；
    ±20% 敏感度警報 0；12 pass / 12 reject 圖已目視；全套 92 tests 與 ruff 通過）

---

## M13–M14 — 全量生成與交接

- [ ] **M13** 全量 2× pool 生成 ＋ 等量 filtered/unfiltered ＋ 巢狀子集
  - **對應規格**：COMP-25 ~ COMP-28、FILT-13
  - **驗證**：**M11 已通過**（前置條件，未過不准開始）；
    像素只寫一次，發兩份 COCO JSON；
    **filtered 與 unfiltered 張數完全相同**（unfiltered 是從同一 pool 均勻抽樣，seed=42），
    且 pool 大小與接受率有記錄；
    `0.5× ⊂ 1× ⊂ 2×` 的巢狀關係以集合包含斷言驗證，且情境配比在三個尺寸相同；
    `set(filtered_ids) ⊆ set(pool_ids)`；
    每筆記錄含 `thresholds_sha256` 與完整 provenance；
    `reports/filter_report.md` 列出**所有仍標 `source: guess` 的門檻**
  - **驗證於**：（等待 M11 通過與 M9 使用者簽核）

- [ ] **M14** 各情境預覽 grid ＋ `instructions_for_me.md` ＋ Phase 1 驗收
  - **對應規格**：PREV-01 ~ PREV-05
  - **驗證**：每個情境一張 `reports/figures/preview_<scenario>.png`，畫框、標類別與分數、標 `sample_id`；
    `preview_hard_negatives.png` **不畫框**且附「這是刻意的、由構造保證正確」的說明；
    通過 vs 被拒並排圖；
    **全部預覽圖自己先打開檢視過**再交使用者；
    `instructions_for_me.md` 明確寫出要看哪幾張、每張要看什麼、怎麼回饋（附可複製的回饋範本）；
    `PLAN.md` M0–M14 全勾且每項都有 `驗證於`；
    `git log` 每個里程碑至少一筆 commit
  - **驗證於**：（等待 M13 與 hard-negative 預覽）

---

## 風險分布

`M3`–`M9` 是**做錯了事後無法補救**的區段：
split 一旦被生成端汙染、cutout bank 一旦混入 Val/Test 來源，整份結論作廢，
而且**不會有任何錯誤訊息**——只會得到一組好看但無效的數字。
`M10` 之後都便宜可重做。

兩道硬閘門：
1. **`M3`–`M5` 全綠之前，一張合成圖都不准生**
2. **`M11` 通過之前，合成總量不得超過 300 張**

---

## 交接與追蹤

每完成一個里程碑：勾選本檔並補 `驗證於 <sha>` → 追加 [docs/worklog.md](docs/worklog.md) 一筆
（含**實際跑過的指令與真實輸出**）→ `git commit`（**不要帶 `Co-Authored-By:`**）→
給使用者「換你做」清單。

流程可直接呼叫 `/safesynth` skill。
