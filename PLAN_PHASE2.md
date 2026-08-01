# PLAN_PHASE2.md — Phase 2 里程碑

> 規則見 [CLAUDE.md](CLAUDE.md)｜Phase 1 里程碑見 [PLAN.md](PLAN.md)｜決策見 [docs/decisions.md](docs/decisions.md)｜工作紀錄見 [docs/worklog.md](docs/worklog.md)
> **Phase 2 目標**：四組對照訓練 → 合規邏輯 → 評測與錯誤分析 → demo → 發佈。
> **前置條件**：Phase 1 的 `M0`–`M14` 全綠，合成資料的 filtered／unfiltered 兩版已就位且等量。

狀態圖例：`[ ]` 未開始｜`[~]` 進行中｜`[x]` 完成
**勾選誠實性**：`[x]` 必須附 `**驗證於**：<sha> @ 日期`，且該 sha 存在於 `git log`。

> 分成兩份 PLAN 檔的理由：`PLAN.md` 已接近 20 KB，
> 而 `publish-repo` 的第 2 關會擋掉過大的 PLAN／PROGRESS 檔。

---

## 開工前：盤點輸入

**每次進入 Phase 2 的第一件事**，先列出預期需要的檔案路徑並逐一確認存在，
**缺件就停下來給使用者清單，不要用假設硬做**：

| 路徑 | 內容 |
|---|---|
| `splits/split_manifest.json` ＋ `MANIFEST.sha256` | 凍結的切分 |
| `splits/test_blocklist.json` | 防洩漏反查用 |
| `${data_root}/synthetic/images/` | 合成影像（像素只有一份） |
| `${data_root}/synthetic/annotations_filtered.json` | filtered 版 COCO |
| `${data_root}/synthetic/annotations_unfiltered.json` | unfiltered 版 COCO（**與 filtered 等量**） |
| `${data_root}/synthetic/records.jsonl` | 每筆 provenance 與過濾分數 |
| `${data_root}/interim/coco_all.json` | 真實資料的 COCO |
| `reports/filter_report.md` | 漏斗統計，含仍是 `guess` 的門檻清單 |

---

## M15 — 訓練 notebook 備妥（本機，**做完停下來交接**）

- [x] **M15** `notebooks/01_train_rtdetrv2.ipynb` ＋ 本機 smoke test ＋ 更新 `instructions_for_me.md`
  - **對應規格**：TRAIN-01 ~ TRAIN-14
  - **驗證**：本機以最小步數（1 step）跑通並存出 checkpoint，且能被重新載回；
    **斷點續跑分支實測**——刪 checkpoint／保留 checkpoint 各跑一次，行為符合預期；
    輸出目錄符合 `runs/<arm>/seed_<n>/` 的唯一命名；
    notebook 內**沒有任何明文 token**（只從 Colab Secrets 讀）；
    資料流程是「解壓到 `/content/data` 再訓練」而非直接從掛載的 Drive 讀圖；
    `instructions_for_me.md` 已寫到照做就行（Drive 路徑、Runtime 選型、Secrets 名稱、
    預估時數與 compute units、跑完要下載哪些檔案放回 `results/colab/` 的哪個路徑）
  - **驗證於**：`5b0a1a0` @ 2026-08-01
  - **實際結果**：本機 smoke test 冷啟動／熱啟動皆通過並回傳真實 COCO 指標
    （`eval_map` / `eval_map_small` 有值，不是 0）；Colab 上四組**完整跑完**，
    最後一格印出 `寫到 Drive: .../results_colab.zip`。
    實測 L4 約 1.7–1.9 it/s，`real_only` 組 `train_runtime_seconds` 6356.6（約 1.77 小時）。
  - **踩到的坑**：第一次 Colab 四組**全數陣亡**（[K-18](docs/troubleshooting.md)）。
    smoke test 當初設 `eval_strategy="no"`，於是 `compute_metrics` 本機零覆蓋。
    現在 smoke test 一定跑 eval（val 的 16 張切片）。

> **這裡是本 phase 的第一個交接點。** M15 做完先停，等使用者跑完 Colab
> 把產出放回 `results/colab/` 之後才繼續 M16。

---

## M16 — 四組 × 1 seed 訓練（Colab，使用者執行）＋ 產出盤點

- [x] **M16** 回收 Colab 產出並盤點完整性
  - **對應規格**：TRAIN-15 ~ TRAIN-18
  - **驗證**：對照 `instructions_for_me.md` 的預期清單逐項確認，**缺檔就列清單停下來問使用者**；
    四組各有 checkpoint 與訓練 log；
    每組的 `runs/` 目錄互不覆蓋；
    **從 log 重新聚合**訓練曲線（不抄 notebook 畫面上顯示的數字）；
    確認四組吃的**真實影像完全相同**（比對訓練資料清單的雜湊）；
    確認 `+Standard Aug` 組的增強清單**含光度增強**（[EXP-01](docs/experiment_protocol.md)）；
    確認 unfiltered 與 filtered 兩組的**張數相同**
  - **驗證於**：`7f2f0b3` @ 2026-08-01
  - **實際結果**：`uv run python -m scripts.audit_colab_results --archive ...` → **PASS，0 fatal、4 warning**。四組 `real_train_digest` 全部相同（`b46e8263…`）、步數皆 10,900、filtered 與 unfiltered 皆 3,500、real_only 與 standard_aug 皆 0。
  - **踩到的坑**：4 個 warning 是打包程式漏抓 `trainer_state.json`——HF 把它寫在 `checkpoint-N/` 裡面，而 glob 只掃 `seed_*/` 那一層，`is_file()` 把「找不到」變成「安靜跳過」。已修並移進被測模組（`src/training/ingest.py`）。
    另見 [K-20](docs/troubleshooting.md)：`run_record.json` 的 `eval_metrics` 與任何 checkpoint 都對不上，只能當「有跑過評測」的存在性證據。

---

## M17–M20 — 合規、評測與分析（本機 4090）

- [x] **M17** `src/inference/compliance.py`：兩種模式 ＋ 操作點選擇
  - **對應規格**：EVAL-01 ~ EVAL-04
  - **驗證**：`class_direct` 與 `geometric_pairing` 兩種模式都實作且由 config 切換；
    **`uv run pytest tests/test_compliance.py -k person_not_load_bearing` 通過**——
    把所有 `person` 偵測刪光後，合規判定結果**逐位元相同**；
    信心門檻在 **Validation** 上掃描選出（**絕不在 Test 上選**），
    掃描曲線與選定值寫進 `reports/compliance_operating_point.md`
  - **驗證於**：`bc44f3f` @ 2026-08-01
  - **實際結果**：EVAL-04 在 Validation 上選出 **0.07**（bare-head recall 0.8575、compliance precision 0.8507，下限 0.80），已凍結進 config。
  - **抓到兩個缺陷**：① 原本的佔位值 0.50 對這個模型是災難——223,200 個偵測的最高分只有 **0.2495**，在 0.50 上它什麼都不預測；② `select_operating_point` 會選出「從不觸發」的退化解（`unfiltered_syn` 選到 recall 0.0000 / precision 1.0000）。零召回現已不合格。
  - **最重要的結果**：各組各選各的操作點後，**`unfiltered_syn` 在任何會偵測到東西的門檻上都達不到 0.80 精確度**，`filtered_syn` 可以（0.8076）。過濾決定了能不能當合規檢查器部署。

- [x] **M18** `scripts/eval.py` ＋ 四組對照主表
  - **對應規格**：EVAL-05 ~ EVAL-14
  - **驗證**：`assert_test_untouched()` 啟動時通過，且訓練資料清單與 Test image id 交集為空；
    **`AP_small` 在每張圖自己的原始標註座標下計算**——用構造案例反向驗證
    （已知 area 略小於與略大於門檻的兩個 GT，斷言分桶正確），
    且座標映射的測試**至少一個 fixture 非正方形**
    （[DATA-25](docs/data_protocol.md#data-25--影像不是單一解析度預測框必須逐圖映射)：
    影像不是單一解析度，全正方形 fixture 抓不到兩軸轉置）；
    同時輸出各 size bucket 的**實例數**；
    `pycocotools` 與 `faster-coco-eval` 在同一輸入上 mAP 差距為 0；
    `results/detection_metrics.csv` 每列一個 arm × seed × 指標，
    且 `reports/` 的每個表格數字都能由它重新聚合出**完全相同**的值
  - **驗證於**：`7f2f0b3` @ 2026-08-01
  - **實際結果**：`results/detection_metrics.csv` 441 列。**兩條獨立實作算出的主表一致到 8.8e-07**（JSON 浮點往返誤差）。凍結 Test 744 張：`real_only` primary AP_small **0.4511**、`standard_aug` 0.4236、`unfiltered_syn` 0.3759、`filtered_syn` 0.3664。**合成沒有提升。**
  - **防洩漏實跑通過**：`assert_test_untouched()` 覆蓋 744 張；四組訓練清單的 digest 皆等於凍結 train split，與 Test 交集為零。
  - **這一輪跑的是 `--bootstrap-resamples 0`**，所以它不滿足 EVAL-09；報告已載明這件事。

- [x] **M19** 錯誤分析：FP/FN 對照 grid ＋ 情境切分表
  - **對應規格**：EVAL-15 ~ EVAL-18
  - **驗證**：四類對照圖（修好的 FN／修好的 FP／**新增的 FP**／兩組都錯）各若干張產出並
    **自己打開檢視**；
    「新增的 FP」這一類**必須呈現，不得省略**——那是合成資料的副作用；
    情境切分表（小物件／擁擠／低光桶）各組指標齊備；
    **檢驗針對性**：若 `small_distant` 佔 25% 預算但小物件桶的進步與其他桶相當，
    如實寫出「這是資料變多而非針對性生效」；
    hard-negative 子集的每圖誤報數獨立成表
  - **驗證於**：`7f2f0b3` @ 2026-08-01
  - **實際結果**：四類對照圖已產出並**逐張打開檢視**（`reports/figures/error_analysis/`）。相對 baseline，`filtered_syn` 修好 73 個漏檢、**新製造 1,304 個**——弄壞的是修好的 18 倍；修好 715 個誤報、新增 291 個。
  - **針對性判定：`target_slice_did_not_improve`**，而且比無效更糟——`small_distant` 佔可切片預算最大份額（21.7%），而 `small_object` 是**移動最不利**的切片（−0.0572，對比 `crowded` −0.0412、`low_light` −0.0477）。
  - **未滿足**：hard-negative 子集在凍結 Test 上是**空的**（744 張全都含 helmet 或 head），EVAL-16 的候選區域 fallback 未實作，該表無法產生。

- [~] **M20** 速度對照組（寬鬆授權模型，**不得用 Ultralytics**）
  - **對應規格**：DEMO-05、[ADR-005](docs/decisions.md#adr-005)
  - **驗證**：所選模型的授權經查證且與 MIT repo 相容，證據寫進 ADR-005；
    `grep -rn "ultralytics" src/ scripts/ notebooks/` → **零命中**；
    速度數字含 batch size、輸入解析度、dtype 三項脈絡
    （**缺這三項的 FPS 沒有意義**）；
    主表仍以 RT-DETRv2 為準，速度對照另立一表
  - **已完成的部分**（`55da06a`）：RF-DETR-Nano 實際載入並前向通過（`RfDetrForObjectDetection`，transformers 5.14.1）；兩個模型的 Hub 授權於量測當下重新查證皆為 `apache-2.0` 並釘住 revision；`grep -rn ultralytics` **零命中**，而且改成由 `scripts/check_forbidden_licences.py` **真的執行掃描**（原本是寫死的字串，種一個違規檔進去照樣說通過）；延遲數字含 batch／解析度／dtype 三項。
  - **實測發現，且它推翻了原本要下的結論**：把輸入從 640 降到 320（像素少 4 倍），**兩個模型都沒有變快**（RT-DETRv2 +0.1%、RF-DETR +3.5%）。batch-1 是 dispatch-bound，量到的是我們的 eager-PyTorch 推論路徑而不是架構。**所以「RF-DETR-Nano 比較快」這句話不能寫**——只量一個解析度就會理直氣壯地寫下錯的結論。
  - **這個 `[~]` 涵蓋的範圍**：① 延遲數字量自**預訓練 80 類 checkpoint**，不是微調後的 3 類權重，報告全篇標著 PROVISIONAL，要在最終權重上重測一次；② ADR-005 範圍裡的「RF-DETR 用同樣四組各訓練一次」需要 GPU，留到 GPU 空出來再跑。
  - **驗證於**：（未完成）

---

## M21 — 補 seeds（條件性）

- [ ] **M21** Real-only 與最佳 Filtered 組各補到 3 seeds
  - **對應規格**：TRAIN-19、EVAL-09
  - **前置判斷**：先看 M18 的主表。**若 Filtered 組沒有提升，補 seed 不會改變結論**——
    此時把額度留給錯誤分析，並在 worklog 記錄這個取捨
  - **驗證**：兩組各 3 個獨立 seed，各用獨立 `runs/` 目錄；
    主表改報 **mean ± std**；
    只有 1 seed 的組別在表格中明確標註「單一 seed」；
    **不得用單 seed 的零點幾個點差距宣稱勝出**（[EVAL-10](docs/evaluation_spec.md)）
  - **驗證於**：（未完成）

---

## M22–M24 — Demo 與發佈

- [ ] **M22** Gradio demo（原生 Windows，**不用 WSL**）＋ 效能量測 ＋ GIF
  - **對應規格**：DEMO-01 ~ DEMO-04
  - **驗證**：上傳圖片與影片兩種輸入都能跑；
    畫面顯示 bbox、**合規狀態用顏色區分**、幀層級 `compliant/total` 與 `compliance_rate`；
    效能量測有 warm-up、報**中位數與 p95**（不是平均）、
    分開報「純模型推論」與「端到端」、
    峰值 VRAM 用 `reset_peak_memory_stats()` 後量測，
    且記錄 batch size／解析度／dtype；
    demo GIF 畫面**同時有戴帽與沒戴帽的人**，最好再有一個遠距小物件；
    GIF 在 `.gitignore` 的 `assets/*` 規則中加了例外
  - **驗證於**：（未完成）

- [ ] **M23** README ＋ `scripts/verify_readme.py` ＋ CI
  - **對應規格**：PUB-01 ~ PUB-05、PUB-10
  - **驗證**：`uv run python scripts/verify_readme.py` 通過——
    README 每張表的數字都能從 `results/` 的原始檔重算；
    README 正文（不是只有 Limitations）交代資料集標註缺陷與
    **「所有主張都是相對的」**；
    主動說明**為什麼是四組不是五組**；
    `grep -n "shields.io" README.md` → 零命中（不放靜態假 badge）；
    `.github/workflows/ci.yml` 至少做 `uv sync --locked` → `pytest` 且為綠；
    `uv.lock` 已提交
  - **驗證於**：（未完成）

- [ ] **M24** Hugging Face 上傳 ＋ 發佈總驗收
  - **對應規格**：PUB-06 ~ PUB-11
  - **驗證**：合成資料集（filtered ＋ unfiltered ＋ `records.jsonl`）與最佳權重已備妥；
    dataset card 含來源授權鏈（CC0 1.0）、生成方法、filtered/unfiltered 差別與**等量**這件事、
    限制、以及**「SAM2 自動 mask 只用於合成素材、未用於任何 ground truth」**的明確聲明；
    model card 含訓練資料組成、四組結果表、
    **`AP_small` 在原始 416 座標計算**的說明、基礎模型授權、
    以及「不要拿絕對 AP 當品質保證」的警語；
    HF card ↔ GitHub README 互相連結；
    **呼叫個人 skill `publish-repo` 跑完整驗收**（不要在本 repo 重寫發佈流程）；
    **commit 歷史零個 `Co-Authored-By:` trailer**，所有 commit 的 author 都是 repo 擁有者；
    轉 public 前讓使用者過目
  - **驗證於**：（未完成）

---

## 交接點

Phase 2 有**一次 Colab 往返**：

```
M15（本機備 notebook）→ 停 → 使用者跑 Colab → 產出放回 results/colab/ → M16 起繼續
```

M15 完成時要給使用者的東西寫在 `instructions_for_me.md`，
標準是**照著做就行、不需要再問問題**。

其餘所有工作都在本機 4090 完成，不燒 Colab 額度。

---

## 風險分布

| 段落 | 錯了的後果 |
|---|---|
| `M16` 四組資料組成 | 若四組吃的真實影像不同、或 unfiltered 與 filtered 不等量，**整份對照失效**且事後看不出來 |
| `M18` `AP_small` 座標系 | 算在 640 座標會讓主敘事指標**安靜地**測量另一件事 |
| `M17` 操作點在 Test 上選 | 等於用測試集調參，結論不可信 |
| `M20` 誤用 Ultralytics | 整個 repo 的授權被汙染成 AGPL |

`M19` 之後的工作都便宜可重做。
