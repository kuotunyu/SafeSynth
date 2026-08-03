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
    **成本已實測，不是憑感覺跳過的**：CPU 上一輪 COCOeval over 744 張、223,200 個偵測要 **12.0 秒**，1000 次重抽 × 4 組 = **14.1 小時**，而且那只是**一個**指標。
    試過把偵測截到每圖前 100 個（2.7× 加速），但**不是無損的**——pycocotools 的 `maxDets` 是**逐類別**套用，整圖取前 100 會砍掉某類別本該入榜的框，實測 mAP 差 2.6e-04；改成逐類別取前 100 只省 8%。
    **判斷**：主表最關鍵的差距是 `real_only` 0.4511 對 `filtered_syn` 0.3664（0.085），遠大於 Test 剖析算出的組間雜訊下限 ±0.031，方向性不成疑問。花 14 小時 CPU 去確認一件已經清楚的事，不如把時間用在別處。GPU 空出來後再補。

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
  - **EVAL-16 已用 spec 自己的 fallback 補上**（2026-08-02）：Test 沒有天然的
    hard negative（744 張全含 helmet 或 head），所以在 Test 上挖候選區域**僅供分析**。
    挖到 290 個區域，contact sheet **逐格看過，64 格裡零個真安全帽**（與 Train 上的 H6 一致）。
    **但這個指標在這裡是退化的**：四組分別是 1／1／0／1 個誤報，
    spread 只有 1，**不能拿 0.005 對 0.000 當排名讀**，報告裡明說了這件事。
    退化的原因也量出來了，而且它證實了 [K-11](docs/troubleshooting.md) 已揭露的代價——
    挖料器是靠**色相與圓度**選的，所以撈到的多半是黃橘色但形狀完全不像安全帽的東西
    （木板、圍籬、機具面板、裸露手臂），偵測器根本不會對它們反應。
    要讓 EVAL-16 有鑑別力，需要**形狀像安全帽但沒被戴著**的干擾物，
    這個資料集在 Test 上沒有足夠數量。

- [~] **M20** 速度對照組（寬鬆授權模型，**不得用 Ultralytics**）
  - **對應規格**：DEMO-05、[ADR-005](docs/decisions.md#adr-005)
  - **驗證**：所選模型的授權經查證且與 MIT repo 相容，證據寫進 ADR-005；
    `grep -rn "ultralytics" src/ scripts/ notebooks/` → **零命中**；
    速度數字含 batch size、輸入解析度、dtype 三項脈絡
    （**缺這三項的 FPS 沒有意義**）；
    主表仍以 RT-DETRv2 為準，速度對照另立一表
  - **已完成的部分**（`55da06a`）：RF-DETR-Nano 實際載入並前向通過（`RfDetrForObjectDetection`，transformers 5.14.1）；兩個模型的 Hub 授權於量測當下重新查證皆為 `apache-2.0` 並釘住 revision；`grep -rn ultralytics` **零命中**，而且改成由 `scripts/check_forbidden_licences.py` **真的執行掃描**（原本是寫死的字串，種一個違規檔進去照樣說通過）；延遲數字含 batch／解析度／dtype 三項。
  - **實測發現，且它推翻了原本要下的結論**：把輸入從 640 降到 320（像素少 4 倍），**兩個模型都沒有變快**（RT-DETRv2 +0.1%、RF-DETR +3.5%）。batch-1 是 dispatch-bound，量到的是我們的 eager-PyTorch 推論路徑而不是架構。**所以「RF-DETR-Nano 比較快」這句話不能寫**——只量一個解析度就會理直氣壯地寫下錯的結論。
  - **① 微調權重重測：完成**（2026-08-02，`reports/speed_baseline_probe.md` 三道閘門全綠）。
    RT-DETRv2-R18 微調 3 類權重：**model-only 12.79 ms / 78.2 FPS、end-to-end 16.23 ms / 61.6 FPS**，
    batch 1、640×640、fp16、**SM clock 鎖定 2520 MHz**。
    contention 0/9 通過、clock spread **1.00**（門檻 1.15）通過。
    **可重現的前提是鎖時脈**：`nvidia-smi -lgc 2520,2520`（管理員），量完 `-rgc`。
    未鎖時同一支 harness 兩次跑出 11.81 ms 與 26.74 ms，見 [K-22](docs/troubleshooting.md)。
    鎖了之後仍需重試 2 次才拿到 p95 乾淨的一輪——時脈鎖住的是頻率，不是別的行程。
  - **① 的歷程（保留，因為它是 K-22 的證據）**：程式先完成、數字量不出來。
    `--weights` 可指向本機微調 checkpoint（demo 實際服務的 `real_only/seed_1337/checkpoint-1752`），
    處理器仍取自 Hub（Trainer 輸出目錄沒有 `preprocessor_config.json`）。
    **PROVISIONAL 標籤改成推導的**：讀 `config.id2label`，只有當所有列都預測
    `helmet/head/person` 才會消失；把 `--weights` 指向 COCO checkpoint 不會讓它消失。
    模型確實換了（20.08 M 參數 vs 80 類的 20.2 M、logits `[1,300,3]`）。
    **但數字無法發佈**：見 [K-22](docs/troubleshooting.md)——同一支 harness 兩次跑出
    11.81 ms 與 26.74 ms，延遲跟著 SM clock 走（2520 MHz→12.89 ms、1215 MHz→27.85 ms）。
    已加 `evaluate_clock_spread()` 與 `benchmark.max_clock_spread_ratio`，
    修好取樣點之後重跑三次，spread 1.50／3.11／3.65，**三次都 FAIL**。
    要取得可發佈數字需 `nvidia-smi -lgc`（**需管理員權限，屬使用者親自執行**）。
  - **修正一句先前的描述**：上面寫的「batch-1 是 dispatch-bound」講得太滿。
    CUDA event 量到的 GPU 側時間 ≈ wall clock（12.17 vs 12.20 ms），
    而且 wall clock 與 SM clock 近乎成反比——時間花在 GPU 上、隨時脈縮放，
    但**不隨像素量縮放**。「不隨解析度變化」這個觀察不變，
    「所以兩個模型不能靠這個數字分高下」這個結論也不變。
  - **② 四組訓練與評測：完成**（2026-08-04）。設定由
    `configs/training_rfdetr.yaml` 載入，並非複製 RT-DETR 設定後只換 checkpoint。
    **不是複製 `training.yaml` 換 checkpoint**——2026-08-02 逐項讀取
    `Roboflow/rf-detr-nano` 之後發現兩者在每一個關鍵預處理鍵上都不同：

    | 鍵 | RT-DETRv2-R18 | RF-DETR-Nano |
    |---|---|---|
    | `do_normalize` | false | **true**（ImageNet mean/std） |
    | 輸入 | 640×640 | **384×384** |
    | `do_pad` | false | **true** |
    | 預訓練類別 | 80 | **91** |
    | backbone | ResNet-18 | **DINOv2** |

    第一列的方向**與 `training.yaml` 裡的警告相反**：RT-DETR 只除以 255，
    自己加 ImageNet 正規化會壞；RF-DETR 需要那個正規化，把
    `do_normalize: false` 抄過來會壞掉另一邊。**兩種都不會報錯。**
    唯一原封不動可用的是 backbone LR 分組——`trainer.py` 以
    `"backbone" in name` 選取，RF-DETR 命名為 `model.backbone.backbone.*`，
    473 個參數中 249 個命中。
    `optimizer:` 底下**每一個值都標 `source: guess`**（把 CNN 的 recipe 套到 ViT），
    且有測試強制這個標記存在。
    最後依使用者決策跑完整四組，不再縮成兩組：`real_only`、`standard_aug`、
    `unfiltered_syn`、`filtered_syn` 都以 seed 1337 跑滿 **10,900 steps**，
    並各自用 best-validation checkpoint 評測。
  - **② 凍結結果**：744 張 frozen Test、1,000 次影像層級 bootstrap 已完成，
    來源為 `results/rfdetr_detection_metrics.csv`（424 列）與
    `results/rfdetr_predictions_index.json`。`primary_map_small`：
    `real_only` **0.4841 [0.4653, 0.5048]**、`standard_aug`
    **0.4970 [0.4727, 0.5219]**、`unfiltered_syn`
    **0.4959 [0.4747, 0.5194]**、`filtered_syn`
    **0.5030 [0.4841, 0.5240]**。四組區間全部重疊，不能宣稱合成資料勝出；
    synthetic arms 又只有一半 real-image exposures，且 H4 AUC 0.9053 失敗。
    與 RT-DETRv2 的負向結果合看，結論是**架構敏感且不確定，沒有穩健提升**。
  - **③ 還沒做的**：用 fine-tuned RF-DETR checkpoint 取得可發布的延遲表。
    使用者已鎖定 2520 MHz，clock-spread gate 三次都通過，但 host-contention
    p95 gate 三次都失敗；`reports/rfdetr_speed_baseline_probe.md` 因此保留 FAIL，
    不能把數字放進 README。只在機器真正安靜時再重跑，不降低門檻。
  - **驗證於**：`reports/rfdetr_detection_main_table.md`、
    `results/rfdetr_detection_metrics.csv`、`results/rfdetr_predictions_index.json`；
    M20 整體仍為 `[~]`，待 ③ 通過後才可標 `[x]`。

---

### EVAL-09 — bootstrap 信賴區間（2026-08-02 完成）

- **驗證**：1,000 次重抽、單位是 Test **影像**不是實例；
  `metrics.bootstrap_workers` 平行化，**區間與 worker 數無關**
  （同輸入在 1/8/12/16/20 workers 下 `BootstrapCI` 逐欄相等）
- **實測時數**：16 workers 共 **2 小時 20 分**（單執行緒換算 9.3 小時，加速 4.48×）
- **對帳**：重跑的 424 個點估計與已 commit 的主表**逐一相同（0 個不一致）**，
  前提是 `--device cpu`——見 [K-23](docs/troubleshooting.md)
- **它改變了三個主張，只有一個對本專案有利**：
  - `real_only` 勝過兩個合成組：**區間不重疊，成立**
  - `real_only` 勝過 `standard_aug`：**區間重疊，不成立**。
    0.0275 的點差落在雜訊內，README 原本把它當成排序，已改掉
  - `filtered_syn` 在偵測指標上勝過 `unfiltered_syn`：**三個指標區間全部重疊，不成立**。
    過濾可量測的效果在合規操作點，不在這裡

---

## M21 — 補 seeds（條件性）

- [x] **M21** Real-only 與最佳 Filtered 組各補到 3 seeds（條件未觸發，記錄後結案）
  - **對應規格**：TRAIN-19、EVAL-09
  - **前置判斷**：先看 M18 的主表。**若 Filtered 組沒有提升，補 seed 不會改變結論**——
    此時把額度留給錯誤分析，並在 worklog 記錄這個取捨
  - **驗證**：兩組各 3 個獨立 seed，各用獨立 `runs/` 目錄；
    主表改報 **mean ± std**；
    只有 1 seed 的組別在表格中明確標註「單一 seed」；
    **不得用單 seed 的零點幾個點差距宣稱勝出**（[EVAL-10](docs/evaluation_spec.md)）
  - **實際決策**：RT-DETRv2 的 `filtered_syn` 顯著低於 `real_only`；RF-DETR 的
    `filtered_syn` 雖然點估計較高，但與 `real_only` 的 95% bootstrap 區間重疊，
    且有 real-image exposure 與 H4 domain-gap 混淆。沒有得到「Filtered 提升」的
    觸發條件，因此不再花 6 個額外長訓練重複一個不成立的方向性主張。
  - **驗證於**：README 的兩模型家族結果段落、`docs/worklog.md` 2026-08-04 記錄。

---

## M22–M24 — Demo 與發佈

- [x] **M22** Gradio demo（原生 Windows，**不用 WSL**）＋ 效能量測 ＋ GIF
  - **對應規格**：DEMO-01 ~ DEMO-04
  - **驗證**：上傳圖片與影片兩種輸入都能跑；
    畫面顯示 bbox、**合規狀態用顏色區分**、幀層級 `compliant/total` 與 `compliance_rate`；
    效能量測有 warm-up、報**中位數與 p95**（不是平均）、
    分開報「純模型推論」與「端到端」、
    峰值 VRAM 用 `reset_peak_memory_stats()` 後量測，
    且記錄 batch size／解析度／dtype；
    demo GIF 畫面**同時有戴帽與沒戴帽的人**，最好再有一個遠距小物件；
    GIF 在 `.gitignore` 的 `assets/*` 規則中加了例外
  - **已完成的部分**（`17e5153`）：`app.py` 圖片與影片兩個分頁都實作；
    合規狀態用顏色區分（綠＝戴帽、紅＝裸頭、灰＝`person` 不帶判定）；
    幀層級 `compliant / total` 與 `compliance_rate` 都顯示；
    效能行含 model-only 與 end-to-end 兩個數字、batch／解析度／dtype 三項脈絡，
    CUDA 時另報 `max_memory_allocated()`；啟動時跑一次 warm-up 不計時。
  - **實測驗收**：對兩張真實 Test 影像跑完整路徑並**打開圖檢視**——
    `hard_hat_workers245` 12/15 compliant、`hard_hat_workers2261` 6/15，
    輸出在 `reports/figures/demo_examples/`。
    第一版渲染是壞的（15 個完整標籤在 416px 上糊成一團），看圖才發現，已改成
    依框寬遞減的標籤。
  - **① DEMO-04 的 GIF 已產出**（`scripts/make_demo_gif.py`，`assets/demo.gif`，
    8 幀 802 KiB）。**改成靜態幀的循環蒙太奇而不是實拍影片**，README 明說這件事：
    手邊沒有工地短片，而資料集也代替不了——凍結的 pHash 分群顯示
    4,808 個群裡有 4,643 個是單張、最大群只有 8 張，**沒有夠長的連續影格**。
    但有 501 張圖同時含戴帽與裸頭（val 佔 70 張），那正是 DEMO-04 真正要求的畫面。
    選幀規則：先看兩種判定的平衡度，再看畫出來的框數，全部取自 Validation。
  - **② CUDA 分支已實跑，而且它是壞的**（現已修好）。demo 的 CUDA 路徑從來沒被執行過，
    第一次跑就連撞兩個錯：處理器輸出 float32 對上 float16 模型
    （`Input type and weight type should be the same`），以及
    `outputs.to("cpu")`——`ModelOutput` 是 dataclass，沒有 `.to()`。
    兩個都修在 `app.py` 與 `scripts/make_demo_gif.py`。
    峰值 VRAM 在 CUDA 上實測：RT-DETRv2 **97.7 MiB**、RF-DETR-Nano **106.0 MiB**。
  - **選圖錯兩次，都是打開圖才發現的**：第一版用「裸頭最多」排序，
    結果選出一群沒人戴安全帽的冬衣人群，38 個框裡只有 2 個合規，**整張全紅**，
    demo 唯一想展示的綠紅對比完全看不到；第二版改用平衡度排序但**濾的是標註數**，
    於是一張標註 8 個的圖畫出 22 個框（0.07 操作點下模型框數遠多於物件數），
    標籤糊成一團。現在濾的是**實際畫出來的框**。
    另有一張自動排序永遠第一、但畫面是「疑似傷者被吊掛在擔架上方」的圖，
    列入 `EXCLUDED_FRAMES` 並寫明理由——這種事沒有自動判準看得出來。
  - **驗證**：`uv run pytest tests/test_demo_gif.py` → 20 passed；
    變異測試 **13/13 killed**（含「用裸頭數排序」與「只看單一判定」這兩個真實錯誤）。
  - **③ 影片分頁第一次被執行**（2026-08-02）。它從發佈以來沒跑過。
    `annotate_video` 已從 Gradio callback 抽成模組層函式回傳 `VideoResult`——
    只能透過瀏覽器到達的 callback 就是沒人跑的 callback，而這個檔案裡
    每一段沒跑過的路徑最後都證實是壞的。
    **圖片與影片 × CPU 與 CUDA 四條路徑全部實跑通過**，加上「餵非影片檔」那條分支。
    CUDA 峰值 VRAM 讀到 **92 MiB**。
  - **順帶更正一個沒量過就寫下的數字**：docstring 原本寫「CPU 約 300 ms 一張」，
    實測（12 張 val、暖機後中位數）是 **204 ms**，CUDA 是 **20 ms**。
    `instructions_for_me.md` 原本寫 1.2 秒，差了 6 倍，已改。
  - **驗證**：`uv run pytest tests/test_demo.py` → 27 passed；
    變異測試 13/13（GIF 選圖）。全套 `1465 passed, 42 skipped`。
  - **驗證於**：`741fd6d` @ 2026-08-02

- [~] **M23** README ＋ `scripts/verify_readme.py` ＋ CI
  - **對應規格**：PUB-01 ~ PUB-05、PUB-10
  - **驗證**：`uv run python scripts/verify_readme.py` 通過——
    README 每張表的數字都能從 `results/` 的原始檔重算；
    README 正文（不是只有 Limitations）交代資料集標註缺陷與
    **「所有主張都是相對的」**；
    主動說明**為什麼是四組不是五組**；
    `grep -n "shields.io" README.md` → 零命中（不放靜態假 badge）；
    `.github/workflows/ci.yml` 至少做 `uv sync --locked` → `pytest` 且為綠；
    `uv.lock` 已提交
  - **本機可驗的四條全部實跑通過**（2026-08-01）：
    `verify_readme` → `PASS: every README number has a source and every disclosure is present`；
    `grep -c shields.io README.md` → `0`；
    `uv lock --check` → `Resolved 120 packages`（exit 0）；
    `git ls-files uv.lock` → 已追蹤。
    `ci.yml` 的四個步驟是 `uv sync --locked`、`pytest -q`、
    `verify_readme`、`check_forbidden_licences`。
  - **這個 `[~]` 只差一件，而且現在無法驗**：「CI 為綠」需要 workflow 真的在
    GitHub 上跑過一次，而 `git remote -v` 是空的——**這個 repo 還沒有 remote，
    CI 一次都沒執行過**。第一次 push 之後這條才驗得到。
    在那之前把它勾成 `[x]` 就是宣稱一件沒發生的事。
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
