# Training Spec — RT-DETRv2 四組對照訓練

> 協定層見 [experiment_protocol.md](experiment_protocol.md)、指標見 [evaluation_spec.md](evaluation_spec.md)。
> **所有數值都在 [`configs/training.yaml`](../configs/training.yaml)，本文件只寫需求與判定式。**

---

## 1. 開工前的強制查證

**TRAIN-01 — 寫 notebook 之前必須先查證當前的訓練 API，不得憑記憶動手。**

RT-DETRv2 在 `transformers` 裡的訓練介面仍在演進，而**錯誤的 API 用法會安靜地產出爛結果**
（例如 label 格式錯了不會報錯，只會讓 loss 不收斂）。M15 開始時必須逐項確認並記進 worklog：

| 要查的事 | 為什麼 |
|---|---|
| 模型與 processor 的**確切類別名** | 版本間改過名 |
| checkpoint id、授權、參數量 | 授權要與 MIT repo 相容 |
| **label 的確切格式**（box 的座標慣例、正規化與否、欄位名） | 這是最容易錯又最不會報錯的一項 |
| 是否支援 `Trainer`，以及需要的 collate function | 決定 notebook 骨架 |
| 官方 example script 或 notebook 是否存在 | 有就照抄，不要自創 |
| 目前版本已知的 fine-tune 問題（loss NaN、eval 相關旗標等） | 先知道比事後除錯便宜 |
| 建議超參（lr、batch size、warmup、是否凍結 backbone） | 回填 `configs/training.yaml` |

查證來源與結論寫進 [decisions.md](decisions.md) 的一則 ADR，並附連結。

**TRAIN-02 — 若查證結果與本規格的假設衝突，以查證為準**，並更新本文件與 config，
在 worklog 記錄差異。規格是為了讓實作有依據，不是為了讓實作將就過時的假設。

### 1.1 2026-07-27 的查證結果（M15 時要重新確認一次）

**類別名——不要抄 HF 文件的 autodoc 範例區塊。**

`rt_detr_v2/` 目錄裡**根本沒有任何 image processing 檔案**，
所以 **`RTDetrV2ImageProcessor` 不存在**。但該 model 頁面的
`RTDetrV2ForObjectDetection` autodoc 範例仍寫著要 import 它，
而且連 checkpoint id 都給錯（`PekingU/RTDetrV2_r50vd`，正確的是小寫 `PekingU/rtdetr_v2_r50vd`）。
同一頁上方的 *Usage tips* 才是對的。

**一律用 Auto 類別**，讓 checkpoint 的 `preprocessor_config.json` 決定正確的 processor：
```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection
```

**`transformers` v5 改了 image processor 命名**（[ADR-006](decisions.md#adr-006)）：
`RTDetrImageProcessor` 現在**就是**快版（torchvision），慢版改叫 `RTDetrImageProcessorPil`，
`RTDetrImageProcessorFast` **仍存在但已 deprecated**（import 時只印一行警告，不會 ImportError）。
**2025 年寫的教學抄下來不會爆炸，會安靜地行為不同——那更難抓**（[ADR-014](decisions.md#adr-014) 實測）。

**三件從 checkpoint 的 `preprocessor_config.json` 讀到的事，違反任一項都會安靜壞掉：**

| 設定 | 值 | 意義 |
|---|---|---|
| `do_normalize` | **`false`** | RT-DETR **不做** ImageNet mean/std 正規化，只除以 255。config 裡的 `image_mean`／`image_std` 是**惰性的**。**自己加正規化會安靜地毀掉模型** |
| `size` | 已經是 640×640 | 與我們的目標解析度相同，**不需要覆寫** |
| `do_convert_annotations` | `true` | 這就是把 COCO `[x,y,w,h]` 絕對座標轉成**正規化 cxcywh ∈ [0,1]** 的地方，正是 loss 要的格式。**絕不要自己做這個轉換** |

**label 格式**（官方 docstring）：`list[dict]`，每個 dict 至少含
`class_labels`（`LongTensor`，長度 = 該圖的框數）與 `boxes`（`FloatTensor`，形狀 `(框數, 4)`）。
collate 時 `labels` 是**一個 list 不是 stack 起來的張量**。

**augmentation 的座標慣例**：albumentations 全程停留在
COCO `[x,y,w,h]` **絕對座標**（`bbox_params` 用 `format="coco"`），
最後由 image processor 轉成正規化 cxcywh。**中間不要自己碰 cxcywh。**

### 1.2 六個會安靜出錯的訓練設定

| 設定 | 為什麼非這樣不可 |
|---|---|
| **`eval_do_concat_batches=False`** | **強制**。官方 README 原文說這對偵測模型的正確評測是必要的。否則 `Trainer` 會試圖把長度不齊的 per-image label list 串起來，得到垃圾或直接崩潰 |
| **`ignore_mismatched_sizes=True`** | **強制**。80 類 → 3 類，分類頭必須重新初始化 |
| **`remove_unused_columns=False`** | 否則 `Trainer` 會把 transform 需要的欄位剝掉 |
| **`max_grad_norm=0.1`** | **不是** `Trainer` 的預設值 1.0。上游的 `clip_max_norm` 就是 0.1。常見的 NaN loss 回報都追溯到 LR 太高／裁切太鬆 |
| **`len(val) % per_device_eval_batch_size != 1`** | 餘數剛好是 1 時，`image_size` 會從 `tensor([H,W])` 塌成 `tensor([H])`，在 `collect_targets` 拋 `not enough values to unpack`。**啟動時斷言，一行防一個真實的崩潰** |
| **category id 重新映射為 `0..K-1`** | 「loss 漂亮下降但 mAP 很低」幾乎都是 COCO category_id 不連續造成的。同時要在 config 上設好 `id2label` / `label2id` |

### 1.3 backbone 的處理

**不要凍結 backbone**，改用 **0.1× 的 backbone LR 參數組**——這是上游作者的做法，
在這種有域偏移的 PPE 資料集上嚴格優於凍結。

`Trainer` 不提供 per-group LR，需要 subclass 覆寫 `create_optimizer()`
（三個參數組：backbone 用 0.1× LR、其餘可衰減參數用正常 LR＋weight decay、
norm 與 bias 用正常 LR＋weight decay=0）。

config 裡沒有 `freeze_backbone` 旗標，只有 `freeze_backbone_batch_norms`（預設已開，
凍結 BN 統計量，是 DETR 系的標準做法）。

**已知缺口**：上游用 `ModelEMA(decay=0.9999)`，約值 0.5–1 AP。`Trainer` 沒有 EMA。
第一版先不做，在報告中列為已知缺口，或之後用 `TrainerCallback` 實作。

---

## 2. 四組的資料組成

**TRAIN-03 — 四組定義**（見 [experiment_protocol.md §1](experiment_protocol.md)）：
`real_only` / `standard_aug` / `unfiltered_syn` / `filtered_syn`。
**沒有第五組**，理由見該處。

**TRAIN-04 — 四組吃的真實影像必須完全相同。**
合成資料只能是**增量**，不得替換或抽換任何真實影像。
驗證方式：四組的真實影像 id 清單排序後雜湊，四者必須相同。

**TRAIN-05 — ⚠️ `standard_aug` 組必須包含光度增強。**
因為合成資料帶有低光與 motion blur（[COMP-15](synthesis_spec.md)），
若基線組只有幾何增強，`filtered_syn` 的勝出就有一部分只是「多拿到一種 augmentation」，
主張會在第一個尖銳提問下崩掉。詳見 [EXP-01](experiment_protocol.md)。

增強清單與參數範圍在 `configs/training.yaml` 的 `augmentation`，
且**必須與 `configs/compose.yaml` 的 `postfx` 範圍相當**——
兩者的對照表要放進 Phase 2 的報告。

**TRAIN-06 — `unfiltered_syn` 與 `filtered_syn` 的合成張數必須相同。**
（[COMP-26](synthesis_spec.md)）否則會把「資料更多」與「資料更好」混為一談。
驗證：兩組的合成影像張數相等，且 unfiltered 那批是從同一個 pool 均勻抽樣（seed=42）。

**TRAIN-07 — 對齊訓練預算。**
四組盡量對齊 optimizer steps、batch size 與**真實樣本的曝光次數**。
因為第 3、4 組的資料集較大，同樣的 epoch 數會讓它們走更多 step——
要嘛固定總 step 數，要嘛明確記錄各組的 step 數差異並在報告中說明。
採用哪一種寫進 `configs/training.yaml` 的 `budget_alignment`。

---

## 3. Colab notebook 規格

**TRAIN-08 — 資料先解壓到 `/content/data` 再訓練。**
**絕不**直接從掛載的 Google Drive 讀圖訓練——Drive 的隨機讀取延遲會讓 GPU 大部分時間在等 I/O。

**TRAIN-09 — checkpoint 定期同步回 Drive** 的
`MyDrive/sdg-portfolio/02-safesynth-ppe/`。同步頻率在 config。

**TRAIN-10 — 必須支援斷點續跑。**
啟動時偵測既有 checkpoint 自動接續。Colab 會斷線，這不是選配功能。
**驗收方式是實測**：刪 checkpoint 跑一次、保留 checkpoint 跑一次，行為都要正確。

**TRAIN-11 — 每個平行 notebook 用唯一輸出目錄**：`runs/<arm>/seed_<n>/`。
四組可以開四個 notebook 平行跑，目錄不得互相覆蓋。

**TRAIN-12 — token 只從 Colab Secrets 讀**，notebook 內**不得有任何明文 token**。

**TRAIN-13 — 本機 1-step smoke test。**
在原生 Windows 的 4090 上以最小步數跑通並存出 checkpoint，確認能重新載回。
**smoke test 沒過不准上 Colab**——在 Colab 上除錯又慢又燒額度。

**TRAIN-14 — Colab 執行說明。**
執行說明必須具體到「照做就行」：notebooks 複製到 Drive 的哪個路徑、從 Drive 開啟、Runtime 選型、
需要的 Secrets 名稱、預估時數與 compute units、
跑完要下載哪些檔案放回 `results/colab/` 的哪個路徑。

---

## 4. 產出回收與盤點

**TRAIN-15 — 對照執行說明的預期產物逐項確認，缺檔就停下來列出清單。**
不要用假設硬做。

**TRAIN-16 — 所有表格數字一律從 raw 輸出（log / metrics 檔）重新聚合計算**，
**不要抄 notebook 畫面上顯示的值**。訓練期間印的驗證指標可能用了不同門檻或不同子集。

**TRAIN-17 — 記錄每組的實際訓練條件**：step 數、batch size、實際 epoch、
使用的 GPU 型號、耗時、消耗的 compute units。這些數字要進 README 的成本揭露
（[PUB-05](release_spec.md)）。

**TRAIN-18 — 訓練資料清單存檔。**
每組訓練時實際讀到的影像 id 清單要存成檔案，供 [EVAL-14](evaluation_spec.md) 的
防洩漏自查比對——斷言它與 Test image id 的交集為空。

---

## 5. Seed 策略

**TRAIN-19** — 四組先各跑 **1 seed**。
之後只對 **Real-only** 與**最佳 Filtered 組**補到 **3 seeds**，報 mean±std。

**條件性**：先看主表。**若 Filtered 組沒有提升，補 seed 不會改變結論**——
此時把額度留給錯誤分析，並在 worklog 記錄這個取捨。

補 seed 時各用獨立的 `runs/<arm>/seed_<n>/`，且**只有 seed 改變**，其餘設定完全相同。

---

## 6. 速度對照組

**TRAIN-20 — 不得使用 Ultralytics / YOLO。**
AGPL-3.0 會讓**我們 `import` 它的那支 Python 檔**成為衍生作品而必須同樣 AGPL，
與 MIT repo 直接牴觸。只在 README 註明授權**擋不住這件事**，程式碼還在 repo 裡。
[ADR-001](decisions.md#adr-001) 已因同一理由否決它。

驗證：`grep -rn "ultralytics" src/ scripts/ notebooks/` → **零命中**。

**TRAIN-21 — 改用 `Roboflow/rf-detr-nano`**（Apache-2.0，`transformers` 原生支援），
選型理由與被否決的選項見 [ADR-005](decisions.md#adr-005)。

邊際成本幾乎是零：它是 `transformers` 的一等公民，同一支 `Trainer` 腳本、
同一個 collator、同一套指標，**差別只有一個 checkpoint 字串**。

⚠️ **只能用 nano / small / medium / base / large**——**XL 與 2XL 是 PML-1.0 不是 Apache-2.0。**

範圍限縮：**一個**模型、**一個** seed、與 `real_only` 相同的 epoch 數，列為次要結果。
主表仍以 RT-DETRv2 為準，速度對照另立一表。

**無論如何都要報 RT-DETRv2 自己在 4090 上的端到端 latency**——
那才是關心 PPE 佈署的讀者真正想看的數字，而且是免費的。

速度數字必須附 **batch size、輸入解析度、dtype** 三項脈絡——
缺這三項的 FPS 沒有意義（[DEMO-03](release_spec.md)）。

---

## 7. Colab 預算

**TRAIN-22** — 方案是 **Colab Pro+，500 compute units／月**，
附帶**最長 24 小時的背景執行**（Pro 沒有這項）。
完整跑是 4 組 × 1 seed ＋ 2 組 × 2 個額外 seed = **8 次訓練**。

**2026-07-27 的估算**：8 次訓練用 **L4** 跑 50 epochs 約需 **113 CU**，
在 500 CU 的額度內綽綽有餘；即使全部拉到 100 epochs（約 226 CU）也吃得下。

**Runtime 選 L4**：它對整個 sweep 而言與 T4 **幾乎等價成本**（約 113 vs 116 CU），
但**牆鐘時間快約 2.4 倍**，有 24 GB 記憶體，而且**每一次訓練都能塞進單一 session**。
T4 跑合成組的 100 epochs 會超過約 12 小時的 session 上限。
A100 的 CU 成本高約 55%、速度快約 2.2 倍，只在趕時間或一直被搶佔時才用。

⚠️ **上表全部是外推估計（誤差約 ±40%），不是實測。**
Colab 上經常是 **dataloader 受限而非 GPU 受限**，T4 尤其如此。
**第一次跑完就把實際值回填 `configs/training.yaml`**——第一次的估計一定不準。

預估用量表在 `configs/training.yaml` 的 `budget`，每次執行前另給出實際的 per-run 估計。

**規則**：
- **實際跑之前先把預估用量算給使用者看**
- 若預估會超過額度，先給取捨選項（減 seed／減 epoch／換 Runtime）**再問使用者**，
  不要自行縮減規模
- 每次跑完記錄**實際**消耗，回填估計值——第一次的估計一定不準，第二次才會準

**Runtime 選型原則**：能用便宜的就用便宜的。
只有在明確算出「用更貴的 Runtime 總 units 反而更省」時才升級
（更快的卡每小時燒更多 units，但時數更短，兩者要相乘比較，不能只看單價）。

---

## 8. 防洩漏

**TRAIN-23** — Validation 與 Test **只用真實資料**，合成資料只加進 Train。

**TRAIN-24** — notebook 啟動時呼叫 `assert_test_untouched()`
（[DATA-20](data_protocol.md)），並斷言載入的訓練清單與 `test_blocklist.json` 交集為空。

**TRAIN-25** — Colab 上傳的資料包**不得包含 Test 影像**。
打包腳本要主動排除，並在打包後列印「本包含 N 張影像，其中 Test 影像 0 張」的斷言結果。

---

## 9. 驗證

| 檢查 | 方法 |
|---|---|
| API 用法正確 | TRAIN-01 的查證清單逐項完成並記進 ADR |
| 四組真實影像相同 | 四組真實影像 id 清單排序後雜湊相同 |
| 兩合成組等量 | `unfiltered` 與 `filtered` 的合成張數相等 |
| 基線含光度增強 | `configs/training.yaml` 的 `augmentation` 與 `compose.yaml` 的 `postfx` 對照表齊備 |
| 斷點續跑 | 刪 checkpoint／保留 checkpoint 各跑一次，行為正確 |
| 無明文 token | `grep -rnE "(hf_|sk-|gho_)[A-Za-z0-9_-]{20,}" notebooks/` → 零命中 |
| 目錄不互相覆蓋 | 四組的 `runs/` 路徑互異 |
| smoke test | 本機 1-step 跑通且 checkpoint 能重新載回 |
| 無 AGPL 依賴 | `grep -rn "ultralytics"` → 零命中 |
| 防洩漏 | `assert_test_untouched()` 通過；訓練清單 ∩ Test id = ∅；上傳包含 Test 影像 0 張 |
| 數字來源 | 主表數字由 raw log 重新聚合，非抄畫面 |
