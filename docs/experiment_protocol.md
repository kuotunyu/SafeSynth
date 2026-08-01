# Experiment Protocol

> **這是實驗設計的協定層**：分組定義、防洩漏機制、指標與呈現規範、seed 策略。
> 這些規則**不會被 Phase 1 的實測結果推翻**，所以先寫死。
>
> 具體怎麼做見：[training_spec.md](training_spec.md)（`TRAIN-*`）、
> [evaluation_spec.md](evaluation_spec.md)（`EVAL-*`）、
> [release_spec.md](release_spec.md)（`DEMO-*`、`PUB-*`）。
> 里程碑見 [PLAN_PHASE2.md](../PLAN_PHASE2.md)。

---

## 1. 四組對照

| # | 組別 | 訓練資料 |
|---|---|---|
| 1 | **Real-only** | 真實 Train |
| 2 | **+ Standard Augmentation** | 真實 Train ＋ 傳統增強（**含光度增強**，見 EXP-01） |
| 3 | **+ Unfiltered Synthetic** | 真實 Train ＋ 未過濾合成 |
| 4 | **+ Filtered Synthetic** | 真實 Train ＋ 已過濾合成（主成果組） |

四組吃的**真實影像必須完全相同**，合成只能是增量。
各組盡量對齊 optimizer steps、batch size 與真實樣本曝光次數。
第 3 與第 4 組的合成資料**張數必須相同**（[COMP-26](synthesis_spec.md)）。

### 為什麼沒有第五組「Full-real 上限」

CLAUDE.md 的通用實驗鐵律列了第五組並標註「適用時」。**本專案不適用**：
那一組的用途是「當 Real-only 是刻意縮減的 few-shot 子集時，
拿完整真實資料當作能力上限來參照」。
SafeSynth 的 **Real-only 本來就吃全部真實 Train**，沒有更高的真實資料上限存在。

**README 要主動說明這件事**（[PUB-03](release_spec.md)），
否則讀者對照通用協定會以為漏做了一組。

---

## 2. 三條現在就必須寫死的規則

這三條若等到 Phase 2 才想，會讓整份結論站不住。

### EXP-01 — `+Standard Aug` 組必須包含同等的光度增強

⚠️ `low_light_blur` 情境會對合成圖施加 gamma／亮度／雜訊／motion blur
（[COMP-15](synthesis_spec.md)）。
**若基線組沒有對應的光度增強，第 4 組的勝出就有一部分只是「它多拿到一種 augmentation」**，
而不是「針對性合成有效」。這個主張會在第一個尖銳提問下崩掉。

因此第 2 組的增強清單**必須**涵蓋：幾何（翻轉、縮放、裁切、mosaic 級）**加上**
與 `configs/compose.yaml` 的 `postfx` 範圍**相當**的 gamma／亮度／雜訊／模糊。
兩者的參數範圍要寫進 Phase 2 的報告並列表對照。

### EXP-02 — 所有主張必須是相對的，永遠不能是絕對的

因為約 2/3 的真實物件在這個資料集裡**未標註**
（[data_protocol.md §1.3](data_protocol.md)），
絕對 AP 對**每一個類別**都被系統性壓低，precision 被系統性低估，
而且**模型越好被罰越重**（偵測到真實但未標註的物件會被判假陽性）。

因此：
- 只能說「第 4 組在同一個凍結 Test 上比第 1 組高 X 點」
- **不能**說「本模型達到 mAP 0.YY」
- 這一點要在 README 明白寫出來

五組對照協定**本質上就是相對比較**，所以把這件事講明白正是可信度的來源，
不是要道歉的弱點。

### EXP-03 — `person` AP 的呈現規範

照實報告，**絕不隱藏**，但每次呈現都必須同時附上：
1. 該 split 的 `person` 實例數與圖片數（Test 約 24 張圖、約 110 個實例）
2. 對測試圖做 1,000 次重抽的 **bootstrap 95% 信賴區間**
3. 常設註腳引用 SHEL5K（75,570 vs 25,502 個標註；`person` 類被描述為標註不良），
   說明 person AP 在此基準上**並不測量 person 偵測品質**

同時報 `mAP_all3` 與 `mAP_helmet_head`，**主張以後者為準**。
理由完整寫在 [ADR-003](decisions.md#adr-003)。

---

## 3. 防洩漏

- **Validation / Test 只用真實資料。** 合成資料只加進 Train
- **generator 與過濾器都不得接觸 Test。**
  機制：`splits/test_blocklist.json` ＋ `assert_test_untouched()`，
  **每一支生成腳本啟動時都必須呼叫**（[DATA-20](data_protocol.md)）
- **split manifest 必須先凍結**（含 seed=42、來源 SHA256、pHash 分群決定）**才能開始生成**
- **近似圖片同群同 split**（[DATA-16](data_protocol.md)），且合成時
  **來源 `group_id` ≠ 背景 `group_id`**（[COMP-03](synthesis_spec.md)）
- cutout bank 內**零個**來源影像落在 Val/Test
- **SAM2 的自動 mask 只用於合成素材，絕不當作 Test ground truth**

---

## 4. 指標

完整定義、計算方式與陷阱見 [evaluation_spec.md](evaluation_spec.md)。摘要：

**主敘事**
- **`AP_small (helmet, head)`** ← 主敘事 #1，最能反映「針對性小物件合成」的效果
- **bare-head recall @ IoU 0.5** ← 主敘事 #2
- **hard-negative 子集的每圖誤報數** ← hard negative 只會移動 precision，
  永遠不可能貢獻 recall（[COMP-24](synthesis_spec.md)），所以要獨立成一個數字
- `mAP50-95 (helmet, head)`
- compliance precision / recall

⚠️ **`AP_small` 必須在每張圖自己的原始標註座標下計算**（[EVAL-07](evaluation_spec.md)）。
影像在約 416 標註、在 640 訓練，若在 640 座標算面積，每個物件會膨脹約 2.37 倍，
大量原本屬於 small 的物件會被歸到 medium——主敘事指標會**安靜地**變成在測量另一件事。
且影像**不是單一解析度**（[DATA-25](data_protocol.md#data-25--影像不是單一解析度預測框必須逐圖映射)），
映射必須逐圖、兩軸分開。

**次要**
- per-class AP（`person` 依 EXP-03 呈現）
- `mAP50`、`mAP_all3`、AP_medium／AP_large
- 速度：單張 latency、FPS、VRAM（**必須附 batch size／解析度／dtype**）

**工具**：`pycocotools` 為正確性基準；`faster-coco-eval` 可作為 drop-in 加速版
（宣稱結果完全一致，值得驗證而非直接相信）。

---

## 5. Seed 策略

- 全組合先跑 **1 seed**
- 只對 **Real-only** 與**最佳 Filtered 組**補到 **3 seeds**，報 mean±std
- 合成資料本身的 seed 固定為 42，且每筆樣本記錄自己的衍生 seed

---

## 6. 合成量的消融

`0.5× / 1× / 2×`，用**巢狀子集**（[COMP-27](synthesis_spec.md)）：
`0.5× ⊂ 1× ⊂ 2×`，且情境配比在每個尺寸都相同。
這樣規模曲線是乾淨的巢狀比較，而不是三次獨立抽樣。

`filtered` 與 `unfiltered` 兩組**必須等量**（[COMP-26](synthesis_spec.md)），
否則會把「資料更多」與「資料更好」混為一談。

---

## 7. 若 synthetic 沒有提升

**如實報告並分析原因，不准挑選性隱藏實驗。**

要檢查的方向至少包含：Spike H4 的貼上痕跡可偵測度是否其實沒過關、
合成分布是否真的落在目標情境（`reports/synthetic_stats.md` 的交叉表）、
過濾門檻是否有某一條在做全部的事（[FILT-14](filtering_spec.md) 的敏感度表）、
以及基線組的增強是否已經涵蓋了合成資料提供的變異（EXP-01）。

負面結果加上誠實分析，比挑選性報告更有價值。
