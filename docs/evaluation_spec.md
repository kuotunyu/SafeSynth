# Evaluation Spec — 合規邏輯、指標與錯誤分析

> **Phase 2。** 對應里程碑見 [PLAN_PHASE2.md](../PLAN_PHASE2.md)。
> 協定層（防洩漏、五組定義、呈現規範）見 [experiment_protocol.md](experiment_protocol.md)。
> 訓練見 [training_spec.md](training_spec.md)。相關決策：[ADR-003](decisions.md#adr-003)。
> **所有門檻數值都在 [`configs/evaluation.yaml`](../configs/evaluation.yaml)，本文件只寫判定式與 config key。**

---

## 1. 合規邏輯（`src/inference/compliance.py`）

### 1.1 為什麼 `person` 不承重

原始構想是「由 Helmet／Head／Person 三者的空間關係推導每個人的戴帽狀態」。
[ADR-003](decisions.md#adr-003) 已否決這條路：`person` 只有 751 個實例、分布在 158 張圖，
且 SHEL5K 論文明指這個類別「標得太差」。
把合規判定架在一個 3% 覆蓋率且標註不可靠的類別上，整條主線會塌。

**改成由 `helmet`（戴帽的頭）vs `head`（裸頭）直接推導。**
這兩類合計 24,751 個實例，是資料集裡真正紮實的部分。

### 1.2 兩種模式，由 config 切換

**EVAL-01 — `compliance.mode` 有兩個合法值，由 Phase 1 的 Spike H1 決定用哪個。**

| 模式 | 適用前提 | 判定 |
|---|---|---|
| `class_direct`（**主定義**） | H1 證實 `helmet` 框的是**戴著安全帽的頭部區域**，且與 `head` 在同一人身上**互斥** | 偵測到的類別本身就是狀態。`helmet → COMPLIANT`、`head → NON_COMPLIANT`。**零空間推理、不需要 person 框** |
| `geometric_pairing`（fallback） | H1 顯示 `helmet` 只框帽殼、底下另有 `head` 共存 | `head` 為 COMPLIANT 若且唯若存在某個 `helmet` 滿足 [FILT-07](filtering_spec.md) 的幾何配對 |

**兩種模式都必須實作**，即使 H1 已經定案——因為 H1 的判定寫在 worklog 裡，
而讓程式碼保留兩條路徑並由 config 選擇，比事後回頭改程式碼誠實得多，
也讓「我們考慮過另一種語意」這件事是可驗證的。

**EVAL-02 — 幀層級指標**
```
compliance_rate = n_COMPLIANT / (n_COMPLIANT + n_NON_COMPLIANT)
```
`person` 偵測**只能**作為選配的分組提示（把同一個人的頭歸給同一個人），
**它的存在與否絕不可改變任何一個判定**。

**EVAL-03 — 強制單元測試：`person` 不可承重**
把所有 `person` 偵測從輸入刪光後重跑合規判定，**結果必須逐位元相同**。

這條測試是 EVAL-02 那句話的機械保證。沒有它，`person` 會在某次重構中偷偷變成必要輸入，
而且不會有人發現。

### 1.3 信心分數門檻

**EVAL-04** — 合規判定使用的偵測信心門檻 `compliance.score_threshold`
與 mAP 評測用的門檻**是分開的兩件事**：
mAP 掃過所有信心值積分，合規判定則需要一個單一的操作點。

門檻的選法：在 **Validation** 上掃描信心值，取讓 `bare-head recall` 最大化
且 `compliance precision` 不低於下限的那個點，**然後凍結**。
**絕不在 Test 上選門檻**——那等於用測試集調參。
選出的值與掃描曲線寫進 `reports/compliance_operating_point.md`。

---

## 2. 指標

### 2.1 主敘事（README 的主表只放這幾個）

**EVAL-05**

| 指標 | 為什麼是它 |
|---|---|
| **`AP_small` (helmet, head)** | 主敘事 #1。`small_distant` 佔了合成預算最大份額，這個數字直接檢驗「針對性合成」有沒有打中弱點 |
| **bare-head recall @ IoU 0.5** | 主敘事 #2。`head` 在圖片層級只出現在 18.4% 的圖裡，是安全上最關鍵也最稀少的訊號 |
| **hard-negative 子集的每圖誤報數** | hard negative 永遠不可能貢獻 recall，只會移動 precision（[COMP-24](synthesis_spec.md)），所以必須獨立成一個數字，不能被 mAP 稀釋 |
| `mAP50-95 (helmet, head)` | 綜合品質 |
| compliance precision / recall | 任務層級的最終產出 |

### 2.2 次要

**EVAL-06** — `mAP50`、per-class AP、`mAP_all3`、AP_medium／AP_large、
以及速度數字（見 [release_spec.md](release_spec.md)）。

### 2.3 ⚠️ `AP_small` 的定義陷阱

**EVAL-07 — `AP_small` 的面積門檻必須在「每張圖自己的原始標註座標」下計算。**

COCO 的 `AP_small` 定義是 `area < 32² = 1024 px²`。
但我們**在約 416 的原生解析度標註、在 640×640 訓練**。
若把預測與 GT 都放大到 640 再算面積，同一個物件的面積會變成約 `2.37` 倍，
**大量原本屬於 small 的物件會被歸到 medium**，`AP_small` 就變成在測量另一件事。

規則：
- COCO GT JSON 的 `area` 欄位一律是**原始標註座標**下的 `w * h`
- 模型輸出的框在進評測前先**映射回原始座標**
- 評測全程在原始座標系進行

⚠️ **「原始座標」是逐圖的，不是一個常數。** 影像**不是單一解析度**——
416×415 才是多數，另有 416×416／415×416／415×415（實測見
[DATA-25](data_protocol.md#data-25--影像不是單一解析度預測框必須逐圖映射)）。
所以映射一律讀該圖自己的 `width`／`height`，
**`scale_x` 與 `scale_y` 不一定相等**，不得用單一純量。

偏差只有 1 px，不會有任何東西報錯；但 `head` 平均約 34 × 34 ≈ 1,156 px²，
**正好卡在 small／medium 邊界（1,024）旁邊**，所以這點偏差會真的換桶。

這件事錯了**不會報錯**，只會讓主敘事指標安靜地失去意義。

參考尺度：`head` 平均約 34×34 ≈ 1,156 px²，正好卡在 small/medium 邊界附近——
所以這個資料集對面積定義**特別敏感**，更要小心。

**EVAL-08** — 同時報告各 size bucket 的**實例數**。
若 Test 的 small bucket 只有極少數實例，`AP_small` 的變動就是雜訊，必須說出來。

**判定式**：small bucket 要能承載主敘事，必須**同時**滿足兩個條件——
① 絕對數量 `n_small_primary >= metrics.small_bucket_min_instances`
② 相對佔比不低於各桶均分（`1 / n_buckets`）。

⚠️ **只用比例會通過一個統計上毫無意義的桶**：20 個實例的 split 裡，
small bucket 佔 100% 也還是 20 個框。比例檢定看不出這件事，絕對下限才看得出。
（實際的凍結 Test 有 2,054 個 primary 小物件，兩個條件都過。
但函式必須在假設情況下也判對，否則它只是恰好對。）

**噪音下限要分清楚是誰的**：單一組的 CI 半寬與**兩組差值**的半寬差了約 √2 倍。
報告要標明引用的是哪一個，且兩者都只用來辨認「明顯小到不值得討論」的差距——
真正的裁決工具是 EVAL-09 的 bootstrap。

### 2.4 統計呈現

**EVAL-09 — 對每一個主表數字提供不確定度。**
- 有 3 seeds 的組別（Real-only 與最佳 Filtered 組）報 **mean ± std**
- 只有 1 seed 的組別報單值，並在表格註明「單一 seed」
- **`person` AP 必須附對測試圖做 1,000 次重抽的 bootstrap 95% 信賴區間**
  （[EXP-03](experiment_protocol.md)），以及該 split 的 person 實例數與圖片數

**EVAL-10 — 「進步」的門檻**
一個組別要被宣稱優於另一個，差距必須大於雜訊。
在只有 1 seed 的情況下，**不得**用「差 0.3 個點」來宣稱勝出；
主張要嘛等 3-seed 的 mean±std，要嘛只描述方向並明說證據強度。

---

## 3. 評測腳本（`scripts/eval.py`，本機 4090）

**EVAL-11** — 本機執行（不燒 Colab 額度）。輸入是 `runs/<arm>/seed_<n>/` 的權重，
輸出 `results/detection_metrics.csv`（每列一個 arm × seed × 指標）。

**EVAL-12 — 所有表格數字一律從 raw 輸出重新聚合，不得抄訓練 log 上顯示的值。**
訓練期間印的驗證指標可能用了不同的門檻、不同的座標系或不同的子集。
主表的每一個數字都必須由 `scripts/eval.py` 在凍結的 Test 上重新算一次。

**EVAL-13 — 評測工具**：`pycocotools` 為正確性基準。
`faster-coco-eval` 可作為加速的 drop-in，但**必須先驗證兩者在同一份輸入上結果一致**
再拿它產出正式數字——「宣稱完全一致」是值得驗證的宣稱，不是可以直接相信的前提。

**EVAL-14 — 防洩漏自查**：`scripts/eval.py` 啟動時呼叫 `assert_test_untouched()`
（[DATA-20](data_protocol.md)），並斷言載入的權重來自的訓練資料清單中
**沒有任何 Test 影像 id**。

---

## 4. 錯誤分析

**EVAL-15 — FP/FN 對照 grid。**
取基線組（Real-only）與最佳組（Filtered Syn）在**同一批 Test 影像**上的預測，
產出並排比較圖，分成四類：

| 類別 | 意義 |
|---|---|
| **修好的 FN** | 基線漏掉、最佳組抓到 → 這是合成資料的功勞，最有說服力 |
| **修好的 FP** | 基線誤報、最佳組沒有 → hard negative 的功勞 |
| **新增的 FP** | 基線沒誤報、最佳組誤報了 → **合成資料的副作用，必須誠實呈現** |
| **兩組都錯** | 剩餘的難題，寫進 Limitations |

每一類各取若干張，**自己先打開檢視**再放進報告。

**EVAL-16 — hard negative 專項分析。**
在含 hard negative 的 Test 子集上（若 Test 沒有天然的 hard negative，
就用挖料器在 Test 上標出候選區域**僅供分析、不進訓練**），
比較四組的每圖誤報數。這是 `hard_negative` 情境唯一能被驗證的地方。

**EVAL-17 — 按情境切分的表現。**
把 Test 影像依「是否含小物件／是否擁擠／是否低光」分桶
（用 GT 的統計自動分桶，不用人工標），分別報各組的指標。
**這是檢驗「針對性」是否真的針對到的關鍵**：
若 `small_distant` 佔了 25% 的合成預算，但小物件桶的進步和其他桶一樣多，
那就不是「針對性」生效，只是「資料變多」生效——**這個發現要如實寫出來**。

**EVAL-18 — 若 synthetic 沒有提升**，照 [experiment_protocol.md §7](experiment_protocol.md)
的清單逐項檢查並如實報告。負面結果加誠實分析比挑選性報告有價值。

---

## 5. 驗證

| 檢查 | 方法 |
|---|---|
| 合規邏輯不依賴 person | EVAL-03 的單元測試：刪光 person 偵測後結果逐位元相同 |
| `AP_small` 定義正確 | 用一個已知面積的合成案例反推：構造一個 `area = 1000 px²` 與一個 `area = 1100 px²` 的 GT，斷言前者進 small bucket、後者不進 |
| 座標映射逐圖、不轉置 | 至少一個 fixture **非正方形**（[DATA-25](data_protocol.md#data-25--影像不是單一解析度預測框必須逐圖映射)）。全正方形的 fixture 無法區分 `scale_x` 與 `scale_y`，也無法抓到 `(h, w)` 寫成 `(w, h)` |
| 測試本身有鑑別力 | 對每條實作 `EVAL-*` 的函式做一次變異注入：改壞一個 token，測試**必須變紅**。綠燈不算通過（[K-19](troubleshooting.md)） |
| 數字可追溯 | `reports/` 每個表格數字都能由 `results/detection_metrics.csv` 重新聚合出完全相同的值 |
| 防洩漏 | `assert_test_untouched()` 通過；訓練資料清單與 Test id 交集為空 |
| 評測工具一致性 | `pycocotools` 與 `faster-coco-eval` 在同一輸入上的 mAP 差距為 0 |
| 操作點沒有用 Test 選 | `reports/compliance_operating_point.md` 的掃描曲線來源標明是 Validation |
