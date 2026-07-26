# Filtering Spec — Geometric & Quality Rules

> 對應里程碑 M6、M12。生成端規格見 [synthesis_spec.md](synthesis_spec.md)。
> **所有數值都在 [`configs/filtering.yaml`](../configs/filtering.yaml)。本文件只寫判定式與 config key，不複製任何數值。**
> 每個門檻在 config 內都帶 `source: fixed | guess | calibrated` 標記；M6 的校準工具負責把 `calibrated` 的欄位換成實測值並翻轉標記。

---

## 1. 數學定義

對合成圖中的實例 *i*：`M_i` 是完整 mask、`V_i` 是可見 mask、`B_i` 是軸對齊外接框。

```
mask_to_box_coverage(i) = |M_i ∩ B_i| / |B_i|                        ∈ [0,1]
visible_fraction(i)     = |V_i| / |M_i|                              ∈ [0,1]
overlap_score(i)        = max_{j≠i} |B_i ∩ B_j| / min(|B_i|, |B_j|)  ∈ [0,1]    (IoMin)
overlap_iou(i)          = max_{j≠i} IoU(B_i, B_j)
inside_ratio(i)         = area(B_i^裁切前 ∩ 影像) / area(B_i^裁切前)
seam_energy_ratio(i)    = 貼上邊界「內側」2 px 帶狀區的平均 |∇I|
                          ÷ 貼上區內部的平均 |∇I|
```

**`overlap_score` 刻意用 IoMin 而不是 IoU。**
IoU 會嚴重低報「小框完全落在大框裡面」這種情形——而那正是
「安全帽被 `person` 框吞掉」與「重複貼疊在既有物件上」的幾何形狀。
兩者都要記錄：IoMin 抓包含關係，IoU 抓一般重疊。

---

## 2. 規則

樣本只要**任一條**規則觸發就被拒絕。第一條觸發的原因用於漏斗統計，
但**所有**觸發的原因都要記錄下來。

### FILT-01 — bbox 必須在影像範圍內
**判定**：`0 ≤ x1 < x2 ≤ W`、`0 ≤ y1 < y2 ≤ H`、邊長達下限、
且裁切**前**的 `inside_ratio` 達下限。
**參數**：`rules.bbox_in_bounds.{min_box_side_px, min_inside_ratio}`
**拒絕原因**：`OUT_OF_BOUNDS`
**實作**：`src/filtering/rules.py`，函式上方需有錨點 `# spec: FILT-01`
**驗證**：`uv run pytest tests/test_rules.py -k filt_01`
**狀態**：[x] M12 已實作並驗證

### FILT-02 — 標註物件必須大到是個真的目標
**判定**：`area(B_i)` 達下限。
**參數**：`rules.min_visible_area.min_area_px`（校準目標：真實框面積的 p1）
**拒絕原因**：`BOX_TOO_SMALL`
**驗證**：`uv run pytest tests/test_rules.py -k filt_02`
**狀態**：[x] M12 已實作並驗證

### FILT-03 — 物件必須有足夠比例是看得見的
**判定**：`visible_fraction(i)` 達 per-class 下限。
**參數**：`rules.visible_fraction.{helmet, head, person}`
**拒絕原因**：`LOW_VISIBLE_FRACTION`
**驗證**：`uv run pytest tests/test_rules.py -k filt_03`
**狀態**：[x] M12 已實作並驗證

### FILT-04 — 物件不得互相堆疊
**判定**：同類別的 `overlap_score` 與 `overlap_iou` 都在上限內。
**例外**：`person` 包含 `helmet` / `head` 的containment 是**正確的**，不算違規，須豁免。
**參數**：`rules.overlap.*`（校準目標：真實資料的 p99）
**拒絕原因**：`EXCESSIVE_OVERLAP`
**驗證**：`uv run pytest tests/test_rules.py -k filt_04`
**狀態**：[x] M12 已實作並驗證

### FILT-05 — mask 必須合理地填滿它的框
**判定**：`mask_to_box_coverage(i)` 落在 per-class 的 `[下限, 上限]` 內。
**上限同樣重要**：它抓的是 mask 退化成整個框的情形。
**參數**：`rules.mask_to_box_coverage.{helmet, head, person}`（校準目標：p1–p99）
**拒絕原因**：`BAD_MASK_COVERAGE`
**驗證**：`uv run pytest tests/test_rules.py -k filt_05`
**狀態**：[x] M12 已實作並驗證

### FILT-06 — SAM2 mask 品質（承接自 cutout bank）
**判定**：該實例所用 cutout 在建庫時通過了 [CUT-09](synthesis_spec.md) 的全部判據。
分數隨 provenance 一路帶進合成記錄，不重算。
**參數**：`cutout_bank.mask_quality.*`（在 `configs/compose.yaml`）
**拒絕原因**：`SAM2_MASK_REJECTED`
**狀態**：[x] M12 已實作並驗證

### FILT-07 — 戴著的 Helmet 必須位於 Head 上方的合理範圍

以 head 框 `(hx, hy, hw, hh)`、helmet 框 `(mx, my, mw, mh)` 正規化出五個量：

```
dx        = (helmet.cx - head.cx) / head.w      橫向偏移，以 head 寬為單位
dy        = (head.y0  - helmet.cy) / head.h     正值 = helmet 中心在 head 上緣之上
overlap_y = (helmet.y1 - head.y0)  / head.h     helmet 底邊探入 head 框的深度
r_w       = helmet.w / head.w
r_h       = helmet.h / head.h
```

**判定**：以下**全部**成立——
`|dx| ≤ k_x`、`dy_min ≤ dy ≤ dy_max`、
**`overlap_y_min ≤ overlap_y ≤ overlap_y_max`**、
`r_w` 與 `r_h` 各在範圍內、`IoU(helmet, head) ≥ 下限`。

**「合理範圍」的確切定義**就是這組不等式，白話是：
*helmet 的底邊探入 head 框頂端的某個深度區間內，
helmet 中心落在 head 上緣的上下某個範圍，橫向偏移不超過 ±某個 head 寬。*

**`overlap_y_min` 就是「接觸測試」**：低於它代表安全帽**根本沒碰到頭**，
也就是**漂浮安全帽**。`overlap_y_max` 則抓「安全帽吞掉整張臉」。

**參數**：`rules.helmet_above_head.*`
**拒絕原因**：`FLOATING_HELMET`（`overlap_y` 過低或 `|dx|` 過大）、
`HELMET_HEAD_MISALIGNED`、`HELMET_SWALLOWS_HEAD`

**⚠️ 校準的前提取決於 Spike H1。**
若 H1 證實 `helmet` 與 `head` 在同一人身上**互斥**（約 2× 的面積比強烈暗示如此），
則**真實資料裡根本沒有 (head, helmet) 配對**可用來校準這六個數字。
那種情況下改用「相對於 `person` 框的 head 位置分布」來校準，
並在報告中註明：**本規則只治理合成的「戴著」構圖，不對應任何真實資料模式。**
否則（helmet 只框帽殼、底下另有 head），就從真實配對取 p1–p99 再放寬 10%。

**驗證**：`uv run pytest tests/test_rules.py -k filt_07`
**狀態**：[x] M12 已實作並驗證

### FILT-08 — Person / Head / Helmet 尺寸比例
**判定**：當 helmet/head 框被 person 框包含（包含比例達門檻）時，
面積比、寬度比、以及 head 在 person 框內的垂直位置都要落在範圍內。
**參數**：`rules.size_ratio.*`
**拒絕原因**：`BAD_SIZE_RATIO`
**校準備註**：起點取自人體比例（頭約佔身高 1/7 ⇒ 面積比 0.02–0.06），
但**上界刻意放寬**，因為工地場景的 `person` 框常常是半身裁切，會把比例推到 0.15–0.25。
751 個真實 person 實例足以算出誠實的百分位——**用實測，不要用這裡的先驗值**。
**驗證**：`uv run pytest tests/test_rules.py -k filt_08`
**狀態**：[x] M12 已實作並驗證

### FILT-09 — 穿模與接縫瑕疵
**判定（兩項）**：
1. **深度不一致的探出**：某個貼上物件與既有物件的 mask 交集比例落在一個**小的**區間內，
   **同時**該貼上物件在 z-order 上位於後方。
   *理由*：真實的遮擋要嘛是大量的、要嘛是零。從後方只探出一小塊，
   代表它出現在幾何上不可能的位置
2. **接縫能量**：`seam_energy_ratio` 在上限內
**參數**：`rules.clipping_artifact.*`
**拒絕原因**：`CLIPPING_ARTIFACT`、`SEAM_ARTIFACT`
**接縫門檻的校準方法**：量**真實**物件邊界的同一個統計量，取其 p95。
那才是這個資料集的自然分布——用猜的會系統性偏離。
**驗證**：`uv run pytest tests/test_rules.py -k filt_09`
**狀態**：[x] M12 已實作並驗證

### FILT-10 — Hard negative 不得汙染真實標註
**判定**：hard negative 與任何標註的 IoU 在上限內，
且不得把任何標註的 `visible_fraction` 壓到 [FILT-03](#filt-03) 的下限以下。
**參數**：`rules.hard_negative_no_overlap.max_iou_with_annotation`
**拒絕原因**：`HARD_NEGATIVE_OVERLAPS_ANNOTATION`
**理由**：用一塊**無標註**的色塊遮住一個真實標註物件，是**汙染標籤**而不是磨銳決策邊界
（[ADR-004](decisions.md#adr-004)）。
**驗證**：`uv run pytest tests/test_rules.py -k filt_10`
**狀態**：[x] M12 已實作並驗證

### FILT-11 — 近似樣本去重
**判定（三項）**：
1. 與任何**已接受**的合成影像的 pHash Hamming 距離高於門檻
2. 與**所有 split 的任何真實影像**的 pHash Hamming 距離高於門檻
3. **「什麼都沒發生」防護**：改動像素比例達下限（防止所有貼上都太小或被完全遮住）
**參數**：`rules.phash_dedup.*`
**拒絕原因**：`NEAR_DUPLICATE_SYNTHETIC`、`NEAR_DUPLICATE_REAL`、`NO_CHANGE`
**預期**：因為合成圖共用背景，碰撞會很多——**那是機制正常運作**，
也正是 `compose.max_composites_per_background` 存在的理由（在上游就分散背景）。
**驗證**：`uv run pytest tests/test_rules.py -k filt_11`
**狀態**：[x] M12 已實作並驗證

### FILT-12 — 不變式（斷言，不是過濾）
以下違反時**必須讓程式 crash**，不可以只是拒絕該樣本：
- 真實標註輸出數 == 輸入數 − `len(intentional_removals)`（[COMP-10](synthesis_spec.md)）
- z-order 與 `y_bottom` 排序一致
- 沒有面積為 0 的標註
- `assert_test_untouched()` 在啟動時通過

**參數**：`assertions.*`
**理由**：這些違反代表 `compose.py` 有 bug。若只是「過濾掉這個樣本」，
**bug 會藏進拒絕統計裡看不見**，然後你會以為過濾器很嚴格，其實是生成器壞了。
**驗證**：`uv run pytest tests/test_rules.py -k filt_12`
**狀態**：[x] M12 已實作並驗證

### FILT-13 — filtered / unfiltered 兩版輸出
**判定**：像素**只寫一次**到 `${data_root}/synthetic/images/`；
發兩份 COCO JSON 依 `filter.passed` 分割。
兩組**必須等量**（見 [COMP-26](synthesis_spec.md)）。
**理由**：一份磁碟拷貝，且消融變成「同一個生成器、不同的接受遮罩」，是最乾淨的比較。
**驗證**：`set(filtered_ids) ⊆ set(pool_ids)`；兩份 JSON 的 `images` 指向同一批檔案。
**狀態**：[~] 由 M13 產生等量 filtered / unfiltered COCO 交付物

### FILT-14 — 門檻敏感度表
**判定**：對每一個門檻**獨立**做 ±20% 擾動並重算接受率。
若任一門檻的 ±20% 讓接受率變動超過 `sensitivity_alarm_points` 個百分點，
**代表那個門檻一個人扛了整個過濾器，必須真正校準而不能靠猜**。
**參數**：`sensitivity_alarm_points`
**理由**：這條檢查專門用來找出那個暗中主宰一切的數字。沒有它，
一堆看起來合理的門檻裡可能只有一個在做事，而你不會知道是哪一個。
**驗證**：`reports/threshold_sensitivity.md` 存在，且沒有任何門檻超標
（超標就必須先校準或在報告中明確說明理由）。
**狀態**：[x] M12 已實作並驗證

---

## 3. 拒絕原因 enum

`rules.py` 只能產出 `configs/filtering.yaml` 的 `reject_reasons` 清單裡的字串。
每一筆被拒記錄的 `reject_reasons` 必須非空，且所有值都在 enum 內。

---

## 4. 每筆樣本的記錄 schema

`${data_root}/synthetic/records.jsonl`，一行一張合成圖。必須包含：

| 區塊 | 內容 |
|---|---|
| 識別 | `sample_id`、`schema_version`、`created_at` |
| generator | 腳本路徑、**git sha**、**config 檔的 sha256**、`seed`、`rank`（巢狀子集用）、`scenario` |
| postfx | 實際採用的 gamma／gain／雜訊 sigma／白平衡／模糊核長與角度／JPEG 品質 |
| background | 影像 id、sha256、split、**`group_id`**、既有標註數 |
| output | 檔名、sha256、寬高、pHash、與已接受樣本的最小 pHash 距離、改動像素比例 |
| instances[] | 每個實例：`kind ∈ {pasted, existing, hard_negative}`、類別、標註 id、cutout id、**來源影像 id 與 `group_id`**、來源 bbox、變換參數（scale／rotation／hflip／貼上座標／z_index／y_bottom）、調和參數、最終 bbox、**各項分數**、`kept` |
| | `existing` 實例另需 `existing_mask_source ∈ {sam2, box}` 與 `bbox_xywh_original` |
| | `hard_negative` 實例的 `class_name` 與 `annotation_id` 為 null、`annotated: false`、`negative_source` |
| pairs[] | (helmet, head) 配對的 `dx` / `dy` / `overlap_y` / `r_w` / `r_h` / `worn_helmet_ok` |
| filter | `passed`、`reject_reasons[]`、每條規則的布林結果、**`thresholds_sha256`** |
| invariants | `n_real_ann_in`、`n_real_ann_out`、`intentional_removals[]` |

`thresholds_sha256` 讓你事後能證明某一批樣本是用哪一組門檻篩的——
門檻在校準後會變，沒有這個欄位就無法重現。

---

## 5. 預覽圖（給人抽查用）

### PREV-01 — 每情境一張預覽 grid
`reports/figures/preview_<scenario>.png`，格數見 `preview.grid_rows/cols`。
每格畫框、標類別、標關鍵過濾分數、標 `sample_id`（這樣使用者回饋時能精確指出是哪一格）。

### PREV-02 — hard negative 的預覽圖**不畫框**
並附說明文字：**這是刻意的，由構造保證正確**——它們本來就沒有標註（[ADR-004](decisions.md#adr-004)）。
若沒有這行說明，看的人會以為是 bug。

### PREV-03 — 通過 vs 被拒 並排比較
各取 N 張並排，被拒的要在下方印出拒絕原因。
**若一個被拒的樣本看起來明明沒問題，那就是門檻錯了**——
去改門檻，並把這次改動記進 [decisions.md](decisions.md)。

### PREV-04 — cutout contact sheet 疊在洋紅色背景上
`preview.cutout_backdrop_rgb`。背景滲漏與光暈在洋紅上無所遁形，在白或黑上看不出來。

### PREV-05 — 所有預覽圖都要**自己先打開檢視**再交給使用者
CLAUDE.md 的工作方式明訂。不合理就先修，不要把明顯有問題的圖丟給使用者。
要使用者看哪幾張、怎麼回饋，寫在 `instructions_for_me.md`。

---

## 6. 校準工具（M6，Spike H7）

**一支腳本，把上面所有 `calibrated` 的門檻一次算出來。**

輸入：凍結後的 Train split。
輸出：每一個幾何量的 `[p1, p5, p50, p95, p99]` 表，寫進 `reports/calibration.md`，
並把對應數值回填 `configs/*.yaml`、把 `source` 從 `calibrated`（待填）翻成實測值。

要算的量至少包含：
per-class 的 `mask_to_box_coverage`、框面積、最短邊、長寬比、solidity；
helmet/head 在 person 框內的包含比例與垂直位置；
**真實物件邊界的 `seam_energy_ratio`**（這是 FILT-09 門檻的唯一正確來源）。

**為什麼這是一個獨立里程碑**：目前 `configs/*.yaml` 裡大量門檻標著 `source: guess`。
在全量生成（M13）之前，每一個還是 `guess` 的門檻都必須被明確列進
`reports/filter_report.md`，讓報告能誠實說出「這幾個數字是猜的」。
