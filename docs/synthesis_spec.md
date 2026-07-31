# Synthesis Spec — Cutout Bank & Composition Engine

> 對應里程碑 M7–M11、M13。相關決策：[ADR-001](decisions.md#adr-001) SAM2 路徑、[ADR-003](decisions.md#adr-003) `person` 處置、[ADR-004](decisions.md#adr-004) hard negatives。
> **所有數值都在 [`configs/compose.yaml`](../configs/compose.yaml)。本文件只寫判定式與 config key，不複製任何數值**，這樣規格與數值不可能漂移。
> 過濾規則見 [filtering_spec.md](filtering_spec.md)。

---

## 1. 方法論命題

這不是「多生一點資料」。命題是：

> **模型的弱點是可以列舉的，而合成資料可以針對性地補在弱點上；
> 且合成的 bbox 標註由生成流程本身產生，標註成本為零。**

六個目標情境對應六個已知弱點：

| 情境 | 對應弱點 | 主要指標影響 |
|---|---|---|
| `small_distant` | 遠距小物件 | **AP_small**（主敘事 #1） |
| `head_no_helmet` | 稀少的裸頭（圖片層級只有 18.4%） | **bare-head recall**（主敘事 #2） |
| `partial_occlusion` | 被器材／人遮擋 | per-class AP |
| `crowded` | 多人重疊 | AP、compliance recall |
| `low_light_blur` | 黃昏與運動模糊 | 全域穩健度 |
| `hard_negative` | 誤判成安全帽的黃色圓形物 | **每圖誤報數**（precision） |

與 copy-paste 增強（Ghiasi et al., *Simple Copy-Paste*, CVPR 2021）同源，
但差別在於**分布是被刻意塑形的**，而不是均勻地多貼一些東西。

**這條命題的成立與否，取決於一件事**：偵測器不能靠「貼上的痕跡」抄捷徑。
Spike H4 就是專門驗證這件事的，且它是放大生成量的硬閘門。

---

## 2. Cutout Bank（`src/synthetic/cutout_bank.py`）

### 2.1 兩趟式 SAM2 — 讓 34 px 小框可用的關鍵

**CUT-01 — 絕不用小框去 prompt 416×416 全圖。**

SAM2 的 mask decoder 在 256×256 的網格上輸出。一個 34 px 的框在 416×416 的圖上，
映射到 SAM2 的 1024×1024 輸入是 84 px，到 decoder 網格只剩約 **21 px**。
光是邊界量化就約佔物件的 5%。在這個尺度上談 mask 品質沒有意義。

**CUT-02 — Pass 1：全圖巡覽。**
對每張 Train 影像算一次 `model.get_image_embeddings()`，該圖的所有框重用同一份 embedding。

Pass 1 的產物**不是** bank，而是三件事：
1. **既有標註物件的 mask** — `compose.py` 在重算被遮擋的真實框時**必須**用到（COMP-09）
2. 校準工具（M6）需要的經驗分布
3. 回答 Spike H1

**CUT-03 — Pass 2：逐 crop 建 bank。** 只對通過 2.2 節閘門的候選執行：
```
ctx    = context_pad_frac * max(w, h)
crop   = 框往四周外擴 ctx，裁到影像範圍內
crop   = 補成正方形，短邊不低於 min_crop_side_px
resize crop -> resize_to (cubic)
prompt = 框映射到 crop 座標，input_boxes 為 XYXY
multimask_output = false
mask 反向映射：resize_to -> crop 邊長 -> 原圖座標
```
有效 mask 解析度提升數倍，量化不再是瓶頸，改由**影像本身的資訊量**設限——這才是誠實的極限。

**CUT-04 — Pass 2 刻意放棄 embedding 重用。** 每個候選 crop 各算一次 embedding，
換取小物件上數倍的有效解析度。兩趟合計 GPU 時間仍低於一小時。
**此取捨由 Spike H2 驗證後才定案。**

**CUT-05 — 推論設定**：bf16 autocast、`torch.inference_mode()`、
`multimask_output=false`。模型與 dtype 見 `configs/compose.yaml` 的 `sam2`。

### 2.2 來源實例閘門

**CUT-06** — 以下閘門在 SAM2 **之前**套用（都很便宜），依序執行。
**每一次拒絕都要寫進 `bank_rejects.jsonl` 並附原因**，讓漏斗可以對帳。

| 閘門 | 判定 | config key |
|---|---|---|
| G1 VOC 旗標 | `truncated == 0 and difficult == 0` | `cutout_bank.respect_voc_flags` |
| G2 硬下限 | 最短邊與面積都達下限 | `cutout_bank.hard_floor` |
| G3 偏好層 | 較高的尺寸門檻，多數貼上從這層取材 | `cutout_bank.preferred_tier` |
| G4 長寬比 | 落在 per-class 範圍內 | `cutout_bank.aspect_ratio` |
| G5 不貼邊 | 框與影像任何邊緣的距離達下限 | `cutout_bank.min_distance_to_image_edge_px` |
| G6 遮擋 | 被其他框遮住的比例低於上限 | `cutout_bank.max_occlusion_by_others` |

- **G1 的理由**：標註者已經告訴你這個框不好，這是免費訊號，用它
- **G5 的理由**：貼邊被切斷的物件，mask 會有一條**筆直的死邊**，那是最顯眼的貼上破綻
- **G6 的例外（重要）**：`person` 框**合理地**包含它自己的 `helmet` 與 `head`。
  因此判定遮擋時，`person` 候選只算**其他 `person` 框**；`helmet`/`head` 候選則**忽略 `person` 框**

**CUT-07 — 刻意偏好大實例。**
`small_distant` 情境**只縮小、不放大**（`compose.max_paste_scale` 對 person 是 1.0、
對 helmet/head 略高於 1）。
*理由*：放大一個低解析度 crop 會注入模糊特徵，讓偵測器學到「模糊 ⇒ 小物件」的捷徑。
拿大的乾淨素材縮小來製造遠距物件，**是正確方向而非妥協**。

### 2.3 Mask 清理與自動剔除

**CUT-08 — 清理流程**（自建，因為 transformers 的對應參數是 no-op，見 [ADR-001](decisions.md#adr-001)）：
1. logits 以 0.0 閾值化成二值
2. `cv2.connectedComponentsWithStats` → 保留最大連通元件，記錄 `second_cc_ratio`
3. `scipy.ndimage.binary_fill_holes` → 記錄 `hole_fill_ratio`
4. 形態學閉運算
5. **記錄外溢比例之後**才硬裁到 prompt box

**CUT-09 — 自動剔除判據。任一項不過就拒絕**（門檻見 `cutout_bank.mask_quality`）：

| 訊號 | 定義 |
|---|---|
| `sam_iou` | `outputs.iou_scores` |
| `obj_logit` | `outputs.object_score_logits` |
| `mask_to_box_coverage` | `\|M ∩ B\| / \|B\|`，**上下限都要**（見下） |
| `mask_outside_box_ratio` | `\|M \ B\| / \|M\|`（裁切前） |
| `second_cc_ratio` | 第二大連通元件面積 / 最大者 |
| `hole_fill_ratio` | 填洞後面積 / 原面積 |
| `solidity` | `\|M\| / \|convex_hull(M)\|` |
| `edge_touch_top` | mask 覆蓋框頂端那一列的比例 |
| `edge_touch_side` | 左右邊同理 |

三個非顯而易見的設計：

- **`mask_to_box_coverage` 的上限在做真工作**：它抓的是 SAM2 的退化失敗模式——
  直接回傳 prompt box 本身。那種結果的 `iou_scores` **看起來很高**，
  沒有上限的話會一路通過
- **`edge_touch_top` 有限制、`edge_touch_bottom` 刻意不限**：安全帽緊貼框頂代表帽冠被切掉；
  但安全帽在頸線處被切斷是**合理的**
- **`solidity` 分類別**：安全帽與頭近似凸形，人有四肢所以不是

### 2.4 儲存

**CUT-10 — 用 RGBA PNG ＋ JSONL manifest，不用 `.npz`。**
約 8,000 個 cutout × 約 5 KB ≈ 40 MB，載入速度無關緊要；真正重要的是
**人（和 Claude）可以直接打開資料夾用眼睛看**——CLAUDE.md 明訂「自己產的圖要自己打開檢視」，
`.npz` 不透明，會讓這條規則失效。

```
${data_root}/cutouts/
  helmet/<img_id>_ann<k>.png       RGBA，crop 原生解析度，alpha = 羽化後的軟遮罩
  head/... person/... hardneg/...
  _ctx/<img_id>_ann<k>_ctx.png     RGB，框 + padding，無 alpha：調和參考與預覽用
  bank_manifest.jsonl              每個「被接受」的 cutout 一行
  bank_rejects.jsonl               每個「被拒絕」的候選一行 + 原因（漏斗必須對得起來）
  previews/bank_<class>_grid_<k>.png
```
alpha 存**軟**值；硬 mask 可由 `alpha >= 128` 還原。
manifest 同時存 `lab_mean` / `lab_std` / `hf_noise_sigma`，
讓 `compose.py` 規劃調和時不必重新開啟像素。

**CUT-11 — cutout 記錄 schema**（每行一個 JSON）需含：
`cutout_id`、類別、來源影像 id 與 SHA256、**來源 split 與 `group_id`**、來源 bbox、
VOC 旗標、SAM2 設定與分數、mask 各項統計、外觀統計（Lab 均值/標準差、雜訊 sigma、主色相）、
檔案路徑、閘門結果、使用次數上限與計數、建置資訊（腳本 git sha、seed、時間）。

**來源 `group_id` 是必要欄位**——COMP-03 會用到它。

**CUT-12 — `person` 的特別處置**（[ADR-003](decisions.md#adr-003)）：
每個近似群最多取 `max_person_cutouts_per_group` 個；
報告必須同時列出 `n_person_cutouts` 與 **`n_distinct_person_groups`**，
後者才是真實多樣性。低於 `min_distinct_person_groups` 時，
`crowded` 情境改用 head + helmet 素材，**並在報告中明說**。

**CUT-14 — 素材本身必須看得出是個物件。**

**判定式**：`source_object_luma >= min_source_object_luma[class_name]`，
量的是 cutout **自己的像素**（alpha ≥ 128 的部分），在任何合成之前。
**參數**：`configs/compose.yaml` → `cutout_bank.min_source_object_luma`
**實作位置**：`src/synthetic/compose.py` 的 `_drop_illegible_source_material`，
在 bank 載入時執行；被排除的數量寫進 `summary.json` 的 `source_material_legibility`
**驗證**：`uv run pytest tests/test_compose.py -k CUT_14` ／ `-k source`

**為什麼 [FILT-15](filtering_spec.md) 蓋不到這件事**——這是本規格最容易被誤以為重複的一條：
FILT-15 量的是**合成結果**，而 Lab 調和在它之前執行。
實測案例 `001610_ann008186` 的素材亮度是 **8.5**（一團純黑剪影，沒有可還原的細節），
調和把它抬到合成圖上的 **45.4**，**通過了 FILT-15 的門檻**，
但它仍然是一塊沒有特徵的黑斑。
**調和搬動的是平均值，它不會生出細節。** 因此不可用的素材只能在貼上之前擋掉。

門檻與 FILT-15 相同（真實 Train 物件遮罩的每類 p1）。
在 7,255 個素材上實測排除 **102 個（1.40%）**：head 27、helmet 74、person 1。

---

## 3. 合成引擎（`src/synthetic/compose.py`）

### 3.1 主流程

**COMP-01**
```
compose(bg_image_id, scenario, seed):
  1. rng 由 sample_id 衍生的 seed 建立，並記錄
  2. 載入背景與它的「凍結後 Train COCO 標註」
     既有物件的 mask 取自 Pass 1；Pass 1 沒過 QC 的退回用框矩形，
     並記錄 existing_mask_source ∈ {"sam2", "box"}
  3. 由情境表決定貼上數量、類別配比、變換範圍
  4. 依約束抽 cutout（3.2）
  5. z-order：所有實例（貼上的 + 既有的）依 y_bottom 遞增排序，最後畫的最靠近鏡頭
  6. 依 z-order 逐一：warp -> 調和 -> 羽化 -> alpha 合成
  7. 由 mask 重算所有 bbox（3.5）
  8. 對整張圖施加後處理特效（3.6）
  9. 寫出影像 + 標註 + 記錄
```

### 3.2 背景與素材選取

**COMP-02** — 背景**只用 Train**。程序啟動先跑 `assert_test_untouched()`（[DATA-20](data_protocol.md)）。

資料集幾乎沒有空景（光 `helmet` 就覆蓋 4,581/5,000 張），所以「乾淨背景板」不可行。
**預設使用帶標註的背景並把它的真實標籤一路帶下去**——在這種標註密度下這是唯一選項，
而且保留了真實的場景脈絡。

**COMP-03 — 硬約束**（`configs/compose.yaml` 的 `compose`）：
- 來源影像 ≠ 背景影像
- **來源 `group_id` ≠ 背景 `group_id`** ← **這條最容易漏掉**。
  資料集源自影片：把第 N 幀的安全帽貼到第 N+1 幀，等於**製造一組近似對並安靜洩漏**
- 同一張合成圖內不重複使用同一來源影像（`crowded` 情境放寬）
- 每張背景最多產出 N 張合成圖（強迫背景多樣性，不要讓 pHash 去重扛全部）
- 每個 cutout 有使用次數上限；`person` 另有更嚴的上限（[ADR-003](decisions.md#adr-003)）

### 3.3 放置先驗

**COMP-04** — **不要均勻隨機放置。**
均勻放置會把安全帽放到天空裡，然後被下游過濾掉——**而這會讓存活下來的樣本產生偏差**
（只有僥倖合理的才留下）。建先驗只要約 30 行，卻能在過濾之前就消除約九成的不合理放置。

- **位置先驗**：每類一張 16×16 的正規化 `(cx, cy)` 直方圖（Laplace 平滑），
  取樣後在格內均勻抖動
- **尺度—深度先驗**：`log(框面積)` 對 `cy` 做 OLS；由 `cy` 取樣 `log_area` 再換算成縮放係數。
  免費得到「越靠下＝越近＝越大」
- **錨定放置**（覆蓋上述）：helmet 貼在 head 上、head 貼在 person 內時，
  位置由**父框推導**而非取樣

**Spike H5** 會把先驗畫成熱圖檢視；若過於發散就改以錨定為主。

### 3.4 變換與混合

**COMP-05 — 變換**（範圍見 `compose.rotation_deg`、`compose.max_paste_scale`）：
縮放上限刻意壓住（CUT-07）；旋轉幅度小（工地影像中頭部近乎直立，
大角度旋轉是最明顯的「假」線索）；水平翻轉可以，**垂直翻轉絕不**。
縮小用 `INTER_AREA`（正確的抗鋸齒降採樣），放大用 `INTER_LINEAR`。

**COMP-06 — 混合用高斯羽化 alpha。不用 Poisson，也不用硬 alpha。**

- **Poisson（`cv2.seamlessClone`）在這裡是錯的**：它在梯度域解 PDE，會**沖掉物件的絕對顏色**——
  而在這個資料集裡，安全帽的**色相**（安全黃／白／藍／紅）**就是類別訊號**。
  它還慢約兩個數量級，而且 ROI 碰到影像邊界時會直接失敗，而我們的放置經常碰邊
- **硬 alpha** 留下一條 1 px 階梯，偵測器可以直接把它當成「有接縫 ⇒ 這裡有物件」
- **羽化 alpha**：先侵蝕硬 mask（讓羽化吃進物件而不是吃進背景），
  再以隨貼上尺寸縮放的 sigma 做高斯模糊。sigma 公式與夾限見
  `compose.blending`

### 3.5 bbox 重算 — 本規格的正確性核心

**COMP-07 — 定義。** 對實例 *i*：`M_i` 是它在影像座標的硬 mask，
`V_i = M_i ∧ ¬(⋃_{j 畫在 i 之上} M_j)` 是可見部分。
`bbox_i = tight_bbox(V_i)`，`visible_fraction_i = |V_i| / |M_i|`。

**COMP-08 — 貼上的物件**：`visible_fraction` 低於門檻時，
**重抽放置位置重試**（上限見 `bbox.max_placement_retries`）；重試耗盡就**根本不貼這個實例**。
**絕不產出「看得見卻沒有標註」的正樣本。**
這是一條**放置約束**，不是「丟掉標籤」的規則——這個區別正是標籤語意能保持嚴密的原因。

**COMP-09 — 被貼上物件遮到的既有真實標註**（決定這條管線是幫忙還是幫倒忙的地方）：

| `visible_fraction` | 動作 | 理由 |
|---|---|---|
| 高（`existing_keep_original_above` 以上） | **保留原框不動** | 真實標註者不會為了輕微遮擋去重新收緊框；避免無謂的標籤擾動 |
| 中（`existing_recompute_above` 以上） | 改用 `tight_bbox(V_e)` | 符合這個物件實際上會被怎麼標 |
| 低 | **拒絕這次放置並重試；絕不刪除標註** | 安靜地刪掉一個真實物件的標註，等於**在真實像素上製造假陰性**——這是「合成資料反而讓模型變差」最可能的單一成因 |

`M_e` 取自 Pass 1。Pass 1 沒過 QC 時退回用框矩形，**並誠實記錄這個偏差**：
貼在框**角落**（那裡其實是背景不是物件）會讓 `visible_fraction_e` 被低估。
它偏向「拒絕」，是安全的方向。每個實例記 `existing_mask_source`，
讓報告能量化退回發生的頻率。

**COMP-10 — 硬不變式（違反就 crash，不是過濾）**：
```
真實標註輸出數 == 真實標註輸入數 − len(intentional_removals)
```
`intentional_removals` 只有在 `head_no_helmet` 的 helmet→head 替換時才非空。
*理由*：若這個不變式被違反卻只是「過濾掉這個樣本」，bug 就會藏在拒絕統計裡看不見。

**COMP-11 — 出界處理**：所有框裁切到影像範圍；
貼上物件裁切**前**的 `inside_ratio` 要達下限、裁切**後**的邊長要達下限，否則重試。

### 3.6 光度調和與後處理

**COMP-12 — 調和：CIELab 局部 Reinhard。**
來源統計取自 cutout 的 alpha 區域；目標統計取自**貼上區周圍的環帶**（排除貼上足跡本身），
有效像素不足時退回整張背景的統計。

**a/b 通道刻意只做半強度平均位移，且不縮放色度。** 這是有原則的設計不是偷懶：
安全帽的色相是類別訊號，**全強度色度匹配會把黃色安全帽拉向背景的色偏，
抹掉偵測器最需要的那個特徵**。半強度的**均值**位移足以修正
「這個 crop 來自偏藍的照片、被放進偏暖的照片」，卻不會替物件重新上色。
L 通道則給強處理（曝光不符是最顯眼的破綻，而且不帶類別資訊）。

**COMP-13 — 雜訊匹配。** 三行程式碼，回報很高：
在 416×416 下，**縮小後的 crop 乾淨得可疑**，這是僅次於邊緣的最強破綻。
量測背景環帶與前景的高頻雜訊標準差，前景較乾淨時補上差額的高斯雜訊（有上限）。

**COMP-14 — 調和在 pipeline 中的位置：每個實例 warp 之後、羽化之前。**
- 在 warp **之後**：縮放會改變雜訊統計（`INTER_AREA` 會去噪），必須量縮放後的值
- 在羽化**之前**：讓羽化混合的是「已經調和過的像素」，而不是把一個顏色不連續處抹開

**COMP-15 — 低光與 motion blur 在合成完成後、對整張圖施加。**

**這是正確性論證，不是方便**：
- 這些是**保持幾何**的運算 → 步驟 7 算出的每一個 bbox 都**維持完全有效**，
  零重新推導、零標籤漂移
- 反之，逐物件退化會在物件與場景之間製造**光度不一致**，讓偵測器有捷徑可抄，
  而且還得重新推導框
- 附帶好處：全域模糊與雜訊會順便蓋掉殘餘的貼上接縫，等於免費的調和

效果與參數見 `configs/compose.yaml` 的 `postfx`。

> ⚠️ **對 Phase 2 實驗設計的推論**：`+Standard Aug` 基線組**必須包含同等的光度增強**
> （gamma／亮度／雜訊／模糊）。否則 Filtered Synthetic 組的勝出有一部分只是
> 「它多拿到一種 augmentation」，主張會在第一個尖銳提問下崩掉。
> 已寫進 [experiment_protocol.md](experiment_protocol.md)。

### 3.7 情境

**COMP-16** — 六個情境的數量、尺度、位置限制、旋轉、目標可見比例與後處理機率，
全部在 `configs/compose.yaml` 的 `scenarios`。以下只記非顯而易見的機制。

**COMP-17 — `partial_occlusion`：直接求解遮擋，不要拒絕取樣。**
先貼目標，再把遮擋物（`person` cutout，或一條背景紋理帶代表鷹架／柱子）
放在能讓遮擋比例落入目標區間的偏移量上。一次到位，而不是試十幾次再挑。

**COMP-18 — `head_no_helmet` 的兩個子模式**（比例見 config）：
- **`paste_head`**：依位置先驗貼裸頭，優先放在既有 `person` 框的上半部，讓它落在身體上
- **`helmet_to_head_swap`** — **本管線價值最高的操作**：
  把豐富的類別**就地**轉成稀少的類別，且保留正確的場景脈絡。
  ```
  選一個既有的 helmet 標註 h（其 Pass 1 mask 需通過 QC）
  先用 cv2.INPAINT_TELEA 把 dilate(M_h) 區域抹掉        <-- 這一步是關鍵
  取一個 head cutout，縮放到與 h 的框相符（見下）
  貼上
  斷言殘留的 helmet 區域比例低於門檻（否則 crash）
  移除標註 h，新增一個 head 標註
  記錄 intentional_removals = [h.id]、swap_anchor_annotation_id = h.id
  ```
  **必須先 inpaint 再貼**，否則會留下帽緣殘影。在 416 下的約 48 px 區域上，
  Telea 又快又夠用，因為之後大部分會被蓋住。

  ⚠️ **尺寸必須繼承 anchor，不是只繼承位置。**
  實作曾經只設 `center_override` 而沒設 `target_bbox_xywh`，於是頭沿用了情境的
  通用 `scale_range`，貼出**大到不成比例**的頭：實測 `s42_011879` 用 **52×68**
  的頭取代 **24×30** 的安全帽，而同場景另外兩頂安全帽是 34×40 與 28×33。
  因為這裡通常沒有 `person` 框（全資料集只有 3.16% 的圖有），
  [FILT-08](filtering_spec.md) 的尺寸比例規則無從比對，這個錯誤不會被任何過濾器攔下。

  **縮放規則**：把 head cutout 的 alpha 框以幾何平均縮放到 h 的框。
  取整個框而不是只取寬度，是因為兩類的長寬比在真實資料上幾乎相同
  （head h/w 中位數 1.208、helmet 1.143，兩種規則差不到 3%）。
  無法做得更細——H1 已確認同一個人身上 helmet 與 head 互斥，
  **真實資料裡不存在可供校準的 (head, helmet) 配對**。
  實測結果：head_w / anchor_w 中位數 0.949、head_h / anchor_h 中位數 1.000。

**COMP-19 — `low_light_blur` 刻意重複計算**：除了它自己的專屬配額，
低光／模糊也會作為**修飾**套用到其他情境的一部分樣本上（比例見各情境的 `postfx_prob`）。
這樣得到的是「退化 × 其他每一個目標情境」的組合，比單獨一個退化桶更有價值。

### 3.8 Hard negatives

**COMP-20** — 素材**全部由程序生成**（[ADR-012](decisions.md#adr-012)
推翻了 [ADR-004](decisions.md#adr-004) 的「挖料為主」）。

- **程序生成**：安全色系的圓頂／橢圓／圓弧／管件，帶高光與邊緣陰影，
  **渲染在安全帽的典型尺寸與近方形長寬比**——夠像安全帽才算 *hard*。
  **必須用陰影調變真實背景紋理，不可平坦填色**。
  （唯一例外的判斷：真實安全帽本來就是光滑均勻的圓頂，所以在這個特定物件上
  「偏平」是寫實而非破綻。）
- **挖料已停用**。原實作（HSV 安全黃／橙窗 ＋ 輪廓面積 ＋ 圓度，再用外接框 prompt SAM2）
  與 H6 證據保留在 repo 內作為記錄。目視發現它挖到的**是顏色區域而不是物件**：
  細長的背景色帶、平坦的紋理方塊。原因見 ADR-012——
  COMP-21 的防護刻意排除安全帽幾何，與「只能找到顏色區域」的方法疊加後，
  在這個資料集上結構性地不可能產出 helmet-like 的負樣本

**COMP-20b — 合成方式與放置。**
- distractor 走**獨立的未標註合成通道**，在標註實例堆疊渲染完之後才貼。
  因為 FILT-10 本來就禁止它們與任何標註重疊，這一輪不會擾動 z-order
  或既有標註的可見度計算——改動面因此最小
- **放置限制在地面帶**。均勻隨機會把圓頂放到天空，那不是 hard negative——
  真實的黃色機具、三角錐、油桶都在地面上。
  （實測：均勻放置時 8 張只過 3 張，改地面帶後 12 張過 11 張）
- **distractor 走與標註貼上完全相同的光度管線**（[K-11](troubleshooting.md#k-11)）。
  原本它只做幾何變換＋硬 alpha 合成，**完全沒有**羽化、邊緣去汙、Lab 調和與雜訊匹配，
  而標註貼上四樣都有。使用者審查 `preview_hard_negative_p1` 時把每一個都判為
  「像後製的圖片」。客觀量測證實了這個判斷：以 Laplacian 變異數量表面紋理，
  真實安全帽 p50 = 1350.9，修前的 distractor 只有 **52.4**（約 1/26），修後 505.2

**COMP-20c — 接地陰影。**
在 distractor 底部畫一個模糊橢圓並乘性壓暗背景，讓它落在場景裡而不是浮在任意深度。
參數在 `configs/compose.yaml` → `hard_negatives.contact_shadow`。三個判定式：
1. 橢圓的**兩個半軸都以物件寬度為基準**——高度等同物件的陰影會被讀成第二個物件
2. 陰影在**畫面座標**上繪製並允許溢出 patch 矩形。patch 貼合物件邊界，
   畫在 patch 內的陰影會被物件本身完全蓋掉——存在於陣列裡，不存在於畫面上
3. 陰影在物件合成**之前**施加，且**不改動任何幾何**，所以標籤完全不受影響

⚠️ **「依深度的尺寸先驗」已量測後放棄，不要再提。**
以 17,815 個真實 Train 標註擬合 `log(min_side) = a + b·cy` 得
**b = −0.0350、R² = 0.0001**，分桶中位數由上而下是 28 / 27 / 22 / 23。
這個資料集**沒有可用的深度—尺寸關係**（416×416 的網路照片沒有一致的相機幾何），
硬加一條先驗只會讓 distractor 離真實分布更遠。

**COMP-21 — 挖料的三層防護**（挖料已停用，以下保留為記錄與 H6 簽核的依據）**。**
⚠️ 因為約 2/3 真實物件未標註（[data_protocol.md §1.3](data_protocol.md)），
**「這裡沒有標註」不等於「這裡沒有安全帽」**。天真的挖料會撈到真實但未標註的安全帽，
把它們當負樣本——等於**教偵測器抑制安全帽**。防護：
1. 與任何既有標註的 IoU 低於門檻
2. 必須**通不過**「像戴著的安全帽」測試：正下方沒有類頭部區域，
   且長寬比或面積落在安全帽經驗範圍之外
3. **凍結素材庫前強制人工過目一張 contact sheet**（`human_signoff_required`）

**Spike H6 是決策點**（見 §5）。

**COMP-22 — hard negative 完全不給標註，這是正確語意。**
理由完整寫在 [ADR-004](decisions.md#adr-004)：標籤空間是 `{helmet, head, person}`，
黃色水桶不屬任何一類，**沒有框本身就是「此區域無任何列出類別」的正面斷言**，
因此該圖是完整且正確標註的。加第四類會改變任務與 mAP 分母；設 ignore 區則與目的相反。

**COMP-23 — 合成期推論**：hard negative 的貼上**不得與任何保留的標註重疊**
（見 [FILT-10](filtering_spec.md)），也不得把任何標註的可見比例壓到門檻以下。
用一塊無標註色塊遮住真實標註物件，是**汙染標籤**而不是磨銳邊界。

**COMP-24 — 指標推論**：hard negative 影像永遠不可能貢獻 recall，只會移動 precision。
因此 **hard-negative 子集的每圖誤報數**要當成獨立的一個數字報告。

---

## 4. 生成量與兩個實驗設計陷阱

**COMP-25 — 基準 1× 與真實 Train 影像約 1:1。**
*理由*：copy-paste 增強的收益在 0.5×–2× 之間就會趨於平緩；超過 2× 則訓練分布向合成瑕疵傾斜，
而且「你只是做了更多 augmentation」這個質疑會變得無法回答。1:1 也讓五組對齊 optimizer steps 變簡單。

情境配比見 `configs/compose.yaml` 的 `scenarios.*.weight`。
`small_distant` 與 `head_no_helmet` 拿最大份額（對應兩個主敘事指標）；
`crowded` 被**刻意壓低**，因為 person 素材庫弱（[ADR-003](decisions.md#adr-003)）。

**COMP-26 — ⚠️ 陷阱一：filtered 與 unfiltered 必須等量。**
若生成 N 張、過濾後剩 M 張，直接拿「N 張 unfiltered」對「M 張 filtered」比較，
會把**資料量**和**資料品質**兩個變因混在一起。正確作法：
1. 持續生成直到 accepted 數量達標
2. `filtered_1x` = 那批 accepted
3. `unfiltered_1x` = 從**整個 pool** 均勻隨機抽**同樣張數**（seed=42）
4. 回報 pool 大小與接受率；可另外加一組「整個 pool」當補充列

**COMP-27 — ⚠️ 陷阱二：0.5×/1×/2× 要用巢狀子集。**
生成一次 2× 的 pool，給每個 accepted 樣本一個穩定的 rank（由 `sample_id` ⊕ seed 雜湊而來），
各尺寸取 rank 最小的前 k 個，並**依情境分層**讓配比在每個尺寸都相同。
這強制 `0.5× ⊂ 1× ⊂ 2×`，讓規模曲線變成乾淨的巢狀比較，
而不是三次獨立抽樣——否則抽樣雜訊會混進你正要報告的那條趨勢裡。

**COMP-28 — 輸出佈局**：像素**只寫一次**到 `${data_root}/synthetic/images/`；
發兩份 COCO JSON（`annotations_filtered.json` / `annotations_unfiltered.json`）依 `passed` 分割。
消融就變成「同一個生成器、不同的接受遮罩」，是最乾淨的比較，也不必複製目錄。

**COMP-29 — 每筆合成樣本的記錄 schema** 見 [filtering_spec.md](filtering_spec.md) §4
（provenance、每個 instance 的來源與分數、過濾結果、不變式檢查全部在同一筆記錄裡）。

---

## 5. 驗證性 Spike

### Spike H2 — SAM2 在 ~34 px 框上的品質（M7，卡住 bank）

**問題**：crop 放大的策略能不能救回小框？如果不能，`head` 素材庫會崩掉，
而 `head_no_helmet` 佔了生成預算的四分之一，會沒有素材。

**作法（約 45 分鐘，需 GPU）**：取 60 個框，依最短邊分成三組（很小／中／較大）。
每個框跑三種模式：全圖 prompt、crop 放大到 1024、crop 放大到 512。
記錄 `iou_scores`、coverage、連通元件數；產出三欄並排的比較 grid。

**產出**：確定採用哪一種模式、真正的尺寸下限（品質明顯崩掉的那一組），
以及校準後的 `sam_iou_min` / `min_object_score_logit`
（取「目視良好」那組的第 10 百分位）。

**決策規則**：若 crop 放大在最小那組仍無法產生可用 mask，
提高 `hard_floor`，並讓 `head_no_helmet` 改以 `helmet_to_head_swap` 為主
（因為 helmet 框較大、素材品質較好）。

### Spike H4 — 貼上痕跡的可偵測度（M11，**放大生成量的硬閘門**）

**問題**：偵測器會不會學到「羽化邊緣／乾淨補丁 ⇒ 這裡有物件」，
於是在真實 Test 上完全不受益？**整個專案的命題就繫在這件事上。**

**作法（約 60 分鐘）**：用當時的設定生成 300 張合成圖，
然後訓練一個小型二元分類器（梯度直方圖特徵 + 邏輯迴歸，或 ResNet18 跑幾個 epoch），
分辨「貼上的 patch」與「真實物件 patch」。

**決策規則**：
- AUC 高 → 貼上痕跡可被輕易偵測。**先修調和與羽化，不准放大生成量**
- AUC 接近隨機 → 通過閘門，可以進入 M13 的全量生成

這一小時直接為上萬張的生成降風險。**在它通過之前，合成總量不得超過 300 張。**

### Spike H6 — hard negative 挖料純度（M9）

**問題**：HSV ＋ 圓度挖出來的候選裡，有多少其實是**真實但未標註的安全帽**？

**作法（約 40 分鐘）**：對 200 張 Train 影像跑挖料器，
把分數最高的候選拼成 8×8 contact sheet，**人工數出其中真的是安全帽的數量**。

**決策規則**：真實安全帽比例超過 `max_tolerated_helmet_rate` →
**翻轉為程序生成為主**，挖料降為輔助且逐張人工複核。

### Spike H1 / H3 / H5

見 [data_protocol.md §3](data_protocol.md)（它們卡的是 split 凍結，排在更前面）。
H1 的結果會回過來決定 `helmet_to_head_swap` 的實作細節與
[FILT-07](filtering_spec.md) 是否有真實資料可校準。

---

## 6. 驗證

| 交付 | 驗證方式 |
|---|---|
| cutout bank | **零個** `src_image_id` 落在 Val/Test（比對 `test_blocklist.json`）；每個 RGBA PNG 都有 4 個通道且 alpha 非全 0 也非全 255；`bank_manifest.jsonl` 行數 == PNG 數；漏斗能從 `bank_rejects.jsonl` 完全重新聚合；同 seed 重跑 100 張 mask 一致 |
| cutout 目視 | `reports/figures/bank_<class>_grid.png` 疊在**洋紅色**背景上——背景滲漏與光暈無所遁形。**要自己打開看**：不能有背景殘留、不能有第二個物件入鏡 |
| compose | **COCO 自評測試**：`pycocotools` 載入產出的 GT 跟自己跑 `COCOeval` → mAP 必須剛好 `1.000`；解析式 bbox 單元測試（已知 cutout + 已知遮擋物，重算結果與閉式解差距在 1 px 內）；每個樣本的不變式（COMP-10）；同 seed 兩次產出影像 SHA256 相同 |
| 情境真的做到它宣稱的事 | `reports/synthetic_stats.md` 的「情境 × 類別 × 尺寸桶」交叉表：斷言 `small_distant` 真的產出了最短邊落在目標區間的框，而不是名字叫 small 但其實不小 |
| 目視 | `--n 32 --draw-boxes` 產出並**自己打開看**。沒有任何自動測試能取代這一步 |
