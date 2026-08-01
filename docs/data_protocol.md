# Data & Split Protocol

> 對應里程碑 M2–M5。**這份協定凍結之前，一張合成圖都不准生。**
> 相關決策：[ADR-002](decisions.md#adr-002) 環境與儲存佈局、[ADR-003](decisions.md#adr-003) `person` 類處置。

---

## 1. 資料集事實表（已查證，2026-07-27）

| 項目 | 值 | 來源 |
|---|---|---|
| 授權 | **CC0 1.0 Universal**（公共領域，可自由衍生與發佈） | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)、[Roboflow](https://public.roboflow.com/object-detection/hard-hat-workers)、[Dataset Ninja](https://datasetninja.com/safety-helmet-detection) |
| Kaggle handle | `andrewmvd/hard-hat-detection` | 同上 |
| 影像數 | **5,000** | Dataset Ninja、HF Voxel51、SHEL5K 論文三方一致 |
| 標註物件總數 | **25,502** | Dataset Ninja（與 per-class 加總完全吻合） |
| 影像格式／解析度 | PNG，⚠️ **不是單一解析度**——見 [DATA-25](#data-25--影像不是單一解析度預測框必須逐圖映射) | 上游文獻寫 416 × 416（[SHEL5K 論文](https://www.mdpi.com/1424-8220/22/6/2315)），**實測不符** |
| 標註格式 | PASCAL VOC XML，每圖一份 | |
| 預設 split | **無** | Dataset Ninja 明確說明 |
| 下載大小 | **約 1.2–1.5 GB** | 來源不一致（1.22 GB vs 1.33 GB），M2 實測後回填 |

### 1.1 類別分布

| 類別 | 實例數 | 出現於幾張圖 | 圖片佔比 | 平均每圖 |
|---|---|---|---|---|
| `helmet` | **18,966** | 4,581 | 91.6% | 4.14 |
| `head` | **5,785** | **920** | **18.4%** | **6.29** |
| `person` | **751** | **158** | **3.16%** | 4.75 |

`18,966 + 5,785 + 751 = 25,502` ✓（SHEL5K 論文寫 25,501，差 1，視為誤植；
M2 實測時記錄差異但不因此中止）

**校正一個常見的誤述**：「Head-without-Helmet 很稀少」在**實例層級並不成立**——
`head` 有 5,785 個，佔全部標註的 22.7%。真正稀少的是**圖片層級**：只有 920/5,000 張圖含 `head`。
而且這 920 張裡平均每張有 6.29 個頭——**有頭的圖是擁擠的**，這對 `crowded` 情境反而是好消息。
文件與 README 一律用這組精確數字，不要沿用模糊敘述。

### 1.2 平均物件面積 — 兩個讀數不一致，待實測

⚠️ 同一個 Dataset Ninja 頁面在兩次查詢中回報了不同的數字：

| 類別 | 讀數 A | 讀數 B |
|---|---|---|
| `helmet` | 1.32% | 5.35% |
| `head` | 0.66% | 4.06% |
| `person` | 7.16% | 32% |

兩者相差約 4 倍，推測是不同欄位定義（單一物件平均面積 vs 該類別總覆蓋面積之類）。
**不採信任何一方**。`prepare_data.py` 直接從 XML 算出真值並寫入 `reports/class_distribution.md`。

**可確定的定性結論**（兩個讀數都支持）：三類都偏小、**`head` 最小、`person` 最大**。
以讀數 A 換算約為 helmet ≈48×48 px、head ≈34×34 px、person ≈111×111 px。
這是**小物件問題**，`AP_small` 會主導指標，也是 `synthesis_spec.md` 兩趟式 SAM2 的存在理由。

### 1.3 ⚠️ 已知標註缺陷（影響整個評測設計）

[SHEL5K 論文](https://www.mdpi.com/1424-8220/22/6/2315)（Sensors 2022）**重新標註了同樣這 5,000 張圖**：

| | 原版（本專案使用） | SHEL5K |
|---|---|---|
| 標註數 | **25,502** | **75,570** |
| 類別數 | 3 | 6 |

**約 2/3 的真實物件在原版是未標註的。** 論文並明確指出 `person` 類「標得太差」——
許多圖中明顯有人卻沒有 `person` 框。

**對本專案的四個推論**：

1. **絕對 AP 在這個基準上沒有意義。** 模型偵測到一個真實但未標註的物件會被判假陽性，
   precision 被系統性低估，且**模型越好被罰越重**。
2. 因此**所有主張必須是相對的**——同一個凍結 Test 上 A 組 vs B 組——永遠不能是絕對值。
   五組對照協定**本質上就是相對比較**，所以把這件事講明白正是可信度的來源，不是要道歉的弱點。
3. `person` 的處置見 [ADR-003](decisions.md#adr-003)：三類保留，但合規邏輯不依賴它，
   指標呈現有專門規範。
4. **這也是 hard negative 挖料的風險來源**：「這裡沒有標註」不等於「這裡沒有安全帽」。
   見 [ADR-004](decisions.md#adr-004)。

### 1.4 溯源與授權鏈

```
Northeastern University, China
  └─ Harvard Dataverse  doi:10.7910/DVN/7CBGOS
       └─ Roboflow "Hard Hat Workers"（7,041 張，原始解析度，CC0 1.0）
            └─ MakeML 鏡像
                 └─ Kaggle andrewmvd/hard-hat-detection（5,000 張 @ 416×416，CC0 1.0）
                      ├─ SHEL5K（Mendeley 9rcv8mm682）同樣 5,000 張，重標 75,570 個標註
                      ├─ HF Voxel51/hard-hat-detection
                      └─ Dataset Ninja "Safety Helmet Detection"
```

- **授權鏈全程 CC0 1.0**，衍生物可公開發佈
- **不是** SHWD / `njvisionpower/Safety-Helmet-Wearing-Dataset`（那是另一個資料集：
  7,581 張、類別為 `hat`/`person`、MIT 授權）。任何說這份 Kaggle 資料集是 SHWD 的說法都是錯的
- **措辭注意**：Kaggle 這份本身已經是 7,041 張原始解析度來源的「416×416 縮放後 5,000 張子集」。
  它的「原始」只是相對於 Roboflow 的**再增強**版本而言。公開發佈時寫
  「the Kaggle release of the Northeastern University China Hard Hat Workers dataset」，
  不要寫「the original dataset」

---

## 2. 需求

### 2.1 下載與校驗（M2）

**DATA-01** — 用 `kagglehub.dataset_download` 下載，**不得 `import kaggle`**（見 [ENV-04](environment.md)）。
首次執行從回傳路徑解析出版本號 N，寫回 `configs/paths.yaml` 的 `dataset.pinned_version`，
之後一律用 `andrewmvd/hard-hat-detection/versions/<N>` **釘住版本**。
*理由*：上游若重新上傳，凍結的 split manifest 就失去意義。
**驗證**：`configs/paths.yaml` 的 `pinned_version` 非 null，且 manifest 內記錄同一個 N。

**DATA-02** — 記錄來源指紋：下載的壓縮檔 SHA256 寫入 `splits/source_checksums.json`。
**不要用 `--unzip`**（它會刪掉壓縮檔，就拿不到雜湊了），自己用 `zipfile` 解壓。
**驗證**：`splits/source_checksums.json` 存在且含 archive 的 SHA256。

**DATA-03** — 事實斷言。解壓後逐項比對第 1 節：
影像數 = 5,000；標註數 = 25,502（容許 ±1，記錄差異）；
per-class 實例數 = 18,966 / 5,785 / 751；per-class 圖片數 = 4,581 / 920 / 158。
**不符就停下來報告，不要自行調整。** 不符代表 Kaggle 重新上傳過，整份協定需要重新檢視。
**驗證**：`uv run python scripts/prepare_data.py --verify` 全數通過，不符則非零退出。

**DATA-04** — 檔案配對不得靠索引運算。遞迴 glob `*.xml` 與 `*.png`，用 `Path.stem` 配對，
斷言兩集合相等，不等就印出對稱差集。
*理由*：檔名樣式（推測為 `hard_hat_workers<N>`）來自下游 repo 的間接證據，**未經一手確認**；
部分 Kaggle 鏡像還會多包一層目錄。
**驗證**：斷言通過，或印出缺漏清單。

### 2.2 VOC → COCO 轉換（M2）

**DATA-05** — **座標索引必須執行期偵測，不得寫死。**

原始 PASCAL VOC devkit 用 **1-based** 座標，所以經典轉換器會對 `xmin`/`ymin` 減 1。
但這份資料集的 XML 幾乎確定是 Roboflow 匯出管線機器產生的，而現代工具輸出 **0-based**。
**對已經是 0-based 的資料再減 1，會產生 -1 座標並讓每一個框都靜靜偏移。**

偵測邏輯：掃過所有框取全域最小座標 `gmin`——
`gmin == 0` → offset 0（不減）；`gmin == 1` → offset 1（減 1）；
其他 → offset 0 並**大聲記錄**。把決定寫進 manifest。

**這是整個轉換裡最容易安靜壞掉的一步，也是六行程式碼裡價值最高的一段。**
**驗證**：轉換 log 印出 `global min coordinate = <gmin> -> offset <n>`，且該值進入 manifest。

**DATA-06** — 寬高用 `w = xmax - xmin`（**不加 1**）。
*理由*：VOC 語意上 `xmax` 是**含**的像素索引，真實寬度是 `xmax - xmin + 1`；
COCO 的 `width` 是**不含**的長度。實務上整個生態系都用不加 1 的版本，
沿用它才能與已發表結果比較。
**但這裡不是免費的**：在 ~34 px 的 `head` 框上，1 px 誤差約是 3% 的面積誤差。
此選擇必須在 README 明載。
**驗證**：程式碼註解與本需求 ID 對應；`reports/class_distribution.md` 註明此約定。

**DATA-07** — `difficult` 旗標**不可**映射到 `iscrowd`。
*理由*：`iscrowd=1` 會讓 `pycocotools` 把 IoU 改成 Intersection-over-Area 並允許多對一匹配，
語意完全不同，**會直接灌水指標**。
作法：保留該標註、`iscrowd=0`，另存自訂欄位 `difficult: 0|1`，
用 `int(obj.findtext("difficult", default="0") or 0)` 讀取並印出直方圖。
**驗證**：輸出的 COCO JSON 中所有 `iscrowd == 0`；`difficult` 直方圖出現在轉換報告。

**DATA-08** — 真實影像尺寸從 PNG 讀（`PIL.Image.open(p).size`，惰性、不解碼像素），
與 XML 的 `<size>` 交叉比對，不符要回報。
*理由*：機器產生的 VOC XML 常有錯誤的 `<size>`；不符是「標註後才縮放影像」的訊號。
**驗證**：轉換報告列出不符的檔案數（預期為 0）。

**DATA-09** — 決定性 ID 指派。
`category_id` 從 **1** 起算，順序由 `configs/paths.yaml` 的 `dataset.classes` 決定
（`helmet`=1、`head`=2、`person`=3）；
`image_id` 由**排序後的檔名索引**決定（`sorted(paths, key=lambda p: p.stem)`），
**絕不依賴 `os.listdir` / `Path.glob` 的順序**（跨檔案系統不穩定）。
**驗證**：兩次執行產生完全相同的 `image_id` 對應。

**DATA-10** — 標籤白名單。`name` 取出後 `.strip().lower()`，比對 `{helmet, head, person}`。
出現任何非預期標籤要**大聲失敗**並印出直方圖。
*理由*：類別字串的實際大小寫**未經一手確認**（所有來源都是二手），
且「安靜地多學了第四類」是最難察覺的一種 bug。
**驗證**：未知標籤計數 = 0，否則非零退出。

**DATA-11** — 邊界處理，每一項都**記錄不靜默修正**：
座標裁切到影像範圍內；`xmin > xmax` 時交換並記錄；
裁切後 `w <= 0` 或 `h <= 0` 的框丟棄並記錄；
`<object>` 沒有 `<bndbox>` 子節點要處理；
座標可能是浮點字串（先 `float()` 再 round，不要對 `"12.5"` 直接 `int()`，會拋例外）。
`ET.parse` 要 wrap try/except 並產出 `parse_failures.json`。
**驗證**：`reports/conversion_report.md` 列出每種修正的計數。

**DATA-12** — COCO schema 相容性：每個 annotation 需有
`segmentation: []`、`area = w * h`、`iscrowd = 0`、唯一整數 `id`；
頂層需有 `info` 與 `licenses`（標 CC0 1.0）；
`images[].file_name` 用**正斜線**（`Path.as_posix()`）。
**驗證**：**COCO 自評測試**——用 `pycocotools.COCO` 載入產出的 GT，
拿它跟自己跑 `COCOeval` → mAP 必須剛好是 `1.000`。
五行程式碼一次擋掉 `[x,y,w,h]` vs xyxy 混淆、`area` 算錯、`iscrowd` 設錯、孤兒 `image_id` 一整類 bug。

### 2.3 近似圖片分群（M4）

**DATA-13** — 這份資料源自工地影片，**近似結構嚴重，是 split 有效性的最大威脅**。

分群邊規則：
```
edge(i, j)  若  phash_hamming(i,j) <= T_phash
            或  ( clip_cosine(i,j) >= T_clip  且  phash_hamming(i,j) <= T_clip_guard )
```
門檻在 `configs/filtering.yaml` 之外另存於 `configs/paths.yaml` 之外的分群設定（M4 決定，
由 Spike H3 校準後寫入 manifest）。

**CLIP 的護欄不是可選的**：CLIP 是語意相似度，兩個**真的不同**的工地只要都有黃色安全帽
與鷹架就會拿到很高的分數。在這種高度同質的資料集上**單用 CLIP 會嚴重過度合併**，
可能把整個資料集吞成一群。pHash 護欄就是防這件事。
**驗證**：見 DATA-15 的強制檢查。

**DATA-14** — 計算方式：5,000×5,000 全對比用 unpackbits 矩陣乘法直接算
（約 3.2 GFLOP，CPU 一秒內），**不需要 BK-tree 或 LSH**。
分群用 `scipy.sparse.csgraph.connected_components(directed=False)`——
孤立影像自動成為單獨一群，所以 `group_id` 是 5,000 張圖的完整分割，
正好是 group split 需要的輸入。
CLIP 若啟用，模型與 pretrained tag 必須釘死並寫進 manifest
（不同實作的 embedding **不可互換**）。
**驗證**：`group_id` 覆蓋全部 5,000 張且無缺漏。

**DATA-15** — **強制檢查，不可跳過。** 印出分群大小直方圖與最大的 20 群。
**若最大群超過全體的 5%（250 張），代表門檻太鬆，必須調緊並重跑。**
這關在凍結之前必須通過。
*理由*：一個吞掉整個資料集的巨群會讓 group split 無法達成目標比例，
而且是**安靜地**失敗——你會得到一個看起來正常但實際上 train/test 高度重疊或極度失衡的切分。
**驗證**：`reports/grouping_report.md` 含直方圖與最大群大小，且最大群 ≤ 250。

### 2.4 70/15/15 group split 與凍結（M4–M5）

**DATA-16** — **同一 `group_id` 的影像必須落在同一個 split。** 這是 hard fail 斷言。

**DATA-17** — **分層分配，不能用天真的隨機 group split。**
`person` 只出現在 158 張圖裡，隨機切分很可能讓某個 split 幾乎沒有 person。
演算法：
1. 算出每群的圖片數與 per-class 實例數
2. **先分配含 `person` 的群**，由大到小，每群給目前 person 配額缺口最大的 split
3. 其餘群依圖片數由大到小做 LPT 裝箱，指派給
   `(n_images, n_helmet, n_head, n_person)` 四維相對缺口最大的 split
4. 平手用 seed=42 的 RNG 決定
**驗證**：每個 split 至少拿到 10% 的 `person` 實例；三分割互斥、聯集 = 5,000；
比例落在 70/15/15 的 ±2% 內（group 切分無法剛好命中）。

**DATA-18** — 凍結產物：
- `splits/split_manifest.json`：每張圖 `{image_id, file_name, sha256, phash, group_id, split}`
- `splits/test_blocklist.json`：Test 影像的 `{image_id, file_name, sha256}` 集合，
  供生成端啟動時反查
- `splits/source_checksums.json`：來源壓縮檔雜湊
- `splits/MANIFEST.sha256`：manifest 自身的 SHA256，**單一指紋**

manifest 必須額外記錄：kagglehub 版本號、座標 offset 決定（DATA-05）、
pHash 門檻、CLIP 模型 tag 與門檻（若啟用）、seed。

序列化用 canonical JSON：`sort_keys=True, separators=(",",":"), ensure_ascii=True`，
寫檔用 `newline="\n"`（見 [ENV-10](environment.md)）。
**驗證**：連跑兩次 `splits/MANIFEST.sha256` 完全相同。

**DATA-19** — **凍結後 `splits/` 只能重新產生，不能手改。**
若非改不可，必須同步重算 `MANIFEST.sha256`、新增一則 ADR、
並在 `worklog.md` 記下**哪些下游產物因此作廢**。

**DATA-20** — `assert_test_untouched()` 輔助函式：讀 `test_blocklist.json`，
重新雜湊每個 Test 檔案並比對。**每一支生成腳本在啟動時都必須呼叫它。**
**驗證**：`grep -L "assert_test_untouched" src/synthetic/*.py` → 空。

### 2.5 統計報告（M5）

**DATA-21** — `reports/class_distribution.md` 需含：per-class 實例數與圖片數、
**實測**的面積分布（解決 1.2 節的讀數衝突）、每圖物件數分布、
框尺寸的百分位表、**特別標出 `head` 的圖片層級稀少度（920/5,000）**
以及 `person` 的極端稀少度（158/5,000）與其標註缺陷。
配圖 `reports/figures/class_distribution.png` **要自己打開檢視**後再交使用者過目。
**驗證**：報告中的每個數字都能從 `split_manifest.json` ＋ COCO JSON 重新聚合出完全相同的值。

---

## 3. 驗證性 Spike

### DATA-24 — 標註語意：這個資料集到底在框什麼（H1 已定案）

> 這一節是 2026-07-31 補寫的。H1 的**做法**原本寫在下面，
> **結論**卻只散落在 [ADR-007](decisions.md#adr-007) 與 evaluation_spec 裡，
> 使用者得開口問才知道。規格的職責是讓人不用問。

**框的是「人的頭」，不是「安全帽這個物體」。**

| 類別 | 實際框的東西 |
|---|---|
| `helmet` | **戴著安全帽的那顆頭**（整個頭部區域，含帽） |
| `head` | **沒戴安全帽的那顆頭** |
| `person` | 人的身體（標註嚴重不完整，見 §1.3） |

`helmet` 與 `head` **在同一個人身上互斥**。實測證據：

- 同圖 9,603 個 `helmet × head` 組合中，只有 **95 個（0.99%）** IoU > 0.1
- 長寬比中位數 `helmet` **0.875**、`head` **0.830**——兩者都是**高大於寬**，
  形狀就是人頭。若 `helmet` 只框帽殼，應為寬扁（寬/高 1.3–1.8）

**⚠️ 沒有人戴的安全帽（放在桌上、地上、掛在架上）不框。**
目視驗證於 `image_id=4029`（`hard_hat_workers4623.png`）：
會議桌上明確擺著 **3 頂紅色安全帽**，該圖標註為 `head=8`、**`helmet=0`**——
八個框全在沒戴帽的人頭上，桌上的帽子一個都沒框。
對照 `image_id=1629`／`3803`：那裡的 `helmet` 框都框在**戴著帽的人**身上。
證據圖：`reports/figures/review/loose_helmet_question.png`。

這是自洽的：判定的是**「這個人有沒有戴」**，沒人戴的帽子與工安合規無關。

**因此合規率可以直接數框，零空間推理**：

```
compliance_rate = n_helmet / (n_helmet + n_head)
```

**對合成的三個直接後果**：
1. hard negative 的干擾物**不給標註是正確的**——它們是地上的假安全帽，
   而真實資料裡地上的**真**安全帽也不框（[COMP-22](synthesis_spec.md)）
2. `helmet_to_head_swap`（[COMP-18](synthesis_spec.md)）是把「戴帽的頭」換成「裸頭」，
   不是把帽子拿掉，所以尺寸繼承 anchor 才合理
3. [FILT-07](filtering_spec.md) 的 helmet-above-head 幾何規則**沒有真實配對可校準**，
   它只治理合成的「戴著」構圖，報告中必須註明這點

---

### DATA-25 — 影像不是單一解析度，預測框必須逐圖映射

**上游全部寫錯，我們自己量過。** SHEL5K 論文、Kaggle、Roboflow 都說這份資料是
416 × 416。實測 `coco_all.json` 的 5,000 張影像：

| 尺寸 | 張數 | 佔比 |
|---|---:|---:|
| **416 × 415** | **2,461** | **49.2%**（多數） |
| 416 × 416 | 2,192 | 43.8% |
| 415 × 416 | 324 | 6.5% |
| 415 × 415 | 23 | 0.5% |

`416 × 416` 不但不是唯一尺寸，**連多數都不是**。
凍結的 Test split（744 張）同樣四種都有：353／340／47／4。

**驗證方式**：抽 40 張 PNG 用 Pillow 直接讀 `Image.size`，與 `coco_all.json`
記錄的 `width`／`height` 逐張比對，**零不符**——所以 COCO 裡的值來自真實像素，
不是從 XML 抄的——**DATA-08** 的 PNG／XML 交叉比對確實生效了。

**判定式**：任何把偵測框從訓練解析度映射回標註座標的程式，
**一律使用該圖自己的 `width`／`height`**，不得使用任何全域縮放因子。

```
scale_x = image.width  / evaluated_width      # 逐圖
scale_y = image.height / evaluated_height     # 逐圖，且與 scale_x 不一定相等
```

**為什麼這條要單獨立一個 ID**：誤差只有 1 px，小到不會有任何東西報錯，
但它同時打壞兩件事——

1. **`scale_x != scale_y`**。用一個純量縮放會讓其中一軸偏 0.24%。
   `head` 平均約 34 × 34 ≈ 1,156 px²，正好卡在 small／medium 邊界（1,024）附近，
   所以面積上的小偏差會真的把物件換桶，直接動到 [EVAL-07](evaluation_spec.md) 的主敘事指標
2. **x 與 y 可以互換而不被發現**。若測試 fixture 全是正方形，
   把 `(height, width)` 寫成 `(width, height)` 的轉置錯誤會安靜通過
   （這正是 [K-19](troubleshooting.md) 抓到的其中一條變異）

**實作**：`Sample`（[`src/training/data.py`](../src/training/data.py)）已經帶
per-image `width`／`height`，`run.py` 的 `target_sizes` 也是逐圖給的，
所以現有路徑是對的。這條規則是為了讓它**保持**是對的。

**測試要求**：任何座標映射的測試，**至少一個 fixture 必須是非正方形**。

---

### Spike H1 — `helmet` 框到底框什麼？（最高槓桿，M3）

**問題**：`helmet` 的 bbox 框的是安全帽本體，還是整顆戴著安全帽的頭部區域？
`helmet` 與 `head` 在同一個人身上是互斥的嗎？

**為什麼重要**：這決定了三件事的寫法——
cutout bank 的素材語意、FILT-07 helmet-above-head 幾何規則是否有真實資料可校準、
以及合規狀態用主定義還是 fallback 定義（[ADR-003](decisions.md#adr-003)）。
**目前沒有任何來源定義過這件事**（Kaggle、Roboflow、Dataset Ninja、SHEL5K 論文皆未說明）。

**作法（約 30 分鐘）**
1. 各隨機抽 40 個 `helmet` 與 `head` 框，帶 50% context 裁切，各拼成一張 contact sheet，**用眼睛看**
2. 同時算：同一張圖內 `helmet` 框與 `head` 框 IoU > 0.1 的配對數
3. 算 per-class 長寬比直方圖

**決策規則**
- IoU>0.1 的配對數 ≈ 0 → **互斥**。走合規主定義（偵測到的類別本身就是狀態，零空間推理）；
  FILT-07 沒有真實配對可校準，改用相對於 `person` 框的位置分布，並在報告註明
  此規則只治理合成構圖、不對應任何真實資料模式
- 配對數顯著 → `helmet` 只框帽殼。走 fallback 定義，FILT-07 可用真實配對校準
- 長寬比旁證：只框帽殼會**寬大於高**（約 1.3–1.8）；框整個頭部區域會接近 1（約 0.85–1.15）

**旁證（尚不足以定案）**：有來源提到標註「包含只有 `person` 與 `head` 的情況，
用於個人未戴安全帽時」，支持 `head` = 裸頭 的讀法。
且 `helmet` 平均面積約為 `head` 的 2 倍，也暗示 `helmet` 框的是較大的頭部區域。

### Spike H3 — 近似分群結構（卡住所有下游，M3）

**問題**：影片衍生的 5,000 張圖被 pHash 收斂成幾群？最大群多大？

**為什麼重要**：若 5,000 張塌縮成例如 400 群，70/15/15 的 **group** split 可能嚴重偏離目標比例，
Test 可能過小或過大；而且 cutout bank **與**背景的有效多樣性都遠低於假設，
`synthesis_spec.md` 的整個生成預算都要重算。**這關過不了，split 不能凍結，下游全部不能動。**

**作法（約 30 分鐘，純 CPU）**
1. 對 5,000 張算 pHash
2. 在 Hamming ≤ 4、≤ 6、≤ 8、≤ 10 四個門檻各做一次連通元件分群
3. 印出每個門檻的群數、群大小直方圖、最大群大小
4. 用不同 seed 模擬 5 次 group split，印出各 split 的圖片數

**決策規則**
- 選一個讓最大群 ≤ 250 張（5%）且群數合理的門檻，寫進 manifest
- **只有在 pHash 分出超過 2,000 群時才啟用 CLIP**——那代表 pHash 沒抓到影片結構
- 若任何門檻都無法同時滿足「最大群 ≤ 5%」與「切分比例合理」，停下來報告，不要硬切

**取捨方向**：這是不對稱損失。漏抓一個近似對會**安靜地**灌水測試指標（很糟、看不見）；
過度合併只是少一點切分彈性（輕微、看得見）。**刻意偏向過度分群。**

### Spike H5 — 放置先驗品質（M3）

**問題**：從真實 Train 框建出的位置直方圖，有足夠資訊量嗎？

**作法（約 20 分鐘）**：建每類 16×16 的正規化 `(cx, cy)` 直方圖（Laplace 平滑），
疊在平均影像上畫成熱圖，**打開來看**。

**決策規則**：若過於發散、看不出結構 → 改以**錨定放置**為主
（helmet 錨在 head 上、head 錨在 person 內），放棄取樣式先驗。

---

## 4. 這份協定被違反時會發生什麼

| 違反 | 後果 |
|---|---|
| DATA-05 座標 offset 判斷錯 | 每一個框都偏移 1 px。**不會報錯**，只會讓所有指標略低，且永遠找不到原因 |
| DATA-07 把 `difficult` 當 `iscrowd` | 指標被灌水，且方向是「看起來變好」 |
| DATA-15 分群過鬆 | Train 與 Test 有近似圖片 → 測試指標灌水 → **整份結論作廢** |
| DATA-16 同群跨 split | 同上 |
| DATA-17 沒有分層 | 某個 split 幾乎沒有 `person`，per-class AP 變成純雜訊 |
| DATA-20 生成端讀到 Test | **整份結論作廢**，且無法事後補救 |
