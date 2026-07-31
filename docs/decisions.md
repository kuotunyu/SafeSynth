# Decision Log (ADR)

> **只追加，不改寫。** 決策改變時新增一則 ADR 並在舊的那則標註「被 ADR-0NN 取代」。
> 格式：脈絡 → 決策 → 後果 → 待查證 → 來源。

| # | 標題 | 日期 | 狀態 |
|---|---|---|---|
| [ADR-001](#adr-001) | SAM2 走 Hugging Face transformers，不用官方 repo／Ultralytics／SAM 3 | 2026-07-27 | 生效 |
| [ADR-002](#adr-002) | Windows 原生、不使用 WSL；Python 3.12 ＋ torch cu130 | 2026-07-27 | 生效 |
| [ADR-003](#adr-003) | 保留 3 類，但合規狀態不依賴 `person` | 2026-07-27 | 生效 |
| [ADR-004](#adr-004) | Hard negatives 挖料為主、程序生成為輔，且完全不給標註 | 2026-07-27 | 生效 |
| [ADR-005](#adr-005) | 速度對照組用 RF-DETR-Nano，不用 Ultralytics YOLO | 2026-07-27 | 生效 |
| [ADR-006](#adr-006) | `transformers` 下限提高到 v5.14.1（v5 改了 image processor 命名） | 2026-07-27 | 生效 |
| [ADR-007](#adr-007) | H1/H3/H5 實測：合規採類別直讀，split 採 guarded CLIP，人物採錨定放置 | 2026-07-27 | 生效 |
| [ADR-008](#adr-008) | SAM2 Pass 2 採 effective crop-512，不直接把小框放大到 1024 | 2026-07-27 | 生效 |
| [ADR-009](#adr-009) | H4 維持硬阻擋；FILT-11 排除自身背景 | 2026-07-27 | **技術判定生效；第 1 點的「無限阻擋」後果被 [ADR-011](#adr-011) 取代** |
| [ADR-010](#adr-010) | H4 Option A 採 FLUX.2 參照圖條件式邊界 inpainting | 2026-07-27 | **停止**（見 [ADR-011](#adr-011)） |
| [ADR-011](#adr-011) | H4 判定為「已知限制」而非無限阻擋；1× 生成並推進 Phase 2 | 2026-07-31 | 生效 |

---

## ADR-001 — SAM2 走 Hugging Face transformers，不用官方 repo／Ultralytics／SAM 3

### 脈絡

我們需要用 bbox 當 prompt，對 Train 的每個標註取得乾淨的 cutout mask，離線批次跑約 5,000 張圖。
環境是 **Windows 11 原生、無 WSL**（[ADR-002](#adr-002)），且程式碼要以 MIT 授權公開發佈，
衍生的合成影像要上傳 Hugging Face。

有五條可行路徑，授權與 Windows 相容性差異很大。

### 決策

**用 `transformers` 的 `Sam2Model` / `Sam2Processor` 搭配 `facebook/sam2.1-hiera-large`。**

- 純 pip 安裝，**零 CUDA 編譯**，這是 Windows 原生環境的決定性優勢
- 原生支援 `input_boxes`（XYXY，形狀 `(batch, num_boxes, 4)`），且支援多圖 × 多框批次
- 程式碼與權重皆 **Apache-2.0**，與 MIT repo 相容，對輸出無任何限制
- `model.get_image_embeddings()` 可讓同一張圖的多個 box 重用影像編碼

**硬性下限 `transformers>=4.57.1`。**（⚠️ 此下限已被 [ADR-006](#adr-006) 提高到 **5.14.1**；
以下保留原始論述作為記錄。）這條不是保守，是必要：
4.56.x 的 `_embed_boxes` 少了原始實作會補的 padding point，導致 box prompt 的
mask 品質從 IoU ~0.98 掉到 ~0.94。這是**安靜的劣化**——不會報錯，只會讓每一個
cutout 都差一點。修正在 PR #40800（2025-09-12）併入、v4.57.0 發佈，但 4.57.0 在
PyPI 被 yank，因此下限訂在 **4.57.1**。

**必須自己做 mask 後處理。** `processor.post_process_masks()` 的
`max_hole_area` / `max_sprinkle_area` 兩個參數在 transformers 裡是
**接受但完全不作用的 no-op**（transformers 原始碼在該處留了一行註解，
說明連通元件 kernel 是預計要補、目前沒有的）。
不知道這件事的人會以為破洞與雜點已經清掉，其實沒有。因此清理流程
（保留最大連通元件 → 填洞 → 形態學閉運算）自己用 `cv2.connectedComponentsWithStats`
與 `scipy.ndimage` 實作，參數見 `configs/compose.yaml` 的 `sam2.cleanup`。
這反而比原本的 CUDA kernel 好：門檻可控、可記錄、可在報告中交代。

### 被否決的四條路徑

| 路徑 | 否決理由 |
|---|---|
| **`pip install sam2`（PyPI）** | **不是 Meta 官方套件**。PyPI 上的作者是第三方，`home_page` 指向個人 fork。Meta 從未發佈官方 PyPI 套件（官方 repo 的 issue #433 至今無人回覆）。視為未經審查的供應鏈相依 |
| **官方 `facebookresearch/sam2`（git 安裝）** | 可用但次選。repo 自 2024-12-16 起實質凍結；官方 `INSTALL.md` 明寫建議 Windows 使用者改用 WSL——正是我們排除的方案；Hydra config 在 Windows 有已知路徑 bug（issue #177 / #304 / #701），且 `SAM2_BUILD_CUDA=0` 也修不掉；找不到任何近期的原生 Windows 成功案例 |
| **Ultralytics `SAM("sam2.1_l.pt")`** | API 最簡單，但 **AGPL-3.0**。輸出資料本身不受影響（程式輸出一般不受程式碼著作權涵蓋），但**我們 `import` 它的那支 Python 檔會成為衍生作品而必須同樣 AGPL-3.0**，與 MIT repo 的目標直接衝突。另外它的權重是自行轉檔重新託管的，非 Meta 原始 CDN，對公開資料集的來源交代也較差 |
| **SAM 3 / SAM 3.1** | 模型更好，但走 Meta 自訂的 "SAM License" 而非 Apache-2.0，含**再散布義務**（散布衍生物時必須一併附上該授權）與**發表致謝義務**。對一個要公開發佈資料集的專案來說，Apache-2.0 的乾淨程度值得放棄那一點品質 |

### 後果

- `pyproject.toml` 釘 `transformers>=4.57.1`；`docs/environment.md` 的驗證表要檢查實際版本
- cutout bank 的後處理程式碼比「呼叫官方 API」多約 30 行，但完全在我們控制之下
- 兩趟式 SAM2 策略（見 `docs/synthesis_spec.md`）建立在 `input_boxes` 與 crop 重投影上，
  這兩者在 transformers 路徑都是原生支援
- 若日後 transformers 移除 SAM2 支援，退路是官方 repo ＋ WSL，但那會違反 ADR-002

### 待查證（M7 前重新確認）

- 執行時實測 `transformers.__version__`，並跑一次已知 box 確認 mask 合理
- `facebook/sam2.1-hiera-large` 的 HF model card 是否仍標 `apache-2.0`

### 來源

- [transformers SAM2 docs](https://huggingface.co/docs/transformers/model_doc/sam2)
- [transformers issue #40787（box prompt 品質劣化）](https://github.com/huggingface/transformers/issues/40787)
- [transformers PR #40800（padding point 修正）](https://github.com/huggingface/transformers/pull/40800)
- [facebookresearch/sam2 INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md)
- [facebookresearch/sam2 issue #433（PyPI 套件？）](https://github.com/facebookresearch/sam2/issues/433)
- [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large)
- [Ultralytics License](https://www.ultralytics.com/license)｜[GNU GPL FAQ — program output](https://www.gnu.org/licenses/gpl-faq.html)
- [facebookresearch/sam3 LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE)

---

## ADR-002 — Windows 原生、不使用 WSL；Python 3.12 ＋ torch cu130

### 脈絡

母計畫 `SDG_portfolio_plan_v2.md` 整份是假設 WSL2 環境寫的。使用者明確要求**這個專案不要用 WSL**。
本機是 Windows 11 + RTX 4090（驅動 591.86）、C: 剩 186 GB、D: 剩 1728 GB、
`LongPathsEnabled = 0`、系統 Python 是 anaconda 3.10.9。

### 決策

**全程原生 Windows，用 uv 管理 Python 3.12 的專案虛擬環境。**

**torch 必須從 `https://download.pytorch.org/whl/cu130` 安裝。**
這是本專案最容易安靜出錯的一步：PyPI 上的 `win_amd64` wheel 是 **CPU-only**
（約 122 MB，對比 Linux CUDA wheel 的約 527 MB——PyPI 的檔案大小限制使得
Windows 的 CUDA 版本從來不上傳 PyPI）。一個裸的 `pip install torch` 會成功安裝、
不會報錯，然後 `torch.cuda.is_available()` 回 `False`，4090 整場閒置。

`pyproject.toml` 用 `[[tool.uv.index]]` ＋ `explicit = true`，讓這個 index
**只**服務 torch 與 torchvision，其餘套件仍走 PyPI。

**cu128 不可用。** 實測 cu128 index 最高只到 torch 2.11.0，沒有 2.12／2.13
（CUDA 12.8 自 torch 2.12 起已從標準發佈矩陣移除）。已實測存在的組合是：
`torch-2.13.0+cu130-cp312-cp312-win_amd64.whl` ＋
`torchvision-0.28.0+cu130-cp312-cp312-win_amd64.whl`。
驅動 591.86 ≥ cu130 要求的 580.88，過關。

**選 Python 3.12** 的理由：`kaggle` 2.x 要求 ≥3.11、`kagglehub` 要求 ≥3.10、
而 `pycocotools` 2.0.11 的 `cp312-abi3-win_amd64` wheel 一顆涵蓋 3.12/3.13/3.14，
在原生 Windows 上**不需要 MSVC 就能裝**（`pycocotools-windows` 那個套件停在 2020，已死）。

**大檔放 D:。** `data_root: "D:/sdg-data/02-safesynth"`，與兄弟專案一致。
C: 只留 repo（程式碼、設定、文件、manifest、小圖）。

### 後果

- 所有指令以 PowerShell 5.1 形式撰寫：**沒有 `&&` / `||`**，用 `A; if ($?) { B }`
- `multiprocessing` 用 spawn 不是 fork：所有平行程式碼必須包 `if __name__ == "__main__":`，
  否則會無限遞迴產生行程。DataLoader 預設 `num_workers=0`
- `LongPathsEnabled = 0`，260 字元上限是活的。HF 快取先沿用 C: 預設；撞到就把
  `hf_home` 設成短路徑（見 `configs/paths.yaml` 註解與 `docs/troubleshooting.md`）
- 檔案雜湊一律 binary 模式、manifest 路徑一律 `as_posix()`，否則 manifest 的 SHA256 跨平台對不上
- 檔案系統大小寫不敏感：`images/` 與 `Images/` 在 Windows 同一個東西、在 Linux 不是。
  manifest 內路徑要正規化大小寫
- 不用 `torch.compile`（Windows CUDA inductor 仍需要從 `vcvars64.bat` 啟動 shell，
  徒增失敗面而本專案不需要那點速度）；不用 flash-attn（無官方 Windows wheel，
  且 transformers 的 SDPA 已足夠）

### 待查證（M1 執行時）

- `uv python install 3.12` 後實跑
  `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`，
  必須印出 `2.13.0+cu130 13.0 True`
- torchvision 交給 uv 從同一個 index 解析，不要盲目釘死版本號

### 相關

**跨專案提醒**：兄弟專案 `1_DefectForge` 的 `pyproject.toml` 同時寫著「torch 從 cu128 index 裝」
與「已查證 torch 2.13.0」。依上述實測，該組合不存在，需一併修正為 cu130。

### 來源

- [PyTorch cu130 wheel index](https://download.pytorch.org/whl/cu130/torch/)（2026-07-27 實地確認 win_amd64 cp312 檔名）
- [PyTorch cu128 wheel index](https://download.pytorch.org/whl/cu128/torch/)（實地確認最高只到 2.11.0）
- [uv — PyTorch 整合指南](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [pycocotools on PyPI](https://pypi.org/project/pycocotools/)

---

## ADR-003 — 保留 3 類，但合規狀態不依賴 `person`

### 脈絡

專案目標是「偵測 Helmet／Head／Person 三類並推導安全帽合規狀態」。
原始構想是用三者的空間關係推導每個人的戴帽狀態。

但查證發現資料集有結構性缺陷：

- `person` 只有 **751 個實例，分布在 158 張圖（佔全部 3.16%）**
- SHEL5K 論文用**同樣這 5,000 張圖**重新標註，得到 **75,570 個標註**（6 類），
  原版只有 **25,502** 個（3 類）——約 **2/3 的真實物件在原版是未標註的**
- 該論文明確指出原版的 `person` 類「標得太差」

這造成三個具體問題：

1. **`person` AP 不是一個測量值。** 158 張圖經 15% 切分後 Test 只剩約 24 張、約 110 個實例。
   在這個樣本數下，per-class AP 的 bootstrap 95% 信賴區間大約 ±0.10–0.15 mAP。
   「合成資料讓 person AP 進步 4 個點」是雜訊。
2. **漏標會讓 person AP 隨著模型變好而被壓得更低。** 模型偵測到一個真實但未標註的人，
   評測器判它假陽性。這是**系統性偏差不是變異**，加再多資料也修不掉，而且方向與本專案的論點相反。
3. **它會汙染 cutout 素材庫。** 751 個框過完閘門大約剩一兩百個，且來自 158 張本身就是
   近似影格的圖，**實際不重複的人可能只有數十位**。拿數十個模板貼三千次，教的是模板記憶不是泛化。

### 決策

**三類全部保留**——模型、COCO JSON、訓練都留三類。**誠實性放在報告層，不是偷偷砍類別。**

**合規狀態改由 `helmet` vs `head` 推導，`person` 完全不承重。**

- **主定義**（待 spike H1 確認 `helmet` 框的是「戴帽的頭部區域」且與 `head` 在同一人身上互斥）：
  兩類已經把「頭部可見的人」切乾淨，所以偵測到的類別本身就是狀態。
  `helmet → COMPLIANT`、`head → NON_COMPLIANT`，
  `compliance_rate = n_helmet / (n_helmet + n_head)`。**零空間推理、不需要 person 框。**
- **Fallback 定義**（若 H1 顯示 `helmet` 只框帽殼、底下另有 `head` 共存）：
  `head` 為 COMPLIANT 若且唯若存在某個 `helmet` 滿足 FILT-07 的幾何配對。
- `src/inference/compliance.py`（Phase 2）**兩種模式都實作、由 config 旗標切換**，由 H1 的結果決定用哪個。
- **強制單元測試**：把所有 `person` 偵測刪光後重跑合規判定，結果必須**逐位元相同**。
  這條測試就是「`person` 不可承重」的機械保證。

**cutout bank 的處置**
- 每個近似群最多取 2 個 `person` cutout（`configs/compose.yaml` 的
  `max_person_cutouts_per_group`）
- 同時回報 `n_person_cutouts` 與 **`n_distinct_person_groups`**——後者才是真實多樣性，
  該進 README 而不是被埋起來
- **預先宣告的退路**：若 `n_distinct_person_groups` 低於門檻，`crowded` 情境改用
  head + helmet 素材堆人群，**並在報告中明說**。情境名稱不變、素材改變、改變被記錄
- **不**為了「修正類別不平衡」而合成 person 實例。從數十個模板灌大訓練集的 person 數量
  對泛化毫無幫助，只會讓 Phase 2 的結果變得無法解讀

**指標呈現規範**（Phase 2 執行前預先登記）
- **主指標**：`mAP50-95 (helmet, head)`、`AP_small (helmet, head)`、
  `bare-head recall @ IoU 0.5`、hard-negative 子集的每圖誤報數、compliance P/R
- **`person` AP 照實報告、絕不隱藏**，但每次都必須同時附上：
  (a) 該 split 的 person 實例數與圖片數、
  (b) 對測試圖做 1,000 次重抽的 bootstrap 95% 信賴區間、
  (c) 常設註腳引用 SHEL5K，說明 person AP 在此基準上並不測量 person 偵測品質
- 同時報 `mAP_all3` 與 `mAP_helmet_head`，**主張以後者為準**

### 後果

- 全域推論：因為約 2/3 真實物件未標註，**每一個類別**的絕對 AP 都被壓低、precision 被系統性低估。
  因此**所有主張必須是相對的**（同一凍結 Test 上 A 組 vs B 組），永遠不能是絕對值。
  這不是要道歉的弱點——五組對照協定**本質上就是相對比較**，把這點講明白正是可信度的來源
- `crowded` 情境的權重刻意壓低（見 `configs/compose.yaml`），因為 person 素材庫弱
- README 必須有一節交代這個資料集缺陷，而不是只在 Limitations 帶一句

### 待查證

- **Spike H1** 決定主定義還是 fallback 定義（見 `docs/data_protocol.md`）
- **延伸機會（可行性未經查證）**：SHEL5K 重標了同樣這 5,000 張圖。若其標註可下載且授權允許，
  對凍結 Test 的那批 image id 做一次「乾淨標註」的次要評測，會是遠更乾淨的測量。
  **可取得性與授權兩者都還沒查**，列為 Phase 2 的選配項

### 來源

- [SHEL5K: An Extended Dataset and Benchmarking for Safety Helmet Detection (Sensors 2022)](https://www.mdpi.com/1424-8220/22/6/2315)｜[PMC8950768](https://pmc.ncbi.nlm.nih.gov/articles/PMC8950768/)
- [Safety Helmet Detection — Dataset Ninja](https://datasetninja.com/safety-helmet-detection)（per-class 實例數與圖片數）
- [SHEL5K — Mendeley Data](https://data.mendeley.com/datasets/9rcv8mm682/4)

---

## ADR-004 — Hard negatives 挖料為主、程序生成為輔，且完全不給標註

### 脈絡

目標情境之一是「容易誤判成安全帽的 hard negatives」——黃色機具、圓形物體。
資料集本身沒有這類標註，素材必須自己造。有兩條路：程序生成，或從原圖挖真實碎片。

### 決策

**兩者都做，以挖料為主（約 7 成）、程序生成為輔（約 3 成）。**

對一個 *hard* negative 而言，**域內的真實材質、光照與雜訊統計，遠比程序生成的可控性值錢**。
挖料能拿到真正的挖土機面板、三角錐、油桶、電纜捲盤——這些正是現場會誤導偵測器的東西。
程序生成則補上資料集裡沒有的形狀，並且完全可 seed 重現。

程序生成有一個陷阱要避開：若用平坦填色，偵測器會學到「程序紋理 ⇒ 負樣本」而不是
「黃色圓形物 ⇒ 負樣本」，那產生的是**簡單**負樣本，與目的正好相反。
因此程序形狀必須**用陰影調變真實背景紋理**，而不是貼上一塊純色。

### 挖料的特定風險與三層防護

**這是本 ADR 最重要的一段。** 因為 SHEL5K 顯示約 2/3 真實物件未標註，
**「這裡沒有標註」不等於「這裡沒有安全帽」**。天真的挖料會撈到真實但未標註的安全帽，
把它們當成負樣本貼進訓練集——等於**教偵測器抑制安全帽**，方向完全反了。

三層防護（門檻見 `configs/compose.yaml` 的 `hard_negatives.mining`）：

1. 候選與任何既有標註的 IoU 必須低於門檻
2. 必須**通不過**「像是戴著的安全帽」測試——正下方沒有類頭部區域，且長寬比或面積落在
   安全帽的經驗範圍之外
3. **凍結素材庫前強制人工過目一張 8×8 contact sheet。** 只花使用者五分鐘，
   而這是唯一真正能抓到這種失敗的手段

**Spike H6 是決策點**：若挖出的候選中真實安全帽的比例超過門檻，
**翻轉為程序生成為主**，挖料降為輔助並逐張人工複核。

### 為什麼完全不給標註（這是正確語意，不是偷懶）

- 標籤空間是 `{helmet, head, person}`。一個黃色水桶不屬於其中任何一類。
  在 COCO / VOC 語意下，**沒有框本身就是一個正面斷言**：此區域不存在任何列出類別的物件。
  因此一張含水桶且水桶上沒有框的圖，是**完整且正確標註**的
- 加第四個 "distractor" 類別會改變任務本身、改變 mAP 的分母，並破壞與 Real-only 基線的可比性
- 設成 `iscrowd` / ignore 區則是叫偵測器「不要從這個區域學習」——與 hard negative 的目的正好相反
- 我們要的訓練訊號正是：這塊區域是一個**未被忽略的背景指派**，而且它落在安全帽流形附近，
  因而迫使分類頭把邊界磨得更銳利。這件事只有在它是一個**普通負樣本**時才會發生

**合成期推論（規則 FILT-10）**：hard negative 的貼上**不得與任何保留的標註重疊**，
也不得把任何標註的可見比例壓到門檻以下。用一塊無標註的色塊去遮住一個真實標註物件，
會是**汙染標籤**，而不是磨銳邊界。

**指標推論**：hard negative 影像永遠不可能貢獻 recall，它只會移動 precision。
因此要把 **hard-negative 子集的每圖誤報數**當成獨立的一個數字報告。

### 後果

- 需要一個 HSV ＋ 輪廓圓度的挖料器，以及一個程序形狀渲染器（皆 Phase 1）
- 需要一次人工簽核，會出現在 `instructions_for_me.md` 的「換你做」清單
- hard negative 佔生成預算約 13%——夠大到能移動誤報率，且因為不增加標註所以很便宜

### 待查證

- **Spike H6**：挖料純度（見 `docs/synthesis_spec.md`）

### 來源

- [SHEL5K (Sensors 2022)](https://www.mdpi.com/1424-8220/22/6/2315)——未標註物件比例，是本 ADR 風險段落的依據

---

## ADR-005 — 速度對照組用 RF-DETR-Nano，不用 Ultralytics YOLO

### 脈絡

原始 Phase 2 計畫的第 4 項寫「YOLO11s 跑同樣四組當速度基準，README 註明 Ultralytics 採 AGPL-3.0」。

但 [ADR-001](#adr-001) 已經因為完全相同的理由否決過 Ultralytics：**AGPL-3.0 會讓我們
`import` 它的那支 Python 檔成為衍生作品而必須同樣 AGPL**，與 MIT repo 直接牴觸。
輸出的數字不受影響（程式輸出一般不受程式碼著作權涵蓋），但**程式碼還在 repo 裡**——
只在 README 註明授權擋不住這件事。

同時，速度對照在原始計畫裡本來就標為「選配」。

### 決策

**用 `Roboflow/rf-detr-nano`**（Apache-2.0，約 30.5 M 參數）當速度對照組，
透過 `transformers` 的 `RfDetrForObjectDetection` 執行。

三個理由：
1. **程式碼與權重都是 Apache-2.0**，與 MIT repo 相容
2. **邊際工程成本幾乎是零**：它是 `transformers` 的一等公民，
   官方文件明說同一支 `examples/pytorch/object-detection` 的 `Trainer` 腳本可直接用。
   我們的 collator、指標、評測 harness 全部原封不動重用——差別只有一個 checkpoint 字串
3. 它是目前的 speed/accuracy 前緣（ICLR 2026），所以這個對照是有意義的比較，
   不是找個弱者來墊背

⚠️ **只能用 nano / small / medium / base / large 這幾個變體**——
**XL 與 2XL 是 PML-1.0 不是 Apache-2.0**，不可使用。

### 備案

**D-FINE**（`ustc-community/dfine-nano-coco`，Apache-2.0，約 3.8 M 參數）。
一個 3.8 M 參數的偵測器放在 20.2 M 的 RT-DETRv2-R18 旁邊，
在「效率軸」上是很有資訊量的資料點，而且同樣是 `transformers` 原生支援。

### 被否決的選項

| 選項 | 否決理由 |
|---|---|
| Ultralytics YOLO11s / YOLOv10 | AGPL-3.0（YOLOv10 內含 Ultralytics） |
| YOLOX | Apache-2.0 沒問題，但**最後一次更新是 2025-06**、約 770 個開放 issue，且需要另一套訓練堆疊 |
| RTMDet / mmdetection | Apache-2.0（Roboflow 說它是 GPL-3.0 是**錯的**，那是 `mmyolo`）。但 **mmdetection 最後一次提交是 2024-08**，形同停止維護，而且 mmcv 在原生 Windows 上要編 CUDA extension，是災難 |
| RT-DETRv3 | 基於 PaddleDetection，導入成本高 |

### 後果

- 範圍限縮：**一個**額外模型、**一個** seed、與 real_only 相同的 epoch 數，列為次要結果
- 無論如何都要報 RT-DETRv2 自己在 4090 上的端到端 latency——
  那才是關心 PPE 佈署的讀者真正想看的數字，而且免費
- 驗證：`grep -rn "ultralytics" src/ scripts/ notebooks/` 必須零命中

### 來源

- [roboflow/rf-detr](https://github.com/roboflow/rf-detr)（Apache-2.0，2026-07-24 仍在更新）
- [RF-DETR in transformers](https://huggingface.co/docs/transformers/main/model_doc/rf_detr)
- [Peterande/D-FINE](https://github.com/Peterande/D-FINE)
- [open-mmlab/mmdetection LICENSE](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE)
- [Ultralytics License](https://www.ultralytics.com/license)

---

## ADR-006 — `transformers` 下限提高到 v5.14.1

### 脈絡

[ADR-001](#adr-001) 當時把下限訂在 `>=4.57.1`（為了 SAM2 的 box prompt 修正）。
Phase 2 的查證發現 **`transformers` 已經進入 v5**（v5.0.0 於 2026-01-26 發佈，
目前 5.14.1）。v5 有一個會直接咬人的改名：

| v4 名稱 | v5 名稱 |
|---|---|
| `RTDetrImageProcessor`（慢版，PIL/numpy） | `RTDetrImageProcessorPil` |
| `RTDetrImageProcessorFast`（torchvision） | **`RTDetrImageProcessor`** ← 現在是預設 |

也就是說在 v5 裡 `RTDetrImageProcessor` **就是**快版，`RTDetrImageProcessorFast` 不存在了。
**任何 2025 年寫的教學抄下來，要嘛 ImportError、要嘛行為悄悄不同。**

### 決策

**下限提高到 `transformers>=5.14.1`**，Phase 1 與 Phase 2 統一在 v5 API。

- SAM2（Phase 1）在 5.14.x 仍然存在且有文件，`>=4.57.1` 的 box prompt 修正早已包含在內
- 兩個 phase 用同一個 API 世代，比讓 Phase 1 停在 v4、Phase 2 跳 v5 單純得多
- **一律用 `AutoImageProcessor` / `AutoModelForObjectDetection`**，不要寫死類別名。
  checkpoint 的 `preprocessor_config.json` 會宣告正確的 processor 型別

### v5 的其他影響

- **`from_pretrained` 的預設 dtype 從 `float32` 改為 `"auto"`**——
  模型會以存檔時的精度載入，可能造成與 v4 的靜默數值差異。
  **明確傳入 `dtype=`**（注意 `torch_dtype=` 已棄用）
- TF / Flax 類別全部移除（純 PyTorch）
- 量化的捷徑 kwargs 移除

### ⚠️ 一併記錄的文件錯誤

HF 的 `rt_detr_v2` model 頁面，其 `RTDetrV2ForObjectDetection` autodoc 範例區塊寫著
`from transformers import RTDetrV2ImageProcessor` 與 `PekingU/RTDetrV2_r50vd`——
**類別名與 checkpoint id 兩者都不存在**。
（`rt_detr_v2/` 目錄裡根本沒有任何 image processing 檔案；正確的 repo id 是小寫的
`PekingU/rtdetr_v2_r50vd`。）同一頁上方的 *Usage tips* 區塊才是對的。
**不要複製 autodoc 那段。**

### 後果

- `pyproject.toml`、`docs/environment.md`、`CLAUDE.md` 三處的版本要同步
- Phase 1 的 SAM2 程式碼要用 v5 idiom 撰寫並明確指定 dtype

### 待查證（M1 執行時）

- 實跑 `python -c "import transformers; print(transformers.__version__)"` 確認 ≥5.14.1
- 實跑 `from transformers import Sam2Model, Sam2Processor` 確認 v5 下仍可匯入

### 來源

- [transformers PyPI 版本history](https://pypi.org/project/transformers/#history)
- [RT-DETRv2 model doc](https://huggingface.co/docs/transformers/main/model_doc/rt_detr_v2)
- [PekingU/rtdetr_v2_r18vd](https://huggingface.co/PekingU/rtdetr_v2_r18vd)

---

## ADR-007 — H1/H3/H5 實測：合規採類別直讀，split 採 guarded CLIP，人物採錨定放置

### 脈絡

M3 在 Kaggle version 1 的 5,000 張凍結來源上完成三個 spike。這三項若猜錯，
後續的合規邏輯、Train/Test 隔離與合成構圖都會安靜地失效。

### 決策

1. **合規走 `class_direct` 主定義。** 40 個 `helmet` 與 40 個 `head` 樣本的
   contact sheets 顯示：`helmet` 通常框住戴帽的整顆頭／臉，`head` 則框裸頭，
   不是「帽殼框＋頭框」的配對標註。同圖全部 9,603 個 `helmet×head` 組合中，
   只有 95 組（0.99%）的 IoU > 0.1；長寬比中位數也接近
   helmet 0.875／head 0.830，而非只有帽殼時預期的寬扁形。
2. **近似分群採 pHash Hamming ≤10，外加 OpenCLIP
   `ViT-B-32`／`laion2b_s34b_b79k` 的 guarded edge：
   cosine ≥0.85 且 pHash Hamming ≤20。** pHash 單獨仍留下 4,875 群，
   觸發協定的 CLIP 分支；12 組候選網格中，選定組合得到 4,808 群、最大群 8。
   最大 20 群已打開檢視：有明顯連拍，也有保守合併的同構場景，但沒有 component collapse。
   這種輕微過度合併只犧牲切分彈性，符合「避免近似圖跨 split」的不對稱風險方向。
3. **位置先驗分流使用。** provisional Train 的 16×16 中心熱圖顯示
   `helmet`／`head` 集中在畫面中央水平帶，可作取樣式先驗；
   `person` 的 normalized entropy 為 0.948，且只有數百個標註，
   因此人物與 crowded 構圖改以錨定放置為主，不讓稀疏直方圖主導。

### 後果

- Phase 2 的 compliance 預設使用類別本身作狀態，`person` 不承重；
  geometric pairing 只保留為診斷分支。
- FILT-07 沒有真實 `helmet-head` 配對可校準，只治理合成構圖；
  校準時改看 head/helmet 相對於 person 或畫面的分布。
- 凍結 manifest 必須記錄 pHash、CLIP 模型/tag、cosine 與 guard 四項；
  `same group -> same split` 是 hard fail。
- person cutout 的來源多樣性與錨點品質要分開報告，不能把稀疏位置直方圖當成可靠生成分布。

### 證據

- [M3 結構化摘要](../reports/data_spikes.json)
- [pHash 門檻報告](../reports/h3_grouping_spike.md)
- [guarded CLIP 候選報告](../reports/h3_clip_grouping_spike.md)
- `reports/figures/h1_{helmet,head}_contact_sheet.png`
- `reports/figures/h3_clip_largest_groups.png`
- `reports/figures/h5_placement_priors.png`

---

## ADR-008 — SAM2 Pass 2 採 effective crop-512，不直接把小框放大到 1024

### 脈絡

Spike H2 從凍結 Train 依最短邊取 60 個框，分為 8–20、21–34、36–133 px
三層；每個框比較全圖 prompt、context crop 直接放大至 1024、以及 crop
放大至 512 後以邊緣複製置中到 SAM2 原生 1024 canvas。三張 20-row grid
都已實際打開檢視。

### 決策

**Pass 1 仍使用全圖、同圖所有框重用一份 embedding；Pass 2 素材庫使用
effective crop-512。** 它在最小層的 IoU p10 / p50 為 0.850 / 0.887，
高於 crop-1024 的 0.799 / 0.865；輪廓目視至少相等，而且不會把少數來源
像素過度放大成塊狀 prompt。模型輸入形狀仍是原生 1024，不改 backbone。

排除五個目視明顯破碎或近乎空白的失敗後，55 個良好 crop-512 mask 的
`iou_scores` p10 為 0.821875、`object_score_logits` p10 為 19.5；
config 分別取 0.82 與 19.5。8–20 px 層沒有整體崩潰，因此硬下限不提高；
仍保留 16 px / 400 px² 的材質與有效 alpha 面積底線。

### 後果

- Pass 2 每個候選各算一次 embedding，不能和 Pass 1 混用快取。
- `resize_to: 512` 表示有效 crop 尺度；`model_canvas_size: 1024` 才是模型輸入。
- 小於 preferred tier 的素材可以保留，但合成抽樣必須偏好較大的來源。
- Pass 1 的全圖 mask 只供既有標註遮擋處理，不直接進 cutout bank。

### 證據

- [H2 報告](../reports/h2_sam2_spike.md)
- `reports/figures/h2_sam2_{very_small,medium,larger}.png`

---

## ADR-009 — H4 維持硬阻擋；FILT-11 排除自身背景

### 脈絡

M11 以當時設定產生上限內的 300 張候選圖。初版 H4 有兩個會灌高 AUC
的混淆因子：合成圖使用 JPEG、真實圖使用 PNG；以及真實對照只按類別抽樣，
未控制物件像素尺寸與 frozen group。兩者都修正後，另加入 soft-alpha
邊緣去污染，仍需判斷剩餘訊號是否足以關閉 scale-up gate。

同一輪 M12 顯示，若 pHash 把合成圖和「它自己的 Train 背景」比較，
copy-paste 必然因相近而大量被拒；這不代表跨樣本洩漏。
另外，cutout 已在素材庫經過 p1–p99 mask coverage gate，下游再次使用同一個
緊上限，會讓 ±20% 敏感度測試由該重複門檻主導。

### 決策

1. **M11 不通過。** 最終 H4 使用 lossless PNG、C=1 的 L2 logistic
   regression、frozen-group disjoint split，並在每個 fold 內按類別與
   log(pixel width, pixel height) 配對真實對照。2,028 個 patch 的
   HOG+HSV AUC 為 **0.7964**（bootstrap 95% CI 0.7481–0.8392），
   高於預先登記的 0.60。即使已修正 soft-alpha source-background halo，
   訊號仍明顯存在，因此不得用調寬門檻宣稱通過。
2. **FILT-11 的 real-image pHash 比較排除該合成樣本自己的背景 image id，**
   但仍和其餘所有 split 的真實影像比較。這使無意義的
   `NEAR_DUPLICATE_REAL` 首因由 169 降到 7，同時保留跨影像撞樣防護。
3. **下游 `helmet`／`head` mask coverage 上限設為 0.95。**
   素材庫的嚴格 p99 gate 不變；0.95 只作最後一道「mask 直接回傳 prompt box」
   的防護。調整後 300 張的 ±20% sensitivity alarm 為 0。
4. **每個 `sample_id` 使用由 root seed 與 sample index 經 SHA256 派生的獨立 RNG。**
   noise matching 即使不需加噪也消耗固定亂數 draw，避免某個光度分支把後續樣本
   的 cutout、位置或 post-effect 全部洗牌。H4 羽化消融因此能固定幾何與 fold。

### 後果

- M12 完成：300 = 196 pass + 104 reject，ledger 與 enum 七項檢查全過，
  12 pass／12 reject 圖已目視。
- M13 全量生成仍被 M11 硬阻擋；也同時等待 M9 的使用者 H6 簽核。
- 下一輪 H4 應預先登記人物脈絡錨定的構圖方法，並在新的 frozen group
  fold 驗證；不得改用較弱分類器或放寬 0.60 來過關。
- 事後 feature-family 診斷顯示 HOG-only AUC 0.7792、HSV-only 0.6816；
  兩類訊號都超標，下一個方法必須同時處理重採樣／邊界與光度，不宜再做
  單一 feather 參數搜尋。

### 證據

- [H4 報告](../reports/h4_artifact_gate.md)
- [H4 controlled ablations](../reports/h4_ablation.md)
- [H4 person-context diagnostic](../reports/h4_context_diagnostic.md)
- [H4 context-matched feasibility](../reports/h4_context_matched.md)
- [H4 context-replacement result](../reports/h4_context_replacement.md)
- [H4 context-replacement diagnostic](../reports/h4_context_replacement_diagnostic.md)
- [H4 feature-family diagnostic](../reports/h4_feature_family_diagnostic.md)
- [H4 Poisson result](../reports/h4_poisson_gate.md)
- [H4 exact-source-pair result](../reports/h4_source_pair.md)
- [H4 ROC 與 ranked patches](../reports/figures/h4_artifact_roc.png)
- [M12 ledger](../reports/filter_ledger.md)
- [M12 sensitivity](../reports/threshold_sensitivity.md)
- `reports/figures/filter_pass_reject_grid.png`

---

## ADR-010 — H4 Option A 採 FLUX.2 參照圖條件式邊界 inpainting

### 脈絡

羽化、multiband、Poisson、同類別原位替換與 exact-source 控制全部在 H4
超過 0.60，顯示手工混合器留下的邊界、重採樣與色彩訊號不能再靠參數搜尋
消除。kuotunyu 於 2026-07-27 批准擴充 Phase 1 的 Option A。

候選模型必須支援局部遮罩、參照物件、Windows Python 3.12 與 RTX 4090，
且授權可和 MIT 程式碼及預定資料 release 共存。FLUX.1 Fill 是
non-commercial，Qwen-Image-Edit 是 20B，SDXL Inpainting 則較舊且沒有同等
參照圖介面。`FLUX.2-klein-base-4B` 是 Apache-2.0；Diffusers 0.39.0 已提供
`Flux2KleinInpaintPipeline` 的 mask 與 `image_reference`。

### 決策

固定使用 `black-forest-labs/FLUX.2-klein-base-4B` revision
`a3b4f4849157f664bdbc776fd7453c2783562f4d`。只下載 Diffusers 所需的
18 個檔案，共 15,980,131,711 bytes（14.88 GiB），置於本專案 D 槽 cache，
避免和同時執行的其他專案共用可變模型目錄。下載需 kuotunyu 另行批准；
推論強制 local-only 並核對 manifest。

方法不是讓模型重畫整張圖：現有 compositor 先建立幾何正確的 draft，
模型只生成物件邊界帶，原物件 protected core 與遮罩外像素在輸出時逐像素
複製回 draft。模型因而不能靠刪除、改色、移位或全圖風格化通過 H4。

先跑 64 圖 identity pilot，由 kuotunyu 在不知道 AUC 的情況下簽核；通過後
才用新 seed 與新 group fold 執行一次 300 圖 H4。分類器、控制配對與 0.60
上限不變，看到結果後不得重選 seed、prompt、遮罩或模型。

### 後果

- 新增 `configs/generative_inpaint.yaml` 作為所有數值的唯一來源。
- 新增模型 metadata preflight、隔離下載、逐檔 hash manifest 與 local-only loader。
- M13 仍被擋住，直到模型下載、64 圖 identity gate 與 one-shot H4 都通過。
- 若 one-shot H4 失敗，本方法視為失敗；不得用同一 fold 做參數搜尋。

### 證據

- [Option A 預註冊](h4_generative_preregistration.md)
- [模型 preflight](../reports/generative_model_preflight.md)
- `src/synthetic/generative_inpaint.py`

---

## ADR-011 — H4 判定為「已知限制」而非無限阻擋；1× 生成並推進 Phase 2

> **取代 [ADR-009](#adr-009) 的第 1 點所隱含的「無限期阻擋」後果。**
> ADR-009 的**技術判定完全維持不變**，見下方第一段。

### ⚠️ 先講清楚我們沒有做什麼

- **沒有宣稱 H4 通過。** AUC 仍是 **0.7964**（bootstrap 95% CI 0.7481–0.8392），
  仍然高於預先登記的上限 0.60。
- **沒有放寬 0.60**，沒有換較弱的分類器，沒有改 fold 或特徵。
  ADR-009 明文禁止的三件事，一件都沒做。
- **沒有隱藏這個結果。** 它會出現在 README 正文、dataset card 與每一張成果表旁。

改變的是**「判定失敗之後要做什麼」**，不是判定本身。

### 脈絡

H4 這道閘門當初的**書面目的**是：
「先證明貼上痕跡不可被輕易偵測，再放大到上萬張」——
它是一道**避免浪費**的閘門，不是一個科學結論。

自 ADR-009 起，為了讓它通過已經嘗試並記錄了：
羽化參數搜尋、multiband、Poisson、同類別原位替換、exact-source 控制、
FLUX.2 參照圖條件式邊界 inpainting（ADR-010）、
whole-person 貼上的 v6／v6b／v7／v8、
regional placement 的 v9／v9b、
以及 whole-image v10 所需的 labeler v6 → v23 共 18 輪迭代。

**全部沒有通過**，且 feature-family 診斷顯示
HOG-only AUC 0.7792、HSV-only 0.6816——重採樣／邊界與光度**兩類訊號都超標**，
不是單一參數能修的。

同時，v23 labeler 的分析顯示它**嚴重欠訓練**
（best epoch 3、48 圖 audit 最高信心 0.14、TP 與 FP 的分數分布幾乎完全重疊，
中位數 0.0583 vs 0.0481——不存在能分開兩者的門檻）。
即使再修一輪，它服務的 v10 也只是第 10 條嘗試路線。

### 決策

**把 H4 的失敗當成本專案的一項實測發現，帶著它推進下游實驗。**

1. **生成量上限從 300 張提高到 1×（與真實 Train 約 1:1）。**
   **2× 不做**——那正是 H4 閘門要保護的「大規模投入」，
   在痕跡已知可偵測的前提下沒有理由燒下去。
   規模消融因此只做 **0.5× / 1×** 兩點，並在報告中說明為何沒有 2×。
2. **Phase 2 照常跑四組對照**，但每一張成果表都必須與 H4 AUC 並列呈現。
3. **要回答的問題因此變得更明確**，而且它本身就有價值：
   - 若合成資料**仍有提升** → 結論是「即使貼上痕跡對一個 HOG+HSV 分類器可偵測，
     它對下游偵測器的遷移仍然有效」。這是有意義的正面結果。
   - 若**沒有提升** → H4 AUC 0.7964 正好提供了機制解釋：
     偵測器學到的是貼上捷徑而非物件特徵。這是有機制的負面結果，
     比單純「沒效」有價值得多。
   - **兩種結果都可發表**，且都終結目前這個迭代迴圈。
4. **ADR-009 的其餘三點（FILT-11 排除自身背景、下游 coverage 上限 0.95、
   per-sample 派生 RNG）全部維持有效。**

### 誠實性要求（不可協商）

- README **正文**（不是只有 Limitations）必須寫出 H4 AUC 0.7964 與 0.60 的預先登記上限，
  並說明它是預先登記後未通過的閘門
- 每一個 Phase 2 的主張都要帶上「本合成資料的貼上痕跡對 HOG+HSV 分類器
  可達 AUC 0.796」這個限定
- 上面列出的失敗路線清單要進 README 的 Limitations——
  **9 條合成路線與 18 輪 labeler 迭代的失敗本身就是這個專案最誠實的部分**
- HF dataset card 同樣要載明

### 後果

- `M11` 標記為 **failed-and-accepted**（不是通過），`scale_up_allowed` 的語意改為
  「允許到 1×，禁止 2×」
- `M9` 的 H6 已由使用者以 0/64 簽核通過，不受影響
- v23 labeler **退回且不再迭代**；whole-image v10 路線一併停止。
  相關腳本與報告保留在 repo 內作為失敗路線的證據，不刪除
- `M13` 改為「1× pool 生成 ＋ 等量 filtered/unfiltered ＋ 0.5×/1× 巢狀子集」
- Phase 2（`M15` 起）解除阻擋

### 待查證

- 1× 生成完成後**重跑一次 H4**，確認 AUC 沒有因為樣本數變化而顯著移動；
  若移動，以新值為準並更新所有引用處

### 來源

- [ADR-009](#adr-009) 的完整 H4 證據鏈與 11 份報告
- v23 分析：`reports/supervised_labeler_v23_audit_evidence.json`
  （TP 91 / FP 13 / FN 19 已由獨立重算逐位驗證）
