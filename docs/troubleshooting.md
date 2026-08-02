# Troubleshooting

> 踩到的坑寫進這裡，一坑一節。格式：症狀 → 根因 → 解法 → 預防。
> 環境類問題可直接呼叫 `/safesynth-env` skill 走一遍自檢。

`K-01`–`K-09` 是 M0 查證時**預先寫入的已知風險**；
`K-10` 起是**實作中真的踩到的**，每一條都附實測數字。

---

## 已知風險（查證所得，實際上還沒踩到）

### K-01 — `torch.cuda.is_available()` 回 False

**症狀**：安裝成功、`import torch` 沒問題，但 CUDA 不可用，4090 完全沒被用到。

**根因**：PyPI 上的 `win_amd64` wheel 是 **CPU-only**（約 122 MB，
對比 Linux CUDA wheel 的約 527 MB）。PyPI 的檔案大小限制使得 Windows 的 CUDA 版本
從來不上傳 PyPI。裸的 `pip install torch` 會安裝成功且不報錯。

**解法**：確認 `pyproject.toml` 有 cu130 的 `[[tool.uv.index]]` 與 `[tool.uv.sources]`，
然後重裝：
```powershell
uv sync --reinstall-package torch --reinstall-package torchvision
```
驗證 `torch.__version__` 應含 `+cu130` 後綴。

**預防**：[ENV-02](environment.md)。永遠不要用 `pip install torch`。

---

### K-02 — SAM2 mask 品質莫名偏低

**症狀**：mask 看起來「差不多對但邊界糊」，`iou_scores` 約 0.94 而不是 0.98，沒有任何錯誤訊息。

**根因**：`transformers` 4.56.x 的 `_embed_boxes` 少了原始實作會補的 padding point。
這是**安靜的劣化**。

**解法**：升級到 `transformers>=4.57.1`（4.57.0 在 PyPI 被 yank，不要用）。

**預防**：[ENV-03](environment.md) / [ADR-001](decisions.md#adr-001)。
`/safesynth-env` 的自檢表第 3 列會檢查版本。

---

### K-03 — mask 有破洞或雜點，明明呼叫了後處理

**症狀**：`post_process_masks(max_hole_area=..., max_sprinkle_area=...)` 呼叫了，
但破洞和雜點還在。

**根因**：這兩個參數在 `transformers` 是**接受但完全不作用的 no-op**——
原始碼在該處留了一行註解，說明連通元件 kernel 是預計要補、目前沒有的。

**解法**：自己用 `cv2.connectedComponentsWithStats`（保留最大連通元件）
＋ `scipy.ndimage.binary_fill_holes` 做清理。參數見
`configs/compose.yaml` 的 `sam2.cleanup`。

**預防**：[CUT-08](synthesis_spec.md) / [ADR-001](decisions.md#adr-001)。

---

### K-04 — Kaggle 下載 401

**症狀**：憑證明明設了，`kagglehub` 仍回 401。

**根因（兩種）**：
1. `KAGGLE_API_TOKEN` 存的其實是舊式 `kaggle.json` 的 **JSON blob**，
   但新式憑證預期的是一個**純字串** token，於是整個 JSON 被當成 bearer token 送出
2. 已經正確設了 `KAGGLE_USERNAME` / `KAGGLE_KEY`，
   但格式錯誤的 `KAGGLE_API_TOKEN` 在解析順序中排在前面，把它們 shadow 掉了

**解法**：見 [ENV-05](environment.md) 的偵測與轉換程式碼。
關鍵是轉換完必須 `os.environ.pop("KAGGLE_API_TOKEN", None)`。

**預防**：ENV-05。

---

### K-05 — 讀 `KAGGLE_API_TOKEN` 拿到 KeyError

**症狀**：程式一開始環境變數還在，某個 import 之後就不見了。

**根因**：`kaggle/__init__.py` 在 import 時自動認證，並執行
`os.environ.pop("KAGGLE_API_TOKEN")`（kaggle-cli issue #882）。

**解法**：**不要 `import kaggle`**，改用 `kagglehub`。

**預防**：[ENV-04](environment.md)。`grep -rn "import kaggle" src/ scripts/` 應為零命中。

---

### K-06 — 行程無限增生 / 記憶體爆掉

**症狀**：跑平行處理時電腦卡死，工作管理員看到大量 python 行程。

**根因**：Windows 的 `multiprocessing` 用 **spawn** 不是 fork。
子行程會重新 import 主模組，若模組頂層就有啟動平行的程式碼，會無限遞迴。

**解法**：所有進入點包在 `if __name__ == "__main__":` 之內。

**預防**：[ENV-08](environment.md)。腳本預設 `num_workers=0`。

---

### K-07 — 兩次執行的 manifest SHA256 不同

**症狀**：`splits/MANIFEST.sha256` 每次跑都不一樣，凍結失去意義。

**根因（可能多重）**：
- 雜湊時用文字模式開檔（Windows 會改寫行尾）
- manifest 內存了 `WindowsPath` 的反斜線而非 `as_posix()`
- JSON 沒有用 canonical 形式（`sort_keys` / `separators`）
- 寫檔時沒指定 `newline="\n"`
- `image_id` 依賴了 `os.listdir` / `Path.glob` 的順序而非排序後的檔名

**解法／預防**：[ENV-10](environment.md) ＋ [DATA-09](data_protocol.md)、[DATA-18](data_protocol.md)。

---

### K-08 — PowerShell 指令直接 parser error

**症狀**：貼上一行含 `&&` 的指令，PowerShell 立刻報語法錯誤（不是執行失敗）。

**根因**：PowerShell 5.1 **沒有** `&&` 與 `||`。

**解法**：`A; if ($?) { B }`。

**預防**：[ENV-07](environment.md)。

---

### K-09 — 路徑過長錯誤

**症狀**：下載或載入模型時出現路徑相關的 IO 錯誤。

**根因**：本機 `LongPathsEnabled = 0`，260 字元上限是活的。
HF 快取的 blob 檔名很長，加上深層目錄容易超標。

**解法**：把 `configs/paths.yaml` 的 `hf_home` 設成短路徑（例如 `D:/hf`），
`kagglehub` 則用 `KAGGLEHUB_CACHE` 重導。

**預防**：[ENV-09](environment.md)。

---

## 實際踩到的坑

### K-10 — `UnicodeDecodeError: 'cp950' codec can't decode byte ...`

**症狀**：讀 `configs/*.yaml`、COCO JSON 或任何含非 ASCII 字元的文字檔時直接拋例外。
**本機 2026-07-27 實測重現**：
```
uv run python -c "import yaml; print(yaml.safe_load(open('configs/paths.yaml'))['data_root'])"
UnicodeDecodeError: 'cp950' codec can't decode byte 0xe2 in position 99
```

**根因**：這台機器的 Python 預設文字編碼是 **cp950**（繁體中文 Windows 的 ANSI codepage），
不是 UTF-8。只要檔案裡有任何非 ASCII 字元（我們的 config 註解、規格引用的破折號、
中文說明），裸的 `open(path)` 就會失敗。

**解法**：**所有文字檔的 `open()` 一律明寫 `encoding="utf-8"`**：
```python
with open(path, encoding="utf-8") as f: ...
Path(p).read_text(encoding="utf-8")
Path(p).write_text(s, encoding="utf-8", newline="\n")
json.dump(obj, open(p, "w", encoding="utf-8", newline="\n"), ...)
```
（`ET.parse(path)` 不受影響——它會自己處理 XML 的編碼宣告，這也是
[ENV-10](environment.md) 要求走路徑而非先 `read()` 的原因。）

**預防**：[ENV-10](environment.md)。可考慮在 CI 加一條 grep，
禁止出現不帶 `encoding=` 的 `open(` 文字模式呼叫。

---

### K-11 — Hard negative 看起來像貼上的色塊，不像場景中的物件（**已修**）

**症狀**：`review/preview_hard_negative_p1.png` 裡的 distractor 明顯「浮」在畫面上——
沒有接地陰影、懸在任意深度、光照與場景不一致，整體像貼紙而非實物。
使用者 2026-07-31 審查後裁決：「每張圖片裡的安全帽都像後製的圖片」。

**根因（原本的推測不完整）**：真正的主因不是缺陰影，而是
**distractor 完全沒有走標註貼上的光度管線**。標註貼上會做羽化、邊緣去汙、
Lab 局部調和、雜訊匹配四件事；`_paste_hard_negatives` 一件都沒做，
只有幾何變換加硬 alpha 合成。結果是**扁平、無紋理、硬邊的色塊**。

**客觀證據**（Laplacian 變異數量測物件表面紋理）：

| | p25 | p50 | p75 | p95 |
|---|---:|---:|---:|---:|
| 真實 helmet 表面 | 801.0 | **1350.9** | 2099.9 | 3885.4 |
| distractor（修前） | 15.4 | **52.4** | 229.0 | 1294.7 |
| distractor（修後） | 161.9 | **505.2** | 1195.9 | 1447.1 |

修前的表面紋理只有真實安全帽的 **1/26**。這就是「像後製」的物理量。
目視對照：`reports/figures/review/k11_hard_negative_before.png` 與
`reports/figures/review/preview_hard_negative_p1.png`。

**解法**（[COMP-20b](synthesis_spec.md)／[COMP-20c](synthesis_spec.md)）：
1. distractor 改走與標註貼上**完全相同**的光度管線
2. 加接地陰影。⚠️ **陷阱**：第一版把橢圓畫在 patch 矩形內，
   而 patch 貼合物件邊界，於是陰影被物件自己完全蓋掉——
   `changed` 剖面在物件底邊直接掉到 0。必須畫在**畫面座標**上並允許溢出

**刻意不做**：依深度的尺寸先驗。17,815 個真實標註擬合
`log(min_side) ~ cy` 得 b = −0.0350、R² = 0.0001（分桶中位數 28/27/22/23），
**這個資料集沒有深度—尺寸關係**，硬加只會讓分布更不像真的。

**預防**：`tests/test_composition.py -k contact_shadow` 三條，
其中一條專門斷言陰影落在物件footprint 之下。

**剩餘限制與裁決**：素材是程序生成的（[ADR-012](decisions.md#adr-012)），
表面紋理仍低於真實安全帽（503 vs 1351）；更重要的是**放置位置**仍可能落在半空中，
因為判斷哪裡有地面需要深度理解，而本資料集沒有可利用的深度線索。

**2026-07-31 由 kuotunyu 授權、Claude 裁決：接受，列為已知限制。**
理由有三：
1. [DATA-24](data_protocol.md) 確認**沒人戴的安全帽本來就不框**，
   所以干擾物不給標註與真實標註規則一致，**標籤語意是正確的**——
   浮空影響的是真實度，不是標籤正確性
2. 真實資料裡「該框 vs 不該框」的界線很乾淨，模型主要從那裡學；
   干擾物只是補強
3. 拿掉整個情境（佔 13%）的代價大於留著它

**代價（必須寫進 README）**：這些 distractor 偏 *easy* negative，
`hard-negative 每圖誤報數` 這個次要指標測到的東西會比預期弱。

⚠️ **這條修正不會改善 H4**。distractor 不進 `instances`，
排除消融已證實它們從來就不在 H4 的量測樣本裡。
這條修的是 `hard-negative 每圖誤報數` 那個指標的有效性。

---

### K-12 — 低光 post-fx 把整張圖壓黑，框卻還留著

**症狀**：`preview_head_no_helmet_p1.png` 第 02 格整張近乎全黑，
三個標註框（1 個貼上的 head、2 個真實 helmet）都框在看不到東西的地方。
使用者回報：「有 1 個 head 框，但物件是一團黑根本看不到」。

目視證據：`reports/figures/review/k12_blackout_evidence.png`
（左為合成結果、右為未動過的背景，同一個框位）。

**根因**：`configs/compose.yaml` 的 `postfx.low_light` 有兩個獨立採樣的參數
`gamma: [1.8, 3.2]` 與 `gain: [0.45, 0.80]`，兩者**相乘**。
該樣本抽到 gamma 3.17 × gain 0.54，把中灰（128）映射到亮度 16。
兩個範圍都標著 `source: guess`，而且**沒有任何檢查看輸出還剩下什麼**。

**規模**（修前的 1× pool，3,664 張接受圖）：
- 26,534 個標註裡有 **4,563 個（17.2%）** 比 99% 的真實標註更暗或更平
- **1,402 / 3,664（38.3%）** 的接受圖至少含一個這種框
- 其中 80.7% 施加過 post-fx
- 合成圖的整張平均亮度 p1 = 12.87，真實圖是 41.81

**解法**（兩層，缺一不可）：
1. **範圍改由真實資料推導**。要求變換把**中位數亮度的真實圖**映到真實圖亮度的
   暗尾（p0.5 到 p5）：gain 0.70 → gamma 2.3942 得 34.52；gain 0.95 → gamma 1.8067 得 70.11。
   工地照片不可能比資料集裡最暗的那張更暗，所以真實分布就是合理性的天花板
2. **每張圖自適應鉗制**。範圍是用中位數亮度校準的，本來就偏暗的背景連最輕的一檔都吃不下。
   `apply_postfx` 對強度做遞減掃描，取「所有保留標註都還在門檻之上」的最大強度。
   ⚠️ 雜訊場必須在掃描**之前**抽一次，否則掃描步數會改變 rng 狀態、破壞決定性
3. FILT-15 作為最後一道，量測**最終**影像

**副作用（刻意）**：`low_light_blur` 情境若強度被鉗制到 0，會**換背景重試**
（`LOW_LIGHT_FULLY_SUPPRESSED`）。否則那張圖會掛在一個承諾「暗」的情境底下卻不暗。
`summary.json` 的 `low_light_clamp` 會報告鉗制發生了幾次。

**預防**：`tests/test_composition.py -k postfx`。
更一般的教訓：**任何兩個相乘的 `guess` 範圍都要檢查乘積的後果**，不能只看各自合理。

---

### K-13 — 用小樣本估出來的接受率，放大後完全不準

**症狀**：以 2,000 張候選實測接受率 **58.4%**，據此估 10,000 張可得約 5,800 張，
實際只有 **3,377 張**（33.8%），低於 1× 需要的 3,500。
最大宗的拒絕原因從 `BOX_TOO_SMALL` 變成 `NEAR_DUPLICATE_SYNTHETIC`（2,809／10,000）。

**根因**：FILT-11 把每個候選拿去和**所有已接受的合成圖**比 pHash。
已接受集合越大，碰撞機率越高，**邊際接受率隨 pool 變大單調下降**。
接受率不是一個常數，是 pool 大小的函數。

**解法**：pool 大小要用**邊際**接受率估，不能用小樣本的平均接受率外插。
需要 N 張 accepted 時，先跑一次量出衰減曲線再決定候選數。
`MAX_POOL_IMAGES` 只是跑多久的上限，真正的科學約束是 `TARGET_ACCEPTED_1X`。

**這不是門檻太緊，是方法的天花板**：3,500 張背景、每張最多 4 次合成，
而貼上的物件又小，本來就會產生大量彼此近似的輸出。
**要記進 README**——它限制了這條路線能產出的最大有效資料量。

**預防**：任何「跑 N 張就會得到 M 張」的估算都要標明它是在多大的 pool 上量的。

---

### K-14 — 過濾器被自己的調和器繞過

**症狀**：FILT-15（標註可辨識度）上線、所有數值檢查通過之後，
重新出的預覽圖裡**還是有 head 框框著一塊黑斑**（`s42_010301`）。

**根因**：FILT-15 量的是**合成結果**，而 Lab 局部調和在它之前執行。

| | 素材亮度 | 合成後亮度 | FILT-15 判定 |
|---|---:|---:|---|
| `001610_ann008186` | **8.5** | **45.4** | **通過**（門檻 23.19） |

調和把純黑剪影的平均值抬過了門檻。
**調和搬動的是平均值，它不會生出細節**——抬亮剪影只會得到灰剪影。

**解法**：[CUT-14](synthesis_spec.md)。以同一組門檻量 **cutout 自己的像素**，
在 bank 載入時就排除，不要等到量合成結果。7,255 個素材排除 102 個（1.40%）。

**更一般的教訓**：**過濾器要放在被測性質仍然存在的位置。**
任何會改變被測量的處理（調和、post-fx、正規化）都可能讓下游的門檻失效。
排管線順序時要問：「在我量之前，有沒有東西動過我要量的那個量？」

**這條是目視發現的，不是測試發現的**——所有自動檢查都通過了。
CLAUDE.md 的「自己產的圖要自己打開檢視」不是客套話。

---

### K-15 — helmet→head 替換只繼承位置，沒繼承尺寸

**症狀**：`preview_head_no_helmet_p1` 裡出現**大到不成比例的頭**貼在人身上。
實測 `s42_011879`：貼上的 head 是 **52×68**，被取代的 helmet 只有 **24×30**
（面積約 5 倍），而同一場景另外兩頂安全帽是 34×40 與 28×33。

**根因**：`_build_sample` 的 `do_swap` 分支只設了 `center_override`，
**沒有設 `target_bbox_xywh`**——對照 `context_replacement` 分支兩個都設。
於是 `_transform_scale` 走到 `else` 分支，沿用情境的通用 `scale_range`。
[COMP-18](synthesis_spec.md) 的虛擬碼本來就寫了要縮放到 anchor 框，
**是實作偏離了規格**，不是規格沒寫。

**為什麼沒有任何過濾器攔下它**：
[FILT-08](filtering_spec.md) 的 head/person 尺寸比例需要一個 `person` 框才能比對，
而全資料集只有 **3.16%** 的圖有 person 標註。這條路徑上通常沒有可比對的對象。
**「有規則」不等於「規則會執行」**——條件式規則要問清楚它的前提多常成立。

**解法**：swap 分支一併設 `target_bbox_xywh = removed["bbox"]`。
修後實測 head_w/anchor_w 中位數 **0.949**、head_h/anchor_h 中位數 **1.000**
（修前那個案例是 2.17 與 2.27）。

**預防**：`tests/test_compose.py -k swap`，其中一條斷言
即使把 `scale_range` 放到極寬，只要給了 anchor，結果就不會變。

**這條同樣是目視發現的**，而且發生在 [K-14](#k-14) 之後——
連續兩個 bug 都是自動檢查全過、打開圖才看到。

---

### K-16 — 函式定義在 `if __name__ == "__main__":` 之後，執行時還不存在

**症狀**：`scripts/run_artifact_gate.py` 跑了 20 分鐘、**JSON 已經寫出來**，
最後在產生 markdown 那一行掛掉：

```
File "scripts/run_artifact_gate.py", line 177, in main
    f"- Source run: `{_repo_relative(run_dir)}` ({n_images} generated images)",
NameError: name '_repo_relative' is not defined
```

**根因**：`_repo_relative` 被定義在檔案**最底部**，也就是
`if __name__ == "__main__": main()` **後面**。
Python 由上而下執行模組層級的程式碼，`main()` 被呼叫時那個 `def` 還沒執行到。
而且底部有**兩份重複的定義**——修過一次但補錯位置。

**為什麼特別難發現**：這條路徑要跑完整個分類器（約 20 分鐘）才會走到，
而且**失敗發生在主要產物寫出之後**，所以 JSON 是好的、只有報告缺了。
單元測試不會執行 `__main__` 區塊，所以測試全過。

**解法**：把 helper 移到 `main()` **上面**，刪掉重複的那份。
加一條 AST 斷言比較 `_repo_relative` 與 `if __name__` 的行號來自查：
```python
assert helper_line < guard_line
```

**預防**：**模組層級只放 import、常數與 def；把 `if __name__` 區塊放在檔案最後一行**。
任何在它之後的 `def` 對 `main()` 而言都不存在。

---

### K-17 — RT-DETRv2 載入時噴一大串 missing / unexpected keys（**是良性的**）

**症狀**：`AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")`
印出兩大段嚇人的清單：

```
There were missing keys in the checkpoint model loaded:
  ['model.encoder.aifi.0.layers.0.self_attn.k_proj.weight', ...,
   'class_embed.0.weight', 'bbox_embed.0.layers.0.weight', ...]
There were unexpected keys in the checkpoint model loaded:
  ['model.encoder.encoder.0.layers.0.fc1.bias', ...,
   'model.decoder.layers.0.self_attn.out_proj.bias', ...]
```

`class_embed` 與 `bbox_embed` 出現在 missing 清單裡，看起來像是**偵測頭根本沒載到權重**。
如果真是這樣，訓練會照樣跑出漂亮的 loss 曲線，然後得到一個什麼都偵測不到的模型。

**根因**：這是 `transformers` **載入時自動改名**的正常行為，
清單印的是**改名前**的鍵名。checkpoint 存的是舊命名
（`self_attn.out_proj`、`fc1`），5.14.1 的模型類別用新命名
（`self_attn.o_proj`、`mlp.fc1`）。改名成功了，訊息只是沒講清楚。

**怎麼確認它是良性的**（不要用讀的，用測的）：
**不改類別數**載入原始 checkpoint（80 類 COCO），對一張真實工地照做推論。

```
hard_hat_workers2463.png 的偵測結果（門檻 0.30）：
  person 0.774 / person 0.627 / person 0.488 / sports ball 0.476 ...
```

抓到 5 個 person、最高信心 0.774。**隨機初始化的 decoder 不可能做到這件事**
（logits 會接近均勻，最高分落在 1/80 附近，0.30 門檻下什麼都不剩）。
順帶一提 `sports ball` / `frisbee` 就是安全帽——COCO 沒有 helmet 類別，
圓形彩色物件被歸到最接近的類。

**只有這 5 個張量是真的重新初始化的**（`MISMATCH` 那段才是真的）：
`enc_score_head.{weight,bias}`、`decoder.class_embed.{weight,bias}`、
`denoising_class_embed.weight`——因為 80 類換 3 類，本來就該重來。

**預防**：不要憑載入訊息判斷權重有沒有載進去，**跑一次推論看它會不會偵測**。
`scripts/smoke_train.py` 的存在就是為了在花掉 Colab 時數之前先跑過這條路徑。

---

### K-18 — smoke test 跳過 eval，四組 Colab 訓練全部陣亡

**症狀**：Colab 上四組**全部**在訓練約 2 分鐘後死在同一個地方，
而且**沒有跳出紅色 Error**（每組的例外被 `try/except` 接住，印完 traceback 就繼續下一組），
所以畫面看起來像在跑。最後 `完成的組別: []`。

```
File "/content/safesynth/src/training/run.py", line 114, in compute_metrics
    logits_batches.append(torch.as_tensor(logits))
RuntimeError: Could not infer dtype of dict
```

**根因（兩層，第二層才是真的）**：

**表層**：`eval_do_concat_batches=False` 時，Trainer 交給 `compute_metrics` 的是
**每個 batch 一個 tuple 的 list**，而那個 tuple 的 **index 0 是 loss dict，不是 logits**。
實測 `transformers 5.14.1` 的結構：

```
predictions: list（每個 eval batch 一個）
  predictions[i]: tuple，長度 14
    [0] -> dict，keys = loss_vfl / loss_bbox / loss_giou / ..._aux_*
    [1] -> ndarray (B, 300, num_labels)   ← logits
    [2] -> ndarray (B, 300, 4)            ← pred_boxes
```

我寫的是 `logits, boxes = batch[0], batch[1]`，抓到 loss dict 和 logits。

**深層（真正的錯）**：**`scripts/smoke_train.py` 當初設了 `eval_strategy="no"`**，
理由是「smoke test 不該花時間評測 756 張 val」。
於是 `compute_metrics` 這條路徑**本機從來沒有被執行過一次**。
訓練路徑測得很仔細（冷啟動、熱啟動、checkpoint 都驗了），評測路徑則是零覆蓋。

**解法**：
1. 用**形狀**而不是**索引**找 logits 與 boxes
   （`extract_logits_and_boxes`：末維 == num_labels 的是 logits，== 4 的是 boxes）。
   tuple 佈局已經變過一次，就會再變
2. **smoke test 一定要跑 eval**。改用 val 的一小片（預設 16 張）——
   夠快，但 batching、形狀萃取、COCOeval 三條路全部走過
3. `run_arm` 結束時明確跑一次 `trainer.evaluate()` 並把指標寫進 `run_record.json`，
   讓「eval 有沒有真的算出東西」變成可稽核的紀錄而不是假設

**代價**：浪費了約 12 分鐘 Colab（四組各約 2–3 分鐘）加使用者一次來回。

**教訓（比這個 bug 本身重要）**：
**smoke test 跳過的路徑就是沒有被覆蓋的路徑，不管測試套件多綠。**
當初為了「跑得快」關掉 eval，等於把最貴的驗證機會關掉。
會被 CI 綠燈掩蓋的缺口，通常就是你為了省時間主動關掉的那一塊。

**預防**：`tests/test_training_metrics.py` 用實測到的 14-tuple 結構（index 0 是 loss dict）
釘住四條測試，其中一條刻意把 tuple 重新排序，確認萃取不依賴位置。

---

### K-19 — 96% 分支覆蓋率、零 partial branch，四個一個 token 的 bug 全部存活

**症狀**：沒有症狀。這正是問題所在。

Phase 2 的四個模組（`detection.py`、`compliance.py`、`slices.py`、`benchmark.py`）
寫完當下全部綠燈、`ruff` 全清。`detection.py` 的分支覆蓋率是
**369 statements / 18 miss / 94 branch / BrPart = 0 / 96%**——
每一行都跑到了，每一個分支的兩個方向都走過了。

然後對原始碼做 mutation testing，把一個 token 改掉：

| 注入的變異 | 語意後果 | 測試反應 |
|---|---|---|
| `detection.py` `.index("small")` → `.index("medium")` | **`AP_small` 變成 `AP_medium`**（本專案主敘事指標 #1） | 38 passed |
| `detection.py` maxDets 軸 `-1` → `0` | 每圖只算 1 個偵測而不是 100 個 | 38 passed |
| `detection.py` bootstrap 尾端少掉 `/2` | 95% 信賴區間變成 90%，仍標示為 95% | 38 passed |
| `compliance.py` 同時拔掉 `_drop_person` 與 `HELMET_CLASS` 過濾 | **EVAL-03「`person` 不可承重」直接失守** | 37 passed |
| `compliance.py` IoU 門檻 `iou_threshold` → `-1.0` | 比對完全不看空間位置 | 37 passed |
| `slices.py` 聚合函式內乘上 `640/416` | **正是 EVAL-07 要防的那個 bug** | 47 passed |

**根因**：這是 [K-18](#k-18--smoke-test-跳過-eval四組-colab-訓練全部陣亡) 的下一層。
K-18 是「測試跳過的路徑沒有覆蓋」；這次是**「有測試，但那個測試不可能失敗」**。

四種具體形態，全部在本專案出現過：

1. **拿程式碼自己的輸出當期望值**
   `assert result.ap_small == compute_ap_small(gt, dt)`——兩邊是同一段程式。
   `config=None` 的預設路徑也一樣：只斷言它等於 `config=CONFIG` 的結果，
   等於拿程式跟自己比，只能抓到兩個呼叫端不一致，抓不到值本身是錯的
2. **fixture 退化到讓錯誤分支給出相同答案**
   `slices.py` 的每一個 fixture 框在乘上 `640/416` 之後**還在同一個 bucket**，
   所以 EVAL-07 的守門測試守不到真正產出報告的那些聚合函式。
   `build_pair_descriptor` 的 `dx` 也一樣：唯一的測試場景兩個中心都在 x=130，
   分子恰好是 0，於是分母寫成 `head_w`、`head_h` 還是 `helmet_w` 都得到 0.0
3. **只測 undefined 分支，不測有值的分支**
   `per_class_ap_small` 只有 `-1.0`（未定義）那條被斷言過，
   從來沒有任何測試斷言它等於一個**已知的數字**——
   所以 small 與 medium 換掉也沒人發現
4. **邊界從來不是決勝點**
   `>= min_compliance_precision` 改成 `>` 還是綠的，
   因為恰好落在門檻上的那個點在所有測試裡都不是贏家，它的資格從未被觀察到

**解法**：接受標準從「測試通過」改成**「注入的變異必須讓測試失敗」**。
每一條修好的測試都要當場做四步：注入 → 跑 → 必須紅 → 還原 → 必須綠。
沒跑過這四步就不算修好。

寫斷言的形態也跟著改：

```python
# 壞：assert result.ap_small is not None
# 壞：assert result.ap_small == compute_ap_small(gt, dt)   # 跟自己比
# 好：
# GT: one 30x30 head (900 px^2 -> small) and one 40x40 head (1600 px^2 -> medium).
# The detector hits the small one exactly and misses the other, so by definition
# AP_small = 1.0 and AP_medium = 0.0.
assert result.ap_small == pytest.approx(1.0)
assert result.ap_medium == pytest.approx(0.0)
```

第三種形態才有鑑別力：它**區分得出 small 和 medium**。前兩種區分不出來，
所以 `.index("small")` 的變異才活得下來。

**教訓**：**分支覆蓋率證明每一行都被執行，不證明任何一個回傳值是對的。**
在這個專案裡，「安靜地算錯」比「拋例外」危險得多——
因為約 2/3 真實物件未標註，我們本來就只能做相對比較，
一個被靜悄悄換成 `AP_medium` 的 `AP_small` 不會有任何一個數字看起來不合理。

**預防**：任何實作編號需求（`EVAL-*`／`FILT-*`／`TRAIN-*`）的函式，
新增測試時要先問一句「**如果我把這行改壞，這條測試會紅嗎？**」，
不確定就當場改壞試一次。這比再寫三條測試有用。

---

### K-20 — `run_record.json` 的 `eval_metrics` 對不上任何一個 checkpoint

**症狀**：`run_arm` 結尾那次 `trainer.evaluate()` 寫進 `run_record.json` 的
`eval_map` 是 **0.3312**，但這個值**既不是最佳也不是最後**：

| 來源 | `eval_map` | 本機獨立重算（CPU、fp32） |
|---|---:|---:|
| `checkpoint-1752`（`best_model_checkpoint`） | 0.3564 | **0.3597** ✓ |
| `checkpoint-10900`（最後） | 0.2657 | **0.2621** ✓ |
| `run_record.json` 的最終 `evaluate()` | **0.3312** | — ✗ |

兩個 checkpoint 檔案各自都對得上（差約 0.004，屬 bf16 訓練 vs fp32 重算的正常誤差），
**唯獨最終那次 evaluate 落在兩者之間、誰都不是**。

**判斷**：checkpoint 檔案本身忠實，問題出在 `load_best_model_at_end` 之後
在記憶體裡的那個模型。最可能是載入時部分權重沒被覆蓋——
[K-17](#k-17--rt-detrv2-載入時噴一大串-missing--unexpected-keys是良性的)
已經記錄過這個模型在載入時會出現 missing／unexpected keys，
`strict=False` 的載入會安靜地把沒對上的參數留在原值，
結果就是「最佳的骨幹 ＋ 最後的某些層」這種混合體，分數自然夾在中間。

**沒有繼續往 HF 內部追**，因為可執行的結論已經足夠明確且已驗證。

**解法（規則）**：
1. **一律從 checkpoint 目錄評測**，不要用訓練進程留下的記憶體狀態
2. **`run_record.json` 的 `eval_metrics` 只能當「這一組確實跑完評測」的存在性證明**
   （[K-18](#k-18--smoke-test-跳過-eval四組-colab-訓練全部陣亡) 的守衛用途），
   **不可用於任何報告數字**
3. 這條與 [EVAL-12](evaluation_spec.md) 本來就一致——主表所有數字都要由
   `scripts/eval.py` 在凍結 Test 上重算。K-20 是它為什麼不只是形式主義的實證

**這個坑的普遍形式**：**當你有兩條路可以得到同一個數字，就去比對它們。**
本例只花了兩趟各約 200 秒的 CPU 推論，就把「權重壞了」和「記錄壞了」分開——
而這兩者的後續處置完全不同（前者要重跑訓練，後者只要改讀取來源）。

---

### K-21 — `git add -A` 把背景 agent 注入的變異提交進 main

**症狀**：`scripts/verify_readme.py` 在 git HEAD 是
`if len(value) > _IDENTIFIER_MIN_LENGTH`，工作樹卻是 `>=`。
沒有人「改壞」它——**是 commit 把改壞的那一瞬間拍了下來。**

**經過**：背景有一個 workflow 在對 `verify_readme.py` 做變異測試，流程是
「注入 → 跑測試 → 還原」。我在同一時間為了提交 worklog 跑了 `git add -A`，
正好落在它注入 A07 變異（`>=` → `>`）而還沒還原的那個窗口。
於是 commit `4987841`（訊息只講 worklog）夾帶了兩樣不屬於它的東西：
被變異的 `verify_readme.py`，以及 agent 當時寫到一半的 193 行測試。

**後果是實測的，不是理論的**：`_IDENTIFIER_MIN_LENGTH = 4`，
而本機 `USERNAME` 的長度**正好是 4**。
用 `>` 的版本收集到 **0 個識別字**，於是 PUB-10 的洩漏掃描
**在這台機器上什麼都不搜尋**，卻照樣印 PASS。
用 `>=` 收集到 1 個。這是「安靜地什麼都不做」的檢查，
和 [K-19](#k-19) 講的假驗證同一族。

> 這段原本把使用者名稱直接寫出來當例子，**修好的掃描器立刻把這份文件本身擋下來**——
> 這是它上線後抓到的第一個真實命中，也順帶證明了它不是裝飾。
> 邊界值用「長度正好是 4」描述就夠了，不需要寫出那個字串。

**怎麼發現的**：不是我發現的。是變異驗證 agent 在比對 HEAD 與工作樹時撞見的，
並且它正確判斷「工作樹才是對的、HEAD 是被污染的那個」——
依據是模組自己的 docstring 與更早的兩個 commit 都寫 `>=`。

**解法**：把 `>=` 提交回去，並新增
`test_an_identifier_exactly_at_the_minimum_length_is_still_searched_for`
——用字面的 4 字元與 3 字元名字斷言邊界，**不 import `_IDENTIFIER_MIN_LENGTH`**
（import 常數就會變成拿程式跟自己比）。實測：把 HEAD 那版放回去，該測試立刻紅。

**預防（這是流程問題，不是程式問題）**：
1. **背景 agent 在跑的時候，不要用 `git add -A`。** 逐一列出自己動過的檔案，
   或至少先 `git status --porcelain` 確認每一個要進 commit 的檔案都是自己改的
2. commit 之後掃一眼 `--stat`：**出現在 stat 裡卻沒出現在 commit 訊息裡的檔案，
   就是夾帶進來的**。本例 `verify_readme.py` 與 `test_verify_readme.py`
   出現在一個標題是 `docs(worklog)` 的 commit 裡，光看 stat 就該起疑
3. 變異測試會「短暫地把程式改壞」，這是它的正常運作方式。
   它與任何自動 staging 的操作**天生互斥**

---

### K-22 — 同一支 benchmark 跑兩次差 2.3 倍，而既有的 contention 檢查看不見

**症狀**
`scripts/benchmark_latency.py` 在幾分鐘內、程式碼零改動的情況下跑兩次，
同一個模型的 model-only 中位數是 **11.81 ms** 和 **26.74 ms**。
兩份報告的 p95 / median 檢查都不覺得有問題，因為**每一列都一起變慢**——
比值不變，只有絕對值整體位移。

**排除掉的三個解釋**（各有實測，不是猜的）

| 假設 | 實測 | 結論 |
|---|---|---|
| CPU 被別的東西吃滿 | 總負載 11%，單執行緒 loop 0.81 s | 否 |
| 進程被排到 E-core（13700 有 P/E 混合核心） | 綁 P-core 26.52 ms、綁 E-core 26.77 ms、預設 27.11 ms | 否 |
| 20 次 warmup 不夠，需要更長的暖機 | 加 10 秒密集 matmul 前導**反而變慢**（27.85 vs 12.89 ms） | 否，且方向相反 |

**真正的原因**
延遲跟著 **SM clock** 走。同一組實測裡，2520 MHz → 12.89 ms、1215 MHz → 27.85 ms：
時脈比 2.07、延遲比 2.16。batch-1 偵測是低佔用率的突發負載，
桌機 GPU 的電源管理不認為它需要拉高時脈，而**要不要拉高取決於之前跑過什麼**。
用 CUDA event 量到 GPU 側時間 ≈ wall clock（12.17 vs 12.20 ms），
所以這是 GPU-bound 而不是 dispatch-bound——時脈就是那個變數。

**處置**
1. `LatencyResult` 多一個 `sm_clock_mhz` 欄位，**時脈變成表格的一欄**。
   理由和 batch／解析度／dtype 完全一樣：少了它，這個數字無法被重現。
2. 新增 `evaluate_clock_spread()` 與 `benchmark.max_clock_spread_ratio`。
   這是 **row 之間**的檢查，補上 p95 那個 **row 之內**檢查的盲點。
   兩個模型若在差距過大的時脈下量測，section 1 的對照是在比電源狀態、不是比網路。
3. **取樣點必須在計時區間之內。** 第一版寫在計時迴圈結束之後，看起來沒問題，
   實際上 `nvidia-smi` 要花約 100 ms，這段時間 GPU 已經開始降頻；
   更糟的是誤差**不是常數**——end-to-end 迭代結尾有 CPU 後處理，
   它的 GPU 比 model-only 多閒置一段。連續量測的四列因此讀到
   2520 / 1770 / 2340 / 1680 MHz，得出一個 1.50 的「spread」，
   那完全是溫度計拿的位置造成的假象。
   現在改在迴圈中點取樣，並**丟棄被打斷的那一次迭代**（它會帶著 subprocess 的成本）。

**目前狀態**
修好之後在這台機器上重跑三次，clock spread 分別是 1.50、3.11、3.65，
三次都判 FAIL。**這台機器在使用中無法產出可發佈的延遲數字**，
這不是 harness 壞掉，是 harness 終於看得見了。

要拿到可發佈的數字，需要鎖定時脈（**需要系統管理員權限，屬使用者親自執行**）：

```
nvidia-smi -lgc 2520,2520
```

量完之後解除：

```
nvidia-smi -rgc
```

**教訓**
`11.81 ms` 那個數字當初是「通過」的——p95 檢查給了 PASS。
一個只會檢查自己想得到的失敗模式的驗收條件，
在遇到沒想到的那一種時，會**主動幫忙背書**。

---

### K-23 — 重跑 `scripts/eval.py` 會得到「幾乎一樣但不完全一樣」的主表

**症狀**
用 `--bootstrap-resamples 2` 做煙霧測試，順手把 `results/detection_metrics.csv`
覆蓋掉了。比對之後發現**共同的 424 個鍵裡有 295 個數值不同**，
抽看的五個差在 5e-5 ~ 7e-4 之間。

**原因兩個，都不是 bug**

1. **裝置不同。** 已 commit 的主表是在 **CPU** 上算的
   （當時 GPU 被另一個專案佔用）。`scripts/eval.py` 沒指定 `--device` 時會自動選 CUDA，
   fp16 與 fp32 的捨入不同，偵測分數就會在小數第四位分家。
   **要重現已發佈的數字，必須 `--device cpu`。**
2. **少了 17 列。** `bare_head_recall_at_op`、`operating_point`、`op_*` 這幾列
   不是 `eval.py` 產的，是 `scripts/append_derived_metrics.py` 事後補上去的。
   重跑 `eval.py` 會把它們沖掉，**必須接著再跑一次那支腳本**。

**處置**
重跑主表的正確指令是兩步，不是一步：

```
uv run python -m scripts.eval --device cpu --bootstrap-resamples 1000 --bootstrap-workers 16
uv run python scripts/append_derived_metrics.py
```

**教訓**
「煙霧測試」用了正式的輸出路徑，就不是煙霧測試了。
這次靠 `git checkout --` 就救回來，是因為那份 CSV 已經進 git；
如果它還在工作樹裡沒 commit，我就把四組跑出來的結果洗掉了。

---

### K-21b — 被 SIGKILL 的變異測試會把變異留在工作樹裡

**K-21 的第二種形式，成因不同、後果一樣。**
K-21 是 `git add -A` 夾到變異注入的窗口；這次沒有人用 `git add -A`，
是**變異 harness 自己被殺掉**。

**發生經過**
`mutate_speed.py` 的 S11 把 `if args.long <= args.short:` 換成 `if args.long < 0:`，
跑測試，然後在 `finally:` 還原。但我當時寫的那條測試會呼叫 `main()`——
**拿掉守衛之後，`main()` 就真的開始跑訓練了**。
harness 卡在那裡，10 分鐘後被 timeout SIGKILL。
`finally:` 對 SIGKILL 無效，於是 **S11 的變異就留在檔案裡**。

我是在改同一個檔案、發現 `grep` 出來的那一行長得不對才察覺的。
`git diff` 一看就清楚：三十幾行合理改動裡夾著一行

```
-    if args.long <= args.short:
+    if args.long < 0:
```

**兩個獨立的錯，都要修**

1. **測試不該能啟動昂貴作業。** 把守衛從 `main()` 裡抽成獨立函式
   `validate_step_pair(short, long)`，測試直接打它。
   這樣任何變異都只會讓一條斷言變紅，不會讓機器開始訓練。
   通則：**變異測試會刻意把程式改壞，所以被測入口點必須是「壞掉也很便宜」的**。
2. **不能只靠 `finally:`。** 它擋得住例外，擋不住 SIGKILL、斷電、關機。

**強制動作（K-21 的規則升級版）**
commit 之前**一定要 `git diff` 逐行看過**，不是只看 `--stat`。
`--stat` 只會告訴你「這個檔案改了 31 行」，而混進去的那一行就藏在裡面。
凡是背景跑過變異 harness 的 session，這一步不可省略。

**沒有踩到的那顆地雷**
這次工作樹裡同時有 `src/training/config.py`（新檔）與另外三個修改檔。
如果我當時用 `git add -A` 直接提交，這個變異會**第二次**進到 main。
按檔名逐一 stage 這條規則不是形式主義——它讓我在 stage 之前非看 diff 不可。

**後續：它留下的不只是變異，還有一份看起來很像實測的假報告。**

清理完變異之後，`reports/` 底下多出兩個沒被追蹤的檔案：
`train_speed.md` 與 `train_speed.json`。它們是那次卡死的 harness 產生的——
守衛被拿掉之後，測試參數 `(short=10, long=0)` 一路穿過 `main()` 跑到底。

報告開頭自己寫著：

```
- Method: two runs (10 and 0 steps)
```

**用「0 步的訓練」減「10 步的訓練」去算每步成本**，然後外推到 10,900 步，
得出 **3.53 小時**。這個數字沒有任何意義，但它：

- 落在合理範圍（比 Colab L4 的 1.6–1.75 小時慢，聽起來很像 4090 的樣子）
- 附著裝置名稱、方法說明、對照基準，格式與真報告一模一樣
- 就躺在 `reports/` 裡等著被讀

**如果我沒去看那個「(10 and 0 steps)」，我會把它當成實測回報給使用者。**
CLAUDE.md 之所以規定時數必須來自實測，防的就是推算；
而這裡示範了更糟的一種——**推算會被人發現，假的「實測」不會**。

**兩條規則因此加上去：**
1. 變異 harness 跑完（或被殺掉）之後，除了 `git diff`，還要看
   **`git status --porcelain` 的未追蹤檔案**。變異期間產生的輸出全部不可信。
2. 報告類腳本的參數守衛要在**產生任何檔案之前**擋下來，而且守衛本身要能被
   單獨測試——不能只存在於 `main()` 裡。
