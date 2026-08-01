# Worklog — 02-safesynth-ppe 施工日誌

<!-- 收工時做兩件事：(1) 覆寫「現況快照」 (2) 在「工作日誌」最上面插入一筆。 -->
<!-- 快照是待證偽的假設，不是真相。真相以 git log 為準——開工時務必交叉驗證。 -->
<!-- 本檔超過約 18 KB 時把舊日誌摺疊或歸檔（publish-repo gate 2 的門檻是 ~20 KB）。 -->

## 現況快照

*每次收工覆寫，只留最新一份。*

- **更新時間**：2026-08-01（M16–M19 完成、demo 上線、README 有結果）
- **最後驗證 commit**：`38e5635` test(eval): make the error-analysis figures provable, not just green
- **目前里程碑**：Phase 1 全綠。Phase 2 的 **`M15`–`M19` 完成**，
  **`M20` 與 `M22` 進行中**（都只差需要 GPU 或需要你選素材的那一半），
  **`M23` 實質完成**（README 有結果與圖、`verify_readme` PASS、CI 已加兩道關卡），
  `M21` 依前置判斷暫緩，`M24` 未開始。
- **⚠️ 未 commit 的改動**：無（收工時工作樹乾淨）。
- **最重要的一句話**：**合成資料在這個資料集上沒有提升，四組主表已在凍結 Test 上算完並交叉驗證。**
  但它在真實標註稀少時確實有效——交叉點約在 4 輪真實資料。
- **已凍結不得再動**：`splits/split_manifest.json`、`test_blocklist.json`、
  `source_checksums.json`、`MANIFEST.sha256`（SHA256
  `ce9d76ee336cfba5e6071727442f7af413a8372f28cc9882093cb784587287a3`）；
  再加上 **`configs/evaluation.yaml` 的 `compliance.score_threshold: 0.07`**——
  EVAL-04 在 Validation 上選定後即凍結，**看過 Test 之後再改就是在 Test 上調參**。
- **資料與素材落地**：Colab 四組權重解壓在 `D:\sdg-data-safesynth
uns\<arm>\seed_1337\`
  （每組 best ＋ last 兩個 checkpoint）；八份偵測結果（4 arms × test/val）在
  `runs/predictions/`，索引在 `results/predictions_index.json`。
  **`results/detection_metrics.csv` 現在是被追蹤的**（441 列）——EVAL-12 要求所有報告數字
  都能從它重算，`scripts/verify_readme.py` 與 CI 都以它為準。
- **環境**：不變。Python 3.12.13、torch 2.13.0+cu130、transformers 5.14.1。
  **本輪全程未使用 GPU**（另一個專案佔用中），評測與分析都在 CPU 上跑，
  744 張 × 4 組約 13 分鐘。
- **下一個動作（一句話、可直接動手）**：等使用者對 `instructions_for_me.md` 裡的 repo 體積問題（437 MB，其中 362.9 MB 是孤兒圖）做決定，那件事在第一次 push 之前處理才便宜。
- **卡住的事**：無阻擋。
- **⚠️ 已知限制（必須寫進 README，且已經寫了）**：
  1. **H4 未通過（AUC 0.9053，上限 0.60），而訓練結果與它的警告一致。**
     這是預先登記的閘門確實有預測力，不是事後找的藉口
  2. **這一輪沒有跑 bootstrap**（`--bootstrap-resamples 0`），所以它不滿足 EVAL-09，
     報告已明載。全部單一 seed，EVAL-10 禁止用零點幾個點宣稱勝出
  3. **EVAL-16 無法產生**：凍結 Test 的 744 張**全都**含 helmet 或 head，
     hard-negative 子集是空的，而候選區域 fallback 未實作
  4. 模型**校準很差**：223,200 個偵測的最高分只有 0.2495。排序good、絕對分數不可用
  5. 接受率天花板（K-13）與 hard negative 放置（K-11）維持原狀
  6. **repo 有 437 MB**，其中 362.9 MB 是 116 個沒有任何文件引用的 Phase 1 診斷圖。
     從 HEAD 刪不會讓 clone 變小，要 `filter-repo`，而且**要在第一次 push 之前做**
- **等使用者做的事**：見 [instructions_for_me.md](../instructions_for_me.md)。
  遠端 GitHub repo 仍未建立（發佈時才需要）。
- **驗證本快照的指令**：
  ```
  uv run python -m scripts.audit_colab_results
  uv run python -m scripts.verify_readme
  uv run python -m scripts.check_forbidden_licences
  uv run ruff check .
  uv run pytest -q
  ```

---

## 工作日誌

*append-only，新的插在最上面。*

### 2026-08-01（下午）· demo、README 結果、以及一個我自己造成的 main 污染

- **對應規格**：DEMO-01~03、PUB-01~05、EVAL-15
- **做了什麼**：Gradio demo（M22）、README 補上結果與 headline 圖、
  CI 加上 README 驗證與授權掃描兩道關卡、error_analysis 渲染層的測試補強。
- **驗證（實跑輸出）**：
  - `uv run python -m scripts.verify_readme` → **PASS**（441 列 CSV、37 份文件）
  - `uv run pytest -q` → **1389 passed, 42 skipped**；`ruff check .` 全清
  - demo 對兩張真實 Test 影像跑完整路徑並**打開圖檢視**：
    12/15 與 6/15 compliant，輸出在 `reports/figures/demo_examples/`
  - error_analysis 的 27 條指定變異 ＋ 我自己補的 3 條，**全部killed**
- **踩到的坑**：
  - [K-21](troubleshooting.md)：**我用 `git add -A` 把背景 agent 注入的變異提交進 main。**
    後果是 PUB-10 的洩漏掃描在這台機器上什麼都不搜尋卻照樣印 PASS
    （`_IDENTIFIER_MIN_LENGTH = 4`，本機 USERNAME 長度正好 4，`>` 就收集到 0 個）。
    不是我發現的，是變異驗證 agent 比對 HEAD 與工作樹時撞見的。
  - 修好的掃描器**第一個真實命中就是 K-21 那篇文件自己**——我在裡面寫了字面的使用者名稱。
  - 我補 A18 測試的第一版**仍然抓不到那條變異**：我另外呼叫 `plan_comparison_grid`
    來斷言，那是測葉子不是測那條線。改成攔截協作者、斷言它收到什麼才killed。
- **量過但決定不做的事**：
  - **EVAL-09 的 bootstrap**。CPU 上一輪 COCOeval 12.0 秒，1000 次 × 4 組 = **14.1 小時**，
    而且只是一個指標。試過截斷偵測（2.7× 加速）但**不是無損的**——
    pycocotools 的 maxDets 逐類別套用，整圖取前 100 會砍掉該入榜的框，mAP 差 2.6e-04。
    主表關鍵差距 0.085 遠大於雜訊下限 ±0.031，方向性不成疑問，不值得 14 小時。
  - **M21 補 seed**：PLAN 的前置判斷已寫明，合成沒贏時補 seed 不改變結論。
- **量到但還沒處理的事**：**repo 有 437.5 MB**，其中 116 個檔案、362.9 MB
  沒有任何文件引用（Phase 1 已放棄路線的診斷圖）。刪除是不可逆的，
  而且要真的瘦身得 `filter-repo`——依規矩是使用者的動作。已寫進交接文件。
- **commit**：`71ac177`、`6f5a906`、`17e5153`、`8368cda`、`7f24640`、`fa1b327`、`38e5635`

### 2026-08-01 · Phase 2 主線：四組結果出爐，合成沒有提升

- **對應規格**：TRAIN-15~18、EVAL-01~18、PUB-01~05
- **做了什麼**：回收 Colab 四組產出並稽核（M16）、合規操作點（M17）、
  凍結 Test 主表（M18）、四類錯誤分析（M19）、README 與 CI 關卡（M23 大部分）。
  全程未使用 GPU。
- **驗證（實跑輸出）**：
  - 盤點：**PASS，0 fatal / 4 warning**。四組 `real_train_digest` 全同、步數皆 10,900、
    filtered 與 unfiltered 皆 3,500。
  - 主表（凍結 Test 744 張，各組取自己最佳 checkpoint），primary = helmet+head：

    | arm | primary AP_small | primary mAP | 真實影像曝光 |
    |---|---:|---:|---:|
    | real_only | **0.4511** | **0.5341** | 49.83 |
    | standard_aug | 0.4236 | 0.4958 | 49.83 |
    | unfiltered_syn | 0.3759 | 0.4597 | 24.91 |
    | filtered_syn | 0.3664 | 0.4858 | 24.91 |

  - **兩條獨立實作一致到 8.8e-07**（`scripts/eval.py` vs 一支獨立的 scratchpad 腳本）。
  - 防洩漏實跑：`assert_test_untouched()` 過 744 張；四組訓練 digest 皆等於凍結 train split。
  - `uv run pytest -q` → **1292 passed, 41 skipped**；`ruff check .` 全清。
- **結論與分析**：
  - **合成沒有提升。** 依鐵律如實報告，不做選擇性呈現。
  - **但它在標註稀少時有效**：改用「真實影像曝光」而不是 optimizer 步數當 x 軸，
    `filtered_syn` 在 1–4 輪領先最多 **+0.090 mAP**，第 4–5 輪被追過。
    本資料集有 5,000 張標註，正是合成增強最無用武之地的區間。
    **注意**：對齊曝光就對不齊算力，每一列都是「相同標註、更多計算」。
  - **過濾的價值在合規操作點上最清楚**：各組各選各的門檻後，
    **`unfiltered_syn` 在任何會偵測到東西的門檻上都達不到 0.80 精確度**，
    `filtered_syn` 可以（0.8076）。過濾決定了能不能部署。
  - **針對性失敗，而且比無效更糟**：`small_object` 是**移動最不利**的切片（−0.0572）。
  - **退步是不對稱的**：修好 73 個漏檢、新製造 1,304 個。
- **決策**：無新 ADR。ADR-011 的預測（H4 未過 ⇒ 可能無提升）得到證實。
- **踩到的坑**：
  - [K-20](troubleshooting.md)：`run_record.json` 的 `eval_metrics` 與任何 checkpoint 都對不上。
    兩趟各 200 秒的 CPU 推論就把「權重壞了」和「記錄壞了」分開——處置完全不同。
  - 打包程式漏抓全部四組的 `trainer_state.json`，**完全沒報錯**：
    HF 寫在 `checkpoint-N/` 裡，glob 只掃 `seed_*/`，`is_file()` 把「找不到」變成「跳過」。
    這段邏輯當初寫在 notebook 字串裡，測不到。已移進 `src/training/ingest.py`。
  - `compliance.score_threshold: 0.50` 這個佔位值對本模型是災難（最高分 0.2495）。
  - `select_operating_point` 會選出「從不觸發」的退化解（precision 1.0、recall 0.0）。
  - bare-head recall 在門檻 0 上四組差距只有 0.0023，**沒有鑑別力**；
    改在操作點上讀，差距變成 0.54。已寫成 EVAL-05b。
- **刻意不做**：
  - **M21 補 seed 暫緩**。PLAN 的前置判斷寫得很清楚：若 Filtered 組沒有提升，
    補 seed 不會改變結論。額度留給錯誤分析與後續的 real-fraction 消融。
  - bootstrap 用 0 跑（因此這一輪不滿足 EVAL-09），因為 1000 次重抽 × 每次一輪 COCOeval
    在 CPU 上是好幾小時，而它不改變方向性結論。這件事報告有寫。
- **commit**：`7f2f0b3`、`bc44f3f`、`735111a`、`8c8a240`

### 2026-07-31 · K-11 裁決為「接受」；補寫 DATA-24 標註語意

- **對應規格**：[DATA-24](data_protocol.md)、[K-11](troubleshooting.md)
- **起因**：使用者問「這個任務是要框安全帽還是框人頭？放在桌上的安全帽要框嗎？」
  ——這是最根本的語意問題，而**規格裡查不到答案**。
  H1 的**做法**寫在 `data_protocol.md`、**結論**只散落在
  [ADR-007](decisions.md#adr-007) 與 `evaluation_spec.md`，
  而「沒人戴的安全帽要不要框」**從來沒有任何文件回答過**。
- **實測回答**（渲染真實圖 + 原始標註驗證）：
  - `helmet` 框的是**戴著安全帽的整顆頭**，`head` 是**沒戴帽的頭**，兩者互斥
    （9,603 個同圖配對只有 95 個 IoU>0.1 = 0.99%；
    長寬比中位數 helmet 0.875／head 0.830，都是高>寬，不是帽殼的寬扁形）
  - **沒人戴的安全帽不框**。`image_id=4029` 會議桌上 3 頂紅色安全帽，
    標註是 `head=8`、**`helmet=0`**；對照 `1629`／`3803` 的 helmet 框都在戴帽的人身上
  - 證據圖 `reports/figures/review/loose_helmet_question.png`
- **自我更正**：前一則日誌寫「GT 自己前後不一致（1629、3803 有框，4029 沒框）」，
  暗示 GT 對未佩戴安全帽的處理不一致。**把這三張圖連同標註渲染出來後，
  它們其實是一致的**——1629／3803 框的是戴著的帽，4029 沒框桌上的帽，
  完全符合「只框戴著的」這條規則。原本的說法沒有再驗證就寫進日誌，已更正。
- **裁決**：K-11 的 hard negative 放置真實度 → **接受，列為已知限制**。
  使用者授權 Claude 判斷。關鍵理由是 DATA-24：
  既然真實資料本來就不框沒人戴的安全帽，**干擾物不給標註在語意上完全正確**，
  浮空影響的是真實度而非標籤正確性。代價限縮在
  `hard-negative 每圖誤報數` 這個次要指標，主敘事指標不受影響。
- **順手修**：`reports/figures/` 有 72 個檔案 + 18 個資料夾，
  使用者找不到要看的 12 張。改成 `reports/figures/review/`（只有 15 個檔案 ＋ README）。
  原本想把歷史檔案搬進 `archive/`，但那會弄壞 21 個測試——
  supervised labeler 的審查頁是**凍結證據，路徑與 SHA 都釘死**，已回退該部分。
- **驗證**：`uv run pytest -q` → **643 passed / 25 skipped**；`uv run ruff check .` → 全綠
- **commit**：見下一筆

### 2026-07-31 · 使用者審查揪出四個缺陷；重生成 M13 pool

- **對應規格**：FILT-15、CUT-14、COMP-18、COMP-20b/20c、[ADR-013](decisions.md#adr-013)
- **起因**：kuotunyu 審查 `preview_head_no_helmet_p1` 與 `preview_hard_negative_p1` 後回報三件事。
  追查結果是**四個不同的缺陷**，其中兩個是我在修的過程中自己造成或發現的。

| # | 回報／發現 | 根因 | 處置 |
|---|---|---|---|
| 1 | 「整張圖黑掉」 | `low_light` 的 gamma×gain 兩個 guess 範圍相乘，實測 3.17×0.54 把中灰壓到 16。**38.3% 的接受圖**中招 | 範圍改由真實圖亮度推導 ＋ 每張圖自適應鉗制 ＋ FILT-15（K-12） |
| 2 | 「hard negative 像後製」 | distractor **完全沒走**標註貼上的光度管線（羽化／去汙／調和／雜訊匹配四樣全無） | 併入同一管線 ＋ 接地陰影（K-11） |
| 3 | 自查發現：FILT-15 上線後**還是**有 head 框著黑斑 | Lab 調和在 FILT-15 之前跑，把素材亮度 8.5 的剪影抬到 45.4 過門檻 | 加 CUT-14 在素材端擋（K-14） |
| 4 | 自查發現：頭大到不成比例 | `do_swap` 只設 `center_override`，沒設 `target_bbox_xywh`，**實作偏離 COMP-18 規格** | swap 一併繼承 anchor 尺寸（K-15） |

- **驗證（實測輸出）**：
  - 表面紋理（Laplacian 變異數，真實 helmet p50 = 1350.9）：distractor 由 **52.4 → 503.3**
  - swap 頭尺寸：`head_w/anchor_w` 中位數由 2.17 → **0.949**，`head_h/anchor_h` 由 2.27 → **1.000**
  - CUT-14 實際排除 **102 / 7,255**（head 27／helmet 74／person 1）
  - 低光鉗制：2,184 次抽樣中 681 次維持全強度、1,368 次降強度、135 次完全抑制
  - `low_light_blur` 仍然真的暗：整圖亮度 p50 = 98.2 vs 其他情境 123–131
  - pool：14,000 候選 / **4,177 接受**（29.8%），COCO 自評 mAP = **1.000**
  - 四份子集六個不變式全過；filtered 與 unfiltered 各 3,500，`0.5× ⊂ 1×`
  - `uv run pytest -q` → **330 passed**；`uv run ruff check .` → All checks passed
- **決策**：[ADR-013](decisions.md#adr-013)（四個決策 ＋ 三個「量測後否決」）
- **踩到的坑**：K-11（改寫）、K-12、K-13、K-14、K-15
- **刻意不做**：
  - 依深度的尺寸先驗——`log(min_side) ~ cy` 的 R² = **0.0001**，本資料集沒有這個關係
  - 框內 RMS 對比當閘門——它量的是框的鬆緊，會誤殺又小又亮的遠距安全帽
  - |物件−周圍| 亮度差當閘門——真實物件的 p1 只有 **0.67**，真實差異在色相不在亮度
  - 刪掉那個 5×30 的扁 helmet 框——那是真實標註，鐵律禁止刪，且 Real-only 組也有
  - 修「暗頭髮貼到暗背景」——物件亮度 32.2 落在真實 head 分布內（p1 = 23.19），
    門檻拉到抓得到它就會連真實資料最暗的裸頭一起丟
- **教訓**：連續兩個缺陷（#3、#4）都是**自動檢查全過、打開圖才看到**。
  #4 更進一步顯示「有規則」不等於「規則會執行」——FILT-08 需要 `person` 框，
  而全資料集只有 3.16% 的圖有。
- **commit**：`974df2e`、`c96339f`
