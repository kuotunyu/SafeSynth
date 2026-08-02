# Worklog — 02-safesynth-ppe 施工日誌

<!-- 收工時做兩件事：(1) 覆寫「現況快照」 (2) 在「工作日誌」最上面插入一筆。 -->
<!-- 快照是待證偽的假設，不是真相。真相以 git log 為準——開工時務必交叉驗證。 -->
<!-- 本檔超過約 18 KB 時把舊日誌摺疊或歸檔（publish-repo gate 2 的門檻是 ~20 KB）。 -->

## 現況快照

*每次收工覆寫，只留最新一份。*

- **更新時間**：2026-08-02 上午（M20 ① 完成；GPU 被 llama-server 佔住）
- **最後驗證 commit**：`741fd6d` feat(demo): run the video path for the first time
- **目前里程碑**：Phase 1 全綠。Phase 2 的 **`M15`–`M19`、`M22` 完成**
  （`M22` 於 `741fd6d` 收掉——影片分頁與 CUDA 路徑第一次實跑，四條路徑全過），
  **`M20` 進行中**（只差 ① 那 4 分鐘的安靜量測，② RF-DETR 訓練未開始），
  **`M23` 是誠實的 `[~]`**（四條本機條件全綠，但「CI 為綠」需要先 push），
  `M21` 依前置判斷暫緩，`M24` 未開始。
- **⚠️ 未 commit 的改動**：無（收工時工作樹乾淨）。
- **最重要的一句話**：**合成資料在這個資料集上沒有提升，而且現在有信賴區間撐著這句話**——
  `real_only` 與兩個合成組的 95% CI **不重疊**。
  但同一批區間也**收回了兩個主張**：`real_only` 勝過 `standard_aug`（重疊，不成立）、
  `filtered_syn` 在偵測指標上勝過 `unfiltered_syn`（三個指標全部重疊，不成立）。
  過濾可量測的效果在合規操作點，不在 AP。
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
  GPU 於本輪後段空出並用於 M20 ①（延遲重測），其餘評測與分析在 CPU 上。
- **下一個動作（一句話、可直接動手）**：等使用者對 `instructions_for_me.md` 裡的 repo 體積問題（437 MB，其中 362.9 MB 是孤兒圖）做決定，那件事在第一次 push 之前處理才便宜。
- **卡住的事**：**M20 ② 卡在 GPU 被別的程式佔住**。使用者機器上有一個
  `llama-server`（不是我開的）吃掉 **23.5 GB / 90%** 的 4090，訓練跑不起來。
  程式已全部備妥：`configs/training_rfdetr.yaml` 可載入了、
  `scripts/probe_train_speed.py` 可量實測速度。GPU 一空出來就能跑。
  另外**時脈還鎖著**（`nvidia-smi -rgc` 要使用者執行）。

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

### 2026-08-02（上午）— M20 ① 收掉，以及變異測試第二次咬到我

- **對應規格**：DEMO-03、DEMO-05、TRAIN-01
- **M20 ① 完成**。使用者鎖了時脈之後，三道閘門第一次同時全綠：
  contention 0/9、clock spread **1.00**（門檻 1.15）、授權掃描 PASS。
  RT-DETRv2-R18 微調 3 類：**model-only 12.79 ms / 78.2 FPS、
  end-to-end 16.23 ms / 61.6 FPS**（batch 1、640、fp16、2520 MHz）。
  **鎖了之後仍重試 2 次才拿到 p95 乾淨的一輪**——鎖住的是頻率，不是其他行程。
- **`configs/training_rfdetr.yaml` 先前根本載入不了**。它是刻意只寫差異的檔案，
  但缺 17 個 `run:` 鍵，`yaml.safe_load` 之後 `run_arm` 第 18 行就
  `KeyError: 'per_device_eval_batch_size'`。
  改法是把關係寫成資料：加 `extends:`，`src/training/config.py` 做深層合併。
  兩個性質有測試撐著——**分節深層合併**（淺層會刪掉子檔沒提的 16 個鍵）與
  **子檔一定贏**（`do_normalize` 必須從 base 的 `false` 翻成 `true`，
  那是唯一一個「安靜繼承會毀掉模型而不是讓它崩潰」的設定）。
- **[K-21b](troubleshooting.md)：被 SIGKILL 的變異 harness 把變異留在工作樹裡。**
  我寫的測試會呼叫 `main()`，於是變異拿掉守衛之後**它真的開始訓練**，
  harness 卡住被 10 分鐘 timeout 殺掉，`finally:` 對 SIGKILL 無效。
  是我改同一個檔案時 grep 到那行長得不對才發現的。**這次沒有進 main。**
  兩個獨立的錯都修了：守衛抽成獨立函式（測試再也啟動不了昂貴作業），
  以及 commit 前**逐行看 `git diff` 而不是只看 `--stat`** 寫成強制動作。
- **驗證**：`1516 passed, 44 skipped`；ruff clean；`verify_readme` PASS；
  變異 10/10（config 合併）、12/12（速度斜率）。
- **卡住**：`llama-server` 佔住 23.5 GB VRAM，M20 ② 的訓練與速度實測都跑不了。
- **刻意不做**：沒有去關使用者的 `llama-server`——那是別人的程序，
  不是我該動的東西。也沒有在被佔用的 GPU 上硬跑量測然後宣稱那是實測。


### 2026-08-02（凌晨）— EVAL-09 補上了，代價是收回兩個主張

- **對應規格**：EVAL-09、DEMO-04、PUB-01
- **EVAL-09 完成**。1,000 次重抽 × 3 指標 × 4 組，單位是 Test **影像**。
  16 workers 實測 **2 小時 20 分**（單執行緒 9.3 小時）。
  平行化的關鍵不是加 process pool，而是**讓區間與 worker 數無關**——
  改用 `SeedSequence(seed)` 的獨立子種子，`(seed, k)` 唯一決定第 k 次抽樣。
  在真實的 223,200 筆偵測上驗過 1/8/12/16/20 workers，`BootstrapCI` 逐欄相等。
- **對帳結果比預期好**：重跑的 424 個點估計與已 commit 的主表**零不一致**。
  這也第一次證實了 [K-23](troubleshooting.md)：`--device cpu` 才重現得了已發佈的數字。
- **它收回了兩個主張，這是這一輪最重要的產出**：
  `real_only` vs `standard_aug` 區間重疊——0.0275 的點差在雜訊內，
  README 原本把它當排序；`filtered_syn` vs `unfiltered_syn` 三個偵測指標區間全部重疊。
  **只有「real_only 勝過兩個合成組」是區間不重疊、成立的。**
- **DEMO-04 的 GIF 產出**，而且不需要工地素材：資料集有 501 張同時含戴帽與裸頭。
  改成靜態幀蒙太奇（pHash 分群顯示 4,643/4,808 是單張，沒有連續影格可用），README 明說。
- **順手發現 demo 的 CUDA 路徑從來沒被執行過，而且是壞的**：processor 輸出 float32
  對上 float16 模型、`outputs.to("cpu")` 而 `ModelOutput` 沒有 `.to()`。都已修。
- **選圖錯兩次，都是打開圖才發現的**（詳見 PLAN M22）。第二次的教訓是
  「濾標註數」不等於「濾畫出來的框」——0.07 操作點下模型框數遠多於物件數。
- **M23 從 `[ ]` 改成 `[~]`**：四條本機條件全綠，但「CI 為綠」需要 workflow
  真的在 GitHub 跑過，而 `git remote -v` 是空的。勾 `[x]` 就是宣稱沒發生的事。
- **驗證**：`1453 passed, 42 skipped`；ruff clean；`verify_readme` PASS；
  變異測試 13/13（GIF 選圖）、14/14（時脈檢查）、10/10（PROVISIONAL 推導）。
- **verify_readme 這一輪抓到我兩次**：CI 欄位不是 CSV 裡的指標名（改成「點估計＋區間同格」
  它就驗得動了），以及我把 `unfiltered_syn` 的區間貼到 `filtered_syn` 那列。
- **主圖原本在宣稱一件區間不支持的事**。左面板把四根柱子**依數值排序**，
  而排序過的長條圖對讀者就是一個排名主張。加上 EVAL-09 的區間之後，
  **三組相鄰配對沒有任何一組在 95% 下分開**——成立的是非相鄰的那個比較
  （`real_only` 對任一合成組）。柱子現在帶區間，圖自己在頁腳點名重疊的配對。
  兩個渲染缺陷是打開圖才看到的：數值標籤原本畫在 value+0.008，
  加了 whisker 之後正好被蓋住；`standard_aug` 在 8.5pt 下與 `unfiltered_syn` 連在一起。
- **M22 收掉了**（`741fd6d`）。收掉的關鍵不是補完功能，是**去執行從沒被執行的路徑**：
  影片分頁自寫好以來一次都沒跑過，CUDA 分支第一次跑就連噴兩個錯。
  `annotate_video` 已抽成模組層函式——**只能透過瀏覽器到達的 callback 就是沒人跑的 callback**。
  圖片／影片 × CPU／CUDA 四條路徑加上「餵非影片檔」全部實跑通過，CUDA 峰值 VRAM 92 MiB。
  順帶更正一個沒量過就寫下的數字：docstring 寫「CPU 約 300 ms」，實測 **204 ms**；
  交接檔寫 1.2 秒，差 6 倍。
- **刻意不做**：沒有把 4 張 8–10 MB 的 `h4_*` contact sheet 轉成 JPEG 省 37 MB——
  那是 H4「貼上痕跡可不可偵測」的證據圖，在一張講痕跡的圖上引入壓縮痕跡不划算，
  而且改寫後的 repo 約 100 MB 本來就可以接受。
- **刻意不做**：沒有為了讓 speed probe 變綠而放寬 `max_clock_spread_ratio`，
  也沒有從三次 FAIL 的執行裡挑好看的數字。


### 2026-08-01（晚）— M20 ①：微調權重重測，以及一個會幫忙背書的驗收條件

- **對應規格**：DEMO-03、DEMO-05
- **做了什麼**
  1. `--weights KEY=PATH` 讓 harness 量本機微調權重（處理器仍取自 Hub，
     因為 Trainer 輸出目錄沒有 `preprocessor_config.json`）。
     模型確實換了：20.08 M 參數、logits `[1,300,3]`。
  2. **PROVISIONAL 標籤從寫死改成推導**。原本是一個常數字串——
     和這支腳本自己 docstring 裡罵過的「授權掃描寫死字串」是同一種錯。
     現在讀 `config.id2label`：只有當每一列都預測 `helmet/head/person` 才會消失，
     而且把 `--weights` 指到 COCO checkpoint 也不會讓它消失。
  3. 新增 `sm_clock_mhz` 欄位、`evaluate_clock_spread()` 與
     `benchmark.max_clock_spread_ratio: 1.15`。
- **驗證（實際輸出）**
  - `10/10 killed`（PROVISIONAL 推導 ＋ `--weights` 解析）
  - `14/14 killed`（時脈檢查與取樣位置）
  - `1428 passed, 42 skipped`；`ruff` All checks passed；`verify_readme` PASS
- **踩到的坑（[K-22](troubleshooting.md)）**：同一支 harness 幾分鐘內兩次跑出
  **11.81 ms** 與 **26.74 ms**，而 p95 檢查兩次都給 PASS——因為每一列一起變慢，比值不動。
  排除了 CPU 負載（11%）、P/E core 排程（綁核前後 26.5 vs 26.8 ms）、
  暖機不足（加 10 秒 matmul 前導**反而變慢**）。真正的變數是 SM clock：
  2520 MHz→12.89 ms、1215 MHz→27.85 ms。
- **我自己的第二個錯**：時脈檢查的第一版把取樣點放在計時迴圈**之後**。
  `nvidia-smi` 要 100 ms，期間 GPU 已在降頻，而且誤差**不是常數**——
  end-to-end 迭代結尾有 CPU 後處理，GPU 閒得更久。四列讀到
  2520/1770/2340/1680 MHz，報出一個 1.50 的假 spread。改到迴圈中點取樣，
  並丟棄被打斷的那次迭代。
- **刻意不做**：沒有為了讓報告變綠而放寬 `max_clock_spread_ratio`，
  也沒有從三次 FAIL 的執行裡挑一個比較好看的數字發佈。
  報告維持 FAIL，README 一個延遲數字都沒引用。
- **決策**：無新 ADR。


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
