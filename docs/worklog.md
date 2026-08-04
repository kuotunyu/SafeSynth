# Worklog — 02-safesynth-ppe 施工日誌

<!-- 本檔已於 v1.0.0 發布後凍結為最終稽核紀錄，不再追加逐日施工日誌。 -->
<!-- 現況快照以 Git／GitHub／Hugging Face 的公開狀態交叉驗證；舊條目保留當時語境。 -->

## 現況快照

*每次收工覆寫，只留最新一份。*

- **更新時間**：2026-08-05（v1.0.0 正式發布與最終唯讀驗收完成）
- **v1.0.0 source checkpoint／tag target**：`abbf5a7` ci: pin Node 24 setup-uv action
- **目前里程碑**：Phase 1 全綠。Phase 2 的 **`M15`–`M20`、`M22`–`M24` 完成**
  （`M22` 於 `741fd6d` 收掉——影片分頁與 CUDA 路徑第一次實跑，四條路徑全過），
  `M20` 的 fine-tuned RF-DETR 延遲五次固定時脈量測都未通過 contention gate，
  因此依預先登記規則撤回速度主張、以負面驗證結果結案，
  **`M23` 已完成**，最終 source checkpoint 的 GitHub Actions run `30940634079`
  全部通過且 annotations 為 0；
  `M21` 因沒有得到受支持的 Filtered 提升而依條件結案，不補 seed；
  **`M24` 已完成**（GitHub `v1.0.0` annotated tag／Release 與兩個 HF repo 均已公開，
  並通過發布後唯讀驗收）。
- **目前待 commit 的改動**：只剩本次發布後狀態紀錄與公開 Release 連結；沒有程式、
  實驗結果、模型或資料包變更。模型權重與 1.94 GB 資料包仍只在
  `<data_root>/publish/`，不進 Git。
- **最重要的一句話**：RT-DETRv2 的合成組顯著較差；RF-DETR-Nano 的合成組
  點估計稍高但四組 95% CI 全部重疊。兩個架構方向不一致，因此目前只有
  **「沒有穩健、可泛化的合成資料提升」**這個結論，不能宣稱 RF 已證明勝出。
- **已凍結不得再動**：`splits/split_manifest.json`、`test_blocklist.json`、
  `source_checksums.json`、`MANIFEST.sha256`（SHA256
  `ce9d76ee336cfba5e6071727442f7af413a8372f28cc9882093cb784587287a3`）；
  再加上 **`configs/evaluation.yaml` 的 `compliance.score_threshold: 0.07`**——
  EVAL-04 在 Validation 上選定後即凍結，**看過 Test 之後再改就是在 Test 上調參**。
- **資料與素材落地**：RT-DETRv2 四組權重位於
  `<data_root>/runs/<arm>/seed_1337/`（每組 best ＋ last）；
  八份偵測結果（4 arms × test/val）索引在 `results/predictions_index.json`。
  **`results/detection_metrics.csv` 現在是被追蹤的**（441 列）——EVAL-12 要求所有報告數字
  都能從它重算，`scripts/verify_readme.py` 與 CI 都以它為準。
  RF 複驗另有 `results/rfdetr_detection_metrics.csv`（424 列）與
  `results/rfdetr_predictions_index.json`；README 用明確的 `metrics-source` 註記選來源，
  防止同名 arm／metric 誤對到 RT-DETRv2 的 CSV。
- **環境**：Python 3.12.13、torch 2.13.0+cu130、transformers 5.14.1。
  RF 四組在本機 RTX 4090 完成；bootstrap 與目前文件凍結主要使用 CPU。
  SafeSynth 的延遲程序已退出、不占 GPU；2026-08-04 重啟後實測為 P8／225 MHz，
  2520 MHz 固定時脈已解除。
- **下一個動作（一句話、可直接動手）**：由 owner 提交並推送本次發布後狀態紀錄，
  確認文件-only CI 綠後封存專案。
- **卡住的事**：沒有。

- **⚠️ 已知限制（必須寫進 README，且已經寫了）**：
  1. **H4 未通過（AUC 0.9053，上限 0.60），而訓練結果與它的警告一致。**
     這是預先登記的閘門確實有預測力，不是事後找的藉口
  2. RF-DETR 已跑 **1,000 次影像層級 bootstrap**，但四組仍全是單一 seed；
     EVAL-10 禁止用區間重疊的零點幾個點宣稱勝出
  3. **EVAL-16 無法產生**：凍結 Test 的 744 張**全都**含 helmet 或 head，
     hard-negative 子集是空的，而候選區域 fallback 未實作
  4. 模型**校準很差**：223,200 個偵測的最高分只有 0.2495。排序good、絕對分數不可用
  5. 接受率天花板（K-13）與 hard negative 放置（K-11）維持原狀
  6. **歷史瘦身已完成**：136 張 DROP 圖已從所有 refs 的歷史移除，Git pack 為
     **4.62 MiB**；14 張 KEEP 圖由 v5 封存精確恢復，仍保留可重現的完整復原包
- **等使用者做的事**：只需親自提交並推送本次發布後狀態紀錄；所有公開發布動作已完成。
- **驗證本快照的指令**：
  ```
  uv run python -m scripts.audit_colab_results
  uv run python -m scripts.verify_readme
  uv run python -m scripts.check_forbidden_licences
  uv run ruff check .
  uv run pytest -q
  ```

### 2026-08-05 — v1.0.0 正式發布與最終驗收

- **GitHub Release**：annotated tag `v1.0.0`（tag object `bd3fcf1`）精確指向
  `abbf5a7`；tagger、source author 與 committer 都是 `kuotunyu`。Release 已公開，
  非 draft／prerelease，發布頁為
  `https://github.com/kuotunyu/SafeSynth/releases/tag/v1.0.0`。
- **最終 CI 與安全 metadata**：Actions run `30940634079` 的 locked install、ruff、
  1,774 passed／51 skipped、README、curated figure evidence 與 licence gates 全部通過，
  annotations 為 0。Node 20 action 已換成釘選 SHA 的 `setup-uv v8.1.0`／Node 24；
  vulnerability alerts、homepage 與六個 topics 已啟用。
- **唯一作者驗收**：GitHub Contributors API 只有 `kuotunyu`（380 contributions）；
  本機／遠端歷史沒有 `Co-Authored-By:` trailer，tag 也由 `kuotunyu` 建立。
- **Hugging Face 再驗**：dataset 與 model 均為 public；SHA 仍分別為
  `ed346b7061b6c7d4f113bddfd1953eed3121480c`、
  `f5621de143756695abc18cc7b3310da131b1bf2c`，沒有發布後漂移。

### 2026-08-05 — v1.0.0 最終 pre-tag 稽核

- **公開模型實跑**：直接從 `steven0226/safesynth-rtdetrv2-r18` 下載，以 CPU 載入
  `RTDetrV2ForObjectDetection` 與 `RTDetrImageProcessor`，完成 640×640 inference；
  logits `[1, 300, 3]`、boxes `[1, 300, 4]`，標籤精確為 helmet/head/person。
- **找到並防止版本漂移**：`pyproject.toml` 還停在 `0.1.0`，與準備建立的 `v1.0.0`
  release notes 不一致。新增跨檔一致性測試，先確認它以 `missing release notes for
  project version 0.1.0` 失敗，再把 project version 對齊為 `1.0.0` 後轉綠。
- **清理誤導性舊狀態**：`HANDOFF.md`、`instructions_for_me.md` 與四份 agentic
  implementation plans 都加上歷史封存說明；修復一個由 `\02`／`\r` 被誤解成控制字元
  而斷裂的資料路徑。M11 仍保留科學閘門的 `[~]`，但標明 failed-and-accepted、不是待辦。
- **稽核基線**：修正前 fresh full suite 為 `1773 passed, 51 skipped`，修正後為
  `1774 passed, 51 skipped`；repository links、curated figure evidence、README 數字、
  forbidden-licence scan、ruff、uv lock、tracked-file control-character scan 與
  `git diff --check` 全部通過。`pip-audit` 未找到已知漏洞；PyTorch／torchvision 的
  `+cu130` 自訂 wheel 不在 PyPI，屬工具無法稽核的明確例外。遠端 CI 仍須在 push 後確認。

### 2026-08-05 — M23 CI 與 M24 公開 owner upload 完成

- **GitHub**：`https://github.com/kuotunyu/SafeSynth` 已為 public；最新 source
  commit `b6133b2` 的 author/committer 都是 `kuotunyu`，Contributors API 只有
  `kuotunyu`（379 contributions）。GitHub Actions run `30927093391` 的 locked
  install、lint、pytest、README、figure evidence 與 license gates 全部通過。
- **Hugging Face**：dataset `steven0226/safesynth-hard-hat` 公開 commit
  `ed346b7061b6c7d4f113bddfd1953eed3121480c`；model
  `steven0226/safesynth-rtdetrv2-r18` 公開 commit
  `f5621de143756695abc18cc7b3310da131b1bf2c`。遠端 manifest 與本地完全一致，
  沒有舊 namespace、optimizer 或 Trainer state。
- **剩餘**：只差 owner 建立 GitHub `v1.0.0` annotated tag 與 Release；完成後再做
  最終公開頁面與乾淨工作樹驗收。

### 2026-08-04 — M24 本機公開發布包完成

- **Dataset 包**：filtered／unfiltered 各 3,500 張、重疊 848 張、唯一影像 6,152 張；
  只保留標註聯集，沒有把 14,000 張候選 pool 全部上傳。6,152 筆 provenance 與影像
  SHA-256 逐筆核對，發布包約 1.94 GB。
- **Model 包**：發布 validation-selected `real_only/checkpoint-1752`，含 20.2M 參數的
  `model.safetensors`、三類 config 與 640px RT-DETR processor；optimizer、scheduler、
  RNG、Trainer state 與 training args 全部排除。
- **文件**：dataset/model cards 完整揭露 CC0／Apache-2.0 授權鏈、SAM2 不是真實 GT、
  filtered/unfiltered 等量、四組負面結果、原始座標 AP_small、H4 AUC 0.9053 與絕對 AP
  禁止保證；GitHub/HF 互相連結，另備 v1.0.0 notes 與 owner-only runbook。
- **防錯**：初版 dataset card 曾把 COCO 的 1/2/3 類別 ID 誤寫成模型內部的 0/1/2；
  實際核對 JSON 後修正，並加入 category-table 回歸測試；發布入口現在也會先
  一次檢查所有來源檔，避免模型缺檔時留下半套資料集；模型驗證器要求
  `0=helmet, 1=head, 2=person` 精確對應。HF 專用測試目前 11/11 通過。
- **發佈邊界**：本段記錄當時仍未做遠端寫入；後續已由 `kuotunyu` 本人完成
  git／gh／hf 寫入，並在 2026-08-05 通過公開頁面唯讀驗收。

---

## 工作日誌

### 2026-08-04 — v5 歷史瘦身與正式驗收完成

- **正式改寫**：不可變 v5 runbook 綁定來源 commit
  `07d97fc77e5b9b8fc301210fcb81e634b36defc1`。owner 在外部 Windows PowerShell
  執行後，`main` 改寫為 `402c187fce7263872c11c91a31eae47f59a8cee8`，
  `codex/rfdetr-four-arm` 改寫為 `1a67a42c7f8d7396cec05e109d5ebd73f81112c2`，
  並正確停在 mandatory STOP。
- **controller checkpoint**：Codex 重開後產生的一個冗餘 tree ref，經名稱、型別與
  object ID 驗證為精確等於改寫後的 `HEAD^{tree}`，再以 expected-old-object 條件式
  刪除。重新檢查後工作區乾淨、單一 worktree、turn-diff namespace 為空、無 remote、
  所有 refs 中可達 `reports/figures/` 路徑為 0，strict `git fsck` 通過。
- **精確恢復**：從 v5 只恢復 14 個 KEEP，逐一比對路徑、大小與 SHA-256，並以
  `c74e2fc11d23ed9441148915ae07bd79a84061d1`（`docs: restore curated figure evidence`）
  單獨提交；136 個 DROP 在全部 374 個可達 commits 中均為 0。
- **完整驗收**：`1755 passed, 50 skipped`；Ruff、README 數值、Markdown 連結、
  forbidden-licence scan、`uv lock --check`、`git diff --check`、strict `git fsck`
  全部通過。Git pack 為 **4.62 MiB**（4,730 KiB），figure history 只有一個恢復
  commit，author／committer 只有
  `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`，無 co-author trailer。
- **v5 復原承諾**：KEEP 14／DROP 136；manifest SHA-256
  `a4ae4c773ffbce263ff5fd65d08df7502d1074bec17a2982016c26b8da684d0e`；bundle SHA-256
  `1ec20bcee16b8dcf0138691afeb4199fac30696b2bc801035bd695ab6827effa`，`git bundle verify`
  通過。GitHub repo 建立、push、model／dataset cards、Hugging Face 發佈與 fine-tuned
  latency 發佈仍是獨立後續工作，本次沒有擴大發佈範圍。

### 2026-08-04 — v5 owner gate 在建檔前完成文件綁定
- **已核准的書面規格**：v5 safety specification 是唯一的 owner gate；v1–v4 都是不可變、僅供復原的封存包，任何舊 runbook 都不得執行。v5 package 的建立留給 Task 4；本筆不宣稱 package 存在，也不預先記錄其 hash。
- **Stage 1 的受保護邊界**：先完成完整 preflight，驗證每個 Codex tree ref 後才可作 conditional deletion；history rewrite 後必須做 all-ref scan、strict `git fsck` 與 object-count report，最後以 mandatory STOP 結束。整個流程不涉及 GPU。
- **owner handoff**：owner 複製 v5 命令後完整關閉 Codex 與 editors，在外部 Windows PowerShell 執行；只在看到 STOP 後重開 Codex 並交回完整輸出。任何 restoration 都只能在另一個 read-only checkpoint 通過後開始。

### 2026-08-04 — RF-DETR 四組完成，延遲以負面驗證結果結案

- **訓練完成**：`real_only`、`standard_aug`、`unfiltered_syn`、`filtered_syn`
  都以 seed 1337 在 RTX 4090 跑滿 10,900 optimizer steps；四組 run record 的
  frozen Train digest 相同，synthetic counts 為 0／0／3500／3500。
- **預測與評測完成**：四組各自解析 best-validation checkpoint，在相同 frozen
  744-image Test 上產生預測並跑 1,000 次 image-level bootstrap。主結果與來源分別在
  `reports/rfdetr_detection_main_table.md`、`results/rfdetr_detection_metrics.csv`、
  `results/rfdetr_predictions_index.json`。
- **結果解讀**：`primary_map_small` 依序為 0.4841、0.4970、0.4959、0.5030；
  四組 95% CI 全部重疊。RF 的點估計方向與 RT 相反，但不能宣稱 synthetic win；
  synthetic arms 又只有一半 real-image exposures，且 H4 AUC 0.9053 仍失敗。
  跨架構結論是「效果敏感且不確定」，不是穩健提升。
- **M21 不觸發**：RT 的 Filtered 沒提升，RF 的 Filtered 也沒有區間分離；依原先
  前置判斷，不再補 6 個長訓練 seed。
- **延遲以負面結果凍結**：管理員鎖定 2520 MHz 的正式量測累計五次，clock spread
  每次都通過，但 contention p95 每次都失敗；最後一次在鎖定畫面下仍有 8/9 rows
  超標。三個模型／解析度一起抖動，且模型停止後同一張顯示 GPU 仍有 3–9% 桌面活動，
  證據指向 Windows 顯示排程／背景負載，而不是權重、降頻或單一模型。正式報告保留
  FAIL；不降低門檻、不挑最好的一輪，也不再重跑。最後一次失敗報告 SHA-256 為
  `0dd18c6262ee03106b8282581f7ab521a06643101600424d4960a84fa57aa8dd`。
- **README 防錯**：先加入兩個會失敗的測試，再讓 metric table 可用
  `<!--metrics-source: ...-->` 指定第二份 CSV。若 RF 表誤貼 RT 的可信數字，現在也會
  FAIL，而不是因 arm／metric 同名而蒙混過關。
- **[x] M20 狀態**：四組準確度複驗、授權檢查與延遲驗證都已執行；延遲品質閘門
  未通過，所以完成狀態代表「驗證完成並撤回速度主張」，不代表延遲數字合格。

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
