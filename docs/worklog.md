# Worklog — 02-safesynth-ppe 施工日誌

<!-- 收工時做兩件事：(1) 覆寫「現況快照」 (2) 在「工作日誌」最上面插入一筆。 -->
<!-- 快照是待證偽的假設，不是真相。真相以 git log 為準——開工時務必交叉驗證。 -->
<!-- 本檔超過約 18 KB 時把舊日誌摺疊或歸檔（publish-repo gate 2 的門檻是 ~20 KB）。 -->

## 現況快照

*每次收工覆寫，只留最新一份。*

- **更新時間**：2026-07-27 06:28 +08:00
- **最後驗證 commit**：`15ec811` docs(synthetic): record failed H4 Poisson spike
- **目前里程碑**：`M0`–`M8`、`M10`、`M12` 完成；`M9` 等使用者 H6
  簽核；`M11` H4 未通過；`M13`/`M14` 依硬閘門暫停。
- **⚠️ 未 commit 的改動**：只有本次 worklog／preflight 更新。
- **已凍結不得再動**：`splits/split_manifest.json`、`test_blocklist.json`、
  `source_checksums.json`、`MANIFEST.sha256`；manifest SHA256 為
  `ce9d76ee336cfba5e6071727442f7af413a8372f28cc9882093cb784587287a3`。
- **資料與素材落地**：Kaggle version 1、Pass-1 masks、7,255 個通過素材、
  300 張 final H4/M12 候選與診斷 run 都在 `D:\sdg-data\02-safesynth`；
  Test 洩漏為 0，cutout 重現抽查 100/100。
- **環境**：已安裝並驗證。`.venv/` ＋ `uv.lock` 已存在，
  Python 3.12.13、torch 2.13.0+cu130、torchvision 0.28.0+cu130、transformers 5.14.1。
  `docs/environment.md §5` 的驗證表**十列全過**，
  含 `torch.cuda.is_available() == True` 與 `NVIDIA GeForce RTX 4090`。
  SAM2 2-pass 推論已完成，權重存在既有 Hugging Face cache。
- **下一個動作（一句話、可直接動手）**：使用者先簽核 H6 64 格圖；
  H4 則需另行預註冊能同時解決空間重採樣與色彩訊號的新生成架構。
- **卡住的事**：H4 feathered-alpha AUC 0.7964 > 0.60；M13 硬阻擋。
- **等使用者做的事**：數 H6 青框內真正安全帽格數並批准／不批准；
  遠端 repo 仍未建立，醒來後再決定。
- **驗證本快照的指令**：
  ```
  uv run python -m scripts.audit_phase1_handoff
  uv run ruff check .
  uv run pytest -q
  ```

---

## 工作日誌

*append-only，新的插在最上面。*

### 2026-07-27 · M6–M12 · 素材、合成、filter 與 H4 硬閘門

- **M6–M8**
  - SAM2.1 Hiera Large 兩趟推論與 calibration 完成；H2 選定 contextual crop
    resize 512、edge-replicate 到 1024。
  - 素材庫 7,255 個：helmet 5,578、head 1,564、person 113；person 跨
    77 groups，依 ADR-003 使用 fallback；Test 洩漏 0。
  - 100 個 cutout 重新建置 SHA256 100/100 一致；三類洋紅底 grid 已目視。
- **M9**
  - 64 格 H6 候選與固定 SHA256
    `0e385d857067aa293c5e3d0dd43ad84b4141ff9bac5c8d4aefed187ee9c45739`
    已完成；不得代替 `kuotunyu` 建立簽核。
- **M10/M12**
  - 決定性 scenario compositor、bbox/visibility 重算、完整 provenance、
    COCO self-map 1.000、per-sample SHA256 RNG 已完成。
  - 同 seed 32/32 影像 SHA256 重現；最終 M12 ledger：
    300 = 196 pass + 104 reject，七項對帳全過，門檻敏感度警報 0。
  - 修正 FILT-11：只排除合成樣本自己的 Train 背景，仍比較其他所有真實圖。
- **M11 / H4**
  - 最終公平 gate：lossless PNG、同類別/同 fold/最近 log 尺寸真實對照、
    group-disjoint HOG+HSV C=1；2,028 patches，AUC 0.7964
    （CI 0.7481–0.8392），高於 0.60，M13 維持關閉。
  - controlled alpha／multiband 變體都沒有可靠改善；feature-family 診斷：
    HOG-only 0.7792、HSV-only 0.6816。
  - 預註冊 context matching 因 frozen fold 缺真實對照而按規則停止；
    同類別原位替換 AUC 0.8312；Poisson AUC 0.8869 且明顯洗掉物件顏色。
    exact-source 成對控制 AUC 0.9049，排除素材選擇偏差是主要假象。
    各路線都如實保存，沒有調弱分類器或放寬門檻。
- **品質與交接**
  - `uv run ruff check .` → PASS；`uv run pytest -q` → 92 passed
    （僅 pycocotools/Numpy 已知 deprecation warning）。
  - prepublication audit：所有 local refs 的 author/committer 只有
    `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`；
    `Co-Authored-By:` 0、remote 0。
  - 主要 commits：`10718ba`、`916c6bf`、`e276d3e`、`dce0b85`、
    `49a51fe`、`6d0d13e`、`15ec811`。
- **外部動作**：沒有建立 GitHub repo、remote、push、發佈或寫入兄弟專案。

### 2026-07-27 · M3–M5 · 資料語意、近似分群與 split 凍結

- **對應規格**：DATA-13~21、Spike H1/H3/H5
- **H1 實測與目視**
  - 40 `helmet`＋40 `head` contact sheets 與三類長寬比圖已打開檢視。
  - `helmet×head` 的 9,603 個同圖組合中，IoU>0.1 只有 95（0.99%）。
  - helmet/head 長寬比中位數 0.875/0.830，圖中 helmet 框通常含整顆戴帽的頭，
    head 則是裸頭；ADR-007 決定 compliance 走 `class_direct`。
- **H3 實測**
  - pHash Hamming ≤4/6/8/10 分別得到 4,907/4,900/4,890/4,875 群，
    最大群皆為 4；五個 seed 都能模擬出 3,500/750/750。
  - 因 pHash 群數 >2,000，依協定啟用 OpenCLIP `ViT-B-32` /
    `laion2b_s34b_b79k`；12 組 cosine×pHash guard 候選完成。
  - 選定 base Hamming≤10，或 cosine≥0.85 且 Hamming≤20：
    4,808 群、最大群 8。最大 20 群 grid 已打開看過，沒有 component collapse。
- **H5 實測與目視**
  - head/helmet 中心熱圖形成中央水平帶；person normalized entropy 0.948 且樣本稀疏。
  - head/helmet 保留取樣式先驗；person/crowded 改以錨定放置為主（ADR-007）。
- **M4/M5 凍結結果**
  - Train/Val/Test = 3,500/756/744；同群同 split hard assertion 通過。
  - annotations = 17,815/3,870/3,817；person = 525/113/113。
  - `split_manifest.json` 完整重建兩次，SHA256 都是
    `ce9d76ee336cfba5e6071727442f7af413a8372f28cc9882093cb784587287a3`。
  - 744 張 Test 依 blocklist 逐檔重雜湊：PASS。
  - `class_distribution.png` 已打開檢查，比例與 class imbalance 呈現合理。
- **測試**：`uv run ruff check src scripts tests` → PASS；
  `uv run pytest -q` → `21 passed`；
  `uv run python scripts/freeze_split.py --verify` → group/split/Test 三項 PASS。
- **commit**：`d29405d` feat(data): freeze guarded group split
- **外部動作**：僅下載約 605 MB 的 OpenCLIP 權重到既有 HF cache；
  沒有 remote、push、repo 建立或上傳。

### 2026-07-27 · M2 · 資料下載與 VOC→COCO 落地

- **對應規格**：DATA-01~12、ENV-10
- **下載前報備**：Kaggle 官方 API 的 `totalBytes` 為 `1,320,154,239`
  （約 1.23 GiB／1.32 GB），低於 2 GB 無人值守停等門檻；D 槽當時剩餘約 1.69 TiB。
- **實作**
  - `scripts/prepare_data.py`：讀官方 metadata、釘 Kaggle version、以
    `kagglehub.dataset_download()` 下載、保留來源 archive、轉換並驗證。
  - `src/data/paths.py`：只從 `configs/paths.yaml` 解析路徑，保留 YAML 註解地回填版本。
  - `src/data/voc_to_coco.py`：stem 配對、座標 offset 執行期偵測、VOC flags 保留但
    `iscrowd=0`、決定性 ID、canonical JSON、來源雜湊、COCO 自評。
  - `kagglehub 1.0.2` 會在解壓後刪除 archive；下載器在其解壓 callback 前複製同一份
    原始位元組再雜湊，避免第二次下載，也沒有修改第三方套件檔案。
- **真實驗證輸出**
  - `uv run pytest -q` → `11 passed`
  - `uv run ruff check ...` → `All checks passed!`
  - `uv run python scripts/prepare_data.py --verify`：
    `global min coordinate = 0 -> offset 0`；
    `images = 5,000`；`annotations = 25,502`；
    instances `18,966 / 5,785 / 751`；
    class images `4,581 / 920 / 158`；
    unknown labels `0`；`iscrowd != 0 = 0`；self-eval mAP `1.000`。
  - 來源 archive 大小 `1,314,241,385` bytes，SHA256
    `aa5c80a85f9f4bd3b27e44256f8e36f9a32c53ee423132fa6cd5ea603781be62`。
- **實測面積（解決 DATA §1.2 衝突）**：helmet `1.27%`、head `0.63%`、
  person `7.05%`；與讀數 A 一致，讀數 B 不是單一 bbox 的平均面積。
- **commit**：`1af595c` feat(data): download and convert frozen hard-hat dataset
- **外部動作**：沒有 remote、push、repo 建立或上傳；commit author 僅為 `kuotunyu`。

### 2026-07-27 · M0（續）· 補上 Phase 2 規格

- **對應規格**：EVAL-01~18、DEMO-01~05、PUB-01~11、EXP-01~03
- **背景**：使用者決定把 Phase 2 的規格也一次寫完，目標是「按圖施工」的完整度。
  原先 M0 的決定是暫緩 Phase 2 規格以免被 Phase 1 實測推翻——
  折衷作法是**照寫，但把受 Phase 1 影響的部分標成 config 旗標的兩條分支**
  （例如合規邏輯的 `class_direct` vs `geometric_pairing` 由 Spike H1 決定）
- **做了什麼**
  - `docs/evaluation_spec.md`（EVAL-01~18）：合規邏輯兩模式、指標、錯誤分析
  - `docs/release_spec.md`（DEMO-01~05、PUB-01~11）：Gradio demo、README、HF 發佈
  - `configs/evaluation.yaml`：合規操作點、COCO 面積桶、效能量測、錯誤分析參數
  - `PLAN_PHASE2.md`：M15–M24。**另開一份檔**是因為 `PLAN.md` 已 16 KB，
    合併會超過 `publish-repo` 第 2 關的 ~20 KB 門檻
  - `CLAUDE.md` 加入【去哪找】路由表；把「五組」修正為「四組」並說明原因
  - `docs/experiment_protocol.md` 從存根擴充為完整協定層

- **從原始 Phase 2 prompt 修正的三處**
  1. **第 4 項的 YOLO11s 移除**：Ultralytics 是 AGPL-3.0，會傳染到 `import` 它的程式碼，
     與 MIT repo 牴觸——ADR-001 早已因同一理由否決它，只在 README 註明授權擋不住這件事。
     使用者選擇改用寬鬆授權的偵測器，選型待 ADR-005
  2. **第 5 項的「WSL2」移除**：本專案原生 Windows（ADR-002）
  3. **「五組」修正為「四組」**：本專案的 Real-only 本來就吃全部真實 Train，
     沒有更高的「Full-real 上限」可言。README 要主動說明，否則讀者會以為漏做一組

- **本階段新寫進規格的兩個陷阱**
  - **`AP_small` 必須在原始 416×416 座標下計算**（EVAL-07）。影像在 416 標註、
    在 640 訓練，若在 640 算面積，每個物件膨脹約 2.37 倍，
    大量原本 small 的物件會被歸到 medium——主敘事指標會**安靜地**變成在測量另一件事。
    `head` 平均約 34×34 ≈ 1,156 px²，正好卡在 32²=1024 的邊界附近，特別敏感
  - **合規操作點只能在 Validation 上選**（EVAL-04），在 Test 上選等於用測試集調參

- **使用者決定**：速度對照改用寬鬆授權模型；Colab 是 **500 units/月**方案（非 100）；
  demo 做上傳圖片/影片的 Gradio（不做 webcam）；無人值守的邊界由 Claude 自行判斷
- **無人值守邊界（Claude 自訂並記錄）**：會停下來等使用者的是——花錢、>2GB 下載、
  git push／發佈、以及結果與規格假設牴觸而需要決定範圍的時刻。
  需要看圖判斷的事自己看（H1、H5、預覽圖品質），記進 worklog 隔天複核。
  M9 的 hard negative 簽核先自己數一遍，只有落在灰色地帶才叫人
- **查證結果（RT-DETRv2 訓練 API，已寫進 `training_spec.md` §1.1–1.3 與 ADR-005/006）**
  - **`transformers` 已進入 v5**（5.14.1）。v5 把 image processor 的快慢版合併並改名：
    `RTDetrImageProcessor` 現在**就是**快版，`RTDetrImageProcessorFast` 不存在了。
    2025 年的教學抄下來會 ImportError 或行為悄悄不同 → **下限提高到 5.14.1（ADR-006）**
  - **`RTDetrV2ImageProcessor` 根本不存在**（`rt_detr_v2/` 目錄裡沒有任何 image processing 檔案）。
    HF 官方文件的 autodoc 範例區塊連類別名帶 checkpoint id 都是錯的 →
    **一律用 `AutoImageProcessor` / `AutoModelForObjectDetection`**
  - **RT-DETR 不做 ImageNet 正規化**（`do_normalize: false`，只除以 255）。
    自己加正規化會安靜地毀掉模型
  - checkpoint 的預設 size 已經是 640×640，`do_convert_annotations: true`
    會自動把 COCO xywh 轉成正規化 cxcywh——兩者都不要覆寫
  - 六個會安靜出錯的訓練設定：`eval_do_concat_batches=False`（強制）、
    `ignore_mismatched_sizes=True`、`remove_unused_columns=False`、
    `max_grad_norm=0.1`（不是預設的 1.0）、
    `len(val) % eval_bs != 1`（餘數為 1 會直接崩潰）、category id 要重新映射成 0..K-1
  - **backbone 不要凍結**，用 0.1× 的 backbone LR 參數組（上游作者的做法）
  - `PekingU/rtdetr_v2_r18vd` 確認存在：Apache-2.0、20,209,716 參數、約 77 MiB
- **Colab 預算**：使用者的 500 CU 方案是 **Pro+**，附帶最長 24 小時背景執行。
  8 次訓練用 **L4** 跑 50 epochs 約 **113 CU**，額度綽綽有餘。
  L4 對整個 sweep 與 T4 幾乎等價成本但快約 2.4 倍，且每次訓練都塞得進單一 session
  （T4 跑合成組 100 epochs 會超過約 12 小時上限）。**這些是外推估計，誤差約 ±40%**
- **ADR-005 速度對照組**：改用 `Roboflow/rf-detr-nano`（Apache-2.0，`transformers` 原生支援），
  邊際成本幾乎是零。⚠️ 只能用 nano/small/medium/base/large，**XL 與 2XL 是 PML-1.0**。
  否決 YOLOX（停更 14 個月）與 RTMDet（mmdetection 停更 23 個月，且 mmcv 在原生 Windows 要編 CUDA extension）
- **驗證結果**：`CLAUDE.md` 127 行（上限 200）；`PLAN.md` 16 KB／`PLAN_PHASE2.md` 12 KB
  各自低於 20 KB 門檻；個資／金鑰／禁用詞／AGPL 依賴掃描全 PASS；
  連結、ADR anchor、config 引用、四個版本號三處一致——**全數通過**
- **commit**：（等使用者執行）
- **刻意不做**：不在規格裡寫死今天查到的 API 細節就了事——
  `TRAIN-01` 仍要求 M15 開工時重新查證一次，因為版本還會動

### 2026-07-27 · M0 · 文件與骨架凍結

- **對應規格**：全部規格文件本身
- **做了什麼**
  - 建立目錄骨架（`src/{data,synthetic,filtering,inference,evaluation}`、`scripts`、
    `configs`、`docs`、`splits`、`reports/figures`、`results`、`assets`、`model_cards`、
    `notebooks`、`tests`）與 `.gitkeep`
  - 根檔：`LICENSE`（MIT ＋ 第三方素材授權說明）、`.gitignore`、`README.md`、
    `pyproject.toml`（含 cu130 index 區塊）
  - `configs/`：`paths.yaml`（路徑唯一來源）、`compose.yaml` 與 `filtering.yaml`
    （**數值唯一來源**，每個門檻帶 `source: fixed|guess|calibrated` 標記）
  - `CLAUDE.md` 五章節，102 行
  - `docs/` 九份：`decisions.md`（ADR-001~004）、`environment.md`（ENV-01~10）、
    `data_protocol.md`（DATA-01~21）、`synthesis_spec.md`（CUT-01~12／COMP-01~29）、
    `filtering_spec.md`（FILT-01~14／PREV-01~05）、`experiment_protocol.md`（EXP-01~03 存根）、
    `troubleshooting.md`（K-01~09 已知風險）、`skills_roadmap.md`、`worklog.md`
  - `PLAN.md`：M0–M14，每項附對應規格 ID 與可執行的驗證方法
  - 兩支 skill：`.claude/skills/safesynth/`（開工／收尾儀式）、
    `.claude/skills/safesynth-env/`（Windows 環境 runbook）

- **本階段的查證結果（都已寫進對應文件）**
  - SAM2 走 HF `transformers` 的 `Sam2Model`，純 pip、零 CUDA 編譯，
    解除了「原生 Windows 做不了 SAM2」的疑慮。硬性下限 `transformers>=4.57.1`
    （4.56.x 的 box prompt 有數值 bug，會**安靜**降低 mask 品質）
  - `post_process_masks` 的 `max_hole_area`／`max_sprinkle_area` 是 **no-op**，
    清理必須自建
  - PyPI 的 Windows torch wheel 是 **CPU-only**；已實地確認
    `torch-2.13.0+cu130-cp312-cp312-win_amd64.whl` 與
    `torchvision-0.28.0+cu130-cp312-cp312-win_amd64.whl` 存在；
    驅動 591.86 ≥ 580.88 過關
  - **cu128 index 最高只到 torch 2.11.0**，沒有 2.12／2.13
  - 資料集三類實例數 18,966／5,785／751，其中 `person` 只在 158 張圖裡；
    SHEL5K 重標同樣 5,000 張圖得 75,570 個標註 vs 原版 25,502，
    約 2/3 真實物件未標註 → **所有主張必須是相對的**
  - 資料集平均物件面積有**兩個互相矛盾的讀數**（相差約 4 倍），已列為 M2 的實測項
  - **`helmet` 框的語意沒有任何來源定義過**（框帽殼還是框整顆戴帽的頭），
    已列為 Spike H1，是 M3 的第一項

- **驗證結果**：`(Get-Content CLAUDE.md).Count` → `102`（上限 200）。
  其餘一致性檢查在 M0 收尾時執行
- **決策**：ADR-001（SAM2 路徑）、ADR-002（Windows 原生 ＋ cu130）、
  ADR-003（3 類保留但合規不依賴 person）、ADR-004（hard negative 挖料為主且不給標註）
- **踩到的坑**：無（`troubleshooting.md` 目前只有預先寫入的已知風險 K-01~09）
- **commit**：（等使用者執行）
- **刻意不做**：不寫任何 Python；不下載資料集；不建 venv；
  不寫 Phase 2 的實質規格（只留協定層存根，避免被 Phase 1 實測推翻）；
  不動兄弟專案 `1_DefectForge` 與 `3_FormosaNLU`（有平行 session 正在建置）

- **跨專案發現（需回報使用者）**：`1_DefectForge/pyproject.toml` 同時寫著
  「torch 從 cu128 index 裝」與「已查證 torch 2.13.0」。依實測該組合不存在
  （cu128 最高 2.11.0），該專案的 M1 需一併改用 cu130
