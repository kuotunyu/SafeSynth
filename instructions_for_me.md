# 換你做 — 02-safesynth-ppe

> 這份檔案只留**需要你親自判斷或執行外部動作**的事。
> 本機可逆的實作、測試與 commit 已授權自動完成；不會自行建立 remote、push 或發佈。

---

## 目前不需要你操作

FLUX.2 v2 的四案例 Colab 診斷已在 A100 40 GB 完成，12/12 輸出已下載、
驗證與分析。三個版本都只修改 edit mask 內的像素，但移除 reference 幾乎
沒有影響，降低 strength 也沒有一致改善，因此沒有選出新版本。原 v1 仍維持
人工 identity gate 失敗，M13 與 Phase 2 仍然關閉。

下一步是先設計並預註冊「輸入有效性＋anchor 定位」防護，再產生一批全新、
未看過的 64 圖 identity pilot。這是本機可完成的研究與實作工作；目前不需要
你重開 Colab、重跑 H4 或做其他外部操作。

結果與科學界線見 `reports/flux2_v2_diagnostic.md` 和
`docs/flux2_v2_colab_diagnostic.md`。

---

## 以下是已完成的舊交接紀錄（不用再操作）

M9、H6、Option A 選擇、模型下載及第一次 identity pilot 都已完成；
下列內容只保留作歷史脈絡，不是目前待辦。

### 1. 請先簽核 M9 的 hard-negative 候選

- 打開 `reports/figures/h6_hard_negative_candidates.png`
- 只數**青色框內其實是真正安全帽**的格數（共 64 格）
- 回覆：「真正安全帽 N 格，批准／不批准」

這是目前唯一不能由程式代替的資料決策。圖的 SHA256 已綁定為
`0e385d857067aa293c5e3d0dd43ad84b4141ff9bac5c8d4aefed187ee9c45739`；
若檔案被改過，簽核會 hard fail。超過 6 格（10%）時，規格要求改為程序生成為主。

### 2. 目前完成度與硬阻擋

- M0–M8、M10、M12 已完成；資料、split、7,255 個 cutout 與 300 張 H4/M12
  候選均已有可重現證據
- M9 只差上面的人工簽核
- M11 H4 **沒有通過**：paste-artifact classifier AUC 0.7964
  （門檻 0.60），因此程式正確地阻擋 M13 全量生成
- 睡眠期間額外預註冊並測過「同類別原位替換」修法，AUC 0.8312，
  比基準更差，已保存證據並排除，沒有拿測試結果偷調門檻
- Poisson blending 也依預註冊方法測過，AUC 0.8869 且會洗掉安全帽顏色，
  已排除；預設仍保留較好的 feathered alpha
- H4 的下一步已整理在 `docs/h4_next_decision.md`；需要你選擇是否把規格
  擴張到生成式 inpainting，或把目前結果凍結成誠實的負結果
- 遠端 GitHub repo 仍未建立，也沒有 remote 或 push
- 本機所有 commit author 都是
  `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`
- `Co-Authored-By:` trailer 為 0；沒有 remote、push 或其他 contributors
- 完整稽核結果在 `reports/phase1_preflight.md`；建立 remote 前可重跑
  `uv run python -m scripts.audit_phase1_handoff`
- repo 已啟用 `.githooks/commit-msg`：身份不符或訊息含
  `Co-Authored-By:` 時，commit 會在進入歷史前被拒絕

### 3. 建立 GitHub repo（等你決定後才做）

本機 repo 已完整存在。醒來後若要發佈，先在 GitHub 建一個**空 repo**
（不要勾 README、LICENSE 或 `.gitignore`），再把 URL 給我；我才會加 remote、
再次掃 author/trailer，並在你確認後 push。這能確保 Contributors 只會有
`kuotunyu`。

### 4. 可選的人眼複核（不阻塞後續）

我已逐張檢查過，結論已記在 ADR-007；你起床後若想複核，依序看：

1. `reports/figures/h1_helmet_contact_sheet.png` 與
   `reports/figures/h1_head_contact_sheet.png`：helmet 是戴帽的整顆頭，head 是裸頭
2. `reports/figures/h3_clip_largest_groups.png`：同一列應是連拍或保守合併的同構場景
3. `reports/figures/h5_placement_priors.png`：head/helmet 有中央水平帶，person 較發散
4. `reports/figures/class_distribution.png`：Train/Val/Test 與三類分布沒有離譜偏斜
5. `reports/figures/filter_pass_reject_grid.png`：上半 12 pass、下半 12 reject
6. `reports/figures/h4_ranked_patches.png`：H4 最易／最難辨識的貼上與真實 patch

若看到問題，回覆「檔名／第幾列第幾格／問題」即可。

### 5. 其他兩個 repo 的學校信箱仍要你親自處理

| repo | 受影響的 commit | 現況 |
|---|---|---|
| `1_DefectForge` | 前 3 筆 | repo-local 身分已修，新 commit 乾淨 |
| `3_FormosaNLU` | 2 筆 | repo-local 身分已修，新 commit 乾淨 |
| `2_SafeSynth` | 0 筆 | 一開始就設對了 |

發佈就會**公開且永久**。既有 commit 要靠改寫歷史（這是你親自執行的動作）：

```bash
git filter-repo --email-callback 'return b"61350295+kuotunyu@users.noreply.github.com" if b"gm.scu.edu.tw" in email else email' --force
```

跑完用 `git log --all --format='%ae' | sort -u` 確認只剩 noreply 那一個。

**根因是全域設定**，建議一併改掉，否則下一個新 repo 又會中招：

```bash
git config --global user.email "61350295+kuotunyu@users.noreply.github.com"
```

（三個 repo 的 `Co-Authored-By` trailer 都是 **0**，這部分乾淨。
你截圖裡出現「claude」的那個 repo 不在這三個之中，要單獨處理。）

---

## 你會被要求「打開圖來看」的地方

這個專案有幾處**只有人眼能判斷**，自動測試無法取代。到時會給你明確的檢查清單。

| 里程碑 | 要看什麼 | 判斷重點 |
|---|---|---|
| M3 | `helmet` 與 `head` 的 contact sheet | **框的是安全帽本體，還是整顆戴著安全帽的頭？** 這決定合規邏輯怎麼寫 |
| M5 | `reports/figures/class_distribution.png` | 分布是否合理、有沒有離譜的離群值 |
| M8 | `reports/figures/bank_<class>_grid.png`（洋紅色背景） | 已檢查：cutout 有沒有背景滲漏、光暈、或第二個物件入鏡 |
| **M9** | hard negative 的 8×8 contact sheet | **裡面有幾張其實是真正的安全帽？** 這是必須人工簽核的一關 |
| M12 | 12 通過 vs 12 被拒 並排圖 | 已檢查並修正 pHash 自身背景混淆；最終 196 pass / 104 reject |
| M14 | 各情境的 `preview_<scenario>.png` | 合成結果像不像真的、框有沒有貼對 |

### 回饋格式

指出**哪張圖、哪一格、什麼問題**就好，不用給解法：

```
grid preview_head_no_helmet 第 (3,4) 格：安全帽浮在頭上方沒接觸到
grid bank_helmet_grid_2 第 (1,1) 格：右下角有一塊背景沒去乾淨
```

每一格都會標上 `sample_id`，直接引用它更精確。

---

## M9 的人工簽核為什麼跑不掉

hard negative 的素材有一部分是從原圖「沒有標註的區域」挖出來的黃色圓形物。
但這個資料集**約 2/3 的真實物件是未標註的**——
也就是說「這裡沒有標註」**不等於**「這裡沒有安全帽」。

挖料很可能撈到真實但未標註的安全帽，把它們當成負樣本貼進訓練集，
等於**教偵測器去抑制安全帽**，方向完全反了。

程式端有三層防護（IoU 上限、「不像戴著的安全帽」測試、大小與長寬比範圍），
但唯一真正能抓到這種失敗的還是人眼。**五分鐘，看一張 8×8 的圖，數一數有幾個是安全帽。**
超過門檻就改成以程序生成為主。

---

## 正式 Phase 1／Phase 2 的原 Colab 規則

原定正式流程中，Colab 要到 Phase 2 的 RT-DETRv2 訓練才會用到。
目前的 FLUX.2 notebook 是因本機 GPU 暫時保留給其他專案而新增的
**方法診斷**，不屬於正式 H4 或 Phase 2 訓練。
