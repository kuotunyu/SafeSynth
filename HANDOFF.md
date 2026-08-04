# HANDOFF — 02-safesynth-ppe（歷史封存）

> **封存於 2026-08-05**：這是公開發布前的交接快照，下方「剩下的事」與狀態表
> 不再是現行待辦。GitHub、CI、repo 瘦身、RF-DETR 複驗與 Hugging Face upload
> 均已完成；權威狀態請看 [PLAN_PHASE2.md](PLAN_PHASE2.md) 與
> [docs/worklog.md](docs/worklog.md)。目前只差 owner 建立 GitHub `v1.0.0` tag／Release。

> **給新 session 的第一句話**：先讀這份，再讀 [CLAUDE.md](CLAUDE.md)。
> 這份檔案是「還剩什麼、怎麼做、哪裡有地雷」；CLAUDE.md 是「永遠適用的規則」。
> 兩者衝突時**以 CLAUDE.md 為準**，並回報衝突。
>
> 建立於 2026-08-02，對應 commit `d39de08`。
> **開工第一件事是證偽這份檔案**——見下方〈開工檢查〉。

---

## 0. 三十秒版本

四組對照訓練跑完、評測完、有信賴區間、README 寫完、demo 與 GIF 都有了。
**主結論：合成資料在這個資料集上沒有提升，而且區間支持這句話。**

剩下的事**沒有一件會改變結論**，都是收尾：

| 項目 | 誰做 | 需要 GPU | 狀態 |
|---|---|:---:|---|
| **A. RF-DETR 速度實測 ＋ 訓練**（M20 ②） | Claude | ✅ 重 | 程式備妥，卡在 GPU 被佔用 |
| **B. repo 從 631 MiB 瘦身** | **使用者** | ❌ | 清單已備妥待過目 |
| **C. 建 GitHub repo ＋ push** | **使用者** | ❌ | Claude 備料，使用者執行 |
| **D. Hugging Face 上傳** | **使用者** | ❌ | 卡片未寫 |
| M21 補 3 seeds | 使用者裁決 | ✅ 重 | 已判定不做，可推翻 |
| M11 / M23 的 `[~]` | — | ❌ | **這兩個永遠不會變 `[x]`**，理由見 §5 |

**建議順序：B → A（若 GPU 空出）→ D 備料 → C。**
B 一定要在 C 之前，理由見 §3.B。

---

## 1. 開工檢查（照 `/safesynth` skill 的流程，但這裡先列具體指令）

```bash
git log --oneline -5
git status --porcelain
uv run pytest -q
uv run python scripts/verify_readme.py
```

**應該看到**：

```
（git status 零行輸出）
1518 passed, 44 skipped
PASS: every README number has a source and every disclosure is present
```

`git log` 的最新 commit **會比這份檔案新**——這份檔案本身就是一筆 commit，
之後還會有收尾的 commit。所以**不要拿 sha 去對**，要對的是那三件事：
工作樹乾淨、測試全過、README 驗證通過。
（這份檔案寫成時的基準是 `d39de08`，僅供參考。）

測試數只會**往上**。變少了就是有東西被刪掉或壞掉，要查。

**這三項對不上就先停下來報告，不要直接開工。**
不一致本身是最重要的資訊——通常代表上一輪收工沒收乾淨。

環境：Python 3.12.13、torch 2.13.0+cu130、transformers 5.14.1。
壞掉的話呼叫 `/safesynth-env`。

---

## 2. 這個專案的結論（別重算，也別在文件裡改寫這些數字）

凍結 Test 744 張，四組各取自己最佳 checkpoint。
兩條獨立實作算出的主表一致到 8.8e-07。

| arm | primary AP_small | 95% CI | primary mAP |
|---|---:|---|---:|
| **real_only** | **0.4511** | 0.4307–0.4753 | **0.5341** |
| standard_aug | 0.4236 | 0.3993–0.4530 | 0.4958 |
| unfiltered_syn | 0.3759 | 0.3474–0.4064 | 0.4597 |
| filtered_syn | 0.3664 | 0.3426–0.3956 | 0.4858 |

**信賴區間收回了兩個主張**（EVAL-09，1,000 次重抽，單位是 Test 影像）：

| 比較 | 判定 |
|---|---|
| `real_only` 勝過兩個合成組 | **區間不重疊，成立** |
| `real_only` 勝過 `standard_aug` | **區間重疊，不成立**——0.0275 在雜訊內 |
| `filtered_syn` 在偵測指標上勝過 `unfiltered_syn` | **三個指標區間全重疊，不成立** |

**過濾的價值在合規操作點，不在 AP**：各組各選各的操作點後，
`unfiltered_syn` 在任何會偵測到東西的門檻上都達不到 0.80 精確度，`filtered_syn` 可以。

**還有一半故事**：改用「真實影像曝光次數」當橫軸，`filtered_syn` 在真實資料
只看過 1–4 輪時領先最多 +0.090 mAP。合成資料在標註稀少時有效，
只是這個資料集有 5,000 張標註，正好是它最沒用武之地的區間。
兩半都畫在 `reports/figures/headline.png`。

---

## 3. 剩下的事，逐項

### A. RF-DETR 速度對照組（M20 ②）— Claude 做，需要 GPU

**目前唯一的阻擋：使用者機器上有 `llama-server` 佔著約 18–23 GB VRAM、95% 使用率。**
不是這個專案開的。在被佔滿的 GPU 上量出來的數字沒有意義（見 §4 的 K-21b）。

要使用者先關掉：

```bash
Stop-Process -Name llama-server
```

**A-1. 先量實測速度，不要推算**（約 20 分鐘）：

```bash
uv run python -m scripts.probe_train_speed --arm real_only --short 40 --long 140
```

它跑兩趟不同步數、**取斜率**，所以固定啟動成本會抵銷掉而不是被攤進每步成本。
輸出寫到 `reports/train_speed.md` 與 `.json`（**現在不存在，跑完才會有**）。

**拿到數字之後停下來報告，不要自己開始訓練。**
換算後若超過 1 小時，依 CLAUDE.md 必須明講「這是過夜的工作」。

**A-2. 訓練**（時數未知，等 A-1）。

⚠️ **本機沒有訓練 CLI，這件事要先做。** 四組 RT-DETRv2 是在
`notebooks/01_train_rtdetrv2.ipynb` 裡跑的，`src/training/run.py` 只提供
`run_arm()` 函式、沒有 `__main__` 進入點。所以 A-2 的第一步是**寫一支還不存在的**
`scripts/train_arms.py` 把 `run_arm()` 包起來（`scripts/probe_train_speed.py`
已經在呼叫它，可以直接抄它的用法）。

範圍已縮成**兩組**（`real_only` 與 `filtered_syn`），寫在 config 的 `arms:` 裡。
原規劃四組；縮成兩組是因為這兩組就是決定性對比——
能回答「換一個架構，合成沒提升的結論是否複現」，成本只有一半。

**config 的關鍵設計**：`configs/training_rfdetr.yaml` **只寫差異**，
靠 `extends: configs/training.yaml` 由 `src/training/config.py` 深層合併。
**不要把它展開成完整副本**，那樣兩個檔案會各自漂移。
三個真正不同的值（都已對兩個 checkpoint 實測驗證，不是猜的）：

| | training.yaml | training_rfdetr.yaml |
|---|---|---|
| `do_normalize` | `false` | **`true`** |
| `image_size` | 640 | 384 |
| `do_pad` | `false` | `true` |

`do_normalize` 是唯一一個**安靜繼承會毀掉模型而不是讓它崩潰**的設定。

---

### B. repo 瘦身 — **使用者執行**，Claude 只備料

**這是最急的一件，而且時機不能拖。**

```
clone 大小（打包後）      631.27 MiB
  其中 reports/figures/   629.1 MB  (94%)
  其他全部               40.9 MB   (6%)
```

**這個 repo 的程式與文件本體只有 41 MB**，剩下全是 Phase 1 已放棄路線的診斷圖
（`supervised_labeler_v12`–`v23` 的 audit 每版約 18 MB）。
149 張圖裡**只有 31 張被任何文件引用**。

**為什麼時機關鍵**：從 HEAD 刪檔案不會讓 clone 變小，歷史裡的 blob 還在。
只有 `git filter-repo` 改寫歷史才有用。而**這個 repo 還沒 push 過**
（`git remote -v` 是空的），現在做零副作用；push 之後就是 force-push
覆蓋已公開歷史，難度差一個等級。

逐檔清單、指令、事後驗證都在 **[reports/repo_slimming_plan.md](reports/repo_slimming_plan.md)**。
那份檔案是腳本產的，可以重跑：

```bash
uv run python -m scripts.plan_repo_slimming
```

**依 CLAUDE.md，`filter-repo` 由使用者親自執行。** Claude 不要代跑。

---

### C. 建 GitHub repo ＋ push — **使用者執行**

依 CLAUDE.md【工作方式】的 git 權限界線：
**`push`、建立 remote repo、Hugging Face 上傳、`filter-repo` 一律由使用者親自執行。**
Claude 準備內容與逐行指令，不按 Enter。

發佈前**呼叫個人 skill `publish-repo` 做完整驗收**——
不要在本 repo 重寫發佈流程。

已知會通過的：commit 歷史零個 `Co-Authored-By:`、author 全部是 repo 擁有者、
無 shields.io 假 badge、`uv.lock` 已追蹤、洩漏掃描 PASS。

---

### D. Hugging Face 上傳（M24）— Claude 備料，使用者上傳

要準備的：
- **dataset card**：來源授權鏈（CC0 1.0）、生成方法、filtered/unfiltered 差別
  與**等量**這件事、限制，以及**「SAM2 自動 mask 只用於合成素材、
  未用於任何 ground truth」**的明確聲明
- **model card**：訓練資料組成、四組結果表、
  **`AP_small` 在原始 416 座標計算**的說明、基礎模型授權、
  以及「不要拿絕對 AP 當品質保證」的警語
- HF card ↔ GitHub README 互相連結

帳號：GitHub `kuotunyu`／Hugging Face `steven0226`。

---

## 4. 地雷區（踩過的，別再踩）

完整清單在 [docs/troubleshooting.md](docs/troubleshooting.md)（24 條）。
**收尾階段最會再遇到的六條：**

| 編號 | 一句話 |
|---|---|
| **K-19** | 綠燈不算數。**變異測試才是驗收標準**——96% 分支覆蓋率下四個一 token 的 bug 全部存活 |
| **K-21** | `git add -A` 會夾到背景變異測試注入的變異。**一律按檔名逐一 stage** |
| **K-21b** | 被 SIGKILL 的變異 harness 會把變異**留在工作樹**，還會留下**看起來很像實測的假報告**。commit 前要看 `git diff` 逐行 ＋ `git status` 的未追蹤檔案 |
| **K-22** | 桌機 GPU 的延遲量測會隨 SM clock 差 2.3 倍。要可發佈的數字必須 `nvidia-smi -lgc 2520,2520`（管理員），量完 `-rgc` |
| **K-23** | 主表只有 `--device cpu` 重現得了。EVAL-09 重跑 424 個點估計零不一致，前提就是這個 |
| **K-10** | 這台機器是 cp950，**所有檔案 I/O 一律明寫 `encoding="utf-8"`** |

**還有三條不在 K 編號裡但同樣重要：**

- **PowerShell 5.1 沒有 `&&` 與 `||`**，用 `A; if ($?) { B }`。沒有 `head`/`tail`/`which`/`touch`
- **不得硬編絕對路徑**，一律讀 [configs/paths.yaml](configs/paths.yaml)
- **數值只寫在 `configs/*.yaml`**，`docs/` 只寫判定式與 config key

---

## 5. 兩個永遠不會變 `[x]` 的里程碑（別去「修好」它們）

### M11 — H4 閘門沒有通過，這是照實記錄

貼上痕跡的可偵測度 **AUC 0.9053**（95% CI 0.9013–0.9090，106,144 個 patch），
預先登記的上限是 **0.60**。試過 9 條合成路線與 18 輪 labeler 迭代都翻不過來。

處置**不是**放寬門檻或換弱一點的分類器來「通過」，而是把它當成本專案的
一項發現照實發表，並把生成量壓在 1×（不做 2×）。見 [ADR-011](docs/decisions.md)。

**所有成果表都必須與 H4 AUC 並列呈現**，README 正文已經寫了。
`[~]` 記錄的是「跑過了、失敗了、後果已處理」，不是「還沒做」。

### M23 — 「CI 為綠」在第一次 push 之前無法驗證

四條本機可驗的條件全綠（`verify_readme` PASS、無 shields.io、
`uv lock --check` exit 0、`uv.lock` 已追蹤）。
但驗收條件裡的「`.github/workflows/ci.yml` 為綠」需要 workflow
真的在 GitHub 上跑過一次，而 `git remote -v` 是空的。

**第一次 push 之後這條才驗得到，屆時才可以勾 `[x]` 並補 `驗證於 <sha>`。**
在那之前勾起來就是宣稱一件沒發生的事。

---

## 6. 這個專案誠實的樣子（介紹它的時候要講）

**三個限制，都可以寫，而且都比假裝沒有誠實：**

1. **H4 預先註冊的閘門沒過**（見 §5）
2. **copy-paste 在 3,500 張背景上會飽和**。接受率不是常數——去重要拿新樣本
   跟所有已接受的比，所以 pool 越大接受率越低：2,000 張候選時 58.4%、
   10,000 張時 33.8%、14,000 張時 29.8%。要湊到 1×（3,500 張）需要 14,000 張候選。
   **這是方法的天花板，不是門檻太緊**
3. **hard negative 的放置真實度只修好一半**（[K-11](docs/troubleshooting.md)）。
   代價是 EVAL-16 的每圖誤報數這個次要指標在 Test 上**退化**——
   四組分別是 1／1／0／1，spread 只有 1，**不能當排名讀**，報告裡明說了

另外**四組不是五組**：通用鐵律的第五組「Full-real 上限」不適用本專案，
因為 Real-only 本來就吃全部真實 Train。README 主動說明了這點，
否則讀者會以為漏做一組。

---

## 7. 檔案地圖

| 要做什麼 | 讀哪份 |
|---|---|
| 永遠適用的規則 | [CLAUDE.md](CLAUDE.md) |
| 里程碑與驗證方法 | [PLAN.md](PLAN.md)（Phase 1）／[PLAN_PHASE2.md](PLAN_PHASE2.md) |
| 施工日誌 ＋ 現況快照 | [docs/worklog.md](docs/worklog.md) |
| 「當初為什麼這樣選」 | [docs/decisions.md](docs/decisions.md)（14 則 ADR，只追加不改寫） |
| 撞到看過的錯誤 | [docs/troubleshooting.md](docs/troubleshooting.md)（24 條） |
| 需要使用者動手的事 | [instructions_for_me.md](instructions_for_me.md) |
| repo 瘦身清單 | [reports/repo_slimming_plan.md](reports/repo_slimming_plan.md) |
| 評測規格 | [docs/evaluation_spec.md](docs/evaluation_spec.md) |
| 發佈規格 | [docs/release_spec.md](docs/release_spec.md) |

大檔在 `D:\sdg-data\02-safesynth\`（權重、合成影像、cutout bank）。
專案資料夾只留程式碼、設定、文件、`splits/`、`reports/` 小圖。

---

## 8. 收工儀式（每個里程碑做完都要跑一次）

1. **當場執行驗證指令並貼出真實輸出**，不接受「應該會過」
2. 產圖類的驗證**要真的把圖打開看**（這條抓到過三次真實缺陷）
3. 勾 `PLAN.md` 並補 `**驗證於**：<sha> @ 日期`。
   **順序：先 commit 實質內容拿到 sha，再開一筆小 commit 填 sha。
   絕對不要用 `git commit --amend` 回填**——amend 會產生新 sha，
   剛填的值當場失效
4. 覆寫 `docs/worklog.md` 頂端快照 ＋ 在日誌最上面插一筆
   （超過 ~18 KB 就把最舊的搬到 `docs/worklog_archive.md`）
5. commit 訊息用英文、conventional 格式、**不要帶 `Co-Authored-By:`**
6. 給使用者「換你做」清單

最後全套驗證：

```bash
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/verify_readme.py
uv run python -m scripts.check_forbidden_licences
```
