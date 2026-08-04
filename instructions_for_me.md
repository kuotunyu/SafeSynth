# 換你做 — 02-safesynth-ppe（歷史封存）

> **封存於 2026-08-05**：下方保留的是實驗期間的人工作業紀錄，不是現在的待辦清單。
> GitHub、CI、兩個 Hugging Face repo 與
> [`v1.0.0` Release](https://github.com/kuotunyu/SafeSynth/releases/tag/v1.0.0)
> 都已公開驗收；目前沒有尚待執行的 owner action。

> 這份檔案只留**需要你親自判斷或執行外部動作**的事。
> 本機可逆的實作、測試與 commit 已授權自動完成；不會自行建立 remote、push 或發佈。
> 最後更新：2026-08-02（M20 ① 完成；GPU 被 llama-server 佔住，M20 ② 卡在那裡）

---

## 🎯 一分鐘看懂現在的狀況

**四組訓練跑完了，數字算完了，結論是：合成資料在這個資料集上沒有提升。**

| arm | primary AP_small | primary mAP |
|---|---:|---:|
| **real_only** | **0.4511** | **0.5341** |
| standard_aug | 0.4236 | 0.4958 |
| unfiltered_syn | 0.3759 | 0.4597 |
| filtered_syn | 0.3664 | 0.4858 |

凍結 Test 744 張，各組取自己最佳 checkpoint。兩條獨立實作算出來一致到 8.8e-07。

**現在有信賴區間了**（EVAL-09，1,000 次重抽，2026-08-02 完成）。它撐住了主結論，
但也**收回了兩個主張**：

| 比較 | 95% CI | 判定 |
|---|---|---|
| `real_only` vs 兩個合成組 | 不重疊 | **成立** |
| `real_only` vs `standard_aug` | 重疊 | **不成立**——0.0275 在雜訊內 |
| `filtered_syn` vs `unfiltered_syn`（偵測） | 三個指標全重疊 | **不成立** |

過濾可量測的效果在**合規操作點**，不在 AP。README 已照這個改寫。

**但這不是全部。** 改用「真實影像曝光次數」而不是 optimizer 步數當橫軸，
`filtered_syn` 在**真實資料只看過 1–4 輪**時領先最多 **+0.090 mAP**，第 4–5 輪才被追過。
合成資料在標註稀少時確實有效——只是這個資料集有 5,000 張標註，
正好是它最沒用武之地的區間。

一張圖把兩半都畫出來：`reports/figures/headline.png`（README 開頭也有）。

**還有一個乾淨的勝利**：各組各選各的合規操作點後，
**`unfiltered_syn` 在任何會偵測到東西的門檻上都達不到 0.80 精確度**，`filtered_syn` 可以。
過濾決定了「能不能當合規檢查器部署」——這是過濾這條線最有力的證據。

---

## 你要做的事

### 1️⃣ 看兩張圖，告訴我合不合理（10 分鐘，非必要但有價值）

```
reports/figures/headline.png                      主結果，兩個面板
reports/figures/error_analysis/new_false_positive.png   合成資料的代價
```

第二張是「baseline 沒誤報、合成組誤報了」的並排對照，**12 張取自 291 個**。
我已經逐張看過，但這種圖最需要第二雙眼睛。
回饋範本：「第 3 排右邊那組，橘框框到的其實是……」

### 2️⃣ 決定要不要補 3 seeds（M21）

PLAN 的前置判斷寫得很清楚：**若 Filtered 組沒有提升，補 seed 不會改變結論**。
我**暫緩了**，理由記在 worklog。你可以推翻這個決定。

### 3️⃣ ✅ M20 ① —— **完成了**（2026-08-02，你鎖的那次時脈成功了）

三道閘門同時全綠：contention 0/9、clock spread **1.00**（門檻 1.15）、授權掃描 PASS。

| RT-DETRv2-R18（微調 3 類） | |
|---|---:|
| model-only | **12.79 ms / 78.2 FPS** |
| end-to-end | **16.23 ms / 61.6 FPS** |

batch 1、640×640、fp16、SM clock 鎖定 2520 MHz。
**鎖時脈是可重現的前提**，不是細節——沒鎖時同一支 harness 連續兩次跑出
11.81 ms 與 26.74 ms。鎖了之後我還是重試了 2 次才拿到 p95 乾淨的一輪，
因為鎖住的是頻率、不是其他行程。

### 🔴 兩件現在就該知道的事

**1. GPU 時脈還鎖著，請解除。**

```bash
nvidia-smi -rgc
```

**2. 你的機器上有一個 `llama-server` 正在吃 GPU。**
不是我開的。我發現時它佔 18.4 GB / 95%，現在是 **23.5 GB / 90%**——
幾乎整張 4090。它會讓任何 GPU 量測失去意義，也讓 RF-DETR 訓練跑不起來
（VRAM 不夠）。如果你不需要它，關掉；如果需要，那 M20 ② 就得排在它之後。

### ⬜ M20 ② —— 程式備好了，卡在 GPU 被佔用

`configs/training_rfdetr.yaml` 現在**真的可以載入**（先前它只寫差異、缺 17 個
`run:` 鍵，載入就 KeyError）。已加 `extends: configs/training.yaml` 與
`src/training/config.py` 的深層合併。

`scripts/probe_train_speed.py` 也備好了，量法是「兩趟取斜率」而不是一趟除以步數。
**但它需要 GPU，而 GPU 現在被 llama-server 佔滿。**
等 GPU 空出來跑這行就會得到實測時數：

```bash
uv run python -m scripts.probe_train_speed --arm real_only --short 40 --long 140
```



### 4️⃣ 想玩 demo 的話

```bash
uv run python app.py --device cpu
```

瀏覽器開 `http://127.0.0.1:7860`。圖片與影片兩個分頁**都實跑驗證過了**（CPU 與 CUDA 各一輪）。暖機後 CPU 一張 **204 ms**、CUDA **20 ms**（12 張 val 影像的中位數）。
綠框＝戴帽、紅框＝裸頭、灰框＝`person`（不帶判定）。

---

## ⚠️ 三件我必須讓你知道的事

1. **`configs/evaluation.yaml` 的 `compliance.score_threshold: 0.07` 已凍結。**
   EVAL-04 在 Validation 上選的。**看過 Test 之後再改就是在 Test 上調參**，不要動它。
   它這麼低是因為這個模型校準很差——223,200 個偵測的最高分只有 0.2495。

2. **我把一個變異提交進 main 又修回來了**（[K-21](docs/troubleshooting.md)）。
   背景有 agent 在對 `verify_readme.py` 做變異測試，我用 `git add -A` 的那一刻
   正好夾到它注入變異、還沒還原的窗口。後果是 PUB-10 的洩漏掃描
   在這台機器上**什麼都不搜尋卻照樣印 PASS**。已修，並加了會抓到它的測試。

3. **變異測試被 SIGKILL 之後，把變異留在了工作樹裡**（[K-21b](docs/troubleshooting.md)）。
   我在 commit 前逐行看 `git diff` 才發現——三十幾行合理改動裡夾著一行
   `if args.long <= args.short:` → `if args.long < 0:`。**這次沒有進 main。**
   根因是我寫的測試會呼叫 `main()`，拿掉守衛後它就真的開始訓練，harness 卡死被殺，
   `finally:` 對 SIGKILL 無效。守衛已抽成獨立函式，測試再也啟動不了昂貴作業。

---

## 背景：Phase 1 的最後一個裁決（已結案，留存備查）

上一輪要你裁決的 hard negative 放置問題，你授權我判斷，**我裁決為「接受」**，
理由記在 [K-11](docs/troubleshooting.md)。三個理由：

1. [DATA-24](docs/data_protocol.md) 確認**沒人戴的安全帽本來就不框**
   （會議室桌上 3 頂安全帽、標註 `helmet=0` 是實測驗證的），
   所以干擾物不給標註**與真實標註規則完全一致**——浮空影響的是真實度，不是標籤正確性
2. 真實資料裡「該框 vs 不該框」的界線很乾淨，模型主要從那裡學
3. 拿掉整個情境（佔 13%）的代價大於留著它

**代價**：那些干擾物偏簡單，`hard-negative 每圖誤報數` 這個**次要**指標會比預期弱。
主敘事指標（AP_small、bare-head recall）不受影響。已寫進 README。

### 想看圖的話（非必要）

全部在 `reports/figures/review/`，那個資料夾只有 15 個檔案，附一份 README 說明怎麼看。
六個情境各兩頁，我已經全部看過。

---

## 接下來會輪到你的時機

### ✅ 時機 1：Phase 2 的 Colab 往返 —— **已完成**

四組各 10,900 步跑完，權重已解壓到 `<data_root>\runs\`，
盤點稽核 PASS（0 fatal），主表已算完。實測 L4 約 1.7–1.9 it/s、
每組約 1.6–1.75 小時、四組約 6.5 小時。

> ⚠️ 我原先估「4–5 小時」，那是用「L4 大概比 4090 慢 2–2.5 倍」**推算**的，
> 實際慢約 3 倍。而且我沒有明講「這是過夜的工作」，害你熬夜盯著跑。
> 規則已寫進 [CLAUDE.md](CLAUDE.md)【工作方式】：超過 1 小時的作業，
> 時數必須來自實測，而且要明說「你現在應該去睡覺」。

**Drive 上的東西現在可以刪了**——權重與訓練狀態都已經在本機 D: 磁碟，
`results/detection_metrics.csv` 也已經進 git。

**notebook 的一個缺陷已修**：原本 checkpoint 是每跑完一組才同步回 Drive，
而且打包時漏抓了全部四組的 `trainer_state.json`（HF 寫在 `checkpoint-N/` 裡，
glob 只掃 `seed_*/`，`is_file()` 把「找不到」變成安靜跳過）。
已移進被測模組並補了 6 條測試，M21 補 seed 時不會再漏。

### 🔴 發佈前必須先決定的一件事：clone 是 631 MiB

> ⚠️ **我先前說「437 MB」是量錯了東西。** 那是工作目錄裡被追蹤的檔案總和。
> 決定「別人 clone 要下載多少」的是 git 物件庫，它裝著每個檔案的**每一個歷史版本**。
> 打包後實測 **631.3 MiB**，比我說的還糟。

| | 歷史累計位元組 | 佔比 |
|---|---:|---:|
| `reports/figures/` | **629.1 MB** | **94%** |
| 其他全部（程式、文件、設定、results） | 40.9 MB | 6% |

也就是說，**這個 repo 的本體只有 41 MB**，剩下全是診斷圖。
現有 149 張圖裡**只有 31 張被任何文件引用**，其餘 118 張的唯一提及處
是產生它的那支腳本——那不是讀者跟得下去的引用。

**為什麼時機關鍵**：從 HEAD 刪檔案不會讓 clone 變小，歷史裡的 blob 還在。
只有 `git filter-repo` 改寫歷史才有用。而**這個 repo 還沒 push 過**
（`git remote -v` 是空的），現在做零副作用；push 之後就是 force-push
覆蓋已公開歷史，難度差一個等級。

**我照你的授權決定了：清掉。** 預估 clone 從 631 MB 降到約 100 MB。
逐檔清單、理由、指令與事後驗證都在 **[reports/repo_slimming_plan.md](reports/repo_slimming_plan.md)**——
它是腳本產的、可以隨時重跑，本身不動任何檔案。**請過目再執行。**

研究過程的證據不會消失：試過哪 18 輪 labeler 迭代、9 條合成路線、
各自結果如何，都在 worklog 的文字裡，而且 grep 得到。

---

### ⏳ 時機 2：發佈前 → **建 GitHub repo 與 push**（預計 20 分鐘）

依規矩，**建立 remote、push、Hugging Face 上傳一律由你親自執行**，我只準備內容與逐行指令。
到時會先跑個人 skill `publish-repo` 做完整驗收。

---

## 一件現在就可以順手做的事（非必要）

全域 `git config user.email` 目前是你的學校信箱 `03131047@gm.scu.edu.tw`。
本專案三個 repo 的 repo-local 設定都已改成 noreply，但**全域沒動**——
下一個新 repo 還是會中招：

```bash
git config --global user.email "61350295+kuotunyu@users.noreply.github.com"
```

另外 `1_DefectForge` 與 `3_FormosaNLU` 早期的 commit 已經寫進學校信箱，
要清掉得改寫歷史（這是你親自執行的動作）：

```bash
cd "<path-to-1_DefectForge>" && git filter-repo --email-callback 'return b"61350295+kuotunyu@users.noreply.github.com" if b"gm.scu.edu.tw" in email else email' --force
```

本專案（`2_SafeSynth`）從第一筆 commit 就是乾淨的，不需要處理。

---

## 這個專案現在誠實的樣子

值得先知道，因為它會影響你怎麼跟別人介紹這個專案。**有三個限制**：

**1. H4 這道預先註冊的閘門沒有通過。**
貼上痕跡的可偵測度 AUC **0.9053**（95% CI 0.9013–0.9090，106,144 個 patch），
預先登記的上限是 0.60。試了 9 條合成路線與 18 輪 labeler 迭代都翻不過來。
我們的處置**不是**放寬門檻或換個弱一點的分類器來「通過」，
而是把它當成本專案的一項發現照實發表，並把生成量上限壓在 1×（不做 2×）。

**2. copy-paste 在 3,500 張背景上會飽和。**
接受率不是常數——去重要拿新樣本跟**所有已接受的**比，所以 pool 越大接受率越低：
2,000 張候選時 58.4%、10,000 張時 33.8%、14,000 張時 29.8%，
而且 `NEAR_DUPLICATE_SYNTHETIC` 變成最大宗的拒絕原因。
要湊到 1×（3,500 張）需要 14,000 張候選。這是方法的天花板，不是門檻太緊。

**3. hard negative 的放置真實度只修好一半**（見上方要你裁決的那件事）。

三個限制**都可以寫，而且都比假裝沒有誠實。**
細節見 [ADR-011](docs/decisions.md#adr-011) 與 [ADR-013](docs/decisions.md#adr-013)。
