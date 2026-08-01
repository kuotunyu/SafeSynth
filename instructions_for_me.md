# 換你做 — 02-safesynth-ppe

> 這份檔案只留**需要你親自判斷或執行外部動作**的事。
> 本機可逆的實作、測試與 commit 已授權自動完成；不會自行建立 remote、push 或發佈。
> 最後更新：2026-08-01（Colab 四組訓練進行中）

---

## 🌅 醒來先看這裡

Colab 的四組訓練是 2026-08-01 深夜開始跑的，需要約 **6.5 小時**（每組約 1.6 小時）。
產出寫在 **Drive**，不是那台虛擬機，所以 runtime 被回收也不會掉。

**第一步：看 Drive 的 `sdg-portfolio/02-safesynth-ppe/` 有沒有 `results_colab.zip`。**

| 情況 | 你要做的 |
|---|---|
| **有 zip** | 下載，然後在 repo 根目錄跑下面那行指令。不用手動解壓 |
| **沒有 zip，Colab 還在跑** | 什麼都不用做，等它跑完 |
| **沒有 zip，Colab 斷了** | 回 Colab 按「執行階段 → 全部執行」。**已完成的組不會重練**——它會從 Drive 抓回 checkpoint，Trainer 看到步數已達標就跳過，只重跑一次評測 |

拿到 zip 之後，這行會自動解壓到 `results/colab/` 並做完整性稽核：

```bash
uv run python -m scripts.audit_colab_results --archive ~/Downloads/results_colab.zip
```

它會檢查四組是不是真的可比較——**四組吃的真實影像是否完全相同**（比對 SHA256 摘要）、
optimizer 步數是否相等、filtered 與 unfiltered 是否等量、`+Standard Aug` 有沒有拿到
光度增強（[EXP-01](docs/experiment_protocol.md)）、以及每組是不是真的評測過
（[K-18](docs/troubleshooting.md) 的教訓）。

有問題它會**列出完整清單**寫進 `reports/m16_colab_audit.md` 並回傳非零；
**在清單清空之前，任何表格都不准建立在這批結果上**。跑完把結果貼給我就行。

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

### ✅ 時機 1：Phase 2 的 Colab 往返 —— **進行中**，見本檔開頭的「醒來先看這裡」

實測數字（取代先前的估計）：L4 上約 **1.7–1.9 it/s**，每組 10,900 步約
**1.6–1.75 小時**，四組約 **6.5 小時、約 27 CU**。

> ⚠️ 我原先估「4–5 小時」，那是用「L4 大概比 4090 慢 2–2.5 倍」**推算**的，
> 實際慢約 3 倍。而且我沒有明講「這是過夜的工作」，害你熬夜盯著跑。
> 規則已寫進 [CLAUDE.md](CLAUDE.md)【工作方式】：超過 1 小時的作業，
> 時數必須來自實測，而且要明說「你現在應該去睡覺」。

你的 Colab 是 **Pro+（500 CU/月，含 24 小時背景執行）**，額度很夠。

**已知限制**：checkpoint 是**每跑完一組**才同步回 Drive，
所以中途斷線會損失「當下那一組」（最多約 1.6 小時），已完成的組全部安全。
改成每存一次就同步是之後要做的改進——不在你睡覺、沒人盯著的時候動正在跑的 notebook。

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
cd "C:/Users/3Hml/Desktop/mySyntheticData/1_DefectForge" && git filter-repo --email-callback 'return b"61350295+kuotunyu@users.noreply.github.com" if b"gm.scu.edu.tw" in email else email' --force
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
