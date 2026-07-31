# 換你做 — 02-safesynth-ppe

> 這份檔案只留**需要你親自判斷或執行外部動作**的事。
> 本機可逆的實作、測試與 commit 已授權自動完成；不會自行建立 remote、push 或發佈。
> 最後更新：2026-07-31（`M11` 由 [ADR-011](docs/decisions.md#adr-011) 結案之後）

---

## 現在：**有兩張圖等你看**（約 10 分鐘）

`M13` 已完成。需要你裁決的只有 hard negative 的真實度，見下方「時機 1」。

已經確認過的：

| 項目 | 狀態 |
|---|---|
| M9 的 H6 hard-negative 簽核 | ✅ 你已批准（0/64 真正安全帽） |
| v23 labeler 的 48 格審查 | **不用看了**——v23 已退回，理由與那 48 格無關（見下） |
| H4 硬閘門 | 已結案為「已知限制」，不再阻擋 |
| 遠端 GitHub repo | 尚未建立（要建的時候由你執行） |

### 為什麼 48 格不用看了

v23 的數值稽核確實通過（precision 0.8750／recall 0.8273／median IoU 0.8303，
已獨立重算驗證逐位相同）。但退回它的理由是**分數校準**，那是目視看不出來的：

- 訓練在 **epoch 3** 就選為最佳
- 48 圖 audit 裡模型的**最高信心只有 0.1396**
- **TP 與 FP 的分數分布幾乎完全重疊**（中位數 0.0583 vs 0.0481）

也就是**不存在任何門檻能把對的和錯的分開**，precision 0.875 是把門檻壓到 0.035
（近乎雜訊底）換來的。這是欠訓練，不是標註品質問題。
而且這條 labeler 路線本身已隨 ADR-011 停止。

（如果你純粹好奇想看一格：`reports/figures/supervised_labeler_v23_model_review/review_page_03.png`
的**格 35**——GT 沒框，模型卻框了 3 個放在桌上、沒人戴的安全帽。）

---

## 接下來會輪到你的三個時機

### ✅ 時機 1（**現在**）：看兩張預覽圖，約 10 分鐘

M13 已完成。每張圖是 **2×3 共 6 格、放大兩倍、左上角有黃底大編號**，
每個情境兩頁（`_p1` / `_p2`）。回饋時報**編號**就行。

**必看這兩張：**

| 檔案 | 要判斷什麼 |
|---|---|
| `reports/figures/preview_head_no_helmet_p1.png`（與 `_p2`） | 主敘事指標 #2 的來源。裸頭有沒有**長在身體上**、helmet→head 替換處**有沒有殘留帽緣** |
| `reports/figures/preview_hard_negative_p1.png`（與 `_p2`） | **這張要你裁決**，見下 |

**hard negative 那張要你回答一個問題。**
我已經自己看過並記成 [K-11](docs/troubleshooting.md)：那些黃色圓頂**明顯像貼上去的色塊**——
沒有接地陰影、浮在任意深度、光照與場景不一致。
我的判斷是「不修，列為已知限制」，因為 ADR-011 已經接受「貼上痕跡可被偵測」是本專案的公開發現。

**但這種程度的貼上感你能接受嗎？**
- 能接受 → K-11 維持「已知限制」，Phase 2 照常推進
- 太假 → 我把它升級成要修的項目（接地陰影 → 依深度的尺寸先驗 → 局部光照方向匹配），
  修完必須重跑 H4 證明 AUC 真的下降，否則不算有效

**其餘四個情境**（`small_distant`／`partial_occlusion`／`crowded`／`low_light_blur`）
我看過了，沒有需要你裁決的問題，想看的話檔案也在同一個資料夾。

回饋格式，報編號就好：
```
preview_head_no_helmet_p1 第 04 格：帽緣有殘影
preview_hard_negative_p2 第 09 格：這顆浮得太誇張
```

### ⏳ 時機 2：Phase 2 的 `M15` → `M16` → **跑一趟 Colab**（預計 4–8 小時等待）

我會先在本機把 notebook 跑通 1-step smoke test，然後給你一份照做就行的清單：
複製到 Drive 的哪個路徑、Runtime 選什麼、需要哪些 Secrets、預估時數與 compute units、
跑完下載哪些檔案放回 `results/colab/` 的哪裡。

你的 Colab 是 **Pro+（500 CU/月，含 24 小時背景執行）**，
估計 8 次訓練用 L4 約 113 CU，額度很夠。

### ⏳ 時機 3：發佈前 → **建 GitHub repo 與 push**（預計 20 分鐘）

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

值得先知道，因為它會影響你怎麼跟別人介紹這個專案：

**H4 這道預先註冊的閘門沒有通過**（貼上痕跡的可偵測度 AUC 0.9159，上限 0.60），
而且試了 9 條合成路線與 18 輪 labeler 迭代都翻不過來。

我們的處置**不是**放寬門檻或換個弱一點的分類器來「通過」，
而是把它當成本專案的一項發現照實發表，並把生成量上限壓在 1×（不做 2×），
帶著這個限制去測「即使痕跡可偵測，合成資料對下游偵測器還有沒有幫助」。

- 有幫助 → 「可偵測的貼上痕跡不阻礙遷移」，是有意義的正面結果
- 沒幫助 → AUC 0.9159 正好提供機制解釋

**兩種結果都可以寫，而且都比假裝通過誠實。** 細節見 [ADR-011](docs/decisions.md#adr-011)。
