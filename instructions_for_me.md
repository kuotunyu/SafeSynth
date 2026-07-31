# 換你做 — 02-safesynth-ppe

> 這份檔案只留**需要你親自判斷或執行外部動作**的事。
> 本機可逆的實作、測試與 commit 已授權自動完成；不會自行建立 remote、push 或發佈。
> 最後更新：2026-07-31（`M11` 由 [ADR-011](docs/decisions.md#adr-011) 結案之後）

---

## 現在：**沒有任何事情在等你**

`M13`（生成 1× 合成資料）不需要你操作，可以無人值守跑完。

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

### ⏳ 時機 1：`M13` 跑完之後 → **看合成預覽圖**（預計 15 分鐘）

這是 Phase 1 唯一真正需要人眼的一關。到時會產出在 `reports/figures/`：

| 檔案 | 要看什麼 |
|---|---|
| `preview_small_distant.png` | 縮小後的安全帽/頭**看起來像遠處的人**，不是像貼上去的小貼紙 |
| `preview_head_no_helmet.png` | 裸頭要**長在身體上**，不是浮空；helmet→head 替換處**不能有帽緣殘影** |
| `preview_partial_occlusion.png` | 遮擋要**像真的被擋住**，不是硬切一半 |
| `preview_crowded.png` | 多人重疊的**前後關係合理**（近的在前、遠的在後） |
| `preview_low_light_blur.png` | 低光/模糊是**整張圖一致**的，不是只有貼上的物件變暗 |
| `preview_hard_negatives.png` | **這張刻意不畫框**——確認畫面裡的黃色圓形物**真的都不是安全帽** |
| `filter_pass_reject_grid.png` | 12 通過 vs 12 被拒並排。**重點是看「被拒」那半邊**：如果某張明明沒問題卻被拒，就是門檻設錯了 |

每格都會標 `sample_id`。回饋只要指出**哪張圖、哪一格、什麼問題**，不用給解法：

```
preview_head_no_helmet 第 (3,4) 格：安全帽浮在頭上方沒接觸到
preview_crowded 第 (1,2) 格：後面那個人蓋在前面的人身上，前後關係反了
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
