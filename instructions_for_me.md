# 換你做 — 02-safesynth-ppe

> 這份檔案只留**需要你親自判斷或執行外部動作**的事。
> 本機可逆的實作、測試與 commit 已授權自動完成；不會自行建立 remote、push 或發佈。

---

## 醒來後先看

### 1. 本專案目前不用你操作

- M0–M5 已完成，資料與 split 已凍結；遠端 GitHub repo 仍未建立
- 本機所有 commit author 都是
  `kuotunyu <61350295+kuotunyu@users.noreply.github.com>`
- `Co-Authored-By:` trailer 為 0；沒有 remote、push 或其他 contributors
- M6 起可繼續本機施工，不需要 Colab

### 2. 可選的人眼複核（不阻塞後續）

我已逐張檢查過，結論已記在 ADR-007；你起床後若想複核，依序看：

1. `reports/figures/h1_helmet_contact_sheet.png` 與
   `reports/figures/h1_head_contact_sheet.png`：helmet 是戴帽的整顆頭，head 是裸頭
2. `reports/figures/h3_clip_largest_groups.png`：同一列應是連拍或保守合併的同構場景
3. `reports/figures/h5_placement_priors.png`：head/helmet 有中央水平帶，person 較發散
4. `reports/figures/class_distribution.png`：Train/Val/Test 與三類分布沒有離譜偏斜

若看到問題，回覆「檔名／第幾列第幾格／問題」即可。

### 3. 其他兩個 repo 的學校信箱仍要你親自處理

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
| M8 | `reports/figures/bank_<class>_grid.png`（洋紅色背景） | cutout 有沒有背景滲漏、光暈、或第二個物件入鏡 |
| **M9** | hard negative 的 8×8 contact sheet | **裡面有幾張其實是真正的安全帽？** 這是必須人工簽核的一關 |
| M12 | 12 通過 vs 12 被拒 並排圖 | **被拒的樣本看起來有問題嗎？** 若沒問題就是門檻設錯了 |
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

## Phase 1 不需要 Colab

本階段全部在你的 4090 上跑完：SAM2 只做推論（兩趟合計不到一小時），
合成是純 numpy/cv2。Colab 要到 Phase 2 的 RT-DETRv2 訓練才會用到。
