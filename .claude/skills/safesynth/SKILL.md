---
name: safesynth
description: 02-safesynth-ppe 專案的開工與收工儀式。當使用者說「繼續」「我回來了」「接手這個專案」「現在做到哪」「這專案的狀態」，或新 session 第一句就要推進 SafeSynth 時，走「開工恢復脈絡」；當使用者說「這個里程碑做完了」「M3 完成」「收尾」「這階段結束」，或你自己判斷某個 M 項目的驗證已全數通過時，走「里程碑收尾」。開工流程的核心是用 git log／git status／檔案存在性／PLAN.md 勾選狀態四路交叉證偽 docs/worklog.md 的現況快照——不要直接相信快照。收尾流程要求驗證指令當場執行並貼出真實輸出才准勾選。
---

# SafeSynth — 開工與收工儀式

本專案的長期記憶分成四層：
[CLAUDE.md](../../../CLAUDE.md)（恆真規則）、
[PLAN.md](../../../PLAN.md)（里程碑與驗證方法）、
[docs/worklog.md](../../../docs/worklog.md)（施工日誌 ＋ 現況快照）、
[docs/decisions.md](../../../docs/decisions.md)（ADR）。

---

## 流程 A — 開工恢復脈絡

### 核心原則：快照是待證偽的假設，不是真相

`docs/worklog.md` 頂端的現況快照是人寫的、會過期。
幾週後它和 git 幾乎一定對不上。
**所以第一步不是「相信快照」，是「讀完立刻用四條指令去打它的臉」。**

少了證偽這一步，這就只是另一份會爛掉的進度文件。

### 步驟

**1. 讀快照（只讀那十幾行，不要讀整份日誌）**

`docs/worklog.md` 的「現況快照」區塊。

**2. 四路交叉證偽**

```bash
git log --oneline -8
git status --porcelain
```
```powershell
# 快照聲稱已存在的產物，逐一確認
Test-Path splits\split_manifest.json, splits\MANIFEST.sha256, uv.lock
```
再讀 `PLAN.md`，找出**第一個 `[~]` 或 `[ ]`** 的項目。

**3. 處理不一致——不一致本身是最重要的資訊**

| 情況 | 意義 | 處置 |
|---|---|---|
| `git log` 最新 commit **比快照記的新** | 上次收工沒有更新快照就 commit 了 | 以 git 為準，重寫快照，並在日誌補一筆說明 |
| working tree **是髒的** | 上次工作到一半被中斷 | **先報告有哪些未提交的改動**，問使用者要繼續還是丟棄，不要自己決定 |
| 快照說某檔案存在但**實際不存在** | 產物被刪、或根本沒產出成功 | 把對應的 `PLAN.md` 項目退回 `[ ]` 或 `[~]`，修正快照 |
| `PLAN.md` 有 `[x]` 但**缺 `驗證於` 行**，或該 sha 不在 `git log` | 勾選不誠實 | 退回該項目狀態並報告 |

**任何不一致都要先修文件、再開始做事。**

**4. 讀規格**

找到下一項要做的里程碑後，讀它「對應規格」欄位指向的 `docs/` 章節。
不要憑記憶做事——規格裡的門檻與判定式是唯一來源。

**5. 回報並等確認**

輸出格式：

```
【現況】
- 最後 commit：<sha> <訊息第一行>
- working tree：乾淨 / 有 N 個未提交改動（列出）
- 已凍結不得再動：<檔案清單>
- 快照與 git 是否一致：一致 / 有 N 處不一致（已修正，列出）

【下一步】
- 里程碑：<M?> <標題>
- 對應規格：<ID 範圍>
- 我打算做：<兩三句>
- 驗證方式：<PLAN.md 該項的驗證欄位>

這樣對嗎？
```

**等使用者確認後才動手。**

---

## 流程 B — 里程碑收尾

### 步驟

**1. 當場執行驗證指令，貼出真實輸出**

從 `PLAN.md` 該里程碑的「驗證」欄位逐條執行。

**不接受「應該會過」「照理說沒問題」。** 沒跑過就是沒過。
產圖類的驗證要**真的把圖打開看**（CLAUDE.md 的工作方式明訂），
看到不合理就先修再說。

**2. 勾選 `PLAN.md`**

`[ ]` / `[~]` → `[x]`，並把
```
  - **驗證於**：（未完成）
```
改成
```
  - **驗證於**：`<sha>` @ YYYY-MM-DD
```
sha 是**這次 commit 之後**的實際值，所以這一步和 commit 是連動的：
先 commit 拿到 sha，再回填，再用一個小 commit 收尾；
或先寫好其餘內容，commit 後補上 sha 再 amend。**不要編造 sha。**

**3. 更新 `docs/worklog.md`**

兩個動作：
- **覆寫**頂端的「現況快照」（只留最新一份）
- 在「工作日誌」**最上面插入**一筆新紀錄（append-only，不改舊的）

日誌格式見該檔頂端的範本。**驗證欄位要貼真實輸出的摘要，不是「通過」兩個字。**

**4. 產生 commit 訊息**

conventional commit 格式，**訊息用英文**（與 README 同樣是對外可見的內容）：
```
feat(data): group images by pHash and freeze the 70/15/15 stratified split
docs(spec): record the calibration caveat for FILT-07
chore: scaffold project skeleton and gitignore
```

⚠️ **不要帶 `Co-Authored-By:` trailer**——它會讓 GitHub repo 首頁多出一個貢獻者。
個人 skill `publish-repo` 的第 1.5 關會擋這件事，事後要用 `git filter-repo` 清很麻煩。

**git 寫入動作交由使用者執行**：把指令列出來給使用者，不要自己 commit。

**5. 「換你做」清單**

結尾一定要給。格式：

```
【換你做】
1. <具體動作>（預計 X 分鐘）
2. 打開 reports/figures/<檔名>.png 抽查：<要看什麼>
   回饋範本：「grid preview_head_no_helmet 第 (3,4) 格：安全帽浮在頭上方 → 調緊 overlap_y_min」
3. 執行以下 git 指令：
   <逐行列出>

沒有的話就寫「無」。
```

---

## 常見情境速查

| 使用者說 | 走哪個流程 |
|---|---|
| 「繼續」「我回來了」「做到哪了」 | A |
| 「這個做完了」「M5 完成」「收尾」 | B |
| 「環境好像壞了」「裝不起來」 | 改呼叫 `/safesynth-env` |
| 「要發佈了」「push 上去」 | 改呼叫個人 skill `publish-repo`（**不要自己重寫發佈流程**） |
| 「這個門檻要不要改」 | 先讀 `docs/filtering_spec.md` 對應的 FILT-ID，改 `configs/*.yaml` 而**不是**改文件裡的數字，並追加一則 ADR |
