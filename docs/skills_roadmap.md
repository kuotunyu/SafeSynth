# Skills Roadmap

> 這個專案的協作習慣之一是**一邊實作、一邊把穩定下來的流程長成 skill**。
> 本檔記錄：現在有哪些、什麼時候該長出新的、以及什麼**不該**變成 skill。

---

## 抽取規則

**同一個流程用手動做過兩次以上、而且步驟已經穩定下來，才寫成 skill。**

這條規則存在的理由是負面的：現在就把「跑合成引擎」寫成 skill，
等於憑空捏造一套還不存在的 CLI 介面與參數名稱，實作完必然要整份重寫。
**先做，做順了，再固化。**

判斷放哪一層：

| 層 | 載入時機 | 放什麼 |
|---|---|---|
| `CLAUDE.md` | 每個 session 全量載入 | **恆真的事實**與紅線 |
| `docs/*.md` | 被明確指名時 | 規格與可查證的細節 |
| `.claude/skills/*/SKILL.md` | 語意觸發或使用者打 `/名稱` | **多步驟、有分支的流程** |

流程有順序、有分支、有「不過就停」——這是 `CLAUDE.md` 的條列格式表達不了的東西，
所以才需要 skill。反過來，一句話講得完的事實不該做成 skill。

---

## 現有

### `safesynth` — 開工恢復脈絡 ＋ 里程碑收尾
**建立於**：M0

兩個流程：

- **開工**：讀 `docs/worklog.md` 的現況快照 → **四路交叉證偽** → 不一致先修 →
  找出 `PLAN.md` 的下一項 → 讀對應規格 → 回報狀態請使用者確認
- **收尾**：當場跑驗證指令並貼真實輸出 → 勾 `PLAN.md` 並補「驗證於 `<sha>`」→
  覆寫快照 ＋ 追加日誌 → 產出 commit 訊息 → 給「換你做」清單

**為什麼不放 `CLAUDE.md`**：這是有分支的多步驟流程（光「快照與現實不一致」就有三種情況、
各有不同處置），寫進 `CLAUDE.md` 要四十行、每個 session 都吃 context，
但實際上一週只用一兩次。典型的 skill。

### `safesynth-env` — Windows 原生環境自檢與修復
**建立於**：M0

逐項跑 `docs/environment.md` 的驗證指令表，
把常見故障對應到 `docs/troubleshooting.md` 的條目與修法。

**為什麼獨立一支**：Windows 原生（不用 WSL）是本專案的獨有風險集中區，
而且環境問題的特徵是「隔很久才遇到一次，遇到時完全想不起來上次怎麼修的」——
正是 skill 最有價值的場景。

---

## 觀察中（等它們穩定下來再寫）

| 候選 | 觸發時機 | 目前為什麼還不寫 |
|---|---|---|
| `safesynth-compose` | 跑 cutout bank／合成引擎、產預覽圖、抽查品質 | M8–M13 的 CLI 介面、參數名、預覽圖要看什麼，現在都還不存在。憑空寫必然重寫 |
| `safesynth-freeze` | split 凍結前的完整驗證與封存 | 只會執行一次（M5）。做完之後如果發現流程值得複用（例如 Phase 2 要重新凍結），再抽出來 |
| `safesynth-eval` | Phase 2 的五組評測與圖表產生 | 屬於 Phase 2，且評測腳本還不存在 |
| `safesynth-calibrate` | 重跑校準工具並回填 config 門檻 | M6 之後如果門檻需要反覆調整（很可能），這支的價值會浮現 |

---

## 明確不做成 skill

### 發佈流程
已有**個人** skill `publish-repo`（涵蓋洩漏掃描、體積檢查、必備檔案、CI、
數字可追溯性、stale 文件偵測、commit 切分、tag/Release）。

**不要重寫**，理由有二：
1. 重複維護，兩份會漂移
2. **同名時個人 skill 會覆蓋專案 skill**，寫了也不會生效

發佈時直接呼叫 `publish-repo`。

### 規格對帳
「哪些規格 ID 還沒實作」是純機械檢查，應該寫成 `scripts/check_traceability.py`
讓 CI 跑，不是 skill。

**判準**：skill 用來裝需要**判斷**的東西；**能寫成斷言的東西就寫成斷言**。

---

## 命名

前綴一律 `safesynth-`（`safesynth` 本身除外，它對應兄弟專案的 `/defectforge` 慣例）。

**已檢查無衝突**：與 29 支個人 skill（`publish-repo`、`find-skills`、`webapp-testing`、
`impeccable`、`canvas-design`、`theme-factory`、`frontend-design`、22 支 `tw-opendata-*`）
及外掛 skill（`dataviz`、`run`、`init`、`review`、`simplify` 等）皆無交集。

**這件事必須檢查**，因為個人 skill 會覆蓋同名的專案 skill——撞名的話專案 skill 直接失效。
