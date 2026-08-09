# SafeSynth README 正體中文與 Mermaid 重構實作計畫

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 SafeSynth README 重構為正體中文主文、精簡 English Abstract、研究結論優先且包含三張可由 GitHub 正確呈現的 Mermaid。

**Architecture:** README 保持單一入口文件，依「問題與結論 → 方法與 protocol → evidence → 使用與重現」排列。既有 verifier 繼續作為數字、disclosure 與連結的權威 gate；Mermaid 另外經 extraction 與 renderer 驗證。

**Tech Stack:** GitHub Flavored Markdown、Mermaid、Python、pytest、Ruff、既有 `scripts.verify_readme` 與 release checks。

## Global Constraints

- 主文使用正體中文；技術名稱、arm、metric、CLI 與路徑保留原文。
- 開頭保留一段精簡 English Abstract。
- 不使用非必要 emoji 或裝飾性 Unicode 圖示。
- 三張 Mermaid 各自只呈現整體 pipeline、四組 ablation、evaluation protocol。
- 不改變任何實驗數據、artifact、frozen protocol 或研究結論。
- 所有現有 metrics source annotation、必要 disclosure 與有效連結都必須保留。
- 本次使用者指示覆寫 `CLAUDE.md` 中 README 使用英文的舊規則。
- commit 不加入 `Co-Authored-By`，push 由使用者執行。

---

### Task 1: 建立研究結論優先的 README 主體

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `results/detection_metrics.csv`、`results/rfdetr_detection_metrics.csv`、現有 README disclosure 與 repository links。
- Produces: 保留現有 verifier contract 的正體中文 README 章節與結果表。

- [ ] **Step 1: 保存基準驗證結果**

Run:

```powershell
uv run python -m scripts.verify_readme
```

Expected: 現有英文 README 通過數字、disclosure 與連結檢查。

- [ ] **Step 2: 重排並改寫 README 主體**

以 approved design 的資訊架構改寫 `README.md`：專案定位、English Abstract、核心結論、發布資源、方法、結果、限制、Demo、Dataset、安裝、重現、文件索引、License。保留兩份 metrics CSV 的 annotation 與所有 verifier 所需文字。

- [ ] **Step 3: 檢查語言與專業風格**

Run:

```powershell
rg -n "[😀-🙏🌀-🫿]" README.md
rg -n "^#{1,4} " README.md
```

Expected: emoji scan 無結果；章節順序符合 approved design。

- [ ] **Step 4: 驗證 README contract**

Run:

```powershell
uv run python -m scripts.verify_readme
```

Expected: PASS。

### Task 2: 加入並驗證三張 Mermaid

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 的方法、ablation 與 evaluation 章節。
- Produces: 三個內嵌 Mermaid code fence，可由 GitHub renderer 直接顯示。

- [ ] **Step 1: 加入整體 pipeline flowchart**

建立三層 flowchart，區分 frozen real-data foundation、Synthetic Data pipeline、training/evaluation/release；Test 不連回 generator 或 training decision。

- [ ] **Step 2: 加入四組 controlled ablation flowchart**

從 real Train 與同源 synthetic pool 分出 `real_only`、`standard_aug`、`unfiltered_syn`、`filtered_syn`，圖中標示等量 synthetic arms、固定 optimizer-step budget 與 real-only Validation/Test。

- [ ] **Step 3: 加入 evaluation protocol sequence diagram**

依 Training、Validation checkpoint selection、Validation operating-point selection、frozen Test evaluation、image-level bootstrap、Reporting 的順序繪製；Test 不參與模型選擇。

- [ ] **Step 4: 抽取 Mermaid code fence**

Run:

```powershell
uv run python "C:\Users\3Hml\.agents\skills\design-doc-mermaid\scripts\extract_mermaid.py" README.md --output-dir .codex-readme-mermaid
```

Expected: 抽出三個 Mermaid source files。

- [ ] **Step 5: 使用 Mermaid renderer 驗證三張圖**

Run:

```powershell
Get-ChildItem .codex-readme-mermaid -Filter *.mmd | ForEach-Object { mmdc -i $_.FullName -o ($_.FullName + '.svg') -b white }
```

Expected: 三個命令皆 exit zero，且各產生非空 SVG。驗證完成後移除 `.codex-readme-mermaid` 暫存目錄。

### Task 3: 完整 repository 驗證與提交

**Files:**
- Modify: `README.md`
- Modify: `docs/worklog.md`

**Interfaces:**
- Consumes: 完成的 README 與三張已驗證 Mermaid。
- Produces: 可提交、可發布且留有 worklog 證據的變更。

- [ ] **Step 1: 記錄 README 重構與驗證證據**

在 `docs/worklog.md` 追加簡短紀錄，包含正體中文重構、三張 Mermaid、執行過的 verifier 與測試，不複製實驗數值。

- [ ] **Step 2: 執行 targeted checks**

Run:

```powershell
uv run pytest -q tests/test_verify_readme.py tests/test_verify_repository_links.py
uv run python -m scripts.verify_readme
uv run python -m scripts.check_forbidden_licences
```

Expected: tests 無 failure；兩個 script 都 exit zero。

- [ ] **Step 3: 執行完整品質檢查**

Run:

```powershell
uv run ruff check .
uv run pytest -q
uv lock --check
git diff --check
```

Expected: 所有命令 exit zero。

- [ ] **Step 4: 檢查 identity 與變更範圍**

Run:

```powershell
git config user.name
git config user.email
git status --short
git diff --stat
```

Expected: identity 為 `kuotunyu`；變更只包含 approved README、worklog 與本計畫。

- [ ] **Step 5: 建立本機 commit**

Run:

```powershell
git add README.md docs/worklog.md docs/superpowers/plans/2026-08-09-readme-zh-tw-mermaid.md
git diff --cached --check
git commit -m "docs: rewrite README in Traditional Chinese"
```

Expected: commit author 與 committer 都是 `kuotunyu`，commit message 不含 trailer。
