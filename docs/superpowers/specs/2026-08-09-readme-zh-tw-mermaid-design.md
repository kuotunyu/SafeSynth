# SafeSynth README 正體中文與 Mermaid 重構設計

## 決策摘要

README 採「研究結論優先」的資訊架構。正文使用正體中文，技術名稱、模型名稱、
指標名稱與翻譯後容易失真的術語保留原文；開頭提供一段精簡 English Abstract，
作為國際讀者入口。

本設計直接依使用者要求覆寫 `CLAUDE.md` 中「README 用英文」的舊規則。其餘專案
規則繼續適用，尤其是數字可追溯、負面結果完整揭露、相對連結有效，以及不得加入
`Co-Authored-By`。

## 目標

- 讓第一次進入 repository 的讀者快速理解研究問題、方法、主要結論與產出。
- 以正體中文清楚說明負面結果，不把沒有統計支持的點估計差異寫成勝負。
- 使用 Mermaid 將 pipeline、controlled ablation 與 evaluation protocol 視覺化。
- 保留 README verifier 所依賴的 metrics source、數字來源、disclosure 與連結。
- 維持專業、理性的視覺語言，不使用非必要 emoji 或裝飾性符號。

## 不納入範圍

- 不改變模型、資料、訓練、評測或發布 artifact。
- 不重算任何實驗結果，也不改寫 frozen protocol。
- 不新增互動式網站或外部文件生成系統。
- 不把 README 擴寫成完整論文；細節仍由 `docs/`、`results/` 與 reports 承接。

## 讀者與語言

主要讀者是懂基本 Computer Vision 與 Object Detection 的正體中文使用者。第二讀者
是從 GitHub 或 Hugging Face 進入、只需要快速理解專案的英文讀者。

語言規則如下：

- 標題、正文、圖說與操作說明以正體中文撰寫。
- 專有名詞優先保留原文，例如 Synthetic Data、Hard-Hat Detection、SAM、
  RT-DETRv2、RF-DETR、artifact gate、bootstrap confidence interval。
- 第一次出現且中文說明有助理解時，採「中文說明（Original Term）」格式；之後只用
  Original Term。
- 程式名稱、arm 名稱、metric key、CLI 與檔案路徑維持原樣。
- English Abstract 只概述問題、controlled experiment 與主要結論，不複製全文。

## 資訊架構

README 依讀者決策順序排列：

1. 專案定位與 English Abstract。
2. 核心結論與發布資源。
3. 整體架構與資料生成方法。
4. 四組 controlled ablation 與防洩漏 evaluation protocol。
5. RT-DETRv2 與 RF-DETR 結果、confidence interval 與結果解讀。
6. artifact gate、copy-paste ceiling、hard-negative placement 等限制。
7. Demo、Dataset、安裝、重現與文件索引。
8. License 與引用資訊。

開頭只保留足以建立正確心智模型的內容；長篇機制解釋放在對應結果或限制章節，避免
同一結論前後重複。

## Mermaid 設計

### 整體 pipeline

使用 `flowchart`。以三個視覺區塊呈現 frozen real-data foundation、Synthetic Data
pipeline、training/evaluation/release。資料只沿單一主方向移動，Test 與 generator
之間不建立連線，以視覺方式強調 leakage boundary。

### 四組 controlled ablation

使用 `flowchart`。從同一個 real Train 與同一個 synthetic pool 分流到四個 arm，
明確標示 filtered 與 unfiltered 等量、各組 optimizer-step budget 相同，以及
Validation/Test 全為 real images。圖下文字解釋 synthetic arms 的 real-image
exposure 較少是研究 confound，而非圖中塞入過多註解。

### Evaluation protocol

使用 `sequenceDiagram`。參與者為 Train、Validation、Test 與 Reporting。時序依序為
訓練、Validation 選 best checkpoint、Validation 選 operating point、frozen Test
final evaluation、image-level bootstrap、結果與限制一起發布。Test 不回傳資訊給
training decision，避免圖意暗示測試集參與調參。

### Mermaid 視覺規則

- 每張圖只回答一個問題。
- 不使用 emoji、圖示字元或漸層背景。
- 使用淺色填色、深色文字與清楚邊框，並讓相同語義沿用相同顏色。
- 節點標籤保持短句；必要細節放在圖後說明。
- 僅使用 GitHub 支援穩定的 Mermaid 語法。
- Mermaid 原始碼直接嵌入 README，使 GitHub 可縮放、複製與檢視原始碼。

## 內容保存與縮減

以下內容必須保存：

- GitHub Release、source repository、Hugging Face dataset 與 model links。
- 四個 arm 的主要結果表與對應 metrics source annotation。
- Confidence interval、single-seed 限制、real-image exposure confound。
- artifact gate 未通過、filtered/unfiltered 結論、Full-real arm 不適用的理由。
- Dataset annotation caveat、逐圖座標映射、Validation/Test leakage guards。
- Demo 的資料來源與用途限制。

可縮減重複敘述，但不得刪除 verifier 所要求的 disclosure。詳細的失敗路線與長篇工程
歷程改用精簡摘要並連回既有文件。

## 驗證策略

完成 README 後執行：

- README verifier，確認表格數字、必要 disclosure、相對連結與禁用 placeholder。
- Mermaid extraction 與 syntax validation；每張圖都必須成功解析。
- Markdown diff 檢查，確認沒有 trailing whitespace 或破損表格。
- README 相關測試與完整 test suite。
- forbidden-licence scanner 與 repository status 檢查。

## 驗收條件

- README 正文以正體中文為主，開頭含精簡 English Abstract。
- 沒有非必要 emoji。
- 三張 Mermaid 分別清楚呈現整體 pipeline、四組 ablation 與 evaluation protocol。
- 所有既有可驗證數字仍能通過 README verifier。
- GitHub、Hugging Face、repository 內部連結均有效。
- 負面結果、統計不確定性與研究限制沒有被行銷式語句淡化。
- 安裝、Demo 與重現入口可由新讀者直接找到。
