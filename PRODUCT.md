# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

主要使用者是從 GitHub 進入專案、正在快速判斷研究品質與工程完成度的訪客。他們需要在數秒內理解 SafeSynth 能做什麼、看到真實推論結果，並能進一步驗證公開的資料、模型與研究結論。

## Product Purpose

SafeSynth 是 hard-hat detection 的 controlled synthetic-data ablation project。Live demo 讓訪客上傳圖片或短片，查看每個頭部的安全帽佩戴判定、frame-level compliance summary 與實際推論條件。成功不是只讓畫面看起來漂亮，而是讓操作、研究證據與限制同時可信。

## Positioning

SafeSynth 同時公開 dataset、model weights、frozen evaluation protocol 與 negative result。它不以 synthetic data 必然有效為前提；四組 ablation 的主要結果顯示 synthetic arms 沒有穩健超越 real-only，demo 因此誠實使用表現最佳的 `real_only` checkpoint。

## Operating Context

- 訪客從 GitHub README 進入專案，在本機啟動 Gradio web demo。
- Image 是主要體驗；Video 支援短片並最多處理前 120 frames。
- Demo 預設使用 CPU，避免與其他 GPU 工作競爭；CLI 仍可選擇 CUDA。
- 使用者可查看 GitHub、Hugging Face Dataset 與 Hugging Face Model 的公開證據。

## Capabilities and Constraints

- 模型：`RT-DETRv2-R18`，公開的 `real_only` checkpoint。
- 類別：`helmet`、`head`、`person`；`person` 只定位，不進入 compliance verdict。
- Operating point：score `threshold=0.07`，由 Validation 選定並凍結。
- Image 輸出 Before／After、已佩戴與未佩戴人數、合規率及 performance metadata。
- 沒有可判定頭部時必須顯示不可判定狀態，不得顯示誤導性的 `0%`。
- 不得改寫模型、threshold、研究結果或公開效能主張來配合 UI。

## Brand Commitments

- 產品名稱固定為 SafeSynth。
- 正體中文（`zh-TW`）為主要語言；專有名詞與不自然的翻譯保留原文。
- 介面必須讓 GitHub 訪客第一眼感到專業、有設計感與研究可信度，但不得花俏或像 marketing hype。
- 字級舒適，不使用過小文字；減少無效空白、重複層級與低價值資訊。
- 低飽和 Morandi palette 搭配 safety yellow、sage green、dusty coral 與 graphite。

## Evidence on Hand

- 公開 repository：<https://github.com/kuotunyu/SafeSynth>
- 公開 dataset：<https://huggingface.co/datasets/steven0226/safesynth-hard-hat>
- 公開 model：<https://huggingface.co/steven0226/safesynth-rtdetrv2-r18>
- README demo animation：`assets/demo.gif`
- Evaluation config：`configs/evaluation.yaml`
- Demo rendering and summaries：`src/inference/demo.py`
- Existing executable surface：`app.py`
- 沒有客戶 testimonial、商業成效或現場部署證據；未來介面不得捏造。

## Product Principles

1. 證據先說話：第一個 viewport 直接展示真實推論，不先堆研究說明。
2. 誠實比宣傳重要：negative result、calibration 與資料限制必須可找到且表述準確。
3. 一個主要動作：訪客應立即知道如何上傳自己的影像。
4. Progressive disclosure：主要結果保持清楚，technical metadata 與限制在需要時展開。
5. 公開且可重現：畫面中的模型、threshold 與效能資訊必須對應實際執行條件。

## Accessibility & Inclusion

- 正文以 18 px 為基準，所有輔助資訊至少 16 px。
- 合規狀態不能只靠紅綠色；必須同時提供文字、符號或框線差異。
- 主要操作必須可用 keyboard 完成，並具清楚 focus state。
- Mobile touch target 至少 44 px；Desktop 與 Mobile 都不得水平溢出。
