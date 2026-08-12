---
name: SafeSynth
description: 以證據優先的方式呈現安全帽偵測與可重現研究結果
colors:
  action-yellow: "#e1b45b"
  action-yellow-deep: "#bd8730"
  compliant-sage: "#7f9d8a"
  compliant-soft: "#dfe8df"
  noncompliant-coral: "#c37d72"
  noncompliant-soft: "#f0deda"
  neutral-slate: "#7f919a"
  canvas: "#e7e3da"
  paper: "#f7f4ed"
  paper-deep: "#ede8de"
  ink: "#17201e"
  muted: "#56625e"
  graphite: "#151b19"
typography:
  display:
    fontFamily: "Microsoft JhengHei, PingFang TC, Noto Sans TC, system-ui, sans-serif"
    fontSize: "clamp(38px, 4.2vw, 66px)"
    fontWeight: 800
    lineHeight: 1.08
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Microsoft JhengHei, PingFang TC, Noto Sans TC, system-ui, sans-serif"
    fontSize: "clamp(27px, 2.4vw, 39px)"
    fontWeight: 800
    lineHeight: 1.25
    letterSpacing: "-0.03em"
  body:
    fontFamily: "Microsoft JhengHei, PingFang TC, Noto Sans TC, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 400
    lineHeight: 1.65
  label:
    fontFamily: "Microsoft JhengHei, PingFang TC, Noto Sans TC, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 700
    lineHeight: 1.45
rounded:
  sm: "10px"
  md: "16px"
spacing:
  xs: "8px"
  sm: "14px"
  md: "24px"
  lg: "46px"
components:
  button-primary:
    backgroundColor: "{colors.action-yellow}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "14px 20px"
    height: "54px"
  button-secondary:
    backgroundColor: "{colors.paper-deep}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "12px 20px"
    height: "50px"
  evidence-stage:
    backgroundColor: "{colors.graphite}"
    textColor: "{colors.paper}"
    rounded: "{rounded.md}"
    padding: "28px"
---

# Design System: SafeSynth

## Overview

**Creative North Star: "The Evidence Desk"**

SafeSynth 的視覺系統像一張整理完畢的工地檢查桌：影像證據置於深色工作區，判定與
實測條件放在溫暖紙張上，任何研究主張都必須緊鄰可檢查的依據。語氣理性、透明且
克制；設計感來自清楚的層級、材料對比與精準配色，不依靠裝飾性特效。

介面以正體中文為主，RT-DETRv2、Validation、threshold、checkpoint 等專有名詞保留
原文。使用者先看 Before／After 與判定，再按需要展開研究方法與限制。

**Key Characteristics:**

- 證據優先，說明緊鄰可驗證的輸出。
- 溫暖 Morandi 紙面搭配 Graphite 影像工作區。
- Safety Yellow 只用於主要操作；Sage、Dusty Coral、Slate 只承擔語義。
- 正文 18px、輔助資訊至少 16px，Desktop 與 Mobile 都不以縮字換取空間。
- 不使用 emoji、裝飾性 gradient、glassmorphism 或無意義卡片層級。

## Colors

配色以低彩度工業材料為基底，唯一較鮮明的 Safety Yellow 用來指出使用者可以採取的
下一步。

### Primary

- **Safety Yellow**：主要 Upload action 與 keyboard focus，稀少使用才能維持方向感。
- **Deep Safety Yellow**：連結 underline 與 focus 等需要更高對比的狀態。

### Secondary

- **Compliant Sage**：已佩戴狀態、ready indicator 與其低彩度底色。
- **Dusty Coral**：未佩戴、錯誤與需要復原的狀態。
- **Neutral Slate**：`person` 定位框；不得暗示 compliance verdict。

### Neutral

- **Warm Canvas / Paper / Deep Paper**：頁面、主要內容面與資料帶的三層材料。
- **Inspection Ink / Muted Ink**：主文與輔助說明；輔助文字仍保持可讀對比。
- **Graphite Stage**：承載影像，讓框線與 Before／After 中線成為視覺中心。

**The Semantic Color Rule.** Sage、Coral、Slate 一律搭配文字標籤，不允許只靠顏色傳達
判定。

## Typography

**Display Font:** Microsoft JhengHei，依序 fallback 至 PingFang TC、Noto Sans TC、
system-ui。

**Body Font:** 與 Display 共用字族，以 weight、字級、行距建立層級，避免額外 network
font 使研究 demo 的啟動依賴外部服務。

**Character:** 直率、穩定、具有現場標示般的清晰度；數值採 tabular figures，技術詞
保留原文但不以 monospace 裝飾。

### Hierarchy

- **Display**（800、responsive clamp、1.08）：只用於主張「先看證據，再讀結論」。
- **Headline**（800、responsive clamp、1.25）：判定句與影片完成狀態。
- **Body**（400、18px、1.65）：所有主要說明；段落控制在約 65ch。
- **Label**（700、16px、1.45）：圖例、狀態、metadata 與控制文字的最低尺寸。

**The Readable Floor Rule.** 有語義的文字不得低於 16px；Mobile 只重排，不縮小正文。

## Layout

主容器最大 1480px。Desktop 第一層先以左右 grid 對齊論點與短說明；主要 evidence hero
使用 8:4 比例，左側是影像工作區，右側是判定與操作。四欄 execution evidence 緊接於
hero 下方，不另建浮動 cards。

1020px 以下 hero 改成單欄，evidence 轉為兩欄；720px 以下所有核心區域轉為單欄，
頁邊距為 12px，按鈕維持至少 46px 高。所有 breakpoint 都必須通過無水平溢位檢查。

## Elevation & Depth

大部分區域依靠 Graphite／Paper／Deep Paper 的 tonal layering 建立深度。只有主要 evidence
hero 與影片 workspace 使用一個寬且柔和、帶垂直位移的 ambient shadow；內容內部不疊加
小卡陰影。

- **Evidence Ambient** (`0 18px 46px -28px rgba(23, 32, 30, 0.52)`)：大型互動工作區。
- **Action Lift** (`0 10px 24px -16px rgba(95, 59, 10, 0.75)`)：主要 Upload action。

**The One Elevation Rule.** 一個區域以 shadow 或 border 表示邊界，避免兩者同時堆疊。

## Shapes

主要工作區採 16px 柔和圓角，按鈕、數值區與影像內框採 10px。Pill 只保留給短小狀態
與來源標籤。Compliance 圖例維持矩形框線，直接對應影像 bounding box 的語言。

## Components

### Buttons

- **Primary:** Safety Yellow、Inspection Ink、54px 高、10px 圓角；全寬置於判定之後。
- **Secondary:** Deep Paper、50px 高；視覺權重明顯低於 Upload action。
- **Hover / Focus:** Hover 只提高黃色明度；keyboard focus 使用 3px Deep Safety Yellow
  outline 與 3px offset。

### Status Chips

- **Source:** Graphite stage 上使用深灰底與淺字，Paper 上使用 Sage Soft 與深綠字。
- **Ready:** 小圓點加 `CPU ready` 文字，不能只顯示狀態點。

### Evidence Stage

Graphite 面承載原始與標註影像，ImageSlider 固定以文字列說明 Before、After 與拖曳方式。
判定面使用普通語句、人數、合規率、count blocks 與三列圖例；它是單一摘要區而不是多層
dashboard cards。

### Evidence Strip

四個欄位依序顯示 model-only、end-to-end、checkpoint 與 operating conditions。數值 18px，
說明 16px；窄螢幕逐步改成兩欄與一欄。

### Navigation

主導覽是兩個文字 tabs。Active tab 以 Deep Safety Yellow 3px underline 表示；GitHub、
Dataset、Model 是頂列固定資源連結，Mobile 允許換行但不得隱藏。

## Do's and Don'ts

### Do:

- **Do** 讓判定、影像與 execution conditions 在同一閱讀序列中可見。
- **Do** 同時使用色彩、文字與數值表達 compliance state。
- **Do** 保留負面結果、低 confidence calibration 與 dataset 限制。
- **Do** 以實測內容填入 latency、checkpoint、resolution、dtype 與 device。

### Don't:

- **Don't** 用 `person` 框推導合規率，或把孤立安全帽當成佩戴事件。
- **Don't** 以小於 16px 的文字塞入更多 metadata。
- **Don't** 使用 emoji、裝飾性 gradients、glass 或重複小卡製造科技感。
- **Don't** 為 demo 個別影像臨時調整 frozen threshold。
