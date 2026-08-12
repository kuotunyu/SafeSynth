<div align="center">

# SafeSynth

**Targeted Synthetic Data 對 Hard-Hat Detection 真的有效嗎？**

以 frozen split、four-arm controlled ablation、artifact gate 與 image-level bootstrap，
把「看起來更像資料」轉成可重現、可否證的實驗。

[![CI](https://github.com/kuotunyu/SafeSynth/actions/workflows/ci.yml/badge.svg)](https://github.com/kuotunyu/SafeSynth/actions/workflows/ci.yml)

[Dataset](https://huggingface.co/datasets/steven0226/safesynth-hard-hat) ·
[Model](https://huggingface.co/steven0226/safesynth-rtdetrv2-r18) ·
[Release](https://github.com/kuotunyu/SafeSynth/releases/tag/v1.0.0) ·
[Experiment Protocol](docs/experiment_protocol.md)

</div>

> **結論先講：** 在 RT-DETRv2-R18 上，加入 Synthetic Data 反而降低
> `primary_map_small`；RF-DETR-Nano 的 point estimate 方向相反，但 confidence
> intervals 重疊。SafeSynth 因此不是一份「Synthetic Data 一定有效」的宣傳，
> 而是一套能辨識失敗、保存負結果、公開證據的研究流程。

---

## 核心發現與一眼看懂

- **RT-DETRv2-R18：** `real_only` 取得最高 `primary_map_small = 0.4511`；最佳
  synthetic arm 為 `unfiltered_syn = 0.3759`。
- **RF-DETR-Nano：** `filtered_syn = 0.5030` 高於 `real_only = 0.4841`，但 95%
  bootstrap intervals 仍重疊，不能宣稱穩健提升。
- **Artifact gate：** pre-registered H4 上限是 AUC 0.60，實測 **AUC 0.9053**；
  H4 **did not pass**，顯示 pasted 與 real patches 存在明顯可分訊號。
- **研究立場：** All claims are relative; never absolute AP。所有主張只比較同一個
  frozen Test 上的 arm-to-arm 差異。

![SafeSynth 主要實驗結果](reports/figures/headline.png)

---

## 系統架構與 Pipeline

### 1. 從資料到結論端到端流程

SafeSynth 不追求無限制 bulk generation；它先定義實際 failure modes，再生成、過濾、
訓練與評估。Validation 負責選 checkpoint 與 operating point，Test 僅在最後執行一次。

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph DataStage ["階段一：凍結資料基礎 (Frozen Data Foundation)"]
        direction LR
        Raw[("Hard Hat Workers 原生資料集<br/>(PASCAL VOC 5,000 張)")] --> Split["pHash 雜湊去重群組劃分<br/>(防禦跨切分洩漏)"] --> Set[("凍結資料切分<br/>(Train · Val · Test)")]
    end

    subgraph SynthStage ["階段二：定向合成與幾何過濾 (Targeted Synthesis)"]
        direction LR
        Fail["四大 Failure Modes 定義<br/>(微小 · 遮蔽 · 密集 · 低光)"] --> SAM["SAM 2.1 高精度摳圖<br/>(情境驅動定向合成)"] --> Filter["幾何檢查與品質過濾器<br/>(產出結構化 Provenance)"]
    end

    subgraph ExpStage ["階段三：四組受控消融實驗 (Controlled Experiment)"]
        direction LR
        Arms[("四組訓練分組 (Four Arms)<br/>(相同 10,900 Step 預算)")] --> Val["Validation 門禁選取<br/>(Checkpoint · Operating Point)"] --> Test[("Frozen Real Test 評測<br/>(Image-level Bootstrap 檢定)")]
    end

    subgraph PubStage ["階段四：客觀證據發布 (Public Evidence)"]
        direction LR
        Test --> Evidence(["指標與負結果公開報告<br/>(GitHub · Hugging Face Hub)"])
    end

    DataStage --> SynthStage --> ExpStage --> PubStage

    classDef srcStyle fill:#e7f5ff,stroke:#1971c2,stroke-width:2px,color:#212529
    classDef procStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#212529
    classDef evalStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class Raw,Set,Arms srcStyle
    class Split,Fail,SAM,Filter,Val,Test procStyle
    class Evidence evalStyle

    style DataStage fill:#f8f9fa,stroke:#1971c2,stroke-width:2px,color:#1971c2,stroke-dasharray: 4 4
    style SynthStage fill:#faf5ff,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2,stroke-dasharray: 4 4
    style ExpStage fill:#fffcf0,stroke:#f59f00,stroke-width:2px,color:#f59f00,stroke-dasharray: 4 4
    style PubStage fill:#f4fbf7,stroke:#0ca678,stroke-width:2px,color:#0ca678,stroke-dasharray: 4 4
```

### 2. 四組受控消融架構 (Four-arm Controlled Ablation)

四組實驗共用同一個 split、base model、seed、optimizer-step budget 與 evaluation code。
Synthetic arms 與 `real_only` 的差別只在 training stream，避免把更多訓練步數誤認成
Synthetic Data 的效果。

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    subgraph InputStage ["階段一：訓練資料串流 (Training Streams)"]
        direction LR
        R[("真實資料 Real Train<br/>(3,500 張影像)")]
        U[("未過濾合成 Unfiltered<br/>(3,500 張影像)")]
        F[("過濾合成 Filtered<br/>(3,500 張影像)")]
    end

    subgraph ArmStage ["階段二：四組消融對照組 (Four Training Arms)"]
        direction LR
        A1["1. real_only<br/>(純真實樣本)"]
        A2["2. standard_aug<br/>(標準影像增強)"]
        A3["3. unfiltered_syn<br/>(真實 + 未過濾合成)"]
        A4["4. filtered_syn<br/>(真實 + 過濾後合成)"]
    end

    subgraph EvalStage ["階段三：等步數預算與嚴格評測 (Budget & Evaluation)"]
        direction LR
        Budget["相同 10,900 Optimizer Steps<br/>(消除訓練步數 Confound)"] --> Val["Real Validation 驗證集<br/>(選取最佳 Checkpoint 與門檻)"] --> Test[("Frozen Real Test 測試集<br/>(744 張圖獨立客觀檢驗)")]
    end

    R --> A1 & A2 & A3 & A4
    U --> A3
    F --> A4
    A1 & A2 & A3 & A4 --> Budget

    classDef srcStyle fill:#e7f5ff,stroke:#1971c2,stroke-width:2px,color:#212529
    classDef armStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#212529
    classDef evalStyle fill:#e6fcf5,stroke:#0ca678,stroke-width:2px,color:#212529

    class R,U,F,Test srcStyle
    class A1,A2,A3,A4 armStyle
    class Budget,Val evalStyle

    style InputStage fill:#f8f9fa,stroke:#1971c2,stroke-width:2px,color:#1971c2,stroke-dasharray: 4 4
    style ArmStage fill:#faf5ff,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2,stroke-dasharray: 4 4
    style EvalStage fill:#f4fbf7,stroke:#0ca678,stroke-width:2px,color:#0ca678,stroke-dasharray: 4 4
```

本研究採 **four-arm** 設計，而不是 five-arm。**第五組 Full-real 不適用**，因為
`real_only` 已使用 **all real Train data**，不存在更高的 real-data ceiling。

---

## 主要結果

數值可由 [`results/detection_metrics.csv`](results/detection_metrics.csv) 重新聚合；CI 會
執行 `scripts.verify_readme`，避免 README 與實驗輸出分離。

| Arm | primary_map_small <!--split: test--> | primary_map <!--split: test--> | bare_head_recall <!--split: test--> | real-image exposures |
|---|---:|---:|---:|---:|
| `real_only` | 0.4511 | 0.5341 | 0.9875 | 49.83 |
| `standard_aug` | 0.4236 | 0.4958 | 0.9875 | 49.83 |
| `unfiltered_syn` | 0.3759 | 0.4597 | 0.9898 | 24.91 |
| `filtered_syn` | 0.3664 | 0.4858 | 0.9886 | 24.91 |

三個重點：

1. `real_only` 在 RT-DETRv2-R18 的主要 detection metrics 上勝出。
2. Synthetic arms 的 real-image exposures 只有一半；這是一個重要的 data exposure
   confound，不應把差距全部歸因於影像品質。
3. `filtered_syn` 達成 compliance precision 的 deployment constraint，但沒有換到更高 AP。

<details>
<summary><strong>RF-DETR-Nano replication</strong></summary>

相同 four-arm protocol 換成 RF-DETR-Nano 後，Synthetic Data 的 point estimate 轉為正向；
然而 interval 仍重疊，所以這裡只報告「model-dependent signal」，不報告穩健 win。

<!--metrics-source: rfdetr_detection_metrics.csv-->
| Arm | primary_map_small <!--split: test--> | primary_map <!--split: test--> | bare_head_recall <!--split: test--> | real_image_exposures <!--split: test--> |
|---|---:|---:|---:|---:|
| `real_only` | 0.4841 [0.4653, 0.5048] | 0.5657 | 0.9761 [0.9643, 0.9863] | 49.83 |
| `standard_aug` | 0.4970 [0.4727, 0.5219] | 0.5789 | 0.9681 [0.9539, 0.9809] | 49.83 |
| `unfiltered_syn` | 0.4959 [0.4747, 0.5194] | 0.5774 | 0.9750 [0.9596, 0.9865] | 24.91 |
| `filtered_syn` | 0.5030 [0.4841, 0.5240] | 0.5818 | 0.9863 [0.9777, 0.9938] | 24.91 |

</details>

---

## 為什麼負結果仍然重要

- **H4 提前指出 domain gap。** AUC 0.9053 表示 real 與 pasted patches 有明顯可分訊號；
  依 [ADR-011](docs/decisions.md#adr-011)，專案停止擴增到 2x，保留 1x 實驗結果。
- **Filter 不是免費午餐。** 它改善特定 deployment constraint，卻沒有保證提升 detector AP。
- **Model choice 會改變方向。** RT-DETRv2 與 RF-DETR-Nano 的結果不同，提醒我們不能從
  single architecture 推廣成普遍結論。

---

## 互動式 Demo 展示

Demo 採 evidence-first 介面：先顯示影像與偵測框，再呈現 compliance verdict、counts、
confidence、latency、checkpoint 與 runtime。沒有裝飾性 dashboard，也不隱藏模型限制。

![SafeSynth evidence-first demo](assets/demo/demo_ui_desktop.png)

<details>
<summary><strong>Validation montage</strong></summary>

![SafeSynth validation montage](assets/demo.gif)

Montage 使用 Validation 影像，不使用 Test。黃色為 helmeted head，紅色為 bare head；
caption 顯示 `compliant / total` 與 compliance rate。

</details>

---

## 證據與資料邊界

- 原始資料是 [Hard Hat Workers](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
  （CC0 1.0），共 5,000 張影像。
- [SHEL5K（Sensors 2022）](https://www.mdpi.com/1424-8220/22/6/2315) 對相同影像重新
  標註後得到 **75,570 labels**，原始版本只有 **25,502**；因此 `person` 不承擔主要結論。
- Synthetic Dataset 公開 filtered / unfiltered annotations、image SHA-256 與
  `records.jsonl` provenance；它們不是 Validation 或 Test ground truth。
- Split 以 pHash groups 凍結，並以 SHA-256 manifest 驗證；generator 與 filter 不讀取
  Validation / Test labels。

<details>
<summary><strong>已知限制</strong></summary>

- 原始 annotation 不完整，因此所有數值只適合做同一 frozen Test 上的相對比較。
- Copy-paste 在有限 backgrounds 上會飽和，且 H4 已證實殘留 artifact signal。
- 主實驗與 replication 都是 single-seed training；bootstrap 衡量 Test image sampling
  uncertainty，不等同 run-to-run variance。
- 公開 release 不主張 RF-DETR latency；固定時脈 benchmark 未通過 host-contention p95 gate。

</details>

---

## 快速開始與重現步驟

### 環境需求

- Windows 11 (原生環境；不使用 WSL)
- Python 3.12 與 [uv](https://docs.astral.sh/uv/)
- CPU 用於驗證與 Demo 展示；CUDA 可選用於訓練或快速推論

### 複製專案與驗證門禁

```powershell
git clone https://github.com/kuotunyu/SafeSynth.git
Set-Location SafeSynth
uv sync --locked

uv run ruff check .
uv run pytest -q
uv run python -m scripts.verify_readme
uv run python -m scripts.check_forbidden_licences
```

### 使用公開權重執行 Demo

```powershell
uvx hf download steven0226/safesynth-rtdetrv2-r18 `
  --local-dir models/safesynth-rtdetrv2-r18

uv run python app.py `
  --device cpu `
  --weights models/safesynth-rtdetrv2-r18
```

伺服器啟動後於瀏覽器開啟 `http://127.0.0.1:7860`。若有相容 NVIDIA GPU，可將 `--device cpu` 換為 `--device cuda`。

### 實驗流程規範

大檔影像與權重依規範存放於 Git 外部。於 [`configs/paths.yaml`](configs/paths.yaml) 設定資料路徑後，依循凍結規範執行：

- [Data protocol](docs/data_protocol.md)
- [Synthesis specification](docs/synthesis_spec.md)
- [Filtering specification](docs/filtering_spec.md)
- [Training specification](docs/training_spec.md)
- [Evaluation specification](docs/evaluation_spec.md)
- [Environment and CUDA notes](docs/environment.md)

公開發布產物可獨立驗證：

```powershell
uv run python -m scripts.verify_hf_release `
  --dataset <dataset-bundle> `
  --model <model-bundle>
```

---

## 專案結構

| 目錄 / 檔案 | 內容職責規範 |
|---|---|
| `src/` | 資料、合成、訓練、推論、評測與發布核心代碼 |
| `configs/` | 凍結實驗組態與超參數設定 |
| `scripts/` | 可重現命令列進入點與驗證工具 |
| `tests/` | 單元測試、合約測試、證據與回歸測試 |
| `results/` | 機器可讀輕量指標（供 README 驗證對齊） |
| `reports/` | 科學報告與精選證據圖表 |
| `publishing/` | 發布說明與 Hugging Face Model Card |

---

## 授權與聲明

原始程式碼採 [MIT License](LICENSE) 釋出。原生資料集採用 CC0 1.0；SAM 2.1 權重遵循 Apache-2.0。引用本研究時，請使用不可變之 [SafeSynth v1.0.0 release](https://github.com/kuotunyu/SafeSynth/releases/tag/v1.0.0) 與 [Hugging Face Dataset](https://huggingface.co/datasets/steven0226/safesynth-hard-hat)。
