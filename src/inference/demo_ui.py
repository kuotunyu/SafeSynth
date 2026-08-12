"""Testable view models and HTML for the evidence-first Gradio demo."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.inference.demo import FrameSummary

DEFAULT_EXAMPLE_SOURCE = "images/hard_hat_workers863.png"


@dataclass(frozen=True)
class ImagePresentation:
    """The four independent UI regions updated by one image inference."""

    comparison: tuple[np.ndarray, np.ndarray]
    summary_html: str
    evidence_html: str
    source_html: str
    error_html: str = ""


def format_source_html(source_label: str) -> str:
    """A compact source indicator that never trusts caller-provided markup."""

    return f'<span class="ss-source-pill">{escape(source_label)}</span>'


def format_summary_html(summary: FrameSummary, *, source_label: str) -> str:
    """Render the verdict in plain zh-TW without inventing a zero rate."""

    source = escape(source_label)
    if summary.compliance_rate is None:
        return f"""
<section class="ss-summary" aria-live="polite">
  <div class="ss-result-status"><span aria-hidden="true"></span>推論完成 · {source}</div>
  <h2>未找到可判定的<br>安全帽佩戴狀態</h2>
  <p class="ss-summary-copy">你可以換一張人物頭部更清楚的影像。</p>
  <div class="ss-rate"><strong>—</strong><span>合規率<br>不適用</span></div>
  <div class="ss-count-grid ss-count-grid-single">
    <div class="ss-count ss-count-neutral"><b>{summary.n_neutral}</b><span>僅定位 person</span></div>
  </div>
  {_legend_html()}
</section>
""".strip()

    rate = round(summary.compliance_rate * 100)
    return f"""
<section class="ss-summary" aria-live="polite">
  <div class="ss-result-status"><span aria-hidden="true"></span>推論完成 · {source}</div>
  <h2>{summary.n_people} 位人員中，<br>{summary.n_compliant} 位正確佩戴安全帽</h2>
  <p class="ss-summary-copy">依照固定的 Validation operating point 判定，不因 demo 影像臨時調整 threshold。</p>
  <div class="ss-rate"><strong>{rate}</strong><span>%<br>合規率</span></div>
  <div class="ss-count-grid">
    <div class="ss-count ss-count-good"><b>{summary.n_compliant}</b><span>已佩戴</span></div>
    <div class="ss-count ss-count-bad"><b>{summary.n_non_compliant}</b><span>未佩戴</span></div>
  </div>
  {_legend_html()}
</section>
""".strip()


def _legend_html() -> str:
    return """
<div class="ss-legend" aria-label="偵測框圖例">
  <div><i class="ss-line ss-line-good" aria-hidden="true"></i>已佩戴：戴安全帽的頭部</div>
  <div><i class="ss-line ss-line-bad" aria-hidden="true"></i>未佩戴：未戴安全帽的頭部</div>
  <div><i class="ss-line ss-line-neutral" aria-hidden="true"></i>僅定位：person，不納入合規率</div>
</div>
""".strip()


def format_evidence_html(
    model_ms: float,
    e2e_ms: float,
    detector: Any,
    *,
    threshold: float,
) -> str:
    """Report the conditions a visitor needs to interpret latency honestly."""

    size = getattr(detector.processor, "size", {})
    resolution = f"{size.get('width', '?')}×{size.get('height', '?')}"
    dtype = str(detector.dtype).replace("torch.", "")
    device = str(detector.device)
    checkpoint = Path(detector.checkpoint).name
    fps = 1000.0 / e2e_ms if e2e_ms > 0 else 0.0
    return f"""
<section class="ss-evidence" aria-label="推論證據">
  <article><b>{model_ms:.0f} ms</b><span>model-only</span></article>
  <article><b>{e2e_ms:.0f} ms · {fps:.1f} FPS</b><span>end-to-end · batch 1</span></article>
  <article><b>{escape(checkpoint)}</b><span>RT-DETRv2-R18 · real_only</span></article>
  <article><b>threshold {threshold:.2f}</b><span>{escape(resolution)} · {escape(dtype)} · {escape(device)}</span></article>
</section>
""".strip()


def format_error_html(message: str) -> str:
    return f'<div class="ss-error" role="alert">{escape(message)}</div>'


def format_video_summary_html(
    *,
    n_frames: int,
    truncated: bool,
    mean_compliance_rate: float | None,
) -> str:
    """Render a clip result without implying that missing verdicts equal zero."""

    if mean_compliance_rate is None:
        rate = "不適用"
        rate_detail = "沒有可判定的安全帽佩戴狀態"
    else:
        rate = f"{round(mean_compliance_rate * 100)}%"
        rate_detail = "平均合規率"
    limit = (
        "<strong>已達處理上限</strong>，較長影片只分析前段。"
        if truncated
        else "已完成全部可解碼 frames。"
    )
    return f"""
<section class="ss-video-summary" aria-live="polite">
  <div>
    <h2>影片分析完成</h2>
    <p>已處理 <strong>{n_frames} frames</strong>；{limit}</p>
  </div>
  <div class="ss-video-rate"><strong>{rate}</strong><span>{rate_detail}</span></div>
</section>
""".strip()


def load_example_image(path: Path) -> np.ndarray:
    """Load a shipped example with camera orientation applied and RGB pinned."""

    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
