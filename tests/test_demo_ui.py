"""Presentation contracts for the evidence-first Gradio demo."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from src.inference.demo import FrameSummary
from src.inference.demo_ui import (
    DEFAULT_EXAMPLE_SOURCE,
    format_error_html,
    format_evidence_html,
    format_source_html,
    format_summary_html,
    load_example_image,
)


class _Processor:
    size: ClassVar = {"width": 640, "height": 640}


class _DetectorMetadata:
    processor = _Processor()
    device = "cpu"
    dtype = "float32"
    checkpoint = Path("safesynth-rtdetrv2-r18")


def test_summary_html_leads_with_plain_language_counts() -> None:
    html = format_summary_html(FrameSummary(4, 3, 1), source_label="精選範例")

    assert "7 位人員中" in html
    assert "4 位正確佩戴安全帽" in html
    assert "57" in html
    assert "精選範例" in html
    assert "已佩戴" in html
    assert "未佩戴" in html


def test_indeterminate_summary_never_claims_zero_percent() -> None:
    html = format_summary_html(FrameSummary(0, 0, 2), source_label="使用者影像")

    assert "未找到可判定" in html
    assert "0%" not in html
    assert "—" in html


def test_dynamic_copy_is_html_escaped() -> None:
    label = '<script>alert("unsafe")</script>'

    summary = format_summary_html(FrameSummary(1, 0, 0), source_label=label)
    source = format_source_html(label)
    error = format_error_html(label)

    assert "<script>" not in summary + source + error
    assert "&lt;script&gt;" in summary
    assert "&lt;script&gt;" in source
    assert "&lt;script&gt;" in error


def test_evidence_html_reports_real_execution_conditions() -> None:
    html = format_evidence_html(191.4, 198.2, _DetectorMetadata(), threshold=0.07)

    assert "191 ms" in html
    assert "198 ms" in html
    assert "5.0 FPS" in html
    assert "640×640" in html
    assert "cpu" in html
    assert "safesynth-rtdetrv2-r18" in html
    assert "0.07" in html


def test_example_loader_applies_rgb_conversion(tmp_path) -> None:
    source = tmp_path / "example.png"
    Image.new("L", (9, 7), color=123).save(source)

    loaded = load_example_image(source)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (7, 9, 3)
    assert loaded.dtype == np.uint8


def test_curated_example_records_its_released_dataset_source() -> None:
    assert DEFAULT_EXAMPLE_SOURCE == "images/hard_hat_workers863.png"
