"""Presentation contracts for the evidence-first Gradio demo."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from src.inference.demo import FrameSummary
from src.inference.demo_ui import (
    DEFAULT_EXAMPLE_SOURCE,
    format_error_html,
    format_evidence_html,
    format_source_html,
    format_summary_html,
    format_video_summary_html,
    load_example_image,
)


class _Processor:
    size: ClassVar = {"width": 640, "height": 640}


class _DetectorMetadata:
    processor = _Processor()
    device = "cpu"
    dtype = "float32"
    checkpoint = Path("safesynth-rtdetrv2-r18")


class _FakeDetector(_DetectorMetadata):
    def __call__(self, _image):
        return (
            [
                {"category_id": 0, "score": 0.9, "bbox": [5, 5, 15, 15]},
                {"category_id": 1, "score": 0.8, "bbox": [30, 5, 15, 15]},
                {"category_id": 2, "score": 0.7, "bbox": [10, 30, 30, 25]},
            ],
            191.4,
            198.2,
        )


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
    assert loaded.flags.writeable


def test_curated_example_records_its_released_dataset_source() -> None:
    assert DEFAULT_EXAMPLE_SOURCE == "images/hard_hat_workers863.png"


def test_present_image_returns_before_after_and_real_metrics() -> None:
    import app

    image = np.zeros((64, 64, 3), dtype=np.uint8)

    result = app.present_image(_FakeDetector(), image, 0.07, source_label="使用者影像")

    assert result.comparison[0] is image
    assert result.comparison[1].shape == image.shape
    assert "使用者影像" in result.summary_html
    assert "1 位正確佩戴" in result.summary_html
    assert "198 ms" in result.evidence_html
    assert result.error_html == ""


def test_present_image_preserves_the_input_when_inference_fails() -> None:
    import app

    class _BrokenDetector(_DetectorMetadata):
        def __call__(self, _image):
            raise RuntimeError("model failed")

    image = np.zeros((32, 48, 3), dtype=np.uint8)
    result = app.present_image(_BrokenDetector(), image, 0.07, source_label="使用者影像")

    assert result.comparison[0] is image
    assert result.comparison[1] is image
    assert "分析失敗" in result.error_html
    assert "model failed" not in result.error_html


def test_uploaded_image_loader_rejects_corrupt_files(tmp_path) -> None:
    import app

    invalid = tmp_path / "broken.png"
    invalid.write_bytes(b"not an image")

    with pytest.raises(app.DemoInputError, match="無法讀取這個影像"):
        app.load_uploaded_image(invalid)


def test_build_interface_exposes_the_evidence_first_surface() -> None:
    import app

    source = Path(app.__file__).read_text(encoding="utf-8")

    assert "gr.ImageSlider(" in source
    assert "gr.UploadButton(" in source
    assert "gr.Textbox(" not in source
    assert 'gr.Tab("圖片偵測")' in source
    assert 'gr.Tab("影片偵測")' in source
    assert '"研究方法與限制"' in source


def test_demo_css_pins_readable_type_and_responsive_evidence_layout() -> None:
    css = (Path(__file__).parents[1] / "assets" / "demo_ui.css").read_text(
        encoding="utf-8"
    )

    assert "font-size: 18px" in css
    assert "font-size: 16px" in css
    assert ".ss-disclosure button span" in css
    assert "@media (max-width: 720px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css


def test_video_summary_is_localized_and_reports_truncation() -> None:
    html = format_video_summary_html(
        n_frames=120,
        truncated=True,
        mean_compliance_rate=0.625,
    )

    assert "影片分析完成" in html
    assert "120 frames" in html
    assert "62%" in html
    assert "已達處理上限" in html


def test_video_summary_does_not_invent_a_zero_rate() -> None:
    html = format_video_summary_html(
        n_frames=4,
        truncated=False,
        mean_compliance_rate=None,
    )

    assert "不適用" in html
    assert "0%" not in html
