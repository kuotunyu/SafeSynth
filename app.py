"""Gradio demo: upload an image or a short clip, get compliance per person.

DEMO-01..DEMO-03. No webcam (a construction scene is not reproducible in front
of a laptop camera, and assuming the user has one is a bad default).

Runs on CPU by default. That is not a limitation being apologised for - the
measured end-to-end latency on this machine is 204 ms median per frame on CPU
(12 validation images, after the warm-up this file performs at startup), which
is usable for the single images and short clips this demo takes, and it means
the demo does not compete with whatever else is using the GPU. On CUDA the same
measurement is 20 ms.

Two honesty notes are surfaced in the UI rather than buried here, because a
reader who sees boxes appear will otherwise draw the wrong conclusion:

  The threshold is 0.07 and that is not a typo. Across 223,200 test detections
  this model's highest score is 0.2495; ranking is good, calibration is not, and
  the operating point was selected on Validation by EVAL-04.

  Synthetic data did NOT improve detection on this dataset. The demo ships the
  real_only weights because they are the best ones, and the project's own result
  says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.detection import load_evaluation_config
from src.inference.demo import draw_on, drawn_boxes, summarise
from src.inference.demo_ui import (
    ImagePresentation,
    format_error_html,
    format_evidence_html,
    format_source_html,
    format_summary_html,
    format_video_summary_html,
    load_example_image,
)
from src.training.ingest import latest_checkpoint
from src.training.metrics import predictions_to_coco

CLASS_NAMES = ("helmet", "head", "person")
PROCESSOR_ID = "PekingU/rtdetr_v2_r18vd"
DEFAULT_ARM = "real_only"
MAX_VIDEO_FRAMES = 120
DEFAULT_EXAMPLE_PATH = PROJECT_ROOT / "assets" / "demo" / "example.jpg"


class DemoStartupError(RuntimeError):
    """Raised when the demo cannot find weights to serve."""


class DemoInputError(ValueError):
    """Raised when a user-selected file is not a readable image."""


def resolve_weights(runs_root: Path, arm: str, seed: int) -> Path:
    """The checkpoint Validation selected (K-20: never the in-memory final model)."""

    seed_dir = Path(runs_root) / arm / f"seed_{seed}"
    newest = latest_checkpoint(seed_dir)
    if newest is None:
        raise DemoStartupError(
            f"no checkpoint under {seed_dir}. Unpack the Colab run first, or pass "
            f"--weights with a directory containing model.safetensors"
        )
    state = json.loads((newest / "trainer_state.json").read_text(encoding="utf-8"))
    recorded = state.get("best_model_checkpoint")
    if not recorded:
        return newest
    # Recorded on Colab as an absolute /content/... path; only the name is portable.
    resolved = seed_dir / Path(recorded).name
    return resolved if resolved.is_dir() else newest


class Detector:
    """Model plus processor, with the two timings DEMO-03 asks to separate."""

    def __init__(self, checkpoint: Path, device: str, dtype: torch.dtype) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(PROCESSOR_ID)
        self.model = (
            AutoModelForObjectDetection.from_pretrained(str(checkpoint), dtype=dtype)
            .eval()
            .to(device)
        )
        self.device = device
        self.dtype = dtype
        self.checkpoint = checkpoint

    def __call__(self, image: np.ndarray) -> tuple[list[dict], float, float]:
        """Detections plus (model_only_ms, end_to_end_ms).

        Both are measured because they answer different questions: the first is
        a property of the network, the second is what a user feels.
        """

        started = time.perf_counter()
        # dtype travels with the device. The processor always returns float32,
        # so a float16 model raises "Input type (torch.cuda.FloatTensor) and
        # weight type (torch.cuda.HalfTensor) should be the same". This went
        # unnoticed because the demo defaults to CPU, where dtype is float32
        # and the mismatch cannot arise - the CUDA path had never been run.
        encoded = self.processor(images=image, return_tensors="pt").to(
            device=self.device, dtype=self.dtype
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        model_started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**encoded)
        if self.device == "cuda":
            torch.cuda.synchronize()
        model_ms = (time.perf_counter() - model_started) * 1000.0

        height, width = image.shape[:2]
        # `outputs` is a ModelOutput dataclass, not a tensor, so it has no .to()
        # - the CUDA branch raised AttributeError the first time it ran. Post
        # processing accepts it where it already lives, given a target_sizes on
        # the same device.
        target = torch.tensor([[height, width]], dtype=torch.float32, device=self.device)
        processed = self.processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target
        )
        detections = predictions_to_coco(processed, [0])
        end_to_end_ms = (time.perf_counter() - started) * 1000.0
        return detections, model_ms, end_to_end_ms


def annotate(detector: Detector, image: np.ndarray, threshold: float):
    detections, model_ms, end_to_end_ms = detector(image)
    boxes = drawn_boxes(
        detections, class_names=CLASS_NAMES, score_threshold=threshold
    )
    return draw_on(image, boxes), summarise(boxes), model_ms, end_to_end_ms


def load_uploaded_image(path: str | Path) -> np.ndarray:
    """Load a browser upload with orientation and colour channels normalized."""

    try:
        with Image.open(path) as image:
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise DemoInputError("無法讀取這個影像。請使用 JPG、PNG 或 WEBP。") from error


def present_image(
    detector: Detector,
    image: np.ndarray,
    threshold: float,
    *,
    source_label: str,
) -> ImagePresentation:
    """Run inference and convert it to the four independently updated UI regions."""

    try:
        annotated, summary, model_ms, e2e_ms = annotate(detector, image, threshold)
    except Exception as error:  # noqa: BLE001 - UI boundary preserves the upload.
        print(f"image inference failed: {type(error).__name__}: {error}", file=sys.stderr)
        message = "分析失敗，請重新執行；若問題持續，請查看終端機紀錄。"
        return ImagePresentation(
            comparison=(image, image),
            summary_html=format_error_html(message),
            evidence_html="",
            source_html=format_source_html(source_label),
            error_html=format_error_html(message),
        )

    return ImagePresentation(
        comparison=(image, annotated),
        summary_html=format_summary_html(summary, source_label=source_label),
        evidence_html=format_evidence_html(
            model_ms, e2e_ms, detector, threshold=threshold
        ),
        source_html=format_source_html(source_label),
    )


def performance_line(model_ms: float, e2e_ms: float, detector: Detector) -> str:
    """DEMO-03: a latency number without batch, resolution and dtype is meaningless."""

    size = detector.processor.size
    resolution = f"{size.get('width', '?')}x{size.get('height', '?')}"
    peak = ""
    if detector.device == "cuda":
        peak = f"  ·  peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
    return (
        f"model-only {model_ms:.0f} ms  ·  end-to-end {e2e_ms:.0f} ms "
        f"({1000.0 / e2e_ms:.1f} FPS)  ·  batch 1, {resolution}, "
        f"{str(detector.dtype).replace('torch.', '')}, {detector.device}{peak}"
    )


@dataclass(frozen=True)
class VideoResult:
    """What the video tab produced, separated from how Gradio displays it."""

    path: Path
    note: str
    model_ms: float
    e2e_ms: float
    n_frames: int
    truncated: bool
    mean_compliance_rate: float | None


# spec: DEMO-02
def annotate_video(
    detector,
    source,
    threshold: float,
    destination: Path | None = None,
    *,
    progress_callback: Callable[[int, int | None], None] | None = None,
):
    """Annotate up to MAX_VIDEO_FRAMES of a clip; None if nothing decoded.

    Lifted out of the Gradio callback so it can be exercised without a browser.
    Everything else in this file that had never been run turned out to be
    broken - the CUDA branch raised twice on its first execution - and an
    untested path is not made safe by being short.
    """

    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 12.0
    raw_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    progress_total = (
        min(int(raw_total), MAX_VIDEO_FRAMES)
        if np.isfinite(raw_total) and raw_total > 0
        else None
    )
    frames, rates, model_times, e2e_times = [], [], [], []
    while len(frames) < MAX_VIDEO_FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated, summary, model_ms, e2e_ms = annotate(detector, rgb, threshold)
        frames.append(annotated)
        model_times.append(model_ms)
        e2e_times.append(e2e_ms)
        if summary.compliance_rate is not None:
            rates.append(summary.compliance_rate)
        if progress_callback is not None:
            progress_callback(len(frames), progress_total)
    capture.release()
    if not frames:
        return None

    out = (
        PROJECT_ROOT / "reports" / "figures" / "demo_annotated.mp4"
        if destination is None
        else Path(destination)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    truncated = len(frames) == MAX_VIDEO_FRAMES
    mean_compliance_rate = sum(rates) / len(rates) if rates else None
    mean_rate = f"{mean_compliance_rate:.2f}" if mean_compliance_rate is not None else "n/a"
    return VideoResult(
        path=out,
        note=(
            f"{len(frames)} frames{' (truncated)' if truncated else ''}"
            f"  ·  mean compliance_rate {mean_rate}"
        ),
        # Median, not mean: DEMO-03 says so, and one stalled frame moves a mean.
        model_ms=float(np.median(model_times)),
        e2e_ms=float(np.median(e2e_times)),
        n_frames=len(frames),
        truncated=truncated,
        mean_compliance_rate=mean_compliance_rate,
    )


def build_interface(detector: Detector, threshold: float):
    import gradio as gr

    default_image = load_example_image(DEFAULT_EXAMPLE_PATH)
    default_presentation = present_image(
        detector,
        default_image,
        threshold,
        source_label="精選範例",
    )

    def presentation_outputs(presentation: ImagePresentation):
        return (
            presentation.comparison,
            presentation.summary_html,
            presentation.evidence_html,
            presentation.source_html,
            presentation.error_html,
        )

    def on_image(path):
        if not path:
            return presentation_outputs(default_presentation)
        try:
            image = load_uploaded_image(path)
        except DemoInputError as error:
            return (
                default_presentation.comparison,
                default_presentation.summary_html,
                default_presentation.evidence_html,
                default_presentation.source_html,
                format_error_html(str(error)),
            )
        return presentation_outputs(
            present_image(detector, image, threshold, source_label="使用者影像")
        )

    def on_reset():
        return presentation_outputs(default_presentation)

    def on_video(path, progress=gr.Progress()):  # noqa: B008 - Gradio injection API.
        if not path:
            return None, "", ""
        try:
            result = annotate_video(
                detector,
                path,
                threshold,
                progress_callback=lambda current, total: progress(
                    (current, total),
                    desc="正在逐幀分析影片",
                    unit="frames",
                ),
            )
        except Exception as error:  # noqa: BLE001 - recover at the browser boundary.
            print(f"video inference failed: {type(error).__name__}: {error}", file=sys.stderr)
            message = "影片分析失敗。請確認檔案可播放，再重新執行。"
            return path, format_error_html(message), ""
        if result is None:
            message = "無法從這個影片讀取任何 frames。請改用一般 MP4 檔案再試一次。"
            return path, format_error_html(message), ""
        return str(result.path), format_video_summary_html(
            n_frames=result.n_frames,
            truncated=result.truncated,
            mean_compliance_rate=result.mean_compliance_rate,
        ), format_evidence_html(
            result.model_ms,
            result.e2e_ms,
            detector,
            threshold=threshold,
        )

    header = """
<!--
THESIS: Evidence before explanation; visitors see a real result before setup prose.
OWN-WORLD: A restrained field-inspection desk: warm paper, graphite evidence stage,
Morandi status colours, safety-yellow action, and measured technical typography.
STORY: Result -> inspect Before/After -> understand counts -> verify execution
conditions -> upload evidence -> open research limitations.
FIRST VIEWPORT: Product identity, image/video navigation, real comparison, plain-
language verdict, upload action, and latency evidence without scrolling.
FORM: User-pinned evidence stage; concept seed c7cf1f99.
FINISH: Unreviewed and undocumented is unfinished; this build ends with browser
verification and an explicit finish verdict.
-->
<header class="ss-product-bar">
  <div class="ss-brand">
    <span class="ss-brand-mark" aria-hidden="true">SS</span>
    <span><strong>SafeSynth</strong><small>Hard-hat compliance research demo</small></span>
  </div>
  <div class="ss-product-links">
    <span class="ss-ready"><i aria-hidden="true"></i>CPU ready</span>
    <a href="https://github.com/kuotunyu/SafeSynth" target="_blank" rel="noreferrer">GitHub</a>
    <a href="https://huggingface.co/datasets/steven0226/safesynth-hard-hat" target="_blank" rel="noreferrer">Dataset</a>
    <a href="https://huggingface.co/steven0226/safesynth-rtdetrv2-r18" target="_blank" rel="noreferrer">Model</a>
  </div>
</header>
<section class="ss-intro">
  <h1>先看偵測證據，<br>再讀研究結論。</h1>
  <p>比較原始影像與模型標註，直接檢查每一位人員的安全帽佩戴狀態。所有結果使用同一個 frozen operating point。</p>
</section>
"""

    research_copy = f"""
<div class="ss-disclosure-copy">
  <section>
    <h3>判定規則</h3>
    <p><strong>helmet</strong> 代表戴安全帽的頭部，標為「已佩戴」；<strong>head</strong> 代表未戴安全帽的頭部，標為「未佩戴」。<strong>person</strong> 只用於定位，不納入合規率。</p>
  </section>
  <section>
    <h3>固定 operating point</h3>
    <p><strong>threshold {threshold:.2f}</strong> 不是輸入錯誤。模型排名能力良好但 confidence calibration 偏低；此門檻依 Validation 的 0.80 compliance-precision floor 選定，之後固定不變。</p>
  </section>
  <section>
    <h3>研究結論與限制</h3>
    <p>四組 ablation 顯示 synthetic data 在這個 dataset 上<strong>沒有改善 detection</strong>。因此 demo 使用實驗中表現最佳的 <strong>real_only</strong> weights，並公開負面結果，而不是挑選對假設有利的展示。</p>
  </section>
</div>
"""

    with gr.Blocks(title="SafeSynth", fill_width=True) as interface:
        gr.HTML(header, elem_classes="ss-header-wrap")
        with gr.Tabs(elem_classes="ss-tabs"):
            with gr.Tab("圖片偵測"):
                with gr.Row(elem_classes="ss-evidence-hero"):
                    with gr.Column(scale=8, elem_classes="ss-image-stage"):
                        image_source = gr.HTML(default_presentation.source_html)
                        comparison = gr.ImageSlider(
                            value=default_presentation.comparison,
                            type="numpy",
                            label="原始影像與模型標註比較",
                            show_label=False,
                            buttons=["fullscreen", "download"],
                            height="auto",
                            max_height=620,
                            elem_classes="ss-comparison",
                        )
                        gr.HTML(
                            '<p class="ss-slider-help"><b>Before</b> 原始影像 <span></span> 拖曳中線比較 <span></span> <b>After</b> 模型標註</p>'
                        )
                    with gr.Column(scale=4, elem_classes="ss-summary-panel"):
                        image_summary = gr.HTML(default_presentation.summary_html)
                        image_upload = gr.UploadButton(
                            "上傳你的工地影像",
                            file_types=["image"],
                            file_count="single",
                            type="filepath",
                            variant="primary",
                            size="lg",
                            elem_classes="ss-upload",
                        )
                        image_reset = gr.Button(
                            "回到精選範例",
                            variant="secondary",
                            size="lg",
                            elem_classes="ss-reset",
                        )
                        image_error = gr.HTML(default_presentation.error_html)
                image_evidence = gr.HTML(
                    default_presentation.evidence_html,
                    elem_classes="ss-evidence-wrap",
                )
                with gr.Accordion(
                    "研究方法與限制",
                    open=False,
                    elem_classes="ss-disclosure",
                ):
                    gr.HTML(research_copy)

                image_outputs = [
                    comparison,
                    image_summary,
                    image_evidence,
                    image_source,
                    image_error,
                ]
                image_upload.upload(
                    on_image,
                    image_upload,
                    image_outputs,
                    show_progress="full",
                )
                image_reset.click(
                    on_reset,
                    inputs=None,
                    outputs=image_outputs,
                    show_progress="hidden",
                )
            with gr.Tab("影片偵測"):
                gr.HTML(
                    f"""
<section class="ss-video-intro">
  <h2>短影片逐幀檢查</h2>
  <p>為維持可預測的處理時間，每次最多分析前 <strong>{MAX_VIDEO_FRAMES}</strong> frames；完成後提供標註影片與實測 latency。</p>
</section>
"""
                )
                with gr.Row(elem_classes="ss-video-workspace"):
                    video_in = gr.Video(label="影片來源", sources=["upload"])
                    video_out = gr.Video(label="標註結果", interactive=False)
                video_summary = gr.HTML()
                video_perf = gr.HTML(elem_classes="ss-evidence-wrap")
                video_in.change(
                    on_video,
                    video_in,
                    [video_out, video_summary, video_perf],
                    show_progress="full",
                )
                with gr.Accordion(
                    "研究方法與限制",
                    open=False,
                    elem_classes="ss-disclosure",
                ):
                    gr.HTML(research_copy)
        gr.HTML(
            """
<footer class="ss-footer">
  <p>SafeSynth · Controlled synthetic-data ablations for hard-hat detection</p>
  <p>可重現的模型、資料與研究紀錄皆已公開。</p>
</footer>
"""
        )
    return interface


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_evaluation_config()
    threshold = float(config["compliance"]["score_threshold"])

    try:
        checkpoint = args.weights or resolve_weights(
            load_project_paths().runs, args.arm, args.seed
        )
    except DemoStartupError as error:
        print(error)
        return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but not available; falling back to cpu")
        args.device = "cpu"
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dtype = torch.float16 if args.device == "cuda" else torch.float32
    print(f"loading {checkpoint} on {args.device} ({dtype})")
    detector = Detector(checkpoint, args.device, dtype)

    # DEMO-03: never time the first inference. It carries import, allocator and
    # autotuning costs that no later frame pays.
    warmup = np.zeros((416, 416, 3), dtype=np.uint8)
    detector(warmup)

    build_interface(detector, threshold).launch(
        server_port=args.port,
        share=args.share,
        inbrowser=False,
        css_paths=PROJECT_ROOT / "assets" / "demo_ui.css",
        footer_links=[
            {"text": "GitHub", "url": "https://github.com/kuotunyu/SafeSynth"},
            {
                "text": "Dataset",
                "url": "https://huggingface.co/datasets/steven0226/safesynth-hard-hat",
            },
            {
                "text": "Model",
                "url": "https://huggingface.co/steven0226/safesynth-rtdetrv2-r18",
            },
        ],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
