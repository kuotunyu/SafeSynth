"""Gradio demo: upload an image or a short clip, get compliance per person.

DEMO-01..DEMO-03. No webcam (a construction scene is not reproducible in front
of a laptop camera, and assuming the user has one is a bad default).

Runs on CPU by default. That is not a limitation being apologised for - the
measured latency on this machine is around 300 ms per frame on CPU, which is
usable for the single images and short clips this demo takes, and it means the
demo does not compete with whatever else is using the GPU.

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
from pathlib import Path

import numpy as np
import torch

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.detection import load_evaluation_config
from src.inference.demo import draw_on, drawn_boxes, summarise
from src.training.ingest import latest_checkpoint
from src.training.metrics import predictions_to_coco

CLASS_NAMES = ("helmet", "head", "person")
PROCESSOR_ID = "PekingU/rtdetr_v2_r18vd"
DEFAULT_ARM = "real_only"
MAX_VIDEO_FRAMES = 120


class DemoStartupError(RuntimeError):
    """Raised when the demo cannot find weights to serve."""


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


def build_interface(detector: Detector, threshold: float):
    import gradio as gr

    def on_image(image):
        if image is None:
            return None, "upload an image", ""
        annotated, summary, model_ms, e2e_ms = annotate(detector, image, threshold)
        return annotated, summary.render(), performance_line(model_ms, e2e_ms, detector)

    def on_video(path):
        if not path:
            return None, "upload a clip", ""
        import cv2

        capture = cv2.VideoCapture(path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 12.0
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
        capture.release()
        if not frames:
            return None, "could not read any frame from that file", ""

        out = PROJECT_ROOT / "reports" / "figures" / "demo_annotated.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

        mean_rate = f"{sum(rates) / len(rates):.2f}" if rates else "n/a"
        note = (
            f"{len(frames)} frames"
            f"{' (truncated)' if len(frames) == MAX_VIDEO_FRAMES else ''}"
            f"  ·  mean compliance_rate {mean_rate}"
        )
        # Median, not mean: DEMO-03 says so, and one stalled frame would move a mean.
        return (
            str(out),
            note,
            performance_line(
                float(np.median(model_times)), float(np.median(e2e_times)), detector
            ),
        )

    header = f"""
# SafeSynth — hard-hat compliance

Green is a helmeted head (**compliant**), red is a bare head
(**non-compliant**), grey is a `person` box, which carries **no verdict** —
that class is badly annotated in this dataset and was deliberately removed from
the compliance path.

**The score threshold is {threshold:.2f}, and that is not a typo.** Across
223,200 test detections this model's highest score is 0.2495: the ranking is
good and the calibration is not. The operating point was selected on Validation
against a 0.80 compliance-precision floor and then frozen.

Serving `{DEFAULT_ARM}` from `{detector.checkpoint.name}` — the four-arm
experiment found that **synthetic data did not improve detection** on this
dataset, so the real-only weights are the best ones and those are what ships.
"""

    with gr.Blocks(title="SafeSynth") as interface:
        gr.Markdown(header)
        with gr.Tab("Image"):
            image_in = gr.Image(type="numpy", label="upload a site photo")
            image_out = gr.Image(label="annotated")
            image_summary = gr.Textbox(label="frame summary", interactive=False)
            image_perf = gr.Textbox(label="performance", interactive=False)
            image_in.change(
                on_image, image_in, [image_out, image_summary, image_perf]
            )
        with gr.Tab("Video"):
            video_in = gr.Video(label=f"upload a clip (first {MAX_VIDEO_FRAMES} frames)")
            video_out = gr.Video(label="annotated")
            video_summary = gr.Textbox(label="clip summary", interactive=False)
            video_perf = gr.Textbox(label="performance (median)", interactive=False)
            video_in.change(
                on_video, video_in, [video_out, video_summary, video_perf]
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
        server_port=args.port, share=args.share, inbrowser=False
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
