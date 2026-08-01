"""DEMO-04: the README's demo GIF, built from the dataset instead of site footage.

WHAT THIS IS, STATED UP FRONT. DEMO-04 says "record a short GIF", and the
obvious reading is a video clip. This project has no site footage, and the
dataset cannot supply one: it is often described as video-derived, but the
frozen pHash grouping says otherwise - 4,643 of 4,808 groups are a single
image and the largest group is 8 frames, so there is no run of consecutive
frames long enough to be a clip.

What the dataset does supply is 501 images containing BOTH a helmeted head and
a bare head, which is DEMO-04's actual acceptance criterion. So this builds a
cycling montage of annotated still frames rather than a recording, and the
README says so in as many words. A montage presented as live video would be a
small lie about a capability the demo does not have.

Frames come from VALIDATION, never Test, matching the convention the latency
probe already follows. Nothing here trains or tunes anything, but a README
figure drawn on the frozen Test split invites a question that costs nothing to
avoid.

SELECTION, AND THE CRITERION THAT WAS WRONG FIRST. The obvious ranking is "most
bare heads", since bare heads are what the project is about. Opening the
resulting GIF showed why that is exactly backwards: ranking on bare heads
selects crowds of people NOT wearing helmets, so the first frame was 38 boxes
of which 2 were compliant, on a picture of a crowd in winter coats that is not
recognisably a construction site. Every box was red. The one thing the demo
exists to show - green helmet against red bare head - was invisible.

The criterion is therefore BALANCE, not count. Frames must carry at least
`MIN_PER_CLASS` of each class so both colours are substantial, and no more than
`MAX_INSTANCES` boxes so a 416 px frame stays readable. Among those, rank by how
balanced the two classes are, then by fewest boxes, then by file name.

AND THE COUNT THAT MATTERS IS THE DRAWN ONE. The second attempt filtered on
GROUND TRUTH instance counts, which is not what a viewer sees: one frame passed
the "at most 12" test with 8 annotations and then drew 22 boxes, because at the
EVAL-04 operating point of 0.07 this poorly-calibrated model emits far more
boxes than there are objects. The result was unreadable - overlapping labels and
person rectangles spanning the image. Candidates are therefore scored AFTER
inference, on the boxes the demo actually draws.

`EXCLUDED_FRAMES` is an editorial list, and deliberately explicit rather than a
silent filter. A README figure is the first thing a reader sees, and one of the
top-ranked frames shows what reads as an injured person suspended from a rope
above a wheeled stretcher. No automatic criterion catches that, so it is named
here with its reason instead of being quietly dropped.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.evaluation.detection import load_evaluation_config
from src.inference.demo import draw_on, drawn_boxes, summarise
from src.training.ingest import latest_checkpoint
from src.training.metrics import predictions_to_coco

CLASS_NAMES = ("helmet", "head", "person")
PROCESSOR_ID = "PekingU/rtdetr_v2_r18vd"
DEFAULT_ARM = "real_only"
DEFAULT_SEED = 1337
FRAME_SPLIT = "val"
GIF_NAME = "demo.gif"

# Readability bounds for a 416 px frame, not statistical parameters: they decide
# what a reader can see, and nothing downstream depends on them.
MIN_PER_CLASS = 2
MAX_INSTANCES = 12
MIN_INSTANCES = 4

# Applied to the boxes the demo DRAWS, which is what a reader counts.
MAX_DRAWN_BOXES = 12
MIN_DRAWN_PER_VERDICT = 2

# Editorial exclusions, named with their reason. Not a silent filter: a reader
# of this file should be able to see every human decision that shaped the GIF.
EXCLUDED_FRAMES = {
    "images/hard_hat_workers1697.png": (
        "reads as an accident - a limp figure suspended from a rope above a "
        "wheeled stretcher. Ranked first on every automatic criterion; no "
        "automatic criterion can see why it is the wrong README image."
    ),
}


class DemoGifError(RuntimeError):
    """Raised when the montage cannot be built from the frozen split."""


@dataclass(frozen=True)
class FrameChoice:
    """One selected image plus the counts that selected it."""

    file_name: str
    n_bare_heads: int
    n_helmets: int
    smallest_box_area: float

    @property
    def balance(self) -> int:
        """How many of the rarer class are present. The ranking quantity."""

        return min(self.n_helmets, self.n_bare_heads)

    @property
    def n_instances(self) -> int:
        return self.n_helmets + self.n_bare_heads

    def readable(self) -> bool:
        return (
            self.balance >= MIN_PER_CLASS
            and MIN_INSTANCES <= self.n_instances <= MAX_INSTANCES
        )

    def sort_key(self) -> tuple[int, int, str]:
        # Negative balance so the most even mix sorts first; then fewest boxes,
        # because a readable frame beats a busy one; then name, for determinism.
        return (-self.balance, self.n_instances, self.file_name)


# spec: DEMO-04
def choose_frames(
    manifest: Mapping[str, object],
    annotations: Mapping[str, object],
    *,
    split: str,
    n_frames: int,
) -> list[FrameChoice]:
    """Validation images holding both a helmeted and a bare head, best first.

    Raises rather than returning a short list: a montage that quietly contains
    four frames because only four qualified would still look deliberate, and
    the caller would never learn the criterion had been relaxed.
    """

    category = {int(c["id"]): str(c["name"]) for c in annotations["categories"]}
    wanted = {entry["file_name"] for entry in manifest["images"] if entry["split"] == split}
    image_name = {
        int(image["id"]): str(image["file_name"])
        for image in annotations["images"]
        if str(image["file_name"]) in wanted
    }

    per_image: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in annotations["annotations"]:
        image_id = int(record["image_id"])
        if image_id in image_name:
            per_image[image_id].append(record)

    choices: list[FrameChoice] = []
    for image_id, records in per_image.items():
        labels = [category[int(r["category_id"])] for r in records]
        # No "does it have both classes" check here on purpose: it would be dead
        # code. `balance` is min(helmets, bare heads), so an image missing either
        # class has balance 0 and `readable()` already rejects it. Mutation
        # testing found the two conditions could not be told apart.
        areas = [float(r["bbox"][2]) * float(r["bbox"][3]) for r in records]
        choices.append(
            FrameChoice(
                file_name=image_name[image_id],
                n_bare_heads=labels.count("head"),
                n_helmets=labels.count("helmet"),
                smallest_box_area=min(areas),
            )
        )

    usable = [
        choice
        for choice in choices
        if choice.readable() and choice.file_name not in EXCLUDED_FRAMES
    ]
    if len(usable) < n_frames:
        raise DemoGifError(
            f"{len(choices)} annotated {split} images, of which only {len(usable)} "
            f"are readable (>={MIN_PER_CLASS} of each class, "
            f"{MIN_INSTANCES}-{MAX_INSTANCES} boxes) and not excluded, "
            f"which is fewer than the {n_frames} asked for"
        )
    usable.sort(key=FrameChoice.sort_key)
    # Everything that survives is a CANDIDATE; the drawn-box test in
    # annotate_frames does the final cut, so hand back the full ranked list.
    return usable


# spec: DEMO-04
def caption_for(summary, file_name: str) -> str:
    """One line under each frame. The rate is the demo's actual output."""

    rate = "n/a" if summary.compliance_rate is None else f"{summary.compliance_rate:.2f}"
    return (
        f"{file_name}  |  {summary.n_compliant} / "
        f"{summary.n_compliant + summary.n_non_compliant} compliant  |  rate {rate}"
    )


# spec: DEMO-04
def stack_caption(frame: np.ndarray, text: str, *, band_px: int = 22) -> np.ndarray:
    """Add a caption band under the frame, without covering any detection."""

    import cv2

    height, width = frame.shape[:2]
    canvas = np.zeros((height + band_px, width, 3), dtype=np.uint8)
    canvas[:height] = frame
    canvas[height:] = (24, 24, 24)
    cv2.putText(
        canvas,
        text,
        (6, height + band_px - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


# spec: DEMO-04
def publishable(*, n_drawn: int, n_compliant: int, n_non_compliant: int) -> str | None:
    """None if the frame may go in the GIF, else the reason it may not.

    Separated from the inference loop so it can be tested without a GPU, and
    so the two rules are visible side by side. Both are about what a READER
    sees, which is why they count drawn boxes and not annotations:

      * too busy - at this model's 0.07 operating point a frame can carry more
        than twice as many boxes as objects, and 416 px cannot hold them.
      * one verdict barely present - a montage of all-green or all-red frames
        does not show the thing the demo exists to show.
    """

    if n_drawn > MAX_DRAWN_BOXES:
        return "too busy"
    if min(n_compliant, n_non_compliant) < MIN_DRAWN_PER_VERDICT:
        return "one verdict barely present"
    return None


def resolve_weights(runs_root: Path, arm: str, seed: int) -> Path:
    """The checkpoint the demo serves, so the GIF shows what app.py shows."""

    seed_dir = Path(runs_root) / arm / f"seed_{seed}"
    newest = latest_checkpoint(seed_dir)
    if newest is None:
        raise DemoGifError(f"no checkpoint under {seed_dir}")
    state = json.loads((newest / "trainer_state.json").read_text(encoding="utf-8"))
    recorded = state.get("best_model_checkpoint")
    if not recorded:
        return newest
    resolved = seed_dir / Path(recorded).name
    return resolved if resolved.is_dir() else newest


def annotate_frames(
    choices: Sequence[FrameChoice],
    *,
    images_root: Path,
    checkpoint: Path,
    device: str,
    threshold: float,
    n_frames: int,
) -> list[np.ndarray]:
    import cv2
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoImageProcessor.from_pretrained(PROCESSOR_ID)
    model = (
        AutoModelForObjectDetection.from_pretrained(str(checkpoint), dtype=dtype)
        .eval()
        .to(device)
    )

    frames: list[np.ndarray] = []
    rejected = 0
    for choice in choices:
        if len(frames) == n_frames:
            break
        path = Path(images_root) / choice.file_name
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        # dtype as well as device: the processor always emits float32, and a
        # float16 model meets it with "Input type and weight type should be the
        # same". app.py had the same gap, unnoticed because its default is CPU.
        encoded = processor(images=image, return_tensors="pt").to(device=device, dtype=dtype)
        with torch.no_grad():
            outputs = model(**encoded)
        # The output object is a ModelOutput dataclass, not a tensor - it has no
        # .to(). Post-processing takes it on whatever device it is already on,
        # so long as target_sizes agrees, which is what this does.
        target = torch.tensor(
            [[image.shape[0], image.shape[1]]], dtype=torch.float32, device=device
        )
        processed = processor.post_process_object_detection(
            outputs, threshold=0.0, target_sizes=target
        )
        boxes = drawn_boxes(
            predictions_to_coco(processed, [0]),
            class_names=CLASS_NAMES,
            score_threshold=threshold,
        )
        summary = summarise(boxes)

        verdict = publishable(
            n_drawn=len(boxes),
            n_compliant=summary.n_compliant,
            n_non_compliant=summary.n_non_compliant,
        )
        if verdict is not None:
            print(
                f"  SKIP {choice.file_name}: {verdict} "
                f"({len(boxes)} drawn, {summary.n_compliant}G/{summary.n_non_compliant}R)"
            )
            rejected += 1
            continue

        frames.append(stack_caption(draw_on(image, boxes), caption_for(summary, choice.file_name)))
        print(f"  keep {choice.file_name}: {summary.render()}")

    if len(frames) < n_frames:
        raise DemoGifError(
            f"only {len(frames)} of {len(choices)} candidates survived the drawn-box "
            f"test ({rejected} rejected); asked for {n_frames}"
        )
    return frames


def write_gif(frames: Sequence[np.ndarray], destination: Path, *, ms_per_frame: int) -> None:
    from PIL import Image

    destination.parent.mkdir(parents=True, exist_ok=True)
    pages = [Image.fromarray(frame) for frame in frames]
    pages[0].save(
        destination,
        save_all=True,
        append_images=pages[1:],
        duration=ms_per_frame,
        loop=0,
        optimize=True,
    )


def parse_args() -> argparse.Namespace:
    paths = load_project_paths()
    parser = argparse.ArgumentParser(description="DEMO-04 montage GIF")
    parser.add_argument("--runs-root", type=Path, default=paths.data_root / "runs")
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--ms-per-frame", type=int, default=1200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "assets" / GIF_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_project_paths()
    config = load_evaluation_config()
    threshold = float(config["compliance"]["score_threshold"])

    manifest = json.loads((paths.splits / "split_manifest.json").read_text(encoding="utf-8"))
    annotations = json.loads((paths.interim / "coco_all.json").read_text(encoding="utf-8"))
    choices = choose_frames(
        manifest, annotations, split=FRAME_SPLIT, n_frames=args.frames
    )
    print(
        f"{len(choices)} {FRAME_SPLIT} candidates passed the annotation filter; "
        f"keeping the first {args.frames} that survive the drawn-box test:"
    )

    if args.device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    checkpoint = resolve_weights(args.runs_root, args.arm, args.seed)
    print(f"weights: {checkpoint}  device={device}  threshold={threshold}")

    frames = annotate_frames(
        choices,
        images_root=paths.hardhat_raw,
        checkpoint=checkpoint,
        device=device,
        threshold=threshold,
        n_frames=args.frames,
    )
    write_gif(frames, args.out, ms_per_frame=args.ms_per_frame)
    size_kb = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({len(frames)} frames, {size_kb:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
