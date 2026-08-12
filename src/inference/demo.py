"""Drawing and summarising for the Gradio demo (DEMO-01, DEMO-02).

Kept out of the Gradio app so the parts that can be wrong are testable without
launching a server: which colour a box gets, what the frame summary says, and
how an empty frame is described.

The colour rule is the whole point of the demo. A detector demo draws boxes; this
project's claim is about COMPLIANCE, so a box is coloured by the verdict its
class implies, not by its class. `helmet` means a helmeted head and reads
COMPLIANT; `head` means a bare head and reads NON_COMPLIANT (ADR-007, DATA-24).
`person` is drawn in a neutral colour and carries NO verdict - ADR-003 removed it
from the compliance path and EVAL-03 pins that by test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.inference.compliance import ComplianceStatus

# BGR, because the drawing goes through cv2. These low-saturation colours align
# with the demo's visual system; the caption and summary carry the verdict too,
# so colour is never the only semantic channel.
COMPLIANT_COLOUR = (138, 157, 127)  # RGB #7F9D8A after reversal
NON_COMPLIANT_COLOUR = (114, 125, 195)  # RGB #C37D72 after reversal
NEUTRAL_COLOUR = (154, 145, 127)  # RGB #7F919A after reversal

STATUS_COLOUR = {
    ComplianceStatus.COMPLIANT: COMPLIANT_COLOUR,
    ComplianceStatus.NON_COMPLIANT: NON_COMPLIANT_COLOUR,
}

SEMANTIC_LABEL = {
    ComplianceStatus.COMPLIANT: "已佩戴",
    ComplianceStatus.NON_COMPLIANT: "未佩戴",
    None: "僅定位",
}


class DemoError(RuntimeError):
    """Raised when a frame cannot be summarised."""


@dataclass(frozen=True)
class DrawnBox:
    """One box as the demo will render it."""

    label: str
    score: float
    bbox_xywh: tuple[float, float, float, float]
    status: ComplianceStatus | None
    colour: tuple[int, int, int]

    @property
    def caption(self) -> str:
        """Class, confidence and verdict, in that order.

        The confidence is shown to two decimals rather than as a percentage
        because this model's scores top out near 0.25 - rendering that as "25%"
        invites the reader to compare it with a calibrated detector's 25%, which
        would be a different thing entirely.
        """

        verdict = f" · {self.status.value}" if self.status is not None else ""
        return (
            f"{self.label} · {SEMANTIC_LABEL[self.status]} · "
            f"{self.score:.2f}{verdict}"
        )

    def caption_for_width(self, pixels_per_character: float) -> str:
        """As much of the caption as fits over a box this wide.

        A crowded site photo at 416 px can carry fifteen boxes, and the full
        caption on every one turns the middle of the frame into unreadable
        overlapping text - which is what the first render of this demo did.

        Dropping text is safe here because COLOUR is the primary channel by
        design: green is compliant, red is not. The words are a convenience for
        boxes with room for them, so they degrade class-and-score first and then
        away entirely, rather than being shrunk to illegibility.
        """

        box_width = self.bbox_xywh[2]
        room = box_width / max(pixels_per_character, 1e-6)
        if room >= len(self.caption):
            return self.caption
        short = f"{self.label} {self.score:.2f}"
        if room >= len(short):
            return short
        if room >= len(self.label):
            return self.label
        return ""


@dataclass(frozen=True)
class FrameSummary:
    """What the demo prints beside the picture."""

    n_compliant: int
    n_non_compliant: int
    n_neutral: int

    @property
    def n_people(self) -> int:
        return self.n_compliant + self.n_non_compliant

    @property
    def compliance_rate(self) -> float | None:
        """None when no head was found at all - not zero.

        Zero would read as "nobody here is wearing a helmet", which is a strong
        claim about a frame the model simply had nothing to say about.
        """

        return None if not self.n_people else self.n_compliant / self.n_people

    def render(self) -> str:
        if not self.n_people:
            extra = f" ({self.n_neutral} person box(es), which carry no verdict)"
            return f"no heads detected{extra if self.n_neutral else ''}"
        return (
            f"{self.n_compliant} / {self.n_people} compliant  ·  "
            f"compliance_rate {self.compliance_rate:.2f}"
        )


# spec: DEMO-02
def status_for_label(label: str) -> ComplianceStatus | None:
    """The verdict a detected class implies under `class_direct` (ADR-007)."""

    if label == "helmet":
        return ComplianceStatus.COMPLIANT
    if label == "head":
        return ComplianceStatus.NON_COMPLIANT
    return None


# spec: DEMO-02
def drawn_boxes(
    detections: Sequence[Mapping[str, Any]],
    *,
    class_names: Sequence[str],
    score_threshold: float,
) -> list[DrawnBox]:
    """Filter to the operating point and attach a colour to every survivor.

    Sorted by ascending score so the confident boxes are drawn last and are not
    hidden under the marginal ones. At this model's operating point of 0.07 a
    frame can carry a lot of low-scoring boxes.
    """

    kept = [
        detection
        for detection in detections
        if float(detection["score"]) >= score_threshold
    ]
    kept.sort(key=lambda detection: float(detection["score"]))

    boxes: list[DrawnBox] = []
    for detection in kept:
        index = int(detection["category_id"])
        if not 0 <= index < len(class_names):
            raise DemoError(
                f"category_id {index} is outside the {len(class_names)} configured "
                f"classes; a detection cannot be drawn without knowing what it is"
            )
        label = class_names[index]
        status = status_for_label(label)
        boxes.append(
            DrawnBox(
                label=label,
                score=float(detection["score"]),
                bbox_xywh=tuple(float(value) for value in detection["bbox"]),
                status=status,
                colour=STATUS_COLOUR.get(status, NEUTRAL_COLOUR),
            )
        )
    return boxes


# spec: DEMO-02
def summarise(boxes: Sequence[DrawnBox]) -> FrameSummary:
    return FrameSummary(
        n_compliant=sum(1 for box in boxes if box.status is ComplianceStatus.COMPLIANT),
        n_non_compliant=sum(
            1 for box in boxes if box.status is ComplianceStatus.NON_COMPLIANT
        ),
        n_neutral=sum(1 for box in boxes if box.status is None),
    )


def draw_on(image, boxes: Sequence[DrawnBox]):
    """Render boxes onto an RGB numpy array. Returns a new array."""

    import cv2
    import numpy as np

    canvas = np.ascontiguousarray(image.copy())
    height, width = canvas.shape[:2]
    thickness = max(1, round(min(height, width) / 300))
    scale = max(0.32, min(height, width) / 1200)
    # HERSHEY_SIMPLEX advances about 19 px per character at scale 1.0. Measured
    # rather than assumed: cv2.getTextSize("MMMMMMMMMM", ...) / 10 at scale 1.0.
    pixels_per_character = 19.0 * scale
    for box in boxes:
        x, y, box_width, box_height = box.bbox_xywh
        first = (round(x), round(y))
        second = (round(x + box_width), round(y + box_height))
        # cv2 wants BGR; the canvas is RGB, so the stored colours are reversed
        # here rather than at the constant, which keeps them readable above.
        colour = box.colour[::-1]
        cv2.rectangle(canvas, first, second, colour, thickness)
        caption = box.caption_for_width(pixels_per_character)
        if not caption:
            continue
        cv2.putText(
            canvas,
            caption,
            (first[0], max(10, first[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas
