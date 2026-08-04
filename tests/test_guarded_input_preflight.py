from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.prepare_guarded_identity_preflight import (
    CpuInputCapture,
    InputEvidence,
    render_input_sheet,
)


def test_cpu_input_capture_returns_draft_without_model_work() -> None:
    capture = CpuInputCapture()
    draft = np.full((24, 24, 3), [30, 60, 90], dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=bool)
    mask[8:16, 9:15] = True

    result = capture.generate(
        draft_rgb=draft,
        object_mask=mask,
        reference_rgba=np.zeros((8, 6, 4), dtype=np.uint8),
        class_name="helmet",
        seed=42,
    )

    assert np.array_equal(result.image_rgb, draft)
    assert np.array_equal(capture.by_seed[42].object_mask, mask)
    assert result.provenance["method"] == "cpu_guarded_input_capture"


def test_input_sheet_renders_registered_8_by_8_grid(tmp_path: Path) -> None:
    evidence: dict[int, InputEvidence] = {}
    records = []
    for index in range(1, 65):
        draft = np.full((32, 32, 3), index, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 11:21] = True
        evidence[index] = InputEvidence(draft, mask, "helmet", index)
        records.append(
            {
                "sample_id": f"sample_{index:02d}",
                "instances": [
                    {"generative_inpaint": {"seed": index}},
                ],
                "context_replacement_input_guard": {
                    "selected_anchor_edge_margin_px": 10.0,
                    "selected_anchor_required_edge_margin_px": 8,
                },
            }
        )
    output_path = tmp_path / "sheet.png"

    render_input_sheet(
        records=records,
        evidence_by_seed=evidence,
        output_path=output_path,
    )

    with Image.open(output_path) as sheet:
        assert sheet.size == (3136, 1808)
    assert output_path.stat().st_size > 0
