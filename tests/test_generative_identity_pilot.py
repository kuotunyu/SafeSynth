from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_generative_identity_pilot import (
    PilotEvidence,
    render_contact_sheet,
)


def test_contact_sheet_contains_registered_four_panels(tmp_path: Path) -> None:
    output_dir = tmp_path / "pilot"
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)
    final = np.full((32, 32, 3), [20, 40, 60], dtype=np.uint8)
    Image.fromarray(final).save(image_dir / "sample.png")
    edit_mask = np.zeros((32, 32), dtype=bool)
    edit_mask[10:22, 12:20] = True
    evidence = PilotEvidence(
        draft_rgb=np.full_like(final, [10, 30, 50]),
        edit_mask=edit_mask,
        reference_rgb=np.full_like(final, [127, 127, 127]),
        output_rgb=final,
        class_name="helmet",
        seed=42,
    )
    records = [
        {
            "file_name": "images/sample.png",
            "passed": True,
            "first_reject_reason": None,
            "instances": [
                {
                    "generative_inpaint": {
                        "seed": 42,
                        "identity_metrics": {
                            "outside_edit_changed_pixels": 0,
                            "protected_core_changed_pixels": 0,
                        },
                    }
                }
            ],
        }
    ]
    output_path = tmp_path / "sheet.png"

    render_contact_sheet(
        records=records,
        evidence_by_seed={42: evidence},
        output_dir=output_dir,
        output_path=output_path,
        rows=1,
        columns=1,
    )

    sheet = Image.open(output_path)
    assert sheet.size == (392, 420)
    assert output_path.stat().st_size > 0
