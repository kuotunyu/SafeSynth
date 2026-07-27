from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_generative_identity_pilot import (
    PilotEvidence,
    _require_input_preflight_approved,
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


def test_gpu_pilot_rejects_failed_input_preflight(tmp_path: Path) -> None:
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        (
            '{"architecture":"guarded_context_replacement_v5",'
            '"root_seed":20260731,"status":"rejected_by_kuotunyu",'
            '"reviewed_by":"kuotunyu","observed_input_issue_count":33}'
        ),
        encoding="utf-8",
    )
    config = {
        "pilot": {
            "architecture": "guarded_context_replacement_v5",
            "root_seed": 20260731,
        }
    }

    with np.testing.assert_raises_regex(RuntimeError, "GPU identity pilot locked"):
        _require_input_preflight_approved(report_path, config)


def test_gpu_pilot_accepts_only_exact_zero_issue_approval(tmp_path: Path) -> None:
    report_path = tmp_path / "preflight.json"
    report_path.write_text(
        (
            '{"architecture":"guarded_context_replacement_v5",'
            '"root_seed":20260731,"status":"approved_by_kuotunyu",'
            '"reviewed_by":"kuotunyu","observed_input_issue_count":0}'
        ),
        encoding="utf-8",
    )
    config = {
        "pilot": {
            "architecture": "guarded_context_replacement_v5",
            "root_seed": 20260731,
        }
    }

    _require_input_preflight_approved(report_path, config)
