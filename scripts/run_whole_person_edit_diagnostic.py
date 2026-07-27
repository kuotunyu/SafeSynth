"""Run the approved four-case v8 FLUX diagnostic on the local GPU."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import _archive_existing
from src.synthetic.generative_inpaint import (
    load_flux2_pipeline,
    load_generative_config,
    model_directory,
)
from src.synthetic.region_inpaint import (
    enforce_outside_edit_exact,
    region_identity_metrics,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "whole_person_edit_diagnostic.yaml"
PREFLIGHT_ROOT = (
    PROJECT_ROOT / "outputs" / "whole_person_edit_preflight_seed20260804"
)
PREFLIGHT_REPORT = (
    PROJECT_ROOT / "reports" / "whole_person_edit_preflight_v8.json"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "whole_person_edit_diagnostic_seed20260804"
)
FIGURE_PATH = (
    PROJECT_ROOT / "reports" / "figures" / "whole_person_edit_diagnostic_v8.png"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "whole_person_edit_diagnostic_v8.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_seed(root_seed: int, case_index: int) -> int:
    payload = f"v8|{root_seed}|{case_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        2**63 - 1
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def require_approved_inputs(
    *,
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    input_root: Path,
) -> None:
    """Block model loading until the exact v8 manifest is approved."""

    manifest_path = input_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("v8 GPU gate locked: exact input manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate = config["model_gate"]
    manifest_sha256 = str(manifest["input_manifest_sha256"])
    if (
        gate.get("allowed") is not True
        or gate.get("review_status") != "approved_by_kuotunyu"
        or gate.get("required_reviewer") != "kuotunyu"
        or gate.get("approved_manifest_sha256") != manifest_sha256
        or report.get("status") != "approved_by_kuotunyu"
        or report.get("reviewed_by") != "kuotunyu"
        or int(report.get("observed_input_issue_count", -1)) != 0
        or report.get("input_manifest_sha256") != manifest_sha256
    ):
        raise RuntimeError(
            "v8 GPU gate locked: kuotunyu has not approved these exact inputs"
        )
    for case in manifest["cases"]:
        for name, relative_path in case["files"].items():
            path = input_root / str(relative_path)
            expected = str(case["sha256"][name])
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(
                    f"v8 GPU gate locked: input hash changed for {path}"
                )


def _render_contact_sheet(
    rows: list[dict[str, Path]],
    *,
    output_path: Path,
) -> None:
    panel = 256
    header = 42
    labels = ("ORIGINAL", "EDIT MASK", "REFERENCE", "FLUX OUTPUT")
    sheet = Image.new(
        "RGB",
        (panel * len(labels), (panel + header) * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        panels = [
            Image.open(row["draft"]).convert("RGB"),
            Image.open(row["edit_mask"]).convert("L").convert("RGB"),
            Image.open(row["reference"]).convert("RGB"),
            Image.open(row["output"]).convert("RGB"),
        ]
        y0 = row_index * (panel + header)
        draw.text((5, y0 + 5), f"CASE {row_index + 1:02d}", fill="black")
        for column, (label, source) in enumerate(
            zip(labels, panels, strict=True)
        ):
            draw.text((column * panel + 5, y0 + 24), label, fill="black")
            sheet.paste(
                source.resize((panel, panel), Image.Resampling.LANCZOS),
                (column * panel, y0 + header),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    report = json.loads(PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    require_approved_inputs(
        config=config,
        report=report,
        input_root=PREFLIGHT_ROOT,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("v8 diagnostic requires an available CUDA GPU")

    paths = load_project_paths()
    model_config = load_generative_config()
    model_dir = model_directory(paths, model_config)
    pipeline = None
    try:
        pipeline = load_flux2_pipeline(
            model_dir=model_dir,
            config=model_config,
        )
        manifest = json.loads(
            (PREFLIGHT_ROOT / "manifest.json").read_text(encoding="utf-8")
        )
        _archive_existing(OUTPUT_ROOT)
        OUTPUT_ROOT.mkdir(parents=True)
        records: list[dict[str, Any]] = []
        sheet_rows: list[dict[str, Path]] = []
        for case in manifest["cases"]:
            case_index = int(case["case_index"])
            case_input_dir = PREFLIGHT_ROOT / f"case_{case_index:02d}"
            case_output_dir = OUTPUT_ROOT / f"case_{case_index:02d}"
            case_output_dir.mkdir()
            draft_path = case_input_dir / "draft.png"
            mask_path = case_input_dir / "edit_mask.png"
            reference_path = case_input_dir / "reference.png"
            draft = np.asarray(
                Image.open(draft_path).convert("RGB"),
                dtype=np.uint8,
            )
            edit_mask = (
                np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
                > 0
            )
            reference = Image.open(reference_path).convert("RGB")
            seed = _case_seed(int(config["root_seed"]), case_index)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = pipeline(
                prompt=str(config["method"]["prompt"]),
                image=Image.fromarray(draft),
                image_reference=reference,
                mask_image=Image.fromarray(
                    edit_mask.astype(np.uint8) * 255
                ),
                padding_mask_crop=int(
                    config["method"]["padding_mask_crop_px"]
                ),
                strength=float(config["method"]["strength"]),
                num_inference_steps=int(
                    config["method"]["num_inference_steps"]
                ),
                guidance_scale=float(config["method"]["guidance_scale"]),
                generator=generator,
                output_type="pil",
            )
            raw_generated = np.asarray(
                result.images[0].convert("RGB"),
                dtype=np.uint8,
            )
            output = enforce_outside_edit_exact(
                draft,
                raw_generated,
                edit_mask=edit_mask,
            )
            metrics = region_identity_metrics(
                draft,
                output,
                edit_mask=edit_mask,
            )
            if int(metrics["outside_edit_changed_pixels"]) != 0:
                raise AssertionError("v8 outside-edit identity invariant failed")
            raw_path = case_output_dir / "raw_model_output.png"
            output_path = case_output_dir / "output.png"
            Image.fromarray(raw_generated).save(raw_path, optimize=True)
            Image.fromarray(output).save(output_path, optimize=True)
            record = {
                "case_index": case_index,
                "seed": seed,
                "identity_metrics": metrics,
                "raw_model_output_sha256": _sha256(raw_path),
                "output_sha256": _sha256(output_path),
                "target": case["target"],
                "reference": case["reference"],
            }
            _write_json(case_output_dir / "record.json", record)
            records.append(record)
            sheet_rows.append(
                {
                    "draft": draft_path,
                    "edit_mask": mask_path,
                    "reference": reference_path,
                    "output": output_path,
                }
            )

        _render_contact_sheet(sheet_rows, output_path=FIGURE_PATH)
        result_report = {
            "schema_version": 1,
            "status": "pending_kuotunyu_output_review",
            "architecture": config["architecture"],
            "root_seed": int(config["root_seed"]),
            "n_cases": len(records),
            "input_manifest_sha256": manifest["input_manifest_sha256"],
            "model_repo_id": model_config["model"]["repo_id"],
            "model_revision": model_config["model"]["revision"],
            "records": records,
            "h4_auc_computed": False,
            "expanded_to_64": False,
        }
        _write_json(OUTPUT_ROOT / "manifest.json", result_report)
        _write_json(REPORT_PATH, result_report)
        print(json.dumps(result_report, indent=2, sort_keys=True))
    finally:
        if pipeline is not None:
            del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
