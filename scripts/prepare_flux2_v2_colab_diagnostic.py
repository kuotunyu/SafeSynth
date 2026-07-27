"""Export four train-only FLUX.2 inpaint inputs for a Colab method diagnostic."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import generate
from src.synthetic.generative_inpaint import (
    GenerativeBoundaryInpainter,
    InpaintResult,
    load_generative_config,
    reference_canvas,
)

ROOT_SEED = 20260727
CANONICAL_CELLS = (7, 13, 17, 52)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "flux2_v2_colab_diagnostic_inputs"
ARCHIVE_PATH = (
    PROJECT_ROOT / "outputs" / "flux2_v2_colab_diagnostic_inputs_portable.zip"
)


class IdentityPipeline:
    """Return the draft unchanged without loading or initializing a GPU model."""

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(images=[kwargs["image"]])


class RecordingInpainter:
    """Capture model-free production inputs while returning the normal result."""

    def __init__(
        self,
        engine: GenerativeBoundaryInpainter,
        config: dict[str, Any],
    ) -> None:
        self.engine = engine
        self.config = config
        self.by_seed: dict[int, dict[str, Any]] = {}

    def generate(
        self,
        *,
        draft_rgb: np.ndarray,
        object_mask: np.ndarray,
        reference_rgba: np.ndarray,
        class_name: str,
        seed: int,
    ) -> InpaintResult:
        result = self.engine.generate(
            draft_rgb=draft_rgb,
            object_mask=object_mask,
            reference_rgba=reference_rgba,
            class_name=class_name,
            seed=seed,
        )
        method = self.config["method"]
        reference = reference_canvas(
            reference_rgba,
            canvas_size=int(method["reference_canvas_size"]),
            max_fill=float(method["reference_max_fill"]),
            background_rgb=tuple(
                int(value) for value in method["reference_background_rgb"]
            ),
        )
        if int(seed) in self.by_seed:
            raise AssertionError(f"Duplicate diagnostic seed: {seed}")
        self.by_seed[int(seed)] = {
            "class_name": str(class_name),
            "draft_rgb": np.asarray(draft_rgb, dtype=np.uint8).copy(),
            "edit_mask": np.asarray(result.edit_mask, dtype=bool).copy(),
            "reference_rgb": reference,
        }
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if not len(xs):
        raise ValueError("Diagnostic edit mask is empty")
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def _render_preview(case_dirs: list[Path], output_path: Path) -> None:
    panel_size = 320
    caption_height = 30
    canvas = Image.new(
        "RGB",
        (panel_size * 3, (panel_size + caption_height) * len(case_dirs)),
        "white",
    )
    for row, case_dir in enumerate(case_dirs):
        draft = Image.open(case_dir / "draft.png").convert("RGB")
        mask = Image.open(case_dir / "edit_mask.png").convert("RGB")
        reference = Image.open(case_dir / "reference.png").convert("RGB")
        for column, (label, panel) in enumerate(
            (("DRAFT", draft), ("EDIT MASK", mask), ("REFERENCE", reference))
        ):
            resized = panel.resize(
                (panel_size, panel_size),
                Image.Resampling.NEAREST
                if label == "EDIT MASK"
                else Image.Resampling.LANCZOS,
            )
            ImageDraw.Draw(resized).rectangle((0, 0, 94, 21), fill="black")
            ImageDraw.Draw(resized).text((5, 4), label, fill="white")
            canvas.paste(
                resized,
                (column * panel_size, row * (panel_size + caption_height)),
            )
        metadata = json.loads(
            (case_dir / "metadata.json").read_text(encoding="utf-8")
        )
        ImageDraw.Draw(canvas).text(
            (5, row * (panel_size + caption_height) + panel_size + 7),
            (
                f"{case_dir.name} | {metadata['class_name']} | "
                f"{metadata['sample_id']}"
            ),
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def main() -> None:
    paths = load_project_paths()
    config = load_generative_config()
    recorder = RecordingInpainter(
        GenerativeBoundaryInpainter(IdentityPipeline(), config),
        config,
    )
    capture_tag = f"flux2_v2_colab_diagnostic_capture_seed{ROOT_SEED}"
    summary = generate(
        paths=paths,
        n=64,
        seed=ROOT_SEED,
        output_tag=capture_tag,
        selected_scenarios=["context_replacement"],
        draw_boxes=False,
        generative_inpainter=recorder,  # type: ignore[arg-type]
    )
    capture_dir = Path(str(summary["output_dir"]))
    records = [
        json.loads(line)
        for line in (capture_dir / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(records) != 64:
        raise AssertionError("Expected the frozen 64-sample geometry replay")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    case_dirs: list[Path] = []
    cases: list[dict[str, Any]] = []
    for cell in CANONICAL_CELLS:
        record = records[cell - 1]
        generated = [
            instance
            for instance in record["instances"]
            if "generative_inpaint" in instance
        ]
        if len(generated) != 1:
            raise AssertionError("Every diagnostic case must have one generated instance")
        instance = generated[0]
        seed = int(instance["generative_inpaint"]["seed"])
        evidence = recorder.by_seed[seed]
        if evidence["class_name"] != "helmet":
            raise AssertionError("The v2 diagnostic intentionally uses helmet cases only")

        case_name = f"case_{cell:02d}"
        case_dir = OUTPUT_ROOT / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(evidence["draft_rgb"]).save(case_dir / "draft.png")
        Image.fromarray(evidence["edit_mask"].astype(np.uint8) * 255).save(
            case_dir / "edit_mask.png"
        )
        evidence["reference_rgb"].save(case_dir / "reference.png")

        background_path = paths.hardhat_raw / str(record["background"]["file_name"])
        metadata = {
            "canonical_contact_sheet_cell": cell,
            "class_name": evidence["class_name"],
            "cutout_id": str(instance["cutout_id"]),
            "edit_mask_bbox_xywh": _mask_bbox(evidence["edit_mask"]),
            "generative_seed": seed,
            "replacement_anchor_annotation_id": int(
                record["replacement_anchor_annotation_id"]
            ),
            "sample_id": str(record["sample_id"]),
            "source_background_file": str(record["background"]["file_name"]),
            "source_background_image_id": int(record["background"]["image_id"]),
            "source_background_sha256": _sha256(background_path),
        }
        (case_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        case_files = {}
        for name in ("draft.png", "edit_mask.png", "reference.png", "metadata.json"):
            path = case_dir / name
            case_files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        cases.append({**metadata, "files": case_files})
        case_dirs.append(case_dir)

    manifest = {
        "cases": cases,
        "diagnostic_only": True,
        "final_h4_auc_computed": False,
        "model": {
            "repo_id": config["model"]["repo_id"],
            "revision": config["model"]["revision"],
            "diffusers_version": config["model"]["diffusers_version"],
            "license": config["model"]["license"],
        },
        "purpose": (
            "Method development only: compare reference conditioning and strength "
            "without changing or evaluating the frozen H4 gate."
        ),
        "root_seed": ROOT_SEED,
        "schema_version": 1,
        "source_split": "train_only",
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _render_preview(case_dirs, OUTPUT_ROOT / "input_preview.png")
    with zipfile.ZipFile(
        ARCHIVE_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT_ROOT).as_posix())
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE_PATH),
                "archive_sha256": _sha256(ARCHIVE_PATH),
                "cases": len(cases),
                "diagnostic_only": True,
                "h4_auc_computed": False,
                "output_root": str(OUTPUT_ROOT),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
