"""Freeze and render the guarded 64-input pilot plan without model inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import generate


@dataclass(frozen=True)
class InputEvidence:
    """One exact draft and target mask captured before generative inference."""

    draft_rgb: np.ndarray
    object_mask: np.ndarray
    class_name: str
    seed: int


class CpuInputCapture:
    """Capture the registered model inputs while returning the draft unchanged."""

    def __init__(self) -> None:
        self.by_seed: dict[int, InputEvidence] = {}

    def generate(
        self,
        *,
        draft_rgb: np.ndarray,
        object_mask: np.ndarray,
        reference_rgba: np.ndarray,
        class_name: str,
        seed: int,
    ) -> SimpleNamespace:
        del reference_rgba
        if seed in self.by_seed:
            raise AssertionError(f"Duplicate guarded-input seed: {seed}")
        draft = np.asarray(draft_rgb, dtype=np.uint8).copy()
        mask = np.asarray(object_mask, dtype=bool).copy()
        self.by_seed[seed] = InputEvidence(
            draft_rgb=draft,
            object_mask=mask,
            class_name=str(class_name),
            seed=int(seed),
        )
        return SimpleNamespace(
            image_rgb=draft,
            provenance={
                "method": "cpu_guarded_input_capture",
                "seed": int(seed),
                "class_name": str(class_name),
                "identity_metrics": {
                    "outside_edit_changed_pixels": 0,
                    "protected_core_changed_pixels": 0,
                    "edit_mask_changed_fraction": 0.0,
                },
            },
        )


def _repo_relative(path) -> str:
    """Repo-relative POSIX path; absolute paths leak the local username."""

    from pathlib import Path as _Path

    candidate = _Path(path)
    try:
        return (
            candidate.resolve()
            .relative_to(_Path(__file__).resolve().parents[1])
            .as_posix()
        )
    except ValueError:
        return candidate.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _crop_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if not len(xs):
        raise ValueError("Guarded input object mask is empty")
    height, width = mask.shape
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    side = min(max(80, span * 4), min(height, width))
    center_x = (int(xs.min()) + int(xs.max()) + 1) // 2
    center_y = (int(ys.min()) + int(ys.max()) + 1) // 2
    left = min(max(0, center_x - side // 2), width - side)
    top = min(max(0, center_y - side // 2), height - side)
    return left, top, left + side, top + side


def _marked(image_rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        outline=(0, 255, 255),
        width=3,
    )
    return image


def render_input_sheet(
    *,
    records: list[dict[str, Any]],
    evidence_by_seed: dict[int, InputEvidence],
    output_path: Path,
) -> None:
    """Render each full draft beside an enlarged cyan-marked anchor crop."""

    if len(records) != 64:
        raise ValueError("Guarded input preflight must contain exactly 64 records")
    panel = 196
    caption = 30
    cell_width = panel * 2
    cell_height = panel + caption
    sheet = Image.new("RGB", (cell_width * 8, cell_height * 8), "white")
    for index, record in enumerate(records, start=1):
        generated = [
            instance
            for instance in record["instances"]
            if "generative_inpaint" in instance
        ]
        if len(generated) != 1:
            raise AssertionError("Each guarded input must contain one target")
        seed = int(generated[0]["generative_inpaint"]["seed"])
        evidence = evidence_by_seed[seed]
        marked = _marked(evidence.draft_rgb, evidence.object_mask)
        full = marked.resize((panel, panel), Image.Resampling.LANCZOS)
        crop = marked.crop(_crop_box(evidence.object_mask)).resize(
            (panel, panel),
            Image.Resampling.LANCZOS,
        )
        x0 = ((index - 1) % 8) * cell_width
        y0 = ((index - 1) // 8) * cell_height
        sheet.paste(full, (x0, y0))
        sheet.paste(crop, (x0 + panel, y0))
        guard = record["context_replacement_input_guard"]
        label = (
            f"{index:02d} | {record['sample_id']} | "
            f"margin {guard['selected_anchor_edge_margin_px']:.0f}/"
            f"{guard['selected_anchor_required_edge_margin_px']}"
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((x0 + 4, y0 + panel + 6), label, fill="black")
        draw.rectangle(
            (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
            outline=(60, 60, 60),
            width=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)


def _geometry_fingerprint(records: list[dict[str, Any]]) -> str:
    plan = []
    for record in records:
        generated = next(
            instance
            for instance in record["instances"]
            if "generative_inpaint" in instance
        )
        plan.append(
            {
                "sample_id": record["sample_id"],
                "background_image_id": record["background"]["image_id"],
                "replacement_anchor_annotation_id": (
                    record["replacement_anchor_annotation_id"]
                ),
                "cutout_id": generated["cutout_id"],
                "bbox_xywh": generated["bbox_xywh"],
                "generative_seed": generated["generative_inpaint"]["seed"],
            }
        )
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "generative_inpaint.yaml").read_text(
            encoding="utf-8"
        )
    )
    pilot = config["pilot"]
    if (
        config["status"] != "guarded_v5_preregistered_no_output"
        or pilot["architecture"] != "guarded_context_replacement_v5"
        or int(pilot["n_images"]) != 64
        or int(pilot["root_seed"]) != 20260731
        or int(pilot["input_preflight_issue_max_count"]) != 0
    ):
        raise RuntimeError("Guarded input preflight configuration is not frozen")

    paths = load_project_paths()
    capture = CpuInputCapture()
    output_tag = f"h4_guarded_input_preflight_seed{int(pilot['root_seed'])}"
    summary = generate(
        paths=paths,
        n=int(pilot["n_images"]),
        seed=int(pilot["root_seed"]),
        output_tag=output_tag,
        selected_scenarios=["context_replacement"],
        draw_boxes=False,
        generative_inpainter=capture,  # type: ignore[arg-type]
    )
    output_dir = Path(str(summary["output_dir"]))
    records = _read_jsonl(output_dir / "records.jsonl")
    if len(capture.by_seed) != 64 or len(records) != 64:
        raise AssertionError("Guarded input capture is incomplete")
    if any("context_replacement_input_guard" not in record for record in records):
        raise AssertionError("A guarded input is missing provenance")

    sheet_path = paths.figures / "h4_guarded_input_preflight.png"
    render_input_sheet(
        records=records,
        evidence_by_seed=capture.by_seed,
        output_path=sheet_path,
    )
    payload = {
        "schema_version": 1,
        "status": "pending_kuotunyu_input_review",
        "architecture": pilot["architecture"],
        "root_seed": int(pilot["root_seed"]),
        "n_images": len(records),
        "model_inference_run": False,
        "h4_auc_computed": False,
        "input_issue_max_count": int(pilot["input_preflight_issue_max_count"]),
        "observed_input_issue_count": None,
        "reviewed_by": None,
        "geometry_fingerprint_sha256": _geometry_fingerprint(records),
        "contact_sheet": str(sheet_path),
        "contact_sheet_sha256": _sha256(sheet_path),
        "output_dir": str(output_dir),
    }
    json_path = paths.reports / "h4_guarded_input_preflight.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = [
        "# H4 guarded-input preflight",
        "",
        "- Status: **pending kuotunyu input review**",
        f"- Inputs: **{payload['n_images']}**",
        f"- Root seed: `{payload['root_seed']}`",
        "- Model inference run: **no**",
        "- H4 AUC computed: **no**",
        f"- Allowed input issues: **{payload['input_issue_max_count']}**",
        (
            "- Geometry fingerprint: "
            f"`{payload['geometry_fingerprint_sha256']}`"
        ),
        f"- Contact sheet: `{_repo_relative(sheet_path)}`",
        "",
        (
            "Each numbered cell shows the full DRAFT on the left and an enlarged "
            "anchor crop on the right. The cyan rectangle marks the exact object "
            "support that will receive boundary inpainting."
        ),
        "",
        (
            "Approval requires zero invalid drafts and zero misplaced cyan boxes. "
            "This approval only unlocks GPU inference for the same frozen inputs; "
            "it does not pass the later output identity gate or H4."
        ),
        "",
    ]
    (paths.reports / "h4_guarded_input_preflight.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
