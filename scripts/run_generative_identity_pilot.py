"""Run the frozen 64-image visual identity gate without computing H4."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.data.paths import ProjectPaths, load_project_paths
from src.synthetic.compose import generate
from src.synthetic.generative_inpaint import (
    GenerativeBoundaryInpainter,
    InpaintResult,
    load_flux2_pipeline,
    load_generative_config,
    model_directory,
    reference_canvas,
)


@dataclass(frozen=True)
class PilotEvidence:
    """The four registered visual panels for one generative edit."""

    draft_rgb: np.ndarray
    edit_mask: np.ndarray
    reference_rgb: np.ndarray
    output_rgb: np.ndarray
    class_name: str
    seed: int


class RecordingInpainter:
    """Record model inputs and outputs while preserving the production engine."""

    def __init__(
        self,
        engine: Any,
        config: dict[str, Any],
    ) -> None:
        self.engine = engine
        self.config = config
        self.by_seed: dict[int, PilotEvidence] = {}

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
            raise AssertionError(f"Duplicate generative seed in pilot: {seed}")
        self.by_seed[int(seed)] = PilotEvidence(
            draft_rgb=np.asarray(draft_rgb, dtype=np.uint8).copy(),
            edit_mask=np.asarray(result.edit_mask, dtype=bool).copy(),
            reference_rgb=np.asarray(reference, dtype=np.uint8).copy(),
            output_rgb=np.asarray(result.image_rgb, dtype=np.uint8).copy(),
            class_name=str(class_name),
            seed=int(seed),
        )
        return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_input_preflight_approved(
    report_path: Path,
    config: dict[str, Any],
) -> None:
    """Block every GPU model call until kuotunyu approves the exact inputs."""

    if not report_path.exists():
        raise RuntimeError("GPU identity pilot locked: input preflight is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    pilot = config["pilot"]
    if (
        report.get("status") != "approved_by_kuotunyu"
        or report.get("reviewed_by") != "kuotunyu"
        or int(report.get("observed_input_issue_count", -1)) != 0
        or report.get("architecture") != pilot["architecture"]
        or int(report.get("root_seed", -1)) != int(pilot["root_seed"])
    ):
        raise RuntimeError(
            "GPU identity pilot locked: the exact input sheet is not approved"
        )


def _label_panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.convert("RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 58, 17), fill="black")
    draw.text((4, 2), label, fill="white")
    return panel


def _marked_image(image_rgb: np.ndarray, edit_mask: np.ndarray) -> Image.Image:
    panel = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    ys, xs = np.where(np.asarray(edit_mask, dtype=bool))
    if len(xs):
        draw = ImageDraw.Draw(panel)
        draw.rectangle(
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            outline=(0, 255, 255),
            width=2,
        )
    return panel


def _detail_crop_box(edit_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return a square context crop centred on the editable band."""

    mask = np.asarray(edit_mask, dtype=bool)
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Pilot edit mask is empty")
    height, width = mask.shape
    object_span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    side = min(max(64, object_span * 3), min(height, width))
    center_x = (int(xs.min()) + int(xs.max()) + 1) // 2
    center_y = (int(ys.min()) + int(ys.max()) + 1) // 2
    left = min(max(0, center_x - side // 2), width - side)
    top = min(max(0, center_y - side // 2), height - side)
    return left, top, left + side, top + side


def render_contact_sheet(
    *,
    records: list[dict[str, Any]],
    evidence_by_seed: dict[int, PilotEvidence],
    output_dir: Path,
    output_path: Path,
    rows: int,
    columns: int,
    crop_to_edit: bool = False,
) -> None:
    """Render draft, edit mask, reference, and final output for every item."""

    if len(records) != rows * columns:
        raise ValueError("Pilot record count does not match the registered grid")
    quadrant = 196
    caption_height = 28
    cell_width = quadrant * 2
    cell_height = quadrant * 2 + caption_height
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        "white",
    )
    draw_sheet = ImageDraw.Draw(sheet)
    resampling = Image.Resampling.LANCZOS

    for index, record in enumerate(records):
        generated_instances = [
            instance
            for instance in record["instances"]
            if "generative_inpaint" in instance
        ]
        if len(generated_instances) != 1:
            raise AssertionError(
                "Registered context-replacement pilot requires one generated "
                f"instance per image; got {len(generated_instances)}"
            )
        instance = generated_instances[0]
        seed = int(instance["generative_inpaint"]["seed"])
        evidence = evidence_by_seed[seed]
        final_path = output_dir / str(record["file_name"])
        final = np.asarray(Image.open(final_path).convert("RGB"), dtype=np.uint8)
        mask_rgb = np.zeros((*evidence.edit_mask.shape, 3), dtype=np.uint8)
        mask_rgb[evidence.edit_mask] = [0, 255, 255]
        draft_panel = _marked_image(evidence.draft_rgb, evidence.edit_mask)
        mask_panel = Image.fromarray(mask_rgb)
        output_panel = _marked_image(final, evidence.edit_mask)
        if crop_to_edit:
            crop_box = _detail_crop_box(evidence.edit_mask)
            draft_panel = draft_panel.crop(crop_box)
            mask_panel = mask_panel.crop(crop_box)
            output_panel = output_panel.crop(crop_box)

        panels = [
            _label_panel(draft_panel, "DRAFT"),
            _label_panel(mask_panel, "EDIT MASK"),
            _label_panel(Image.fromarray(evidence.reference_rgb), "REFERENCE"),
            _label_panel(output_panel, "OUTPUT"),
        ]
        panels = [
            panel.resize(
                (quadrant, quadrant),
                Image.Resampling.NEAREST if panel_index == 1 else resampling,
            )
            for panel_index, panel in enumerate(panels)
        ]
        cell = Image.new("RGB", (cell_width, cell_height), "white")
        cell.paste(panels[0], (0, 0))
        cell.paste(panels[1], (quadrant, 0))
        cell.paste(panels[2], (0, quadrant))
        cell.paste(panels[3], (quadrant, quadrant))
        status = "PASS" if record["passed"] else str(record["first_reject_reason"])
        ImageDraw.Draw(cell).text(
            (4, quadrant * 2 + 6),
            f"{index + 1:02d} | {evidence.class_name} | {status}",
            fill="black",
        )
        x0 = (index % columns) * cell_width
        y0 = (index // columns) * cell_height
        sheet.paste(cell, (x0, y0))
        draw_sheet.rectangle(
            (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1),
            outline=(50, 50, 50),
            width=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, optimize=True)


def _identity_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "generated_instances": 0,
        "outside_edit_changed_pixels": 0,
        "protected_core_changed_pixels": 0,
    }
    for record in records:
        for instance in record["instances"]:
            if "generative_inpaint" not in instance:
                continue
            metrics = instance["generative_inpaint"]["identity_metrics"]
            counts["generated_instances"] += 1
            counts["outside_edit_changed_pixels"] += int(
                metrics["outside_edit_changed_pixels"]
            )
            counts["protected_core_changed_pixels"] += int(
                metrics["protected_core_changed_pixels"]
            )
    return counts


def _write_report(
    *,
    paths: ProjectPaths,
    config: dict[str, Any],
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    contact_sheet_path: Path,
    detail_sheet_path: Path,
) -> None:
    counts = _identity_counts(records)
    if counts["generated_instances"] != int(config["pilot"]["n_images"]):
        raise AssertionError("Every pilot image must contain one generated instance")
    if (
        counts["outside_edit_changed_pixels"] != 0
        or counts["protected_core_changed_pixels"] != 0
    ):
        raise AssertionError("Pilot violated a pixel-exact identity invariant")

    payload = {
        "schema_version": 1,
        "status": "pending_kuotunyu_visual_review",
        "h4_auc_computed": False,
        "root_seed": int(config["pilot"]["root_seed"]),
        "n_images": int(config["pilot"]["n_images"]),
        "scenario": "context_replacement",
        "contact_sheet": str(contact_sheet_path),
        "contact_sheet_sha256": _sha256(contact_sheet_path),
        "detail_contact_sheet": str(detail_sheet_path),
        "detail_contact_sheet_sha256": _sha256(detail_sheet_path),
        "evidence_alignment": "inline_or_deterministic_geometry_verified",
        "identity_counts": counts,
        "identity_metric_scope": "immediate_inpaint_result_before_global_postfx",
        "automated_filter_passed": int(summary["passed"]),
        "automated_filter_rejected": int(summary["rejected"]),
        "human_gate": {
            "label_mismatch_max_count": int(
                config["pilot"]["label_mismatch_max_count"]
            ),
            "severe_identity_failure_max_count": int(
                config["pilot"]["severe_identity_failure_max_count"]
            ),
            "observed_label_mismatch_count": None,
            "observed_severe_identity_failure_count": None,
        },
    }
    json_path = paths.reports / "h4_generative_identity_pilot.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    markdown = [
        "# H4 generative identity pilot",
        "",
        "- Status: **pending kuotunyu visual review**",
        f"- Images: **{payload['n_images']}**",
        "- Scenario: `context_replacement`",
        f"- Root seed: `{payload['root_seed']}`",
        "- H4 AUC computed: **no**",
        (
            "- Outside-edit changed pixels: "
            f"**{counts['outside_edit_changed_pixels']}**"
        ),
        (
            "- Protected-core changed pixels: "
            f"**{counts['protected_core_changed_pixels']}**"
        ),
        "- Pixel-invariant scope: immediate inpaint result before global postfx",
        (
            "- Human gate: 0 label mismatches and at most "
            f"{payload['human_gate']['severe_identity_failure_max_count']} "
            "severe identity failures."
        ),
        f"- Contact sheet: `{_repo_relative(contact_sheet_path)}`",
        f"- Detail contact sheet: `{_repo_relative(detail_sheet_path)}`",
        "",
        "Each numbered cell is ordered DRAFT / EDIT MASK / REFERENCE / OUTPUT.",
        "The cyan rectangle marks the registered editable boundary band.",
        "The detail sheet crops DRAFT, EDIT MASK, and OUTPUT around that band.",
        "",
    ]
    (paths.reports / "h4_generative_identity_pilot.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-details-from-existing",
        action="store_true",
        help="replay composition without model inference and render detail evidence",
    )
    return parser.parse_args()


class _IdentityPipeline:
    """Return the draft unchanged while the production engine records evidence."""

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(images=[kwargs["image"]])


def _render_details_from_existing(
    *,
    paths: ProjectPaths,
    config: dict[str, Any],
    n_images: int,
    root_seed: int,
    rows: int,
    columns: int,
) -> None:
    recorder = RecordingInpainter(
        GenerativeBoundaryInpainter(_IdentityPipeline(), config),
        config,
    )
    capture_tag = f"h4_generative_identity_capture_seed{root_seed}"
    generate(
        paths=paths,
        n=n_images,
        seed=root_seed,
        output_tag=capture_tag,
        selected_scenarios=["context_replacement"],
        draw_boxes=False,
        generative_inpainter=recorder,  # type: ignore[arg-type]
    )
    real_output_dir = (
        paths.synthetic / f"h4_generative_identity_pilot_seed{root_seed}"
    )
    records = _read_jsonl(real_output_dir / "records.jsonl")
    captured_records = _read_jsonl(
        paths.synthetic / capture_tag / "records.jsonl"
    )
    real_keys = [
        (
            record["sample_id"],
            record["background"]["image_id"],
            next(
                instance["cutout_id"]
                for instance in record["instances"]
                if "generative_inpaint" in instance
            ),
        )
        for record in records
    ]
    captured_keys = [
        (
            record["sample_id"],
            record["background"]["image_id"],
            next(
                instance["cutout_id"]
                for instance in record["instances"]
                if "generative_inpaint" in instance
            ),
        )
        for record in captured_records
    ]
    if real_keys != captured_keys:
        raise AssertionError("Capture-only replay did not reproduce pilot geometry")

    full_sheet = paths.figures / "h4_generative_identity_pilot.png"
    detail_sheet = paths.figures / "h4_generative_identity_pilot_detail.png"
    render_contact_sheet(
        records=records,
        evidence_by_seed=recorder.by_seed,
        output_dir=real_output_dir,
        output_path=full_sheet,
        rows=rows,
        columns=columns,
    )
    render_contact_sheet(
        records=records,
        evidence_by_seed=recorder.by_seed,
        output_dir=real_output_dir,
        output_path=detail_sheet,
        rows=rows,
        columns=columns,
        crop_to_edit=True,
    )
    summary = json.loads(
        (real_output_dir / "summary.json").read_text(encoding="utf-8")
    )
    _write_report(
        paths=paths,
        config=config,
        summary=summary,
        records=records,
        contact_sheet_path=full_sheet,
        detail_sheet_path=detail_sheet,
    )
    print(
        json.dumps(
            {
                "capture_geometry_verified": True,
                "contact_sheet": str(full_sheet),
                "detail_contact_sheet": str(detail_sheet),
                "h4_auc_computed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = _parse_args()
    config = load_generative_config()
    paths = load_project_paths()
    n_images = int(config["pilot"]["n_images"])
    root_seed = int(config["pilot"]["root_seed"])
    rows = int(config["pilot"]["contact_sheet_rows"])
    columns = int(config["pilot"]["contact_sheet_columns"])
    if n_images != 64 or rows != 8 or columns != 8:
        raise RuntimeError("The registered identity pilot must remain an 8x8 grid")
    if args.render_details_from_existing:
        _render_details_from_existing(
            paths=paths,
            config=config,
            n_images=n_images,
            root_seed=root_seed,
            rows=rows,
            columns=columns,
        )
        return

    _require_input_preflight_approved(
        paths.reports / "h4_guarded_input_preflight.json",
        config,
    )
    engine = GenerativeBoundaryInpainter(
        load_flux2_pipeline(
            model_dir=model_directory(paths, config),
            config=config,
        ),
        config,
    )
    recorder = RecordingInpainter(engine, config)
    output_tag = f"h4_generative_identity_pilot_seed{root_seed}"
    summary = generate(
        paths=paths,
        n=n_images,
        seed=root_seed,
        output_tag=output_tag,
        selected_scenarios=["context_replacement"],
        draw_boxes=False,
        generative_inpainter=recorder,  # type: ignore[arg-type]
    )
    output_dir = Path(str(summary["output_dir"]))
    records = _read_jsonl(output_dir / "records.jsonl")
    contact_sheet = paths.figures / "h4_generative_identity_pilot.png"
    detail_sheet = paths.figures / "h4_generative_identity_pilot_detail.png"
    render_contact_sheet(
        records=records,
        evidence_by_seed=recorder.by_seed,
        output_dir=output_dir,
        output_path=contact_sheet,
        rows=rows,
        columns=columns,
    )
    render_contact_sheet(
        records=records,
        evidence_by_seed=recorder.by_seed,
        output_dir=output_dir,
        output_path=detail_sheet,
        rows=rows,
        columns=columns,
        crop_to_edit=True,
    )
    _write_report(
        paths=paths,
        config=config,
        summary=summary,
        records=records,
        contact_sheet_path=contact_sheet,
        detail_sheet_path=detail_sheet,
    )
    print(
        json.dumps(
            {
                "contact_sheet": str(contact_sheet),
                "h4_auc_computed": False,
                "n_images": n_images,
                "output_dir": str(output_dir),
                "status": "pending_kuotunyu_visual_review",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


def _repo_relative(path) -> str:
    """Render a path as repo-relative POSIX for inclusion in a report.

    Absolute paths embed the local username, which publish-repo gate 1 rejects,
    and they are meaningless to anyone who clones the repository.
    """
    from pathlib import Path as _Path

    candidate = _Path(path)
    try:
        candidate = candidate.resolve().relative_to(_Path(__file__).resolve().parents[1])
    except ValueError:
        pass
    return candidate.as_posix()
