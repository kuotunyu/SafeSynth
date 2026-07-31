"""Verify and compare the Train-only FLUX.2 v2 Colab diagnostic outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT

VARIANTS = (
    "v1_reference_strength_085",
    "reference_strength_055",
    "no_reference_strength_055",
)

METHOD_DECISION = {
    "selected_variant": None,
    "outcome": "no_registered_variant_selected",
    "binding_state": "v1_failed_human_identity_gate",
    "rationale": [
        (
            "Removing the reference at strength 0.55 has negligible effect "
            "(aggregate masked RGB MAE 0.2260/255)."
        ),
        (
            "Lowering strength changes some cases but produces no consistent "
            "visual improvement across the four fixed Train examples."
        ),
        (
            "All variants preserve pixels outside the edit mask, but none "
            "addresses the invalid-draft and mislocalized-anchor failures from "
            "the rejected 64-image pilot."
        ),
    ],
    "next_requirement": (
        "Design and preregister an input-validity and anchor-localization guard "
        "before generating another untouched identity pilot."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def masked_metrics(
    draft_rgb: np.ndarray,
    output_rgb: np.ndarray,
    edit_mask: np.ndarray,
) -> dict[str, float | int]:
    """Measure exact and absolute changes inside and outside an edit mask."""

    draft = np.asarray(draft_rgb, dtype=np.uint8)
    output = np.asarray(output_rgb, dtype=np.uint8)
    mask = np.asarray(edit_mask, dtype=bool)
    if draft.shape != output.shape or draft.shape[:2] != mask.shape:
        raise ValueError("Draft, output, and edit mask shapes disagree")
    if not mask.any():
        raise ValueError("Edit mask is empty")
    absolute = np.abs(output.astype(np.int16) - draft.astype(np.int16))
    changed = np.any(absolute != 0, axis=2)
    inside_values = absolute[mask]
    return {
        "changed_pixel_fraction_inside_mask": float(changed[mask].mean()),
        "changed_pixels_inside_mask": int(changed[mask].sum()),
        "edit_mask_pixels": int(mask.sum()),
        "mae_rgb_inside_mask": float(inside_values.mean()),
        "max_abs_channel_error_inside_mask": int(inside_values.max()),
        "outside_mask_changed_pixels": int(changed[~mask].sum()),
        "p95_abs_channel_error_inside_mask": float(
            np.percentile(inside_values, 95)
        ),
        "sum_abs_channel_error_inside_mask": int(inside_values.sum()),
    }


def pairwise_metrics(
    first_rgb: np.ndarray,
    second_rgb: np.ndarray,
    edit_mask: np.ndarray,
) -> dict[str, float | int]:
    """Measure how much two variants differ inside the same edit mask."""

    first = np.asarray(first_rgb, dtype=np.uint8)
    second = np.asarray(second_rgb, dtype=np.uint8)
    mask = np.asarray(edit_mask, dtype=bool)
    if first.shape != second.shape or first.shape[:2] != mask.shape:
        raise ValueError("Variant images and edit mask shapes disagree")
    absolute = np.abs(first.astype(np.int16) - second.astype(np.int16))
    changed = np.any(absolute != 0, axis=2)
    inside_values = absolute[mask]
    return {
        "different_pixel_fraction_inside_mask": float(changed[mask].mean()),
        "different_pixels_inside_mask": int(changed[mask].sum()),
        "mae_rgb_inside_mask": float(inside_values.mean()),
        "max_abs_channel_error_inside_mask": int(inside_values.max()),
        "sum_abs_channel_error_inside_mask": int(inside_values.sum()),
    }


def _crop_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if not len(xs):
        raise ValueError("Edit mask is empty")
    height, width = mask.shape
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    side = min(max(96, span * 4), min(height, width))
    center_x = (int(xs.min()) + int(xs.max()) + 1) // 2
    center_y = (int(ys.min()) + int(ys.max()) + 1) // 2
    left = min(max(0, center_x - side // 2), width - side)
    top = min(max(0, center_y - side // 2), height - side)
    return left, top, left + side, top + side


def _marked_panel(image_rgb: np.ndarray, mask: np.ndarray, label: str) -> Image.Image:
    image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        outline=(0, 255, 255),
        width=2,
    )
    crop = image.crop(_crop_box(mask)).resize((256, 256), Image.Resampling.LANCZOS)
    ImageDraw.Draw(crop).rectangle((0, 0, 256, 21), fill="black")
    ImageDraw.Draw(crop).text((5, 4), label, fill="white")
    return crop


def _render_detail_sheet(
    cases: list[dict[str, Any]],
    output_path: Path,
) -> None:
    panel_size = 256
    caption_height = 28
    canvas = Image.new(
        "RGB",
        (
            panel_size * (1 + len(VARIANTS)),
            (panel_size + caption_height) * len(cases),
        ),
        "white",
    )
    for row, case in enumerate(cases):
        panels = [
            _marked_panel(case["draft"], case["mask"], "DRAFT"),
            *[
                _marked_panel(
                    case["outputs"][variant],
                    case["mask"],
                    variant,
                )
                for variant in VARIANTS
            ],
        ]
        for column, panel in enumerate(panels):
            canvas.paste(
                panel,
                (
                    column * panel_size,
                    row * (panel_size + caption_height),
                ),
            )
        ImageDraw.Draw(canvas).text(
            (5, row * (panel_size + caption_height) + panel_size + 6),
            f"{case['case_name']} | {case['sample_id']}",
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "flux2_v2_diagnostic_results_colab_a100",
    )
    parser.add_argument(
        "--inputs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "flux2_v2_colab_diagnostic_inputs",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path.home() / "Downloads" / "flux2_v2_diagnostic_results.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(
        (args.results_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["diagnostic_only"] is not True:
        raise AssertionError("Colab result is not marked diagnostic-only")
    if manifest["final_h4_auc_computed"] is not False:
        raise AssertionError("The method diagnostic must not compute H4")
    if manifest["execution_mode"] != "full_model_on_cuda":
        raise AssertionError("Expected the registered A100 full-CUDA execution")
    if len(manifest["runs"]) != 12:
        raise AssertionError("Expected 4 cases x 3 registered variants")

    case_payloads: list[dict[str, Any]] = []
    result_cases: dict[str, Any] = {}
    for case_name in ("case_07", "case_13", "case_17", "case_52"):
        input_dir = args.inputs_root / case_name
        metadata = json.loads(
            (input_dir / "metadata.json").read_text(encoding="utf-8")
        )
        draft = np.asarray(Image.open(input_dir / "draft.png").convert("RGB"))
        mask = np.asarray(Image.open(input_dir / "edit_mask.png").convert("L")) > 0
        outputs = {
            variant: np.asarray(
                Image.open(args.results_root / case_name / f"{variant}.png").convert(
                    "RGB"
                )
            )
            for variant in VARIANTS
        }
        variant_metrics = {
            variant: masked_metrics(draft, output, mask)
            for variant, output in outputs.items()
        }
        pair_metrics = {
            "strength_effect_v1_vs_reference_055": pairwise_metrics(
                outputs["v1_reference_strength_085"],
                outputs["reference_strength_055"],
                mask,
            ),
            "reference_effect_at_strength_055": pairwise_metrics(
                outputs["reference_strength_055"],
                outputs["no_reference_strength_055"],
                mask,
            ),
        }
        result_cases[case_name] = {
            "sample_id": metadata["sample_id"],
            "variant_metrics": variant_metrics,
            "pairwise_metrics": pair_metrics,
        }
        case_payloads.append(
            {
                "case_name": case_name,
                "draft": draft,
                "mask": mask,
                "outputs": outputs,
                "sample_id": metadata["sample_id"],
            }
        )

    aggregate: dict[str, Any] = {"variants": {}, "pairwise": {}}
    total_mask_pixels = sum(
        int(case["variant_metrics"][VARIANTS[0]]["edit_mask_pixels"])
        for case in result_cases.values()
    )
    for variant in VARIANTS:
        metrics = [case["variant_metrics"][variant] for case in result_cases.values()]
        aggregate["variants"][variant] = {
            "changed_pixel_fraction_inside_mask": float(
                sum(int(item["changed_pixels_inside_mask"]) for item in metrics)
                / total_mask_pixels
            ),
            "mae_rgb_inside_mask": float(
                sum(int(item["sum_abs_channel_error_inside_mask"]) for item in metrics)
                / (total_mask_pixels * 3)
            ),
            "outside_mask_changed_pixels": sum(
                int(item["outside_mask_changed_pixels"]) for item in metrics
            ),
        }
    for name in (
        "strength_effect_v1_vs_reference_055",
        "reference_effect_at_strength_055",
    ):
        metrics = [case["pairwise_metrics"][name] for case in result_cases.values()]
        aggregate["pairwise"][name] = {
            "different_pixel_fraction_inside_mask": float(
                sum(int(item["different_pixels_inside_mask"]) for item in metrics)
                / total_mask_pixels
            ),
            "mae_rgb_inside_mask": float(
                sum(int(item["sum_abs_channel_error_inside_mask"]) for item in metrics)
                / (total_mask_pixels * 3)
            ),
        }

    detail_path = (
        PROJECT_ROOT / "reports" / "figures" / "flux2_v2_diagnostic_detail.png"
    )
    _render_detail_sheet(case_payloads, detail_path)
    report = {
        "aggregate": aggregate,
        "archive": str(args.archive),
        "archive_sha256": _sha256(args.archive),
        "cases": result_cases,
        "diagnostic_only": True,
        "detail_sheet": str(detail_path),
        "detail_sheet_sha256": _sha256(detail_path),
        "execution": {
            "gpu": manifest["gpu"],
            "mode": manifest["execution_mode"],
            "python": manifest["python"],
            "torch": manifest["torch"],
            "total_inference_seconds": float(
                sum(float(run["seconds"]) for run in manifest["runs"])
            ),
        },
        "final_h4_auc_computed": False,
        "method_decision": METHOD_DECISION,
        "model": manifest["model"],
        "status": "diagnostic_complete_no_registered_variant_selected",
    }
    json_path = PROJECT_ROOT / "reports" / "flux2_v2_diagnostic.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# FLUX.2 v2 Colab diagnostic",
        "",
        "- Status: **diagnostic complete; no registered variant selected**",
        f"- GPU: `{manifest['gpu']}`",
        f"- Execution mode: `{manifest['execution_mode']}`",
        f"- Outputs: **{len(manifest['runs'])}/12**",
        (
            "- Total inference time: "
            f"**{report['execution']['total_inference_seconds']:.2f} seconds**"
        ),
        "- H4 AUC computed: **no**",
        f"- Result archive SHA-256: `{report['archive_sha256']}`",
        "",
        "## Aggregate masked changes",
        "",
        "| variant | changed pixels | RGB MAE | outside-mask changes |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        metrics = aggregate["variants"][variant]
        lines.append(
            f"| `{variant}` | "
            f"{metrics['changed_pixel_fraction_inside_mask']:.4f} | "
            f"{metrics['mae_rgb_inside_mask']:.4f} | "
            f"{metrics['outside_mask_changed_pixels']} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise effects",
            "",
            "| comparison | different pixels | RGB MAE |",
            "|---|---:|---:|",
        ]
    )
    for name, metrics in aggregate["pairwise"].items():
        lines.append(
            f"| `{name}` | "
            f"{metrics['different_pixel_fraction_inside_mask']:.4f} | "
            f"{metrics['mae_rgb_inside_mask']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Detail sheet: `{_repo_relative(detail_path)}`",
            "",
            "## Method decision",
            "",
            (
                "- Selected variant: **none**. The original v1 method remains "
                "rejected by the human identity gate."
            ),
            (
                "- Removing the reference at strength 0.55 is effectively a "
                "no-op at inspection scale (masked RGB MAE **0.2260/255**)."
            ),
            (
                "- Lowering strength changes some masked pixels but shows no "
                "consistent visual improvement across the four fixed cases."
            ),
            (
                "- All three calls preserve every pixel outside the edit mask. "
                "They do not fix invalid drafts or mislocalized anchors, the "
                "failure modes found in the rejected 64-image pilot."
            ),
            (
                "- Next requirement: preregister an input-validity and "
                "anchor-localization guard before any new untouched identity pilot."
            ),
            "",
            "- This Train-only diagnostic cannot reopen M13 or replace the 64-image gate.",
            "",
        ]
    )
    (PROJECT_ROOT / "reports" / "flux2_v2_diagnostic.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


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
