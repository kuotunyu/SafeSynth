"""Reference-conditioned generative boundary inpainting for Option A."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

from src.data.paths import PROJECT_ROOT, ProjectPaths

CONFIG_PATH = PROJECT_ROOT / "configs" / "generative_inpaint.yaml"


class InpaintPipeline(Protocol):
    """Minimal Diffusers-compatible interface used by the testable engine."""

    def __call__(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class InpaintResult:
    """A generated image plus identity and reproducibility evidence."""

    image_rgb: np.ndarray
    edit_mask: np.ndarray
    protected_core: np.ndarray
    provenance: dict[str, Any]


def load_generative_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the frozen generative-inpaint configuration."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in {path}")
    model = config["model"]
    if model["license"] != "apache-2.0":
        raise RuntimeError("The registered model must remain Apache-2.0")
    if model["pipeline_class"] != "Flux2KleinInpaintPipeline":
        raise RuntimeError("The registered inpaint pipeline changed")
    if int(model["required_download_bytes"]) <= 2 * 1024**3:
        raise RuntimeError("The model must remain behind explicit large-download approval")
    if config["runtime"]["local_files_only"] is not True:
        raise RuntimeError("Generative runtime must be local-only")
    if float(config["final_h4"]["max_auc_for_scaleup"]) != 0.60:
        raise RuntimeError("Option A may not weaken the registered H4 threshold")
    pilot = config["pilot"]
    if (
        config["status"] != "guarded_v2_preregistered_no_output"
        or pilot["architecture"] != "guarded_context_replacement_v2"
        or int(pilot["root_seed"]) != 20260728
        or int(pilot["previous_failed_root_seed"]) != 20260727
    ):
        raise RuntimeError("The guarded v2 identity pilot must remain preregistered")
    return config


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = 2 * int(radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def boundary_edit_mask(
    object_mask: np.ndarray,
    *,
    outer_dilate_px: int,
    protected_core_erode_px: int,
    minimum_protected_core_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a boundary edit band and an identity-preserving object core."""

    support = np.asarray(object_mask, dtype=bool)
    area = int(support.sum())
    if support.ndim != 2 or area == 0:
        raise ValueError("object_mask must be a non-empty 2D mask")
    if not 0 < minimum_protected_core_fraction <= 1:
        raise ValueError("minimum_protected_core_fraction must be in (0, 1]")

    outer = cv2.dilate(
        support.astype(np.uint8), _ellipse_kernel(outer_dilate_px)
    ).astype(bool)
    core = cv2.erode(
        support.astype(np.uint8), _ellipse_kernel(protected_core_erode_px)
    ).astype(bool)

    minimum_core_pixels = max(
        1, int(np.ceil(area * float(minimum_protected_core_fraction)))
    )
    if int(core.sum()) < minimum_core_pixels:
        distances = distance_transform_edt(support)
        ranked = np.argsort(distances.ravel(), kind="stable")[::-1]
        support_flat = support.ravel()
        selected = [index for index in ranked if support_flat[index]][
            :minimum_core_pixels
        ]
        core = np.zeros_like(support)
        core.ravel()[selected] = True

    edit_mask = outer & ~core
    if not edit_mask.any():
        raise RuntimeError("Registered boundary mask produced no editable pixels")
    return edit_mask, core


def reference_canvas(
    rgba: np.ndarray,
    *,
    canvas_size: int,
    max_fill: float,
    background_rgb: tuple[int, int, int],
) -> Image.Image:
    """Place a cutout on a neutral square canvas for reference conditioning."""

    source = np.asarray(rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("rgba must have shape HxWx4")
    if not source[..., 3].any():
        raise ValueError("rgba alpha channel is empty")
    if not 0 < max_fill <= 1:
        raise ValueError("max_fill must be in (0, 1]")

    height, width = source.shape[:2]
    target = max(1, round(canvas_size * max_fill))
    scale = min(target / width, target / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized_rgb = cv2.resize(
        source[..., :3],
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    resized_alpha = cv2.resize(
        source[..., 3],
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.empty((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[...] = np.asarray(background_rgb, dtype=np.uint8)
    left = (canvas_size - resized_width) // 2
    top = (canvas_size - resized_height) // 2
    alpha = resized_alpha.astype(np.float32)[..., None] / 255
    target_rgb = canvas[top : top + resized_height, left : left + resized_width]
    canvas[top : top + resized_height, left : left + resized_width] = np.clip(
        resized_rgb.astype(np.float32) * alpha
        + target_rgb.astype(np.float32) * (1 - alpha),
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(canvas)


def enforce_identity_regions(
    draft_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    *,
    edit_mask: np.ndarray,
    protected_core: np.ndarray,
) -> np.ndarray:
    """Keep model pixels only in the registered edit band."""

    draft = np.asarray(draft_rgb, dtype=np.uint8)
    generated = np.asarray(generated_rgb, dtype=np.uint8)
    editable = np.asarray(edit_mask, dtype=bool)
    core = np.asarray(protected_core, dtype=bool)
    if draft.shape != generated.shape or draft.shape[:2] != editable.shape:
        raise ValueError("Draft, generated image, and edit mask shapes disagree")
    if editable.shape != core.shape or np.any(editable & core):
        raise ValueError("Edit mask and protected core must be disjoint")

    output = draft.copy()
    output[editable] = generated[editable]
    return output


def identity_metrics(
    draft_rgb: np.ndarray,
    output_rgb: np.ndarray,
    *,
    edit_mask: np.ndarray,
    protected_core: np.ndarray,
) -> dict[str, float | int]:
    """Measure the two pixel-exact identity invariants and edit activity."""

    draft = np.asarray(draft_rgb, dtype=np.uint8)
    output = np.asarray(output_rgb, dtype=np.uint8)
    editable = np.asarray(edit_mask, dtype=bool)
    core = np.asarray(protected_core, dtype=bool)
    changed = np.any(draft != output, axis=2)
    outside = ~editable
    return {
        "outside_edit_changed_pixels": int((changed & outside).sum()),
        "protected_core_changed_pixels": int((changed & core).sum()),
        "edit_mask_changed_fraction": float(
            (changed & editable).sum() / max(int(editable.sum()), 1)
        ),
    }


def model_directory(paths: ProjectPaths, config: dict[str, Any]) -> Path:
    """Return the project-isolated model directory on the bulk-data drive."""

    slug = str(config["model"]["repo_id"]).replace("/", "--")
    return paths.cache / "models" / slug / str(config["model"]["revision"])


def require_verified_model(model_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Hard fail unless a post-download manifest matches the frozen model."""

    manifest_path = model_dir / "SAFESYNTH_MODEL_MANIFEST.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "Registered model is not downloaded and verified. "
            "Run the approved model preflight download first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config["model"]
    if (
        manifest.get("repo_id") != expected["repo_id"]
        or manifest.get("revision") != expected["revision"]
        or manifest.get("license") != expected["license"]
        or int(manifest.get("download_bytes", -1))
        != int(expected["required_download_bytes"])
    ):
        raise RuntimeError("Local model manifest does not match preregistration")
    return manifest


def load_flux2_pipeline(
    *,
    model_dir: Path,
    config: dict[str, Any],
) -> InpaintPipeline:
    """Load the pinned model without allowing any implicit network download."""

    require_verified_model(model_dir, config)
    import diffusers
    from diffusers import Flux2KleinInpaintPipeline

    if diffusers.__version__ != str(config["model"]["diffusers_version"]):
        raise RuntimeError(
            f"Expected diffusers {config['model']['diffusers_version']}, "
            f"got {diffusers.__version__}"
        )
    dtype_name = str(config["runtime"]["dtype"])
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    pipeline = Flux2KleinInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        local_files_only=True,
    )
    if config["runtime"]["model_cpu_offload"]:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(str(config["runtime"]["device"]))
    return pipeline


class GenerativeBoundaryInpainter:
    """Execute the frozen Option A method with an injected pipeline."""

    def __init__(self, pipeline: InpaintPipeline, config: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.config = config

    def generate(
        self,
        *,
        draft_rgb: np.ndarray,
        object_mask: np.ndarray,
        reference_rgba: np.ndarray,
        class_name: str,
        seed: int,
    ) -> InpaintResult:
        method = self.config["method"]
        edit_mask, core = boundary_edit_mask(
            object_mask,
            outer_dilate_px=int(method["outer_dilate_px"]),
            protected_core_erode_px=int(method["protected_core_erode_px"]),
            minimum_protected_core_fraction=float(
                method["minimum_protected_core_fraction"]
            ),
        )
        reference = reference_canvas(
            reference_rgba,
            canvas_size=int(method["reference_canvas_size"]),
            max_fill=float(method["reference_max_fill"]),
            background_rgb=tuple(
                int(value) for value in method["reference_background_rgb"]
            ),
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        result = self.pipeline(
            prompt=str(self.config["prompts"][class_name]),
            image=Image.fromarray(np.asarray(draft_rgb, dtype=np.uint8)),
            image_reference=reference,
            mask_image=Image.fromarray(edit_mask.astype(np.uint8) * 255),
            padding_mask_crop=int(method["padding_mask_crop_px"]),
            strength=float(method["strength"]),
            num_inference_steps=int(method["num_inference_steps"]),
            guidance_scale=float(method["guidance_scale"]),
            generator=generator,
            output_type="pil",
        )
        generated = np.asarray(result.images[0].convert("RGB"), dtype=np.uint8)
        output = enforce_identity_regions(
            draft_rgb,
            generated,
            edit_mask=edit_mask,
            protected_core=core,
        )
        metrics = identity_metrics(
            draft_rgb,
            output,
            edit_mask=edit_mask,
            protected_core=core,
        )
        if (
            metrics["outside_edit_changed_pixels"] != 0
            or metrics["protected_core_changed_pixels"] != 0
        ):
            raise AssertionError("Generative identity invariants were violated")
        return InpaintResult(
            image_rgb=output,
            edit_mask=edit_mask,
            protected_core=core,
            provenance={
                "method": method["name"],
                "model_repo_id": self.config["model"]["repo_id"],
                "model_revision": self.config["model"]["revision"],
                "seed": int(seed),
                "class_name": class_name,
                "identity_metrics": metrics,
            },
        )
