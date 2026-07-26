"""Correctness-critical composition geometry, blending, harmonization, and bbox updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class Layer:
    """One existing or pasted instance in full-image coordinates."""

    instance_id: str
    class_name: str
    kind: str
    mask: np.ndarray
    bbox_xywh_original: list[float]
    y_bottom: float
    z_index: int = 0
    intentional_removal: bool = False
    existing_mask_source: str | None = None


@dataclass(frozen=True)
class VisibilityResult:
    accepted: bool
    annotations: tuple[dict[str, Any], ...]
    rejected_instance_ids: tuple[str, ...]
    reason: str | None


def tight_bbox(mask: np.ndarray) -> list[float] | None:
    """Return a half-open COCO XYWH bbox tightly enclosing a boolean mask."""

    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return [float(left), float(top), float(right - left), float(bottom - top)]


def box_to_mask(shape: tuple[int, int], bbox_xywh: list[float]) -> np.ndarray:
    """Rasterize a clipped half-open COCO box for safe Pass 1 fallback."""

    height, width = shape
    x, y, box_width, box_height = (float(value) for value in bbox_xywh)
    left = max(0, min(width, int(np.floor(x))))
    top = max(0, min(height, int(np.floor(y))))
    right = max(left, min(width, int(np.ceil(x + box_width))))
    bottom = max(top, min(height, int(np.ceil(y + box_height))))
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def assign_z_order(layers: list[Layer]) -> list[Layer]:
    """Sort far-to-near by y_bottom, breaking ties by stable instance ID."""

    ordered = sorted(layers, key=lambda layer: (layer.y_bottom, layer.instance_id))
    for z_index, layer in enumerate(ordered):
        layer.z_index = z_index
    return ordered


def recompute_visible_annotations(
    layers: list[Layer],
    *,
    min_visible_fraction_pasted: float,
    existing_keep_original_above: float,
    existing_recompute_above: float,
) -> VisibilityResult:
    """Apply COMP-07..10 without silently deleting any real annotation."""

    ordered = assign_z_order(layers)
    union_above = np.zeros_like(ordered[0].mask, dtype=bool) if ordered else np.zeros((0, 0), bool)
    annotations_reversed: list[dict[str, Any]] = []
    rejected: list[str] = []
    for layer in reversed(ordered):
        mask = np.asarray(layer.mask, dtype=bool)
        visible = mask & ~union_above
        original_area = int(mask.sum())
        visible_fraction = int(visible.sum()) / max(original_area, 1)
        visible_bbox = tight_bbox(visible)
        union_above |= mask
        if layer.kind == "hard_negative" or layer.intentional_removal:
            continue
        if layer.kind == "pasted":
            if visible_fraction < min_visible_fraction_pasted or visible_bbox is None:
                rejected.append(layer.instance_id)
                continue
            output_bbox = visible_bbox
        elif layer.kind == "existing":
            if visible_fraction >= existing_keep_original_above:
                output_bbox = list(layer.bbox_xywh_original)
            elif visible_fraction >= existing_recompute_above and visible_bbox is not None:
                output_bbox = visible_bbox
            else:
                # Reject the placement. Deleting this label would create a false negative.
                return VisibilityResult(
                    accepted=False,
                    annotations=(),
                    rejected_instance_ids=(layer.instance_id,),
                    reason="EXISTING_OBJECT_TOO_OCCLUDED",
                )
        else:
            raise ValueError(f"Unknown layer kind: {layer.kind}")
        annotations_reversed.append(
            {
                "instance_id": layer.instance_id,
                "class_name": layer.class_name,
                "kind": layer.kind,
                "bbox_xywh": output_bbox,
                "bbox_xywh_original": list(layer.bbox_xywh_original),
                "visible_fraction": visible_fraction,
                "z_index": layer.z_index,
                "y_bottom": layer.y_bottom,
                "existing_mask_source": layer.existing_mask_source,
                "kept": True,
            }
        )
    annotations = tuple(reversed(annotations_reversed))
    return VisibilityResult(
        accepted=True,
        annotations=annotations,
        rejected_instance_ids=tuple(rejected),
        reason=None,
    )


def warp_rgba(
    rgba: np.ndarray,
    *,
    scale: float,
    rotation_deg: float,
    hflip: bool,
) -> np.ndarray:
    """Transform a cutout using appropriate interpolation for RGB and alpha."""

    source = np.asarray(rgba, dtype=np.uint8)
    if hflip:
        source = source[:, ::-1].copy()
    height, width = source.shape[:2]
    scaled_width = max(1, round(width * scale))
    scaled_height = max(1, round(height * scale))
    rgb_interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    rgb = cv2.resize(
        source[..., :3], (scaled_width, scaled_height), interpolation=rgb_interpolation
    )
    alpha = cv2.resize(
        source[..., 3], (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR
    )
    diagonal = int(np.ceil(np.hypot(scaled_width, scaled_height))) + 2
    center = (diagonal / 2, diagonal / 2)
    rgb_canvas = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((diagonal, diagonal), dtype=np.uint8)
    left = (diagonal - scaled_width) // 2
    top = (diagonal - scaled_height) // 2
    rgb_canvas[top : top + scaled_height, left : left + scaled_width] = rgb
    alpha_canvas[top : top + scaled_height, left : left + scaled_width] = alpha
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    rotated_rgb = cv2.warpAffine(
        rgb_canvas,
        matrix,
        (diagonal, diagonal),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rotated_alpha = cv2.warpAffine(
        alpha_canvas,
        matrix,
        (diagonal, diagonal),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    nonzero = rotated_alpha > 0
    bbox = tight_bbox(nonzero)
    if bbox is None:
        return np.zeros((1, 1, 4), dtype=np.uint8)
    x, y, crop_width, crop_height = (round(value) for value in bbox)
    return np.dstack(
        (
            rotated_rgb[y : y + crop_height, x : x + crop_width],
            rotated_alpha[y : y + crop_height, x : x + crop_width],
        )
    )


def placement_slices(
    *,
    frame_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    center_xy: tuple[float, float],
) -> tuple[tuple[slice, slice], tuple[slice, slice], float]:
    """Return clipped frame/patch slices and the pre-clip inside ratio."""

    frame_height, frame_width = frame_shape
    patch_height, patch_width = patch_shape
    left = round(center_xy[0] - patch_width / 2)
    top = round(center_xy[1] - patch_height / 2)
    right = left + patch_width
    bottom = top + patch_height
    frame_left = max(0, left)
    frame_top = max(0, top)
    frame_right = min(frame_width, right)
    frame_bottom = min(frame_height, bottom)
    width = max(frame_right - frame_left, 0)
    height = max(frame_bottom - frame_top, 0)
    patch_left = frame_left - left
    patch_top = frame_top - top
    inside_ratio = width * height / max(patch_width * patch_height, 1)
    return (
        (slice(frame_top, frame_bottom), slice(frame_left, frame_right)),
        (slice(patch_top, patch_top + height), slice(patch_left, patch_left + width)),
        inside_ratio,
    )


def annulus_mask(
    shape: tuple[int, int],
    footprint: np.ndarray,
    *,
    outer_scale: float,
) -> np.ndarray:
    """Build a local target-statistics ring around a placement footprint."""

    binary = np.asarray(footprint, dtype=np.uint8)
    bbox = tight_bbox(binary.astype(bool))
    if bbox is None:
        return np.zeros(shape, dtype=bool)
    _, _, width, height = bbox
    dilation = max(1, round((outer_scale - 1) * max(width, height) / 2))
    kernel_size = 2 * dilation + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    outer = cv2.dilate(binary, kernel).astype(bool)
    return outer & ~binary.astype(bool)


def harmonize_lab(
    patch_rgb: np.ndarray,
    alpha: np.ndarray,
    background_rgb: np.ndarray,
    target_mask: np.ndarray,
    *,
    config: dict[str, Any],
) -> np.ndarray:
    """Apply local CIELab Reinhard matching while protecting chroma class signal."""

    source_mask = alpha >= 128
    if not source_mask.any() or not target_mask.any():
        return patch_rgb
    source_lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_pixels = source_lab[source_mask]
    target_pixels = target_lab[target_mask]
    source_mean = source_pixels.mean(axis=0)
    source_std = np.maximum(source_pixels.std(axis=0), 1.0)
    target_mean = target_pixels.mean(axis=0)
    target_std = np.maximum(target_pixels.std(axis=0), 1.0)

    result = source_lab.copy()
    l_lambda = float(target_std[0] / source_std[0])
    l_lambda = float(np.clip(l_lambda, *config["L_std_lambda_clip"]))
    l_mean_target = source_mean[0] + float(config["L_mean_beta"]) * (
        target_mean[0] - source_mean[0]
    )
    l_mean_target = float(
        np.clip(
            l_mean_target,
            source_mean[0] - float(config["max_delta_mean_L"]),
            source_mean[0] + float(config["max_delta_mean_L"]),
        )
    )
    result[..., 0] = (result[..., 0] - source_mean[0]) * l_lambda + l_mean_target
    ab_beta = float(config["ab_mean_beta"])
    for channel in (1, 2):
        delta = float(
            np.clip(
                target_mean[channel] - source_mean[channel],
                -float(config["max_delta_mean_ab"]),
                float(config["max_delta_mean_ab"]),
            )
        )
        result[..., channel] += ab_beta * delta
    return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def alpha_composite(
    background_rgb: np.ndarray,
    patch_rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    frame_slice: tuple[slice, slice],
    patch_slice: tuple[slice, slice],
) -> np.ndarray:
    """Composite one clipped patch without mutating the caller's background."""

    output = np.asarray(background_rgb, dtype=np.uint8).copy()
    foreground = patch_rgb[patch_slice].astype(np.float32)
    alpha_float = alpha[patch_slice].astype(np.float32)[..., None] / 255
    target = output[frame_slice].astype(np.float32)
    output[frame_slice] = np.clip(
        foreground * alpha_float + target * (1 - alpha_float), 0, 255
    ).astype(np.uint8)
    return output


def inpaint_masked_object(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    dilate_px: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a helmet before a head swap with deterministic Telea inpainting."""

    kernel_size = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inpaint_mask = cv2.dilate(mask.astype(np.uint8), kernel)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = cv2.inpaint(bgr, inpaint_mask * 255, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB), inpaint_mask.astype(bool)


def feather_alpha(alpha: np.ndarray, *, config: dict[str, Any]) -> np.ndarray:
    """Erode then feather inward so the paste never alters outside pixels."""

    source = np.asarray(alpha, dtype=np.uint8)
    original_support = source > 0
    erode_px = int(config["erode_before_feather_px"])
    if erode_px > 0:
        kernel_size = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        source = cv2.erode(source, kernel)
    bbox = tight_bbox(original_support)
    if bbox is None:
        return source
    sigma = float(config["feather_sigma_base"]) + float(
        config["feather_sigma_per_px"]
    ) * min(float(bbox[2]), float(bbox[3]))
    sigma = float(np.clip(sigma, *config["feather_sigma_clip"]))
    blurred = cv2.GaussianBlur(source, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Taper through the original mask fringe. Clipping to the *eroded* support
    # would reintroduce a hard one-pixel step at its new boundary.
    blurred[~original_support] = 0
    return blurred


def decontaminate_soft_edge(
    patch_rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    core_alpha_min: int,
) -> np.ndarray:
    """Replace source-background halo colours with nearest foreground colours."""

    alpha_array = np.asarray(alpha, dtype=np.uint8)
    core = alpha_array >= int(core_alpha_min)
    fringe = (alpha_array > 0) & ~core
    if not core.any() or not fringe.any():
        return np.asarray(patch_rgb, dtype=np.uint8).copy()
    _, nearest = distance_transform_edt(~core, return_indices=True)
    output = np.asarray(patch_rgb, dtype=np.uint8).copy()
    output[fringe] = output[nearest[0][fringe], nearest[1][fringe]]
    return output


def match_high_frequency_noise(
    patch_rgb: np.ndarray,
    alpha: np.ndarray,
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    *,
    sigma_cap: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add only the missing local high-frequency energy after downscaling."""

    source_mask = np.asarray(alpha) >= 128
    target_mask = np.asarray(target_mask, dtype=bool)
    if not source_mask.any() or not target_mask.any():
        return patch_rgb

    def noise_sigma(image: np.ndarray, mask: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
        return float(residual[mask].std())

    missing = max(
        noise_sigma(target_rgb, target_mask) - noise_sigma(patch_rgb, source_mask),
        0.0,
    )
    missing = min(missing, float(sigma_cap))
    if missing == 0:
        return patch_rgb
    noise = rng.normal(0, missing, patch_rgb.shape[:2])[..., None]
    output = patch_rgb.astype(np.float32)
    output[source_mask] += noise[source_mask]
    return np.clip(output, 0, 255).astype(np.uint8)


def seam_energy_ratio(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    band_px: int,
) -> float:
    """Measure boundary gradient energy relative to the pasted interior."""

    binary = np.asarray(mask, dtype=np.uint8)
    if not binary.any():
        return float("inf")
    kernel_size = 2 * int(band_px) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    inner = cv2.erode(binary, kernel).astype(bool)
    band = binary.astype(bool) & ~inner
    if not band.any() or not inner.any():
        return float("inf")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    interior_energy = max(float(magnitude[inner].mean()), 1e-6)
    return float(magnitude[band].mean() / interior_energy)


def apply_postfx(
    image_rgb: np.ndarray,
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply geometry-preserving whole-image degradation deterministically."""

    output = np.asarray(image_rgb, dtype=np.uint8).copy()
    applied: dict[str, Any] = {}
    low_light = config["low_light"]
    if rng.random() < float(low_light["prob_given_postfx"]):
        gamma = float(rng.uniform(*low_light["gamma"]))
        gain = float(rng.uniform(*low_light["gain"]))
        noise_sigma = float(rng.uniform(*low_light["noise_sigma"]))
        wb_r = float(rng.uniform(*low_light["wb_gain_r"]))
        wb_b = float(rng.uniform(*low_light["wb_gain_b"]))
        normalized = np.power(output.astype(np.float32) / 255.0, gamma) * gain
        normalized[..., 0] *= wb_r
        normalized[..., 2] *= wb_b
        normalized += rng.normal(0, noise_sigma / 255.0, output.shape)
        output = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        applied["low_light"] = {
            "gamma": gamma,
            "gain": gain,
            "noise_sigma": noise_sigma,
            "wb_gain_r": wb_r,
            "wb_gain_b": wb_b,
        }
    blur = config["motion_blur"]
    if rng.random() < float(blur["prob_given_postfx"]):
        length = int(rng.choice(blur["kernel_lengths"]))
        angle = float(rng.uniform(*blur["angle_deg"]))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        center = (length / 2 - 0.5, length / 2 - 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, 1)
        kernel = cv2.warpAffine(kernel, matrix, (length, length))
        kernel /= max(float(kernel.sum()), 1e-6)
        output = cv2.filter2D(output, -1, kernel)
        applied["motion_blur"] = {"kernel_length": length, "angle_deg": angle}
    return output, applied
