"""Audit the preregistered context-replacement input guard without model inference."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import (
    _load_configs,
    _load_context,
    _load_pass1,
    prepare_context_replacement_background,
)


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


def main() -> None:
    paths = load_project_paths()
    compose_config, _ = _load_configs()
    generative_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "generative_inpaint.yaml").read_text(
            encoding="utf-8"
        )
    )
    guard_config = compose_config["compose"]["context_replacement"]["input_guard"]
    coco, _, train_images, annotations, _, _ = _load_context(paths)
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco["categories"]
    }

    reason_counts: Counter[str] = Counter()
    accepted_background_ids: set[int] = set()
    eligible_anchor_count = 0
    decisions: dict[int, Any] = {}
    normalizations: dict[int, Any] = {}
    normalization_side_counts: Counter[str] = Counter()
    for image_id in sorted(train_images):
        image = train_images[image_id]
        image_rgb = np.asarray(
            Image.open(paths.hardhat_raw / str(image["file_name"])).convert("RGB")
        )
        prepared = prepare_context_replacement_background(
            image_rgb=image_rgb,
            annotations=annotations[image_id],
            categories=categories,
            pass1=_load_pass1(paths, image_id),
            guard_config=guard_config,
            output_shape=image_rgb.shape[:2],
            transform_masks=False,
        )
        result = prepared.guard
        decisions[image_id] = result
        normalizations[image_id] = prepared.normalization
        normalization_side_counts.update(prepared.normalization.detected_sides)
        if result.accepted:
            accepted_background_ids.add(image_id)
            eligible_anchor_count += len(result.eligible_annotation_ids)
        else:
            reason_counts[str(result.reject_reason)] += 1

    def historical_case(
        *,
        seed: int,
        output_tag: str,
        known_failure_cells: list[int],
    ) -> dict[str, Any]:
        records = _read_jsonl(
            paths.synthetic / output_tag.format(seed=seed) / "records.jsonl"
        )
        normalized_cells = [
            index
            for index, record in enumerate(records, start=1)
            if normalizations[int(record["background"]["image_id"])].applied
        ]
        rejected_cells = [
            index
            for index, record in enumerate(records, start=1)
            if not decisions[int(record["background"]["image_id"])].accepted
        ]
        resolved_cells = sorted(set(normalized_cells) | set(rejected_cells))
        known_resolved = all(
            cell in resolved_cells for cell in known_failure_cells
        )
        if not known_resolved:
            raise AssertionError(
                f"The v5 preflight missed known failures for seed {seed}"
            )
        return {
            "root_seed": seed,
            "n_images": len(records),
            "normalized_cell_count": len(normalized_cells),
            "normalized_cells": normalized_cells,
            "rejected_after_normalization_cell_count": len(rejected_cells),
            "rejected_after_normalization_cells": rejected_cells,
            "known_failure_cells": known_failure_cells,
            "known_failure_cells_normalized_or_rejected": known_resolved,
        }

    pilot = generative_config["pilot"]
    previous_seed = int(pilot["previous_failed_root_seed"])
    v4_history = historical_case(
        seed=previous_seed,
        output_tag="h4_guarded_input_preflight_seed{seed}",
        known_failure_cells=[7, 54],
    )
    v3_history = historical_case(
        seed=int(pilot["v3_failed_root_seed"]),
        output_tag="h4_guarded_input_preflight_seed{seed}",
        known_failure_cells=[64],
    )
    v2_history = historical_case(
        seed=int(pilot["v2_failed_root_seed"]),
        output_tag="h4_guarded_input_preflight_seed{seed}",
        known_failure_cells=[11, 25],
    )
    original_history = historical_case(
        seed=int(pilot["original_failed_root_seed"]),
        output_tag="h4_generative_identity_pilot_seed{seed}",
        known_failure_cells=[10, 12],
    )

    payload = {
        "schema_version": 1,
        "status": "guard_audit_complete_no_model_inference",
        "scope": (
            "Train pixels, metadata, Pass-1 QC, and failed-pilot provenance only"
        ),
        "validation_images_read": 0,
        "test_images_read": 0,
        "model_inference_run": False,
        "h4_auc_computed": False,
        "guard_config": guard_config,
        "compose_config_sha256": _sha256(PROJECT_ROOT / "configs" / "compose.yaml"),
        "train_backgrounds": {
            "total": len(train_images),
            "accepted": len(accepted_background_ids),
            "rejected": len(train_images) - len(accepted_background_ids),
            "reject_reasons": dict(sorted(reason_counts.items())),
            "eligible_anchor_count": eligible_anchor_count,
            "normalization_applied": sum(
                normalization.applied
                for normalization in normalizations.values()
            ),
            "normalized_side_counts": dict(
                sorted(normalization_side_counts.items())
            ),
        },
        "historical_failed_inputs": {
            "v4": v4_history,
            "v3": v3_history,
            "v2": v2_history,
            "original": original_history,
        },
        "next_pilot": {
            "architecture": generative_config["pilot"]["architecture"],
            "root_seed": int(generative_config["pilot"]["root_seed"]),
            "n_images": int(generative_config["pilot"]["n_images"]),
            "generated": False,
        },
    }
    json_path = (
        PROJECT_ROOT / "reports" / "h4_reflection_normalization_v5_audit.json"
    )
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    reasons = payload["train_backgrounds"]["reject_reasons"]
    markdown = [
        "# H4 reflected-padding normalization v5 CPU audit",
        "",
        "- Status: **complete; no model inference**",
        "- Data scope: Train metadata, Pass-1 QC, and rejected-pilot provenance",
        "- Validation/Test images read: **0 / 0**",
        "- H4 AUC computed: **no**",
        (
            "- Train backgrounds accepted: "
            f"**{len(accepted_background_ids):,}/{len(train_images):,}**"
        ),
        (
            "- Train backgrounds normalized: "
            f"**{payload['train_backgrounds']['normalization_applied']:,}/"
            f"{len(train_images):,}**"
        ),
        f"- Eligible anchors remaining: **{eligible_anchor_count:,}**",
        (
            "- Known v4 failure cells 7 and 54 normalized or rejected: "
            f"**{'yes' if v4_history['known_failure_cells_normalized_or_rejected'] else 'no'}**"
        ),
        (
            "- Known v3 failure cell 64 normalized or rejected: "
            f"**{'yes' if v3_history['known_failure_cells_normalized_or_rejected'] else 'no'}**"
        ),
        (
            "- Known v2 failure cells 11 and 25 normalized or rejected: "
            f"**{'yes' if v2_history['known_failure_cells_normalized_or_rejected'] else 'no'}**"
        ),
        (
            "- Original known failure cells 10 and 12 normalized or rejected: "
            f"**{'yes' if original_history['known_failure_cells_normalized_or_rejected'] else 'no'}**"
        ),
        "",
        "## Rejection counts",
        "",
        "| reason | backgrounds |",
        "|---|---:|",
        *[f"| `{name}` | {count:,} |" for name, count in reasons.items()],
        "",
        "## Scientific boundary",
        "",
        (
            "This audit only checks deterministic CPU normalization and the "
            "post-transform input guards. It does not select a model-call variant, "
            "generate a new identity pilot, compute H4, or reopen M13."
        ),
        "",
        (
            "The next untouched pilot is registered for root seed "
            f"`{payload['next_pilot']['root_seed']}` and has not been generated."
        ),
        "",
    ]
    markdown_path = (
        PROJECT_ROOT / "reports" / "h4_reflection_normalization_v5_audit.md"
    )
    markdown_path.write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
