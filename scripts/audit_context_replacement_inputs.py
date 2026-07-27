"""Audit the preregistered context-replacement input guard without model inference."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.compose import (
    _load_configs,
    _load_context,
    _load_pass1,
    context_replacement_input_guard,
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
    for image_id in sorted(train_images):
        image = train_images[image_id]
        result = context_replacement_input_guard(
            image_shape=(int(image["height"]), int(image["width"])),
            annotations=annotations[image_id],
            categories=categories,
            pass1=_load_pass1(paths, image_id),
            guard_config=guard_config,
        )
        decisions[image_id] = result
        if result.accepted:
            accepted_background_ids.add(image_id)
            eligible_anchor_count += len(result.eligible_annotation_ids)
        else:
            reason_counts[str(result.reject_reason)] += 1

    previous_seed = int(generative_config["pilot"]["previous_failed_root_seed"])
    previous_records_path = (
        paths.synthetic
        / f"h4_generative_identity_pilot_seed{previous_seed}"
        / "records.jsonl"
    )
    previous_records = _read_jsonl(previous_records_path)
    rejected_previous_cells = [
        index
        for index, record in enumerate(previous_records, start=1)
        if not decisions[int(record["background"]["image_id"])].accepted
    ]
    known_failure_cells = [10, 12]
    known_failure_cells_rejected = all(
        cell in rejected_previous_cells for cell in known_failure_cells
    )
    if not known_failure_cells_rejected:
        raise AssertionError("The preregistered guard missed a known pilot failure")

    payload = {
        "schema_version": 1,
        "status": "guard_audit_complete_no_model_inference",
        "scope": "Train metadata, Pass-1 QC, and rejected pilot provenance only",
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
        },
        "rejected_previous_pilot": {
            "root_seed": previous_seed,
            "n_images": len(previous_records),
            "rejected_cell_count": len(rejected_previous_cells),
            "rejected_cells": rejected_previous_cells,
            "known_failure_cells": known_failure_cells,
            "known_failure_cells_rejected": known_failure_cells_rejected,
        },
        "next_pilot": {
            "architecture": generative_config["pilot"]["architecture"],
            "root_seed": int(generative_config["pilot"]["root_seed"]),
            "n_images": int(generative_config["pilot"]["n_images"]),
            "generated": False,
        },
    }
    json_path = PROJECT_ROOT / "reports" / "h4_guarded_input_audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    reasons = payload["train_backgrounds"]["reject_reasons"]
    markdown = [
        "# H4 guarded-input CPU audit",
        "",
        "- Status: **complete; no model inference**",
        "- Data scope: Train metadata, Pass-1 QC, and rejected-pilot provenance",
        "- Validation/Test images read: **0 / 0**",
        "- H4 AUC computed: **no**",
        (
            "- Train backgrounds accepted: "
            f"**{len(accepted_background_ids):,}/{len(train_images):,}**"
        ),
        f"- Eligible anchors remaining: **{eligible_anchor_count:,}**",
        (
            "- Previous failed-pilot cells rejected by the guard: "
            f"**{len(rejected_previous_cells)}/{len(previous_records)}**"
        ),
        (
            "- Known failure cells 10 and 12 rejected: "
            f"**{'yes' if known_failure_cells_rejected else 'no'}**"
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
            "This audit only checks whether the fixed guard removes unsafe inputs "
            "while leaving a usable Train pool. It does not select a model-call "
            "variant, generate a new identity pilot, compute H4, or reopen M13."
        ),
        "",
        (
            "The next untouched pilot is registered for root seed "
            f"`{payload['next_pilot']['root_seed']}` and has not been generated."
        ),
        "",
    ]
    markdown_path = PROJECT_ROOT / "reports" / "h4_guarded_input_audit.md"
    markdown_path.write_text(
        "\n".join(markdown),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
