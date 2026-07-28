"""Record kuotunyu's decision on the exact frozen v6 review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import load_whole_image_config
from src.synthetic.supervised_labeler import (
    load_supervised_labeler_config,
    require_verified_audited_checkpoint,
)
from src.synthetic.whole_image import human_review_evidence_sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_problem_cells(value: str) -> list[int]:
    """Parse a comma-separated, unique list of canonical cells 01-48."""

    if not value.strip():
        return []
    try:
        cells = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Problem cells must be comma-separated integers"
        ) from error
    if any(cell < 1 or cell > 48 for cell in cells):
        raise argparse.ArgumentTypeError("Problem cells must be between 1 and 48")
    return cells


def build_review_evidence(
    *,
    registration: dict[str, Any],
    decision: str,
    reviewed_on: str,
    problem_cells: list[int],
    note: str,
) -> dict[str, Any]:
    """Build the canonical review record after validating decision semantics."""

    date.fromisoformat(reviewed_on)
    if decision == "approve" and problem_cells:
        raise ValueError("An approved review cannot contain problem cells")
    if decision == "reject" and not problem_cells:
        raise ValueError("A rejected review must identify problem cells")
    review = registration["human_review"]
    evidence = {
        "schema_version": 1,
        "status": (
            "approved_by_kuotunyu"
            if decision == "approve"
            else "rejected_by_kuotunyu"
        ),
        "reviewed_by": "kuotunyu",
        "reviewed_on": reviewed_on,
        "review_note": note,
        "experiment_id": registration["experiment_id"],
        "checkpoint_sha256": registration["checkpoint_sha256"],
        "split_manifest_sha256": registration["split_manifest_sha256"],
        "score_threshold": float(registration["score_threshold"]),
        "audit_images": int(registration["audit_images"]),
        "figure": review["figure"],
        "figure_sha256": review["figure_sha256"],
        "pages": review["pages"],
        "separated_pages": review["separated_pages"],
        "problem_count": len(problem_cells),
        "problem_cells": problem_cells,
        "validation_images_read": 0,
        "test_images_read": 0,
        "whole_image_generation_run": False,
    }
    evidence["evidence_sha256"] = human_review_evidence_sha256(evidence)
    return evidence


def _verify_review_images(registration: dict[str, Any]) -> None:
    review = registration["human_review"]
    records = [
        {
            "path": review["figure"],
            "sha256": review["figure_sha256"],
        },
        *review["pages"],
        *review["separated_pages"],
    ]
    for record in records:
        path = PROJECT_ROOT / str(record["path"])
        if not path.is_file() or _sha256(path) != str(record["sha256"]):
            raise RuntimeError(f"Frozen review image changed: {record['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        required=True,
        choices=("approve", "reject"),
    )
    parser.add_argument("--reviewed-on", required=True)
    parser.add_argument(
        "--problem-cells",
        default="",
        type=parse_problem_cells,
        help="Comma-separated canonical cells 01-48; empty only for approval.",
    )
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    whole_config = load_whole_image_config()
    registration = whole_config["supervised_labeler"]
    labeler_config = load_supervised_labeler_config(
        PROJECT_ROOT / str(registration["config_path"])
    )
    labeler_report = json.loads(
        (PROJECT_ROOT / str(registration["report_path"])).read_text(
            encoding="utf-8"
        )
    )
    labeler_split = json.loads(
        (PROJECT_ROOT / str(registration["split_path"])).read_text(
            encoding="utf-8"
        )
    )
    require_verified_audited_checkpoint(
        config=labeler_config,
        registration=registration,
        report=labeler_report,
        split=labeler_split,
    )
    _verify_review_images(registration)
    evidence = build_review_evidence(
        registration=registration,
        decision=args.decision,
        reviewed_on=args.reviewed_on,
        problem_cells=args.problem_cells,
        note=args.note,
    )
    output_path = PROJECT_ROOT / str(
        registration["human_review"]["evidence_path"]
    )
    if output_path.exists():
        raise RuntimeError(f"Review evidence already exists: {output_path}")
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
