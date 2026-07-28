"""Record kuotunyu's decision on exact frozen v7 review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.record_supervised_labeler_v6_review import (
    build_review_evidence,
    parse_problem_cells,
)
from scripts.train_supervised_labeler import (
    AUDIT_EVIDENCE_PATH,
    FIGURE_PATH,
    REPORT_PATH,
)
from src.data.paths import PROJECT_ROOT
from src.synthetic.supervised_labeler import (
    CONFIG_PATH,
    SPLIT_PATH,
    load_supervised_labeler_config,
    require_verified_audited_checkpoint,
)
from src.synthetic.whole_image import human_review_evidence_sha256

OUTPUT_PATH = PROJECT_ROOT / "reports" / "supervised_labeler_v7_human_review.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"Frozen v7 review evidence is missing: {path}")
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": _sha256(path),
    }


def build_v7_registration(
    *,
    config: dict[str, Any],
    split: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Construct and verify the exact numeric-pass review registration."""

    if (
        report.get("status") != "supervised_labeler_audit_passed"
        or report.get("checks")
        != {
            "audit_precision": True,
            "audit_recall": True,
            "audit_median_matched_iou": True,
        }
    ):
        raise RuntimeError("v7 must pass every numeric gate before human review")
    best = report["best_calibration"]
    metrics = report["audit_metrics"]
    postprocessing = report["postprocessing"]
    pages = [
        _record(
            FIGURE_PATH.with_name(
                f"{FIGURE_PATH.stem}_page_{page_index:02d}.png"
            )
        )
        for page_index in range(1, 4)
    ]
    separated_pages = [
        _record(
            FIGURE_PATH.with_name(
                f"{FIGURE_PATH.stem}_separated_page_{page_index:02d}.png"
            )
        )
        for page_index in range(1, 4)
    ]
    audit_evidence = _record(AUDIT_EVIDENCE_PATH)
    if (
        report.get("audit_evidence_path") != audit_evidence["path"]
        or report.get("audit_evidence_sha256") != audit_evidence["sha256"]
    ):
        raise RuntimeError("Exact v7 audit box evidence changed")
    registration = {
        "experiment_id": config["experiment_id"],
        "architecture": config["architecture"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "split_manifest_sha256": split["manifest_sha256"],
        "score_threshold": float(best["threshold"]),
        "max_relative_area": float(postprocessing["max_relative_area"]),
        "max_relative_height": float(postprocessing["max_relative_height"]),
        "audit_images": int(report["untouched_audit_images_read"]),
        "audit_precision": float(metrics["precision"]),
        "audit_recall": float(metrics["recall"]),
        "audit_median_matched_iou": float(
            metrics["median_matched_iou"]
        ),
        "human_review": {
            "figure": _record(FIGURE_PATH)["path"],
            "figure_sha256": _sha256(FIGURE_PATH),
            "pages": pages,
            "separated_pages": separated_pages,
            "evidence_path": OUTPUT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "audit_evidence": audit_evidence,
    }
    require_verified_audited_checkpoint(
        config=config,
        registration=registration,
        report=report,
        split=split,
    )
    return registration


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
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"Review evidence already exists: {OUTPUT_PATH}")

    config = load_supervised_labeler_config(CONFIG_PATH)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    registration = build_v7_registration(
        config=config,
        split=split,
        report=report,
    )
    evidence = build_review_evidence(
        registration=registration,
        decision=args.decision,
        reviewed_on=args.reviewed_on,
        problem_cells=args.problem_cells,
        note=args.note,
    )
    evidence["audit_evidence"] = registration["audit_evidence"]
    evidence["evidence_sha256"] = human_review_evidence_sha256(evidence)
    OUTPUT_PATH.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
