"""Record kuotunyu's decision on the exact four-case v10 FLUX diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import load_whole_image_config
from src.synthetic.whole_image import (
    canonical_mapping_sha256,
    diagnostic_manifest,
    human_review_evidence_sha256,
)

DIAGNOSTIC_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "whole_image_v10_diagnostic.json"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "whole_image_v10_seed20260808"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_problem_cases(value: str) -> list[int]:
    """Parse a comma-separated, unique list of diagnostic cases 01-04."""

    if not value.strip():
        return []
    try:
        cases = sorted({int(part.strip()) for part in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Problem cases must be comma-separated integers"
        ) from error
    if any(case < 1 or case > 4 for case in cases):
        raise argparse.ArgumentTypeError("Problem cases must be between 1 and 4")
    return cases


def build_output_review_evidence(
    *,
    diagnostic_report: dict[str, Any],
    decision: str,
    reviewed_on: str,
    problem_cases: list[int],
    note: str,
) -> dict[str, Any]:
    """Build the canonical four-case output-review record."""

    date.fromisoformat(reviewed_on)
    if decision == "approve" and problem_cases:
        raise ValueError("An approved review cannot contain problem cases")
    if decision == "reject" and not problem_cases:
        raise ValueError("A rejected review must identify problem cases")
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
        "manifest_sha256": diagnostic_report["manifest_sha256"],
        "diagnostic_report_sha256": canonical_mapping_sha256(
            diagnostic_report
        ),
        "figure": diagnostic_report["figure"],
        "figure_sha256": diagnostic_report["figure_sha256"],
        "reviewed_case_indices": [1, 2, 3, 4],
        "problem_count": len(problem_cases),
        "problem_cases": problem_cases,
        "validation_images_read": 0,
        "test_images_read": 0,
        "expanded_to_64": False,
    }
    evidence["evidence_sha256"] = human_review_evidence_sha256(evidence)
    return evidence


def _verify_diagnostic_outputs(
    *,
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    manifest = diagnostic_manifest(config)
    supervised = config["supervised_labeler"]
    cases = report.get("cases", [])
    if (
        report.get("status") != "pending_kuotunyu_visual_review"
        or report.get("manifest_sha256") != manifest["manifest_sha256"]
        or report.get("labeler_checkpoint_sha256")
        != supervised["checkpoint_sha256"]
        or report.get("labeler_split_manifest_sha256")
        != supervised["split_manifest_sha256"]
        or len(cases) != 4
        or int(report.get("validation_images_read", -1)) != 0
        or int(report.get("test_images_read", -1)) != 0
        or report.get("expanded_to_64") is not False
    ):
        raise RuntimeError("The v10 diagnostic report is incomplete or changed")
    figure_path = PROJECT_ROOT / str(report["figure"])
    if (
        not figure_path.is_file()
        or _sha256(figure_path) != report.get("figure_sha256")
    ):
        raise RuntimeError("The v10 diagnostic review figure changed")
    for case in cases:
        case_index = int(case["case_index"])
        image_path = OUTPUT_ROOT / f"case_{case_index:02d}" / "flux.png"
        if (
            not image_path.is_file()
            or _sha256(image_path) != case.get("image_sha256")
        ):
            raise RuntimeError(f"v10 case {case_index:02d} image changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision",
        required=True,
        choices=("approve", "reject"),
    )
    parser.add_argument("--reviewed-on", required=True)
    parser.add_argument(
        "--problem-cases",
        default="",
        type=parse_problem_cases,
        help="Comma-separated cases 01-04; empty only for approval.",
    )
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    config = load_whole_image_config()
    if not DIAGNOSTIC_REPORT_PATH.is_file():
        raise RuntimeError("The four-case v10 diagnostic has not run")
    report = json.loads(
        DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8")
    )
    _verify_diagnostic_outputs(config=config, report=report)
    evidence = build_output_review_evidence(
        diagnostic_report=report,
        decision=args.decision,
        reviewed_on=args.reviewed_on,
        problem_cases=args.problem_cases,
        note=args.note,
    )
    output_path = PROJECT_ROOT / str(
        config["diagnostic"]["output_review"]["evidence_path"]
    )
    if output_path.exists():
        raise RuntimeError(f"Output-review evidence already exists: {output_path}")
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
