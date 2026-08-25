from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVACY_LINK_MARKER = "privacy_and_responsible_use.md"
REPORT_GROUPS = (
    "headline comparison",
    "calibration and operating point",
    "error analysis",
    "synthetic composition",
    "reproducibility",
    "historical diagnostics",
)


def test_all_four_public_entry_points_link_privacy_guidance() -> None:
    entry_points = (
        REPO_ROOT / "README.md",
        REPO_ROOT / "publishing/huggingface/dataset/README.md",
        REPO_ROOT / "publishing/huggingface/model/README.md",
        REPO_ROOT / "reports/README.md",
    )

    for entry_point in entry_points:
        assert PRIVACY_LINK_MARKER in entry_point.read_text(encoding="utf-8"), entry_point


def test_privacy_document_names_identifiability_and_prohibited_uses() -> None:
    privacy = (REPO_ROOT / "docs/privacy_and_responsible_use.md").read_text(
        encoding="utf-8"
    )

    for phrase in ("identifiable", "surveillance", "employment decisions"):
        assert phrase in privacy.lower()


def test_report_index_has_only_the_six_public_evidence_groups() -> None:
    report_index = (REPO_ROOT / "reports/README.md").read_text(encoding="utf-8")
    headings = [
        line.removeprefix("## ").strip()
        for line in report_index.splitlines()
        if line.startswith("## ")
    ]

    assert headings == list(REPORT_GROUPS)
