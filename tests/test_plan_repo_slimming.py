"""The release-size audit must work exactly as its documentation invokes it."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.release.markdown_links import (
    RepositoryLinkError,
    collect_local_destinations,
    resolve_local_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_plan_repo_slimming_runs_as_a_direct_script(tmp_path: Path) -> None:
    """`python scripts/...py` must not depend on pytest adding the repo to sys.path."""

    scripts = tmp_path / "scripts"
    figures = tmp_path / "reports" / "figures"
    scripts.mkdir()
    figures.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "plan_repo_slimming.py", scripts)
    (tmp_path / "README.md").write_text(
        "# Fixture\n\n![kept](reports/figures/keep.png)\n",
        encoding="utf-8",
    )
    (figures / "keep.png").write_bytes(b"small fixture")

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )

    completed = subprocess.run(
        [sys.executable, "scripts/plan_repo_slimming.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = tmp_path / "reports" / "repo_slimming_plan.md"
    assert report.is_file()
    assert "`reports/figures/keep.png`" in report.read_text(encoding="utf-8")


def test_only_real_markdown_destinations_are_collected(tmp_path: Path) -> None:
    """Code and prose mentions must not be mistaken for Markdown destinations."""

    doc = tmp_path / "docs" / "audit.md"
    doc.parent.mkdir()
    doc.write_text(
        """# Audit
![inline](../reports/figures/a.png)
[angle](<../reports/figures/with space.png>)
[ref]: ../reports/figures/reference.png
`reports/figures/code.png`
plain reports/figures/prose.png
```text
![fenced](../reports/figures/fenced.png)
```
""",
        encoding="utf-8",
    )

    links = collect_local_destinations(tmp_path, ["docs/audit.md"])

    assert [link.resolved_path for link in links] == [
        "reports/figures/a.png",
        "reports/figures/with space.png",
        "reports/figures/reference.png",
    ]


def test_relative_paths_are_resolved_from_the_document_directory() -> None:
    """A document's directory is the base for relative link resolution."""

    assert (
        resolve_local_target("docs/audit/report.md", "../../reports/figures/a.png")
        == "reports/figures/a.png"
    )


def test_external_urls_and_anchors_are_not_local_targets() -> None:
    """External URLs and same-document anchors must remain outside the audit."""

    assert resolve_local_target("README.md", "https://example.com/x.png") is None
    assert resolve_local_target("README.md", "#results") is None


def test_path_escape_fails_closed() -> None:
    """Links escaping the repository must be rejected instead of normalized."""

    with pytest.raises(RepositoryLinkError, match="escapes repository"):
        resolve_local_target("README.md", "../outside.png")


def test_absolute_markdown_source_path_fails_closed(tmp_path: Path) -> None:
    """The collector must not read an absolute document outside its root."""

    with pytest.raises(RepositoryLinkError, match="absolute repository path"):
        collect_local_destinations(tmp_path, ["/outside.md"])


@pytest.mark.parametrize(
    ("raw_target", "expected"),
    [
        ("reports/figures/a%23b.png", "reports/figures/a#b.png"),
        ("reports/figures/a%3Fb.png", "reports/figures/a?b.png"),
    ],
)
def test_encoded_filename_delimiters_are_retained(raw_target: str, expected: str) -> None:
    """Encoded delimiters are filename characters, not URL suffix separators."""

    assert resolve_local_target("README.md", raw_target) == expected


@pytest.mark.parametrize("raw_target", [r"C:\outside.png", "C%3A/outside.png"])
def test_windows_absolute_target_fails_closed(raw_target: str) -> None:
    """A drive-letter path must not be misclassified as an external scheme."""

    with pytest.raises(RepositoryLinkError, match="absolute local path"):
        resolve_local_target("README.md", raw_target)


def test_malformed_markdown_link_opener_fails_closed(tmp_path: Path) -> None:
    """An unclosed link opener is unsafe rather than a valid destination."""

    document = tmp_path / "docs" / "audit.md"
    document.parent.mkdir()
    document.write_text("[broken](../reports/figures/a.png\n", encoding="utf-8")

    with pytest.raises(RepositoryLinkError, match="malformed Markdown destination"):
        collect_local_destinations(tmp_path, ["docs/audit.md"])
