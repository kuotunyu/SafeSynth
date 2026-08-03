"""The release-size audit must work exactly as its documentation invokes it."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import plan_repo_slimming
from src.release.markdown_links import (
    RepositoryLinkError,
    collect_local_destinations,
    resolve_local_target,
)
from src.release.repository_curation import plan_figure_curation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write(root: Path, path: str, text: str) -> None:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _write_bytes(root: Path, path: str, contents: bytes) -> None:
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(contents)


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    _write(tmp_path, "README.md", "# Fixture\n\n![keep](reports/figures/keep.png)\n")
    _write_bytes(tmp_path, "reports/figures/keep.png", b"keep")
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
    return tmp_path


def test_generated_report_cannot_promote_drop_entries(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "![keep](reports/figures/a/keep.png)\n")
    _write(
        tmp_path,
        "reports/repo_slimming_plan.md",
        "`reports/figures/drop.png`\n",
    )
    _write_bytes(tmp_path, "reports/figures/a/keep.png", b"keep")
    _write_bytes(tmp_path, "reports/figures/drop.png", b"drop")
    files = [
        "README.md",
        "reports/repo_slimming_plan.md",
        "reports/figures/a/keep.png",
        "reports/figures/drop.png",
    ]

    planned = {item.path: item for item in plan_figure_curation(tmp_path, files)}

    assert planned["reports/figures/a/keep.png"].keep is True
    assert planned["reports/figures/drop.png"].keep is False


def test_same_basename_in_two_directories_is_not_conflated(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "![keep](reports/figures/a/same.png)\n")
    _write_bytes(tmp_path, "reports/figures/a/same.png", b"a")
    _write_bytes(tmp_path, "reports/figures/b/same.png", b"b")
    files = [
        "README.md",
        "reports/figures/a/same.png",
        "reports/figures/b/same.png",
    ]

    planned = {item.path: item for item in plan_figure_curation(tmp_path, files)}

    assert planned["reports/figures/a/same.png"].keep is True
    assert planned["reports/figures/b/same.png"].keep is False


def test_existing_figure_directory_link_is_not_a_missing_figure(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "[directory](reports/figures/a/)\n")
    _write_bytes(tmp_path, "reports/figures/a/keep.png", b"keep")
    files = ["README.md", "reports/figures/a/keep.png"]

    planned = plan_figure_curation(tmp_path, files)

    assert planned[0].keep is False


def test_unresolved_local_destination_blocks_curation(tmp_path: Path) -> None:
    _write(tmp_path, "reports/foo.md", "![missing](reports/figures/x.png)\n")

    with pytest.raises(RepositoryLinkError, match="reports/reports/figures/x.png"):
        plan_figure_curation(tmp_path, ["reports/foo.md"])


def test_fence_info_inside_code_does_not_close_an_outer_fence(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/plan.md",
        """```python
example = '''
```text
![ignored](../reports/figures/fenced.png)
```
'''
```
""",
    )

    assert collect_local_destinations(tmp_path, ["docs/plan.md"]) == ()


def test_report_is_deterministic_and_names_keep_sources(
    fixture_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        plan_repo_slimming,
        "history_bytes_by_area",
        lambda _root: (100.0, {"reports/figures/": 75.0, "everything else": 25.0}),
    )

    first = plan_repo_slimming.render(fixture_repo)
    second = plan_repo_slimming.render(fixture_repo)

    assert first == second
    assert "README.md:3" in first
    assert "## Exact-path correction" in first
    assert "No real Markdown destination\nlinks to them" in first
    assert "every tracked `.md` except this generated report" in first
    assert "now resolve\nto tracked KEEP figures" in first


def test_generated_report_defers_owner_steps_to_verified_external_runbook(
    fixture_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inventory must not become a stale executable rewrite procedure."""

    monkeypatch.setattr(
        plan_repo_slimming,
        "history_bytes_by_area",
        lambda _root: (100.0, {"reports/figures/": 75.0, "everything else": 25.0}),
    )

    report = plan_repo_slimming.render(fixture_repo)

    assert "## Owner-only, non-executable next step" in report
    assert (
        "sole approved source only for the owner-operated history rewrite and "
        "verified KEEP restoration commands" in report
    )
    assert "Task 8 full acceptance gates remain mandatory" in report
    assert "not embedded in this report" in report
    assert "OWNER_HISTORY_REWRITE_RUNBOOK.txt" in report
    assert "bytes across ALL history" not in report
    assert "Generated-report-only historical blobs are excluded; shared blobs remain counted." in report
    assert re.search(r"filter[-_\s]*repo", report, flags=re.IGNORECASE) is None
    assert "restore_curated_figures" not in report.lower()
    assert re.search(
        r"(?im)^\s*git\s+add(?:\s|$)|\bgit\s+add\s+(?:--\s+)?reports/figures\b",
        report,
    ) is None
    assert re.search(
        r"(?is)\b(?:keep(?:ers|\s+files?)?|figures)\b.{0,120}\bsurviv\w*\b"
        r".{0,120}\brewrite\b",
        report,
    ) is None


def test_history_metrics_ignore_generated_report_commits(fixture_repo: Path) -> None:
    """A regenerated report must not change the history metrics it displays."""

    before = plan_repo_slimming.history_bytes_by_area(fixture_repo)
    generated = fixture_repo / "reports" / "repo_slimming_plan.md"
    generated.parent.mkdir(exist_ok=True)
    generated.write_text("generated inventory\n", encoding="utf-8")
    subprocess.run(["git", "add", "reports/repo_slimming_plan.md"], cwd=fixture_repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "generated report",
        ],
        cwd=fixture_repo,
        check=True,
    )

    assert plan_repo_slimming.history_bytes_by_area(fixture_repo) == before


def test_history_metrics_count_report_blob_shared_with_other_path(fixture_repo: Path) -> None:
    """A blob is excluded only when no retained path has ever used it."""

    before_total, before_areas = plan_repo_slimming.history_bytes_by_area(fixture_repo)
    shared = "shared report content\n"
    _write(fixture_repo, "reports/repo_slimming_plan.md", shared)
    _write(fixture_repo, "zzz_retained_evidence.md", shared)
    subprocess.run(
        ["git", "add", "reports/repo_slimming_plan.md", "zzz_retained_evidence.md"],
        cwd=fixture_repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "shared generated blob",
        ],
        cwd=fixture_repo,
        check=True,
    )

    total, areas = plan_repo_slimming.history_bytes_by_area(fixture_repo)

    assert total == before_total + len(shared.encode("utf-8"))
    assert areas["everything else"] == before_areas["everything else"] + len(
        shared.encode("utf-8")
    )


def test_history_metrics_handle_a_valid_empty_commit_repository(tmp_path: Path) -> None:
    """Reachable history with no blob objects is a valid zero-byte inventory."""

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "empty fixture",
        ],
        cwd=tmp_path,
        check=True,
    )

    assert plan_repo_slimming.history_bytes_by_area(tmp_path) == (0.0, {})


def test_plan_repo_slimming_runs_as_a_direct_script(tmp_path: Path) -> None:
    """`python scripts/...py` must not depend on pytest adding the repo to sys.path."""

    scripts = tmp_path / "scripts"
    figures = tmp_path / "reports" / "figures"
    scripts.mkdir()
    figures.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "scripts" / "plan_repo_slimming.py", scripts)
    shutil.copytree(PROJECT_ROOT / "src", tmp_path / "src")
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
