"""The release-size audit must work exactly as its documentation invokes it."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

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
