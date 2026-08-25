from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"


def test_windows_ci_configures_uv_paths_at_runtime() -> None:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    job = workflow["jobs"]["lint-and-test"]

    job_env = job["env"]
    assert job_env["UV_LINK_MODE"] == "copy"
    assert all("${{ runner." not in str(value) for value in job_env.values())

    steps = job["steps"]
    checkout_index = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    setup_uv_index = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    )
    sync_index = next(
        index for index, step in enumerate(steps) if "uv sync" in step.get("run", "")
    )
    runtime_steps = [
        (index, step)
        for index, step in enumerate(steps)
        if step.get("shell") == "pwsh"
        and "$env:RUNNER_TEMP" in step.get("run", "")
        and "$env:GITHUB_ENV" in step.get("run", "")
    ]

    assert len(runtime_steps) == 1
    runtime_index, runtime_step = runtime_steps[0]
    assert checkout_index < runtime_index < setup_uv_index < sync_index

    script = runtime_step["run"]
    assert 'Join-Path $env:RUNNER_TEMP "uv-cache"' in script
    assert 'Join-Path $env:RUNNER_TEMP "safesynth-venv"' in script
    assert '"UV_CACHE_DIR=$uvCacheDir"' in script
    assert '"UV_PROJECT_ENVIRONMENT=$uvProjectEnvironment"' in script
    assert "Out-File -FilePath $env:GITHUB_ENV" in script
    assert "-Append" in script
