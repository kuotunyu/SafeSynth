"""Preflight or explicitly download the pinned Option A inpainting model."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from src.data.paths import load_project_paths
from src.synthetic.generative_inpaint import (
    load_generative_config,
    model_directory,
    require_verified_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("preflight", "download", "check"),
        help="download is destructive to bandwidth/disk and requires prior user approval",
    )
    return parser.parse_args()


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Verify current Hugging Face metadata against the frozen registration."""

    model = config["model"]
    info = HfApi().model_info(str(model["repo_id"]), files_metadata=True)
    selected = [
        sibling
        for sibling in info.siblings
        if _matches(str(sibling.rfilename), list(model["allow_patterns"]))
    ]
    selected_bytes = sum(int(sibling.size or 0) for sibling in selected)
    license_name = str(info.card_data.license) if info.card_data else None
    checks = {
        "revision_matches": str(info.sha) == str(model["revision"]),
        "license_matches": license_name == str(model["license"]),
        "download_bytes_match": selected_bytes
        == int(model["required_download_bytes"]),
        "all_selected_sizes_known": all(sibling.size is not None for sibling in selected),
    }
    return {
        "repo_id": model["repo_id"],
        "registered_revision": model["revision"],
        "remote_revision": info.sha,
        "registered_license": model["license"],
        "remote_license": license_name,
        "download_bytes": selected_bytes,
        "download_gib": selected_bytes / 1024**3,
        "selected_files": [
            {"path": sibling.rfilename, "bytes": int(sibling.size or 0)}
            for sibling in selected
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }


def _write_preflight_report(
    report: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    local_status = (
        "The registered model is present with a matching verified manifest."
        if report["already_verified"]
        else (
            "The download has not been performed unless `action` is `download` "
            "and kuotunyu approved the 14.88 GiB transfer explicitly."
        )
    )
    lines = [
        "# Option A model preflight",
        "",
        f"- Model: `{report['repo_id']}`",
        f"- Pinned revision: `{report['registered_revision']}`",
        f"- License: `{report['remote_license']}`",
        f"- Required download: **{report['download_gib']:.2f} GiB**",
        f"- Destination: `{report['model_dir']}`",
        f"- Destination free space: **{report['destination_free_gib']:.1f} GiB**",
        f"- Already verified locally: **{report['already_verified']}**",
        f"- Remote metadata checks: **{'PASS' if report['passed'] else 'FAIL'}**",
        (
            "- Model card: "
            "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B"
        ),
        "- Diffusers release: https://github.com/huggingface/diffusers/releases/tag/v0.39.0",
        "",
        local_status,
        "",
    ]
    markdown_path.write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )


def _safe_model_directory(model_dir: Path, cache_root: Path) -> Path:
    resolved = model_dir.resolve()
    resolved.relative_to(cache_root.resolve())
    return resolved


def _materialize_model_file(
    *,
    target: Path,
    staging: Path,
    model: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Download one missing file through a short, resumable staging path."""

    relative_path = Path(str(record["path"]))
    destination = (target / relative_path).resolve()
    destination.relative_to(target)
    expected_bytes = int(record["bytes"])
    if destination.exists():
        if destination.is_file() and destination.stat().st_size == expected_bytes:
            return
        raise RuntimeError(
            f"Existing model file has the wrong size; refusing overwrite: {destination}"
        )

    staged = Path(
        hf_hub_download(
            repo_id=str(model["repo_id"]),
            filename=relative_path.as_posix(),
            revision=str(model["revision"]),
            local_dir=staging,
        )
    )
    if not staged.is_file() or staged.stat().st_size != expected_bytes:
        raise RuntimeError(f"Staged model file failed size check: {staged}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staged), str(destination))


def download_model(
    *,
    config: dict[str, Any],
    model_dir: Path,
    remote: dict[str, Any],
) -> dict[str, Any]:
    """Download only the frozen Diffusers components and hash every file."""

    if not remote["passed"]:
        raise RuntimeError("Remote model metadata changed; refusing download")
    paths = load_project_paths()
    target = _safe_model_directory(model_dir, paths.cache)
    target.mkdir(parents=True, exist_ok=True)
    model = config["model"]
    staging = _safe_model_directory(
        paths.cache / ".hf-stage-flux2-klein-4b",
        paths.cache,
    )
    for record in remote["selected_files"]:
        _materialize_model_file(
            target=target,
            staging=staging,
            model=model,
            record=record,
        )
    if staging.exists():
        shutil.rmtree(staging)

    files: list[dict[str, Any]] = []
    for record in remote["selected_files"]:
        path = target / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Downloaded model file failed size check: {path}")
        files.append(
            {
                "path": record["path"],
                "bytes": int(record["bytes"]),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "license": model["license"],
        "download_bytes": int(remote["download_bytes"]),
        "files": files,
    }
    (target / "SAFESYNTH_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    args = parse_args()
    config = load_generative_config()
    paths = load_project_paths()
    target = model_directory(paths, config)
    if args.action == "check":
        print(json.dumps(require_verified_model(target, config), indent=2))
        return

    remote = remote_preflight(config)
    destination = paths.cache if paths.cache.exists() else paths.data_root
    local_verified = False
    try:
        require_verified_model(target, config)
        local_verified = True
    except RuntimeError:
        pass
    report = {
        **remote,
        "model_dir": str(target),
        "destination_free_gib": shutil.disk_usage(destination).free / 1024**3,
        "already_verified": local_verified,
        "action": args.action,
    }
    _write_preflight_report(
        report,
        json_path=paths.reports / "generative_model_preflight.json",
        markdown_path=paths.reports / "generative_model_preflight.md",
    )
    if not remote["passed"]:
        raise RuntimeError("Pinned model metadata no longer matches Hugging Face")
    if args.action == "download":
        manifest = download_model(
            config=config,
            model_dir=target,
            remote=remote,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
