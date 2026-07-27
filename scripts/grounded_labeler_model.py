"""Download or verify the pinned Grounding DINO automatic labeler."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.synthetic.grounded_labeler import (
    labeler_directory,
    load_whole_image_config,
    require_verified_labeler,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_preflight(config: dict[str, Any]) -> dict[str, Any]:
    labeler = config["labeler"]
    info = HfApi().model_info(str(labeler["repo_id"]), files_metadata=True)
    remote_sizes = {
        str(sibling.rfilename): int(sibling.size or -1)
        for sibling in info.siblings
    }
    checks = {
        "revision_matches": str(info.sha) == str(labeler["revision"]),
        "license_matches": (
            str(info.card_data.license) if info.card_data else None
        )
        == str(labeler["license"]),
        "file_sizes_match": all(
            remote_sizes.get(name) == int(size)
            for name, size in labeler["allow_files"].items()
        ),
    }
    return {
        "repo_id": labeler["repo_id"],
        "revision": info.sha,
        "license": str(info.card_data.license) if info.card_data else None,
        "download_bytes": sum(
            int(value) for value in labeler["allow_files"].values()
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _safe_directory(path: Path, cache_root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(cache_root.resolve())
    return resolved


def _download(config: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    paths = load_project_paths()
    target = _safe_directory(model_dir, paths.cache)
    staging = _safe_directory(
        paths.cache / ".hf-stage-grounding-dino-tiny",
        paths.cache,
    )
    target.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    labeler = config["labeler"]
    files = []
    for filename, expected_size in labeler["allow_files"].items():
        destination = target / str(filename)
        if not destination.exists():
            staged = Path(
                hf_hub_download(
                    repo_id=str(labeler["repo_id"]),
                    filename=str(filename),
                    revision=str(labeler["revision"]),
                    local_dir=staging,
                )
            )
            if staged.stat().st_size != int(expected_size):
                raise RuntimeError(f"Wrong staged size for {filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged), str(destination))
        if destination.stat().st_size != int(expected_size):
            raise RuntimeError(f"Wrong local size for {filename}")
        files.append(
            {
                "path": str(filename),
                "bytes": int(expected_size),
                "sha256": _sha256(destination),
            }
        )
    manifest = {
        "schema_version": 1,
        "repo_id": labeler["repo_id"],
        "revision": labeler["revision"],
        "license": labeler["license"],
        "download_bytes": int(labeler["required_download_bytes"]),
        "files": files,
    }
    manifest_path = target / "SAFESYNTH_MODEL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "download", "check"))
    args = parser.parse_args()
    config = load_whole_image_config()
    paths = load_project_paths()
    model_dir = labeler_directory(paths, config)
    remote = _remote_preflight(config)
    if not remote["passed"]:
        raise RuntimeError("Grounding DINO remote metadata changed")
    if args.action == "download":
        manifest = _download(config, model_dir)
    elif args.action == "check":
        manifest = require_verified_labeler(model_dir, config)
    else:
        manifest = None
    report = {
        **remote,
        "model_dir": str(model_dir),
        "already_verified": manifest is not None,
        "manifest": manifest,
    }
    report_path = PROJECT_ROOT / "reports" / "grounded_labeler_model.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
