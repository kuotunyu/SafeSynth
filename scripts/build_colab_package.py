"""Build the single archive that goes to Google Drive for Colab training.

TRAIN-25 forbids the upload from containing any Test image, so the archive is
built from an explicit allowlist derived from the frozen split manifest and then
audited afterwards: every name in the zip is checked against the Test set. A
directory copy would be one typo away from leaking the test split into a
training environment, and that leak would be invisible in every downstream
number.

TRAIN-08 wants the data unzipped to /content/data before training rather than
read from mounted Drive, so this produces one file to upload and unzip, not a
folder tree to sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from src.data.paths import PROJECT_ROOT, ProjectPaths, load_project_paths

POOL_TAG = "m13_pool_1x"
SUBSETS = ("filtered_1x", "unfiltered_1x", "filtered_0_5x", "unfiltered_0_5x")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_lookup(paths: ProjectPaths) -> dict[str, str]:
    manifest = json.loads(
        (paths.splits / "split_manifest.json").read_text(encoding="utf-8")
    )
    entries = manifest["images"] if isinstance(manifest, dict) else manifest
    return {str(e["file_name"]).split("/")[-1]: str(e["split"]) for e in entries}


def plan(paths: ProjectPaths) -> dict[str, Any]:
    """Decide exactly which files ship, before touching the archive."""

    pool = paths.synthetic / POOL_TAG
    split_of = _split_lookup(paths)

    real_names = sorted(n for n, s in split_of.items() if s in {"train", "val"})
    test_names = {n for n, s in split_of.items() if s == "test"}
    if set(real_names) & test_names:
        raise RuntimeError("Train/val allowlist intersects the Test split")

    synthetic_names: set[str] = set()
    for subset in SUBSETS:
        payload = json.loads((pool / f"annotations_{subset}.json").read_text(encoding="utf-8"))
        synthetic_names |= {im["file_name"].split("/")[-1] for im in payload["images"]}

    return {
        "pool": pool,
        "real_names": real_names,
        "test_names": test_names,
        "synthetic_names": sorted(synthetic_names),
    }


def build(output: Path, *, paths: ProjectPaths) -> dict[str, Any]:
    selection = plan(paths)
    pool = selection["pool"]
    test_names = selection["test_names"]

    output.parent.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    # Stored, not deflated: PNG and these JSONs are already compressed, so
    # deflate costs minutes of CPU for a rounding error of size.
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in selection["real_names"]:
            arc = f"real/images/{name}"
            archive.write(paths.hardhat_raw / "images" / name, arc)
            written.append(arc)
        for name in selection["synthetic_names"]:
            arc = f"synthetic/images/{name}"
            archive.write(pool / "images" / name, arc)
            written.append(arc)

        arc = "real/coco_all.json"
        archive.write(paths.interim / "coco_all.json", arc)
        written.append(arc)
        for subset in SUBSETS:
            arc = f"synthetic/annotations_{subset}.json"
            archive.write(pool / f"annotations_{subset}.json", arc)
            written.append(arc)
        for name in ("split_manifest.json", "test_blocklist.json", "MANIFEST.sha256"):
            arc = f"splits/{name}"
            archive.write(paths.splits / name, arc)
            written.append(arc)
        arc = "configs/training.yaml"
        archive.write(PROJECT_ROOT / "configs" / "training.yaml", arc)
        written.append(arc)

    # TRAIN-25 audit: read the archive back and prove no Test image is inside.
    with zipfile.ZipFile(output) as archive:
        members = archive.namelist()
    leaked = sorted(
        name for name in members
        if name.startswith("real/images/") and Path(name).name in test_names
    )
    if leaked:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"TRAIN-25 violated: archive contained Test images {leaked[:5]}")

    return {
        "archive": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "n_members": len(members),
        "n_real_images": len(selection["real_names"]),
        "n_synthetic_images": len(selection["synthetic_names"]),
        "n_test_images_excluded": len(test_names),
        "test_images_in_archive": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = load_project_paths()
    output = (
        Path(args.output)
        if args.output
        else paths.data_root / "colab" / "safesynth_train_data.zip"
    )
    result = build(output, paths=paths)
    result["path"] = str(output)
    result["size_gb"] = round(result["bytes"] / 1e9, 2)
    (PROJECT_ROOT / "reports" / "colab_package.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
