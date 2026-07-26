"""Freeze the group-aware 70/15/15 split and its leakage blocklist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.integrity import assert_test_untouched
from src.data.paths import load_project_paths, load_raw_config
from src.data.split import (
    aggregate_split_stats,
    build_group_stats,
    build_split_manifest,
    build_test_blocklist,
    class_distribution_markdown,
    distribution_summary,
    grouping_markdown,
    hash_images,
    image_split_map,
    load_json,
    manifest_fingerprint,
    save_class_distribution_figure,
    stratified_group_split,
    verify_split_invariants,
    write_manifest_fingerprint,
)
from src.data.voc_to_coco import (
    DataInvariantError,
    canonical_json_bytes,
    sha256_file,
    write_canonical_json,
)


def load_spike_inputs(paths):
    coco = load_json(paths.interim / "coco_all.json")
    phash_payload = load_json(paths.interim / "phash_spike.json")
    grouping_decision = load_json(paths.interim / "h3_spike.json")
    phashes = {int(record["image_id"]): str(record["phash"]) for record in phash_payload["records"]}
    image_to_group = {
        int(record["image_id"]): int(record["group_id"])
        for record in grouping_decision["group_ids"]
    }
    return coco, phashes, grouping_decision, image_to_group


def freeze() -> str:
    paths = load_project_paths()
    path_config = load_raw_config(paths.config_path)
    grouping_config = load_raw_config(paths.grouping_config_path)
    classes = paths.classes
    fractions = {name: float(path_config["split"][name]) for name in ("train", "val", "test")}
    coco, phashes, grouping_decision, image_to_group = load_spike_inputs(paths)
    groups = build_group_stats(coco, image_to_group, classes)
    assignments = stratified_group_split(
        groups,
        classes=classes,
        fractions=fractions,
        seed=paths.seed,
    )
    image_splits = image_split_map(groups, assignments)
    split_stats = aggregate_split_stats(coco, image_splits, classes)
    verify_split_invariants(
        groups=groups,
        group_assignments=assignments,
        image_splits=image_splits,
        split_stats=split_stats,
        fractions=fractions,
        person_min_fraction=0.10,
        ratio_tolerance=float(grouping_config["guardrails"]["split_tolerance"]),
    )

    print("Hashing 5,000 source images for the frozen manifest")
    image_hashes = hash_images(
        coco,
        paths.hardhat_raw,
        workers=int(grouping_config["runtime"]["phash_workers"]),
    )
    source_checksums = load_json(paths.splits / "source_checksums.json")
    manifest = build_split_manifest(
        coco=coco,
        phash_records=phashes,
        image_hashes=image_hashes,
        image_to_group=image_to_group,
        image_splits=image_splits,
        source_checksums=source_checksums,
        grouping_decision=grouping_decision,
        fractions=fractions,
        seed=paths.seed,
    )
    manifest_path = paths.splits / "split_manifest.json"
    write_canonical_json(manifest_path, manifest)
    fingerprint = write_manifest_fingerprint(
        manifest_path,
        paths.splits / "MANIFEST.sha256",
    )
    blocklist = build_test_blocklist(manifest)
    if blocklist["manifest_sha256"] != fingerprint:
        raise DataInvariantError("Test blocklist and MANIFEST.sha256 fingerprints differ")
    write_canonical_json(paths.splits / "test_blocklist.json", blocklist)

    summary = distribution_summary(
        coco=coco,
        image_splits=image_splits,
        classes=classes,
    )
    (paths.reports / "class_distribution.md").write_text(
        class_distribution_markdown(summary),
        encoding="utf-8",
        newline="\n",
    )
    save_class_distribution_figure(
        summary,
        paths.figures / "class_distribution.png",
    )
    (paths.reports / "grouping_report.md").write_text(
        grouping_markdown(
            groups=groups,
            group_assignments=assignments,
            split_stats=split_stats,
            grouping_decision=grouping_decision,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(f"manifest SHA256 = {fingerprint}")
    print(f"split stats = {split_stats}")
    print(f"groups = {len(groups):,}; max group = {max(group.image_count for group in groups)}")
    return fingerprint


def verify() -> str:
    paths = load_project_paths()
    manifest_path = paths.splits / "split_manifest.json"
    fingerprint_path = paths.splits / "MANIFEST.sha256"
    for required in (
        manifest_path,
        fingerprint_path,
        paths.splits / "test_blocklist.json",
        paths.splits / "source_checksums.json",
        paths.reports / "class_distribution.md",
        paths.reports / "grouping_report.md",
        paths.figures / "class_distribution.png",
    ):
        if not required.is_file():
            raise DataInvariantError(f"Missing frozen-split artifact: {required}")
    manifest = load_json(manifest_path)
    canonical = canonical_json_bytes(manifest) + b"\n"
    if canonical != manifest_path.read_bytes():
        raise DataInvariantError("split_manifest.json is not canonical JSON")
    actual = manifest_fingerprint(manifest_path)
    recorded = fingerprint_path.read_text(encoding="utf-8").split()[0]
    if actual != recorded:
        raise DataInvariantError("MANIFEST.sha256 does not match split_manifest.json")
    source = load_json(paths.splits / "source_checksums.json")
    archive = paths.raw / source["archive"]["file_name"]
    if sha256_file(archive) != source["archive"]["sha256"]:
        raise DataInvariantError("Source archive checksum changed")

    image_to_group = {
        int(record["image_id"]): int(record["group_id"]) for record in manifest["images"]
    }
    image_splits = {int(record["image_id"]): str(record["split"]) for record in manifest["images"]}
    coco = load_json(paths.interim / "coco_all.json")
    groups = build_group_stats(coco, image_to_group, paths.classes)
    assignments = {group.group_id: image_splits[group.image_ids[0]] for group in groups}
    split_stats = aggregate_split_stats(coco, image_splits, paths.classes)
    verify_split_invariants(
        groups=groups,
        group_assignments=assignments,
        image_splits=image_splits,
        split_stats=split_stats,
        fractions=manifest["split"]["fractions"],
        person_min_fraction=0.10,
        ratio_tolerance=0.02,
    )
    assert_test_untouched(paths=paths)
    print(f"manifest SHA256 = {actual}")
    print(f"split stats = {split_stats}")
    print("same group -> same split = PASS")
    print("Test blocklist re-hash = PASS")
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the frozen artifacts without rewriting them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.verify:
            verify()
        else:
            freeze()
        return 0
    except (DataInvariantError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
