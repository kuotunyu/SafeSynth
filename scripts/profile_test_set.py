"""EVAL-08 + EVAL-17: profile the frozen Test split before any model exists.

This is the highest-value thing that can be computed with no checkpoint at all.
AP_small is headline metric #1, and if the frozen Test split holds very few
`small` instances then AP_small is measuring noise — EVAL-08 requires that to be
stated in the report rather than discovered after the analysis is written.

Reading Test here is legitimate and required: EVAL-08 asks for exactly these
counts. Nothing produced by this script feeds a generator, a filter or a
threshold — it reads ground-truth statistics and writes a markdown report. The
frozen split is treated as read-only.

Usage:
    uv run python -m scripts.profile_test_set
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.paths import load_project_paths
from src.evaluation.slices import (
    EVALUATION_CONFIG,
    assert_areas_in_annotation_space,
    build_profile,
    image_mean_luminances,
    load_slice_config,
    render_profile_markdown,
)
from src.training.arms import split_real_images
from src.training.data import load_coco_samples

SPLIT_NAME = "test"
REPORT_NAME = "test_set_profile.md"

# Every way a bad input file can surface: a missing/unreadable path (OSError), a
# malformed JSON document (json.JSONDecodeError, a ValueError), a document whose
# shape is wrong (KeyError, IndexError) or whose types are wrong (TypeError).
# These become SystemExit with the offending path named, because a traceback
# through split_real_images tells the operator nothing about which file to fix.
_INPUT_ERRORS = (OSError, ValueError, KeyError, IndexError, TypeError)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    paths = load_project_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=paths.splits / "split_manifest.json")
    parser.add_argument("--annotations", type=Path, default=paths.interim / "coco_all.json")
    parser.add_argument("--images-root", type=Path, default=paths.hardhat_raw / "images")
    parser.add_argument("--output", type=Path, default=paths.reports / REPORT_NAME)
    # Defaulting rather than branching in main(): a `None` default would give
    # main() a config-loading path that only a real invocation ever takes, which
    # is the one path a test suite can least afford to leave unexecuted.
    parser.add_argument("--config", type=Path, default=EVALUATION_CONFIG)
    return parser.parse_args(argv)


def load_test_samples(manifest: Path, annotations: Path, images_root: Path):
    """Load exactly the frozen Test images, and prove that is what happened."""

    try:
        splits = split_real_images(manifest)
    except _INPUT_ERRORS as error:
        raise SystemExit(
            f"Cannot read the frozen split manifest {manifest}: {type(error).__name__}: {error}"
        ) from error
    if SPLIT_NAME not in splits:
        raise SystemExit(f"{manifest} declares no {SPLIT_NAME!r} split")
    expected = set(splits[SPLIT_NAME])

    try:
        payload = json.loads(Path(annotations).read_text(encoding="utf-8"))
    except _INPUT_ERRORS as error:
        raise SystemExit(
            f"Cannot read the COCO annotations {annotations}: {type(error).__name__}: {error}"
        ) from error
    # EVAL-07: the source COCO must agree with its own boxes, or the areas we are
    # about to bucket are in a different coordinate space than the boxes. This
    # raises AnnotationSpaceError, which is deliberately NOT caught here.
    assert_areas_in_annotation_space(payload)

    try:
        samples = load_coco_samples(annotations, images_root, keep_names=sorted(expected))
    except _INPUT_ERRORS as error:
        raise SystemExit(
            f"Cannot build samples from {annotations}: {type(error).__name__}: {error}"
        ) from error
    loaded = {sample.image_path.name for sample in samples}
    if loaded != expected:
        raise SystemExit(
            f"Loaded {len(loaded)} images but the frozen {SPLIT_NAME} split names "
            f"{len(expected)}; missing={sorted(expected - loaded)[:5]} "
            f"unexpected={sorted(loaded - expected)[:5]}"
        )
    return samples


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_slice_config(args.config)

    samples = load_test_samples(args.manifest, args.annotations, args.images_root)
    print(f"Loaded {len(samples)} frozen {SPLIT_NAME} images from {args.annotations}")

    luminances = image_mean_luminances(samples)
    profile = build_profile(samples, luminance_by_image=luminances, config=config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_profile_markdown(profile), encoding="utf-8", newline="\n")

    verdict = profile["verdict"]
    print(json.dumps(profile["bucket_counts"], indent=2))
    print(json.dumps(profile["slice_counts"], indent=2))
    print(
        f"AP_small verdict: {verdict['n_small_primary']} primary-class small instances "
        f"({100 * verdict['small_share']:.1f}% of {verdict['n_primary_total']}), "
        f"floor={verdict['min_instances']}, thin={verdict['is_thin']} "
        f"(below_floor={verdict['is_below_floor']}, "
        f"below_even_share={verdict['is_below_even_share']}); worst-case half width "
        f"±{verdict['single_arm_half_width_points']:.2f} points for one arm, "
        f"±{verdict['two_arm_gap_floor_points']:.2f} points for a two-arm gap"
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
