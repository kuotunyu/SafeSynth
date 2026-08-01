"""M16 driver: unpack what came back from Colab and audit it before trusting it.

Usage (either form):

    uv run python -m scripts.audit_colab_results --archive ~/Downloads/results_colab.zip
    uv run python -m scripts.audit_colab_results

The first form extracts the archive into results/colab/ and then audits; the
second audits whatever is already there. Exits non-zero when anything fatal is
found, so it can gate the rest of Phase 2 rather than merely inform it.

The checks live in src/training/ingest.py and are unit-tested there. This file
is only wiring: locate inputs, extract, render, decide the exit code.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import yaml

from src.data.paths import PROJECT_ROOT
from src.training.ingest import audit_colab_results, render_audit_markdown

RESULTS_DIR = PROJECT_ROOT / "results" / "colab"
REPORT_PATH = PROJECT_ROOT / "reports" / "m16_colab_audit.md"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training.yaml"


class UnsafeArchiveError(RuntimeError):
    """Raised when an archive member would write outside the target directory."""


def safe_extract(archive: Path, destination: Path) -> list[str]:
    """Extract, refusing any member that escapes `destination`.

    zipfile does not protect against `../` members or absolute paths on its own.
    This archive comes off a Colab VM rather than the open internet, so the risk
    is low, but a check that costs four lines should not be skipped on the
    strength of where a file is assumed to have come from.
    """

    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            target = (destination / member).resolve()
            if target != resolved_root and resolved_root not in target.parents:
                raise UnsafeArchiveError(f"{member!r} would extract outside {destination}")
            extracted.append(member)
        bundle.extractall(destination)
    return extracted


def load_augmentation_config(path: Path = TRAINING_CONFIG) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config.get("augmentation", {})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="results_colab.zip to extract into results/colab/ before auditing",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.archive is not None:
        if not args.archive.is_file():
            print(f"archive not found: {args.archive}")
            return 2
        members = safe_extract(args.archive, args.results_dir)
        print(f"extracted {len(members)} entries into {args.results_dir}")

    if not args.results_dir.is_dir():
        print(
            f"{args.results_dir} does not exist. Download results_colab.zip from "
            f"Drive and pass it with --archive."
        )
        return 2

    report = audit_colab_results(
        args.results_dir, augmentation_config=load_augmentation_config()
    )
    rendered = render_audit_markdown(report)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(rendered, encoding="utf-8", newline="\n")

    print(rendered)
    print(f"wrote {args.report}")

    if not report.ok:
        print(
            f"\n{len(report.fatal)} fatal finding(s). Do NOT build any table on these "
            f"results until they are resolved."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
