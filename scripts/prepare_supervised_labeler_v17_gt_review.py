"""Freeze and render the independent v17 GT-only review pool."""

from __future__ import annotations

import scripts.prepare_supervised_labeler_v15_gt_review as engine
from src.data.paths import PROJECT_ROOT

VERSION = 17
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "supervised_labeler_v17_gt_review.yaml"
)
POOL_PATH = PROJECT_ROOT / "splits" / "supervised_labeler_v17_gt_pool.json"
EVIDENCE_PATH = (
    PROJECT_ROOT / "reports" / "supervised_labeler_v17_gt_review.json"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "supervised_labeler_v17_gt_review"
)


def main() -> None:
    engine.VERSION = VERSION
    engine.CONFIG_PATH = CONFIG_PATH
    engine.POOL_PATH = POOL_PATH
    engine.EVIDENCE_PATH = EVIDENCE_PATH
    engine.FIGURE_DIR = FIGURE_DIR
    engine.main()


if __name__ == "__main__":
    main()
