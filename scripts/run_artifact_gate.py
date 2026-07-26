"""Run M11's group-disjoint paste-artifact classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.filtering.artifact_gate import (
    build_patch_examples,
    train_artifact_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="m11_h4_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match-person-context", action="store_true")
    parser.add_argument("--report-tag", default="h4_artifact_gate")
    return parser.parse_args()


def _render_roc(labels: np.ndarray, scores: np.ndarray, output: Path) -> None:
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    false_positive: list[float] = []
    true_positive: list[float] = []
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    for threshold in thresholds:
        predicted = scores >= threshold
        true_positive.append(float((predicted & (labels == 1)).sum()) / positives)
        false_positive.append(float((predicted & (labels == 0)).sum()) / negatives)
    figure, axis = plt.subplots(figsize=(5.2, 5.2))
    axis.plot(false_positive, true_positive, color="#d95f02", linewidth=2)
    axis.plot((0, 1), (0, 1), "--", color="#666666")
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="H4 paste-artifact ROC",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    paths = load_project_paths()
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "filtering.yaml").read_text(encoding="utf-8")
    )
    run_dir = paths.synthetic / args.run_tag
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    if int(summary["n_images"]) != 300:
        raise RuntimeError("H4 requires exactly 300 generated images")
    try:
        examples = build_patch_examples(
            paths=paths,
            run_dir=run_dir,
            config=config,
            seed=args.seed,
            match_person_context=args.match_person_context,
        )
    except ValueError as error:
        if not args.match_person_context:
            raise
        result = {
            "run_dir": str(run_dir),
            "seed": args.seed,
            "person_context_matching": True,
            "status": "infeasible",
            "reason": str(error),
            "passed": False,
        }
        (paths.reports / f"{args.report_tag}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        lines = [
            "# Spike H4 — context-matched feasibility",
            "",
            "- Status: **INFEASIBLE**",
            f"- Frozen seed: `{args.seed}`",
            f"- Reason: `{error}`",
            "",
            "The pre-registration forbids fold reselection or silently relaxing",
            "class/fold/context matching. The original H4 failure remains binding.",
            "",
        ]
        (paths.reports / f"{args.report_tag}.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
            newline="\n",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(3) from error
    result = train_artifact_classifier(
        examples,
        seed=args.seed,
        bootstrap_samples=int(config["artifact_gate"]["bootstrap_samples"]),
        logistic_c=float(config["artifact_gate"]["logistic_c"]),
    )
    threshold = float(config["artifact_gate"]["max_auc_for_scaleup"])
    result.update(
        {
            "run_dir": str(run_dir),
            "max_auc_for_scaleup": threshold,
            "passed": float(result["auc"]) <= threshold,
            "split": "group-disjoint 4/5 train, 1/5 test",
            "real_control_matching": (
                "same class + same fold + nearest log(pixel width, pixel height)"
            ),
            "labels": {"0": "real", "1": "pasted"},
            "person_context_matching": bool(args.match_person_context),
        }
    )
    output_json = paths.reports / f"{args.report_tag}.json"
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _render_roc(
        np.asarray(result["test_labels"]),
        np.asarray(result["test_scores"]),
        paths.figures / f"{args.report_tag}_roc.png",
    )
    status = "PASS — scale-up gate open" if result["passed"] else "FAIL — fix blending first"
    lines = [
        "# Spike H4 — paste-artifact detectability",
        "",
        f"- Source run: `{run_dir}` (300 images)",
        (
            f"- Examples: {result['n_examples']} "
            f"({result['n_train']} train / {result['n_test']} group-disjoint test)"
        ),
        f"- HOG + HSV logistic-regression AUC: **{result['auc']:.4f}**",
        f"- Bootstrap 95% CI: {result['auc_ci95'][0]:.4f}–{result['auc_ci95'][1]:.4f}",
        f"- Scale-up maximum AUC: {threshold:.2f}",
        f"- Decision: **{status}**",
        "",
        (
            "Real controls match class, H4 fold, "
            + (
                "person-context state, and "
                if args.match_person_context
                else ""
            )
            + "nearest log pixel width/height."
        ),
        "Both labels use the same frozen-group hash, so video near-duplicates cannot",
        "cross the split; fold-level class and target-resolution counts are paired.",
        "",
    ]
    (paths.reports / f"{args.report_tag}.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "auc": result["auc"],
                "auc_ci95": result["auc_ci95"],
                "passed": result["passed"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
