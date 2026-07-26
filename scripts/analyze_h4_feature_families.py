"""Decompose the failed H4 shortcut into HOG and HSV feature families."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from src.data.paths import PROJECT_ROOT, load_project_paths
from src.filtering.artifact_gate import (
    PatchExample,
    build_patch_examples,
    train_artifact_classifier,
)

HSV_FEATURES = 3 * 8


def _slice_examples(
    examples: list[PatchExample],
    *,
    feature_family: str,
) -> list[PatchExample]:
    if feature_family == "hog":
        selector = slice(None, -HSV_FEATURES)
    elif feature_family == "hsv":
        selector = slice(-HSV_FEATURES, None)
    elif feature_family == "hog+hsv":
        selector = slice(None)
    else:
        raise ValueError(f"Unknown feature family: {feature_family}")
    return [
        replace(example, feature=example.feature[selector])
        for example in examples
    ]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    paths = load_project_paths()
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "filtering.yaml").read_text(encoding="utf-8")
    )
    seed = 42
    examples = build_patch_examples(
        paths=paths,
        run_dir=paths.synthetic / "m11_h4_seed42",
        config=config,
        seed=seed,
    )
    expected = _read_json(paths.reports / "h4_artifact_gate.json")
    results: dict[str, dict[str, Any]] = {}
    for feature_family in ("hog+hsv", "hog", "hsv"):
        result = train_artifact_classifier(
            _slice_examples(examples, feature_family=feature_family),
            seed=seed,
            bootstrap_samples=int(config["artifact_gate"]["bootstrap_samples"]),
            logistic_c=float(config["artifact_gate"]["logistic_c"]),
        )
        results[feature_family] = {
            key: result[key]
            for key in (
                "auc",
                "auc_ci95",
                "feature_dimensions",
                "n_train",
                "n_test",
            )
        }
    if abs(float(results["hog+hsv"]["auc"]) - float(expected["auc"])) > 1e-9:
        raise AssertionError("Combined-feature diagnostic did not reproduce frozen H4")

    output = {
        "scope": "post-failure diagnostic on the frozen H4 run and fold",
        "decision_effect": "none; the registered combined-feature gate remains binding",
        "results": results,
    }
    (paths.reports / "h4_feature_family_diagnostic.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# H4 feature-family diagnostic",
        "",
        "This is a post-failure diagnostic on the frozen run and fold. The combined",
        "HOG+HSV classifier remains the registered gate; weaker feature subsets",
        "cannot be selected to pass.",
        "",
        "| features | dimensions | AUC | bootstrap 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for feature_family, result in results.items():
        lines.append(
            f"| {feature_family} | {result['feature_dimensions']} | "
            f"{result['auc']:.4f} | "
            f"{result['auc_ci95'][0]:.4f}–{result['auc_ci95'][1]:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpret the stronger subset as an engineering clue only. It does not",
            "change the AUC 0.60 maximum or reopen M13.",
            "",
        ]
    )
    (paths.reports / "h4_feature_family_diagnostic.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
