"""Measure whether person-context anchoring reduces the existing H4 shortcut."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.paths import load_project_paths
from src.filtering.artifact_gate import has_person_context, roc_auc


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentiles(values: list[float]) -> list[float]:
    return np.percentile(values, (10, 50, 90)).astype(float).tolist()


def main() -> None:
    paths = load_project_paths()
    run_dir = paths.synthetic / "m11_h4_seed42"
    h4 = _read_json(paths.reports / "h4_artifact_gate.json")
    records = {
        str(record["sample_id"]): record
        for record in _read_jsonl(run_dir / "records.jsonl")
    }
    scores: dict[str, list[float]] = {
        "anchored_pasted": [],
        "unanchored_pasted": [],
        "real": [],
    }
    for example_id, label, score, class_name in zip(
        h4["test_example_ids"],
        h4["test_labels"],
        h4["test_scores"],
        h4["test_classes"],
        strict=True,
    ):
        if class_name not in {"head", "helmet"}:
            continue
        if int(label) == 0:
            scores["real"].append(float(score))
            continue
        sample_id, instance_id = str(example_id).split(":", 1)
        record = records[sample_id]
        instance = next(
            item
            for item in record["instances"]
            if item["instance_id"] == instance_id
        )
        person_boxes = [
            item["bbox_xywh"]
            for item in record["instances"]
            if item["class_name"] == "person" and item.get("kept", True)
        ]
        key = (
            "anchored_pasted"
            if has_person_context(instance["bbox_xywh"], person_boxes)
            else "unanchored_pasted"
        )
        scores[key].append(float(score))

    result: dict[str, Any] = {
        "source_h4": str(paths.reports / "h4_artifact_gate.json"),
        "scope": "held-out head/helmet examples; existing H4 classifier scores",
        "counts": {key: len(values) for key, values in scores.items()},
        "score_p10_p50_p90": {
            key: _percentiles(values) for key, values in scores.items()
        },
    }
    for key in ("anchored_pasted", "unanchored_pasted"):
        labels = np.r_[
            np.ones(len(scores[key]), dtype=np.int64),
            np.zeros(len(scores["real"]), dtype=np.int64),
        ]
        combined_scores = np.r_[scores[key], scores["real"]]
        result[f"{key}_vs_real_score_auc"] = roc_auc(labels, combined_scores)
    output_json = paths.reports / "h4_context_diagnostic.json"
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# H4 person-context diagnostic",
        "",
        "This is a read-only stratification of the already held-out H4 scores,",
        "not a retrained classifier or a replacement scale-up gate.",
        "",
        f"- Anchored pasted headlike patches: {result['counts']['anchored_pasted']}",
        f"- Unanchored pasted headlike patches: {result['counts']['unanchored_pasted']}",
        f"- Real headlike controls: {result['counts']['real']}",
        (
            "- Anchored-vs-real score AUC: "
            f"**{result['anchored_pasted_vs_real_score_auc']:.4f}**"
        ),
        (
            "- Unanchored-vs-real score AUC: "
            f"**{result['unanchored_pasted_vs_real_score_auc']:.4f}**"
        ),
        "",
        "Anchoring is associated with a smaller shortcut, but the anchored sample",
        "is small and still well above the 0.60 gate. Treat this only as the",
        "pre-registration clue for a new context-anchored composition spike.",
        "",
    ]
    (paths.reports / "h4_context_diagnostic.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
