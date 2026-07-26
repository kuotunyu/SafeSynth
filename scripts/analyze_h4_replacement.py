"""Explain the frozen H4 context-replacement failure without retuning it."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.data.paths import load_project_paths
from src.filtering.artifact_gate import roc_auc


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _area(bbox_xywh: list[float]) -> float:
    return max(float(bbox_xywh[2]), 1) * max(float(bbox_xywh[3]), 1)


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "score_median": float(np.median([row["score"] for row in rows])),
        "source_min_side_median": float(
            np.median([row["source_min_side"] for row in rows])
        ),
        "paste_to_source_area_ratio_median": float(
            np.median([row["paste_to_source_area_ratio"] for row in rows])
        ),
        "paste_to_anchor_area_ratio_median": float(
            np.median([row["paste_to_anchor_area_ratio"] for row in rows])
        ),
        "postfx_fraction": float(np.mean([row["postfx_applied"] for row in rows])),
        "filter_pass_fraction": float(np.mean([row["filter_pass"] for row in rows])),
    }


def main() -> None:
    paths = load_project_paths()
    run_dir = paths.synthetic / "m11_h4_context_replace"
    h4 = _read_json(paths.reports / "h4_context_replacement.json")
    records = {
        str(record["sample_id"]): record
        for record in _read_jsonl(run_dir / "records.jsonl")
    }
    bank = {
        str(item["cutout_id"]): item
        for item in _read_jsonl(paths.cutouts / "bank_manifest.jsonl")
    }
    coco = _read_json(paths.interim / "coco_all.json")
    annotations = {
        int(annotation["id"]): annotation for annotation in coco["annotations"]
    }

    rows: list[dict[str, Any]] = []
    for example_id, label, score, class_name in zip(
        h4["test_example_ids"],
        h4["test_labels"],
        h4["test_scores"],
        h4["test_classes"],
        strict=True,
    ):
        if int(label) != 1:
            continue
        sample_id, instance_id = str(example_id).split(":", 1)
        record = records[sample_id]
        instance = next(
            item
            for item in record["instances"]
            if item["instance_id"] == instance_id
        )
        source = bank[str(instance["cutout_id"])]
        anchor = annotations[int(record["replacement_anchor_annotation_id"])]
        paste_area = _area(instance["bbox_xywh"])
        source_area = _area(source["src_bbox_xywh"])
        anchor_area = _area(anchor["bbox"])
        rows.append(
            {
                "sample_id": sample_id,
                "class_name": str(class_name),
                "score": float(score),
                "source_min_side": min(
                    float(source["src_bbox_xywh"][2]),
                    float(source["src_bbox_xywh"][3]),
                ),
                "paste_to_source_area_ratio": paste_area / source_area,
                "paste_to_anchor_area_ratio": paste_area / anchor_area,
                "postfx_applied": bool(record["postfx"]),
                "filter_pass": bool(record["passed"]),
            }
        )

    rows.sort(key=lambda row: float(row["score"]))
    quartile_size = max(len(rows) // 4, 1)
    bottom = rows[:quartile_size]
    top = rows[-quartile_size:]
    scores = np.asarray([row["score"] for row in rows], dtype=np.float64)
    correlations: dict[str, float] = {}
    for field in (
        "source_min_side",
        "paste_to_source_area_ratio",
        "paste_to_anchor_area_ratio",
    ):
        statistic = spearmanr(
            scores,
            np.asarray([row[field] for row in rows], dtype=np.float64),
        ).statistic
        correlations[f"score_vs_{field}_spearman"] = float(statistic)

    binary_auc: dict[str, float | None] = {}
    for field in ("postfx_applied", "filter_pass"):
        labels = np.asarray([row[field] for row in rows], dtype=np.int64)
        binary_auc[f"{field}_from_score_auc"] = (
            roc_auc(labels, scores) if len(np.unique(labels)) == 2 else None
        )

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["class_name"])].append(row)
    result = {
        "scope": "held-out pasted examples from the pre-registered replacement spike",
        "n_pasted_test_examples": len(rows),
        "bottom_score_quartile": _group_summary(bottom),
        "top_score_quartile": _group_summary(top),
        "correlations": correlations,
        "binary_score_auc": binary_auc,
        "by_class": {
            class_name: _group_summary(class_rows)
            for class_name, class_rows in sorted(by_class.items())
        },
        "interpretation": (
            "exploratory failure analysis only; no H4 threshold or method changes"
        ),
    }
    (paths.reports / "h4_context_replacement_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    low = result["bottom_score_quartile"]
    high = result["top_score_quartile"]
    lines = [
        "# H4 context-replacement failure diagnostic",
        "",
        "This is an exploratory read-only analysis of the already failed held-out",
        "scores. It cannot alter the pre-registered decision.",
        "",
        f"- Held-out pasted examples: {len(rows)}",
        (
            "- Bottom/top score-quartile median source min side: "
            f"{low['source_min_side_median']:.1f} / "
            f"{high['source_min_side_median']:.1f} px"
        ),
        (
            "- Bottom/top paste-to-source area ratio: "
            f"{low['paste_to_source_area_ratio_median']:.3f} / "
            f"{high['paste_to_source_area_ratio_median']:.3f}"
        ),
        (
            "- Bottom/top paste-to-anchor area ratio: "
            f"{low['paste_to_anchor_area_ratio_median']:.3f} / "
            f"{high['paste_to_anchor_area_ratio_median']:.3f}"
        ),
        (
            "- Bottom/top whole-image post-effect fraction: "
            f"{low['postfx_fraction']:.3f} / {high['postfx_fraction']:.3f}"
        ),
        "",
        "The ranked patch grid should be treated as the primary qualitative",
        "evidence. These correlations only help distinguish size/resampling and",
        "post-effect associations; they do not establish a causal fix.",
        "",
    ]
    (paths.reports / "h4_context_replacement_diagnostic.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
