"""Export the M13 training subsets from one generated pool (COMP-26..28, FILT-13).

Two experiment-design traps are handled here, and both are easy to get wrong in
a way that silently invalidates the whole four-arm comparison:

1. **Size matching.** Comparing "everything we generated" against "what survived
   the filter" confounds *more data* with *better data*. The unfiltered arm is
   therefore a uniform random sample of the SAME SIZE drawn from the whole pool,
   not the whole pool (COMP-26).

2. **Nested scaling.** 0.5x and 1x are drawn as nested prefixes of one stable
   ranking, stratified by scenario, so 0.5x is a strict subset of 1x and the
   scenario mix is identical at both sizes. Three independent draws would let
   sampling noise leak into the very trend the ablation is meant to measure
   (COMP-27). 2x is deliberately not produced: H4 did not pass, so ADR-011 caps
   accepted samples at 1x.

Pixels are written exactly once by the compositor. This script only emits COCO
JSONs that point at those files, so the ablation is "same generator, different
acceptance mask" and there is no second copy on disk (FILT-13).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.data.paths import load_project_paths

TARGET_ACCEPTED_1X = 3_500
FRACTIONS = (0.5, 1.0)


def _stable_rank(sample_id: str, seed: int) -> float:
    """Deterministic uniform rank; independent of iteration or filesystem order."""

    digest = hashlib.sha256(f"{seed}|{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _stratified_prefix(
    records: list[dict[str, Any]], *, size: int, seed: int
) -> list[dict[str, Any]]:
    """Take `size` records, keeping the scenario mix of the input.

    Ranking inside each scenario and slicing a prefix is what makes the smaller
    size a strict subset of the larger one.
    """

    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scenario[str(record["scenario"])].append(record)
    for items in by_scenario.values():
        items.sort(key=lambda item: _stable_rank(str(item["sample_id"]), seed))

    total = len(records)
    chosen: list[dict[str, Any]] = []
    quotas: dict[str, int] = {}
    for scenario, items in by_scenario.items():
        quotas[scenario] = min(len(items), round(size * len(items) / total))
    # Rounding can miss or overshoot the target by a few; settle the remainder on
    # the scenarios that still have material, largest pool first.
    while sum(quotas.values()) != size:
        delta = 1 if sum(quotas.values()) < size else -1
        candidates = sorted(
            by_scenario,
            key=lambda name: (len(by_scenario[name]) - quotas[name]) * delta,
            reverse=True,
        )
        for scenario in candidates:
            new_value = quotas[scenario] + delta
            if 0 <= new_value <= len(by_scenario[scenario]):
                quotas[scenario] = new_value
                break
        else:  # pragma: no cover - only reachable if every scenario is exhausted
            raise RuntimeError("Cannot settle stratified quota")
    for scenario, quota in quotas.items():
        chosen.extend(by_scenario[scenario][:quota])
    chosen.sort(key=lambda item: str(item["sample_id"]))
    return chosen


def _coco_subset(
    coco: dict[str, Any], sample_ids: set[str], *, description: str
) -> dict[str, Any]:
    images = [image for image in coco["images"] if str(image["sample_id"]) in sample_ids]
    keep_image_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in coco["annotations"]
        if int(annotation["image_id"]) in keep_image_ids
    ]
    info = dict(coco["info"])
    info["description"] = description
    return {
        "info": info,
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export(*, pool_tag: str, seed: int) -> dict[str, Any]:
    paths = load_project_paths()
    pool_dir = paths.synthetic / pool_tag
    records = [
        json.loads(line)
        for line in (pool_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    coco = json.loads((pool_dir / "annotations.json").read_text(encoding="utf-8"))

    accepted = [record for record in records if record["passed"]]
    if len(accepted) < TARGET_ACCEPTED_1X:
        raise RuntimeError(
            f"Pool yielded {len(accepted)} accepted samples, below the {TARGET_ACCEPTED_1X} "
            "needed for 1x. Generate a larger pool rather than lowering the target."
        )

    outputs: dict[str, Any] = {}
    filtered_1x = _stratified_prefix(accepted, size=TARGET_ACCEPTED_1X, seed=seed)
    filtered_ids = {str(record["sample_id"]) for record in filtered_1x}

    # COMP-26: the unfiltered arm is size-matched, drawn from the WHOLE pool.
    unfiltered_1x = _stratified_prefix(records, size=TARGET_ACCEPTED_1X, seed=seed + 1)
    unfiltered_ids = {str(record["sample_id"]) for record in unfiltered_1x}

    for fraction in FRACTIONS:
        size = round(TARGET_ACCEPTED_1X * fraction)
        tag = f"{fraction:g}x".replace(".", "_")
        for arm, source in (("filtered", filtered_1x), ("unfiltered", unfiltered_1x)):
            subset = (
                source
                if fraction == 1.0
                else _stratified_prefix(source, size=size, seed=seed)
            )
            ids = {str(record["sample_id"]) for record in subset}
            path = pool_dir / f"annotations_{arm}_{tag}.json"
            sha256 = _write_json(
                path,
                _coco_subset(
                    coco,
                    ids,
                    description=f"SafeSynth M13 {arm} {fraction:g}x (ADR-011: 1x cap)",
                ),
            )
            outputs[f"{arm}_{tag}"] = {
                "file": path.name,
                "n_images": len(ids),
                "sha256": sha256,
                "scenarios": dict(
                    Counter(str(record["scenario"]) for record in subset)
                ),
            }

    checks = {
        "filtered_and_unfiltered_are_size_matched": (
            outputs["filtered_1x"]["n_images"] == outputs["unfiltered_1x"]["n_images"]
        ),
        "filtered_is_subset_of_accepted": filtered_ids
        <= {str(record["sample_id"]) for record in accepted},
        "no_2x_emitted": all("2x" not in key for key in outputs),
        "unfiltered_is_subset_of_pool": unfiltered_ids
        <= {str(record["sample_id"]) for record in records},
    }
    for fraction_tag in ("0_5x",):
        for arm in ("filtered", "unfiltered"):
            small = _read_ids(pool_dir / f"annotations_{arm}_{fraction_tag}.json")
            large = _read_ids(pool_dir / f"annotations_{arm}_1x.json")
            checks[f"{arm}_{fraction_tag}_nested_in_1x"] = small <= large
    if not all(checks.values()):
        raise RuntimeError(f"Subset invariants failed: {checks}")

    return {
        "acceptance_rate": len(accepted) / len(records),
        "checks": checks,
        "n_accepted": len(accepted),
        "n_pool": len(records),
        "outputs": outputs,
        "pool_tag": pool_tag,
    }


def _read_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(image["sample_id"]) for image in payload["images"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-tag", default="m13_pool_1x")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(export(pool_tag=args.pool_tag, seed=args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
