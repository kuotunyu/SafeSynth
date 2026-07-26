"""One-at-a-time threshold sensitivity analysis for the synthetic filter."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.filtering.rules import filter_sample


@dataclass(frozen=True)
class NumericLeaf:
    path: tuple[str | int, ...]
    value: float


def numeric_rule_leaves(
    value: Any, path: tuple[str | int, ...] = ()
) -> tuple[NumericLeaf, ...]:
    """Enumerate numeric thresholds, excluding booleans and enum-like settings."""

    leaves: list[NumericLeaf] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            leaves.extend(numeric_rule_leaves(value[key], (*path, str(key))))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            leaves.extend(numeric_rule_leaves(item, (*path, index)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        leaves.append(NumericLeaf(path=path, value=float(value)))
    return tuple(leaves)


def _set_path(root: dict[str, Any], path: tuple[str | int, ...], value: float) -> None:
    current: Any = root
    for part in path[:-1]:
        current = current[part]
    final = path[-1]
    original = current[final]
    current[final] = round(value) if isinstance(original, int) else value


def acceptance_rate(
    samples: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> float:
    if not samples:
        raise ValueError("Threshold sensitivity requires at least one sample")
    passed = sum(
        filter_sample(sample, config, check_invariants=False).passed
        for sample in samples
    )
    return passed / len(samples)


def analyze_threshold_sensitivity(
    samples: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    perturbation: float = 0.20,
) -> dict[str, Any]:
    """Perturb each numeric rule leaf independently and recompute acceptance."""

    if not 0 < perturbation < 1:
        raise ValueError("perturbation must lie in (0, 1)")
    baseline = acceptance_rate(samples, config)
    rows: list[dict[str, Any]] = []
    for leaf in numeric_rule_leaves(config["rules"], ("rules",)):
        rates: dict[str, float] = {}
        effective_values: dict[str, float | int] = {}
        for label, factor in (("minus_20pct", 1 - perturbation), ("plus_20pct", 1 + perturbation)):
            modified = copy.deepcopy(config)
            _set_path(modified, leaf.path, leaf.value * factor)
            effective: Any = modified
            for part in leaf.path:
                effective = effective[part]
            effective_values[label] = effective
            rates[label] = acceptance_rate(samples, modified)
        maximum_swing_points = (
            max(abs(rate - baseline) for rate in rates.values()) * 100
        )
        rows.append(
            {
                "path": ".".join(str(part) for part in leaf.path),
                "baseline_value": leaf.value,
                **effective_values,
                "baseline_acceptance": baseline,
                **rates,
                "max_swing_percentage_points": maximum_swing_points,
                "alarm": maximum_swing_points
                > float(config["sensitivity_alarm_points"]),
            }
        )
    rows.sort(
        key=lambda row: (-float(row["max_swing_percentage_points"]), str(row["path"]))
    )
    return {
        "n_samples": len(samples),
        "perturbation": perturbation,
        "baseline_acceptance": baseline,
        "alarm_points": float(config["sensitivity_alarm_points"]),
        "alarm_count": sum(bool(row["alarm"]) for row in rows),
        "rows": rows,
    }
