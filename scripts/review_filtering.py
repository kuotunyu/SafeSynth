"""Verify M12's filter ledger and render 12 pass versus 12 reject samples."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from src.data.paths import PROJECT_ROOT, load_project_paths

COLORS = {
    "helmet": (255, 196, 0),
    "head": (230, 70, 70),
    "person": (30, 170, 240),
}
GRID_COLUMNS = 6
TILE_SIZE = 208
CAPTION_HEIGHT = 38
SECTION_HEIGHT = 28


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _diverse(records: list[dict[str, Any]], *, key: str, n: int) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for record in sorted(records, key=lambda item: str(item["sample_id"])):
        buckets[str(record[key])].append(record)
    selected: list[dict[str, Any]] = []
    names = sorted(buckets)
    while len(selected) < n and names:
        next_names: list[str] = []
        for name in names:
            if buckets[name] and len(selected) < n:
                selected.append(buckets[name].popleft())
            if buckets[name]:
                next_names.append(name)
        names = next_names
    if len(selected) != n:
        raise RuntimeError(f"Need {n} review samples, found {len(selected)}")
    return selected


def _render_tile(run_dir: Path, record: dict[str, Any]) -> Image.Image:
    image = Image.open(run_dir / record["file_name"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    for instance in record["instances"]:
        if not instance.get("kept", True):
            continue
        x, y, width, height = (float(value) for value in instance["bbox_xywh"])
        color = COLORS[str(instance["class_name"])]
        line_width = 3 if instance["kind"] == "pasted" else 1
        draw.rectangle(
            (x, y, x + width, y + height),
            outline=color,
            width=line_width,
        )
    image.thumbnail((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TILE_SIZE, TILE_SIZE + CAPTION_HEIGHT), "white")
    x_offset = (TILE_SIZE - image.width) // 2
    y_offset = (TILE_SIZE - image.height) // 2
    canvas.paste(image, (x_offset, y_offset))
    caption = (
        "PASS"
        if record["passed"]
        else str(record["first_reject_reason"]).replace("_", " ")
    )
    caption = caption[:31]
    tile_draw = ImageDraw.Draw(canvas)
    tile_draw.text(
        (4, TILE_SIZE + 3),
        f"{record['sample_id']} | {record['scenario']}",
        fill=(25, 25, 25),
    )
    tile_draw.text(
        (4, TILE_SIZE + 19),
        caption,
        fill=(25, 120, 45) if record["passed"] else (190, 45, 35),
    )
    return canvas


def _render_section(
    run_dir: Path,
    title: str,
    records: list[dict[str, Any]],
) -> Image.Image:
    rows = (len(records) + GRID_COLUMNS - 1) // GRID_COLUMNS
    section = Image.new(
        "RGB",
        (
            GRID_COLUMNS * TILE_SIZE,
            SECTION_HEIGHT + rows * (TILE_SIZE + CAPTION_HEIGHT),
        ),
        (242, 244, 247),
    )
    ImageDraw.Draw(section).text((6, 7), title, fill=(15, 15, 15))
    for index, record in enumerate(records):
        x = (index % GRID_COLUMNS) * TILE_SIZE
        y = SECTION_HEIGHT + (index // GRID_COLUMNS) * (
            TILE_SIZE + CAPTION_HEIGHT
        )
        section.paste(_render_tile(run_dir, record), (x, y))
    return section


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="m11_h4_seed42")
    args = parser.parse_args()

    paths = load_project_paths()
    run_dir = paths.synthetic / args.run_tag
    records = _read_jsonl(run_dir / "records.jsonl")
    summary = _read_json(run_dir / "summary.json")
    filter_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "filtering.yaml").read_text(encoding="utf-8")
    )
    allowed_reasons = set(filter_config["reject_reasons"])
    passed = [record for record in records if record["passed"]]
    rejected = [record for record in records if not record["passed"]]
    first_reasons = Counter(
        str(record["first_reject_reason"]) for record in rejected
    )
    unknown_reasons = sorted(
        {
            str(reason)
            for record in rejected
            for reason in record["reject_reasons"]
            if reason not in allowed_reasons
        }
    )
    empty_rejections = [
        str(record["sample_id"])
        for record in rejected
        if not record["reject_reasons"]
    ]
    checks = {
        "records_equal_summary_total": len(records) == int(summary["n_images"]),
        "pass_plus_reject_equal_total": len(passed) + len(rejected) == len(records),
        "pass_count_equal_summary": len(passed) == int(summary["passed"]),
        "reject_count_equal_summary": len(rejected) == int(summary["rejected"]),
        "first_reason_funnel_equal_summary": dict(sorted(first_reasons.items()))
        == summary["first_reject_reasons"],
        "rejected_reasons_nonempty": not empty_rejections,
        "all_reasons_in_enum": not unknown_reasons,
    }
    if not all(checks.values()):
        raise RuntimeError(f"M12 ledger check failed: {checks}")

    passed_review = _diverse(passed, key="scenario", n=12)
    rejected_review = _diverse(rejected, key="first_reject_reason", n=12)
    pass_section = _render_section(run_dir, "12 PASSED samples", passed_review)
    reject_section = _render_section(run_dir, "12 REJECTED samples", rejected_review)
    grid = Image.new(
        "RGB",
        (pass_section.width, pass_section.height + reject_section.height),
        "white",
    )
    grid.paste(pass_section, (0, 0))
    grid.paste(reject_section, (0, pass_section.height))
    figure_path = paths.figures / "filter_pass_reject_grid.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(figure_path, optimize=True)

    result = {
        "source_run": str(run_dir),
        "n_total": len(records),
        "n_pass": len(passed),
        "n_reject": len(rejected),
        "first_reject_reasons": dict(sorted(first_reasons.items())),
        "checks": checks,
        "review_pass_sample_ids": [item["sample_id"] for item in passed_review],
        "review_reject_sample_ids": [item["sample_id"] for item in rejected_review],
        "review_figure": str(figure_path),
    }
    (paths.reports / "filter_ledger.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# M12 filter ledger verification",
        "",
        f"- Source: `{run_dir}`",
        f"- Total / pass / reject: {len(records)} / {len(passed)} / {len(rejected)}",
        f"- First-reason funnel: `{dict(sorted(first_reasons.items()))}`",
        "- All seven ledger and enum checks: **PASS**",
        f"- Human-review grid: `{figure_path}`",
        "",
    ]
    (paths.reports / "filter_ledger.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
