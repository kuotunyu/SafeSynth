"""Generator-level regression tests for public path serialization."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

from scripts import prepare_guarded_identity_preflight as guarded
from scripts import prepare_paired_person_preflight as paired
from scripts import review_filtering
from src.release.public_paths import LOCAL_ONLY_NOT_PUBLISHED


def _public_paths(repository: Path, external_data: Path) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=repository,
        synthetic=external_data / "synthetic",
        interim=external_data / "interim",
        reports=repository / "reports",
        figures=repository / "reports" / "figures",
    )


def test_filter_ledger_uses_one_portable_source_in_json_and_markdown(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repository"
    external_data = tmp_path / "external-data"
    paths = _public_paths(repository, external_data)
    run_dir = paths.synthetic / "review-run"
    run_dir.mkdir(parents=True)
    paths.reports.mkdir(parents=True)
    (repository / "configs").mkdir(parents=True)
    (repository / "configs" / "filtering.yaml").write_text(
        yaml.safe_dump({"reject_reasons": ["too_small"]}), encoding="utf-8"
    )
    Image.new("RGB", (16, 16), "white").save(run_dir / "sample.png")
    records = []
    for index in range(24):
        passed = index < 12
        records.append(
            {
                "sample_id": f"sample-{index:02d}",
                "scenario": "baseline",
                "passed": passed,
                "first_reject_reason": None if passed else "too_small",
                "reject_reasons": [] if passed else ["too_small"],
                "instances": [],
                "file_name": "sample.png",
            }
        )
    (run_dir / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_images": 24,
                "passed": 12,
                "rejected": 12,
                "first_reject_reasons": {"too_small": 12},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(review_filtering, "PROJECT_ROOT", repository)
    monkeypatch.setattr(review_filtering, "load_project_paths", lambda: paths)
    monkeypatch.setattr(
        sys, "argv", ["review_filtering.py", "--run-tag", "review-run"]
    )

    review_filtering.main()

    json_text = (paths.reports / "filter_ledger.json").read_text(encoding="utf-8")
    markdown = (paths.reports / "filter_ledger.md").read_text(encoding="utf-8")
    payload = json.loads(json_text)
    raw_source = str(run_dir)
    assert raw_source not in json_text
    assert raw_source not in markdown
    assert payload["source_run"] == LOCAL_ONLY_NOT_PUBLISHED
    assert f"Source: `{payload['source_run']}`" in markdown
    assert payload["review_figure"] == "reports/figures/filter_pass_reject_grid.png"


def test_guarded_preflight_uses_portable_output_in_json_and_markdown(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repository"
    external_data = tmp_path / "external-data"
    paths = _public_paths(repository, external_data)
    paths.reports.mkdir(parents=True)
    (repository / "configs").mkdir(parents=True)
    (repository / "configs" / "generative_inpaint.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "guarded_v5_preregistered_no_output",
                "pilot": {
                    "architecture": "guarded_context_replacement_v5",
                    "n_images": 64,
                    "root_seed": 20260731,
                    "input_preflight_issue_max_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = paths.synthetic / "guarded-output"
    output_dir.mkdir(parents=True)

    def fake_generate(*, generative_inpainter, **_kwargs):
        records = []
        for index in range(64):
            seed = index + 1
            draft = np.zeros((4, 4, 3), dtype=np.uint8)
            mask = np.zeros((4, 4), dtype=bool)
            generative_inpainter.by_seed[seed] = guarded.InputEvidence(
                draft, mask, "helmet", seed
            )
            records.append(
                {
                    "sample_id": f"guarded-{seed:02d}",
                    "background": {"image_id": seed},
                    "replacement_anchor_annotation_id": seed,
                    "instances": [
                        {
                            "cutout_id": f"cutout-{seed:02d}",
                            "bbox_xywh": [0, 0, 2, 2],
                            "generative_inpaint": {"seed": seed},
                        }
                    ],
                    "context_replacement_input_guard": {},
                }
            )
        (output_dir / "records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        return {"output_dir": str(output_dir)}

    def fake_render(*, output_path: Path, **_kwargs) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"guarded-sheet")

    monkeypatch.setattr(guarded, "PROJECT_ROOT", repository)
    monkeypatch.setattr(guarded, "load_project_paths", lambda: paths)
    monkeypatch.setattr(guarded, "generate", fake_generate)
    monkeypatch.setattr(guarded, "render_input_sheet", fake_render)

    guarded.main()

    json_text = (paths.reports / "h4_guarded_input_preflight.json").read_text(
        encoding="utf-8"
    )
    markdown = (paths.reports / "h4_guarded_input_preflight.md").read_text(
        encoding="utf-8"
    )
    payload = json.loads(json_text)
    raw_output = str(output_dir)
    assert raw_output not in json_text
    assert raw_output not in markdown
    assert payload["output_dir"] == LOCAL_ONLY_NOT_PUBLISHED
    assert f"Output directory: `{payload['output_dir']}`" in markdown
    assert payload["contact_sheet"] == "reports/figures/h4_guarded_input_preflight.png"


def test_paired_preflight_uses_portable_output_in_json_and_markdown(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repository"
    external_data = tmp_path / "external-data"
    paths = _public_paths(repository, external_data)
    paths.reports.mkdir(parents=True)
    (repository / "configs").mkdir(parents=True)
    seed = 20260802
    (repository / "configs" / "paired_person_preflight.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "candidate_v7_cpu_preflight_no_model_output",
                "architecture": "paired_person_scene_position_insert_v7",
                "n_images": 64,
                "root_seed": seed,
                "input_issue_max_count": 0,
                "model_gate": {"model_inference_allowed": False},
                "donor": {"max_uses_per_cutout": 64},
                "scene_matching": {
                    "feature_file": "features.npz",
                    "model_name": "fixture-model",
                    "pretrained": "fixture-weights",
                    "min_full_image_cosine_similarity": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    train_images = {index: {} for index in range(1, 65)}
    coco = {
        "categories": [{"id": 1, "name": "person"}],
        "images": [{"id": index, "height": 64} for index in train_images],
    }
    unit = SimpleNamespace(person={"src_group_id": 1, "src_image_id": 1})

    def fake_background(*, background_id: int, **_kwargs):
        rendered = np.zeros((4, 4, 3), dtype=np.uint8)
        record = {
            "background_image_id": background_id,
            "anchor_annotation_id": background_id,
            "donor_cutout_id": "cutout-01",
            "donor_group_id": 1,
            "person_bbox_xywh": [0, 0, 3, 3],
            "head_bbox_xywh": [1, 0, 1, 1],
            "scale": 1.0,
            "hflip": False,
            "scene_clip_cosine_similarity": 1.0,
        }
        return rendered, record

    def fake_write_image(path: Path, _image, _quality: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"paired-image")

    def fake_render_sheet(*, output_path: Path, **_kwargs) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"paired-sheet")

    monkeypatch.setattr(paired, "PROJECT_ROOT", repository)
    monkeypatch.setattr(paired, "load_project_paths", lambda: paths)
    monkeypatch.setattr(paired, "_load_configs", lambda: ({}, {}))
    monkeypatch.setattr(
        paired,
        "_load_context",
        lambda _paths: (coco, {}, train_images, {}, {}, set()),
    )
    monkeypatch.setattr(
        paired,
        "eligible_units",
        lambda **_kwargs: ([unit], Counter()),
    )
    monkeypatch.setattr(
        paired,
        "load_train_only_clip_embeddings",
        lambda *_args, **_kwargs: (
            np.zeros((64, 1), dtype=np.float32),
            {index: index - 1 for index in train_images},
        ),
    )
    monkeypatch.setattr(paired, "_try_background", fake_background)
    monkeypatch.setattr(paired, "_archive_existing", lambda _path: None)
    monkeypatch.setattr(paired, "_write_image", fake_write_image)
    monkeypatch.setattr(paired, "render_sheet", fake_render_sheet)

    paired.main()

    stem = f"h4_paired_person_input_preflight_seed{seed}"
    json_text = (paths.reports / f"{stem}.json").read_text(encoding="utf-8")
    markdown = (paths.reports / f"{stem}.md").read_text(encoding="utf-8")
    payload = json.loads(json_text)
    raw_output = str(paths.synthetic / f"h4_paired_person_preflight_seed{seed}")
    assert raw_output not in json_text
    assert raw_output not in markdown
    assert payload["output_dir"] == LOCAL_ONLY_NOT_PUBLISHED
    assert f"Output directory: `{payload['output_dir']}`" in markdown
    assert payload["contact_sheet"] == (
        f"reports/figures/{stem}.png"
    )
