from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

from src.synthetic.grounded_labeler import (
    box_iou_xyxy,
    greedy_detection_metrics,
    load_whole_image_config,
    predict_single_phrase,
    require_verified_labeler,
)


def test_whole_image_generation_is_locked_before_label_audit() -> None:
    config = load_whole_image_config()

    assert config["generation_gate"]["allowed"] is False
    assert config["labeler"]["license"] == "apache-2.0"
    assert config["labeler"]["required_download_bytes"] == 690_305_545


def test_open_generation_gate_requires_owner_review(tmp_path) -> None:
    config = load_whole_image_config()
    config["generation_gate"]["allowed"] = True
    path = tmp_path / "whole_image.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="without owner approval"):
        load_whole_image_config(path)


def test_verified_labeler_rehashes_registered_files(tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.safetensors"
    model_path.write_bytes(b"fixed model")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    config = {
        "labeler": {
            "repo_id": "owner/model",
            "revision": "fixed-revision",
            "license": "apache-2.0",
            "required_download_bytes": len(b"fixed model"),
            "allow_files": {"model.safetensors": len(b"fixed model")},
        }
    }
    manifest = {
        "repo_id": "owner/model",
        "revision": "fixed-revision",
        "license": "apache-2.0",
        "download_bytes": len(b"fixed model"),
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(b"fixed model"),
                "sha256": digest,
            }
        ],
    }
    (model_dir / "SAFESYNTH_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    require_verified_labeler(model_dir, config)
    model_path.write_bytes(b"wrong model")

    with pytest.raises(RuntimeError, match="failed integrity"):
        require_verified_labeler(model_dir, config)


def test_box_iou_and_greedy_matching() -> None:
    assert box_iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    metrics = greedy_detection_metrics(
        [(0, 0, 10, 10), (20, 20, 30, 30)],
        [
            (0.9, (0, 0, 10, 10)),
            (0.8, (1, 1, 9, 9)),
            (0.7, (20, 20, 30, 30)),
        ],
        score_threshold=0.5,
        match_iou=0.5,
    )

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0


class _FakeInputs(dict):
    input_ids = torch.tensor([[1, 2]])

    def to(self, device: str) -> _FakeInputs:
        self["device"] = device
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.text: list[list[str]] | None = None
        self.target_sizes: list[tuple[int, int]] | None = None

    def __call__(self, **kwargs: object) -> _FakeInputs:
        self.text = kwargs["text"]  # type: ignore[assignment]
        return _FakeInputs(pixel_values=torch.ones(1))

    def post_process_grounded_object_detection(
        self,
        _outputs: object,
        _input_ids: object,
        **kwargs: object,
    ) -> list[dict[str, torch.Tensor]]:
        self.target_sizes = kwargs["target_sizes"]  # type: ignore[assignment]
        return [
            {
                "scores": torch.tensor([0.75]),
                "boxes": torch.tensor([[1.0, 2.0, 30.0, 40.0]]),
            }
        ]


class _FakeModel:
    def __call__(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace()


def test_single_phrase_prediction_does_not_mix_candidate_prompts() -> None:
    processor = _FakeProcessor()
    image = Image.new("RGB", (64, 48), "white")

    detected = predict_single_phrase(
        processor=processor,
        model=_FakeModel(),
        images=[image],
        phrase="  a safety hard hat  ",
        device="cuda",
        score_floor=0.05,
        text_threshold=0.1,
    )

    assert processor.text == [["a safety hard hat"]]
    assert processor.target_sizes == [(48, 64)]
    assert detected == [[(0.75, [1.0, 2.0, 30.0, 40.0])]]


def test_single_phrase_prediction_rejects_blank_phrase() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        predict_single_phrase(
            processor=_FakeProcessor(),
            model=_FakeModel(),
            images=[Image.new("RGB", (8, 8))],
            phrase=" ",
            device="cpu",
            score_floor=0.05,
            text_threshold=0.1,
        )
