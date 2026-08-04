from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.prepare_hf_release import main as prepare_hf_release_main
from scripts.verify_hf_release import main as verify_hf_release_main
from src.release.hf_bundle import (
    ReleaseBundleError,
    prepare_dataset_bundle,
    prepare_model_bundle,
    verify_dataset_bundle,
    verify_model_bundle,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")


def _coco(*file_names: str) -> dict:
    return {
        "images": [
            {"id": index, "file_name": f"images/{name}", "width": 4, "height": 4}
            for index, name in enumerate(file_names, start=1)
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "helmet"}],
    }


def _record(name: str, payload: bytes | None = None) -> dict:
    image_bytes = name.encode("ascii") if payload is None else payload
    return {
        "sample_id": name.removesuffix(".png"),
        "file_name": f"images/{name}",
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
    }


def test_dataset_bundle_keeps_the_annotation_union_once_with_exact_provenance(
    tmp_path: Path,
) -> None:
    """Catch copying the whole generation pool or duplicating shared release images."""

    source = tmp_path / "source"
    images = source / "images"
    images.mkdir(parents=True)
    for name in ("a.png", "b.png", "c.png", "unused.png"):
        (images / name).write_bytes(name.encode("ascii"))

    _write_json(source / "annotations_filtered_1x.json", _coco("a.png", "b.png"))
    _write_json(source / "annotations_unfiltered_1x.json", _coco("b.png", "c.png"))
    records = [_record(name) for name in ("a.png", "b.png", "c.png", "unused.png")]
    (source / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    card = tmp_path / "DATASET_CARD.md"
    card.write_text("# Mini dataset\n", encoding="utf-8", newline="\n")

    output = tmp_path / "bundle"
    manifest = prepare_dataset_bundle(source, output, card, link_mode="copy")

    assert sorted(path.name for path in (output / "images").iterdir()) == [
        "a.png",
        "b.png",
        "c.png",
    ]
    released_records = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["file_name"] for record in released_records] == [
        "images/a.png",
        "images/b.png",
        "images/c.png",
    ]
    assert (output / "annotations_filtered.json").is_file()
    assert (output / "annotations_unfiltered.json").is_file()
    assert (output / "README.md").read_text(encoding="utf-8") == "# Mini dataset\n"
    assert manifest["dataset"]["filtered_images"] == 2
    assert manifest["dataset"]["unfiltered_images"] == 2
    assert manifest["dataset"]["shared_images"] == 1
    assert manifest["dataset"]["unique_images"] == 3
    assert json.loads((output / "release_manifest.json").read_text(encoding="utf-8")) == manifest


def test_dataset_bundle_refuses_an_annotation_without_matching_provenance(
    tmp_path: Path,
) -> None:
    """Catch silently publishing an image whose generation record was lost."""

    source = tmp_path / "source"
    images = source / "images"
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(b"a")
    _write_json(source / "annotations_filtered_1x.json", _coco("a.png"))
    _write_json(source / "annotations_unfiltered_1x.json", _coco("a.png"))
    (source / "records.jsonl").write_text("", encoding="utf-8", newline="\n")
    card = tmp_path / "DATASET_CARD.md"
    card.write_text("# Card\n", encoding="utf-8", newline="\n")

    with pytest.raises(ReleaseBundleError, match="missing provenance for images/a.png"):
        prepare_dataset_bundle(source, tmp_path / "bundle", card, link_mode="copy")


def test_dataset_bundle_refuses_image_bytes_that_disagree_with_provenance(
    tmp_path: Path,
) -> None:
    """Catch publishing a replaced or corrupted image under a valid sample id."""

    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "a.png").write_bytes(b"actual")
    _write_json(source / "annotations_filtered_1x.json", _coco("a.png"))
    _write_json(source / "annotations_unfiltered_1x.json", _coco("a.png"))
    bad = _record("a.png", payload=b"different")
    (source / "records.jsonl").write_text(
        json.dumps(bad) + "\n", encoding="utf-8", newline="\n"
    )
    card = tmp_path / "DATASET_CARD.md"
    card.write_text("# Card\n")
    output = tmp_path / "bundle"

    with pytest.raises(ReleaseBundleError, match="image hash disagrees.*images/a.png"):
        prepare_dataset_bundle(source, output, card, link_mode="copy")
    assert not output.exists()


def test_model_bundle_includes_only_inference_files_and_the_card(tmp_path: Path) -> None:
    """Catch uploading Trainer state or omitting the processor needed for inference."""

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"rt_detr_v2"}\n')
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "optimizer.pt").write_bytes(b"do not publish")
    processor = tmp_path / "preprocessor_config.json"
    processor.write_text('{"image_processor_type":"RTDetrImageProcessor"}\n')
    card = tmp_path / "MODEL_CARD.md"
    card.write_text("# Mini model\n", encoding="utf-8", newline="\n")

    output = tmp_path / "model-bundle"
    manifest = prepare_model_bundle(checkpoint, processor, output, card)

    assert sorted(path.name for path in output.iterdir()) == [
        "README.md",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "release_manifest.json",
    ]
    assert manifest["model"]["published_files"] == [
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ]
    assert manifest["model"]["excluded_trainer_state"] is True


def test_release_cli_builds_both_owner_review_directories(tmp_path: Path) -> None:
    """Catch wiring only one of the two required Hugging Face payloads."""

    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "a.png").write_bytes(b"a")
    _write_json(source / "annotations_filtered_1x.json", _coco("a.png"))
    _write_json(source / "annotations_unfiltered_1x.json", _coco("a.png"))
    (source / "records.jsonl").write_text(
        json.dumps(_record("a.png", payload=b"a")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    processor = tmp_path / "preprocessor_config.json"
    processor.write_text("{}\n")
    dataset_card = tmp_path / "DATASET_CARD.md"
    dataset_card.write_text("# Dataset\n")
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("# Model\n")
    dataset_output = tmp_path / "dataset-output"
    model_output = tmp_path / "model-output"

    exit_code = prepare_hf_release_main(
        [
            "--dataset-source",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--processor-config",
            str(processor),
            "--dataset-card",
            str(dataset_card),
            "--model-card",
            str(model_card),
            "--dataset-output",
            str(dataset_output),
            "--model-output",
            str(model_output),
            "--link-mode",
            "copy",
        ]
    )

    assert exit_code == 0
    assert (dataset_output / "release_manifest.json").is_file()
    assert (model_output / "release_manifest.json").is_file()


def test_release_cli_does_not_leave_half_a_release_when_model_source_is_missing(
    tmp_path: Path,
) -> None:
    """Catch building the dataset before discovering that the model cannot build."""

    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "a.png").write_bytes(b"a")
    _write_json(source / "annotations_filtered_1x.json", _coco("a.png"))
    _write_json(source / "annotations_unfiltered_1x.json", _coco("a.png"))
    (source / "records.jsonl").write_text(
        json.dumps(_record("a.png", payload=b"a")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    processor = tmp_path / "preprocessor_config.json"
    processor.write_text("{}\n")
    dataset_card = tmp_path / "DATASET_CARD.md"
    dataset_card.write_text("# Dataset\n")
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text("# Model\n")
    dataset_output = tmp_path / "dataset-output"

    exit_code = prepare_hf_release_main(
        [
            "--dataset-source",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--processor-config",
            str(processor),
            "--dataset-card",
            str(dataset_card),
            "--model-card",
            str(model_card),
            "--dataset-output",
            str(dataset_output),
            "--model-output",
            str(tmp_path / "model-output"),
            "--link-mode",
            "copy",
        ]
    )

    assert exit_code == 2
    assert not dataset_output.exists()


def test_dataset_bundle_verifier_detects_bytes_changed_after_packaging(
    tmp_path: Path,
) -> None:
    """Catch a corrupt upload payload even when preparation originally passed."""

    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "a.png").write_bytes(b"a")
    _write_json(source / "annotations_filtered_1x.json", _coco("a.png"))
    _write_json(source / "annotations_unfiltered_1x.json", _coco("a.png"))
    (source / "records.jsonl").write_text(
        json.dumps(_record("a.png", payload=b"a")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    card = tmp_path / "DATASET_CARD.md"
    card.write_text("---\nlicense: cc0-1.0\n---\nSAM2 ground truth filtered unfiltered\n")
    output = tmp_path / "bundle"
    prepare_dataset_bundle(source, output, card, link_mode="copy")
    assert verify_dataset_bundle(output)["dataset"]["unique_images"] == 1

    (output / "images" / "a.png").write_bytes(b"corrupt")
    with pytest.raises(ReleaseBundleError, match="image hash disagrees.*images/a.png"):
        verify_dataset_bundle(output)


def test_dataset_bundle_verifier_rejects_category_ids_documented_incorrectly(
    tmp_path: Path,
) -> None:
    """Catch a card that presents zero-based IDs for one-based COCO categories."""

    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "a.png").write_bytes(b"a")
    _write_json(source / "annotations_filtered_1x.json", _coco("a.png"))
    _write_json(source / "annotations_unfiltered_1x.json", _coco("a.png"))
    (source / "records.jsonl").write_text(
        json.dumps(_record("a.png", payload=b"a")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    card = tmp_path / "DATASET_CARD.md"
    card.write_text(
        "---\nlicense: cc0-1.0\n---\nSAM2 ground truth filtered unfiltered\n"
        "| 0 | `helmet` | wrong id |\n"
    )
    output = tmp_path / "bundle"
    prepare_dataset_bundle(source, output, card, link_mode="copy")

    with pytest.raises(ReleaseBundleError, match="dataset card category table disagrees"):
        verify_dataset_bundle(output)


def test_model_bundle_verifier_rejects_trainer_state_added_after_packaging(
    tmp_path: Path,
) -> None:
    """Catch accidentally placing optimizer state in the owner upload directory."""

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_json(
        checkpoint / "config.json",
        {
            "model_type": "rt_detr_v2",
            "id2label": {"0": "helmet", "1": "head", "2": "person"},
        },
    )
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "optimizer.pt").write_bytes(b"excluded trainer state")
    processor = tmp_path / "preprocessor_config.json"
    _write_json(
        processor,
        {
            "image_processor_type": "RTDetrImageProcessor",
            "do_normalize": False,
            "size": {"height": 640, "width": 640},
        },
    )
    card = tmp_path / "MODEL_CARD.md"
    card.write_text(
        "---\nlicense: apache-2.0\nbase_model: PekingU/rtdetr_v2_r18vd\n---\n"
        "absolute AP synthetic four-arm\n"
    )
    output = tmp_path / "bundle"
    prepare_model_bundle(checkpoint, processor, output, card)
    assert verify_model_bundle(output)["model"]["excluded_trainer_state"] is True

    (output / "optimizer.pt").write_bytes(b"secret trainer state")
    with pytest.raises(ReleaseBundleError, match="unexpected model payload file.*optimizer.pt"):
        verify_model_bundle(output)


def test_model_bundle_verifier_rejects_swapped_label_ids(tmp_path: Path) -> None:
    """Catch a loadable checkpoint whose displayed class IDs are silently swapped."""

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_json(
        checkpoint / "config.json",
        {
            "model_type": "rt_detr_v2",
            "id2label": {"0": "head", "1": "helmet", "2": "person"},
        },
    )
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    processor = tmp_path / "preprocessor_config.json"
    _write_json(
        processor,
        {
            "image_processor_type": "RTDetrImageProcessor",
            "do_normalize": False,
            "size": {"height": 640, "width": 640},
        },
    )
    card = tmp_path / "MODEL_CARD.md"
    card.write_text(
        "---\nlicense: apache-2.0\nbase_model: PekingU/rtdetr_v2_r18vd\n---\n"
        "absolute AP synthetic four-arm\n"
    )
    output = tmp_path / "bundle"
    prepare_model_bundle(checkpoint, processor, output, card)

    with pytest.raises(ReleaseBundleError, match="model config label IDs are not exact"):
        verify_model_bundle(output)


def test_verify_release_cli_accepts_both_complete_payloads(tmp_path: Path) -> None:
    """Catch a release command that verifies only one of the two Hub payloads."""

    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "images" / "a.png").write_bytes(b"a")
    _write_json(source / "annotations_filtered_1x.json", _coco("a.png"))
    _write_json(source / "annotations_unfiltered_1x.json", _coco("a.png"))
    (source / "records.jsonl").write_text(
        json.dumps(_record("a.png", payload=b"a")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    dataset_card = tmp_path / "DATASET_CARD.md"
    dataset_card.write_text(
        "---\nlicense: cc0-1.0\n---\nSAM2 ground truth filtered unfiltered\n"
    )
    dataset_output = tmp_path / "dataset-bundle"
    prepare_dataset_bundle(source, dataset_output, dataset_card, link_mode="copy")

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_json(
        checkpoint / "config.json",
        {
            "model_type": "rt_detr_v2",
            "id2label": {"0": "helmet", "1": "head", "2": "person"},
        },
    )
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    processor = tmp_path / "preprocessor_config.json"
    _write_json(
        processor,
        {
            "image_processor_type": "RTDetrImageProcessor",
            "do_normalize": False,
            "size": {"height": 640, "width": 640},
        },
    )
    model_card = tmp_path / "MODEL_CARD.md"
    model_card.write_text(
        "---\nlicense: apache-2.0\nbase_model: PekingU/rtdetr_v2_r18vd\n---\n"
        "absolute AP synthetic four-arm\n"
    )
    model_output = tmp_path / "model-bundle"
    prepare_model_bundle(checkpoint, processor, model_output, model_card)

    assert (
        verify_hf_release_main(
            [
                "--dataset",
                str(dataset_output),
                "--model",
                str(model_output),
            ]
        )
        == 0
    )
