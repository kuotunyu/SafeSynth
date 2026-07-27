from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.generative_model import (
    _matches,
    _materialize_model_file,
    remote_preflight,
)
from src.synthetic.generative_inpaint import load_generative_config


def test_registered_allow_patterns_exclude_duplicate_single_file() -> None:
    config = load_generative_config()
    patterns = config["model"]["allow_patterns"]

    assert _matches("transformer/diffusion_pytorch_model.safetensors", patterns)
    assert _matches("text_encoder/model-00001-of-00002.safetensors", patterns)
    assert not _matches("flux-2-klein-base-4b.safetensors", patterns)
    assert not _matches("editing.jpg", patterns)


class FakeApi:
    def model_info(self, repo_id: str, *, files_metadata: bool) -> SimpleNamespace:
        assert repo_id == "black-forest-labs/FLUX.2-klein-base-4B"
        assert files_metadata is True
        return SimpleNamespace(
            sha="a3b4f4849157f664bdbc776fd7453c2783562f4d",
            card_data=SimpleNamespace(license="apache-2.0"),
            siblings=[
                SimpleNamespace(rfilename="model_index.json", size=422),
                SimpleNamespace(
                    rfilename="transformer/diffusion_pytorch_model.safetensors",
                    size=15_980_131_289,
                ),
                SimpleNamespace(
                    rfilename="flux-2-klein-base-4b.safetensors",
                    size=7_751_105_712,
                ),
            ],
        )


def test_remote_preflight_counts_only_runtime_files(monkeypatch) -> None:
    monkeypatch.setattr("scripts.generative_model.HfApi", FakeApi)

    report = remote_preflight(load_generative_config())

    assert report["download_bytes"] == 15_980_131_711
    assert report["passed"] is True


def test_materialize_model_file_stages_then_moves(
    monkeypatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "long-model-destination"
    staging = tmp_path / "short-stage"
    staged = staging / "vae" / "weights.safetensors"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"verified")
    calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(staged)

    monkeypatch.setattr("scripts.generative_model.hf_hub_download", fake_download)
    _materialize_model_file(
        target=target.resolve(),
        staging=staging.resolve(),
        model={"repo_id": "owner/model", "revision": "fixed-revision"},
        record={"path": "vae/weights.safetensors", "bytes": 8},
    )

    assert (target / "vae" / "weights.safetensors").read_bytes() == b"verified"
    assert not staged.exists()
    assert calls == [
        {
            "repo_id": "owner/model",
            "filename": "vae/weights.safetensors",
            "revision": "fixed-revision",
            "local_dir": staging.resolve(),
        }
    ]
