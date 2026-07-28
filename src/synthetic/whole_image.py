"""Frozen whole-image FLUX diagnostic registration and safety gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import diffusers
import torch

from src.data.paths import ProjectPaths


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def diagnostic_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact prompt and seed manifest shown to the reviewer."""

    diagnostic = config["diagnostic"]
    root_seed = int(diagnostic["root_seed"])
    cases = []
    seen_indices: set[int] = set()
    seen_scenarios: set[str] = set()
    for raw_case in diagnostic["cases"]:
        case_index = int(raw_case["case_index"])
        scenario = str(raw_case["scenario"])
        prompt = " ".join(str(raw_case["prompt"]).split())
        if case_index in seen_indices or scenario in seen_scenarios:
            raise RuntimeError("Whole-image diagnostic cases must be unique")
        if not prompt:
            raise RuntimeError("Whole-image diagnostic prompt cannot be empty")
        seen_indices.add(case_index)
        seen_scenarios.add(scenario)
        seed_material = f"v10|{root_seed}|{case_index}|{scenario}".encode()
        seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            "big",
        ) & (2**63 - 1)
        cases.append(
            {
                "case_index": case_index,
                "scenario": scenario,
                "prompt": prompt,
                "seed": seed,
            }
        )
    cases.sort(key=lambda value: int(value["case_index"]))
    if [int(case["case_index"]) for case in cases] != [1, 2, 3, 4]:
        raise RuntimeError("Whole-image diagnostic must contain cases 01-04")
    manifest = {
        "schema_version": 1,
        "architecture": str(config["architecture"]),
        "root_seed": root_seed,
        "generator": {
            key: config["generator"][key]
            for key in (
                "repo_id",
                "revision",
                "pipeline_class",
                "width",
                "height",
                "num_inference_steps",
                "guidance_scale",
            )
        },
        "cases": cases,
        "validation_images_read": 0,
        "test_images_read": 0,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_json(manifest)
    ).hexdigest()
    return manifest


def human_review_evidence_sha256(report: Mapping[str, Any]) -> str:
    """Hash a review record without trusting its embedded digest."""

    payload = dict(report)
    payload.pop("evidence_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping in canonical key order."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def generator_directory(paths: ProjectPaths, config: Mapping[str, Any]) -> Path:
    """Return the project-isolated pinned FLUX model directory."""

    generator = config["generator"]
    slug = str(generator["repo_id"]).replace("/", "--")
    return paths.cache / "models" / slug / str(generator["revision"])


def require_verified_generator(
    model_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Hard fail unless the existing FLUX manifest matches the registration."""

    manifest_path = model_dir / "SAFESYNTH_MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("Pinned FLUX generator is not downloaded")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config["generator"]
    if (
        manifest.get("repo_id") != expected["repo_id"]
        or manifest.get("revision") != expected["revision"]
        or manifest.get("license") != expected["license"]
        or int(manifest.get("download_bytes", -1))
        != int(expected["required_download_bytes"])
    ):
        raise RuntimeError("FLUX generator manifest does not match registration")
    return manifest


def require_generation_approval(
    *,
    config: Mapping[str, Any],
    labeler_report: Mapping[str, Any],
    human_review_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Keep FLUX locked until both machine and human gates pass."""

    review = config["diagnostic"]["input_review"]
    supervised = config["supervised_labeler"]
    labeler_review = supervised["human_review"]
    gate = config["generation_gate"]
    manifest_sha256 = str(manifest["manifest_sha256"])
    checks = labeler_report.get("checks", {})
    metrics = labeler_report.get("audit_metrics", {})
    best = labeler_report.get("best_calibration", {})
    postprocessing = labeler_report.get("postprocessing", {})
    if (
        labeler_report.get("status") != "supervised_labeler_audit_passed"
        or checks.get("audit_precision") is not True
        or checks.get("audit_recall") is not True
        or checks.get("audit_median_matched_iou") is not True
        or int(labeler_report.get("validation_images_read", -1)) != 0
        or int(labeler_report.get("test_images_read", -1)) != 0
        or int(labeler_report.get("untouched_audit_images_read", -1))
        != int(supervised["audit_images"])
        or labeler_report.get("whole_image_generation_run") is not False
        or labeler_report.get("checkpoint_sha256")
        != supervised["checkpoint_sha256"]
        or labeler_report.get("split_manifest_sha256")
        != supervised["split_manifest_sha256"]
        or float(best.get("threshold", -1))
        != float(supervised["score_threshold"])
        or float(metrics.get("precision", -1))
        != float(supervised["audit_precision"])
        or float(metrics.get("recall", -1))
        != float(supervised["audit_recall"])
        or float(metrics.get("median_matched_iou", -1))
        != float(supervised["audit_median_matched_iou"])
        or float(postprocessing.get("max_relative_area", -1))
        != float(supervised["max_relative_area"])
        or float(postprocessing.get("max_relative_height", -1))
        != float(supervised["max_relative_height"])
        or gate.get("allowed") is not True
        or gate.get("required_reviewer") != "kuotunyu"
        or labeler_review.get("required_reviewer") != "kuotunyu"
        or labeler_review.get("status") != "approved_by_kuotunyu"
        or human_review_report.get("status") != "approved_by_kuotunyu"
        or human_review_report.get("reviewed_by") != "kuotunyu"
        or not human_review_report.get("reviewed_on")
        or human_review_report.get("experiment_id")
        != supervised["experiment_id"]
        or human_review_report.get("checkpoint_sha256")
        != supervised["checkpoint_sha256"]
        or human_review_report.get("split_manifest_sha256")
        != supervised["split_manifest_sha256"]
        or float(human_review_report.get("score_threshold", -1))
        != float(supervised["score_threshold"])
        or int(human_review_report.get("audit_images", -1))
        != int(supervised["audit_images"])
        or human_review_report.get("figure")
        != labeler_review["figure"]
        or human_review_report.get("figure_sha256")
        != labeler_review["figure_sha256"]
        or human_review_report.get("pages") != labeler_review["pages"]
        or human_review_report.get("separated_pages")
        != labeler_review["separated_pages"]
        or int(human_review_report.get("problem_count", -1)) != 0
        or human_review_report.get("problem_cells") != []
        or int(human_review_report.get("validation_images_read", -1)) != 0
        or int(human_review_report.get("test_images_read", -1)) != 0
        or human_review_report.get("evidence_sha256")
        != human_review_evidence_sha256(human_review_report)
        or review.get("required_reviewer") != "kuotunyu"
        or review.get("status") != "approved_by_kuotunyu"
        or review.get("approved_manifest_sha256") != manifest_sha256
    ):
        raise RuntimeError(
            "v10 GPU gate locked: verified v6 audit or exact kuotunyu "
            "approval is missing"
        )


def require_scaleup_approval(
    *,
    config: Mapping[str, Any],
    diagnostic_report: Mapping[str, Any],
    output_review_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Keep 64/300-image expansion locked until the exact v10 pilot passes."""

    supervised = config["supervised_labeler"]
    review = config["diagnostic"]["output_review"]
    gate = config["scaleup_gate"]
    cases = diagnostic_report.get("cases", [])
    if (
        diagnostic_report.get("status")
        != "pending_kuotunyu_visual_review"
        or diagnostic_report.get("manifest_sha256")
        != manifest["manifest_sha256"]
        or diagnostic_report.get("labeler_checkpoint_sha256")
        != supervised["checkpoint_sha256"]
        or diagnostic_report.get("labeler_split_manifest_sha256")
        != supervised["split_manifest_sha256"]
        or len(cases) != 4
        or [int(case.get("case_index", -1)) for case in cases]
        != [1, 2, 3, 4]
        or any(not case.get("image_sha256") for case in cases)
        or int(diagnostic_report.get("validation_images_read", -1)) != 0
        or int(diagnostic_report.get("test_images_read", -1)) != 0
        or diagnostic_report.get("expanded_to_64") is not False
        or gate.get("allowed") is not True
        or gate.get("required_reviewer") != "kuotunyu"
        or review.get("required_reviewer") != "kuotunyu"
        or review.get("status") != "approved_by_kuotunyu"
        or int(review.get("required_problem_count", -1)) != 0
        or output_review_report.get("status") != "approved_by_kuotunyu"
        or output_review_report.get("reviewed_by") != "kuotunyu"
        or not output_review_report.get("reviewed_on")
        or output_review_report.get("manifest_sha256")
        != manifest["manifest_sha256"]
        or output_review_report.get("diagnostic_report_sha256")
        != canonical_mapping_sha256(diagnostic_report)
        or output_review_report.get("figure")
        != diagnostic_report.get("figure")
        or output_review_report.get("figure_sha256")
        != diagnostic_report.get("figure_sha256")
        or output_review_report.get("reviewed_case_indices")
        != [1, 2, 3, 4]
        or int(output_review_report.get("problem_count", -1)) != 0
        or output_review_report.get("problem_cases") != []
        or int(output_review_report.get("validation_images_read", -1)) != 0
        or int(output_review_report.get("test_images_read", -1)) != 0
        or output_review_report.get("evidence_sha256")
        != human_review_evidence_sha256(output_review_report)
    ):
        raise RuntimeError(
            "v10 scale-up gate locked: exact four-case output approval is missing"
        )


def load_flux2_text_to_image(
    *,
    model_dir: Path,
    config: Mapping[str, Any],
) -> Any:
    """Load the pinned local-only FLUX.2 Klein text-to-image pipeline."""

    require_verified_generator(model_dir, config)
    generator = config["generator"]
    if diffusers.__version__ != str(generator["diffusers_version"]):
        raise RuntimeError(
            f"Expected diffusers {generator['diffusers_version']}, "
            f"got {diffusers.__version__}"
        )
    if generator["pipeline_class"] != "Flux2KleinPipeline":
        raise RuntimeError("Registered FLUX pipeline class changed")
    if generator["local_files_only"] is not True:
        raise RuntimeError("FLUX runtime must remain local-only")
    from diffusers import Flux2KleinPipeline

    pipeline = Flux2KleinPipeline.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    if generator["model_cpu_offload"] is True:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")
    return pipeline
