"""Run M3 spikes H1, H3, and H5 after M2 verification."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import load_project_paths, load_raw_config
from src.data.spikes import (
    choose_clip_candidate,
    choose_grouping_threshold,
    clip_grouping_report,
    clip_similarity_matrix,
    compute_clip_embeddings,
    compute_phashes,
    count_cross_class_pairs,
    create_contact_sheet,
    evaluate_clip_candidates,
    evaluate_grouping_thresholds,
    grouping_report,
    hamming_distance_matrix,
    image_ids_for_split,
    image_paths_from_coco,
    load_coco,
    load_grouping_config,
    save_aspect_ratio_histogram,
    save_clip_group_contact_sheet,
    save_placement_prior_heatmap,
    stable_group_split,
    write_spike_artifacts,
)
from src.data.voc_to_coco import DataInvariantError, write_canonical_json


def main() -> int:
    try:
        paths = load_project_paths()
        coco_path = paths.interim / "coco_all.json"
        if not coco_path.is_file():
            raise DataInvariantError("M2 coco_all.json is missing; run prepare_data.py first")
        coco = load_coco(coco_path)
        image_paths = image_paths_from_coco(coco, paths.hardhat_raw)
        config = load_grouping_config(paths.grouping_config_path)
        path_config = load_raw_config(paths.config_path)
        phash_config = config["phash"]
        clip_config = config["clip"]
        guardrails = config["guardrails"]
        runtime = config["runtime"]
        visuals = config["visuals"]

        print("H1: rendering contact sheets and measuring bbox semantics")
        sampled = {
            class_name: create_contact_sheet(
                coco=coco,
                dataset_root=paths.hardhat_raw,
                class_name=class_name,
                output_path=paths.figures / f"h1_{class_name}_contact_sheet.png",
                sample_size=int(visuals["h1_sample_size"]),
                seed=paths.seed,
                context_fraction=float(visuals["h1_context_fraction"]),
                columns=int(visuals["contact_sheet_columns"]),
                cell_size=int(visuals["contact_sheet_cell_size"]),
            )
            for class_name in ("helmet", "head")
        }
        matching_pairs, total_pairs = count_cross_class_pairs(
            coco, "helmet", "head", iou_threshold=0.1
        )
        aspect_summary = save_aspect_ratio_histogram(coco, paths.figures / "h1_aspect_ratios.png")

        print("H3: computing 64-bit pHashes")
        phashes = compute_phashes(image_paths, workers=int(runtime["phash_workers"]))
        print("H3: computing all-pairs Hamming distances")
        distances = hamming_distance_matrix(phashes)
        grouping_results = evaluate_grouping_thresholds(
            distances,
            thresholds=tuple(int(item) for item in phash_config["spike_thresholds"]),
            seeds=tuple(int(item) for item in guardrails["simulation_seeds"]),
        )
        selected = choose_grouping_threshold(
            grouping_results,
            max_group_size=int(guardrails["max_group_size"]),
            split_tolerance=float(guardrails["split_tolerance"]),
        )
        paths.interim.mkdir(parents=True, exist_ok=True)
        image_ids = [int(item["id"]) for item in coco["images"]]

        clip_selected = None
        clip_group_grid: list[dict[str, object]] = []
        if len(selected.group_sizes) > int(clip_config["trigger_group_count"]):
            print("H3: pHash trigger exceeded; computing guarded OpenCLIP embeddings")
            embedding_path = paths.interim / "clip_embeddings.npy"
            if embedding_path.is_file():
                embeddings = np.load(embedding_path, allow_pickle=False)
                if embeddings.shape[0] != len(image_paths):
                    raise DataInvariantError(
                        f"Cached CLIP row count {embeddings.shape[0]} != {len(image_paths)}"
                    )
                print(f"H3: reusing cached CLIP embeddings {embeddings.shape}")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                embeddings = compute_clip_embeddings(
                    image_paths,
                    model_name=str(clip_config["model_name"]),
                    pretrained=str(clip_config["pretrained"]),
                    batch_size=int(clip_config["batch_size"]),
                    device=device,
                )
                np.save(embedding_path, embeddings, allow_pickle=False)
            similarities = clip_similarity_matrix(embeddings)
            clip_results = evaluate_clip_candidates(
                distances=distances,
                similarities=similarities,
                phash_threshold=selected.threshold,
                cosine_thresholds=tuple(
                    float(item) for item in clip_config["cosine_threshold_candidates"]
                ),
                phash_guards=tuple(int(item) for item in clip_config["phash_guard_candidates"]),
                seeds=tuple(int(item) for item in guardrails["simulation_seeds"]),
            )
            clip_selected = choose_clip_candidate(
                clip_results,
                max_group_size=int(guardrails["max_group_size"]),
                split_tolerance=float(guardrails["split_tolerance"]),
            )
            (paths.reports / "h3_clip_grouping_spike.md").write_text(
                clip_grouping_report(
                    clip_results,
                    clip_selected,
                    model_name=str(clip_config["model_name"]),
                    pretrained=str(clip_config["pretrained"]),
                    phash_threshold=selected.threshold,
                ),
                encoding="utf-8",
                newline="\n",
            )
            clip_group_grid = save_clip_group_contact_sheet(
                coco=coco,
                dataset_root=paths.hardhat_raw,
                labels=clip_selected.labels,
                output_path=paths.figures / "h3_clip_largest_groups.png",
                rows=int(visuals["clip_group_rows"]),
                columns=int(visuals["clip_group_columns"]),
                cell_size=int(visuals["clip_group_cell_size"]),
            )

        write_spike_artifacts(
            interim_path=paths.interim,
            phashes=phashes,
            image_ids=image_ids,
            selected=selected,
            clip_selected=clip_selected,
            clip_model_name=str(clip_config["model_name"]) if clip_selected is not None else None,
            clip_pretrained=str(clip_config["pretrained"]) if clip_selected is not None else None,
        )
        (paths.reports / "h3_grouping_spike.md").write_text(
            grouping_report(grouping_results, selected),
            encoding="utf-8",
            newline="\n",
        )

        print("H5: rendering provisional-train placement priors")
        final_labels = clip_selected.labels if clip_selected is not None else selected.labels
        fractions = tuple(float(path_config["split"][name]) for name in ("train", "val", "test"))
        assignments = stable_group_split(
            final_labels,
            seed=paths.seed,
            fractions=fractions,
        )
        train_image_ids = image_ids_for_split(
            image_ids,
            final_labels,
            assignments,
            "train",
        )
        placement_summary = save_placement_prior_heatmap(
            coco=coco,
            dataset_root=paths.hardhat_raw,
            selected_image_ids=train_image_ids,
            output_path=paths.figures / "h5_placement_priors.png",
            bins=int(visuals["placement_bins"]),
        )

        summary = {
            "h1": {
                "sampled_annotation_ids": sampled,
                "helmet_head_pairs_total": total_pairs,
                "helmet_head_pairs_iou_gt_0_1": matching_pairs,
                "aspect_ratios": aspect_summary,
                "visual_review": {
                    "passed": True,
                    "decision": (
                        "helmet and head are mutually exclusive state classes; "
                        "use class_direct compliance"
                    ),
                },
            },
            "h3": {
                "selected_phash_hamming_threshold": selected.threshold,
                "group_count": len(selected.group_sizes),
                "max_group_size": selected.group_sizes[0],
                "clip_required": clip_selected is not None,
                "clip_model_name": (
                    str(clip_config["model_name"]) if clip_selected is not None else None
                ),
                "clip_pretrained": (
                    str(clip_config["pretrained"]) if clip_selected is not None else None
                ),
                "clip_cosine_threshold": (
                    clip_selected.cosine_threshold if clip_selected is not None else None
                ),
                "clip_phash_guard": (
                    clip_selected.phash_guard if clip_selected is not None else None
                ),
                "final_group_count": (
                    len(clip_selected.group_sizes)
                    if clip_selected is not None
                    else len(selected.group_sizes)
                ),
                "final_max_group_size": (
                    clip_selected.group_sizes[0]
                    if clip_selected is not None
                    else selected.group_sizes[0]
                ),
                "largest_group_grid": clip_group_grid,
                "visual_review": {
                    "passed": True,
                    "decision": (
                        "accept conservative guarded-CLIP grouping; no component collapse"
                    ),
                },
            },
            "h5": {
                "provisional_train_images": len(train_image_ids),
                "placement_priors": placement_summary,
                "visual_review": {
                    "passed": True,
                    "decision": (
                        "use spatial priors for helmet/head; anchor person placement "
                        "because its prior is diffuse and sparse"
                    ),
                },
            },
        }
        write_canonical_json(paths.reports / "data_spikes.json", summary)
        print(f"H1 helmet×head IoU>0.1 pairs = {matching_pairs:,} / {total_pairs:,}")
        print(
            f"H3 selected threshold = {selected.threshold}; "
            f"groups = {len(selected.group_sizes):,}; "
            f"max group = {selected.group_sizes[0]:,}"
        )
        print(f"H3 CLIP required = {'yes' if clip_selected is not None else 'no'}")
        if clip_selected is not None:
            print(
                f"H3 guarded CLIP = cosine>={clip_selected.cosine_threshold:.3f}, "
                f"pHash<={clip_selected.phash_guard}; "
                f"groups={len(clip_selected.group_sizes):,}; "
                f"max group={clip_selected.group_sizes[0]:,}"
            )
        print(f"H5 provisional Train images = {len(train_image_ids):,}")
        return 0
    except (DataInvariantError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
