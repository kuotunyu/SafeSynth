---
license: cc0-1.0
task_categories:
  - object-detection
language:
  - en
tags:
  - computer-vision
  - object-detection
  - synthetic-data
  - construction-safety
  - hard-hat
pretty_name: SafeSynth Hard-Hat Synthetic Data
size_categories:
  - 1K<n<10K
---

# SafeSynth Hard-Hat Synthetic Data

SafeSynth is a controlled synthetic-data ablation for hard-hat detection. This
release contains two **equal-sized** COCO annotation sets drawn from the same
14,000-image candidate pool:

| Release view | Images | Annotations | Meaning |
|---|---:|---:|---|
| `annotations_filtered.json` | 3,500 | 25,278 | Images that passed every pre-registered geometry, photometry, and quality rule |
| `annotations_unfiltered.json` | 3,500 | 29,998 | A deterministic size-matched sample from the full pool before filtering |

The two views overlap by 848 images and refer to 6,152 unique image files in
total. Images are stored once under `images/`; both COCO files point into that
shared directory. `records.jsonl` contains exactly one provenance record for
each released image.

This dataset accompanies the
[SafeSynth source repository](https://github.com/kuotunyu/SafeSynth) and the
[released RT-DETRv2-R18 checkpoint](https://huggingface.co/kuotunyu/safesynth-rtdetrv2-r18).

## What the labels mean

The COCO categories are:

| ID | Class | Interpretation |
|---:|---|---|
| 1 | `helmet` | A safety helmet worn on a visible head |
| 2 | `head` | A visible bare head (non-compliant) |
| 3 | `person` | Person boxes inherited from the upstream dataset |

The released annotations are generator outputs derived from the frozen real
Train split. Validation and Test images were never used by the generator,
filter, or synthetic-data selection logic.

## Files

```text
README.md
annotations_filtered.json
annotations_unfiltered.json
records.jsonl
release_manifest.json
images/
  s42_000001.png
  ...
```

Both annotation files use standard COCO detection fields. To select a view,
load the corresponding JSON and resolve each `images[].file_name` relative to
the repository root.

## Generation method

1. Freeze a group-disjoint real Train/Validation/Test split.
2. Use [SAM 2.1 Hiera Large](https://huggingface.co/facebook/sam2.1-hiera-large)
   (Apache-2.0) to segment source objects from **Train only**.
3. Build cutouts for six targeted scenarios: small/distant objects, partial
   occlusion, crowded scenes, low light or blur, bare heads, and hard negatives.
4. Contextually copy-paste cutouts into Train backgrounds and recompute boxes
   from the post-compositing visible masks.
5. Apply pre-registered geometry, visibility, contact, photometry, clipping,
   deduplication, and quality rules.
6. Export the passing set and a deterministic, equal-sized pre-filter sample.

The filtered view contains 3,500 passing records. The unfiltered view contains
991 passing and 2,509 rejected records; this is intentional and makes the
filtering ablation size-matched rather than confounded by data quantity.

**SAM2's automatic masks were used only to obtain synthesis material and to
recompute boxes during compositing. They were not used as ground truth for any
real Validation or Test image.**

## Provenance (`records.jsonl`)

Each JSON line is keyed by `file_name` and includes:

- `sample_id`, `schema_version`, `scenario`, image dimensions, and
  `image_sha256`;
- the real background file, image ID, and frozen pHash group;
- deterministic generation seeds, retry counts, and hard-negative flags;
- every retained real or pasted instance, its source IDs, class, cutout ID,
  pre-clip/original/final box, visibility, SAM2 quality fields, and z-order;
- filter verdict (`passed`), first/all rejection reasons, invariants,
  deduplication evidence, and any intentional removals;
- post-processing and JPEG settings when applicable.

The release builder verifies every image SHA-256 against this ledger and
refuses to package a record or image that is missing, duplicated, path-unsafe,
or inconsistent.

## Source and license chain

- Immediate source: [Hard Hat Workers / Safety Helmet Detection on Kaggle](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
  (`andrewmvd/hard-hat-detection`), 5,000 PASCAL VOC images, released under
  CC0 1.0 / Public Domain.
- Project provenance notes attribute the Kaggle release to Northeastern
  University–China material redistributed through Harvard Dataverse and
  [Roboflow](https://public.roboflow.com/object-detection/hard-hat-workers).
  This release treats the immediate Kaggle CC0 page as its license authority;
  it does not claim that every redistribution hand-off was independently
  revalidated.
- The [SHEL5K dataset paper](https://doi.org/10.3390/s22062315) separately
  documents the annotation incompleteness of the same 5,000-image SHD set and
  compares it with the related Hardhat and Hard Hat Workers datasets.
- SAM 2.1 model and code: Apache-2.0.
- These derived synthetic images and annotations are released under
  **CC0 1.0** to match the immediate source dataset.

## Intended use

This release is intended for research on synthetic-data filtering, controlled
augmentation ablations, hard-hat detection, provenance-aware generation, and
failure analysis. It is not a certified occupational-safety system and must not
be used as the sole basis for workplace enforcement or safety decisions.

## Known limitations

- The source annotations are incomplete. SHEL5K re-annotated the same 5,000
  images with 75,570 labels versus 25,502 in the source and reports that the
  `person` class is poorly labelled. The synthetic labels inherit those defects.
- Image sizes are not uniformly 416x416: the originals include 415-pixel edges.
  Evaluation must map predictions back per image instead of applying one global
  scale factor.
- A pre-registered pasted-artifact classifier reached AUC **0.9053** against a
  maximum allowed AUC of 0.60. The synthetic domain is detectably different.
- Filtered and unfiltered are equal-sized views, not statistically independent
  datasets; 848 images occur in both.
- These data did not produce a robust improvement in the repository's
  four-arm experiments. Do not use this release to claim an absolute AP level
  or an unconditional synthetic-data benefit.
- Real workers may be identifiable in inherited source pixels. Apply the same
  privacy and responsible-use care as for the upstream public dataset.

## Reproducibility and audit

The complete generator, frozen split protocol, filter rules, experiment
configuration, source hashes, and result tables are available in
[GitHub](https://github.com/kuotunyu/SafeSynth). `release_manifest.json` records
the counts and SHA-256 digests of the packaged metadata files.
