# SafeSynth privacy and responsible use

## What the public release contains

The public dataset contains filtered and size-matched unfiltered synthetic composites,
COCO annotations, `records.jsonl` provenance, and release hashes. The public model is a
research checkpoint selected from the frozen four-arm experiment. Neither artifact is a
certified occupational-safety product.

Some composites preserve background or other pixels from real workers in the upstream
public Hard Hat Workers dataset. The release changes or inserts selected objects; it does
not replace every person or anonymize every source pixel.

## Identifiability boundary

Real workers whose pixels remain in a composite may still be identifiable. In this
release, “synthetic” describes how samples were composed; it does not mean anonymous,
de-identified, fictional, or safe for identity analysis.

The immediate upstream dataset and this derived dataset use CC0 1.0. Dataset licensing
and privacy are separate boundaries: CC0 does not eliminate privacy risk, establish
consent for a new use, or remove applicable responsible-use obligations.

## Allowed research use

Appropriate uses include reproducible synthetic-data ablations, object-detection and
calibration research, provenance auditing, bias and failure analysis, privacy-risk
assessment, and teaching with the limitations kept visible. Researchers should minimize
retention, avoid unnecessary copies, and report only aggregate findings when individual
workers are not relevant to the research question.

## Prohibited or high-risk use

Do not use this research release for identity recognition or re-identification, worker
surveillance, employment decisions, discipline, access control, or other decisions that
can harm a worker. Do not use the dataset, model, demo, or reported metrics for safety
certification or as the sole basis for a workplace-safety verdict. A qualified human
safety process must remain authoritative.

## Removal and incident contact

To request removal of an image or report a privacy incident, contact the repository owner
through the [SafeSynth GitHub repository](https://github.com/kuotunyu/SafeSynth/issues).
Do not post a worker's identity or other sensitive personal details in a public issue;
provide only the artifact path or SHA-256 needed to locate the file, and arrange a private
follow-up with the owner when sensitive context is required.
