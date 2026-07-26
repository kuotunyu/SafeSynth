# H4 next-method decision brief

## Current evidence

The registered feathered-alpha compositor fails the paste-artifact gate:
AUC 0.7964 against a maximum of 0.60. This is not explained away by the
obvious confounds:

| check or method | outcome |
|---|---:|
| Final pooled-control H4 | AUC 0.7964 |
| HOG-only diagnostic | AUC 0.7792 |
| HSV-only diagnostic | AUC 0.6816 |
| Exact original source as each real control | AUC 0.9049 |
| Same-class in-place context replacement | AUC 0.8312 |
| Poisson `NORMAL_CLONE` | AUC 0.8869 |
| RGB multiband alpha blend | AUC 0.8507 |

The exact-source result is especially important: the classifier still
separates the pasted object from the same object in its raw source image.
The shortcut is therefore substantive. It is not just the cutout bank selecting
cleaner objects than the pooled real-control population.

## What is no longer scientifically safe

- trying more feather widths, alpha thresholds, or held-out fold seeds;
- choosing HOG-only or HSV-only because it gives a lower number;
- widening the AUC maximum;
- declaring Poisson successful when it reduces changed pixels by washing out
  helmet colour;
- starting M13 and hoping detector training hides the artifact.

Those actions would optimize against the gate rather than fix synthesis.

## Decisions that require the repository owner

### Option A — expand Phase 1 to generative object insertion

Replace pixel cut-and-paste with a model that synthesizes or inpaints the
head/helmet into the target context. Before implementation this requires:

1. approval to change the registered synthesis architecture;
2. current model/license verification;
3. approval before any model download over 2 GB;
4. a new untouched H4 fold and a pre-registered visual identity-preservation
   check, so a model cannot pass merely by erasing or recolouring the object.

This is the most plausible route to continuing the original project claim, but
it materially expands scope and compute.

### Option B — freeze Phase 1 as a negative engineering result

Keep the current reproducible pipeline and publish the finding that guarded
copy-paste did not meet the anti-shortcut criterion. M13 and Phase 2 would not
run, because doing so would contradict the registered gate.

This produces a defensible artifact and unusually honest failure analysis, but
not the planned four-arm detector experiment.

### Option C — redesign the research question

Retain the data/split/SAM2/filter infrastructure but define a new intervention
whose claim does not depend on visually indistinguishable pasted objects. This
needs a new experiment protocol and should not reuse H4's held-out results for
method selection.

## Recommendation

First complete the independent H6 human review. Then choose **Option A** if the
goal is still the full detector study, or **Option B** if preserving the current
scope and methodological integrity matters more than completing every planned
arm. Do not change H4's 0.60 maximum.
