"""Which images each of the four arms trains on (TRAIN-03..07).

This module is the single source of truth for arm composition, and it is a
module rather than notebook cells for one reason: notebooks cannot be tested,
and every rule here is a rule whose violation is silent. An arm that quietly
trains on a different set of real images, or synthetic arms of unequal size,
produces a clean-looking result table that means nothing.

`scripts/build_training_notebook.py` embeds this file into the Colab notebook,
so the code that runs on Colab is the code the tests ran against.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARMS = ("real_only", "standard_aug", "unfiltered_syn", "filtered_syn")
SYNTHETIC_SUBSET = {
    "real_only": None,
    "standard_aug": None,
    "unfiltered_syn": "unfiltered",
    "filtered_syn": "filtered",
}
# The arms differ in augmentation only between real_only and the rest: TRAIN-05
# requires standard_aug's photometric range to bracket what synthetic images
# carry, and the synthetic arms must reuse that same augmentation or "synthetic
# helps" becomes confounded with "different augmentation".
AUGMENTATION_PROFILE = {
    "real_only": "real_only",
    "standard_aug": "standard_aug",
    "unfiltered_syn": "standard_aug",
    "filtered_syn": "standard_aug",
}


class ArmCompositionError(RuntimeError):
    """Raised when an arm would train on the wrong data."""


@dataclass(frozen=True)
class ArmComposition:
    arm: str
    real_train: tuple[str, ...]
    real_val: tuple[str, ...]
    synthetic: tuple[str, ...]
    augmentation_profile: str
    real_train_digest: str = field(default="")

    @property
    def n_train_images(self) -> int:
        return len(self.real_train) + len(self.synthetic)

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "n_real_train": len(self.real_train),
            "n_real_val": len(self.real_val),
            "n_synthetic": len(self.synthetic),
            "n_train_total": self.n_train_images,
            "augmentation_profile": self.augmentation_profile,
            "real_train_digest": self.real_train_digest,
        }


def digest_names(names: Sequence[str]) -> str:
    """Order-independent digest, so TRAIN-04 compares sets not orderings."""

    payload = "\n".join(sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _coco_image_names(path: Path) -> tuple[str, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(sorted(image["file_name"].split("/")[-1] for image in payload["images"]))


def split_real_images(manifest_path: Path) -> dict[str, tuple[str, ...]]:
    """Read the frozen split. Test is returned so callers can assert against it."""

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = manifest["images"] if isinstance(manifest, dict) else manifest
    grouped: dict[str, list[str]] = {}
    for entry in entries:
        name = str(entry["file_name"]).split("/")[-1]
        grouped.setdefault(str(entry["split"]), []).append(name)
    return {split: tuple(sorted(names)) for split, names in grouped.items()}


def build_arm(
    arm: str,
    *,
    real_splits: Mapping[str, Sequence[str]],
    synthetic_annotations: Mapping[str, Path],
) -> ArmComposition:
    if arm not in ARMS:
        raise ArmCompositionError(f"Unknown arm {arm!r}; expected one of {ARMS}")

    real_train = tuple(sorted(real_splits["train"]))
    real_val = tuple(sorted(real_splits["val"]))
    if not real_train or not real_val:
        raise ArmCompositionError("Frozen split yielded an empty train or val set")

    subset = SYNTHETIC_SUBSET[arm]
    synthetic: tuple[str, ...] = ()
    if subset is not None:
        path = synthetic_annotations.get(subset)
        if path is None:
            raise ArmCompositionError(f"Arm {arm!r} needs the {subset!r} annotations")
        synthetic = _coco_image_names(Path(path))
        if not synthetic:
            raise ArmCompositionError(f"{subset!r} annotations contain no images")

    return ArmComposition(
        arm=arm,
        real_train=real_train,
        real_val=real_val,
        synthetic=synthetic,
        augmentation_profile=AUGMENTATION_PROFILE[arm],
        real_train_digest=digest_names(real_train),
    )


def build_all_arms(
    *,
    manifest_path: Path,
    synthetic_annotations: Mapping[str, Path],
) -> dict[str, ArmComposition]:
    real_splits = split_real_images(manifest_path)
    compositions = {
        arm: build_arm(
            arm, real_splits=real_splits, synthetic_annotations=synthetic_annotations
        )
        for arm in ARMS
    }
    assert_arm_invariants(compositions, real_splits=real_splits)
    return compositions


def assert_arm_invariants(
    compositions: Mapping[str, ArmComposition],
    *,
    real_splits: Mapping[str, Sequence[str]],
) -> None:
    """Every rule here fails silently if unchecked, so all of them crash instead."""

    missing = set(ARMS) - set(compositions)
    if missing:
        raise ArmCompositionError(f"Missing arms: {sorted(missing)}")

    # TRAIN-23 — checked FIRST because it is the most severe failure and the one
    # least likely to be noticed. A leak that touches all four arms equally would
    # sail past the digest comparison below, and a leak in one arm would
    # otherwise be reported as a mere "arms disagree", burying the real problem.
    test_names = set(real_splits.get("test", ()))
    if test_names:
        for arm, comp in compositions.items():
            leaked = test_names & (set(comp.real_train) | set(comp.real_val))
            if leaked:
                raise ArmCompositionError(
                    f"Arm {arm!r} would train or validate on {len(leaked)} Test images"
                )

    # TRAIN-04 — all four arms must consume exactly the same real images.
    digests = {arm: comp.real_train_digest for arm, comp in compositions.items()}
    if len(set(digests.values())) != 1:
        raise ArmCompositionError(
            f"Arms disagree on their real training images: {digests}"
        )

    # TRAIN-06 — the two synthetic arms must be the same size, or the ablation
    # confounds "more data" with "better data".
    unfiltered = len(compositions["unfiltered_syn"].synthetic)
    filtered = len(compositions["filtered_syn"].synthetic)
    if unfiltered != filtered:
        raise ArmCompositionError(
            f"Synthetic arms are not size matched: unfiltered={unfiltered} "
            f"filtered={filtered}"
        )

    # The baseline arms must carry no synthetic data at all.
    for arm in ("real_only", "standard_aug"):
        if compositions[arm].synthetic:
            raise ArmCompositionError(f"Arm {arm!r} must not receive synthetic images")

    # Real val must be identical across arms too; a differing val set makes the
    # headline numbers incomparable while looking perfectly normal.
    val_digests = {arm: digest_names(comp.real_val) for arm, comp in compositions.items()}
    if len(set(val_digests.values())) != 1:
        raise ArmCompositionError(f"Arms disagree on their validation images: {val_digests}")


def equal_step_budget(
    compositions: Mapping[str, ArmComposition],
    *,
    reference_arm: str,
    reference_epochs: int,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Give every arm the same optimizer steps (TRAIN-07, `equal_steps`).

    The three quantities TRAIN-07 asks to align - optimizer steps, batch size and
    real-image exposure - cannot all hold once the datasets differ in size, so
    one has to give. Fixing steps controls compute, which answers the sharper
    objection ("it only won because it trained longer"). The cost is that the
    synthetic arms see each real image fewer times, so that number is returned
    per arm and belongs in the report rather than in a footnote.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    reference = compositions[reference_arm]
    steps_per_epoch = max(1, reference.n_train_images // batch_size)
    total_steps = steps_per_epoch * reference_epochs

    plan: dict[str, dict[str, Any]] = {}
    for arm, comp in compositions.items():
        arm_steps_per_epoch = max(1, comp.n_train_images // batch_size)
        epochs = total_steps / arm_steps_per_epoch
        real_fraction = len(comp.real_train) / comp.n_train_images
        plan[arm] = {
            "total_steps": total_steps,
            "steps_per_epoch": arm_steps_per_epoch,
            "epochs": round(epochs, 3),
            "n_train_images": comp.n_train_images,
            # What the report has to state plainly: how often each real image is
            # actually seen once the step budget is held fixed.
            "real_image_exposures": round(
                total_steps * batch_size * real_fraction / len(comp.real_train), 2
            ),
        }
    return plan
