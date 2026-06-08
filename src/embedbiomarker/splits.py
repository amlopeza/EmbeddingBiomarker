"""Deterministic, seeded train/val/test splits by PATIENT_ID.

The split is reproducible by construction: patient ids are sorted into a stable
canonical order *before* any shuffling, and all randomness flows from a single
``np.random.default_rng(seed)``. This fixes the bug inherited from the old
project, where ``np.random.shuffle`` ran unseeded before a seeded
``train_test_split``, making the partition irreproducible.

When ``strata`` is provided the 80/10/10 split is performed independently within
each stratum (e.g. CANCER_TYPE x OS_STATUS) and concatenated, so each partition
keeps comparable tumor and event proportions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

Split = dict[str, list[str]]

DEFAULT_RATIOS = (0.8, 0.1, 0.1)


def _split_one_group(
    ids: list[str], rng: np.random.Generator, ratios: tuple[float, float, float]
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle a single (already canonically ordered) id list and cut 80/10/10."""
    train_r, val_r, _ = ratios
    arr = np.array(ids, dtype=object)
    perm = rng.permutation(len(arr))
    shuffled = arr[perm].tolist()

    n = len(shuffled)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    # test takes the remainder so the three parts always sum to n exactly.
    n_val = min(n_val, n - n_train)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def make_splits(
    patient_ids: Sequence[str],
    seed: int = 42,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    strata: Mapping[str, str] | None = None,
) -> Split:
    """Return a deterministic {"train","val","test"} split of ``patient_ids``.

    Args:
        patient_ids: iterable of unique patient ids.
        seed: RNG seed; identical input + seed always yields the identical split.
        ratios: (train, val, test) fractions; must sum to 1.0.
        strata: optional mapping patient_id -> stratum label. When given, the
            split is done per stratum and concatenated (stratified split).

    Raises:
        ValueError: on duplicate ids, ratio mismatch, or missing strata labels.
    """
    ids = list(patient_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("patient_ids contains duplicates")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} -> {sum(ratios)}")

    rng = np.random.default_rng(seed)
    out: Split = {"train": [], "val": [], "test": []}

    if strata is None:
        # Single group: stable sort, then shuffle.
        groups = {"__all__": sorted(ids)}
    else:
        missing = [i for i in ids if i not in strata]
        if missing:
            raise ValueError(f"{len(missing)} ids have no stratum label (e.g. {missing[:3]})")
        grouped: dict[str, list[str]] = defaultdict(list)
        for i in sorted(ids):  # stable order before grouping
            grouped[str(strata[i])].append(i)
        # Iterate strata in a fixed (sorted) key order for full determinism.
        groups = {k: grouped[k] for k in sorted(grouped)}

    for group_ids in groups.values():
        tr, va, te = _split_one_group(group_ids, rng, ratios)
        out["train"].extend(tr)
        out["val"].extend(va)
        out["test"].extend(te)

    # Sort each partition for a stable, diff-friendly artifact.
    for k in out:
        out[k] = sorted(out[k])
    return out
