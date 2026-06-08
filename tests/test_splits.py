"""Integrity tests for the seeded splitter."""

import numpy as np
import pytest

from embedbiomarker.splits import make_splits


def _ids(n):
    return [f"P-{i:05d}" for i in range(n)]


def test_no_overlap_and_full_coverage():
    ids = _ids(1000)
    s = make_splits(ids, seed=42)
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    # Zero overlap.
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    # Every patient lands in exactly one partition, none lost or invented.
    assert train | val | test == set(ids)
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == len(ids)


def test_ratios_are_approximately_80_10_10():
    ids = _ids(10000)
    s = make_splits(ids, seed=42)
    n = len(ids)
    assert abs(len(s["train"]) / n - 0.8) < 0.01
    assert abs(len(s["val"]) / n - 0.1) < 0.01
    assert abs(len(s["test"]) / n - 0.1) < 0.01


def test_deterministic_same_seed():
    ids = _ids(500)
    assert make_splits(ids, seed=42) == make_splits(ids, seed=42)


def test_input_order_does_not_matter():
    # Stable canonical ordering => shuffling the input must not change the split.
    ids = _ids(500)
    shuffled = list(ids)
    np.random.default_rng(7).shuffle(shuffled)
    assert make_splits(ids, seed=42) == make_splits(shuffled, seed=42)


def test_different_seed_changes_split():
    ids = _ids(500)
    assert make_splits(ids, seed=1) != make_splits(ids, seed=2)


def test_duplicates_rejected():
    with pytest.raises(ValueError):
        make_splits(["A", "A", "B"], seed=42)


def test_bad_ratios_rejected():
    with pytest.raises(ValueError):
        make_splits(_ids(10), ratios=(0.7, 0.1, 0.1))


def test_stratified_split_preserves_proportions_and_no_overlap():
    # Two strata with very different sizes; each must be ~80/10/10 on its own.
    ids = _ids(2000)
    strata = {i: ("A" if int(i.split("-")[1]) < 1500 else "B") for i in ids}
    s = make_splits(ids, seed=42, strata=strata)

    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert train & val == set() and train & test == set() and val & test == set()
    assert train | val | test == set(ids)

    # Stratum A (1500) should contribute ~80% to train; check its share.
    a_ids = {i for i in ids if strata[i] == "A"}
    a_in_train = len(a_ids & train)
    assert abs(a_in_train / len(a_ids) - 0.8) < 0.02


def test_stratified_requires_all_labels():
    ids = _ids(10)
    strata = {i: "A" for i in ids[:-1]}  # one id missing a label
    with pytest.raises(ValueError):
        make_splits(ids, seed=42, strata=strata)
