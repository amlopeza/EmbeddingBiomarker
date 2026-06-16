"""Phase 2 — treatment taxonomy + out-of-fold leakage guards.

Fast, hermetic tests: the regimen/class assignment logic on canonical patients,
and the OOF cross-fit partition property (every patient is predicted exactly once,
from a fold whose train set never contains them).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embedbiomarker import analysis as A  # noqa: E402

TAXONOMY = A.load_treatment_taxonomy(REPO_ROOT / "config/treatments.yaml")


def _frame(cancer: str, histories: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        A.CANCER_COL: cancer,
        A.TRT_COL: histories,
    })


# --------------------------------------------------------------------------- #
# Pancreas — regimen (exclusive arms, priority order)
# --------------------------------------------------------------------------- #
def test_pancreas_regimen_arms():
    frame = _frame("Pancreatic Cancer", [
        "OXALIPLATIN, IRINOTECAN, FLUOROURACIL, LEUCOVORIN",  # FOLFIRINOX-like
        "GEMCITABINE, PACLITAXEL PROTEIN-BOUND",              # gemcitabine-based
        "OXALIPLATIN, FLUOROURACIL",                         # no irinotecan -> other
        "CAPECITABINE",                                      # treated, no backbone -> other
        "Unknown",                                           # untreated
        "GEMCITABINE, OXALIPLATIN, IRINOTECAN",              # both backbones -> FOLFIRINOX wins
    ])
    mode, strata = A.treatment_strata(frame, TAXONOMY["Pancreatic Cancer"])
    assert mode == "regimen"
    assert list(strata) == [
        "FOLFIRINOX-like", "gemcitabine-based", "other-treated",
        "other-treated", "untreated", "FOLFIRINOX-like",
    ]


def test_pancreas_untreated_tokens():
    # empty / sentinel cells collapse to untreated (same tokenizer as the features)
    frame = _frame("Pancreatic Cancer", ["", "Not available", "none", np.nan])
    _, strata = A.treatment_strata(frame, TAXONOMY["Pancreatic Cancer"])
    assert (strata == "untreated").all()


# --------------------------------------------------------------------------- #
# Breast — non-exclusive class flags
# --------------------------------------------------------------------------- #
def test_breast_class_flags_are_non_exclusive():
    frame = _frame("Breast Cancer", [
        "LETROZOLE, PALBOCICLIB",   # endocrine + CDK4/6
        "PACLITAXEL",               # chemo only
        "TRASTUZUMAB, PERTUZUMAB",  # HER2 only
        "Unknown",                  # untreated
    ])
    mode, strata = A.treatment_strata(frame, TAXONOMY["Breast Cancer"])
    assert mode == "classes"
    assert bool(strata.loc[0, "endocrine"]) and bool(strata.loc[0, "cdk4_6"])
    assert not bool(strata.loc[0, "untreated"])
    assert bool(strata.loc[1, "chemo"]) and not bool(strata.loc[1, "endocrine"])
    assert bool(strata.loc[2, "her2_targeted"])
    assert bool(strata.loc[3, "untreated"]) and not strata.loc[3, ["endocrine", "chemo"]].any()


def test_assign_treatment_strata_covers_present_cancers():
    df = pd.concat([
        _frame("Pancreatic Cancer", ["GEMCITABINE"]),
        _frame("Breast Cancer", ["LETROZOLE"]),
    ], ignore_index=True)
    out = A.assign_treatment_strata(df, TAXONOMY)
    assert set(out) == {"Pancreatic Cancer", "Breast Cancer"}
    assert out["Pancreatic Cancer"][0] == "regimen"
    assert out["Breast Cancer"][0] == "classes"


# --------------------------------------------------------------------------- #
# Risk dichotomization + OOF leakage guard
# --------------------------------------------------------------------------- #
def test_risk_high_low_median_split():
    groups = A.risk_high_low(np.array([1.0, 2.0, 3.0, 4.0]))
    assert list(groups) == ["low", "low", "high", "high"]


def test_oof_folds_partition_without_leakage():
    # The leakage guarantee of crossfit_risk: across folds the test indices form a
    # disjoint cover of all patients, and no fold's train set overlaps its test set.
    n = 200
    rng = np.random.default_rng(0)
    strata = rng.integers(0, 4, size=n)  # stand-in for CANCER x STATUS strata
    folds = list(A._stratified_folds(n, strata, n_splits=5, seed=42))

    test_concat = np.concatenate([te for _, te in folds])
    assert sorted(test_concat.tolist()) == list(range(n))  # each patient predicted once
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(test_idx.tolist())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
