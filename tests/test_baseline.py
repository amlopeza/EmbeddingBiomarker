"""Guards for the tabular baseline: no target in the matrix, no leakage across split.

Mirrors the rigor of ``test_no_leakage.py`` but for the numeric feature matrix
fed to Cox-PH / RSF / XGBoost-Cox, rather than the text prompts.
"""

import numpy as np
import pandas as pd
import pytest

from embedbiomarker import survival as S


def _config() -> dict:
    return {
        "target": {
            "event_status": "OS_STATUS",
            "time_months": "OS_MONTHS",
            "event_positive_token": "1:DECEASED",
        },
        "missing_tokens": ["", "NA", "nan", "None"],
        "normalize_casing": {"Do Not Report": "Do not report"},
        "features": [
            {"label": "Age", "column": "AGE", "kind": "numeric"},
            {"label": "Stage", "column": "STAGE", "kind": "categorical"},
            {"label": "Mutations", "column": "MUT", "kind": "text_list"},
        ],
    }


def _frame(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "PATIENT_ID": [f"P{i}" for i in range(n)],
        "AGE": rng.integers(40, 85, n).astype(float),
        "STAGE": rng.choice(["Stage 1-3", "Stage 4", None], n),
        "MUT": rng.choice(["TP53, KRAS", "EGFR", "TP53, EGFR, BRAF", "Not available"], n),
        "OS_STATUS": rng.choice(["0:LIVING", "1:DECEASED"], n),
        "OS_MONTHS": rng.uniform(1, 100, n),
    })


def test_featurizer_rejects_target_as_feature():
    cfg = _config()
    cfg["features"].append({"label": "Leak", "column": "OS_STATUS", "kind": "categorical"})
    with pytest.raises(ValueError):
        S.TabularFeaturizer(cfg)


def test_target_columns_never_in_matrix():
    cfg = _config()
    fz = S.TabularFeaturizer(cfg, {"features": {}})
    X = fz.fit_transform(_frame())
    forbidden = {"OS_STATUS", "OS_MONTHS"}
    assert not (forbidden & set(X.columns)), "target column leaked into feature matrix"
    # No column may even mention a target name as a substring.
    assert not [c for c in X.columns if "OS_STATUS" in c or "OS_MONTHS" in c]


def test_fit_on_train_columns_stable_on_val():
    """Encoder fit on train must produce the identical column set on unseen data."""
    cfg = _config()
    fz = S.TabularFeaturizer(cfg, {"features": {"top_k_mutations": 3}})
    X_train = fz.fit_transform(_frame(60))
    X_val = fz.transform(_frame(20))  # different rows, possibly unseen tokens
    assert list(X_train.columns) == list(X_val.columns) == fz.columns_
    assert not X_val.isna().any().any()


def test_concordance_orientation():
    """Higher risk must mean shorter survival -> C-index well above 0.5."""
    cfg = _config()
    rng = np.random.default_rng(1)
    time = rng.uniform(1, 100, 200)
    event = np.ones(200, dtype=bool)
    df = pd.DataFrame({"OS_STATUS": "1:DECEASED", "OS_MONTHS": time})
    y = S.make_target(df, cfg)
    risk = -time  # perfectly anti-correlated with survival time
    assert S.concordance(y, risk) > 0.9
