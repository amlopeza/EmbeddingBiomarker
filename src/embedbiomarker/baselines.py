"""The three Phase 0 tabular survival baselines, behind one interface.

Each estimator wraps a different library but exposes the same two methods::

    model.fit(X_train, y_train)          # X: DataFrame, y: sksurv structured array
    risk = model.predict_risk(X)         # higher == worse prognosis (shorter survival)

``predict_risk`` is normalized so that **larger means higher hazard** for all three
models, which is exactly the orientation :func:`embedbiomarker.survival.concordance`
expects — so the same scoring call works regardless of backend.

  * :class:`CoxPHBaseline`   — lifelines Cox proportional hazards (ridge-penalized).
  * :class:`RSFBaseline`     — scikit-survival Random Survival Forest.
  * :class:`XGBoostCoxBaseline` — XGBoost with the ``survival:cox`` objective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest

# Internal column names for the duration/event frame lifelines wants.
_DURATION = "_duration"
_EVENT = "_event"


def _unpack(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the sksurv structured target into (event_bool, time_float)."""
    return y["event"].astype(bool), y["time"].astype(float)


class CoxPHBaseline:
    """Cox proportional hazards (lifelines), ridge-penalized for stability."""

    def __init__(self, penalizer: float = 0.1):
        self.penalizer = penalizer
        self.model = CoxPHFitter(penalizer=penalizer)
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CoxPHBaseline":
        event, time = _unpack(y)
        self.columns_ = list(X.columns)
        df = X.copy()
        df[_DURATION] = time
        df[_EVENT] = event.astype(int)
        self.model.fit(df, duration_col=_DURATION, event_col=_EVENT)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        # partial hazard = exp(beta . x); strictly increasing in linear risk.
        return self.model.predict_partial_hazard(X[self.columns_]).to_numpy()


class RSFBaseline:
    """Random Survival Forest (scikit-survival)."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 5,
        min_samples_leaf: int = 15,
        max_features: str | int | float = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RSFBaseline":
        self.columns_ = list(X.columns)
        self.model.fit(X.to_numpy(), y)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        # RSF.predict returns the risk score (sum of cumulative hazard); higher == worse.
        return self.model.predict(X[self.columns_].to_numpy())


class XGBoostCoxBaseline:
    """Gradient-boosted Cox (XGBoost ``survival:cox``)."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs,
    ):
        # eval_metric/objective are passed through from config; drop non-constructor keys.
        kwargs.pop("objective", None)
        self.model = xgb.XGBRegressor(
            objective="survival:cox",
            eval_metric=kwargs.pop("eval_metric", "cox-nloglik"),
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs,
        )
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBoostCoxBaseline":
        event, time = _unpack(y)
        # survival:cox encodes censoring in the label sign: positive time for an
        # observed event, negative time for a censored observation.
        label = np.where(event, time, -time)
        self.columns_ = list(X.columns)
        self.model.fit(X.to_numpy(), label)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        # survival:cox prediction is the relative risk e^f(x); higher == worse.
        return self.model.predict(X[self.columns_].to_numpy())


# Registry consumed by scripts/10_baseline_tabular.py.
def build_model(name: str, config: dict):
    """Instantiate a baseline by name using its sub-block of ``baseline`` config."""
    builders = {
        "coxph": lambda c: CoxPHBaseline(**c.get("coxph", {})),
        "rsf": lambda c: RSFBaseline(**c.get("rsf", {})),
        "xgboost_cox": lambda c: XGBoostCoxBaseline(**c.get("xgboost_cox", {})),
    }
    if name not in builders:
        raise ValueError(f"unknown baseline model {name!r}; have {sorted(builders)}")
    return builders[name](config)
