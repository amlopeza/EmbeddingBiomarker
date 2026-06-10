"""Tabular featurization, target construction and concordance for the baseline.

This is the Phase 0 "line to beat": the 12 raw context features (config/data.yaml)
encoded as a plain numeric matrix and handed to Cox-PH / RSF / XGBoost-Cox. The
survival target (``OS_STATUS``, ``OS_MONTHS``) is built separately by
:func:`make_target` and is NEVER part of the feature matrix — :class:`TabularFeaturizer`
refuses to encode a target column, mirroring the structural guard in ``prompts.py``.

Leakage discipline: every encoder (the top-K token vocabularies, the one-hot
category sets, the numeric medians and z-score statistics) is FIT ON THE TRAIN
FOLD ONLY via :meth:`TabularFeaturizer.fit`, then applied unchanged to val/test
via :meth:`~TabularFeaturizer.transform`. No statistic crosses the split.

Featurization per ``kind`` (config/data.yaml):
  * ``numeric``      -> median-impute (train median) + z-score, plus an optional
                       binary "was missing" indicator (missingness is signal).
  * ``categorical``  -> one-hot over the train categories with the first (sorted)
                       level dropped as reference, so each block is not collinear
                       with the model intercept; missing/blank tokens collapse to
                       an explicit "Not available" level; casing normalized per config.
  * ``text_list``    -> token count + multi-hot of the K most frequent tokens in
                       the train fold (genes for MUTATIONS, agents for
                       TREATMENT_HISTORY; repeats counted in the total).

Columns that are constant across the train fold (e.g. a "was missing" indicator
for a feature with no missing values in train) are dropped at fit time: they
carry no signal and make the Cox-PH design matrix singular.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ID_COLUMN = "PATIENT_ID"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_config(path: Path | str) -> dict:
    """Load a YAML config (data.yaml or survival.yaml)."""
    return yaml.safe_load(Path(path).read_text())


def load_table(path: Path | str) -> pd.DataFrame:
    """Load the canonical patient table; PATIENT_ID as string for split joins."""
    df = pd.read_csv(path)
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    return df


def load_splits(path: Path | str) -> dict[str, list[str]]:
    """Load the frozen train/val/test split (ids as strings)."""
    raw = json.loads(Path(path).read_text())
    return {part: [str(i) for i in ids] for part, ids in raw.items()}


def split_frames(
    df: pd.DataFrame, splits: dict[str, list[str]], id_column: str = ID_COLUMN
) -> dict[str, pd.DataFrame]:
    """Slice ``df`` into {train,val,test} frames by PATIENT_ID, preserving order."""
    by_id = df.set_index(id_column)
    out: dict[str, pd.DataFrame] = {}
    for part, ids in splits.items():
        present = [i for i in ids if i in by_id.index]
        out[part] = by_id.loc[present].reset_index()
    return out


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
def make_target(df: pd.DataFrame, data_config: dict) -> np.ndarray:
    """Build the sksurv structured target (event: bool, time: float).

    Event is ``OS_STATUS == event_positive_token`` ("1:DECEASED"); time is
    ``OS_MONTHS``. Returns a structured array usable by scikit-survival and
    decomposable into (event, time) for lifelines / xgboost.
    """
    target = data_config["target"]
    status_col = target["event_status"]
    time_col = target["time_months"]
    positive = target["event_positive_token"]

    event = (df[status_col].astype(str) == positive).to_numpy()
    time = df[time_col].astype(float).to_numpy()
    if np.isnan(time).any():
        raise ValueError(f"{int(np.isnan(time).sum())} rows have missing {time_col}")
    return Surv.from_arrays(event=event, time=time)


def target_columns(data_config: dict) -> set[str]:
    """The target column names (must never appear in the feature matrix)."""
    target = data_config.get("target", {})
    return {target.get("event_status"), target.get("time_months")} - {None}


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def concordance(target: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index. ``risk`` is higher == worse prognosis (shorter survival).

    Single source of truth so all three baselines (and later the embedding heads)
    are scored identically. ``target`` is the structured array from
    :func:`make_target`.
    """
    event = target["event"]
    time = target["time"]
    return float(concordance_index_censored(event, time, risk)[0])


# --------------------------------------------------------------------------- #
# Featurization
# --------------------------------------------------------------------------- #
def _tokenize(value: object) -> list[str]:
    """Split a text_list cell ("A, B, C") into stripped, upper-cased tokens."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"not available", "unknown", "none"}:
        return []
    return [t.strip().upper() for t in text.split(",") if t.strip()]


class TabularFeaturizer:
    """Fit-on-train encoder for the 12 context features -> numeric matrix.

    Use::

        fz = TabularFeaturizer(data_config, baseline_config)
        X_train = fz.fit_transform(train_df)
        X_val   = fz.transform(val_df)

    The feature spec and missing/casing rules come from ``data_config``
    (config/data.yaml); the top-K sizes and the missing-indicator toggle come
    from ``baseline_config`` (the ``baseline.features`` block of survival.yaml).
    """

    def __init__(self, data_config: dict, baseline_config: dict | None = None):
        self.features = data_config["features"]
        self.missing_tokens = {t.lower() for t in data_config.get("missing_tokens", [])}
        self.casing = data_config.get("normalize_casing", {})
        self._target_cols = target_columns(data_config)

        feat_cfg = (baseline_config or {}).get("features", {})
        self.top_k = {
            "MUTATIONS": int(feat_cfg.get("top_k_mutations", 50)),
            "TREATMENT_HISTORY": int(feat_cfg.get("top_k_treatments", 30)),
        }
        self.add_missing_indicator = bool(feat_cfg.get("add_missing_indicator", True))

        # Structural leakage guard: a target column must never be a feature.
        leak = {f["column"] for f in self.features} & self._target_cols
        if leak:
            raise ValueError(f"feature columns collide with target columns: {sorted(leak)}")

        self._fitted = False
        self.columns_: list[str] = []
        # Per-feature learned state (set in fit):
        self._num_stats: dict[str, tuple[float, float]] = {}   # col -> (median, std)
        self._cat_levels: dict[str, list[str]] = {}            # col -> ordered categories
        self._vocab: dict[str, list[str]] = {}                 # col -> top-K tokens

    # -- categorical normalization -----------------------------------------
    def _norm_cat(self, series: pd.Series) -> pd.Series:
        s = series.astype("object").where(series.notna(), "Not available")
        s = s.map(lambda v: "Not available" if str(v).strip().lower() in self.missing_tokens else v)
        s = s.map(lambda v: self.casing.get(str(v).strip(), str(v).strip()))
        return s.replace("", "Not available")

    # -- fit ----------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "TabularFeaturizer":
        self._num_stats.clear()
        self._cat_levels.clear()
        self._vocab.clear()

        for feat in self.features:
            col, kind = feat["column"], feat["kind"]
            if kind == "numeric":
                values = pd.to_numeric(df[col], errors="coerce")
                median = float(values.median())
                std = float(values.std(ddof=0)) or 1.0
                self._num_stats[col] = (median, std)
            elif kind == "categorical":
                levels = sorted(self._norm_cat(df[col]).unique().tolist())
                self._cat_levels[col] = levels
            elif kind == "text_list":
                counts: dict[str, int] = {}
                for cell in df[col]:
                    for tok in set(_tokenize(cell)):  # presence per patient
                        counts[tok] = counts.get(tok, 0) + 1
                top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                self._vocab[col] = [tok for tok, _ in top[: self.top_k.get(col, 50)]]
            else:
                raise ValueError(f"unknown feature kind {kind!r} for {col}")

        self._fitted = True
        # Build the full column set, then drop columns that are constant on TRAIN
        # (zero variance -> no signal, and singular for Cox). columns_ is the
        # surviving, fit-time-ordered set that transform() reindexes to.
        self.columns_ = self._build_columns()
        full = self._transform_full(df)
        keep = [c for c in self.columns_ if float(full[c].std(ddof=0)) > 0.0]
        self.columns_ = keep
        return self

    def _build_columns(self) -> list[str]:
        cols: list[str] = []
        for feat in self.features:
            col, kind = feat["column"], feat["kind"]
            if kind == "numeric":
                cols.append(f"{col}__z")
                if self.add_missing_indicator:
                    cols.append(f"{col}__missing")
            elif kind == "categorical":
                # Drop the first sorted level as reference (avoid collinearity
                # with the intercept the Cox baseline hazard already absorbs).
                cols.extend(f"{col}={lvl}" for lvl in self._cat_levels[col][1:])
            elif kind == "text_list":
                cols.append(f"{col}__count")
                cols.extend(f"{col}:{tok}" for tok in self._vocab[col])
        return cols

    # -- transform ----------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode ``df`` and reindex to the columns kept at fit time."""
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        return self._transform_full(df)[self.columns_]

    def _transform_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode every feature into the full (pre-pruning) column set."""
        blocks: list[pd.DataFrame] = []

        for feat in self.features:
            col, kind = feat["column"], feat["kind"]
            if kind == "numeric":
                values = pd.to_numeric(df[col], errors="coerce")
                median, std = self._num_stats[col]
                z = (values.fillna(median) - median) / std
                block = {f"{col}__z": z.to_numpy()}
                if self.add_missing_indicator:
                    block[f"{col}__missing"] = values.isna().astype(float).to_numpy()
                blocks.append(pd.DataFrame(block, index=df.index))
            elif kind == "categorical":
                norm = self._norm_cat(df[col])
                block = {
                    f"{col}={lvl}": (norm == lvl).astype(float).to_numpy()
                    for lvl in self._cat_levels[col]
                }
                blocks.append(pd.DataFrame(block, index=df.index))
            elif kind == "text_list":
                toks = df[col].map(_tokenize)
                block = {f"{col}__count": toks.map(len).astype(float).to_numpy()}
                for tok in self._vocab[col]:
                    block[f"{col}:{tok}"] = toks.map(lambda ts, t=tok: float(t in ts)).to_numpy()
                blocks.append(pd.DataFrame(block, index=df.index))

        return pd.concat(blocks, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
