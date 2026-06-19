"""Phase 2 — treatment-aware survival analysis.

The Phase 1 finding is that the MedGemma embedding adds genuine,
*semantic* prognostic signal over the 12 tabular features, concentrated in
pancreas. Phase 2 asks the clinical question that is the paper's headline: the
LM-extracted risk score — is it **prognostic** (it ranks survival regardless of
the therapy given) or **predictive** (its effect is modified by the therapy)?

Pipeline (all leakage-safe, all built on the existing ``survival`` / ``baselines``
machinery):

  1. :func:`crossfit_risk` — an out-of-fold risk score for every patient, so the
     downstream KM / log-rank / Cox analyses run on the full cohort (pancreas
     n=2940, not the 294 of the frozen test). Two flavours, both returned:
       * ``mode="pan"``      — one XGBoost-Cox (tab ⊕ emb) trained across all five
                               tumors, cross-fitted (the Phase 1 model).
       * ``mode="specific"`` — one model PER tumor, cross-fitted within tumor.
  2. :func:`assign_treatment_strata` — per-cancer therapy taxonomy from
     ``config/treatments.yaml`` (regimen arms for pancreas, drug-class flags for
     the rest). Untreated = zero agents.
  3. :func:`km_logrank`, :func:`cox_adjusted_hr`, :func:`cox_interaction` — thin
     wrappers over lifelines, the single source of truth for the stats.

Honest caveat (reported in the paper): ``TREATMENT_HISTORY`` is post-baseline
(living longer -> receiving more lines), so treatment groups carry immortal-time
bias and there are no line/date boundaries. The framing is therefore *stratified
prognostic validation* + interaction as an effect-modification probe, NOT a causal
treatment-efficacy claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from . import survival as S
from .baselines import build_model

CANCER_COL = "CANCER_TYPE"
TRT_COL = "TREATMENT_HISTORY"
AGE_COL = "CURRENT_AGE_DEID"
STAGE_COL = "STAGE_HIGHEST_RECORDED"


# --------------------------------------------------------------------------- #
# Out-of-fold risk score
# --------------------------------------------------------------------------- #
def _emb_cols(emb: pd.DataFrame) -> list[str]:
    return [c for c in emb.columns if c.startswith("e")]


def _both_matrix(frame: pd.DataFrame, fz: S.TabularFeaturizer, emb: pd.DataFrame,
                 emb_cols: list[str], *, fit: bool) -> pd.DataFrame:
    """tab ⊕ emb for one fold; featurizer fit on TRAIN only (``fit=True``)."""
    Xt = (fz.fit_transform(frame) if fit else fz.transform(frame)).reset_index(drop=True)
    ids = frame[S.ID_COLUMN].to_numpy()
    Xe = emb.loc[ids, emb_cols].reset_index(drop=True)
    return pd.concat([Xt, Xe], axis=1)


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, emb: pd.DataFrame,
                 emb_cols: list[str], data_cfg: dict, baseline_cfg: dict) -> np.ndarray:
    """Fit XGBoost-Cox(tab⊕emb) on the train fold, return risk on the test fold."""
    fz = S.TabularFeaturizer(data_cfg, baseline_cfg)
    Xtr = _both_matrix(train_df, fz, emb, emb_cols, fit=True)
    Xte = _both_matrix(test_df, fz, emb, emb_cols, fit=False)
    ytr = S.make_target(train_df, data_cfg)
    model = build_model("xgboost_cox", baseline_cfg).fit(Xtr, ytr)
    return model.predict_risk(Xte)


def _stratified_folds(n: int, strata: np.ndarray, n_splits: int, seed: int):
    """StratifiedKFold split positions (lazy import to keep module import light)."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return skf.split(np.zeros(n), strata)


def crossfit_risk(df: pd.DataFrame, emb: pd.DataFrame, data_cfg: dict,
                  baseline_cfg: dict, *, mode: str = "pan", n_splits: int = 5,
                  seed: int = 42) -> pd.Series:
    """Per-patient out-of-fold risk (higher == worse). Leakage-safe by construction.

    ``mode="pan"`` cross-fits one model over the whole cohort, stratified by
    CANCER_TYPE x OS_STATUS. ``mode="specific"`` cross-fits a separate model
    within each tumor, stratified by OS_STATUS. ``emb`` is indexed by PATIENT_ID.
    Returns a Series indexed by PATIENT_ID in ``df`` order.
    """
    emb_cols = _emb_cols(emb)
    status_col = data_cfg["target"]["event_status"]
    n = len(df)
    risk = np.full(n, np.nan)

    if mode == "pan":
        strata = (df[CANCER_COL].astype(str) + "|" + df[status_col].astype(str)).to_numpy()
        for tr, te in _stratified_folds(n, strata, n_splits, seed):
            risk[te] = _fit_predict(df.iloc[tr], df.iloc[te], emb, emb_cols, data_cfg, baseline_cfg)
    elif mode == "specific":
        cancers = df[CANCER_COL].to_numpy()
        for ct in sorted(pd.unique(cancers)):
            pos = np.where(cancers == ct)[0]
            sub = df.iloc[pos]
            strata = sub[status_col].astype(str).to_numpy()
            for tr_l, te_l in _stratified_folds(len(pos), strata, n_splits, seed):
                risk[pos[te_l]] = _fit_predict(
                    sub.iloc[tr_l], sub.iloc[te_l], emb, emb_cols, data_cfg, baseline_cfg)
    else:
        raise ValueError(f"mode must be 'pan' or 'specific', got {mode!r}")

    if np.isnan(risk).any():
        raise RuntimeError(f"{int(np.isnan(risk).sum())} patients got no OOF risk")
    return pd.Series(risk, index=df[S.ID_COLUMN].to_numpy(), name=f"risk_{mode}")


# --------------------------------------------------------------------------- #
# Treatment strata
# --------------------------------------------------------------------------- #
def load_treatment_taxonomy(path: Path | str) -> dict:
    """Load the per-cancer agent->class taxonomy (config/treatments.yaml)."""
    return yaml.safe_load(Path(path).read_text())


def _agent_set(cell: object) -> set[str]:
    """The set of agents a patient received (same tokenizer as the features)."""
    return set(S._tokenize(cell))


def treatment_strata(frame: pd.DataFrame, tax_entry: dict):
    """Treatment strata for ONE cancer's frame, per its taxonomy entry.

    Returns ``(mode, strata)`` where:
      * ``mode="regimen"`` -> a Series of exclusive arm labels (incl. "untreated",
        "other-treated"); arms matched in config order, first match wins.
      * ``mode="classes"`` -> a DataFrame of non-exclusive boolean flag columns
        (one per drug class) plus an "untreated" column.
    Index is aligned to ``frame.index``.
    """
    toks = frame[TRT_COL].map(_agent_set)
    treated = toks.map(len) > 0

    if tax_entry["mode"] == "regimen":
        labels = []
        for agents, is_tr in zip(toks, treated):
            if not is_tr:
                labels.append("untreated")
                continue
            arm = None
            for spec in tax_entry["arms"]:
                req_all = set(spec.get("require_all", []))
                req_any = set(spec.get("require_any", []))
                if req_all and not req_all.issubset(agents):
                    continue
                if req_any and not (req_any & agents):
                    continue
                arm = spec["name"]
                break
            labels.append(arm or "other-treated")
        return "regimen", pd.Series(labels, index=frame.index, name="regimen")

    if tax_entry["mode"] == "classes":
        cols = {}
        for cname, agents in tax_entry["classes"].items():
            agset = set(a.upper() for a in agents)
            cols[cname] = toks.map(lambda t, s=agset: bool(t & s))
        out = pd.DataFrame(cols, index=frame.index)
        out["untreated"] = ~treated
        return "classes", out

    raise ValueError(f"unknown taxonomy mode {tax_entry['mode']!r}")


def assign_treatment_strata(df: pd.DataFrame, taxonomy: dict) -> dict:
    """Apply the taxonomy per cancer. Returns {cancer: (mode, strata)} for the
    cancers present both in ``df`` and the taxonomy."""
    out = {}
    for ct in sorted(df[CANCER_COL].unique()):
        if ct in taxonomy:
            out[ct] = treatment_strata(df[df[CANCER_COL] == ct], taxonomy[ct])
    return out


# --------------------------------------------------------------------------- #
# Risk dichotomization
# --------------------------------------------------------------------------- #
def risk_high_low(risk: np.ndarray) -> np.ndarray:
    """Split a risk vector into "high"/"low" at its median (the per-tumor cutpoint)."""
    r = np.asarray(risk, dtype=float)
    med = float(np.median(r))
    return np.where(r > med, "high", "low")


# --------------------------------------------------------------------------- #
# lifelines wrappers (single source of truth for the stats)
# --------------------------------------------------------------------------- #
def _finite(x: float):
    """JSON-safe float: map inf/nan (e.g. an unreached median survival) to None."""
    return float(x) if np.isfinite(x) else None


def km_logrank(time: np.ndarray, event: np.ndarray, group: np.ndarray) -> dict:
    """Median OS per group + multivariate log-rank p across groups."""
    time = np.asarray(time, float)
    event = np.asarray(event, bool)
    group = np.asarray(group)
    groups = [g for g in pd.unique(group)]

    medians, ns, events = {}, {}, {}
    kmf = KaplanMeierFitter()
    for g in groups:
        m = group == g
        kmf.fit(time[m], event[m])
        medians[str(g)] = _finite(kmf.median_survival_time_)
        ns[str(g)] = int(m.sum())
        events[str(g)] = int(event[m].sum())

    p = None
    if len(groups) >= 2 and all(v >= 1 for v in ns.values()):
        p = float(multivariate_logrank_test(time, group, event).p_value)
    return {"logrank_p": p, "median_os": medians, "n": ns, "events": events}


def _design(frame: pd.DataFrame, risk: np.ndarray, data_cfg: dict) -> pd.DataFrame:
    """lifelines design: standardized risk + age + one-hot stage + (_t,_e target).

    Zero-variance covariates are dropped (singular otherwise). Risk and age are
    z-scored so the risk coefficient is a HR-per-SD.

    Risk is log-transformed first: ``predict_risk`` returns the relative risk
    e^f(x) (heavy right tail), and z-scoring that raw exponential destabilizes the
    Newton-Raphson fit when covariates are added (NSCLC diverged to a nonsense
    coef). log(risk) IS the Cox linear predictor f(x) and is well-behaved; the
    monotonic transform leaves the C-index and the median high/low split unchanged.
    """
    y = S.make_target(frame, data_cfg)
    r = np.log(np.clip(np.asarray(risk, float), 1e-12, None))
    d = pd.DataFrame(index=range(len(frame)))
    d["risk_z"] = (r - r.mean()) / (r.std(ddof=0) or 1.0)

    age = pd.to_numeric(frame[AGE_COL], errors="coerce")
    d["age_z"] = ((age - age.mean()) / (age.std(ddof=0) or 1.0)).fillna(0.0).to_numpy()

    stage = frame[STAGE_COL].astype(str).fillna("NA").to_numpy()
    dummies = pd.get_dummies(stage, prefix="stage", drop_first=True).astype(float).reset_index(drop=True)
    d = pd.concat([d.reset_index(drop=True), dummies], axis=1)

    d["_t"] = y["time"]
    d["_e"] = y["event"].astype(int)
    cov = [c for c in d.columns if c not in ("_t", "_e")]
    keep = [c for c in cov if float(d[c].std(ddof=0)) > 0.0]
    return d[keep + ["_t", "_e"]]


def cox_adjusted_hr(frame: pd.DataFrame, risk: np.ndarray, data_cfg: dict,
                    penalizer: float = 0.1) -> dict | None:
    """HR of the risk score PER SD, adjusted for age + stage. None if it can't fit."""
    d = _design(frame, risk, data_cfg)
    if "risk_z" not in d.columns or int(d["_e"].sum()) < 5:
        return None
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(d, duration_col="_t", event_col="_e")
    except Exception:
        return None
    s = cph.summary.loc["risk_z"]
    return {
        "hr_per_sd": float(np.exp(s["coef"])),
        "ci_low": float(np.exp(s["coef lower 95%"])),
        "ci_high": float(np.exp(s["coef upper 95%"])),
        "p": float(s["p"]),
        "n": int(len(d)),
        "events": int(d["_e"].sum()),
    }


def cox_interaction(frame: pd.DataFrame, risk: np.ndarray, treat_binary: np.ndarray,
                    data_cfg: dict, penalizer: float = 0.1) -> dict | None:
    """Cox: risk + treatment + risk×treatment + age + stage.

    ``treat_binary`` is a 0/1 indicator (received a given regimen/class). The
    risk×treatment term tests effect modification: a significant interaction is
    the "predictive" signal, a null one means the score is purely prognostic.
    Returns the interaction term stats, or None if it degenerates.
    """
    d = _design(frame, risk, data_cfg).copy()
    t = np.asarray(treat_binary, float)
    if "risk_z" not in d.columns or np.std(t) == 0 or int(d["_e"].sum()) < 10:
        return None
    d.insert(0, "trt", t)
    d.insert(0, "risk_x_trt", d["risk_z"].to_numpy() * t)
    if float(d["risk_x_trt"].std(ddof=0)) == 0.0:
        return None
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(d, duration_col="_t", event_col="_e")
    except Exception:
        return None
    s = cph.summary.loc["risk_x_trt"]
    return {
        "interaction_hr": float(np.exp(s["coef"])),
        "ci_low": float(np.exp(s["coef lower 95%"])),
        "ci_high": float(np.exp(s["coef upper 95%"])),
        "p": float(s["p"]),
        "n_treated": int(t.sum()),
        "n": int(len(d)),
    }
