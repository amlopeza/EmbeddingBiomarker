#!/usr/bin/env python
"""Frozen-model external validation on GENIE BPC (Phase 3, step 14.5).

Principle: ZERO refit. The TabularFeaturizer and the XGBoost-Cox head are fit on
MSK-CHORD (train+val, decision D1) and applied UNCHANGED to the external cohort —
no statistic crosses cohorts. The external cohort is a fully independent test set,
so risk is the direct ``predict_risk`` (no cross-fit needed: there is no leakage to
guard against, unlike the in-cohort Phase 2 OOF risk).

For each feature set (``tab`` = the 12 features only; ``both`` = tab + MedGemma
embeddings) it reports on the external cohort:
  * C-index, global and per CANCER_TYPE (concordance is rank-based -> robust to a
    uniform OS time-origin shift between cohorts; see caveat 2 in plans.md).
  * KM medians + log-rank for high/low risk at TWO cutpoints (decision D2):
      - intra-external median (pure discrimination within the external cohort), and
      - the MSK reference median (does the MSK-learned threshold transfer?).
  * Cox HR of risk per SD, adjusted for age + stage.

Inputs (must already exist):
  data/interim/data_prompts.csv               (MSK table)
  data/interim/splits.json                    (frozen MSK split)
  data/interim/genie_<cohort>.csv             (external table; scripts/40)
  data/processed/embeddings/<model>__<tid>__<pool>__by_patient.parquet        (MSK)
  data/processed/embeddings/genie_<cohort>__<model>__<tid>__<pool>__by_patient.parquet

Usage:
    python scripts/41_external_validate.py                  # PANC pilot, medgemma15
    python scripts/41_external_validate.py --cohort PANC --model medgemma15 --pooling mean
    python scripts/41_external_validate.py --all            # all 5 solid tumors (step 14.7)
    python scripts/41_external_validate.py --cohort NSCLC CRC   # a subset

The MSK frozen fit (featurizer + XGBoost-Cox, for both ``tab`` and ``both``) is
identical across cohorts, so with ``--all`` it is computed ONCE and reused; only
each cohort's external table + embeddings are loaded per iteration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run without an editable install

from embedbiomarker import analysis as A
from embedbiomarker import survival as S
from embedbiomarker.baselines import build_model

CANCER_COL = "CANCER_TYPE"


def _matrix(frame: pd.DataFrame, fz: S.TabularFeaturizer, emb: pd.DataFrame | None,
            emb_cols: list[str], *, fit: bool, use_emb: bool) -> pd.DataFrame:
    """tab (and optionally ⊕ emb) for one frame; featurizer fit only when fit=True."""
    Xt = (fz.fit_transform(frame) if fit else fz.transform(frame)).reset_index(drop=True)
    if not use_emb:
        return Xt
    ids = frame[S.ID_COLUMN].to_numpy()
    Xe = emb.loc[ids, emb_cols].reset_index(drop=True)
    return pd.concat([Xt, Xe], axis=1)


def _fit_frozen(trainval: pd.DataFrame, emb: pd.DataFrame | None, emb_cols: list[str],
                data_cfg: dict, bcfg: dict, *, use_emb: bool):
    """Fit featurizer + XGBoost-Cox on MSK train+val. Returns (model, featurizer)."""
    fz = S.TabularFeaturizer(data_cfg, bcfg)
    X = _matrix(trainval, fz, emb, emb_cols, fit=True, use_emb=use_emb)
    y = S.make_target(trainval, data_cfg)
    model = build_model("xgboost_cox", bcfg).fit(X, y)
    return model, fz


def _km_block(time: np.ndarray, event: np.ndarray, risk: np.ndarray,
              cutpoint: float | None) -> dict:
    """KM/log-rank for high/low risk. cutpoint=None -> intra-cohort median."""
    if cutpoint is None:
        group = A.risk_high_low(risk)
    else:
        group = np.where(np.asarray(risk, float) > cutpoint, "high", "low")
    return A.km_logrank(time, event, group)


def _evaluate(ext: pd.DataFrame, risk_ext: np.ndarray, msk_median: float,
              data_cfg: dict) -> dict:
    """All external metrics for one risk vector (one feature set)."""
    target = S.make_target(ext, data_cfg)
    time = target["time"].astype(float)
    event = target["event"].astype(bool)

    # Per-tumor C-index (global == the single tumor for the PANC pilot).
    per_cancer = {}
    for ct in sorted(pd.unique(ext[CANCER_COL])):
        m = (ext[CANCER_COL] == ct).to_numpy()
        if m.sum() >= 10 and event[m].sum() >= 3:
            per_cancer[str(ct)] = {
                "n": int(m.sum()),
                "concordance": S.concordance(target[m], risk_ext[m]),
            }

    return {
        "n": int(len(ext)),
        "events": int(event.sum()),
        "concordance": S.concordance(target, risk_ext),
        "concordance_per_cancer": per_cancer,
        "cox_adjusted_hr": A.cox_adjusted_hr(ext, risk_ext, data_cfg),
        "km_intra_external_median": _km_block(time, event, risk_ext, None),
        "km_msk_reference_median": _km_block(time, event, risk_ext, msk_median),
    }


# 5 solid tumors of MSK-CHORD. BLADDER is excluded on purpose: it is not a
# MSK-CHORD tumor, so the frozen model has no prior for it (see external.py).
SOLID_TUMORS = ["PANC", "NSCLC", "CRC", "BrCa", "Prostate"]


def run_cohort(cohort_name: str, frozen: list, ext: pd.DataFrame, ext_emb: pd.DataFrame,
               emb_cols: list[str], data_cfg: dict, args, tid: str,
               out: Path | None, out_patients: Path | None) -> dict:
    """Apply the pre-fit frozen models to one external cohort; write JSON + parquet."""
    cohort = cohort_name.lower()
    results = {
        "cohort": cohort_name,
        "model": args.model,
        "template_id": tid,
        "pooling": args.pooling,
        "fit_on": "MSK train+val",
        "external_n": int(len(ext)),
        "feature_sets": {},
        "msk_reference_median": {},
    }
    target = S.make_target(ext, data_cfg)
    patients = pd.DataFrame({
        S.ID_COLUMN: ext[S.ID_COLUMN].to_numpy(),
        CANCER_COL: ext[CANCER_COL].to_numpy(),
        "os_months": target["time"].astype(float),
        "event": target["event"].astype(int),
    })
    for name, use_emb, model, _fz, msk_median in frozen:
        Xext = _matrix(ext, _fz, ext_emb, emb_cols, fit=False, use_emb=use_emb)
        risk_ext = model.predict_risk(Xext)
        patients[f"risk_{name}"] = risk_ext
        results["msk_reference_median"][name] = msk_median
        results["feature_sets"][name] = _evaluate(ext, risk_ext, msk_median, data_cfg)
        c = results["feature_sets"][name]["concordance"]
        print(f"  [{name:4s}] external C-index = {c:.4f}")

    out = out or REPO_ROOT / f"results/external_genie_{cohort}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"  wrote {out.relative_to(REPO_ROOT)}")

    # Per-patient risk + survival (gitignored) so the figures notebook can draw KM
    # curves without re-fitting — mirrors data/processed/treatment_oof_patients.parquet.
    pat_out = out_patients or REPO_ROOT / f"data/processed/external_{cohort}_patients.parquet"
    pat_out.parent.mkdir(parents=True, exist_ok=True)
    patients.to_parquet(pat_out, index=False)
    print(f"  wrote {pat_out.relative_to(REPO_ROOT)}: {len(patients)} patients")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", nargs="+", default=["PANC"],
                    help="one or more cohorts (e.g. --cohort NSCLC CRC); ignored if --all")
    ap.add_argument("--all", action="store_true",
                    help=f"run all 5 solid tumors: {', '.join(SOLID_TUMORS)} (step 14.7)")
    ap.add_argument("--model", default="medgemma15")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--template-id", default=None)
    ap.add_argument("--table", type=Path, default=REPO_ROOT / "data/interim/data_prompts.csv")
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "data/interim/prompts.parquet")
    ap.add_argument("--splits", type=Path, default=REPO_ROOT / "data/interim/splits.json")
    ap.add_argument("--emb-dir", type=Path, default=REPO_ROOT / "data/processed/embeddings")
    ap.add_argument("--external-dir", type=Path, default=REPO_ROOT / "data/interim")
    ap.add_argument("--data-config", type=Path, default=REPO_ROOT / "config/data.yaml")
    ap.add_argument("--survival-config", type=Path, default=REPO_ROOT / "config/survival.yaml")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/external_genie_<cohort>.json")
    ap.add_argument("--out-patients", type=Path, default=None,
                    help="per-patient risk+survival parquet (gitignored) for KM curves; "
                         "default: data/processed/external_<cohort>_patients.parquet")
    args = ap.parse_args()

    cohorts = SOLID_TUMORS if args.all else list(args.cohort)
    if (args.out or args.out_patients) and len(cohorts) > 1:
        raise SystemExit("--out/--out-patients are only valid for a single cohort")

    data_cfg = S.load_config(args.data_config)
    bcfg = S.load_config(args.survival_config)["baseline"]
    tid = args.template_id or str(pd.read_parquet(args.prompts)["template_id"].iloc[0])

    def emb_file(prefix: str) -> Path:
        stem = f"{args.model}__{tid}__{args.pooling}__by_patient.parquet"
        return args.emb_dir / (f"{prefix}{stem}" if prefix else stem)

    # --- load MSK (train+val) + MSK embeddings, FIT FROZEN MODELS ONCE -------
    msk = S.load_table(args.table)
    splits = S.load_splits(args.splits)
    trainval_ids = [*splits["train"], *splits["val"]]
    trainval = msk.set_index(S.ID_COLUMN).loc[
        [i for i in trainval_ids if i in set(msk[S.ID_COLUMN])]
    ].reset_index()

    msk_emb_path = emb_file("")
    if not msk_emb_path.exists():
        raise SystemExit(
            f"missing MSK embeddings: {msk_emb_path}\n"
            f"  place the Phase-1 parquet under {args.emb_dir}/"
        )
    msk_emb = pd.read_parquet(msk_emb_path).set_index(S.ID_COLUMN)
    emb_cols = A._emb_cols(msk_emb)

    print(f"MSK train+val: {len(trainval)} | emb dims: {len(emb_cols)} | "
          f"template: {tid} | cohorts: {', '.join(cohorts)}")

    # Fit featurizer + XGBoost-Cox for each feature set once; reuse across cohorts.
    frozen = []  # (name, use_emb, model, featurizer, msk_reference_median)
    for name, use_emb in [("tab", False), ("both", True)]:
        model, fz = _fit_frozen(trainval, msk_emb, emb_cols, data_cfg, bcfg, use_emb=use_emb)
        # MSK reference median: median risk on the (in-sample) MSK fit data — a
        # transferable high/low threshold to test against the intra-external one.
        Xtv = _matrix(trainval, fz, msk_emb, emb_cols, fit=False, use_emb=use_emb)
        msk_median = float(np.median(model.predict_risk(Xtv)))
        frozen.append((name, use_emb, model, fz, msk_median))

    # --- iterate over cohorts: load external table + embeddings, evaluate ----
    summary = []
    for cohort_name in cohorts:
        cohort = cohort_name.lower()
        ext_path = args.external_dir / f"genie_{cohort}.csv"
        ext_emb_path = emb_file(f"genie_{cohort}__")
        if not ext_path.exists():
            print(f"[{cohort_name}] SKIP: missing table {ext_path.relative_to(REPO_ROOT)}")
            continue
        if not ext_emb_path.exists():
            print(f"[{cohort_name}] SKIP: missing embeddings {ext_emb_path.name} "
                  f"(run the Colab extractor)")
            continue
        ext = S.load_table(ext_path)
        ext_emb = pd.read_parquet(ext_emb_path).set_index(S.ID_COLUMN)
        # A patient embedded twice (identical prompt -> identical vector) would make
        # emb.loc[ids] fan out and misalign with the feature matrix; keep one.
        ext_emb = ext_emb[~ext_emb.index.duplicated(keep="first")]
        assert set(emb_cols) <= set(ext_emb.columns), \
            f"{cohort_name}: external embeddings missing e-columns"
        print(f"[{cohort_name}] external n={len(ext)}")
        res = run_cohort(cohort_name, frozen, ext, ext_emb, emb_cols, data_cfg, args,
                         tid, args.out, args.out_patients)
        summary.append((cohort_name, res["feature_sets"]["tab"]["concordance"],
                        res["feature_sets"]["both"]["concordance"]))

    if not summary:
        raise SystemExit("no cohort produced results (missing tables/embeddings)")
    print("\nSummary (external C-index):")
    print(f"  {'cohort':10s} {'tab':>8s} {'both':>8s} {'Δ':>8s}")
    for name, c_tab, c_both in summary:
        print(f"  {name:10s} {c_tab:8.4f} {c_both:8.4f} {c_both - c_tab:+8.4f}")


if __name__ == "__main__":
    main()
