#!/usr/bin/env python
"""Phase 2 — treatment-aware survival analysis (the paper's headline).

Builds an out-of-fold LM risk score for every patient (both a pan-cancer and a
cancer-specific model), then, per tumor and per treatment stratum, asks whether
the score is prognostic and/or predictive:

  * C-index of the risk score, overall and WITHIN each treatment stratum;
  * Kaplan-Meier high/low-risk split (per-tumor median cutpoint) + log-rank p;
  * stage/age-adjusted HR of the risk score (per SD);
  * risk x treatment interaction term (effect modification: prognostic vs predictive).

Pancreas is stratified by regimen (FOLFIRINOX-like / gemcitabine-based / ...),
the other four tumors by drug-class flags (config/treatments.yaml).

Usage:
    python scripts/30_treatment_analysis.py --model medgemma15 --pooling mean
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embedbiomarker import analysis as A
from embedbiomarker import survival as S

MIN_N = 20      # minimum patients in a stratum to compute within-stratum stats
MIN_EVENTS = 5  # minimum events in a stratum to compute within-stratum stats


def _clean(obj):
    """Recursively make a payload JSON-safe (numpy scalars, non-finite floats)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _cindex(target: np.ndarray, risk: np.ndarray):
    try:
        return float(S.concordance(target, risk))
    except (ValueError, ZeroDivisionError):
        return None


def _analyze_group(frame: pd.DataFrame, risk: np.ndarray, data_cfg: dict) -> dict:
    """Within-group prognostic summary: C-index, KM high/low, adjusted HR."""
    target = S.make_target(frame, data_cfg)
    out = {
        "c_index": _cindex(target, risk),
        "km_high_low": A.km_logrank(
            target["time"], target["event"], A.risk_high_low(risk)),
        "adjusted_hr": A.cox_adjusted_hr(frame, risk, data_cfg),
    }
    return out


def _strata_pairs(mode: str, strata):
    """Normalize regimen/classes strata into [(name, boolean mask), ...]."""
    if mode == "regimen":
        return [(g, (strata.to_numpy() == g)) for g in sorted(strata.unique())]
    return [(c, strata[c].to_numpy(dtype=bool)) for c in strata.columns]


def _analyze_cancer(frame: pd.DataFrame, risks: dict, tax_entry: dict,
                    data_cfg: dict) -> dict:
    """All treatment-aware stats for one tumor, for every risk mode in ``risks``."""
    mode, strata = A.treatment_strata(frame, tax_entry)
    target = S.make_target(frame, data_cfg)
    pairs = _strata_pairs(mode, strata)

    res = {
        "n": int(len(frame)),
        "events": int(target["event"].sum()),
        "taxonomy_mode": mode,
        "by_risk_mode": {},
    }
    for rmode, risk_series in risks.items():
        risk = risk_series.loc[frame[S.ID_COLUMN].to_numpy()].to_numpy()
        rmode_out = {
            "global": _analyze_group(frame, risk, data_cfg),
            "strata": {},
        }
        for name, mask in pairs:
            n, ev = int(mask.sum()), int(target["event"][mask].sum())
            entry = {"n": n, "events": ev}
            if n >= MIN_N and ev >= MIN_EVENTS:
                entry.update(_analyze_group(frame[mask].reset_index(drop=True),
                                            risk[mask], data_cfg))
                # interaction is fit on the WHOLE tumor: arm/class vs the rest.
                entry["interaction"] = A.cox_interaction(
                    frame, risk, mask.astype(float), data_cfg)
            else:
                entry["skipped"] = "insufficient n/events"
            rmode_out["strata"][str(name)] = entry
        res["by_risk_mode"][rmode] = rmode_out
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="medgemma15")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--template-id", default=None)
    ap.add_argument("--table", type=Path, default=REPO_ROOT / "data/interim/data_prompts.csv")
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "data/interim/prompts.parquet")
    ap.add_argument("--emb-dir", type=Path, default=REPO_ROOT / "data/processed/embeddings")
    ap.add_argument("--treatments", type=Path, default=REPO_ROOT / "config/treatments.yaml")
    ap.add_argument("--data-config", type=Path, default=REPO_ROOT / "config/data.yaml")
    ap.add_argument("--survival-config", type=Path, default=REPO_ROOT / "config/survival.yaml")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/treatment_analysis.json")
    ap.add_argument("--out-patients", type=Path,
                    default=REPO_ROOT / "data/processed/treatment_oof_patients.parquet",
                    help="per-patient OOF risk + survival (gitignored) for KM curves")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--risk-modes", nargs="+", default=["pan", "specific"])
    ap.add_argument("--reuse-risk", action="store_true",
                    help="load OOF risks from --out-patients instead of cross-fitting "
                         "(cross-fit is deterministic; use to re-run downstream stats fast)")
    args = ap.parse_args()

    data_cfg = S.load_config(args.data_config)
    surv_cfg = S.load_config(args.survival_config)
    bcfg = surv_cfg["baseline"]
    seed = int(surv_cfg.get("seed", 42))

    template_id = args.template_id or str(pd.read_parquet(args.prompts)["template_id"].iloc[0])
    emb_path = args.emb_dir / f"{args.model}__{template_id}__{args.pooling}__by_patient.parquet"

    df = S.load_table(args.table)
    emb = pd.read_parquet(emb_path).set_index(S.ID_COLUMN)
    taxonomy = A.load_treatment_taxonomy(args.treatments)
    print(f"patients: {len(df)} | emb dims: {len(A._emb_cols(emb))} | template: {template_id}")

    # --- out-of-fold risk scores (both models) ---------------------------
    risks = {}
    if args.reuse_risk and args.out_patients.exists():
        cached = pd.read_parquet(args.out_patients).set_index(S.ID_COLUMN)
        for rmode in args.risk_modes:
            risks[rmode] = cached[f"risk_{rmode}"]
        print(f"  reused OOF risks from {args.out_patients.relative_to(REPO_ROOT)}")
    else:
        for rmode in args.risk_modes:
            t0 = _time.time()
            risks[rmode] = A.crossfit_risk(
                df, emb, data_cfg, bcfg, mode=rmode, n_splits=args.n_splits, seed=seed)
            gc = _cindex(S.make_target(df, data_cfg),
                         risks[rmode].loc[df[S.ID_COLUMN].to_numpy()].to_numpy())
            print(f"  risk[{rmode}]: OOF pan-cohort C-index={gc:.4f}  ({_time.time()-t0:.0f}s)")

    # --- per-patient OOF table (gitignored) so the notebook can draw KM ---
    target_all = S.make_target(df, data_cfg)
    patients = df[[S.ID_COLUMN, A.CANCER_COL, A.TRT_COL]].copy()
    patients["os_months"] = target_all["time"]
    patients["event"] = target_all["event"].astype(int)
    for rmode, rs in risks.items():
        patients[f"risk_{rmode}"] = rs.loc[df[S.ID_COLUMN].to_numpy()].to_numpy()
    args.out_patients.parent.mkdir(parents=True, exist_ok=True)
    patients.to_parquet(args.out_patients, index=False)
    print(f"  wrote per-patient OOF table -> {args.out_patients.relative_to(REPO_ROOT)}")

    # --- per-cancer treatment-aware analysis -----------------------------
    per_cancer = {}
    for ct in sorted(df[A.CANCER_COL].unique()):
        if ct not in taxonomy:
            continue
        frame = df[df[A.CANCER_COL] == ct].reset_index(drop=True)
        per_cancer[ct] = _analyze_cancer(frame, risks, taxonomy[ct], data_cfg)
        print(f"  {ct:28s} n={per_cancer[ct]['n']:5d} "
              f"events={per_cancer[ct]['events']:5d} mode={per_cancer[ct]['taxonomy_mode']}")

    payload = {
        "model": args.model, "pooling": args.pooling, "template_id": template_id,
        "n_patients": int(len(df)),
        "risk_modes": list(args.risk_modes),
        "n_splits": int(args.n_splits),
        "min_n": MIN_N, "min_events": MIN_EVENTS,
        "caveat": ("Stratified prognostic validation + an EXPLORATORY effect-modification "
                   "probe, not a causal treatment-efficacy claim. Interaction caveats: "
                   "(1) TREATMENT_HISTORY is post-baseline -> immortal-time bias; "
                   "(2) evaluated CASE BY CASE per tumor (separate populations / "
                   "treatments), not one joint family -> no global multiple-testing "
                   "correction; cancer-specific score is the primary lens, pan-cancer a "
                   "robustness check; (3) circularity -- the risk score is trained on "
                   "features that include treatment history, so a risk x treatment "
                   "interaction is partly mechanical. Adjusted HR/CI/p use an unpenalized "
                   "Cox (penalizer=0.0) for valid inference."),
        "per_cancer": per_cancer,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_clean(payload), indent=2))
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
