#!/usr/bin/env python
"""Phase 1 — XGBoost-Cox over embeddings, tabular, and their concatenation.

The de-risking probe: does an embedding head beat the tabular baseline (~0.755),
and — more importantly — do embeddings ADD signal on top of the 12 tabular features
(complementarity)? Reuses the SAME frozen split, target and ``concordance`` as the
Phase 0 baseline so numbers compare directly.

Three feature sets, each fed to the same XGBoost-Cox config:
    emb   — embeddings only
    tab   — the 100 tabular features (Phase 0)
    both  — emb ⊕ tab (concatenated)

Usage:
    python scripts/22_cox_grid.py                    # default model medcpt
    python scripts/22_cox_grid.py --model medcpt --pooling mean
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embedbiomarker import survival as S
from embedbiomarker.baselines import build_model

CANCER_COL = "CANCER_TYPE"


def _per_cancer(frame: pd.DataFrame, risk, data_cfg) -> dict:
    out = {}
    for ct, idx in frame.groupby(CANCER_COL).groups.items():
        pos = frame.index.get_indexer(idx)
        suby = S.make_target(frame.loc[idx], data_cfg)
        if int(suby["event"].sum()) >= 5:
            out[str(ct)] = S.concordance(suby, risk[pos])
    return out


def _per_cancer_delta_ci(frame: pd.DataFrame, risk_a, risk_b, data_cfg, n_boot: int) -> dict:
    """Paired bootstrap CI of the C-index delta (b - a) within each cancer stratum.

    Same patients resampled for both scores per replicate (paired), restricted to
    one tumour at a time. Tells whether the embedding's edge over tabular is real
    where it matters most (e.g. pancreas) or just small-n noise. Skips strata with
    < 5 events, as ``_per_cancer`` does.
    """
    out = {}
    for ct, idx in frame.groupby(CANCER_COL).groups.items():
        pos = frame.index.get_indexer(idx)
        suby = S.make_target(frame.loc[idx], data_cfg)
        if int(suby["event"].sum()) >= 5:
            out[str(ct)] = S.concordance_delta_ci(
                suby, risk_a[pos], risk_b[pos], n_boot=n_boot)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medcpt")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--template-id", default=None, help="defaults to the prompts' template")
    parser.add_argument("--table", type=Path, default=REPO_ROOT / "data/interim/data_prompts.csv")
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "data/interim/prompts.parquet")
    parser.add_argument("--emb-dir", type=Path, default=REPO_ROOT / "data/processed/embeddings")
    parser.add_argument("--data-config", type=Path, default=REPO_ROOT / "config/data.yaml")
    parser.add_argument("--survival-config", type=Path, default=REPO_ROOT / "config/survival.yaml")
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "data/interim/splits.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="paired bootstrap replicates for the test-set C-index CIs (0 = skip)")
    args = parser.parse_args()

    data_cfg = S.load_config(args.data_config)
    surv_cfg = S.load_config(args.survival_config)
    bcfg = surv_cfg["baseline"]

    template_id = args.template_id or str(pd.read_parquet(args.prompts)["template_id"].iloc[0])
    emb_path = args.emb_dir / f"{args.model}__{template_id}__{args.pooling}__by_patient.parquet"
    if not emb_path.exists():
        raise SystemExit(f"embeddings not found: {emb_path}\nRun 20_extract_embeddings.py first.")

    # --- load + split ----------------------------------------------------
    df = S.load_table(args.table)
    splits = S.load_splits(args.splits)
    frames = S.split_frames(df, splits, id_column=S.ID_COLUMN)
    emb = pd.read_parquet(emb_path).set_index(S.ID_COLUMN)
    emb_cols = [c for c in emb.columns if c.startswith("e")]
    print(f"Model {args.model} | emb dim={len(emb_cols)} | template={template_id} pooling={args.pooling}")

    # --- build the 3 feature sets per split (aligned to frame order) -----
    fz = S.TabularFeaturizer(data_cfg, bcfg)
    y = {p: S.make_target(frames[p], data_cfg) for p in ("train", "val", "test")}
    X_tab, X_emb, X_both = {}, {}, {}
    for p in ("train", "val", "test"):
        ids = frames[p][S.ID_COLUMN]
        X_tab[p] = (fz.fit_transform(frames[p]) if p == "train" else fz.transform(frames[p])).reset_index(drop=True)
        X_emb[p] = emb.loc[ids, emb_cols].reset_index(drop=True)
        X_both[p] = pd.concat([X_tab[p], X_emb[p]], axis=1)

    feature_sets = {"emb": X_emb, "tab": X_tab, "both": X_both}

    # --- fit + score -----------------------------------------------------
    results = {}
    risk_test = {}  # per feature set, for the paired bootstrap below
    for name, X in feature_sets.items():
        t0 = time.time()
        model = build_model("xgboost_cox", bcfg).fit(X["train"], y["train"])
        scores = {p: S.concordance(y[p], model.predict_risk(X[p])) for p in ("val", "test")}
        risk_test[name] = model.predict_risk(X["test"])
        results[name] = {
            "n_features": X["train"].shape[1],
            "c_index": scores,
            "c_index_per_cancer_test": _per_cancer(frames["test"], risk_test[name], data_cfg),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"  {name:5s} ({X['train'].shape[1]:4d} feat)  val={scores['val']:.4f}  test={scores['test']:.4f}")

    # --- bootstrap CIs (test set resampled, models fixed) ----------------
    deltas = {}
    delta_per_cancer = {}
    if args.n_boot > 0:
        for name in feature_sets:
            results[name]["c_index_ci_test"] = S.concordance_ci(
                y["test"], risk_test[name], n_boot=args.n_boot)
        # paired deltas vs the tabular bar: does the embedding add signal?
        for name in ("emb", "both"):
            deltas[f"{name}_minus_tab"] = S.concordance_delta_ci(
                y["test"], risk_test["tab"], risk_test[name], n_boot=args.n_boot)
            # ... and per tumour: is the edge real where tabular is weakest?
            delta_per_cancer[f"{name}_minus_tab"] = _per_cancer_delta_ci(
                frames["test"], risk_test["tab"], risk_test[name], data_cfg, args.n_boot)

    # --- persist + verdict ----------------------------------------------
    payload = {
        "model": args.model, "template_id": template_id, "pooling": args.pooling,
        "emb_dim": len(emb_cols),
        "n_patients": {p: len(frames[p]) for p in ("train", "val", "test")},
        "feature_sets": results,
        "delta_vs_tab_test": deltas,
        "delta_per_cancer_test": delta_per_cancer,
    }
    out = (args.out or (REPO_ROOT / f"results/embedding_grid__{args.model}.json")).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        shown = out
    print(f"\nWrote {shown}")

    tab_t, emb_t, both_t = (results[k]["c_index"]["test"] for k in ("tab", "emb", "both"))
    print(f"\nTest C-index — tab={tab_t:.4f}  emb={emb_t:.4f}  both={both_t:.4f}")
    if args.n_boot > 0:
        for k in ("tab", "emb", "both"):
            ci = results[k]["c_index_ci_test"]
            print(f"  {k:5s} {ci['point']:.4f}  [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  ({int(ci['ci_level']*100)}% CI)")
        for name, label in (("emb_minus_tab", "embeddings vs tabular"),
                            ("both_minus_tab", "complementarity (both-tab)")):
            d = deltas[name]
            if d["ci_low"] > 0:
                sig = "SIGNIFICANT better (CI > 0)"
            elif d["ci_high"] < 0:
                sig = "SIGNIFICANT worse (CI < 0)"
            else:
                sig = "not significant (CI includes 0)"
            print(f"  {label}: {d['point']:+.4f}  [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
                  f"p(>0)={d['p_gt0']:.3f}  -> {sig}")
        print("\n  complementarity (both-tab) per cancer:")
        for ct, d in sorted(delta_per_cancer["both_minus_tab"].items()):
            flag = "*" if d["ci_low"] > 0 else (" " if d["ci_high"] > 0 else "-")
            print(f"  {flag} {ct:30s} {d['point']:+.4f}  [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
                  f"p(>0)={d['p_gt0']:.3f}  (n_boot={d['n_boot']})")
    else:
        print(f"  embeddings vs tabular:   {emb_t - tab_t:+.4f}")
        print(f"  complementarity (both-tab): {both_t - tab_t:+.4f}")


if __name__ == "__main__":
    main()
