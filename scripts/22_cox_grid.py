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
    for name, X in feature_sets.items():
        t0 = time.time()
        model = build_model("xgboost_cox", bcfg).fit(X["train"], y["train"])
        scores = {p: S.concordance(y[p], model.predict_risk(X[p])) for p in ("val", "test")}
        risk_test = model.predict_risk(X["test"])
        results[name] = {
            "n_features": X["train"].shape[1],
            "c_index": scores,
            "c_index_per_cancer_test": _per_cancer(frames["test"], risk_test, data_cfg),
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"  {name:5s} ({X['train'].shape[1]:4d} feat)  val={scores['val']:.4f}  test={scores['test']:.4f}")

    # --- persist + verdict ----------------------------------------------
    payload = {
        "model": args.model, "template_id": template_id, "pooling": args.pooling,
        "emb_dim": len(emb_cols),
        "n_patients": {p: len(frames[p]) for p in ("train", "val", "test")},
        "feature_sets": results,
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
    print(f"  embeddings vs tabular:   {emb_t - tab_t:+.4f}")
    print(f"  complementarity (both-tab): {both_t - tab_t:+.4f}")


if __name__ == "__main__":
    main()
