#!/usr/bin/env python
"""Phase 0 tabular baseline — the line every embedding head must beat.

Encodes the 12 raw context features (config/data.yaml) into a numeric matrix
(fit on TRAIN only), fits Cox-PH / RSF / XGBoost-Cox, and reports Harrell's
C-index on val and test — pan-cancer and per cancer type. Writes
``results/baseline_tabular.json`` (versioned) plus a printed summary table.

Usage:
    python scripts/10_baseline_tabular.py
    python scripts/10_baseline_tabular.py --models coxph xgboost_cox
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run without an editable install

from embedbiomarker import survival as S
from embedbiomarker.baselines import build_model

CANCER_COL = "CANCER_TYPE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=REPO_ROOT / "data/interim/data_prompts.csv")
    parser.add_argument("--data-config", type=Path, default=REPO_ROOT / "config/data.yaml")
    parser.add_argument("--survival-config", type=Path, default=REPO_ROOT / "config/survival.yaml")
    parser.add_argument("--splits", type=Path, default=REPO_ROOT / "data/interim/splits.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results/baseline_tabular.json")
    parser.add_argument("--models", nargs="+", default=None, help="subset of baseline models to run")
    args = parser.parse_args()

    data_cfg = S.load_config(args.data_config)
    surv_cfg = S.load_config(args.survival_config)
    baseline_cfg = surv_cfg["baseline"]
    model_names = args.models or baseline_cfg["models"]

    # --- load + split -----------------------------------------------------
    df = S.load_table(args.table)
    splits = S.load_splits(args.splits)
    frames = S.split_frames(df, splits, id_column=S.ID_COLUMN)
    print(f"Loaded {len(df)} patients — "
          + ", ".join(f"{p}={len(frames[p])}" for p in ("train", "val", "test")))

    # --- featurize (fit on TRAIN only) ------------------------------------
    fz = S.TabularFeaturizer(data_cfg, baseline_cfg)
    X = {part: fz.fit_transform(frames[part]) if part == "train" else fz.transform(frames[part])
         for part in ("train", "val", "test")}
    y = {part: S.make_target(frames[part], data_cfg) for part in ("train", "val", "test")}
    print(f"Featurized {len(fz.columns_)} columns from the 12 context features "
          f"(target held out: {sorted(S.target_columns(data_cfg))}).")

    # --- fit + score ------------------------------------------------------
    results: dict[str, dict] = {}
    for name in model_names:
        t0 = time.time()
        model = build_model(name, baseline_cfg)
        model.fit(X["train"], y["train"])

        scores = {part: S.concordance(y[part], model.predict_risk(X[part]))
                  for part in ("val", "test")}

        # Per-cancer test C-index (treatment-aware context for later phases).
        per_cancer: dict[str, float] = {}
        test_df = frames["test"]
        risk_test = model.predict_risk(X["test"])
        for ct, idx in test_df.groupby(CANCER_COL).groups.items():
            pos = test_df.index.get_indexer(idx)
            sub_y = S.make_target(test_df.loc[idx], data_cfg)
            if sub_y["event"].sum() >= 5:  # need events to compute concordance
                per_cancer[str(ct)] = S.concordance(sub_y, risk_test[pos])

        results[name] = {
            "c_index": scores,
            "c_index_per_cancer_test": per_cancer,
            "fit_seconds": round(time.time() - t0, 1),
        }
        print(f"  {name:12s} val={scores['val']:.4f}  test={scores['test']:.4f}  "
              f"({results[name]['fit_seconds']}s)")

    # --- persist ----------------------------------------------------------
    payload = {
        "n_features": len(fz.columns_),
        "n_patients": {p: len(frames[p]) for p in ("train", "val", "test")},
        "metric": surv_cfg.get("metrics", {}).get("primary", "c_index"),
        "models": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}")

    # --- summary ----------------------------------------------------------
    best = max(results.items(), key=lambda kv: kv[1]["c_index"]["test"])
    print(f"\nPan-cancer test C-index (line to beat):")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["c_index"]["test"]):
        star = "  <-- best" if name == best[0] else ""
        print(f"  {name:12s} {r['c_index']['test']:.4f}{star}")


if __name__ == "__main__":
    main()
