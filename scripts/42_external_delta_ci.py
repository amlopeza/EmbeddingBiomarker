"""Phase 3 add-on: paired-bootstrap 95% CI for the EXTERNAL embedding contribution.

The external validation (scripts/41_external_validate.py) reports the contribution
delta = C(both) - C(tab) per tumor as a POINT estimate only (results/external_genie_*.json,
Table 3 / Fig. 3b). This script attaches an uncertainty interval to each external delta
using the SAME paired bootstrap as the internal complementarity CI
(survival.concordance_delta_ci), so internal and external deltas are directly comparable.

No refit, no GPU: it reuses the cached per-patient predictions in
data/processed/external_<tumor>_patients.parquet, which already hold risk_tab and risk_both.

Usage (from repo root):
  python scripts/42_external_delta_ci.py
Writes results/external_delta_ci.json and prints a report table.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd
from sksurv.util import Surv

from embedbiomarker import survival as S

REPO = pathlib.Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "results" / "external_delta_ci.json"

# tumor key -> (parquet stem, display label)
TUMORS = [
    ("panc", "Pancreas"),
    ("prostate", "Prostate"),
    ("crc", "Colorectal"),
    ("brca", "Breast"),
    ("nsclc", "NSCLC"),
]

N_BOOT = 1000
SEED = 42


def main() -> None:
    results: dict[str, dict] = {}
    for key, label in TUMORS:
        df = pd.read_parquet(PROC / f"external_{key}_patients.parquet")
        target = Surv.from_arrays(
            event=df["event"].astype(bool).to_numpy(),
            time=df["os_months"].astype(float).to_numpy(),
        )
        risk_tab = df["risk_tab"].to_numpy()
        risk_both = df["risk_both"].to_numpy()
        d = S.concordance_delta_ci(target, risk_tab, risk_both, n_boot=N_BOOT, seed=SEED)
        results[key] = {
            "label": label,
            "n": int(len(df)),
            "events": int(df["event"].astype(bool).sum()),
            "c_tab": S.concordance(target, risk_tab),
            "c_both": S.concordance(target, risk_both),
            "delta": d["point"],
            "ci_low": d["ci_low"],
            "ci_high": d["ci_high"],
            "p_gt0": d["p_gt0"],
            "n_boot": d["n_boot"],
        }

    meta = {"n_boot": N_BOOT, "seed": SEED, "method": "paired percentile bootstrap (both - tab)"}
    OUT.write_text(json.dumps({"meta": meta, "per_cancer": results}, indent=2))

    # report
    print(f"\nExternal embedding contribution  delta = C(both) - C(tab)   "
          f"[{N_BOOT}-resample paired bootstrap, seed={SEED}]\n")
    print(f"{'Tumor':<12}{'n':>5}{'ev':>6}{'C_tab':>9}{'C_both':>9}"
          f"{'delta':>10}{'95% CI':>22}{'P(>0)':>8}")
    for key, _ in TUMORS:
        r = results[key]
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"{r['label']:<12}{r['n']:>5}{r['events']:>6}{r['c_tab']:>9.4f}"
              f"{r['c_both']:>9.4f}{r['delta']:>+10.4f}{ci:>22}{r['p_gt0']:>8.2f}")
    print(f"\nJSON -> {OUT}")


if __name__ == "__main__":
    main()
