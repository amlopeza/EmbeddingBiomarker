#!/usr/bin/env python
"""Phase 1 — interpretability of the pancreas complementarity.

MedGemma embeddings add a real, significant edge over the 12 tabular features
ONLY in pancreas (+0.0386, CI [+0.0200, +0.0583]). This probe asks WHY: what does
the text prompt carry that the multi-hot tabular encoding throws away?

The multi-hot encoder loses four things the embedder keeps:
  1. rare tokens beyond the global top-K vocab (mutations>50, treatments>30),
  2. treatment ORDER (sequence is flattened to presence),
  3. treatment REPETITION (a drug given twice is still presence=1),
  4. semantics / co-occurrence (only the LM represents these).

Three falsifiable analyses, per cancer, on the frozen test split:
  A. out-of-vocab burden   -> tests (1): is pancreas systematically worse covered?
  B. order/repetition stats -> tests (2,3): does pancreas have richer sequences?
  C. enriched-tab refit     -> tests (1-3) jointly: do hand-engineered count/OOV
     features CLOSE the pancreas gap? If yes, the mechanism is countable structure;
     if a gap survives, MedGemma captures semantics no count reproduces.

Usage:
    python scripts/24_pancreas_interpret.py --model medgemma15 --pooling mean
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embedbiomarker import survival as S
from embedbiomarker.baselines import build_model

CANCER_COL = "CANCER_TYPE"
MUT_COL = "MUTATIONS"
TRT_COL = "TREATMENT_HISTORY"


def _tokens(cell, *, unique: bool) -> list[str]:
    toks = S._tokenize(cell)
    return sorted(set(toks)) if unique else toks


def _coverage_stats(frame: pd.DataFrame, col: str, vocab: set[str], *, unique: bool) -> dict:
    """Per-patient out-of-vocab burden for a text_list column, averaged over the group."""
    n_total, n_oov, has_oov, has_repeat = [], [], [], []
    for cell in frame[col]:
        toks = _tokens(cell, unique=unique)
        oov = [t for t in toks if t not in vocab]
        n_total.append(len(toks))
        n_oov.append(len(oov))
        has_oov.append(len(oov) > 0)
        if not unique:
            raw = S._tokenize(cell)
            has_repeat.append(len(raw) > len(set(raw)))
    out = {
        "mean_tokens_per_patient": float(np.mean(n_total)),
        "mean_oov_per_patient": float(np.mean(n_oov)),
        "frac_tokens_oov": float(np.sum(n_oov) / max(np.sum(n_total), 1)),
        "pct_patients_with_oov": float(np.mean(has_oov)),
    }
    if has_repeat:
        out["pct_patients_with_repeat"] = float(np.mean(has_repeat))
    return out


def _engineer(frame: pd.DataFrame, mut_vocab: set[str], trt_vocab: set[str]) -> pd.DataFrame:
    """Hand-crafted features that capture what the multi-hot loses (counts/OOV/repetition)."""
    rows = []
    for _, r in frame.iterrows():
        mut = S._tokenize(r[MUT_COL])
        trt = S._tokenize(r[TRT_COL])
        rows.append({
            "eng__n_mut": len(set(mut)),
            "eng__n_mut_oov": sum(1 for t in set(mut) if t not in mut_vocab),
            "eng__n_trt": len(trt),                      # with repeats == sequence length
            "eng__n_trt_unique": len(set(trt)),
            "eng__n_trt_repeats": len(trt) - len(set(trt)),
            "eng__n_trt_oov": sum(1 for t in trt if t not in trt_vocab),
        })
    return pd.DataFrame(rows, index=range(len(frame)))


def _per_cancer_delta(frame, risk_a, risk_b, data_cfg, n_boot) -> dict:
    out = {}
    for ct, idx in frame.groupby(CANCER_COL).groups.items():
        pos = frame.index.get_indexer(idx)
        suby = S.make_target(frame.loc[idx], data_cfg)
        if int(suby["event"].sum()) >= 5:
            out[str(ct)] = S.concordance_delta_ci(suby, risk_a[pos], risk_b[pos], n_boot=n_boot)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="medgemma15")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--template-id", default=None)
    ap.add_argument("--table", type=Path, default=REPO_ROOT / "data/interim/data_prompts.csv")
    ap.add_argument("--prompts", type=Path, default=REPO_ROOT / "data/interim/prompts.parquet")
    ap.add_argument("--emb-dir", type=Path, default=REPO_ROOT / "data/processed/embeddings")
    ap.add_argument("--data-config", type=Path, default=REPO_ROOT / "config/data.yaml")
    ap.add_argument("--survival-config", type=Path, default=REPO_ROOT / "config/survival.yaml")
    ap.add_argument("--splits", type=Path, default=REPO_ROOT / "data/interim/splits.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/interpret_pancreas.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    data_cfg = S.load_config(args.data_config)
    surv_cfg = S.load_config(args.survival_config)
    bcfg = surv_cfg["baseline"]

    template_id = args.template_id or str(pd.read_parquet(args.prompts)["template_id"].iloc[0])
    emb_path = args.emb_dir / f"{args.model}__{template_id}__{args.pooling}__by_patient.parquet"

    df = S.load_table(args.table)
    splits = S.load_splits(args.splits)
    frames = S.split_frames(df, splits, id_column=S.ID_COLUMN)
    emb = pd.read_parquet(emb_path).set_index(S.ID_COLUMN)
    emb_cols = [c for c in emb.columns if c.startswith("e")]

    # Fit the tabular featurizer once -> its learned top-K vocab is the "what the
    # tabular model can see" boundary. Anything outside it lives only in the text.
    fz = S.TabularFeaturizer(data_cfg, bcfg)
    fz.fit(frames["train"])
    mut_vocab = set(fz._vocab.get(MUT_COL, []))
    trt_vocab = set(fz._vocab.get(TRT_COL, []))
    print(f"tabular vocab: mutations top-{len(mut_vocab)}, treatments top-{len(trt_vocab)}")

    test = frames["test"]
    cancers = sorted(test[CANCER_COL].unique())

    # --- A & B: coverage + order/repetition per cancer -------------------
    coverage = {}
    for ct in cancers:
        sub = test[test[CANCER_COL] == ct]
        coverage[ct] = {
            "n": int(len(sub)),
            "mutations": _coverage_stats(sub, MUT_COL, mut_vocab, unique=True),
            "treatments": _coverage_stats(sub, TRT_COL, trt_vocab, unique=False),
        }

    print("\n=== A/B. Out-of-vocab burden & treatment structure (test) ===")
    print(f"{'cancer':28s} {'mut_oov%':>9s} {'trt_oov%':>9s} {'trt_len':>8s} "
          f"{'repeats%':>9s} {'mut/pt':>7s}")
    for ct in cancers:
        m, t = coverage[ct]["mutations"], coverage[ct]["treatments"]
        print(f"{ct:28s} {m['frac_tokens_oov']*100:8.1f}% {t['frac_tokens_oov']*100:8.1f}% "
              f"{t['mean_tokens_per_patient']:8.2f} {t.get('pct_patients_with_repeat',0)*100:8.1f}% "
              f"{m['mean_tokens_per_patient']:7.2f}")

    # --- C: enriched-tab falsification -----------------------------------
    y = {p: S.make_target(frames[p], data_cfg) for p in ("train", "val", "test")}
    X_tab, X_enr, X_both = {}, {}, {}
    for p in ("train", "val", "test"):
        ids = frames[p][S.ID_COLUMN]
        Xt = (fz.fit_transform(frames[p]) if p == "train" else fz.transform(frames[p])).reset_index(drop=True)
        eng = _engineer(frames[p], mut_vocab, trt_vocab)
        Xe = emb.loc[ids, emb_cols].reset_index(drop=True)
        X_tab[p] = Xt
        X_enr[p] = pd.concat([Xt, eng], axis=1)
        X_both[p] = pd.concat([Xt, Xe], axis=1)

    risk = {}
    for name, X in (("tab", X_tab), ("enriched", X_enr), ("both", X_both)):
        model = build_model("xgboost_cox", bcfg).fit(X["train"], y["train"])
        risk[name] = model.predict_risk(X["test"])

    d_enr = _per_cancer_delta(test, risk["tab"], risk["enriched"], data_cfg, args.n_boot)
    d_both = _per_cancer_delta(test, risk["tab"], risk["both"], data_cfg, args.n_boot)

    print("\n=== C. Does enriched-tab (counts/OOV/repetition) close the gap? ===")
    print(f"{'cancer':28s} {'enriched-tab':>22s} {'both-tab (MedGemma)':>24s} {'closed':>7s}")
    for ct in cancers:
        if ct not in d_both:
            continue
        de, db = d_enr.get(ct), d_both[ct]
        frac = (de["point"] / db["point"] * 100) if de and abs(db["point"]) > 1e-9 else float("nan")
        es = f"{de['point']:+.4f} [{de['ci_low']:+.3f},{de['ci_high']:+.3f}]" if de else "  n/a"
        bs = f"{db['point']:+.4f} [{db['ci_low']:+.3f},{db['ci_high']:+.3f}]"
        print(f"{ct:28s} {es:>22s} {bs:>24s} {frac:6.0f}%")

    payload = {
        "model": args.model, "pooling": args.pooling, "template_id": template_id,
        "tabular_vocab": {"mutations_top_k": len(mut_vocab), "treatments_top_k": len(trt_vocab)},
        "coverage_per_cancer": coverage,
        "gap_closure_per_cancer": {
            ct: {"enriched_minus_tab": d_enr.get(ct), "both_minus_tab": d_both.get(ct)}
            for ct in cancers if ct in d_both
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
