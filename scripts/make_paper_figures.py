"""Regenerate the paper figures in a Nature-Communications visual style.

Reads only versioned analysis outputs (results/*.json) and the gitignored
out-of-fold per-patient table (data/processed/treatment_oof_patients.parquet);
recomputes no statistics. Writes three multi-panel PNGs into paper/figures/.

Style: Liberation Sans (Arial-metric compatible), Wong colour-blind-safe
palette, panel labels a/b/..., no in-image titles (captions live in the paper).

Run:  .venv_fig/bin/python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from embedbiomarker import analysis as A  # noqa: E402

RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# House style
# --------------------------------------------------------------------------- #
# Register Liberation Sans (Arial-metric compatible) so figures read as Nature.
for fp in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"):
    fm.fontManager.addfont(str(fp))
SANS = "Liberation Sans" if any(
    f.name == "Liberation Sans" for f in fm.fontManager.ttflist
) else "DejaVu Sans"

plt.rcParams.update({
    "font.family": SANS,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Wong (2011) colour-blind-safe palette.
C_TAB = "#999999"   # grey   -- tabular / baseline reference
C_EMB = "#0072B2"   # blue   -- language-model embedding
C_LOW = "#0072B2"   # blue   -- low risk
C_HIGH = "#D55E00"  # vermillion -- high risk
C_POS = "#009E73"   # green  -- positive contribution
C_NEG = "#D55E00"   # vermillion -- negative contribution
C_INK = "#222222"

SHORT = {
    "Pancreatic Cancer": "Pancreas",
    "Breast Cancer": "Breast",
    "Colorectal Cancer": "Colorectal",
    "Non-Small Cell Lung Cancer": "NSCLC",
    "Prostate Cancer": "Prostate",
}


def panel_label(ax, letter, dx=-0.085, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="right")


def errbars(d):
    """asymmetric xerr from a {point, ci_low, ci_high} dict."""
    return [[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]]


# --------------------------------------------------------------------------- #
# Figure 1 -- semantic mechanism (a: gap-closure forest, b: OOV refutation)
# --------------------------------------------------------------------------- #
def figure1():
    data = json.loads((RESULTS / "interpret_pancreas.json").read_text())
    gap = data["gap_closure_per_cancer"]
    cov = data["coverage_per_cancer"]
    order = sorted(gap, key=lambda c: gap[c]["both_minus_tab"]["point"], reverse=True)
    y = np.arange(len(order))[::-1]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- panel a: dumbbell forest ----------------------------------------- #
    OFF = 0.17
    for yi, c in zip(y, order):
        de, db = gap[c]["enriched_minus_tab"], gap[c]["both_minus_tab"]
        axA.plot([de["point"], db["point"]], [yi + OFF, yi - OFF],
                 color="#d6d6d6", lw=1.0, zorder=1)
        axA.errorbar(de["point"], yi + OFF, xerr=errbars(de), fmt="o",
                     color=C_TAB, ms=5, lw=1.4, capsize=2.2, zorder=3)
        axA.errorbar(db["point"], yi - OFF, xerr=errbars(db), fmt="o",
                     color=C_EMB, ms=5.5, lw=1.6, capsize=2.2, zorder=3)
    axA.axvline(0, color=C_INK, lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    axA.set_yticks(y)
    axA.set_yticklabels([SHORT[c] for c in order])
    axA.set_ylim(-0.6, len(order) - 0.4)
    axA.set_xlabel(r"$\Delta$ C-index over tabular baseline")
    handles = [
        plt.Line2D([0], [0], marker="o", color=C_TAB, lw=0, ms=5,
                   label="engineered counts"),
        plt.Line2D([0], [0], marker="o", color=C_EMB, lw=0, ms=5,
                   label="MedGemma embedding"),
    ]
    axA.legend(handles=handles, loc="lower right", handletextpad=0.4)
    panel_label(axA, "a", dx=-0.30)

    # --- panel b: OOV vs gain (refutation) -------------------------------- #
    for c in gap:
        x = cov[c]["mutations"]["frac_tokens_oov"] * 100
        yv = gap[c]["both_minus_tab"]["point"]
        is_p = c == "Pancreatic Cancer"
        axB.scatter(x, yv, s=110 if is_p else 55,
                    color=C_EMB if is_p else C_TAB,
                    edgecolor=C_INK, linewidth=0.6, zorder=3)
        axB.annotate(SHORT[c], (x, yv), textcoords="offset points",
                     xytext=(7, 5), fontsize=7,
                     fontweight="bold" if is_p else "normal",
                     color=C_EMB if is_p else C_INK)
    axB.axhline(0, color=C_INK, lw=0.8, ls=(0, (4, 3)), alpha=0.6)
    axB.set_xlabel("Mutation tokens out of tabular vocabulary (%)")
    axB.set_ylabel(r"MedGemma gain ($\Delta$ C-index)")
    axB.margins(x=0.18, y=0.22)
    panel_label(axB, "b", dx=-0.22)

    fig.tight_layout(w_pad=2.0)
    out = FIGDIR / "fig1_mechanism.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 2 -- pancreas KM within each regimen (4 panels)
# --------------------------------------------------------------------------- #
def figure2():
    data = json.loads((RESULTS / "treatment_analysis.json").read_text())
    pts = pd.read_parquet(ROOT / "data/processed/treatment_oof_patients.parquet")
    tax = A.load_treatment_taxonomy(ROOT / "config/treatments.yaml")
    PANC = "Pancreatic Cancer"
    REGIMENS = ["FOLFIRINOX-like", "gemcitabine-based", "other-treated", "untreated"]
    NICE = {
        "FOLFIRINOX-like": "FOLFIRINOX-like",
        "gemcitabine-based": "Gemcitabine-based",
        "other-treated": "Other treated",
        "untreated": "Untreated",
    }

    panc = pts[pts["CANCER_TYPE"] == PANC].reset_index(drop=True)
    _, regimen = A.treatment_strata(panc, tax[PANC])
    panc = panc.assign(regimen=regimen.to_numpy())
    cut = float(np.median(panc["risk_pan"]))
    panc = panc.assign(grp=np.where(panc["risk_pan"] > cut, "high", "low"))
    strata_json = data["per_cancer"][PANC]["by_risk_mode"]["pan"]["strata"]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharex=True, sharey=True)
    kmf = KaplanMeierFitter()
    for ax, arm, letter in zip(axes.ravel(), REGIMENS, "abcd"):
        sub = panc[panc["regimen"] == arm]
        for grp, color in (("low", C_LOW), ("high", C_HIGH)):
            g = sub[sub["grp"] == grp]
            if len(g) == 0:
                continue
            kmf.fit(g["os_months"], g["event"], label=f"{grp} risk (n={len(g)})")
            kmf.plot_survival_function(ax=ax, color=color, lw=1.6, ci_alpha=0.15)
        p = strata_json.get(arm, {}).get("km_high_low", {}).get("logrank_p")
        ptxt = (f"log-rank P = {p:.0e}".replace("e-0", "e-")
                if p else "log-rank n/a")
        ax.text(0.95, 0.95, ptxt, transform=ax.transAxes, ha="right",
                va="top", fontsize=7)
        ax.set_title(f"{NICE[arm]}  (n={len(sub)})", fontsize=8, pad=4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Months since first treatment timeline entry")
        ax.set_ylabel("Overall survival probability")
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86),
                  handlelength=1.4, handletextpad=0.5)
        panel_label(ax, letter, dx=-0.13, dy=1.10)

    fig.tight_layout(w_pad=1.6, h_pad=1.8)
    out = FIGDIR / "fig2_pancreas_km.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Figure 3 -- frozen-model external validation
#   a: external C-index (tab vs both) per tumor
#   b: embedding contribution, internal (CI) vs external
# --------------------------------------------------------------------------- #
def figure3():
    COHORTS = {
        "panc": "Pancreatic Cancer", "prostate": "Prostate Cancer",
        "crc": "Colorectal Cancer", "brca": "Breast Cancer",
        "nsclc": "Non-Small Cell Lung Cancer",
    }
    internal = json.loads((RESULTS / "embedding_grid__medgemma15.json").read_text())
    int_tab = internal["feature_sets"]["tab"]["c_index_per_cancer_test"]
    int_both = internal["feature_sets"]["both"]["c_index_per_cancer_test"]
    int_delta = internal["delta_per_cancer_test"]["both_minus_tab"]

    rows = []
    for stem, ct in COHORTS.items():
        ext = json.loads((RESULTS / f"external_genie_{stem}.json").read_text())
        ft = ext["feature_sets"]["tab"]
        fb = ext["feature_sets"]["both"]
        dl = int_delta[ct]
        rows.append({
            "tumor": SHORT[ct], "n": ext["external_n"],
            "f3_tab": ft["concordance"], "f3_both": fb["concordance"],
            "f1_d": int_both[ct] - int_tab[ct],
            "f1_lo": dl["ci_low"], "f1_hi": dl["ci_high"],
            "f3_d": fb["concordance"] - ft["concordance"],
        })
    df = pd.DataFrame(rows)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.5),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # --- panel a: external C-index forest --------------------------------- #
    da = df.sort_values("f3_both").reset_index(drop=True)
    y = np.arange(len(da))
    axA.hlines(y, da["f3_tab"], da["f3_both"], color="#d0d0d0", lw=2.0, zorder=1)
    axA.scatter(da["f3_tab"], y, s=55, color=C_TAB, zorder=2,
                label="12 features")
    axA.scatter(da["f3_both"], y, s=55, color=C_EMB, zorder=2,
                label="+ MedGemma")
    for yi, r in da.iterrows():
        d = r["f3_d"]
        axA.text(max(r["f3_tab"], r["f3_both"]) + 0.008, yi, f"{d:+.3f}",
                 va="center", fontsize=7,
                 color=C_POS if d > 0 else C_NEG)
    axA.axvline(0.5, ls=":", color=C_INK, lw=0.8)
    axA.set_yticks(y)
    axA.set_yticklabels([f"{t}\n(n={n})" for t, n in zip(da["tumor"], da["n"])])
    axA.set_xlim(0.5, 0.80)
    axA.set_xlabel("External C-index (GENIE BPC, non-MSK)")
    axA.set_ylim(-0.6, len(da) - 0.4)
    axA.legend(loc="upper left", handletextpad=0.4)
    panel_label(axA, "a", dx=-0.28)

    # --- panel b: contribution internal vs external ----------------------- #
    db = df.sort_values("f1_d").reset_index(drop=True)
    y = np.arange(len(db))
    axB.errorbar(db["f1_d"], y + 0.14,
                 xerr=[db["f1_d"] - db["f1_lo"], db["f1_hi"] - db["f1_d"]],
                 fmt="o", color=C_EMB, ms=5, lw=1.4, capsize=2.2,
                 label="internal MSK (95% CI)")
    axB.scatter(db["f3_d"], y - 0.14, marker="s", s=42, color=C_NEG,
                zorder=3, label="external GENIE")
    axB.axvline(0, ls=":", color=C_INK, lw=0.8)
    axB.set_yticks(y)
    axB.set_yticklabels(db["tumor"])
    axB.set_xlabel(r"Embedding contribution ($\Delta$ C-index, both $-$ tab)")
    axB.legend(loc="lower right", handletextpad=0.4)
    panel_label(axB, "b", dx=-0.22)

    fig.tight_layout(w_pad=2.2)
    out = FIGDIR / "fig3_external_forest.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
