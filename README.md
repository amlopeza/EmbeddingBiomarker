# EmbeddingBiomarker

**A small/efficient language model as a treatment-aware prognostic feature
extractor** on the MSK-CHORD 2024 cohort (~23,777 patients, 5 solid tumors).

The model does **not** predict survival directly. Instead it turns a patient's
leakage-safe clinical description into an embedding; that embedding (optionally
concatenated with 12 structured features) feeds a survival head (XGBoost-Cox).
The headline is the **treatment-aware analysis by therapy strata** and an
**external, frozen-model validation** on an independent cohort.

The design rationale and every methodological decision (with the argument behind
it) are documented in [`decisions.md`](decisions.md).

## Install

```bash
pip install -e ".[dev]"             # Phase 0: splits, prompts, tabular baseline
pip install -e ".[dev,embeddings]"  # Phase 1+: torch / transformers extractors
```

## Data

The raw cBioPortal **MSK-CHORD 2024** files must be present under
`data/raw/msk_chord_2024/` (`data_clinical_patient.txt`,
`data_timeline_treatment.txt`, `data_mutations.txt`,
`msk_chord_2024_clinical_data.tsv`). The external **AACR GENIE BPC** cohorts go
under `data/raw/genie_bpc/<COHORT>/cBioPortal_files/` (Phase 3).

Everything under `data/` is gitignored and **rebuilt from code** — no opaque
artifacts. The patient table (`data_prompts.csv`) is a *derived* artifact, not
raw: it is regenerated from the raw files by `embedbiomarker.data` and written to
`data/interim/` (it must **not** be placed under `data/raw/`).

### Obtaining the raw data

- **MSK-CHORD 2024 (internal cohort).** Public on cBioPortal, study id
  `msk_chord_2024`. Download the study tarball with the *Download* button on the
  [study page](https://www.cbioportal.org/study/summary?id=msk_chord_2024), or
  pull it from the [cBioPortal datahub](https://github.com/cBioPortal/datahub).
  Extract the four files named above into `data/raw/msk_chord_2024/`.
- **AACR GENIE BPC (external cohort, Phase 3).** Create a [Synapse](https://www.synapse.org/genie)
  account and accept the data terms, then download the BPC public releases (Files
  → Data Releases → cohort → version). Place each cohort under
  `data/raw/genie_bpc/<COHORT>/cBioPortal_files/`. Validation keeps only the
  non-MSK centers (DFCI / VICC / UHN); MSK overlap is dropped automatically to
  avoid leakage with the MSK-CHORD training cohort.

The exact patient counts and column shape are asserted in code
(`scripts/00_build_table.py` checks 23,777 × 26), so a successful rebuild
confirms you have the right inputs.

### Leakage guard

`OS_STATUS` / `OS_MONTHS` are the **target only** — they never enter a prompt or
a feature. A test (`tests/test_no_leakage.py`) fails if `OS_STATUS`,
`OS_MONTHS`, or `Overall Survival` appears in any prompt.

## Phase 0 — Consolidation + tabular baseline (the line to beat)

```bash
python scripts/00_build_table.py    # raw cBioPortal -> data/interim/data_prompts.csv
python scripts/01_make_splits.py    # frozen 80/10/10 seeded split -> data/interim/splits.json
python scripts/02_make_prompts.py   # one leakage-safe prompt/patient -> data/interim/prompts.parquet
python scripts/10_baseline_tabular.py  # Cox-PH / RSF / XGBoost-Cox -> results/baseline_tabular.json
pytest                              # split integrity + no-leakage guard
```

Tabular baseline (12 features, XGBoost-Cox): **test C-index = 0.7549**. This is
the bar every embedding model must beat.

## Phase 1 — Extractor grid (embeddings → XGBoost-Cox)

Embeddings are extracted once on a GPU (Colab T4) and scored locally on CPU. The
extraction notebook is versioned and reproducible:

```text
notebooks/21_extract_embeddings_phase1.ipynb   # MedCPT + MedGemma-1.5-4B embeddings of the MSK prompts
```

Then score embeddings-only / tabular / concatenated with the same frozen split:

```bash
python scripts/22_cox_grid.py --model medgemma15 --pooling mean --n-boot 1000 \
    --out "$PWD/results/embedding_grid__medgemma15.json"
```

**Result — of the extractors evaluated (MedGemma and MedCPT), MedGemma-1.5-4B
(2560-d, masked-mean) is the only one whose combined model (emb ⊕ tab) beats the
baseline and adds complementary signal (embeddings alone stay below the bar):**

| Feature set | Test C-index | Δ vs tabular |
|---|---:|---:|
| tabular (12 features) | 0.7549 | — |
| embeddings only | 0.7503 | −0.0046 |
| **both (emb ⊕ tab)** | **0.7649** | **+0.0099** (95% CI [+0.0032, +0.0172]) |

The complementarity is **concentrated in pancreas**: +0.0386 (95% CI
[+0.0200, +0.0583]) — the only per-tumor stratum individually significant, and
exactly where the tabular branch is weakest. MedCPT, by contrast, adds nothing
(+0.0003, CI crosses 0). Interpretability (`scripts/24_pancreas_interpret.py`)
shows ~74% of the pancreas gain is **semantic** — it cannot be reproduced by
count/OOV/sequence engineering, only by the language model.

## Phase 2 — Treatment-aware analysis (the headline)

```bash
python scripts/30_treatment_analysis.py            # cross-fit OOF risk + strata stats
python scripts/30_treatment_analysis.py --reuse-risk   # re-run stats without re-cross-fitting
```

A cross-fit produces an out-of-fold risk score per patient (pan-cancer
C=0.7556, cancer-specific C=0.7544). The LM-derived risk is a **robust
prognostic factor in all 5 tumors** and within every therapy stratum (log-rank
p ≪ 1e-10). Adjusted HR per +1 SD of risk (age + stage adjusted, **unpenalized**
Cox so the CI/p are valid inference):

| Tumor | adjusted HR per SD [95% CI] |
|---|---:|
| Breast | 3.79 [3.56, 4.04] |
| Prostate | 3.50 [3.22, 3.80] |
| Colorectal | 2.22 [2.10, 2.33] |
| NSCLC | 2.08 [2.00, 2.16] |
| Pancreas | 1.93 [1.81, 2.05] |

(all p ≤ 1e-97.)

A **risk × treatment interaction** term probes prognostic vs predictive signal,
evaluated **case by case (per tumor)** and read as **exploratory**. Each tumor is
a separate population with its own therapies — the interactions are not pooled
into one joint claim across experiments, so they are not corrected as a single
multiple-testing family (the cancer-specific score is the primary lens; the
pan-cancer score is a robustness comparison). It is also partly **circular**: the
risk score is trained on features that already include treatment history.
Empirically, untreated strata give interaction HR ≈ 1 (pure prognosis) in 4 of 5
tumors (NSCLC the exception); treated strata > 1 (effect modification). Figures:
`notebooks/30_treatment_figures.ipynb`.

## Phase 3 — External validation (frozen model, zero retraining)

The featurizer and XGBoost-Cox head are fit **once** on MSK (train+val) and
applied **unchanged** to the independent **AACR GENIE BPC** cohorts. Only the
external embeddings are computed anew. Patients from MSK centers are dropped
(MSK contributes to both cohorts → leakage); validation runs on DFCI / VICC /
UHN only.

```bash
python scripts/40_build_external.py --cohort PANC      # external table + prompts (per cohort)
# notebooks/42_extract_embeddings_external.ipynb       # MedGemma embeddings of the external prompts (Colab T4)
python scripts/41_external_validate.py --all           # frozen-model validation, all 5 solid tumors
```

**Discrimination transfers out of MSK in all 5 tumors** (C-index ≥ 0.65, no
retraining). Internal (MSK test) vs external (GENIE BPC, frozen):

| Tumor | n ext | int. both | ext. tab | ext. both | ext. Δ (both−tab) |
|---|---:|---:|---:|---:|---:|
| Pancreas | 551 | 0.6879 | 0.7150 | 0.7253 | **+0.0103** |
| Prostate | 517 | 0.7875 | 0.6559 | 0.6946 | **+0.0387** |
| Colorectal | 776 | 0.7499 | 0.7137 | 0.7145 | +0.0007 |
| Breast | 596 | 0.7945 | 0.6826 | 0.6786 | −0.0040 |
| NSCLC | 955 | 0.7367 | 0.7180 | 0.6820 | −0.0360 |

The embedding's contribution is positive and consistent where Phase 1 predicted
it (pancreas, prostate), neutral in colorectal/breast, and negative in NSCLC.
The cause is open: prompt truncation was measured and ruled out (median ~125
tokens, <0.2% hit the 512 cap), so a candidate but **untested** explanation is
divergent gene panels → more out-of-vocabulary tokens. Figures:
`notebooks/40_external_figures.ipynb`, `notebooks/41_external_forest.ipynb`.

## Repository structure

```text
config/      data.yaml · models.yaml · survival.yaml · treatments.yaml
src/embedbiomarker/   data · splits · prompts · embeddings · baselines ·
                      survival · analysis · external
scripts/     00_build_table · 01_make_splits · 02_make_prompts ·
             10_baseline_tabular · 20_extract_embeddings · 22_cox_grid ·
             24_pancreas_interpret · 30_treatment_analysis ·
             40_build_external · 41_external_validate
notebooks/   embedding extraction (21 internal, 42 external) +
             figures (10, 20, 24, 30, 40, 41)
results/      versioned JSON outputs (the only data-derived files in git)
tests/        splits · no-leakage · baseline · treatment strata · external
```

## Conventions

- No opaque artifacts: everything under `data/` is regenerated from `data/raw`
  with seeded code. `results/` is versioned; `data/` is not.
- The tabular baseline must exist before any "serious" embedding run.
