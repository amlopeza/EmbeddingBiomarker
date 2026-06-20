# Design decisions

This document records the methodological decisions behind EmbeddingBiomarker and
the argument for each one, so that someone outside the project can understand and
reproduce it. It complements [`README.md`](README.md) (what to run) by explaining
*why* the pipeline is shaped the way it is. Findings are included only where they
justify a decision.

The project reuses the MSK-CHORD 2024 cohort from an earlier internship codebase,
but is a clean rewrite. Several decisions exist specifically to avoid pitfalls
inherited from that older code; those are flagged **[inherited pitfall]**.

---

## Data and reproducibility

**Regenerate every derived artifact from raw cBioPortal files (never trust a
shipped CSV).** `data_prompts.csv` is treated as a *derived* artifact, rebuilt by
`embedbiomarker.data.build_patient_table()` from the raw MSK-CHORD files.
*Argument* **[inherited pitfall]**: in the old repo a file named
`dataset2_sin_os_status.csv` actually *contained* `OS_STATUS` and was the wrong
corpus entirely. File names cannot be trusted; only code that runs from
`data/raw/` is trustworthy. Everything under `data/` is gitignored and
regenerated; only `results/` (JSON outputs) is versioned.

**Seeded, stratified split (seed 42, 80/10/10, stratified by
`CANCER_TYPE × OS_STATUS`), frozen and reused everywhere.** *Argument*
**[inherited pitfall]**: the old split called `np.random.shuffle` *without a seed*
before `train_test_split(random_state=42)`, so it was not reproducible despite the
fixed `random_state`. Stratifying on tumor × event keeps the tumor mix and event
rate comparable across folds. The same frozen split is reused by Phase 0, Phase 1
and Phase 3 so all numbers compare directly.

**One leakage-safe prompt per patient (~24k), no question, no replicas.**
*Argument* **[inherited pitfall]**: the old pipeline generated ~713k MCQ prompts
(5 replicas × 6–7 variables). Survival modelling needs exactly one descriptive
prompt per patient. The MCQ corpus is only relevant to generative fine-tuning,
not to feature extraction.

**The target never enters the model input.** `OS_STATUS` and `OS_MONTHS` are the
target only — they appear in neither the 12-feature context nor the prompt. A
mandatory guard test (`tests/test_no_leakage.py`) fails if `OS_STATUS`,
`OS_MONTHS`, or `Overall Survival` appears in any prompt. The 12 context features
are fixed: mutations history, age, treatment history, HER2, cancer stage, gender,
smoking history, history of PD-L1, fraction genome altered, MSI type, mutation
count, number of tumor diagnoses (ICD-O).

**Missingness is signal — keep it, do not silently impute it away.** Categorical
missing values are kept as an explicit `Not available` / `Unknown` category;
numeric features add a binary "was missing" indicator. The two numeric features
that *are* median-imputed (`Mutation_Count`, `Fraction_Genome_Altered`) are
imputed because their missingness is technical (assay not run), never a true 0 —
and the missing-indicator preserves the information that they were absent.

---

## Phase 0 — Tabular baseline first

**Build the tabular baseline before any embedding work.** Cox-PH, Random Survival
Forest and XGBoost-Cox over the 12 raw features. *Argument*: without a baseline
no embedding result is interpretable. **Result**: XGBoost-Cox reaches test
C-index **0.7549**, well above the old project's 0.594. This 0.7549 is the bar
every later model must beat. (The earlier 0.594 was confirmed leakage-free; the
jump is from clean featurization, not from leakage.)

**Featurize fit-on-train-only.** Top-K token vocabularies (mutations, treatments),
one-hot categories and median-imputation values are all learned on the train fold
and applied unchanged to val/test — no statistic crosses the split.

---

## Phase 1 — Extractor grid

**Embedding cache key = `(model_id, template_id, prompt_hash)`, with pooling in
the filename.** *Argument*: the expensive forward pass must run once and be reused
across the whole Cox grid; keying on the prompt text hash re-embeds only prompts
that actually changed. Folding `template_id` and pooling into the key guarantees
two prompt formats or two poolings of the same model never mix.

**Hold the comparison apples-to-apples.** Same frozen split, same target, same
concordance metric, same prompt template (`ctx_v1`, a natural-language preamble
that suits instruction-tuned backbones) and the same `max_length=512` for every
extractor.

**The extractor is the lever, not the prompt or the pooling.** *Findings that
drove this*: (1) MedCPT off-the-shelf does **not** beat the baseline (~0.718) and
adds no complementarity; (2) all three poolings (mean/cls/last) land at
~0.718–0.720, so pooling is not the bottleneck; (3) the prompt already contains
more raw information (ordered mutations + treatments as text, with repeats) than
the tabular multi-hot. The conclusion: an off-the-shelf encoder simply cannot
*extract* the signal — the real levers are a stronger extractor or fine-tuning,
not richer templates.

**Adopt MedGemma-1.5-4B as the primary extractor.** *Argument*: a stronger
backbone should extract more from the same 12 features. **Result**: MedGemma
(2560-d, masked-mean) gives `both` = 0.7649, complementarity **+0.0099** with a
paired-bootstrap 95% CI **[+0.0032, +0.0172]** that excludes 0 — whereas the best
MedCPT delta (+0.0003) is indistinguishable from noise.

**Report complementarity per tumor, not just globally.** **Finding**: the
embedding's added value is **concentrated in pancreas** (+0.0386, 95% CI
[+0.0200, +0.0583]) — the only individually significant stratum, and exactly where
the tabular branch is weakest. The global +0.0099 is significant because it
aggregates the five tumors; pancreas is its engine. This per-tumor result is the
most defensible Phase 1 claim.

**Pool MedGemma's text backbone, not its multimodal head.** MedGemma is a
multimodal model; the extraction notebook locates the text transformer
(`get_text_backbone`) and masked-mean-pools its `last_hidden_state`. Using the
raw multimodal wrapper output would not give a clean text embedding.

**Pancreas interpretability (falsification).** A counts/OOV/sequence
feature-engineering branch closes only **~26%** of MedGemma's pancreas advantage;
the remaining ~74% is **semantic** — structure (mutation/treatment combinations,
clinical meaning) that only the language model represents. *Consequence*: this
rejects the deflating alternative "just add count features and skip the LM", and
empirically motivates survival-aware fine-tuning as a future lever (the residual
is in the LM's territory).

---

## Phase ordering

**Do Phase 2 (treatment-aware) before fine-tuning.** *Argument*: Phase 2 secures
the paper with results that are already solid (the pancreas interpretability
supports it rigorously) and does not depend on a GPU or on fine-tuning working.
Fine-tuning a small model (SmolLM-130M, with a same-protocol MedGemma control to
isolate size from the fine-tuning effect) is a later reinforcement of the
"small/efficient" claim, not a prerequisite.

---

## Phase 2 — Treatment-aware analysis

**Compute both a pan-cancer and a cancer-specific risk score, cross-fit
out-of-fold.** *Argument*: the cross-fit yields a leakage-free OOF risk per
patient for both modelling choices (pan C=0.7556, specific C=0.7544, consistent
with Phase 1). High/low strata use the intra-tumor median of the OOF score.

**Use `log(risk)` (the linear predictor `f(x)`), not the raw `e^{f(x)}`, in the
strata Cox models.** *Argument*: the raw hazard ratio diverged in the NSCLC Cox
fit (a spurious HR of 0.00). `log` is monotone, so the C-index and KM curves are
unchanged; it only stabilizes the reported HRs.

**Separate prognostic from predictive signal with a risk × treatment
interaction.** **Finding**: the LM risk score is a robust *prognostic* factor in
all 5 tumors and within every therapy stratum (log-rank p ≪ 1e-10; age+stage
adjusted HR per SD 1.93–3.79, unpenalized Cox). The interaction term is
**exploratory** and read with three caveats: (1) immortal-time bias
(`TREATMENT_HISTORY` is post-baseline); (2) it is evaluated **case by case (per
tumor)** — each tumor is a separate population with its own treatments, so the
interactions are not one joint family and are not globally multiple-testing
corrected (the cancer-specific score is the primary lens, the pan-cancer score a
robustness check; within a tumor the few strata share the cohort, a small family
at most); (3) **circularity** — the risk score is trained on features that
include treatment history, so a risk×treatment interaction is partly mechanical.
Empirically, untreated strata give interaction
HR ≈ 1 (pure prognosis) in 4 of 5 tumors — **NSCLC is the exception** (untreated
interaction HR ≈ 1.18, p≈8e-9); treated strata > 1 (effect modification).

**Report HR/CI/p from an UNPENALIZED Cox.** *Argument*: the adjusted-HR and
interaction models are fit with `penalizer=0.0` so the Wald CI and p-value are
valid inference. An earlier ridge (`penalizer=0.1`) was carried over for numeric
stability, but it shrank the HRs 13–36% toward 1 and invalidates the CI/p; once
risk enters as `log(risk)` all 50 fits converge unpenalized (0 fallbacks), so the
penalty is unnecessary. `cox_*` keep a stability fallback that re-adds a small
ridge ONLY if an unpenalized fit fails to converge, and record it in a
`penalizer` field so a penalized (non-inferential) estimate is never reported as a
clean one.

---

## Phase 3 — External validation

**Validate on AACR GENIE BPC.** *Argument*: it covers the same 5 solid tumors and
adds exactly what the genomic GENIE lacks — systemic treatments with dates, OS
(death/censor) and curated stage — with public releases. About 7/12 features map
cleanly; the rest fall to "Not available" and the model degrades gracefully.

**Frozen model, zero retraining (decision D1: fit on MSK train+val).** The
`TabularFeaturizer` and XGBoost-Cox head are fit once on MSK (train+val) and
applied unchanged to the external cohort. No statistic crosses cohorts, so the
external cohort is a fully independent test set (and the internal MSK test number
from Phase 1 stays intact). Because there is no in-cohort leakage to guard
against, external risk is the direct `predict_risk` — no cross-fit needed (unlike
the Phase 2 OOF risk).

**Drop MSK-center patients — the decisive leakage control.** MSK contributes to
*both* MSK-CHORD (train) and GENIE BPC, so validating on BPC's MSK patients would
be leakage. Only non-MSK centers (DFCI, VICC, UHN) are kept. This is enforced in
`build_genie_table` and is a mandatory test: it must fail if an MSK id enters.
Bonus: it strengthens the cross-institution generalization claim.

**Two KM cutpoints (decision D2).** Report the intra-external median (pure
discrimination within the external cohort) *and* the MSK reference median (does
the MSK-learned threshold transfer / ranking calibration).

**Schema mapping decisions (BPC → the 12 MSK features).**
- **Age (D3):** use `AGE_AT_SEQUENCING` from `data_clinical_sample.txt` (clean
  integer, age at sequencing).
- **Agent vocabulary (D4):** 0 names match exactly between BPC and MSK (BPC is
  Title-Case with salt suffixes, MSK is UPPERCASE simple names). For the *tabular*
  branch, normalize with `upper()` + strip salt suffixes (HCL/liposome/…) + a
  small rename dict (e.g. `Nabpaclitaxel` → `PACLITAXEL PROTEIN-BOUND`). The
  normalization is applied once when building the table, so BOTH the tabular
  features and the prompt/embedding see the normalized agent names (keeps the
  external prompts apples-to-apples with MSK).
- **Missing features land on the correct one-hot column.** Features with no BPC
  source are set to MSK's *own* missing token per feature (`Not available` /
  `Unknown` / NaN). In particular Smoking must be `Unknown`, not `Not available`,
  or every patient would collapse onto the dropped reference level ("Former/Current
  Smoker") and read as a smoker.
- **PATIENT_ID** is the first 3 barcode segments in GENIE (`GENIE-DFCI-136491`)
  vs 2 in MSK (`P-0000001`).
- **Multi-sample patients** are collapsed to one row (lowest `AGE_AT_SEQUENCING` /
  first sample).
- **BLADDER is excluded**: it is not an MSK-CHORD tumor, so the frozen model has
  no prior for it.

**The C-index is robust to an OS time-origin shift between cohorts.** OS may be
timed from a different origin (diagnosis vs sequencing) in BPC. Because
concordance is rank-based, a uniform scale shift does not change it; the
discrimination claim holds. (Absolute KM medians across cohorts should be read
with this caveat.)

**Pilot pancreas first, then scale to all 5.** Validate the whole pipeline on the
public PANC release (the headline tumor) before extending the `CANCER_TYPE`
mapping to the other four.

**Data fix:** CRC had one duplicated patient (`GENIE-VICC-382607`, two identical
survival rows) that the survival merge expanded into two rows. Fixed by
de-duplicating survival by `PATIENT_ID` before the merge in `external.py`, plus a
defensive de-dup of the embedding index in `41_external_validate.py` (777 → 776).

**Result.** Discrimination transfers out of MSK in all 5 tumors (C-index ≥ 0.65,
no retraining). The embedding's contribution is positive and consistent where
Phase 1 predicted it (pancreas, prostate), ~neutral in colorectal/breast, and
negative in NSCLC. The cause is an open question: prompt truncation was measured
and ruled out (MedCPT-tokenizer length: median ~125 tokens, p95 ~185, <0.2% of
prompts exceed the 512 cap, NSCLC not an outlier), so a candidate but **untested**
explanation is divergent gene panels → more out-of-vocabulary tokens. See
`README.md` for the per-tumor table.

---

