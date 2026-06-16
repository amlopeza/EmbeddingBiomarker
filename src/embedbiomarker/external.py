"""Build the 12-feature patient table from AACR GENIE BPC raw (Phase 3 adapter).

Parallel to :mod:`embedbiomarker.data` (which targets MSK-CHORD), but for the
external validation cohort. It emits a table with the SAME column names ``data.py``
produces, so the MSK-fit :class:`~embedbiomarker.survival.TabularFeaturizer` and the
prompt renderer run on it UNCHANGED — that is the whole point of frozen-model
external validation: no re-fitting, only ``transform`` / ``predict_risk``.

Differences from MSK-CHORD, handled here (decisions logged in plans.md, step 14.0):
  * PATIENT_ID barcode is 3 segments (``GENIE-DFCI-136491``), not 2 (``P-0000001``).
  * Leakage filter: MSK contributes to BOTH MSK-CHORD (train) and GENIE BPC, so
    patients from ``exclude_centers`` (MSK) are dropped — validate only on unseen
    institutions (DFCI / UHN / VICC).
  * Agent vocabulary differs (BPC Title-Case + salt suffix vs MSK UPPERCASE); the
    tabular branch needs them normalized to the MSK form. The embedding branch
    consumes the raw text and does not.
  * Several MSK features have no BPC source. They are set to MSK's OWN missing
    token per feature (``Not available`` / ``Unknown`` / NaN) so they land on the
    correct one-hot column instead of silently collapsing to the dropped reference
    level (e.g. Smoking must be ``Unknown``, not ``Not available``, or every patient
    reads as the reference level "Former/Current Smoker").

The frozen model then degrades gracefully on the missing features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the non-functional variant filter so MUTATIONS matches data.py exactly.
from .data import NON_FUNCTIONAL_VARIANTS

# --- Raw GENIE BPC layout --------------------------------------------------- #
DEFAULT_GENIE_DIR = Path("data/raw/genie_bpc")
CBIO_SUBDIR = "cBioPortal_files"
CLINICAL_PATIENT = "data_clinical_patient.txt"
CLINICAL_SAMPLE = "data_clinical_sample.txt"
SURVIVAL_SUPP = "data_clinical_supp_survival.txt"
TIMELINE_TREATMENT = "data_timeline_treatment.txt"
MUTATIONS = "data_mutations_extended.txt"

# Cohort folder -> the MSK-CHORD CANCER_TYPE value it must match for per-tumor
# reporting. PANC is verified ("Pancreatic Cancer"); the rest are filled in /
# verified against the MSK table when Phase 3 scales out (step 14.7).
COHORT_CANCER_TYPE = {
    "PANC": "Pancreatic Cancer",
}

# Centers whose patients are dropped (leakage: also in MSK-CHORD train).
DEFAULT_EXCLUDE_CENTERS = ("MSK",)

# --- Agent normalization (BPC -> MSK vocabulary, tabular branch only) -------- #
# 0 exact matches BPC<->MSK (step 14.0): MSK is UPPERCASE + simple name, BPC is
# Title-Case + salt suffix. upper() fixes most; this dict handles the rest.
AGENT_RENAME = {
    "GEMCITABINE HCL": "GEMCITABINE",
    "IRINOTECAN HCL": "IRINOTECAN",
    "IRINOTECAN LIPOSOME": "IRINOTECAN LIPOSOMAL",
    "NABPACLITAXEL": "PACLITAXEL PROTEIN-BOUND",
}


def normalize_agent(name: str) -> str:
    """Map a BPC agent name onto the MSK-CHORD agent vocabulary."""
    upper = str(name).strip().upper()
    return AGENT_RENAME.get(upper, upper)


def _read_cbio(path: Path) -> pd.DataFrame:
    """Read a cBioPortal clinical file (``#``-commented 5-line header)."""
    return pd.read_csv(path, sep="\t", comment="#")


def _patient_id_from_barcode(barcode: pd.Series) -> pd.Series:
    """GENIE sample barcode -> patient id (first 3 dash-separated segments)."""
    return barcode.str.split("-").str[:3].str.join("-")


def _treatment_history(cbio: Path) -> pd.DataFrame:
    """One TREATMENT_HISTORY string per patient, agents (MSK-normalized) by date.

    Mirrors :func:`embedbiomarker.data._treatment_history`: post-baseline only
    (START_DATE >= 0), stable sort by date, real repeats kept (re-treatment lines).
    """
    trt = pd.read_csv(cbio / TIMELINE_TREATMENT, sep="\t")
    trt = trt[trt["START_DATE"] >= 0].copy()
    trt["AGENT"] = trt["AGENT"].map(normalize_agent)

    def join_agents(group: pd.DataFrame) -> str:
        return ", ".join(group.sort_values("START_DATE", kind="stable")["AGENT"].tolist())

    return (
        trt.groupby("PATIENT_ID")[["START_DATE", "AGENT"]]
        .apply(join_agents)
        .reset_index(name="TREATMENT_HISTORY")
    )


def _mutations(cbio: Path) -> pd.DataFrame:
    """Per patient: MUTATIONS (sorted gene set) and Mutation_Count (functional).

    Same functional filter as MSK (drop Silent/Intron). Mutation_Count here is the
    count of functional mutation rows (a proxy for the cBioPortal nonsynonymous
    count, which MSK pulls from its clinical .tsv); the featurizer z-scores it.
    """
    mut = pd.read_csv(cbio / MUTATIONS, sep="\t", low_memory=False)
    mut["PATIENT_ID"] = _patient_id_from_barcode(mut["Tumor_Sample_Barcode"])
    mut = mut[~mut["Variant_Classification"].isin(NON_FUNCTIONAL_VARIANTS)]

    genes = (
        mut.groupby("PATIENT_ID")["Hugo_Symbol"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index(name="MUTATIONS")
    )
    counts = mut.groupby("PATIENT_ID").size().reset_index(name="Mutation_Count")
    return genes.merge(counts, on="PATIENT_ID", how="outer")


def _sample_level(cbio: Path) -> pd.DataFrame:
    """Collapse the sample table to one row per patient (some have 2-3 samples).

    Age -> earliest sample (min AGE_AT_SEQUENCING). PD-L1 history -> "Yes" if any
    sample was tested positive-eligible, else the recorded PDL1_TESTING value.
    """
    smp = _read_cbio(cbio / CLINICAL_SAMPLE)
    age = smp.groupby("PATIENT_ID")["AGE_AT_SEQUENCING"].min()

    def pdl1(group: pd.Series) -> str:
        vals = set(group.dropna().astype(str))
        if "Yes" in vals:
            return "Yes"
        if "No" in vals:
            return "No"
        return "Not available"

    pdl1_hist = smp.groupby("PATIENT_ID")["PDL1_TESTING"].apply(pdl1)
    return pd.DataFrame({
        "CURRENT_AGE_DEID": age,
        "HISTORY_OF_PDL1": pdl1_hist,
    }).reset_index()


def _map_stage(stage: pd.Series) -> pd.Series:
    """BPC STAGE_DX (Stage I..IV, NOS) -> MSK STAGE_HIGHEST_RECORDED bins."""
    return np.where(
        stage.astype(str).str.strip().eq("Stage IV"), "Stage 4", "Stage 1-3"
    )


def build_genie_table(
    cohort: str = "PANC",
    raw_dir: Path | str = DEFAULT_GENIE_DIR,
    cancer_type: str | None = None,
    exclude_centers: tuple[str, ...] = DEFAULT_EXCLUDE_CENTERS,
) -> pd.DataFrame:
    """Build the external (GENIE BPC) patient table in MSK-CHORD column schema.

    Args:
        cohort: cohort folder under ``raw_dir`` (e.g. "PANC").
        raw_dir: GENIE BPC root holding the per-cohort folders.
        cancer_type: CANCER_TYPE value to stamp (defaults to the cohort's MSK match).
        exclude_centers: institutions dropped for leakage (default: MSK).

    Returns:
        One row per non-excluded patient with the 12 feature columns, PATIENT_ID,
        OS_STATUS/OS_MONTHS, CANCER_TYPE and CENTER. Patients missing MUTATIONS or
        OS are dropped (same discipline as ``data.build_patient_table``).
    """
    raw_dir = Path(raw_dir)
    cbio = raw_dir / cohort / CBIO_SUBDIR
    cancer_type = cancer_type or COHORT_CANCER_TYPE.get(cohort)
    if cancer_type is None:
        raise ValueError(f"no CANCER_TYPE known for cohort {cohort!r}; pass cancer_type=")

    # 1. Patient base: center filter (leakage), stage, sex, ICD-O proxy.
    pat = _read_cbio(cbio / CLINICAL_PATIENT)
    pat = pat[~pat["CENTER"].isin(exclude_centers)].copy()

    df = pd.DataFrame({
        "PATIENT_ID": pat["PATIENT_ID"].astype(str),
        "CENTER": pat["CENTER"],
        "GENDER": pat["SEX"],
        "STAGE_HIGHEST_RECORDED": _map_stage(pat["STAGE_DX"]),
        "NUM_ICDO_DX": pd.to_numeric(pat["N_CANCERS"], errors="coerce"),
    })

    # 2. Merge derived blocks (left joins keep all non-excluded patients for now).
    df = df.merge(_sample_level(cbio), on="PATIENT_ID", how="left")
    df = df.merge(_treatment_history(cbio), on="PATIENT_ID", how="left")
    df = df.merge(_mutations(cbio), on="PATIENT_ID", how="left")

    # 3. Survival target (token identical to MSK: "1:DECEASED" / "0:LIVING").
    surv = _read_cbio(cbio / SURVIVAL_SUPP)
    df = df.merge(
        surv[["PATIENT_ID", "OS_STATUS", "OS_MONTHS"]], on="PATIENT_ID", how="left"
    )

    # 4. Features with no BPC source -> MSK's own missing token (graceful + lands
    #    on the right one-hot level). Smoking="Unknown" (MSK token), NOT "Not
    #    available", which would collapse to the dropped reference level.
    df["HER2"] = "Not available"
    df["MSI_Type"] = "Not available"
    df["SMOKING_PREDICTIONS_3_CLASSES"] = "Unknown"
    df["Fraction_Genome_Altered"] = np.nan  # numeric -> featurizer median-imputes
    df["CANCER_TYPE"] = cancer_type

    # 5. Fill text/categorical gaps the same way data.py does, then drop patients
    #    without the two fields required downstream (MUTATIONS, OS).
    df["TREATMENT_HISTORY"] = df["TREATMENT_HISTORY"].fillna("Unknown")
    df["HISTORY_OF_PDL1"] = df["HISTORY_OF_PDL1"].fillna("Not available")
    df = df.dropna(subset=["MUTATIONS", "OS_STATUS", "OS_MONTHS"]).reset_index(drop=True)

    return df
