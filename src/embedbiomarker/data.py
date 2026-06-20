"""Build the patient-level table from the raw cBioPortal MSK-CHORD 2024 dump.

Versioned port of the ``01_data_processing.ipynb`` notebook. It rebuilds the
patient table from the raw cBioPortal inputs, so the result is a derived artifact
(written under ``data/interim/``) that is fully traceable to source.

The pipeline (one row per patient):
  1. Load ``data_clinical_patient.txt`` (base demographics/clinical/targets).
  2. ``TREATMENT_HISTORY`` from ``data_timeline_treatment.txt`` (START_DATE >= 0,
     agents ordered by date).
  3. ``MUTATIONS`` from ``data_mutations.txt`` (functional variants, gene set).
  4. Left-merge clinical + treatments + mutations.
  5. Clean: drop patients missing CURRENT_AGE_DEID or MUTATIONS; impute the rest.
  6. ``CANCER_TYPE`` from ``msk_chord_2024_clinical_data.tsv`` (one type/patient).
  7. Collapse 9 organ columns into ``METASTATIC_SITES``.
  8. Add 6 genomic columns from the clinical-data table.

Design decisions:
  * Mutations are aggregated as ``sorted(set(...))`` so the gene order in
    ``MUTATIONS`` is deterministic across runs.
  * Treatment agents keep their real repeats by default (they encode actual
    re-treatment lines); ``dedup_treatments=True`` collapses them, order-preserving.
  * Inclusion criterion: patients missing age or with NO functional mutation are
    excluded (step 5; ~1,170 of 24,950 -> 23,777), because the prompt/embedding is
    built around the mutation profile. For every OTHER feature, missing values
    become explicit categories ("Unknown" / "Not available") rather than being
    dropped — missingness there is itself signal.

Expected output: 23,777 patients x 26 columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Raw cBioPortal files (true raw inputs) --------------------------------
DEFAULT_RAW_DIR = Path("data/raw/msk_chord_2024")
CLINICAL_PATIENT = "data_clinical_patient.txt"
TIMELINE_TREATMENT = "data_timeline_treatment.txt"
MUTATIONS = "data_mutations.txt"
CLINICAL_DATA = "msk_chord_2024_clinical_data.tsv"

# Variant classes excluded as non-functional (matches the old notebook).
NON_FUNCTIONAL_VARIANTS = ["Silent", "Intron"]

# Organ columns collapsed into METASTATIC_SITES ("OTHER" is dropped: not a site).
METASTATIC_COLUMNS = [
    "ADRENAL_GLANDS", "BONE", "CNS_BRAIN", "INTRA_ABDOMINAL", "LIVER",
    "LUNG", "LYMPH_NODES", "PLEURA", "REPRODUCTIVE_ORGANS",
]

# Missing-value imputation (missingness is itself signal -> explicit category).
FILL_VALUES = {
    "TREATMENT_HISTORY": "Unknown",
    "GLEASON_FIRST_REPORTED": "Not available",
    "GLEASON_HIGHEST_REPORTED": "Not available",
    "HR": "Not available",
    "HER2": "Not available",
    "HISTORY_OF_PDL1": "Not available",
}

# Six extra columns pulled from the clinical-data .tsv (raw label -> our label).
EXTRA_COLUMNS = {
    "Fraction Genome Altered": "Fraction_Genome_Altered",
    "MSI Type": "MSI_Type",
    "Mutation Count": "Mutation_Count",
    "Primary Tumor Site": "Primary_Tumor_Site",
    "TMB (nonsynonymous)": "TMB_nonsynonymous",
    "Tumor Purity": "Tumor_Purity",
}


def _treatment_history(raw_dir: Path, dedup: bool) -> pd.DataFrame:
    """One TREATMENT_HISTORY string per patient, agents ordered by START_DATE.

    Only post-baseline treatments (START_DATE >= 0) are kept, matching the
    notebook. ``dedup`` collapses consecutive/repeated agents while preserving
    first-seen order; the default (False) keeps repeats, which encode real
    re-treatment lines.
    """
    treatment = pd.read_csv(raw_dir / TIMELINE_TREATMENT, sep="\t")
    treatment = treatment[treatment["START_DATE"] >= 0]

    def join_agents(group: pd.DataFrame) -> str:
        # stable sort -> deterministic order even when START_DATE ties.
        agents = group.sort_values("START_DATE", kind="stable")["AGENT"].tolist()
        if dedup:
            agents = list(dict.fromkeys(agents))  # order-preserving dedup
        return ", ".join(agents)

    # Select only the columns the callback needs so PATIENT_ID (the group key)
    # is not passed into the callback (avoids the pandas grouping-column warning).
    history = (
        treatment.groupby("PATIENT_ID")[["START_DATE", "AGENT"]]
        .apply(join_agents)
        .reset_index(name="TREATMENT_HISTORY")
    )
    return history


def _mutations_list(raw_dir: Path) -> pd.DataFrame:
    """One MUTATIONS string per patient: sorted set of functionally-mutated genes.

    PATIENT_ID is derived from the first two segments of the sample barcode.
    Genes are ``sorted(set(...))`` so the order is deterministic across runs.
    """
    mut = pd.read_csv(raw_dir / MUTATIONS, sep="\t", low_memory=False)
    mut["PATIENT_ID"] = mut["Tumor_Sample_Barcode"].str.split("-").str[:2].str.join("-")
    mut = mut[~mut["Variant_Classification"].isin(NON_FUNCTIONAL_VARIANTS)]

    genes = (
        mut.groupby("PATIENT_ID")["Hugo_Symbol"]
        .apply(lambda s: ", ".join(sorted(set(s))))
        .reset_index(name="MUTATIONS")
    )
    return genes


def _metastatic_sites(row: pd.Series) -> str:
    """Collapse the 9 organ 'Yes/No' columns into a readable, comma-joined list."""
    sites = [
        organ.replace("_", " ").title()
        for organ in METASTATIC_COLUMNS
        if str(row[organ]).strip().lower() == "yes"
    ]
    return ", ".join(sites) or "Unknown"


def build_patient_table(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    dedup_treatments: bool = False,
) -> pd.DataFrame:
    """Regenerate the patient-level table (``data_prompts.csv``) from raw inputs.

    Args:
        raw_dir: directory holding the raw cBioPortal MSK-CHORD files.
        dedup_treatments: if True, collapse repeated treatment agents per patient
            (order-preserving). Default False reproduces the old notebook output.

    Returns:
        DataFrame with one row per patient and 26 columns (~23,777 rows).
    """
    raw_dir = Path(raw_dir)

    # 1. Base clinical/demographic table (includes the 9 organ columns + targets).
    patients = pd.read_csv(raw_dir / CLINICAL_PATIENT, sep="\t", comment="#")

    # 2-3. Derived string features.
    treatment_history = _treatment_history(raw_dir, dedup=dedup_treatments)
    mutations = _mutations_list(raw_dir)

    # 4. Left-merge so patients without treatment/mutation rows are kept (then
    #    cleaned/imputed below), not silently dropped.
    df = patients.merge(treatment_history, on="PATIENT_ID", how="left")
    df = df.merge(mutations, on="PATIENT_ID", how="left")

    # 5a. Drop patients missing the two fields required downstream.
    df = df.dropna(subset=["CURRENT_AGE_DEID", "MUTATIONS"]).copy()

    # 5b. Impute the rest. The notebook normalized the literal "null" string only
    #     in GLEASON_FIRST_REPORTED before filling; ported faithfully.
    df["GLEASON_FIRST_REPORTED"] = df["GLEASON_FIRST_REPORTED"].replace(
        ["null", None], pd.NA
    )
    df = df.fillna(value=FILL_VALUES)

    # 6. Cancer type (one per patient) from the clinical-data table. Read once and
    #    reused in step 8.
    clinical_data = pd.read_csv(raw_dir / CLINICAL_DATA, sep="\t", comment="#")
    cancer_type = (
        clinical_data[["Patient ID", "Cancer Type"]]
        .rename(columns={"Patient ID": "PATIENT_ID", "Cancer Type": "CANCER_TYPE"})
        # Keep the first cancer type per patient (drops multi-type duplicates).
        .drop_duplicates(subset="PATIENT_ID")
    )
    df = df.merge(cancer_type, on="PATIENT_ID", how="left")

    # 7. Collapse organ columns into a single METASTATIC_SITES, then drop them
    #    plus the uninformative "OTHER" column.
    df["METASTATIC_SITES"] = df.apply(_metastatic_sites, axis=1)
    df = df.drop(columns=METASTATIC_COLUMNS + ["OTHER"])

    # 8. Enrich with six genomic/context columns from the clinical-data table.
    extra = (
        clinical_data[["Patient ID", *EXTRA_COLUMNS.keys()]]
        .rename(columns={"Patient ID": "PATIENT_ID", **EXTRA_COLUMNS})
        .drop_duplicates(subset="PATIENT_ID")
    )
    df = df.merge(extra, on="PATIENT_ID", how="left")

    return df.reset_index(drop=True)
