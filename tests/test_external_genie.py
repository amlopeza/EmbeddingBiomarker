"""Phase 3 adapter guards: the GENIE BPC external table must be leakage-safe and
schema-compatible with the MSK-fit featurizer / prompt renderer.

These run only when the raw GENIE BPC PANC dump is present (it is gitignored), so
they skip cleanly on a fresh checkout.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from embedbiomarker import prompts
from embedbiomarker.external import (
    DEFAULT_GENIE_DIR,
    build_genie_table,
    normalize_agent,
)

PANC_DIR = DEFAULT_GENIE_DIR / "PANC" / "cBioPortal_files"
pytestmark = pytest.mark.skipif(
    not PANC_DIR.exists(), reason="GENIE BPC PANC raw not present (gitignored)"
)

# The 12 feature columns the MSK-fit featurizer/prompt renderer require.
FEATURE_COLUMNS = {
    "MUTATIONS", "CURRENT_AGE_DEID", "TREATMENT_HISTORY", "HER2",
    "STAGE_HIGHEST_RECORDED", "GENDER", "SMOKING_PREDICTIONS_3_CLASSES",
    "HISTORY_OF_PDL1", "Fraction_Genome_Altered", "MSI_Type", "Mutation_Count",
    "NUM_ICDO_DX",
}


@pytest.fixture(scope="module")
def panc() -> pd.DataFrame:
    return build_genie_table("PANC")


def test_nonempty(panc: pd.DataFrame):
    assert len(panc) > 0


def test_all_feature_columns_present(panc: pd.DataFrame):
    missing = FEATURE_COLUMNS - set(panc.columns)
    assert not missing, f"missing feature columns: {sorted(missing)}"


def test_no_msk_patients_leakage(panc: pd.DataFrame):
    # The decisive guard: no MSK-center patient may survive the filter.
    assert "MSK" not in set(panc["CENTER"])
    assert not panc["PATIENT_ID"].str.contains("-MSK-").any()


def test_one_row_per_patient(panc: pd.DataFrame):
    assert panc["PATIENT_ID"].is_unique


def test_patient_id_is_three_segments(panc: pd.DataFrame):
    # GENIE-<CENTER>-<id>, e.g. GENIE-DFCI-136491.
    assert (panc["PATIENT_ID"].str.split("-").str.len() == 3).all()


def test_os_present_and_complete(panc: pd.DataFrame):
    assert {"OS_STATUS", "OS_MONTHS"} <= set(panc.columns)
    assert panc["OS_STATUS"].notna().all()
    assert panc["OS_MONTHS"].notna().all()
    # Token must match MSK so make_target works unchanged.
    assert set(panc["OS_STATUS"]) <= {"0:LIVING", "1:DECEASED"}


def test_categoricals_match_msk_vocabulary(panc: pd.DataFrame):
    assert set(panc["GENDER"]) <= {"Male", "Female"}
    assert set(panc["STAGE_HIGHEST_RECORDED"]) <= {"Stage 1-3", "Stage 4"}
    # Smoking must be the MSK missing token "Unknown", never "Not available".
    assert set(panc["SMOKING_PREDICTIONS_3_CLASSES"]) == {"Unknown"}


def test_prompts_have_no_target_leakage(panc: pd.DataFrame):
    # The rendered prompt must never contain the target (structural guard).
    cfg = prompts.load_config("config/data.yaml")
    rendered = prompts.build_prompts(panc, cfg)
    blob = "\n".join(rendered["prompt"]).lower()
    for forbidden in ("os_status", "os_months", "overall survival", "deceased", "living"):
        assert forbidden not in blob, f"leakage: {forbidden!r} in a prompt"


def test_agent_normalization():
    assert normalize_agent("Gemcitabine HCL") == "GEMCITABINE"
    assert normalize_agent("Nabpaclitaxel") == "PACLITAXEL PROTEIN-BOUND"
    assert normalize_agent("Fluorouracil") == "FLUOROURACIL"
    assert normalize_agent("Irinotecan liposome") == "IRINOTECAN LIPOSOMAL"
