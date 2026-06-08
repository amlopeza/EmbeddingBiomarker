"""Render one leakage-safe survival prompt per patient (the 12-feature context).

Each patient becomes a single text prompt (``prompt_no_question``) built ONLY from
the 12 prognostic context features declared in ``config/data.yaml``. The survival
target (``OS_STATUS``, ``OS_MONTHS``) is excluded by construction: it is never in
the feature list, and ``build_prompts`` refuses to run if a feature column
collides with a target column. The independent string-content guard that scans
the rendered prompts (preamble included) lives in ``tests/test_no_leakage.py``.

Two formats, controlled by the ``prompt`` section of ``config/data.yaml``:
  * ``include_preamble: true`` — a natural-language framing (instruction + context
    lead-in) followed by the features. Instruction-tuned backbones (MedGemma,
    Llama-Instruct) embed better with this than with a flat list. ``style`` picks
    "bulleted" (one feature per line) or "inline" (joined with '. ').
  * ``include_preamble: false`` — the minimal inline format (features joined with
    '. '), for ablations.

Missing values render as ``"Not available"`` (missingness is signal, not dropped)
— consistent with the imputation in ``data.py``.

The ``template_id`` from config identifies the rendered format. It MUST be part of
the embedding cache key (model_id, template_id, prompt_hash) so embeddings from
different prompt formats are never mixed in the extractor grid (Phase 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# The 12 context features (label -> raw column), defined authoritatively in
# config/data.yaml. Listed here for reference only — code reads them from config.
#   1.  Mutations history             -> MUTATIONS
#   2.  Age                           -> CURRENT_AGE_DEID
#   3.  Treatment history             -> TREATMENT_HISTORY
#   4.  HER2                          -> HER2
#   5.  Cancer stage                  -> STAGE_HIGHEST_RECORDED
#   6.  Gender                        -> GENDER
#   7.  Smoking history               -> SMOKING_PREDICTIONS_3_CLASSES
#   8.  History of PDL-1              -> HISTORY_OF_PDL1
#   9.  Fraction Genome altered       -> Fraction_Genome_Altered
#   10. MSI Type                      -> MSI_Type
#   11. Mutation Count                -> Mutation_Count
#   12. Number of tumor diagnoses     -> NUM_ICDO_DX
# OS_STATUS / OS_MONTHS are the TARGET and never enter a prompt.

ID_COLUMN = "PATIENT_ID"

# Defaults used when config has no `prompt` section (minimal inline format).
DEFAULT_PROMPT_CFG = {
    "include_preamble": False,
    "instruction": "",
    "context_lead": "",
    "style": "inline",
    "template_id": "inline_v0",
}


def load_config(config_path: Path | str = "config/data.yaml") -> dict:
    """Load the feature/target spec (single source of truth for prompts)."""
    return yaml.safe_load(Path(config_path).read_text())


def template_id(config: dict) -> str:
    """The prompt format identifier; part of the embedding cache key (Phase 1)."""
    return config.get("prompt", DEFAULT_PROMPT_CFG).get("template_id", DEFAULT_PROMPT_CFG["template_id"])


def _format_value(value: Any, kind: str) -> str:
    """Render one feature value; missing -> 'Not available', whole floats -> int."""
    if pd.isna(value) or str(value).strip() == "":
        return "Not available"
    if kind == "numeric":
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value).strip()


def render_prompt(row: pd.Series, features: list[dict], prompt_cfg: dict | None = None) -> str:
    """Render a single patient's prompt from the ordered feature spec.

    Feature order is deterministic (config order — never shuffled). With a
    preamble, the layout is::

        Instruction: <instruction>
        Context: <context_lead>
         * <Label>: <value>
         ...

    Without a preamble, features are joined inline with '. '. ``style`` selects
    bulleted vs inline rendering of the feature block.
    """
    cfg = {**DEFAULT_PROMPT_CFG, **(prompt_cfg or {})}
    pairs = [
        (feat["label"], _format_value(row[feat["column"]], feat["kind"]))
        for feat in features
    ]

    if cfg["style"] == "bulleted":
        feature_block = "\n".join(f" * {label}: {value}" for label, value in pairs)
    else:  # inline
        feature_block = ". ".join(f"{label}: {value}" for label, value in pairs) + "."

    if not cfg["include_preamble"]:
        # Minimal format: features only (inline regardless of style for ablation).
        return ". ".join(f"{label}: {value}" for label, value in pairs) + "."

    lines = [
        f"Instruction: {cfg['instruction']}",
        f"Context: {cfg['context_lead']}",
        feature_block,
    ]
    return "\n".join(lines)


def build_prompts(df: pd.DataFrame, config: dict, id_column: str = ID_COLUMN) -> pd.DataFrame:
    """Build a ``{id_column, prompt}`` table, one row per patient.

    Raises:
        ValueError: if a feature column is missing from ``df``, or if any feature
            column is also a target column (structural leakage guard — the target
            must never be rendered into a prompt).
    """
    features = config["features"]
    feature_cols = {feat["column"] for feat in features}

    # Early validation: every feature column must exist in the dataframe.
    missing = feature_cols - set(df.columns)
    if missing:
        raise ValueError(f"feature columns not found: {sorted(missing)}")

    # Structural leakage guard: no feature column may be a target column.
    target = config.get("target", {})
    target_cols = {target.get("event_status"), target.get("time_months")} - {None}
    leak = feature_cols & target_cols
    if leak:
        raise ValueError(f"feature columns collide with target columns: {sorted(leak)}")

    prompt_cfg = config.get("prompt", DEFAULT_PROMPT_CFG)
    prompts = df.apply(lambda row: render_prompt(row, features, prompt_cfg), axis=1)
    return pd.DataFrame({id_column: df[id_column].astype(str), "prompt": prompts})
