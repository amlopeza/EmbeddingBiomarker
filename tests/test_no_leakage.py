"""Mandatory leakage guard: the survival target must never appear in any prompt.

``OS_STATUS`` / ``OS_MONTHS`` / "Overall Survival" — and the target value tokens
themselves — must not appear in any patient prompt. The forbidden-token list is
defined HERE, independently of ``prompts.py``, so this test is a real adversarial
check on the rendered output, not a mirror of the code it guards.
"""

from pathlib import Path

import pandas as pd
import pytest

from embedbiomarker.prompts import build_prompts, load_config, render_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = REPO_ROOT / "config/data.yaml"
PATIENT_TABLE = REPO_ROOT / "data/interim/data_prompts.csv"

# Any of these appearing (case-insensitively) in a prompt means the target leaked.
FORBIDDEN = ["OS_STATUS", "OS_MONTHS", "OVERALL SURVIVAL", "DECEASED", "LIVING"]


def _leaks(text: str) -> list[str]:
    upper = text.upper()
    return [token for token in FORBIDDEN if token in upper]


def _synthetic_row(config: dict) -> dict:
    """One patient row: a value for every feature + obviously leaky target columns."""
    row = {feat["column"]: (1 if feat["kind"] == "numeric" else "X")
           for feat in config["features"]}
    row["PATIENT_ID"] = "P-TEST"
    row["OS_STATUS"] = "1:DECEASED"   # present in the data, must NOT reach the prompt
    row["OS_MONTHS"] = 42.0
    return row


def test_render_prompt_excludes_target():
    # Render WITH the configured preamble so the scan covers the full prompt
    # (instruction + context lead-in + features), not just the feature block.
    config = load_config(DATA_CONFIG)
    prompt = render_prompt(
        pd.Series(_synthetic_row(config)), config["features"], config.get("prompt")
    )
    assert _leaks(prompt) == [], f"target leaked into prompt: {_leaks(prompt)}"


def test_static_preamble_is_clean():
    # The preamble is static text; it must not carry any target token either.
    config = load_config(DATA_CONFIG)
    preamble = " ".join(
        str(config.get("prompt", {}).get(k, "")) for k in ("instruction", "context_lead")
    )
    assert _leaks(preamble) == [], f"target token in static preamble: {_leaks(preamble)}"


def test_build_prompts_rejects_target_as_feature():
    config = load_config(DATA_CONFIG)
    bad = dict(config)
    bad["features"] = config["features"] + [
        {"label": "Leaky", "column": "OS_STATUS", "kind": "categorical"}
    ]
    df = pd.DataFrame([_synthetic_row(config)])
    with pytest.raises(ValueError):
        build_prompts(df, bad)


@pytest.mark.skipif(not PATIENT_TABLE.exists(), reason="patient table not built yet")
def test_real_prompts_have_no_leakage():
    config = load_config(DATA_CONFIG)
    df = pd.read_csv(PATIENT_TABLE)
    prompts = build_prompts(df, config)
    leaking = {
        pid: _leaks(text)
        for pid, text in zip(prompts["PATIENT_ID"], prompts["prompt"])
        if _leaks(text)
    }
    assert not leaking, f"{len(leaking)} prompts leak the target, e.g. {list(leaking.items())[:3]}"
