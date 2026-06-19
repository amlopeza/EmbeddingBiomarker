#!/usr/bin/env python
"""Build the external GENIE BPC cohort table + prompts for frozen-model validation.

Phase 3: it builds the external cohort table and its prompts in one pass (they
share the same cohort):
  1. ``embedbiomarker.external.build_genie_table(cohort)`` -> the 12-feature table
     in MSK-CHORD schema, non-MSK centers only -> ``data/interim/genie_<cohort>.csv``.
  2. ``embedbiomarker.prompts.build_prompts`` with the SAME ``config/data.yaml``
     (template ctx_v1) so the external prompts are apples-to-apples with MSK ->
     ``data/interim/genie_<cohort>_prompts.parquet`` (the Colab extractor input).

The prompts carry the ``template_id`` column so the embedding cache key downstream
stays (model_id, template_id, prompt_hash) — identical contract to Phase 1.

Usage:
    python scripts/40_build_external.py                 # PANC pilot
    python scripts/40_build_external.py --cohort PANC --genie-dir DIR --config FILE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run without an editable install

from embedbiomarker.external import DEFAULT_GENIE_DIR, build_genie_table
from embedbiomarker.prompts import build_prompts, load_config, template_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", default="PANC",
        help="GENIE BPC cohort folder (default: PANC pilot)",
    )
    parser.add_argument(
        "--genie-dir", type=Path, default=REPO_ROOT / DEFAULT_GENIE_DIR,
        help="GENIE BPC root with the per-cohort folders",
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "config/data.yaml",
        help="feature/target/prompt spec (same ctx_v1 as MSK -> apples-to-apples)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "data/interim",
        help="output directory for the table CSV and prompts Parquet",
    )
    args = parser.parse_args()

    # 1. External table (non-MSK, MSK-CHORD schema).
    df = build_genie_table(args.cohort, raw_dir=args.genie_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort = args.cohort.lower()
    table_path = args.out_dir / f"genie_{cohort}.csv"
    df.to_csv(table_path, index=False)

    centers = ", ".join(f"{c}={n}" for c, n in df["CENTER"].value_counts().items())
    event_rate = (df["OS_STATUS"] == "1:DECEASED").mean()
    print(f"Wrote {table_path.relative_to(REPO_ROOT)}: {len(df)} patients "
          f"({centers}); OS event rate {event_rate:.3f}")

    # 2. Prompts with the SAME template as MSK (ctx_v1).
    config = load_config(args.config)
    prompts = build_prompts(df, config)
    prompts["template_id"] = template_id(config)

    prompts_path = args.out_dir / f"genie_{cohort}_prompts.parquet"
    prompts.to_parquet(prompts_path, index=False)

    tid = template_id(config)
    mean_len = int(prompts["prompt"].str.len().mean())
    print(f"Wrote {prompts_path.relative_to(REPO_ROOT)}: {len(prompts)} prompts, "
          f"template_id={tid}, mean length {mean_len} chars")


if __name__ == "__main__":
    main()
