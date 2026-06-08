#!/usr/bin/env python
"""Render one leakage-safe prompt per patient and write it to disk.

Reads ``data/interim/data_prompts.csv`` and ``config/data.yaml``, calls
``embedbiomarker.prompts.build_prompts``, and writes
``data/interim/prompts.parquet`` (the input to the Phase 1 extractor grid).

The output carries a ``template_id`` column: the embedding cache key downstream
must be (model_id, template_id, prompt_hash), so the format id travels with the
prompts and never gets lost.

Usage:
    python scripts/02_make_prompts.py
    python scripts/02_make_prompts.py --table FILE --config FILE --out FILE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run without an editable install

from embedbiomarker.prompts import build_prompts, load_config, template_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table", type=Path, default=REPO_ROOT / "data/interim/data_prompts.csv",
        help="patient table produced by 00_build_table.py",
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "config/data.yaml",
        help="YAML with feature / target / prompt spec",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data/interim/prompts.parquet",
        help="output Parquet path",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = pd.read_csv(args.table)

    prompts = build_prompts(df, config)
    prompts["template_id"] = template_id(config)  # travels with the prompts

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompts.to_parquet(args.out, index=False)

    out = args.out.relative_to(REPO_ROOT)
    tid = template_id(config)
    mean_len = int(prompts["prompt"].str.len().mean())
    print(f"Wrote {out}: {len(prompts)} prompts, template_id={tid}, "
          f"mean length {mean_len} chars")


if __name__ == "__main__":
    main()
