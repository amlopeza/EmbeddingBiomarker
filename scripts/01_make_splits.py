#!/usr/bin/env python
"""Compute the frozen, seeded train/val/test split and write it to disk.

Reads ``data/interim/data_prompts.csv``, builds one composite stratum label per
patient (``CANCER_TYPE x OS_STATUS`` by default), calls
``embedbiomarker.splits.make_splits``, and writes ``data/interim/splits.json``.
Split config (seed, ratios, stratify keys) comes from ``config/survival.yaml``.

Usage:
    python scripts/01_make_splits.py
    python scripts/01_make_splits.py --table FILE --config FILE --out FILE
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run without an editable install

from embedbiomarker.splits import make_splits

ID_COLUMN = "PATIENT_ID"
# Below this stratum size, an 80/10/10 cut can leave val or test empty.
SMALL_STRATUM = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table", type=Path, default=REPO_ROOT / "data/interim/data_prompts.csv",
        help="patient table produced by 00_build_table.py",
    )
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "config/survival.yaml",
        help="YAML with seed / ratios / stratify keys",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data/interim/splits.json",
        help="output JSON path",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = cfg["seed"]
    split_cfg = cfg["split"]
    ratios = (split_cfg["train"], split_cfg["val"], split_cfg["test"])
    stratify = split_cfg.get("stratify", False)
    keys = split_cfg.get("stratify_keys", [])

    df = pd.read_csv(args.table)
    ids = df[ID_COLUMN].astype(str).tolist()

    strata = None
    if stratify:
        # make_splits wants ONE label per patient, so fold the stratify_keys
        # columns into a single composite label, e.g. "Breast Cancer|1:DECEASED".
        labels = df[keys].astype(str).agg("|".join, axis=1)
        strata = dict(zip(ids, labels))

        small = {lab: n for lab, n in Counter(labels).items() if n < SMALL_STRATUM}
        if small:
            print(f"WARNING: {len(small)} stratum(s) below {SMALL_STRATUM} patients "
                  f"(val/test may be empty there):")
            for lab, n in sorted(small.items(), key=lambda kv: kv[1]):
                print(f"  {lab}: {n}")

    splits = make_splits(ids, seed=seed, ratios=ratios, strata=strata)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2))

    n = len(ids)
    out = args.out.relative_to(REPO_ROOT)
    strat_note = f"stratified by {' x '.join(keys)}" if stratify else "unstratified"
    print(f"Wrote {out}: {strat_note}, seed={seed}")
    for part in ("train", "val", "test"):
        k = len(splits[part])
        print(f"  {part:5s} {k:6d}  ({k / n:.1%})")


if __name__ == "__main__":
    main()
