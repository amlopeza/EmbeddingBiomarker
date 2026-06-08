#!/usr/bin/env python
"""Regenerate the patient table from the raw cBioPortal MSK-CHORD dump.

Runs ``embedbiomarker.data.build_patient_table()`` and writes the result to
``data/interim/data_prompts.csv`` — the derived starting point for splits,
prompts and the tabular baseline.

Usage:
    python scripts/00_build_table.py
    python scripts/00_build_table.py --raw-dir DIR --out FILE --dedup-treatments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # run without an editable install

from embedbiomarker.data import DEFAULT_RAW_DIR, build_patient_table

EXPECTED_SHAPE = (23777, 26)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=REPO_ROOT / DEFAULT_RAW_DIR,
        help="directory with the raw cBioPortal MSK-CHORD files",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data/interim/data_prompts.csv",
        help="output CSV path",
    )
    parser.add_argument(
        "--dedup-treatments", action="store_true",
        help="collapse repeated treatment agents per patient (order-preserving)",
    )
    args = parser.parse_args()

    df = build_patient_table(args.raw_dir, dedup_treatments=args.dedup_treatments)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    out = args.out.relative_to(REPO_ROOT)
    print(f"Wrote {out}: {df.shape[0]} patients x {df.shape[1]} columns")
    if df.shape != EXPECTED_SHAPE:
        print(f"  WARNING: expected {EXPECTED_SHAPE}, got {df.shape}")


if __name__ == "__main__":
    main()
