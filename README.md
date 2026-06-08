# EmbeddingBiomarker

LLM as a prognostic feature extractor (treatment-aware) on the MSK-CHORD 2024
cohort. A small/efficient language model produces patient embeddings that feed a
survival head (XGBoost-Cox); the emphasis is the treatment-aware analysis by
strata.

## Install

```bash
pip install -e ".[dev]"            # Phase 0: splits, prompts, tabular baseline
pip install -e ".[dev,embeddings]" # Phase 1+: torch / transformers extractors
```

## Data

The raw cBioPortal MSK-CHORD 2024 files must be present under
`data/raw/msk_chord_2024/` (`data_clinical_patient.txt`,
`data_timeline_treatment.txt`, `data_mutations.txt`,
`msk_chord_2024_clinical_data.tsv`). The patient table (`data_prompts.csv`) is a
*derived* artifact regenerated from them by `embedbiomarker.data` — it is not raw.
`data/` is gitignored; everything under it is rebuilt from code.

## Phase 0

```bash
python scripts/00_build_table.py   # raw cBioPortal -> data/interim/data_prompts.csv
python scripts/01_make_splits.py   # -> data/interim/splits.json
python scripts/02_make_prompts.py  # -> data/interim/prompts.parquet
pytest                             # split integrity + no-leakage guard
```
