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

## Phase 0

```bash
python scripts/00_make_splits.py   # -> data/interim/splits.json
python scripts/01_make_prompts.py  # -> data/interim/prompts.parquet
pytest                             # split integrity + no-leakage guard
```
