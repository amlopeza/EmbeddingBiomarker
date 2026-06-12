#!/usr/bin/env python
"""Phase 1 — extract (and cache) prompt embeddings for one extractor.

Reads ``data/interim/prompts.parquet`` (one leakage-safe prompt per patient) and a
model entry from ``config/models.yaml``, runs the encoder, and writes the embedding
matrix to ``data/processed/embeddings/`` keyed by patient. The forward pass is cached
by (model, template_id, pooling, prompt_hash) so re-runs are instant; only changed
prompts are re-embedded.

Usage:
    python scripts/20_extract_embeddings.py                 # default model: medcpt
    python scripts/20_extract_embeddings.py --model medcpt
    python scripts/20_extract_embeddings.py --model smollm-135m --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embedbiomarker.embeddings import extract_with_cache

ID_COLUMN = "PATIENT_ID"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="medcpt", help="model name in config/models.yaml")
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "data/interim/prompts.parquet")
    parser.add_argument("--models-config", type=Path, default=REPO_ROOT / "config/models.yaml")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/processed/embeddings")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/processed/embeddings")
    parser.add_argument("--batch-size", type=int, default=None, help="override config batch_size")
    parser.add_argument("--max-length", type=int, default=None, help="override config max_length")
    parser.add_argument("--pooling", default=None, help="override config pooling (mean | cls | last)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.models_config.read_text())
    defaults = cfg.get("defaults", {})
    entry = next((m for m in cfg["models"] if m["name"] == args.model), None)
    if entry is None:
        names = [m["name"] for m in cfg["models"]]
        raise SystemExit(f"model {args.model!r} not in models.yaml (have {names})")

    pooling = args.pooling or entry.get("pooling", defaults.get("pooling", "mean"))
    max_length = args.max_length or defaults.get("max_length", 512)
    batch_size = args.batch_size or defaults.get("batch_size", 32)

    prompts = pd.read_parquet(args.prompts)
    template_id = str(prompts["template_id"].iloc[0])
    print(f"Model {args.model} ({entry['hf_id']}) | pooling={pooling} "
          f"max_length={max_length} batch={batch_size}")
    print(f"Prompts: {len(prompts)} | template_id={template_id}")

    t0 = time.time()
    emb = extract_with_cache(
        prompts,
        model_name=args.model,
        hf_id=entry["hf_id"],
        template_id=template_id,
        pooling=pooling,
        max_length=max_length,
        batch_size=batch_size,
        cache_dir=args.cache_dir,
        id_column=ID_COLUMN,
    )
    dim = emb.shape[1] - 1
    out_path = args.out_dir / f"{args.model}__{template_id}__{pooling}__by_patient.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb.to_parquet(out_path, index=False)

    print(f"Embedded -> {dim} dims in {time.time() - t0:.1f}s")
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}  ({len(emb)} patients)")


if __name__ == "__main__":
    main()
