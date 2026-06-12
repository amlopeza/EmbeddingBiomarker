"""Off-the-shelf prompt embeddings with an on-disk cache (Phase 1 extractor grid).

Turns the one-prompt-per-patient table (``prompts.parquet``) into a dense matrix
via a HuggingFace encoder (MedCPT, SmolLM, ...). Embeddings are **cached on disk
keyed by (model, template_id, pooling, prompt_hash)** so the expensive forward
pass runs once and is reused across the whole Cox grid — re-running only embeds
prompts whose text changed. The cache key follows the project rule
(model_id, template_id, prompt_hash); pooling is folded into the cache filename so
two poolings of the same model never collide.

CPU-only friendly: forces float32 (fp16 matmuls are slow/unsupported on CPU),
batches, and runs under ``torch.no_grad()``. Pooling is configurable
(mean | cls | last); ``mean`` is a masked mean over real tokens.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "PATIENT_ID"


def prompt_hash(text: str) -> str:
    """Stable content hash of a prompt (part of the embedding cache key)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _pool(last_hidden, attention_mask, pooling: str):
    """Pool token embeddings -> one vector per sequence. Returns a torch tensor."""
    import torch

    if pooling == "cls":
        return last_hidden[:, 0]
    if pooling == "last":
        # last non-pad token per sequence
        lengths = attention_mask.sum(dim=1) - 1
        return last_hidden[torch.arange(last_hidden.size(0)), lengths]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
    raise ValueError(f"unknown pooling {pooling!r} (use mean | cls | last)")


class EmbeddingExtractor:
    """Wraps a HF encoder + tokenizer; ``embed(texts) -> np.ndarray (n, dim)``."""

    def __init__(self, hf_id: str, pooling: str = "mean", max_length: int = 512,
                 batch_size: int = 32, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.hf_id = hf_id
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        # float32 everywhere: CPU fp16 is slow/unsupported; on GPU the encoders
        # here are small enough that fp32 is fine for one-time extraction.
        self.model = AutoModel.from_pretrained(hf_id).to(self.device).eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch

        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            enc = self.tokenizer(
                batch, truncation=True, padding=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            vecs = _pool(hidden, enc["attention_mask"], self.pooling)
            out.append(vecs.float().cpu().numpy())
        return np.vstack(out)


def _cache_file(cache_dir: Path, model_name: str, template_id: str, pooling: str) -> Path:
    return Path(cache_dir) / f"{model_name}__{template_id}__{pooling}.parquet"


def extract_with_cache(
    prompts: pd.DataFrame,
    *,
    model_name: str,
    hf_id: str,
    template_id: str,
    pooling: str = "mean",
    max_length: int = 512,
    batch_size: int = 32,
    cache_dir: Path | str = "data/processed/embeddings",
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """Return ``{id_column, e0..e{d-1}}`` embeddings, embedding only cache misses.

    ``prompts`` must have columns [id_column, "prompt"]. The on-disk cache maps
    prompt_hash -> vector for this (model, template, pooling); only prompts whose
    hash is absent are run through the model, then appended to the cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_file(cache_dir, model_name, template_id, pooling)

    df = prompts[[id_column, "prompt"]].copy()
    df["prompt_hash"] = df["prompt"].map(prompt_hash)

    cache = pd.read_parquet(cache_path) if cache_path.exists() else None
    cached_hashes = set(cache["prompt_hash"]) if cache is not None else set()

    # Unique prompts not yet cached (dedupe identical prompts before the forward pass).
    missing = df.loc[~df["prompt_hash"].isin(cached_hashes), ["prompt_hash", "prompt"]]
    missing = missing.drop_duplicates("prompt_hash")

    if len(missing):
        extractor = EmbeddingExtractor(hf_id, pooling, max_length, batch_size)
        mat = extractor.embed(missing["prompt"].tolist())
        dim = mat.shape[1]
        new = pd.DataFrame(mat, columns=[f"e{i}" for i in range(dim)])
        new.insert(0, "prompt_hash", missing["prompt_hash"].to_numpy())
        cache = new if cache is None else pd.concat([cache, new], ignore_index=True)
        cache.to_parquet(cache_path, index=False)

    # Align cached vectors back to the requested patient order.
    emb_cols = [c for c in cache.columns if c.startswith("e")]
    merged = df.merge(cache, on="prompt_hash", how="left")
    result = merged[[id_column] + emb_cols].copy()
    return result
