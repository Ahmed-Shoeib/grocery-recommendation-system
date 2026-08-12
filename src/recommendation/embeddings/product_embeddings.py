"""Precomputed, cached, content-hash-validated product embeddings.

`get_or_compute_product_embeddings` is the only entrypoint most callers
need: it loads the on-disk cache if present and still valid for the
current catalog (same model, same product ids, same text content per
product), otherwise encodes once and writes the cache. Training (Phase 4)
and serving never call the Sentence Transformer directly - they read this
cache, satisfying "do not repeatedly encode products during training."

Cache files (`*.npz` + `*.meta.json`) are written under `data/processed/`,
which is gitignored (regenerable artifact, not source).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recommendation.data.schemas.product import Product
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.embeddings.text_builder import build_product_text


@dataclass
class ProductEmbeddingCache:
    model_name: str
    embedding_dim: int
    product_ids: list[int]
    embeddings: np.ndarray  # shape (len(product_ids), embedding_dim), row-aligned with product_ids
    content_hashes: dict[int, str]  # product_id -> hash of the text that produced its embedding

    def as_dict(self) -> dict[int, np.ndarray]:
        return {pid: self.embeddings[i] for i, pid in enumerate(self.product_ids)}

    def get(self, product_id: int) -> np.ndarray | None:
        try:
            return self.embeddings[self.product_ids.index(product_id)]
        except ValueError:
            return None


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compute_product_embeddings(products: list[Product], encoder: SentenceTransformerEncoder) -> ProductEmbeddingCache:
    texts = [build_product_text(p) for p in products]
    vectors = encoder.encode(texts)
    return ProductEmbeddingCache(
        model_name=encoder.model_name,
        embedding_dim=vectors.shape[1] if len(products) else encoder.embedding_dim,
        product_ids=[p.id for p in products],
        embeddings=vectors,
        content_hashes={p.id: _content_hash(t) for p, t in zip(products, texts)},
    )


def save_cache(cache: ProductEmbeddingCache, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=cache.embeddings, product_ids=np.array(cache.product_ids, dtype=np.int64))
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "model_name": cache.model_name,
                "embedding_dim": cache.embedding_dim,
                "content_hashes": {str(pid): h for pid, h in cache.content_hashes.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_cache(path: str | Path) -> ProductEmbeddingCache | None:
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if not path.exists() or not meta_path.exists():
        return None
    data = np.load(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return ProductEmbeddingCache(
        model_name=meta["model_name"],
        embedding_dim=meta["embedding_dim"],
        product_ids=[int(x) for x in data["product_ids"]],
        embeddings=data["embeddings"],
        content_hashes={int(pid): h for pid, h in meta["content_hashes"].items()},
    )


def get_or_compute_product_embeddings(
    products: list[Product],
    encoder: SentenceTransformerEncoder,
    cache_path: str | Path,
    force_recompute: bool = False,
) -> tuple[ProductEmbeddingCache, bool]:
    """Returns (cache, was_recomputed). Cache is considered valid only if
    the model name matches AND every current product's text-content hash
    matches the cached one AND the product id sets match exactly - so a
    changed description, a new product, or a removed product all correctly
    trigger a recompute rather than silently serving stale embeddings.
    """
    current_hashes = {p.id: _content_hash(build_product_text(p)) for p in products}

    if not force_recompute:
        cached = load_cache(cache_path)
        if (
            cached is not None
            and cached.model_name == encoder.model_name
            and set(cached.product_ids) == set(current_hashes.keys())
            and cached.content_hashes == current_hashes
        ):
            return cached, False

    cache = compute_product_embeddings(products, encoder)
    save_cache(cache, cache_path)
    return cache, True
