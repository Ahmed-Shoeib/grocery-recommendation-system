"""ANN retrieval quality/speed evaluation: how much recall does the
configured approximate `VectorIndex` backend (FAISS HNSW / ScaNN tree+AH,
see `retrieval/index/faiss_index.py`, `scann_index.py`) give up relative
to exact brute-force nearest-neighbor search, and what does it cost in
latency in exchange.

This is deliberately SEPARATE from `scripts/generate_offline_report.py`:
this script measures SEARCH-SYSTEM performance (ANN recall vs. exact,
latency) over the raw item-embedding space, not recommendation quality
(Precision/Recall/NDCG/MRR against real held-out user purchases). Mixing
the two would make it impossible to tell "the ANN approximation is
weak" apart from "the ranker/cold-start/re-ranking logic changed" -
see that script's own report for recommendation-quality metrics.

Reuses the real trained catalog (`models/sqlite_baseline/two_tower/
item_embeddings.npz`, ~1,200 items) as both the corpus AND the query set
(sampled item embeddings as queries) - legitimate here since the goal is
measuring the ANN structure's fidelity to exact search, not personalized
relevance (that's the offline report's job). The exact-search reference
is a plain numpy dot product (embeddings are already L2-normalized, so
this is exact cosine similarity) - the same `brute_force_top_k` pattern
already used in `scripts/build_vector_index.py`/`tests/test_vector_index
.py`, not a second FAISS index, since it's the simplest already-proven
correct reference available.

Usage:
    python scripts/evaluate_ann_recall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from recommendation.api.dependencies import resolve_models_root
from recommendation.evaluation.latency import measure_latency
from recommendation.evaluation.retrieval_metrics import mean_recall_at_k
from recommendation.retrieval.index.embeddings_io import load_item_embeddings
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.utils.config import get_config
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

RECALL_K_VALUES = [10, 20, 50]
NUM_SAMPLE_QUERIES = 200
RECALL_AT_10_WARNING_THRESHOLD = 0.90


def brute_force_top_k(query: np.ndarray, embeddings: np.ndarray, k: int) -> np.ndarray:
    """Row indices (not product ids) of the exact top-k by cosine
    similarity - a plain numpy dot product, since embeddings are already
    L2-normalized.
    """
    scores = embeddings @ query
    return np.argsort(-scores)[:k]


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)

    two_tower_dir = resolve_models_root(config) / "two_tower"
    item_ids, embeddings = load_item_embeddings(two_tower_dir)
    logger.info("Loaded %d item embeddings (dim=%d) from %s", len(item_ids), embeddings.shape[1], two_tower_dir)

    max_k = max(RECALL_K_VALUES)
    rng = np.random.default_rng(config.random_seed)
    sample_rows = rng.choice(len(item_ids), size=min(NUM_SAMPLE_QUERIES, len(item_ids)), replace=False)
    query_embeddings = embeddings[sample_rows]

    print(f"\n=== ANN vs. exact recall: {config.retrieval.backend} backend, {len(sample_rows)} sampled queries, "
          f"{len(item_ids)}-item catalog ===")

    index = build_vector_index(config.retrieval)
    index.build(item_ids, embeddings)

    # One exact top-max_k pass per query, sliced per k below - avoids
    # recomputing the full similarity matrix once per k value.
    exact_top_k_rows = [brute_force_top_k(query_embeddings[i], embeddings, max_k) for i in range(len(sample_rows))]

    ann_results = index.search(query_embeddings, max_k)

    for k in RECALL_K_VALUES:
        ann_rankings = [result.item_ids[:k] for result in ann_results]
        relevant_sets = [{item_ids[row] for row in exact_rows[:k]} for exact_rows in exact_top_k_rows]
        recall = mean_recall_at_k(ann_rankings, relevant_sets, k)
        flag = "" if recall >= RECALL_AT_10_WARNING_THRESHOLD else "  <-- below tuning threshold, consider raising efSearch/leaves_to_search"
        print(f"  Recall@{k:<3} vs. exact = {recall:.4f}{flag if k == 10 else ''}")

    # --- Latency: ANN backend vs. exact brute-force numpy, single query and batch ---
    single_query = query_embeddings[0:1]
    batch_query = query_embeddings[: min(32, len(query_embeddings))]
    eval_k = 10

    ann_single = measure_latency(lambda: index.search(single_query, eval_k), num_runs=200)
    ann_batch = measure_latency(lambda: index.search(batch_query, eval_k), num_runs=200)
    exact_single = measure_latency(lambda: brute_force_top_k(single_query[0], embeddings, eval_k), num_runs=200)
    exact_batch = measure_latency(
        lambda: [brute_force_top_k(q, embeddings, eval_k) for q in batch_query], num_runs=200
    )

    print(f"\n=== Retrieval latency (top-{eval_k}) ===")
    print(f"  ANN single query   : mean={ann_single.mean_ms:.4f}ms  p50={ann_single.p50_ms:.4f}ms  p95={ann_single.p95_ms:.4f}ms")
    print(f"  Exact single query : mean={exact_single.mean_ms:.4f}ms  p50={exact_single.p50_ms:.4f}ms  p95={exact_single.p95_ms:.4f}ms")
    print(f"  ANN batch of {batch_query.shape[0]:<3}  : mean={ann_batch.mean_ms:.4f}ms  p50={ann_batch.p50_ms:.4f}ms  p95={ann_batch.p95_ms:.4f}ms")
    print(f"  Exact batch of {batch_query.shape[0]:<3}: mean={exact_batch.mean_ms:.4f}ms  p50={exact_batch.p50_ms:.4f}ms  p95={exact_batch.p95_ms:.4f}ms")

    print(
        "\nNote: recall/latency above are SEARCH-SYSTEM performance only - they say nothing about "
        "recommendation quality (Precision/Recall/NDCG/MRR against real held-out purchases). "
        "See scripts/generate_offline_report.py for that, and do not conflate the two."
    )


if __name__ == "__main__":
    main()
