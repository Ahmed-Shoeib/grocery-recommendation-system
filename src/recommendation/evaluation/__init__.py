"""Offline evaluation: Recall@K, HitRate@K, Precision@K, NDCG@K, MRR,
catalog coverage, and retrieval/end-to-end latency measurement.

`retrieval_metrics.py` (Recall@K, HitRate@K) landed in Phase 4, generic
enough for reuse by Phase 5's ANN evaluation and Phase 6's ranking
evaluation unchanged. The remaining metrics (Precision@K, NDCG@K, MRR,
coverage, latency) land across Phases 5-6. Online metrics (CTR,
impressions, add-to-cart/purchase conversion) require the deferred
event-tracking pipeline and are out of scope for V1 - see
docs/data-mapping.md, section 8.
"""
