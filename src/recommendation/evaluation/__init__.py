"""Offline evaluation: Recall@K, HitRate@K, Precision@K, NDCG@K, MRR,
catalog coverage, and retrieval/end-to-end latency measurement.

`retrieval_metrics.py` (Recall@K, HitRate@K, and - landed in Phase 6 -
Precision@K, NDCG@K, reciprocal rank/MRR) is generic enough for reuse by
Phase 5's ANN evaluation and Phase 6's ranking evaluation unchanged.
`latency.py` (generic timing/percentile report) landed in Phase 5, used
to measure `VectorIndex.search` latency and reusable unchanged for
Phase 8's end-to-end latency. Catalog coverage is deferred - not
requested for V1's ranking evaluation. Online metrics (CTR, impressions,
add-to-cart/purchase conversion) require the deferred event-tracking
pipeline and are out of scope for V1 - see docs/data-mapping.md, section 8.
"""
