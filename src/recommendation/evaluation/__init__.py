"""Offline evaluation: Recall@K, HitRate@K, Precision@K, NDCG@K, MRR,
catalog coverage, and retrieval/end-to-end latency measurement.

`retrieval_metrics.py` and `latency.py` are generic (ranked-id-list /
zero-arg-callable based), so both ANN retrieval evaluation and ranking
evaluation reuse them unchanged. Catalog coverage is deferred - not
needed for V1's ranking evaluation. Online metrics (CTR, impressions,
add-to-cart/purchase conversion) require the deferred event-tracking
pipeline and are out of scope for V1 - see docs/data-mapping.md, section 8.
"""
