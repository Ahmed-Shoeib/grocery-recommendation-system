"""Offline evaluation: Recall@K, HitRate@K, Precision@K, NDCG@K, MRR,
catalog coverage, and retrieval/end-to-end latency measurement.

Implemented across Phases 5-6. Online metrics (CTR, impressions,
add-to-cart/purchase conversion) require the deferred event-tracking
pipeline and are out of scope for V1 - see docs/data-mapping.md, section 8.
"""
