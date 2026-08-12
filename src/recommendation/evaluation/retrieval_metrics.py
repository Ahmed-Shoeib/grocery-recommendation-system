"""Generic offline retrieval metrics: Recall@K and HitRate@K.

Deliberately generic (a ranked id list + a relevant-id set per query) so
Phase 5/6 can reuse these unchanged against ANN retrieval or ranked
candidates - nothing here is Two-Tower-specific.

Note: V1's leave-one-out evaluation (docs/data-mapping.md section 3/12,
`retrieval.two_tower.splitting`) holds out exactly one relevant item per
query. With a single relevant item, Recall@K and HitRate@K are
numerically identical per query (both are 0 or 1) and therefore identical
in aggregate too - this is an honest property of the single-target
leave-one-out protocol, not a bug in either metric. Both are still
implemented generally (multi-relevant-item capable) so they diverge
correctly whenever a query has more than one relevant item.
"""

from __future__ import annotations


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def hit_rate_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


def mean_recall_at_k(rankings: list[list[int]], relevant_sets: list[set[int]], k: int) -> float:
    if not rankings:
        return 0.0
    scores = [recall_at_k(ranked, relevant, k) for ranked, relevant in zip(rankings, relevant_sets)]
    return sum(scores) / len(scores)


def mean_hit_rate_at_k(rankings: list[list[int]], relevant_sets: list[set[int]], k: int) -> float:
    if not rankings:
        return 0.0
    scores = [hit_rate_at_k(ranked, relevant, k) for ranked, relevant in zip(rankings, relevant_sets)]
    return sum(scores) / len(scores)
