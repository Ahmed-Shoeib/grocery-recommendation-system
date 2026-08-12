"""Generic offline ranking/retrieval metrics: Recall@K, HitRate@K,
Precision@K, NDCG@K, and (Mean) Reciprocal Rank.

Deliberately generic (a ranked id list + a relevant-id set per query) so
Phase 5's ANN retrieval and Phase 6's ranking evaluation share these
unchanged - nothing here is Two-Tower- or ranker-specific, and it's what
lets Phase 6 compare the neural ranker against the raw retrieval-score
ordering baseline on identical terms.

Note: V1's leave-one-out evaluation (docs/data-mapping.md section 3/12,
`retrieval.two_tower.splitting`) holds out exactly one relevant item per
query. With a single relevant item, Recall@K and HitRate@K are
numerically identical per query (both are 0 or 1) and therefore identical
in aggregate too - this is an honest property of the single-target
leave-one-out protocol, not a bug in either metric. Both are still
implemented generally (multi-relevant-item capable) so they diverge
correctly whenever a query has more than one relevant item. The same
holds for Precision@K vs. NDCG@K vs. reciprocal rank at this protocol:
they diverge only in HOW they weight the position of the (single) hit,
which is exactly why reporting more than one is useful even here - NDCG
and MRR are sensitive to rank position within top-K, Precision@K and
Recall@K/HitRate@K are not.
"""

from __future__ import annotations

import math


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


def precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / k


def mean_precision_at_k(rankings: list[list[int]], relevant_sets: list[set[int]], k: int) -> float:
    if not rankings:
        return 0.0
    scores = [precision_at_k(ranked, relevant, k) for ranked, relevant in zip(rankings, relevant_sets)]
    return sum(scores) / len(scores)


def ndcg_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Binary-relevance NDCG@K: DCG using rank-discounted relevance,
    divided by the ideal DCG (all relevant items - up to K of them - at
    the top). 0.0 when there is nothing relevant to rank (IDCG is 0).
    """
    if not relevant_ids or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, item_id in enumerate(ranked_ids[:k]) if item_id in relevant_ids)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg_at_k(rankings: list[list[int]], relevant_sets: list[set[int]], k: int) -> float:
    if not rankings:
        return 0.0
    scores = [ndcg_at_k(ranked, relevant, k) for ranked, relevant in zip(rankings, relevant_sets)]
    return sum(scores) / len(scores)


def reciprocal_rank(ranked_ids: list[int], relevant_ids: set[int]) -> float:
    """1 / (1-indexed rank of the first relevant item), uncapped (searches
    the full ranked list, not just a top-K window) - the standard MRR
    definition. 0.0 if no relevant item appears anywhere in `ranked_ids`.
    """
    for i, item_id in enumerate(ranked_ids):
        if item_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(rankings: list[list[int]], relevant_sets: list[set[int]]) -> float:
    if not rankings:
        return 0.0
    scores = [reciprocal_rank(ranked, relevant) for ranked, relevant in zip(rankings, relevant_sets)]
    return sum(scores) / len(scores)
