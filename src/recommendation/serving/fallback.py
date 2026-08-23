"""Non-personalized fallback candidate sources and merge strategies for
SPARSE_HISTORY (weighted blend) and NO_HISTORY (ordered waterfall) users
(docs/data-mapping.md section 3).

"Popularity" is all-time purchase/cart volume - this fallback path does
not apply recency/decay, so it does not compute a genuine "trending"
signal (docs/data-mapping.md section 6). `category_popularity_ranking`
is reused for two distinct fallback sources: `"preferred_category"` (the
confirmed `UserProfile
.preferred_category` attribute) and `"category_popularity"` (the user's
single highest-affinity category from `UserFeatures.category_affinity`,
which factors in preferred_category plus whatever sparse purchase/cart/
search/chatbot signal exists - see `features.user_features`). For a true
zero-signal user these two coincide (affinity has nothing else to draw
on); for a SPARSE_HISTORY user with some real signal they can genuinely
differ, which is the point of listing them as separate fallback tiers.
"""

from __future__ import annotations

from recommendation.features.product_features import ProductFeatures
from recommendation.reranking.candidates import RankedCandidate


def _rank_score(rank: int, n: int) -> float:
    """Synthetic descending score for a plain id list with no native
    score of its own, so fallback candidates are comparable to
    ranker-scored personalized candidates in downstream re-ranking.
    """
    return 1.0 if n <= 1 else 1.0 - (rank / (n - 1))


def to_ranked_candidates(product_ids: list[int], source: str) -> list[RankedCandidate]:
    n = len(product_ids)
    return [RankedCandidate(pid, _rank_score(i, n), source) for i, pid in enumerate(product_ids)]


def global_popularity_ranking(product_features: dict[int, ProductFeatures]) -> list[int]:
    ranked = sorted(product_features.values(), key=lambda pf: (-pf.purchase_count, -pf.cart_add_count, pf.product_id))
    return [pf.product_id for pf in ranked]


def category_popularity_ranking(product_features: dict[int, ProductFeatures], category_name: str | None) -> list[int]:
    if category_name is None:
        return []
    in_category = [pf for pf in product_features.values() if pf.category_name == category_name]
    ranked = sorted(in_category, key=lambda pf: (-pf.purchase_count, -pf.cart_add_count, pf.product_id))
    return [pf.product_id for pf in ranked]


def top_affinity_category(category_affinity: dict[str, float]) -> str | None:
    if not category_affinity:
        return None
    return max(category_affinity.items(), key=lambda kv: kv[1])[0]


def blend_candidate_lists(sources: list[tuple[str, list[int], float]], pool_size: int) -> list[RankedCandidate]:
    """Weighted merge for SPARSE_HISTORY: each source contributes
    `weight * (its own best-first normalized rank score)`; a product
    appearing in multiple sources keeps its highest blended score
    (dedup-by-max, not summed - so appearing in two weak sources can't
    artificially outrank a strong single-source signal).
    """
    best: dict[int, RankedCandidate] = {}
    for name, product_ids, weight in sources:
        for candidate in to_ranked_candidates(product_ids, name):
            blended_score = weight * candidate.score
            existing = best.get(candidate.product_id)
            if existing is None or blended_score > existing.score:
                best[candidate.product_id] = RankedCandidate(candidate.product_id, blended_score, name)
    ranked = sorted(best.values(), key=lambda c: -c.score)
    return ranked[:pool_size]


def waterfall_candidates(sources: list[tuple[str, list[int]]], pool_size: int) -> list[RankedCandidate]:
    """Ordered fallback for NO_HISTORY: sources are tried in the
    configured order (`cold_start.no_history_fallback_order`), each
    contributing only NEW ids, until `pool_size` is reached or sources
    are exhausted - a strict priority chain, not a weighted blend.
    """
    seen: set[int] = set()
    result: list[RankedCandidate] = []
    for name, product_ids in sources:
        for candidate in to_ranked_candidates(product_ids, name):
            if candidate.product_id in seen:
                continue
            seen.add(candidate.product_id)
            result.append(candidate)
            if len(result) >= pool_size:
                return result
    return result
