"""Deduplication and diversity re-ranking (Phase 7).

`deduplicate` removes repeated product ids (a SPARSE-tier blend of
personalized + fallback candidate lists can easily surface the same
product from more than one source - see `serving.fallback
.blend_candidate_lists`, which already dedupes across sources, but
`rerank` deduplicates unconditionally so ANY candidate list, blended or
not, is safe to pass through here).

`apply_diversity` is a greedy MMR-style pass: at each step, pick the
remaining candidate maximizing `score - diversity_strength *
(repetition penalty for categories/brands already selected)`. This is a
continuous trade-off, not a hard per-category/brand quota - a candidate
is never excluded outright for lacking diversity, only reordered, so it
can't "over-diversify" a list into irrelevance; at
`diversity_strength=0` it exactly reproduces input order (ties broken by
original position, since the scan keeps the first-seen maximum).
"""

from __future__ import annotations

from recommendation.features.product_features import ProductFeatures
from recommendation.reranking.candidates import RankedCandidate
from recommendation.utils.config import ReRankingConfig


def deduplicate(candidates: list[RankedCandidate]) -> list[RankedCandidate]:
    seen: set[int] = set()
    deduped = []
    for c in candidates:
        if c.product_id in seen:
            continue
        seen.add(c.product_id)
        deduped.append(c)
    return deduped


def apply_diversity(
    candidates: list[RankedCandidate],
    product_features: dict[int, ProductFeatures],
    config: ReRankingConfig,
) -> list[RankedCandidate]:
    remaining = list(candidates)
    selected: list[RankedCandidate] = []
    category_counts: dict[str, int] = {}
    brand_counts: dict[str, int] = {}

    while remaining:
        best_index = 0
        best_adjusted = float("-inf")
        for i, c in enumerate(remaining):
            pf = product_features.get(c.product_id)
            category = pf.category_name if pf else None
            brand = pf.brand if pf else None
            penalty = config.diversity_strength * (
                category_counts.get(category, 0) * config.category_repetition_penalty
                + brand_counts.get(brand, 0) * config.brand_repetition_penalty
            )
            adjusted = c.score - penalty
            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = i

        chosen = remaining.pop(best_index)
        selected.append(chosen)
        pf = product_features.get(chosen.product_id)
        if pf and pf.category_name:
            category_counts[pf.category_name] = category_counts.get(pf.category_name, 0) + 1
        if pf and pf.brand:
            brand_counts[pf.brand] = brand_counts.get(pf.brand, 0) + 1

    return selected


def rerank(
    candidates: list[RankedCandidate],
    product_features: dict[int, ProductFeatures],
    config: ReRankingConfig,
) -> list[RankedCandidate]:
    return apply_diversity(deduplicate(candidates), product_features, config)
