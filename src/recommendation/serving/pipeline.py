"""The full V1 serving pipeline (Phase 7):

Two-Tower -> VectorIndex retrieval -> Neural Ranker -> Re-ranking
-> Business Rules / Eligibility -> Final Top-N

`generate_recommendations` is a pure function of an already-built
`UserFeatures` (built with full visible history for real serving, or
with held-out val/test excluded for evaluation - see `serving
.evaluation`) plus the catalog/model artifacts - it does no adapter
lookups itself, so it works identically whether called from a live
`recommend()` request or from an offline evaluation loop.

Eligibility runs LAST, over the FULL re-ranked candidate pool (not just
the top requested N) so that filtering out inactive/out-of-stock items
still leaves as many eligible candidates as possible to fill the
requested Top-N from - the reason retrieval/ranking/re-ranking always
operate on `pool_size` candidates (config-driven, Phase 5's
`candidate_pool_size`), not just N.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.schemas.product import Product
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures, build_user_features
from recommendation.ranking.features import build_ranking_feature_vector
from recommendation.reranking.candidates import RankedCandidate
from recommendation.reranking.diversity import rerank
from recommendation.retrieval.index.base import VectorIndex
from recommendation.retrieval.index.factory import candidate_pool_size
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.serving.cold_start import HistoryTier, determine_history_tier
from recommendation.serving.eligibility import build_eligibility_rules, apply_eligibility
from recommendation.serving.fallback import (
    blend_candidate_lists,
    category_popularity_ranking,
    global_popularity_ranking,
    top_affinity_category,
    waterfall_candidates,
)
from recommendation.utils.config import AppConfig
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RecommendationResult:
    user_id: int
    tier: HistoryTier
    product_ids: list[int]
    scores: list[float]
    sources: list[str]
    requested_n: int
    pool_size: int
    num_before_eligibility: int
    num_excluded_by_eligibility: int
    excluded_reasons: dict[int, list[str]] = field(default_factory=dict)
    # Full ranked pool BEFORE re-ranking/eligibility (ranker-only order for
    # STRONG/SPARSE, pre-dedup fallback order for NO_HISTORY) - lets
    # evaluation isolate exactly what Phase 7's re-ranking + eligibility
    # stages changed, from an otherwise identical run.
    pre_rerank_product_ids: list[int] = field(default_factory=list)
    # Full pool AFTER dedup/diversity but BEFORE eligibility filtering -
    # what business rules actually acted on (used to demonstrate/inspect
    # exactly which candidates isActive/stockQuantity excluded).
    pre_eligibility_product_ids: list[int] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        return len(self.product_ids) / self.requested_n if self.requested_n > 0 else 0.0


def _personalized_candidates(
    user_features: UserFeatures,
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
    tt_encoder: TwoTowerFeatureEncoder,
    user_tower: tf.keras.Model,
    ranker_model: tf.keras.Model,
    vector_index: VectorIndex,
    pool_size: int,
) -> list[RankedCandidate]:
    user_batch = tt_encoder.encode_user_batch([user_features])
    user_embedding = user_tower.predict(user_batch, verbose=0)
    [result] = vector_index.search(user_embedding, k=pool_size)

    # A VectorIndex candidate missing from the current product_features
    # would mean the index and the live catalog have drifted apart (e.g.
    # a Two-Tower trained against a stale product snapshot) - skip it
    # rather than crash the whole request; a corrupt/inconsistent
    # artifact must degrade the result, not the availability, of
    # recommendations for everyone.
    valid = [(pid, score) for pid, score in zip(result.item_ids, result.scores) if pid in product_features]
    if len(valid) < len(result.item_ids):
        logger.warning(
            "VectorIndex returned %d candidate(s) not present in the current product catalog - skipped",
            len(result.item_ids) - len(valid),
        )
    if not valid:
        return []

    feature_rows = np.stack(
        [
            build_ranking_feature_vector(
                user_features, product_features[pid], product_embeddings.get(pid), score, rank, pool_size, tt_encoder.max_price
            )
            for rank, (pid, score) in enumerate(valid)
        ]
    )
    ranker_scores = ranker_model.predict(feature_rows, verbose=0).reshape(-1)
    order = np.argsort(-ranker_scores)
    valid_ids = [pid for pid, _ in valid]
    return [RankedCandidate(valid_ids[i], float(ranker_scores[i]), "personalized") for i in order]


def generate_recommendations(
    user_features: UserFeatures,
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
    all_item_ids: list[int],
    tt_encoder: TwoTowerFeatureEncoder,
    user_tower: tf.keras.Model,
    ranker_model: tf.keras.Model,
    vector_index: VectorIndex,
    config: AppConfig,
    top_n: int,
) -> RecommendationResult:
    tier = determine_history_tier(user_features.total_engagement_events, config.cold_start)
    pool_size = candidate_pool_size(config.retrieval, limit=top_n, catalog_size=len(all_item_ids))

    personalized: list[RankedCandidate] = []
    if tier in (HistoryTier.STRONG, HistoryTier.SPARSE):
        personalized = _personalized_candidates(
            user_features, product_features, product_embeddings, tt_encoder, user_tower, ranker_model, vector_index, pool_size
        )

    if tier is HistoryTier.STRONG:
        ranked = personalized[:pool_size]
    elif tier is HistoryTier.SPARSE:
        blend = config.cold_start.sparse_blend
        preferred_ids = category_popularity_ranking(product_features, user_features.preferred_category)
        global_ids = global_popularity_ranking(product_features)
        sources = [
            ("personalized", [c.product_id for c in personalized], blend.personalized),
            ("preferred_category", preferred_ids, blend.preferred_category),
            ("popularity", global_ids, blend.popularity),
        ]
        ranked = blend_candidate_lists(sources, pool_size)
    else:  # NO_HISTORY
        preferred_ids = category_popularity_ranking(product_features, user_features.preferred_category)
        category_pop_ids = category_popularity_ranking(product_features, top_affinity_category(user_features.category_affinity))
        global_ids = global_popularity_ranking(product_features)
        source_lookup = {
            "preferred_category": preferred_ids,
            "category_popularity": category_pop_ids,
            "global_popularity": global_ids,
        }
        ordered_sources = [(name, source_lookup[name]) for name in config.cold_start.no_history_fallback_order]
        ranked = waterfall_candidates(ordered_sources, pool_size)

    pre_rerank_ids = [c.product_id for c in ranked]

    reranked = rerank(ranked, product_features, config.reranking)
    eligibility_rules = build_eligibility_rules(config.eligibility)
    eligibility_result = apply_eligibility([c.product_id for c in reranked], product_features, eligibility_rules)
    eligible_set = set(eligibility_result.eligible_ids)

    final = [c for c in reranked if c.product_id in eligible_set][:top_n]

    result = RecommendationResult(
        user_id=user_features.user_id,
        tier=tier,
        product_ids=[c.product_id for c in final],
        scores=[c.score for c in final],
        sources=[c.source for c in final],
        requested_n=top_n,
        pool_size=pool_size,
        num_before_eligibility=len(reranked),
        num_excluded_by_eligibility=len(eligibility_result.excluded_ids),
        excluded_reasons=eligibility_result.excluded_reasons,
        pre_rerank_product_ids=pre_rerank_ids,
        pre_eligibility_product_ids=[c.product_id for c in reranked],
    )

    # Logged HERE (not by each caller) so the API and the dashboard - and
    # any future caller of this function - get identical, non-duplicated
    # observability for free (Phase 10). Deliberately no CTR/conversion
    # metrics: V1 has no event-tracking pipeline to compute those from.
    logger.info(
        "recommendation generated user_id=%s tier=%s requested=%d returned=%d fill_rate=%.2f "
        "pool_size=%d excluded_by_eligibility=%d",
        result.user_id, result.tier.value, result.requested_n, len(result.product_ids), result.fill_rate,
        result.pool_size, result.num_excluded_by_eligibility,
    )
    if result.fill_rate < 1.0:
        logger.warning(
            "recommendation under-filled for user_id=%s: requested=%d returned=%d (fill_rate=%.2f) - "
            "the eligible candidate pool did not have enough items to satisfy the request",
            result.user_id, result.requested_n, len(result.product_ids), result.fill_rate,
        )
    return result


def recommend(
    user_id: int,
    bundle: AdapterBundle,
    product_lookup: dict[int, Product],
    product_features: dict[int, ProductFeatures],
    product_embeddings: dict[int, np.ndarray],
    text_embeddings: dict[str, np.ndarray],
    all_item_ids: list[int],
    tt_encoder: TwoTowerFeatureEncoder,
    user_tower: tf.keras.Model,
    ranker_model: tf.keras.Model,
    vector_index: VectorIndex,
    config: AppConfig,
    limit: int,
    context: dict | None = None,  # noqa: ARG001 - extensibility hook, unused by any V1 model (see module docstring)
) -> RecommendationResult:
    """Convenience wrapper matching `serving/__init__.py`'s documented
    `recommend(user_id, limit, context=None)` shape: looks up the user's
    full engagement history (no exclusion - this is live serving, not
    leave-one-out evaluation) and runs `generate_recommendations`.
    """
    from recommendation.data.adapters.engagement import build_engagement_profile

    profile = build_engagement_profile(
        user_id, bundle.users, bundle.purchases, bundle.cart, bundle.search, bundle.chatbot, bundle.reviews
    )
    user_features = build_user_features(profile, product_lookup, product_embeddings, config.features, text_embeddings=text_embeddings)
    return generate_recommendations(
        user_features, product_features, product_embeddings, all_item_ids, tt_encoder, user_tower, ranker_model, vector_index, config, limit
    )
