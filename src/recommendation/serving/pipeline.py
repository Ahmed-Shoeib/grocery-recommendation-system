"""The full serving pipeline.

Catalog -> HARD PRE-RETRIEVAL ELIGIBILITY -> Two-Tower / VectorIndex
retrieval (eligible products only) -> Neural Ranker -> Re-ranking
-> remaining business rules -> FINAL LIGHTWEIGHT ELIGIBILITY VALIDATION
-> Final Top-N

`generate_recommendations` is a pure function of an already-built
`UserFeatures` (built with full visible history for real serving, or
with held-out val/test excluded for evaluation - see `serving
.evaluation`) plus the catalog/model artifacts - it does no adapter
lookups itself, so it works identically whether called from a live
`recommend()` request or from an offline evaluation loop.

**Pre-retrieval eligibility**: `isActive`/`stockQuantity` are hard,
global catalog-eligibility gates, not model knowledge, so they gate
candidate generation itself rather than filtering results afterward
(docs/data-mapping.md §5). Before any candidate generation -
personalized (Two-Tower/VectorIndex) or fallback (category/global
popularity, cold-start waterfall) - the current eligible product id set
is computed once from `product_features` and threaded through every
candidate source, via `serving.eligibility.apply_eligibility` (the SAME
predicate rules used at the final stage) and `retrieval.index
.eligibility_filter.EligibilityRestrictedIndex` for the VectorIndex path
specifically. This never touches the Two-Tower, the ranker, or the
VectorIndex's built structure - stock/active state changes only require
refreshing `product_features`, never a retrain or index rebuild.

**Final lightweight validation**: `apply_eligibility` runs a SECOND time
at the very end, over the full re-ranked pool, as a defense-in-depth
safety net against a product becoming unavailable between pre-retrieval
filtering and the final response. In the common case this stage excludes
nothing; it exists to make that an enforced invariant, not an assumption.

Both eligibility stages operate on the FULL candidate/re-ranked pool
(not just the top requested N) - `pool_size` is capped by the *eligible*
catalog size - so a late exclusion at the final stage still leaves as
many eligible candidates as possible to fill the requested Top-N from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.schemas.product import Product
from recommendation.features.price import PriceCatalogContext
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures, build_user_features
from recommendation.ranking.features import build_ranking_feature_vector
from recommendation.reranking.candidates import RankedCandidate
from recommendation.reranking.diversity import rerank
from recommendation.retrieval.index.base import VectorIndex
from recommendation.retrieval.index.eligibility_filter import EligibilityRestrictedIndex
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
from recommendation.utils.config import AppConfig, RetrievalConfig
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
    # How many catalog products the hard PRE-RETRIEVAL eligibility gate
    # excluded before candidate generation ever ran (Phase 11) - i.e. how
    # much of the catalog was inactive/out-of-stock at request time. Not
    # to be confused with `num_excluded_by_eligibility` below, which is
    # the separate, final safety-net re-check.
    num_excluded_pre_retrieval: int
    num_before_eligibility: int
    # Exclusions caught by the FINAL lightweight eligibility validation
    # (Phase 11) - expected to be 0 in the common case, since candidate
    # generation is already restricted to pre-retrieval-eligible products;
    # a nonzero value means a product became ineligible between
    # pre-retrieval filtering and this final check (see
    # `final_product_features` on `generate_recommendations`).
    num_excluded_by_eligibility: int
    excluded_reasons: dict[int, list[str]] = field(default_factory=dict)
    # Full ranked pool BEFORE re-ranking/eligibility (ranker-only order for
    # STRONG/SPARSE, pre-dedup fallback order for NO_HISTORY) - lets
    # evaluation isolate exactly what Phase 7's re-ranking + eligibility
    # stages changed, from an otherwise identical run.
    pre_rerank_product_ids: list[int] = field(default_factory=list)
    # Full pool AFTER dedup/diversity but BEFORE FINAL eligibility
    # validation - what the final safety-net check actually acted on.
    # Pre-retrieval filtering means this is normally already
    # all-eligible; kept for observability/inspection parity with Phase 7.
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
    eligible_ids: set[int],
    retrieval_config: RetrievalConfig,
) -> list[RankedCandidate]:
    user_batch = tt_encoder.encode_user_batch([user_features])
    user_embedding = user_tower.predict(user_batch, verbose=0)
    # Hard pre-retrieval eligibility gate (Phase 11): the Two-Tower/
    # VectorIndex retrieval step itself never sees inactive/out-of-stock
    # products - see `retrieval.index.eligibility_filter
    # .EligibilityRestrictedIndex`. This restricts *results*, not the
    # index structure, so no rebuild/retrain is triggered by eligibility
    # changing.
    [result] = EligibilityRestrictedIndex(vector_index, retrieval_config).search(
        user_embedding, k=pool_size, allowed_ids=eligible_ids
    )

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
                user_features, product_features[pid], product_embeddings.get(pid), score, rank, pool_size, tt_encoder.max_price,
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
    all_item_ids: list[int],  # noqa: ARG001 - kept for call-site/interface stability; pool sizing is now driven by the eligible-id count (see body), not the full catalog
    tt_encoder: TwoTowerFeatureEncoder,
    user_tower: tf.keras.Model,
    ranker_model: tf.keras.Model,
    vector_index: VectorIndex,
    config: AppConfig,
    top_n: int,
    final_product_features: dict[int, ProductFeatures] | None = None,
) -> RecommendationResult:
    """`final_product_features` defaults to `product_features` (today's
    only real caller, `recommend()`, has a single catalog-state snapshot
    per request - see the module docstring). Passing a distinct dict lets
    a caller with a fresher read of catalog state (e.g. a future live
    re-fetch right before responding) make the final validation stage
    genuinely re-check against current state rather than trusting
    pre-retrieval filtering - exercised directly in tests.
    """
    if final_product_features is None:
        final_product_features = product_features

    tier = determine_history_tier(user_features.total_engagement_events, config.cold_start)

    # --- Hard pre-retrieval eligibility (Phase 11): compute the current
    # eligible product id set ONCE, from catalog state, before any
    # candidate generation - personalized or fallback - runs. Reuses the
    # exact same rules as the final validation stage below (one policy,
    # applied twice) so pre-retrieval and final eligibility can never
    # silently drift apart. ---
    eligibility_rules = build_eligibility_rules(config.eligibility)
    pre_retrieval = apply_eligibility(list(product_features.keys()), product_features, eligibility_rules)
    eligible_ids = set(pre_retrieval.eligible_ids)
    eligible_product_features = {pid: product_features[pid] for pid in eligible_ids}

    if not eligible_ids:
        # No eligible product exists in the whole catalog - a valid,
        # honest zero-fill result, not an error (requirement: fewer than
        # N eligible products means returning fewer than N).
        logger.warning("no eligible products in catalog for user_id=%s - returning zero recommendations", user_features.user_id)
        return RecommendationResult(
            user_id=user_features.user_id,
            tier=tier,
            product_ids=[],
            scores=[],
            sources=[],
            requested_n=top_n,
            pool_size=0,
            num_excluded_pre_retrieval=len(product_features),
            num_before_eligibility=0,
            num_excluded_by_eligibility=0,
        )

    pool_size = candidate_pool_size(config.retrieval, limit=top_n, catalog_size=len(eligible_ids))

    personalized: list[RankedCandidate] = []
    if tier in (HistoryTier.STRONG, HistoryTier.SPARSE):
        personalized = _personalized_candidates(
            user_features, product_features, product_embeddings, tt_encoder, user_tower, ranker_model, vector_index, pool_size, eligible_ids, config.retrieval
        )

    if tier is HistoryTier.STRONG:
        ranked = personalized[:pool_size]
    elif tier is HistoryTier.SPARSE:
        blend = config.cold_start.sparse_blend
        preferred_ids = category_popularity_ranking(eligible_product_features, user_features.preferred_category)
        global_ids = global_popularity_ranking(eligible_product_features)
        sources = [
            ("personalized", [c.product_id for c in personalized], blend.personalized),
            ("preferred_category", preferred_ids, blend.preferred_category),
            ("popularity", global_ids, blend.popularity),
        ]
        ranked = blend_candidate_lists(sources, pool_size)
    else:  # NO_HISTORY
        preferred_ids = category_popularity_ranking(eligible_product_features, user_features.preferred_category)
        category_pop_ids = category_popularity_ranking(eligible_product_features, top_affinity_category(user_features.category_affinity))
        global_ids = global_popularity_ranking(eligible_product_features)
        source_lookup = {
            "preferred_category": preferred_ids,
            "category_popularity": category_pop_ids,
            "global_popularity": global_ids,
        }
        ordered_sources = [(name, source_lookup[name]) for name in config.cold_start.no_history_fallback_order]
        ranked = waterfall_candidates(ordered_sources, pool_size)

    pre_rerank_ids = [c.product_id for c in ranked]

    reranked = rerank(ranked, product_features, config.reranking)

    # --- Final lightweight eligibility validation (Phase 11): re-checks
    # the SAME rules against `final_product_features`, independently of
    # whether pre-retrieval filtering already guaranteed eligibility -
    # see the module docstring and `generate_recommendations`'s own
    # docstring above. ---
    final_eligibility = apply_eligibility([c.product_id for c in reranked], final_product_features, eligibility_rules)
    eligible_set = set(final_eligibility.eligible_ids)

    final = [c for c in reranked if c.product_id in eligible_set][:top_n]

    result = RecommendationResult(
        user_id=user_features.user_id,
        tier=tier,
        product_ids=[c.product_id for c in final],
        scores=[c.score for c in final],
        sources=[c.source for c in final],
        requested_n=top_n,
        pool_size=pool_size,
        num_excluded_pre_retrieval=len(product_features) - len(eligible_ids),
        num_before_eligibility=len(reranked),
        num_excluded_by_eligibility=len(final_eligibility.excluded_ids),
        excluded_reasons=final_eligibility.excluded_reasons,
        pre_rerank_product_ids=pre_rerank_ids,
        pre_eligibility_product_ids=[c.product_id for c in reranked],
    )

    # Logged HERE (not by each caller) so the API and the dashboard - and
    # any future caller of this function - get identical, non-duplicated
    # observability for free (Phase 10). Deliberately no CTR/conversion
    # metrics: V1 has no event-tracking pipeline to compute those from.
    logger.info(
        "recommendation generated user_id=%s tier=%s requested=%d returned=%d fill_rate=%.2f "
        "pool_size=%d excluded_pre_retrieval=%d excluded_by_eligibility=%d",
        result.user_id, result.tier.value, result.requested_n, len(result.product_ids), result.fill_rate,
        result.pool_size, result.num_excluded_pre_retrieval, result.num_excluded_by_eligibility,
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
    price_context: PriceCatalogContext | None = None,
) -> RecommendationResult:
    """Convenience wrapper matching `serving/__init__.py`'s documented
    `recommend(user_id, limit, context=None)` shape: looks up the user's
    full engagement history (no exclusion - this is live serving, not
    leave-one-out evaluation) and runs `generate_recommendations`.

    Recency (docs/data-mapping.md section 14) is opt-in per
    `build_user_features` call (`features.user_features` docstring) - this
    is the live-request boundary, so it explicitly supplies
    `reference_time=` the current naive-UTC time (wall-clock "now" is
    legitimate here, unlike in offline evaluation or the non-temporal
    training path). Deliberately UTC, not the server process's local
    clock (`datetime.now()`) - `data.sqlite.loader._parse_timestamp`
    parses every `action_time` under the same naive-UTC convention, so a
    fresh `User_events` row (see `api.dependencies.RecommendationService
    .maybe_refresh`) compares correctly against this reference_time
    regardless of which timezone the server machine itself runs in. Using
    the server's local clock here instead would make a genuinely-recent,
    correctly-UTC-timestamped event look "in the future" on any server not
    itself running in UTC, incorrectly raising `features.recency
    .RecencyLeakageError`.

    `price_context` (STEP 6, docs/data-mapping.md section 15) is likewise
    threaded straight through to `build_user_features` - `None` (default)
    leaves `UserFeatures.price_profile` unset for this request, degrading
    gracefully (neutral price features) rather than crashing; the real
    caller (`RecommendationService.recommend`) builds and passes one.
    """
    from datetime import datetime, timezone

    from recommendation.data.adapters.engagement import build_engagement_profile

    profile = build_engagement_profile(
        user_id, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    # Naive UTC "now" - NOT datetime.now() (the server process's local
    # clock) - so this matches data.sqlite.loader._parse_timestamp's
    # naive-UTC parsing convention regardless of server timezone. See this
    # function's docstring.
    now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    user_features = build_user_features(
        profile, product_lookup, product_embeddings, config.features,
        text_embeddings=text_embeddings, reference_time=now_utc_naive, price_context=price_context,
    )
    return generate_recommendations(
        user_features, product_features, product_embeddings, all_item_ids, tt_encoder, user_tower, ranker_model, vector_index, config, limit
    )
