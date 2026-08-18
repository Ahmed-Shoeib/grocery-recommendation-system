"""Per-candidate ranking feature vector: user features, item features,
explicit user-item cross features, and the retrieval signal itself.

Deliberately richer AND structurally different from what the Two-Tower
towers consume (`retrieval.two_tower.feature_encoding`): the towers learn
a shared dense embedding space for ANN search from (semantic embedding +
learned category/brand embeddings + coarse numeric features); this
module instead builds explicit, interpretable cross features (does this
candidate match the user's affinity distributions? how does the
retrieval stage itself already rank it?) that a retrieval embedding
can't directly expose, meant to be scored by a plain feature-concatenation
MLP (`ranking.model`) - the classic "learned embeddings for recall, hand
-built features for precision" split in two-stage recommenders.

`stock_quantity`/`is_active` are included as plain numeric/boolean
features (kept AVAILABLE to the ranker), not used to filter or exclude
candidates here - eligibility filtering is the serving pipeline's job,
not the ranker's (`serving.eligibility`, applied both as a hard
pre-retrieval gate and a final lightweight validation since Phase 11,
per docs/data-mapping.md section 5). Since every candidate the ranker
scores is already pre-retrieval-eligible, `is_active` is effectively
constant (`True`) and `stock_quantity` is always `> 0` for every row in
V1 - the feature is still computed/passed through unchanged (no ranker
redesign), it just carries less discriminative signal than before.
"""

from __future__ import annotations

import numpy as np

from recommendation.features.price import price_relative_distance
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures

RANKING_FEATURE_NAMES = [
    # user (7)
    "user_log_purchase_count",
    "user_log_cart_item_count",
    "user_log_search_count",
    "user_log_total_engagement_events",
    "user_has_chatbot_context",
    "user_has_preferred_category",
    "user_has_age_group",
    # item (11) - stock_quantity/is_active kept available, not a filter (Phase 7 does that, at the end).
    # category_relative_price/is_discounted added STEP 6 (docs/data-mapping.md section 15).
    "item_normalized_price",
    "item_discount_fraction",
    "item_log_purchase_count",
    "item_log_cart_add_count",
    "item_log_review_count",
    "item_average_rating",
    "item_has_rating",
    "item_log_stock_quantity",
    "item_is_active",
    "item_category_relative_price",
    "item_is_discounted",
    # user-item cross features (8) - explicit signals the Two-Tower embedding doesn't expose directly.
    # The 3 price ones (STEP 6) mirror the existing category/brand/preferred
    # "affinity match" pattern - a small, interpretable, non-redundant set
    # (docs/data-mapping.md section 15): a raw normalized reference point
    # (user_normalized_typical_price), the derived compatibility distance
    # (price_relative_distance), and a coarse tier match - NOT every
    # possible price-distance formulation.
    "category_affinity_match",
    "brand_affinity_match",
    "preferred_category_match",
    "semantic_cosine_similarity",
    "has_semantic_similarity",
    "user_normalized_typical_price",
    "user_has_price_profile",
    "price_relative_distance",
    "price_tier_match",
    # retrieval signal (2) - what Phase 5's VectorIndex already computed for this candidate
    "retrieval_score",
    "retrieval_rank_normalized",
]


def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> tuple[float, bool]:
    if a is None or b is None:
        return 0.0, False
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0, False
    return float(np.dot(a, b) / (norm_a * norm_b)), True


def build_ranking_feature_vector(
    user_features: UserFeatures,
    product_features: ProductFeatures,
    item_semantic_embedding: np.ndarray | None,
    retrieval_score: float,
    retrieval_rank: int,
    pool_size: int,
    max_price: float,
) -> np.ndarray:
    """`retrieval_rank` is 0-indexed (0 = top of the retrieved list);
    `pool_size` is the number of candidates the VectorIndex was asked to
    retrieve (used only to normalize rank to [0, 1], not to filter).
    """
    semantic_similarity, has_similarity = _cosine_similarity(user_features.semantic_embedding, item_semantic_embedding)

    category_match = user_features.category_affinity.get(product_features.category_name or "", 0.0)
    brand_match = user_features.brand_affinity.get(product_features.brand or "", 0.0)
    preferred_match = (
        1.0
        if user_features.preferred_category is not None
        and user_features.preferred_category == product_features.category_name
        else 0.0
    )
    normalized_price = min(product_features.effective_price / max_price, 1.0) if max_price > 0 else 0.0

    # STEP 6 (docs/data-mapping.md section 15): all degrade to neutral
    # (0.0) when the user has no price profile at all (an old call site
    # that didn't build/pass `price_context` to `build_user_features`) -
    # `user_has_price_profile` is what lets the model tell that apart from
    # a genuine "price matches exactly" (distance=0) case.
    price_profile = user_features.price_profile
    user_normalized_typical_price = 0.0
    price_tier_match = 0.0
    if price_profile is not None:
        if max_price > 0:
            user_normalized_typical_price = min(price_profile.typical_price / max_price, 1.0)
        price_tier_match = 1.0 if price_profile.price_tier == product_features.price_tier else 0.0
    price_distance = price_relative_distance(
        product_features.effective_price, price_profile.typical_price if price_profile is not None else None
    )

    return np.array(
        [
            np.log1p(user_features.purchase_count),
            np.log1p(user_features.cart_item_count),
            np.log1p(user_features.search_count),
            np.log1p(user_features.total_engagement_events),
            1.0 if user_features.has_chatbot_context else 0.0,
            1.0 if user_features.has_preferred_category else 0.0,
            1.0 if user_features.has_age_group else 0.0,
            normalized_price,
            (product_features.discount_percentage or 0.0) / 100.0,
            np.log1p(product_features.purchase_count),
            np.log1p(product_features.cart_add_count),
            np.log1p(product_features.review_count),
            product_features.average_rating if product_features.average_rating is not None else 0.0,
            1.0 if product_features.average_rating is not None else 0.0,
            np.log1p(product_features.stock_quantity),
            1.0 if product_features.is_active else 0.0,
            product_features.category_relative_price,
            1.0 if product_features.is_discounted else 0.0,
            category_match,
            brand_match,
            preferred_match,
            semantic_similarity,
            1.0 if has_similarity else 0.0,
            user_normalized_typical_price,
            1.0 if price_profile is not None else 0.0,
            price_distance,
            price_tier_match,
            retrieval_score,
            retrieval_rank / max(pool_size - 1, 1),
        ],
        dtype=np.float32,
    )
