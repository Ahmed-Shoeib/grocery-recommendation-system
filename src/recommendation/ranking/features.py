"""Per-candidate ranking feature vector: user features, item features,
explicit user-item cross features, and the retrieval signal itself.

Deliberately richer AND structurally different from what the Two-Tower
towers consume (`retrieval.two_tower.feature_encoding`): the towers learn
a shared dense embedding space for ANN search from (semantic embedding +
learned category embeddings + coarse numeric features); this module
instead builds explicit, interpretable cross features (does this
candidate match the user's affinity distributions? how does the
retrieval stage itself already rank it?) that a retrieval embedding
can't directly expose, meant to be scored by a plain feature-concatenation
MLP (`ranking.model`) - the classic "learned embeddings for recall, hand
-built features for precision" split in two-stage recommenders.

`stock_quantity` is included as a plain numeric feature (kept AVAILABLE to
the ranker), not used to filter or exclude candidates here - eligibility
filtering is the serving pipeline's job, not the ranker's
(`serving.eligibility`, applied both as a hard pre-retrieval gate and a
final lightweight validation - docs/data-mapping.md section 5). Since
every candidate the ranker scores is already pre-retrieval-eligible,
`stock_quantity` is always `> 0` for every row - the feature is still
computed/passed through unchanged, it just carries less discriminative
signal than before the pre-retrieval gate existed.

PRODUCTION-SAFE FEATURE CONTRACT (docs/production-feature-parity-audit.md,
"production-safe feature contract redesign"): the real SQL Server
`Products` table has no `Brand`, `isActive`, `SalePrice`/
`DiscountPercentage`, or `Ingredients` column, and the real `Users` table
has no `AgeGroup` column at all. Five features that depended on those
fields have been REMOVED from `RANKING_FEATURE_NAMES` (29 -> 24 features):
`user_has_age_group`, `item_discount_fraction`, `item_is_active`,
`item_is_discounted`, `brand_affinity_match`. `user_has_preferred_category`/
`preferred_category_match` are RETAINED but reassessed: the real backend
models preferred/favorite categories as a LIST
(`UserProfile.preferred_categories`), so both now mean "has >=1
preferred category" / "candidate's category is one of the user's
preferred categories" rather than a single-scalar match.
`semantic_cosine_similarity`/`has_semantic_similarity`/`retrieval_score`/
`retrieval_rank_normalized` are RETAINED unchanged - their mechanisms are
still valid production features; only the Two-Tower encoder that feeds
them changes (brand/tag/ingredient/parent-category text and inputs
removed - see `embeddings.text_builder` and
`retrieval.two_tower.feature_encoding` module docstrings). They will
become production-safe again once the Two-Tower is retrained on the new,
safe inputs - not before, but the feature slot itself is correct and
should not be deleted.
"""

from __future__ import annotations

import numpy as np

from recommendation.features.price import price_relative_distance
from recommendation.features.product_features import ProductFeatures
from recommendation.features.user_features import UserFeatures

RANKING_FEATURE_NAMES_USER = [
    "user_log_purchase_count",
    "user_log_cart_item_count",
    "user_log_search_count",
    "user_log_total_engagement_events",
    "user_has_chatbot_context",
    "user_has_preferred_category",
]
# stock_quantity kept available, not a filter - eligibility is serving's job (serving.eligibility).
RANKING_FEATURE_NAMES_ITEM_BASE = [
    "item_normalized_price",
    "item_log_purchase_count",
    "item_log_cart_add_count",
    "item_log_review_count",
    "item_average_rating",
    "item_has_rating",
    "item_log_stock_quantity",
]
# Price-aware extras (docs/data-mapping.md section 15). `item_is_discounted`/
# `item_discount_fraction` were removed here (no real SalePrice/
# DiscountPercentage in production) - `item_category_relative_price` stays:
# it is a pure function of Price + CategoryId, both real production fields.
RANKING_FEATURE_NAMES_ITEM_PRICE_EXTRA = ["item_category_relative_price"]
# Explicit signals the Two-Tower embedding doesn't expose directly.
# `brand_affinity_match` was removed (no real Product.Brand in production).
RANKING_FEATURE_NAMES_CROSS_BASE = [
    "category_affinity_match",
    "preferred_category_match",
    "semantic_cosine_similarity",
    "has_semantic_similarity",
]
# Mirrors the existing category/preferred "affinity match" pattern -
# a small, interpretable, non-redundant set (docs/data-mapping.md section
# 15): a raw normalized reference point (user_normalized_typical_price),
# the derived compatibility distance (price_relative_distance), and a
# coarse tier match - not every possible price-distance formulation.
RANKING_FEATURE_NAMES_CROSS_PRICE_EXTRA = [
    "user_normalized_typical_price", "user_has_price_profile", "price_relative_distance", "price_tier_match",
]
# What VectorIndex already computed for this candidate.
RANKING_FEATURE_NAMES_RETRIEVAL = ["retrieval_score", "retrieval_rank_normalized"]

RANKING_FEATURE_NAMES = (
    RANKING_FEATURE_NAMES_USER
    + RANKING_FEATURE_NAMES_ITEM_BASE
    + RANKING_FEATURE_NAMES_ITEM_PRICE_EXTRA
    + RANKING_FEATURE_NAMES_CROSS_BASE
    + RANKING_FEATURE_NAMES_CROSS_PRICE_EXTRA
    + RANKING_FEATURE_NAMES_RETRIEVAL
)


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

    Builds the current 24-entry `RANKING_FEATURE_NAMES` vector (see that
    module-level docstring for the production-safe contract this reflects),
    including the price-aware entries (docs/data-mapping.md section 15).
    """
    semantic_similarity, has_similarity = _cosine_similarity(user_features.semantic_embedding, item_semantic_embedding)

    category_match = user_features.category_affinity.get(product_features.category_name or "", 0.0)
    preferred_match = (
        1.0
        if product_features.category_name is not None
        and product_features.category_name in user_features.preferred_categories
        else 0.0
    )
    normalized_price = min(product_features.effective_price / max_price, 1.0) if max_price > 0 else 0.0

    values = [
        np.log1p(user_features.purchase_count),
        np.log1p(user_features.cart_item_count),
        np.log1p(user_features.search_count),
        np.log1p(user_features.total_engagement_events),
        1.0 if user_features.has_chatbot_context else 0.0,
        1.0 if user_features.has_preferred_category else 0.0,
        normalized_price,
        np.log1p(product_features.purchase_count),
        np.log1p(product_features.cart_add_count),
        np.log1p(product_features.review_count),
        product_features.average_rating if product_features.average_rating is not None else 0.0,
        1.0 if product_features.average_rating is not None else 0.0,
        np.log1p(product_features.stock_quantity),
        product_features.category_relative_price,
    ]

    values += [category_match, preferred_match, semantic_similarity, 1.0 if has_similarity else 0.0]

    # Price-aware cross features (docs/data-mapping.md section 15): all
    # degrade to neutral (0.0) when the user has no price profile at all
    # (a call site that didn't build/pass `price_context` to
    # `build_user_features`) - `user_has_price_profile` is what lets the
    # model tell that apart from a genuine "price matches exactly"
    # (distance=0) case.
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
    values += [user_normalized_typical_price, 1.0 if price_profile is not None else 0.0, price_distance, price_tier_match]

    values += [retrieval_score, retrieval_rank / max(pool_size - 1, 1)]

    return np.array(values, dtype=np.float32)
