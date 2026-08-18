"""Pure data-access and formatting helpers for the dashboard - no
Streamlit import anywhere in this module, so every function here is
unit-testable without a Streamlit runtime. `dashboard.py` calls these
and only handles rendering (`st.*` calls); this module never calls the
Two-Tower/ranker itself except via `serving.pipeline.generate
_recommendations` (through `run_recommendations`) - no recommendation
logic is reimplemented here, only read/formatted.
"""

from __future__ import annotations

from dataclasses import dataclass

from recommendation.api.dependencies import RecommendationService
from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.product import Product
from recommendation.features.user_features import UserFeatures, build_user_features
from recommendation.serving.cold_start import HistoryTier, determine_history_tier
from recommendation.serving.pipeline import RecommendationResult, generate_recommendations


@dataclass
class UserListRow:
    user_id: int
    preferred_category: str | None
    age_group: str | None


def list_users(service: RecommendationService) -> list[UserListRow]:
    rows = []
    for user_id in sorted(service.bundle.users.list_user_ids()):
        profile = service.bundle.users.get_user_profile(user_id)
        rows.append(
            UserListRow(
                user_id=user_id,
                preferred_category=profile.preferred_category if profile else None,
                age_group=profile.age_group if profile else None,
            )
        )
    return rows


@dataclass
class UserDetail:
    user_id: int
    preferred_category: str | None
    age_group: str | None
    engagement: EngagementProfile
    features: UserFeatures
    tier: HistoryTier


def load_user_detail(service: RecommendationService, user_id: int) -> UserDetail | None:
    """Returns None for a genuinely unknown user_id (no `UserProfile`) -
    the same distinction `api.errors.UnknownUserError` draws (see
    docs/data-mapping.md section 3's Phase 8 note).
    """
    profile = service.bundle.users.get_user_profile(user_id)
    if profile is None:
        return None

    engagement = service.engagement_profiles.get(user_id) or build_engagement_profile(
        user_id, service.bundle.users, service.bundle.purchases, service.bundle.cart, service.bundle.clicks,
        service.bundle.search, service.bundle.chatbot, service.bundle.reviews
    )
    features = build_user_features(
        engagement, service.product_lookup, service.product_embeddings, service.config.features,
        text_embeddings=service.text_embeddings, price_context=service.price_context,
    )
    tier = determine_history_tier(features.total_engagement_events, service.config.cold_start)
    return UserDetail(
        user_id=user_id,
        preferred_category=profile.preferred_category,
        age_group=profile.age_group,
        engagement=engagement,
        features=features,
        tier=tier,
    )


def run_recommendations(service: RecommendationService, detail: UserDetail, top_n: int) -> RecommendationResult:
    return generate_recommendations(
        detail.features,
        service.product_features,
        service.product_embeddings,
        service.all_item_ids,
        service.tt_encoder,
        service.user_tower,
        service.ranker_model,
        service.vector_index,
        service.config,
        top_n,
    )


# --- formatting: engagement signals -----------------------------------------

def format_clicks(engagement: EngagementProfile, product_lookup: dict[int, Product]) -> list[dict]:
    return [
        {
            "product_id": c.product_id,
            "product_name": product_lookup[c.product_id].name if c.product_id in product_lookup else "(unknown product)",
        }
        for c in engagement.clicks
    ]


def format_purchases(engagement: EngagementProfile, product_lookup: dict[int, Product]) -> list[dict]:
    return [
        {
            "product_id": p.product_id,
            "product_name": product_lookup[p.product_id].name if p.product_id in product_lookup else "(unknown product)",
            "quantity": p.quantity,
            "unit_price": p.unit_price,
            "order_id": p.order_id,
            "order_status": p.order_status,
        }
        for p in engagement.purchases
    ]


def format_cart_items(engagement: EngagementProfile, product_lookup: dict[int, Product]) -> list[dict]:
    return [
        {
            "product_id": c.product_id,
            "product_name": product_lookup[c.product_id].name if c.product_id in product_lookup else "(unknown product)",
            "quantity": c.quantity,
        }
        for c in engagement.cart_items
    ]


def format_searches(engagement: EngagementProfile, product_lookup: dict[int, Product]) -> list[dict]:
    return [
        {
            "search_term": s.search_term,
            "matched_product": (
                product_lookup[s.matched_product_id].name
                if s.matched_product_id is not None and s.matched_product_id in product_lookup
                else None
            ),
        }
        for s in engagement.searches
    ]


def format_chatbot_context(engagement: EngagementProfile, product_lookup: dict[int, Product]) -> dict | None:
    chatbot = engagement.chatbot_context
    if chatbot is None:
        return None
    return {
        "summary": chatbot.summary,
        "preferred_category": chatbot.preferred_category,
        "product_interest": chatbot.product_interest,
        "keywords": ", ".join(chatbot.keywords) if chatbot.keywords else None,
        "mentioned_products": [
            product_lookup[pid].name if pid in product_lookup else f"#{pid}" for pid in chatbot.mentioned_product_ids
        ],
    }


# --- formatting: recommendations --------------------------------------------

def format_recommendation_table(result: RecommendationResult, product_lookup: dict[int, Product]) -> list[dict]:
    rows = []
    for rank, (pid, score, source) in enumerate(zip(result.product_ids, result.scores, result.sources), start=1):
        product = product_lookup.get(pid)
        rows.append(
            {
                "rank": rank,
                "product_id": pid,
                "score": round(score, 4),
                "source": source,
                "name": product.name if product else "(unknown product)",
                "category": product.category_name if product else None,
                "brand": product.brand if product else None,
                "price": product.price if product else None,
                "is_active": product.is_active if product else None,
                "stock_quantity": product.stock_quantity if product else None,
            }
        )
    return rows


def category_distribution(product_ids: list[int], product_lookup: dict[int, Product]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pid in product_ids:
        product = product_lookup.get(pid)
        category = product.category_name if product and product.category_name else "(uncategorized)"
        counts[category] = counts.get(category, 0) + 1
    return counts


def source_distribution(result: RecommendationResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in result.sources:
        counts[source] = counts.get(source, 0) + 1
    return counts
