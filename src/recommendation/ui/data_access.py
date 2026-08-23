"""Pure data-access and formatting helpers - no Streamlit import anywhere
in this module, so every function here is unit-testable without a
Streamlit runtime.

These functions run SERVER-SIDE only (docs/data-mapping.md section 18),
called from `api.routes`'s read-only endpoints (`GET /v1/users`,
`GET /v1/users/{id}/profile`) - the Streamlit dashboard
no longer imports this module or `RecommendationService` at all; it goes
through `ui.api_client.RecommendationApiClient` (HTTP) instead. This
module still never reimplements recommendation/ranking/feature-
engineering logic - it only reads already-computed `RecommendationService`
state and formats it.

`run_recommendations`/`category_distribution`/`source_distribution` (the
dashboard-only helpers this module used to expose) were removed: the live
recommendation call now goes through `api.routes.get_recommendations`
directly (which - as a side effect - fixes a pre-existing inconsistency
where the dashboard's OWN `UserFeatures` never had `reference_time` set,
so its recommendations were built WITHOUT recency weighting even though
the real API path always has); the two distribution helpers are trivial
enough (`Counter` over already-returned response fields) that the
dashboard now computes them client-side from the API response directly,
with no server involvement needed.
"""

from __future__ import annotations

from dataclasses import dataclass

from recommendation.api.dependencies import RecommendationService
from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.product import Product
from recommendation.features.user_features import UserFeatures, build_user_features
from recommendation.serving.cold_start import HistoryTier, determine_history_tier
from recommendation.serving.pipeline import RecommendationResult


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
    docs/data-mapping.md section 3's note).
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
