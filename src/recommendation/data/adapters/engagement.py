"""Assembles the canonical engagement layer from the individual adapters.

This is the "Canonical Engagement Model" box in the architecture diagram:
the single place that combines previous purchases, add-to-cart behavior,
searched items, chatbot context, the user profile, and reviews into one
`EngagementProfile`. Feature engineering (Phase 3) consumes
`EngagementProfile`, never the individual adapters directly.
"""

from __future__ import annotations

from recommendation.data.adapters.base import (
    CartAdapter,
    ChatbotContextAdapter,
    ClickAdapter,
    PurchaseAdapter,
    ReviewAdapter,
    SearchAdapter,
    UserAdapter,
)
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.schemas.user import UserProfile


def build_engagement_profile(
    user_id: int,
    user_adapter: UserAdapter,
    purchase_adapter: PurchaseAdapter,
    cart_adapter: CartAdapter,
    click_adapter: ClickAdapter,
    search_adapter: SearchAdapter,
    chatbot_adapter: ChatbotContextAdapter,
    review_adapter: ReviewAdapter,
) -> EngagementProfile:
    profile = user_adapter.get_user_profile(user_id) or UserProfile(user_id=user_id)
    return EngagementProfile(
        user_id=user_id,
        profile=profile,
        clicks=click_adapter.get_clicks(user_id),
        purchases=purchase_adapter.get_purchases(user_id),
        cart_items=cart_adapter.get_cart_items(user_id),
        searches=search_adapter.get_search_history(user_id),
        chatbot_context=chatbot_adapter.get_chatbot_context(user_id),
        reviews=review_adapter.get_reviews(user_id),
    )
