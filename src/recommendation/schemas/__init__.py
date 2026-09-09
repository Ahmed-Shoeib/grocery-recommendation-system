"""Canonical pydantic schemas shared by every adapter, model, and service
layer, independent of whether the data came from the synthetic generators
or a real backend.
"""

from recommendation.schemas.category import Category
from recommendation.schemas.engagement import (
    CartAffinityRecord,
    ChatbotContextRecord,
    ClickRecord,
    EngagementProfile,
    PurchaseRecord,
    ReviewRecord,
    SearchRecord,
)
from recommendation.schemas.events import ActionType, UserInteraction
from recommendation.schemas.product import Product
from recommendation.schemas.user import UserProfile

__all__ = [
    "Category",
    "Product",
    "UserProfile",
    "ClickRecord",
    "PurchaseRecord",
    "CartAffinityRecord",
    "SearchRecord",
    "ChatbotContextRecord",
    "ReviewRecord",
    "EngagementProfile",
    "ActionType",
    "UserInteraction",
]
