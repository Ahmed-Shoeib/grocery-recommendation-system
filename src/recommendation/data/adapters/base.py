"""Adapter interfaces.

Every concrete adapter (synthetic in-memory today, a real SQL/API-backed
implementation later) implements one of these and returns the canonical
schemas from `recommendation.data.schemas`. Feature engineering and models
depend only on these interfaces, never on a concrete implementation, so a
real-backend swap is a constructor change, not a model change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from recommendation.data.schemas.engagement import (
    CartAffinityRecord,
    ChatbotContextRecord,
    PurchaseRecord,
    ReviewRecord,
    SearchRecord,
)
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile


class ProductCatalogAdapter(ABC):
    """Backed by ERD entities: Product, Category, Tag, ProductTags."""

    @abstractmethod
    def list_products(self) -> list[Product]: ...

    @abstractmethod
    def get_product(self, product_id: int) -> Product | None: ...


class UserAdapter(ABC):
    """Backed by ERD entity: User (including the confirmed-pending
    `preferredCategory` / `ageGroup` additions).
    """

    @abstractmethod
    def get_user_profile(self, user_id: int) -> UserProfile | None: ...

    @abstractmethod
    def list_user_ids(self) -> list[int]: ...


class PurchaseAdapter(ABC):
    """Backed by ERD entities: Order, OrderItem, Product. The strongest V1
    engagement signal (previous purchases).
    """

    @abstractmethod
    def get_purchases(self, user_id: int) -> list[PurchaseRecord]: ...

    @abstractmethod
    def list_all_purchases(self) -> list[PurchaseRecord]:
        """Catalog-wide purchase records, for aggregate features like
        product popularity. A real backend implementation would run one
        aggregate query here rather than fanning out per user.
        """
        ...


class CartAdapter(ABC):
    """Backed by ERD entities: Cart, CartItem, Product (add-to-cart habit signal)."""

    @abstractmethod
    def get_cart_items(self, user_id: int) -> list[CartAffinityRecord]: ...

    @abstractmethod
    def list_all_cart_items(self) -> list[CartAffinityRecord]:
        """Catalog-wide cart records, for aggregate features."""
        ...


class ReviewAdapter(ABC):
    """Backed by ERD entity: Review (auxiliary ranking signal)."""

    @abstractmethod
    def get_reviews(self, user_id: int) -> list[ReviewRecord]: ...

    @abstractmethod
    def list_all_reviews(self) -> list[ReviewRecord]:
        """Catalog-wide review records, for aggregate rating features."""
        ...


class SearchAdapter(ABC):
    """No backend table exists yet (docs/data-mapping.md section 4). V1
    implementation is synthetic-only; a real backend implementation
    (API/table/event pipeline) implements this same interface.
    """

    @abstractmethod
    def get_search_history(self, user_id: int) -> list[SearchRecord]: ...


class ChatbotContextAdapter(ABC):
    """No backend table exists yet (docs/data-mapping.md section 4). V1
    implementation is synthetic-only; a real backend implementation
    (API/DB entity/event summary) implements this same interface.
    """

    @abstractmethod
    def get_chatbot_context(self, user_id: int) -> ChatbotContextRecord | None: ...


@dataclass
class AdapterBundle:
    """The seven interfaces wired together. Feature engineering, models,
    the API, and the dashboard depend on this type - never on which
    concrete adapters (synthetic in-memory today, a real backend later)
    populated it.
    """

    products: ProductCatalogAdapter
    users: UserAdapter
    purchases: PurchaseAdapter
    cart: CartAdapter
    reviews: ReviewAdapter
    search: SearchAdapter
    chatbot: ChatbotContextAdapter
