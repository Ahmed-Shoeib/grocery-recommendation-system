"""External DTOs - the backend's HTTP response shapes, nothing more.

These mirror what the live API actually returns (verified by probing every
endpoint - the published OpenAPI spec declares request bodies only, not
response schemas). They are deliberately tolerant: `extra="ignore"` so a
backend-side field addition never breaks ingestion, and every
recommendation-irrelevant field is simply omitted here rather than
modeled.

DTOs never leave this package. `recommendation.data.backend.loader`
translates them into the canonical `Raw*` / `UserInteraction` models that
the rest of the codebase already consumes, so no backend field name
(`categorySlug`, `userId`, ...) or wire convention (camelCase, slug/GUID
identity, the `{success, data}` envelope) reaches feature engineering or
the models.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

_WIRE = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class ApiPagination(BaseModel):
    """Union of the two pagination shapes the backend uses: cursor-based
    (`/api/products`, `/api/categories`, `/api/user-activities`) exposes
    `next_cursor` / `has_next`; page-number-based (`/api/tags`) exposes
    `total_pages` / `has_next`. Only `has_next` + `next_cursor` are read by
    this integration.
    """

    model_config = _WIRE

    has_next: bool = False
    next_cursor: str | None = None
    page_size: int | None = None
    current_page: int | None = None
    total_pages: int | None = None
    total_count: int | None = None


class ApiProduct(BaseModel):
    """`/api/products` (list) and `/api/products/{slug}` (detail). The list
    projection omits `description` and `tags`; the detail projection
    includes them. The backend exposes NO numeric/UUID id, NO brand, NO
    sale price / discount, NO `isActive` flag, NO ingredients - see
    `loader` for how the canonical `RawProduct` is populated from what
    exists.
    """

    model_config = _WIRE

    slug: str
    name: str
    price: float
    stock_quantity: int = 0
    category_slug: str | None = None
    description: str | None = None
    alt_text: str | None = None
    product_image_url: str | None = None
    creation_date: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class ApiCategory(BaseModel):
    """`/api/categories` (list). Exposes slug/name only - NO numeric id and
    NO parent-category reference, so `RawCategory.parent_id` is always
    `None` for this source.
    """

    model_config = _WIRE

    slug: str
    name: str
    image_url: str | None = None
    created_at: datetime | None = None


class ApiActivity(BaseModel):
    """One `/api/user-activities` row: a user GUID, a PascalCase action
    type, an optional product slug (null for actions the backend records
    without resolving a product, e.g. some `RemoveFromCart` rows), and a
    naive local timestamp.
    """

    model_config = _WIRE

    user_id: str
    action_type: str
    slug: str | None = None
    timestamp: datetime | None = None


class ApiUser(BaseModel):
    """`/api/users/{userId}`. Currently auth-protected on the dev backend;
    the backend team has confirmed it will be opened up. Modeled
    defensively from the Swagger request DTOs + register contract - every
    field optional, so a bare `{"id": "<guid>"}` (or a 401 that yields no
    body at all) still produces a usable, low-signal profile rather than
    raising. `preferred_category` / `age_group` are populated ONLY if the
    live response actually carries them - never derived/invented (e.g. from
    `birth_date`), per the canonical-schema contract.
    """

    model_config = _WIRE

    id: str | None = None
    user_id: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    preferred_category: str | None = None
    preferred_category_slug: str | None = None
    age_group: str | None = None

    def guid(self) -> str | None:
        return self.id or self.user_id
