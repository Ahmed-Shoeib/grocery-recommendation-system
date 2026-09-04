"""External DTOs - the backend's HTTP response shapes, nothing more.

These mirror what the live API actually returns (verified by probing every
endpoint). As of the 2026-09-01 probe the published OpenAPI spec declared
request bodies only, no response schemas; a later probe (2026-09-04, see
`ApiUser`) found the spec now also documents `/api/users/{userId}`'s
response (`UserResponseApiResponse` -> `UserResponse`) and it matches the
live payload exactly - still verified against the live response here
rather than trusted blindly, since the other list endpoints remain
undocumented. They are deliberately tolerant: `extra="ignore"` so a
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


class ApiCategoryRef(BaseModel):
    """The nested `category` object inside one `ApiFavoriteCategory` entry -
    the same shape `/api/categories` exposes (slug/name), just nested here
    instead of top-level.
    """

    model_config = _WIRE

    slug: str | None = None
    name: str | None = None


class ApiFavoriteCategory(BaseModel):
    """One entry of `/api/users/{userId}`'s `preferredCategories` array -
    verified live 2026-09-04 to be the backend's `FavoriteCategory` join
    row (`{id, userId, categoryId, category: {...}, addedAt}`), NOT a bare
    category slug/name. Only the nested `category` ref is modeled - the
    join row's own numeric id/timestamp are irrelevant here.
    """

    model_config = _WIRE

    category: ApiCategoryRef | None = None


class ApiUser(BaseModel):
    """`/api/users/{userId}`. Bearer-gated (verified live 2026-09-04 via a
    `POST /api/auth/service/token` client-credentials exchange - see
    docs/data-mapping.md section 19.1/19.8); this integration sends no
    Authorization header, so every call still degrades to a bare profile
    until the backend team decides how the recommender should authenticate
    (best-effort - see `client.BackendApiClient.get_user`).

    Modeled from the now-published `UserResponse` OpenAPI schema + a live
    sample, every field still optional so a bare `{"guid": "<guid>"}` (or a
    401 that yields no body at all) produces a usable, low-signal profile
    rather than raising. `preferred_categories` reflects the *actual* wire
    shape - a list, each entry nesting a `category` object - not the
    singular `preferredCategory`/`preferredCategorySlug` guessed pre-
    verification. `age_group` has **no equivalent field in the live
    schema at all**; it is kept only so a future backend addition needs no
    code change here - never derived/invented (e.g. from `birth_date`),
    per the canonical-schema contract.
    """

    model_config = _WIRE

    guid: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    preferred_categories: list[ApiFavoriteCategory] = Field(default_factory=list)
    age_group: str | None = None

    def first_preferred_category_slug(self) -> str | None:
        for entry in self.preferred_categories:
            if entry.category and entry.category.slug:
                return entry.category.slug
        return None

    def first_preferred_category_name(self) -> str | None:
        for entry in self.preferred_categories:
            if entry.category and entry.category.name:
                return entry.category.name
        return None
