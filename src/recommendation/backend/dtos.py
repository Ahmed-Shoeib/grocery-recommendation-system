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

DTOs never leave this package. `recommendation.backend.loader`
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


class ApiServiceToken(BaseModel):
    """`POST /api/auth/service/token` response payload (inside the usual
    `{success, data}` envelope). Swagger declares the *request*
    (`ServiceTokenRequest: {clientId, clientSecret}`) but documents the 200
    response as bare "OK" with no schema, so this is modeled from the live
    exchange: `{accessToken, expiresAtUtc}`, lifetime ~15 minutes.

    `expires_at_utc` is optional on purpose - if the backend ever stops
    sending it, `auth.ServiceTokenProvider` falls back to a conservative
    fixed TTL rather than treating the token as immortal. This object is
    never logged, never persisted, and never leaves the backend package.
    """

    model_config = _WIRE

    access_token: str
    expires_at_utc: datetime | None = None


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
    includes them. As verified live 2026-09-14, `ProductResponse` /
    `ProductSummaryResponse` (the schemas backing these two routes) still
    expose NO numeric id, NO brand, NO sale price / discount, NO
    `isActive` flag, NO ingredients - see `loader` for how the canonical
    `RawProduct` is populated from what exists.

    `product_id` models the backend's stable `Product.Id` - present on the
    new `AiProductResponse` schema (`GET /api/ai/products`, `productId:
    int32`, verified via the live OpenAPI document 2026-09-14) but NOT on
    `ProductResponse`/`ProductSummaryResponse` itself. It is kept on this
    DTO (rather than a separate one) so `loader.load_backend_catalog` can
    key identity resolution on it the moment a product source populates it
    - `extra="ignore"`/default-`None` means today, with `/api/products`
    the only product source this integration reads, the field is always
    `None` and slug remains the resolution key (see `identity.py`'s
    per-namespace key-scheme guard for what happens the day this flips).
    `/api/ai/products` itself is NOT consumed here: it is Bearer-gated and
    the recommender's service credentials return `403` on it as of
    2026-09-14 (no granted scope) - see docs/data-mapping.md section 19.5.
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
    product_id: int | None = None


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
    naive timestamp - confirmed by the backend team (2026-09-14) to be
    UTC wall-clock, not local time (`loader._as_naive_utc` already treated
    it that way; this is now a confirmed contract, not an assumption).

    `product_id` mirrors `ApiProduct.product_id`: modeled from the new
    `AiUserActivityResponse` schema (`GET /api/ai/user-activities`,
    `productId: int32?`, verified via the live OpenAPI document
    2026-09-14), always `None` on `/api/user-activities` itself (schema
    `UserActivitiesResponse` has no such field, verified live) - kept here
    so `loader.load_backend_events` can prefer it the day this endpoint
    (or its Ai-tagged counterpart) actually populates it.
    """

    model_config = _WIRE

    user_id: str
    action_type: str
    slug: str | None = None
    timestamp: datetime | None = None
    product_id: int | None = None


class ApiReview(BaseModel):
    """One `GET /api/reviews` row (`AiProductReviewResponse` in Swagger,
    tag `AiProductReview` - the endpoint the backend team added for this
    recommender, distinct from the browser-facing
    `/api/products/{slug}/reviews`).

    Access status (re-verified live 2026-09-14 with a freshly-minted
    service token): still **403**. The decoded token's `scope` claim is
    `"users:read"` only - `reviews:read` has not (yet) been granted to
    this integration's actual service-client credentials, despite the
    backend team's report that granting both scopes to a service client
    returns `200` in their own testing. `client.list_reviews()` raises
    `BackendAuthError` and `loader.load_backend_reviews` degrades to `[]`,
    exactly as it does for any other authorization gap - see
    docs/data-mapping.md section 19.6 for the exact evidence and the
    action needed (grant `reviews:read` to *this* client, then re-run the
    smoke test).

    Identity note: `user_id` and `product_id` are the backend's **int32
    primary keys**, while `/api/user-activities`/`/api/users/{guid}`
    address users by GUID and `/api/products` addresses products by slug.
    `loader.load_backend_reviews` -> `_resolve_review_product` now joins
    automatically the moment any consumed product source populates
    `ApiProduct.product_id` (`BackendCatalog.product_id_by_backend_id`);
    the user side still has no equivalent - no endpoint this integration
    can reach exposes a GUID<->int user-id mapping, so
    `_resolve_review_user` remains a lookup restricted to users already
    seen in the activity stream (never a mint). Both gaps are unverified
    against real row *values* until the scope above is granted - the
    contract below is from the live OpenAPI document only.

    `rating` is `int32` with no declared bounds on this response, though
    the write side (`CreateProductReviewRequest`) constrains it to 1-5;
    range validation is the loader's job, not the DTO's, so an
    out-of-range row is dropped with a count rather than raising and
    failing the whole load.
    """

    model_config = _WIRE

    review_id: int | None = None
    user_id: int | None = None
    product_id: int | None = None
    rating: float | None = None
    comment: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


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
