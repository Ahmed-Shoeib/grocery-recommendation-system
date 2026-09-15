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
    """Models two related but distinct wire shapes with one tolerant DTO
    (`extra="ignore"`, every backend-specific field optional):

    - `GET /api/ai/products` (`AiProductResponse`) - the **authoritative
      `backend_api` product source since the 2026-09-15 atomic switch**
      (docs/data-mapping.md 19.5). Bearer-gated, a flat array (no
      pagination). Carries `productId: int32` (`product_id` below) on
      every row - the stable backend `Product.Id`, now the primary
      resolver key in `loader.load_backend_catalog`. Also carries `slug`,
      `name`, `description`, `price`, `stockQuantity`, `categorySlug`,
      `tags` - but NOT `altText`/`productImageUrl` (always `None` for this
      source; UI-display metadata only, not consumed by features/embeddings).
    - `GET /api/products`/`GET /api/products/{slug}` (`ProductResponse`/
      `ProductSummaryResponse`) - the legacy, public, slug-only shape.
      Still modeled here (rather than a separate DTO) because
      `BackendApiClient.get_product` - a general-purpose, low-risk,
      currently-unused single-lookup helper - still calls the detail
      route. No production `backend_api` data path calls the legacy list
      route anymore. This shape has NO `product_id` (always `None`), NO
      brand, NO sale price/discount, NO `isActive`, NO ingredients -
      verified live, most recently 2026-09-15.

    See `loader.load_backend_catalog` for how the canonical `RawProduct`
    is populated from whichever shape actually arrived.
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
    """Models both the authoritative and legacy `user-activities` row
    shapes with one tolerant DTO:

    - `GET /api/ai/user-activities` (`AiUserActivityResponse`) - the
      **authoritative `backend_api` activity source since the 2026-09-15
      atomic switch** (docs/data-mapping.md 19.5). Bearer-gated,
      cursor-paginated. Carries `productId: int32?` (`product_id` below,
      nullable in the schema but not observed null live) instead of
      `slug` - this shape has **no slug field at all**.
    - `GET /api/user-activities` (`UserActivitiesResponse`) - the legacy,
      public, slug-only shape. No production `backend_api` data path
      calls it anymore; kept modeled here for the shared DTO and for
      tests that exercise the loader's generic id-or-slug resolution
      logic without needing a live Ai-shaped fixture.

    Both shapes share `userId` (GUID) and `actionType` (PascalCase). The
    naive `timestamp` is confirmed by the backend team to be UTC
    wall-clock, not local time (`loader._as_naive_utc` already treated it
    that way; this is now a confirmed contract, not an assumption).
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

    Access status (re-verified live 2026-09-15 with a freshly-minted
    service token): **200 OK**. The decoded token's `scope` claim is now
    `["users:read", "reviews:read", "products:read", "activities:read"]`
    - the previously-missing `reviews:read` grant landed. Live response:
    11 review rows (2026-09-15).

    Identity note - **user side closed 2026-09-15**: the response now
    also carries `userGuid` (verified live, present and non-null on all
    11 rows), the GUID `/api/user-activities`/`/api/users/{guid}` use -
    this is the bridge that was previously missing. `_resolve_review_user`
    now looks it up directly against the activity-stream's GUID->internal-id
    map (still a **lookup, never a mint**: a review by a user with no
    recorded activity is still dropped, matching the eligibility contract
    - see docs/data-mapping.md 19.6). `user_id` (the int32 primary key) is
    kept only as non-authoritative metadata now that `user_guid` is the
    real join key.

    Product side: `product_id` is the backend's int32 `Product.Id`,
    matching `ApiProduct.product_id`/`ApiActivity.product_id`. **Closed
    and live-verified 2026-09-15**: `loader.load_backend_reviews` ->
    `_resolve_review_product` joins against
    `BackendCatalog.product_id_by_backend_id`, which `load_backend_catalog`
    now populates from `GET /api/ai/products` (the authoritative product
    source since the atomic switch - docs/data-mapping.md 19.5) for every
    product in the catalog. See docs/data-mapping.md 19.6 for live join
    counts.

    `rating` is `int32` with no declared bounds on this response, though
    the write side (`CreateProductReviewRequest`) constrains it to 1-5;
    range validation is the loader's job, not the DTO's, so an
    out-of-range row is dropped with a count rather than raising and
    failing the whole load.
    """

    model_config = _WIRE

    review_id: int | None = None
    user_id: int | None = None
    user_guid: str | None = None
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
