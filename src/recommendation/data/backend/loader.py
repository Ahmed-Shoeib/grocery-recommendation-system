"""Backend DTOs -> canonical `Raw*` / `UserInteraction` models.

The exact counterpart of `recommendation.data.sqlite.loader`: it produces
the SAME models the synthetic generator and the SQLite loader produce, so
`adapters.backend_factory.build_backend_api_adapters` can hand them
straight to the existing `InMemoryProductCatalogAdapter` /
`InMemoryUserAdapter` / `InMemoryReviewAdapter` / `UserEventsAdapter`
without a single new adapter class.

All slug/GUID -> int translation goes through `ExternalIdentityResolver`.
Catalog objects (products, categories) are *assigned* ids; cross
references inside the activity stream are *looked up* only - an activity
that names a product slug absent from the catalog is dropped (counted +
logged), never allowed to mint a phantom product id or attach to the
wrong one (docs/data-mapping.md section 19, and the eligibility/data
validation contract in section 5).

Field-availability vs the ERD-backed paths (verified against the live API
2026-09-01): the backend product projection has NO brand, sale price,
discount, `isActive` flag, ingredients, parent-category link, or tag list,
and NO numeric ids anywhere. The canonical `Raw*` models keep those fields
optional / defaulted, so the mapping below is lossy-but-valid rather than
a schema change. See each `# backend gap:` note.
"""

from __future__ import annotations

from collections import Counter

from recommendation.data.backend.auth import ENV_CLIENT_ID, ENV_CLIENT_SECRET
from recommendation.data.backend.client import BackendApiClient
from recommendation.data.backend.dtos import ApiActivity, ApiReview
from recommendation.data.backend.errors import BackendAuthError, BackendCredentialsError
from recommendation.data.backend.identity import ExternalIdentityResolver
from recommendation.data.backend.mapping import is_known, map_action_type
from recommendation.data.schemas.events import UserInteraction
from recommendation.data.synthetic.raw_schemas import (
    RawCategory,
    RawProduct,
    RawProductTag,
    RawReview,
    RawTag,
    RawUser,
)
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)

# If the first N per-user profile fetches all fail auth, stop trying (the
# endpoint is still gated) rather than emit one 401 per active user.
_ENRICH_AUTH_PROBE_LIMIT = 3


class BackendCatalog:
    """The catalog half of a backend load (products + categories + the
    empty tag join). Kept together because the product mapping needs the
    category slug->id map the category load produced.
    """

    def __init__(
        self,
        categories: list[RawCategory],
        products: list[RawProduct],
        tags: list[RawTag],
        product_tags: list[RawProductTag],
        category_id_by_slug: dict[str, int],
        category_id_by_name: dict[str, int],
        product_id_by_backend_id: dict[int, int] | None = None,
    ) -> None:
        self.categories = categories
        self.products = products
        self.tags = tags
        self.product_tags = product_tags
        self.category_id_by_slug = category_id_by_slug
        self.category_id_by_name = category_id_by_name
        self.product_slugs = {p.slug for p in products}
        # Backend int32 product id -> internal product id. Empty for now:
        # `/api/products` exposes no numeric id, so nothing can populate it.
        # `_resolve_review_product` reads it, which is what makes
        # `/api/reviews` (keyed by that int id) resolvable the moment the
        # backend's in-progress product-id work lands - see 19.5/19.6.
        self.product_id_by_backend_id = dict(product_id_by_backend_id or {})


def load_backend_catalog(client: BackendApiClient, resolver: ExternalIdentityResolver) -> BackendCatalog:
    api_categories = client.list_categories()
    raw_categories: list[RawCategory] = []
    cat_id_by_slug: dict[str, int] = {}
    cat_id_by_name: dict[str, int] = {}
    for c in api_categories:
        cid = resolver.resolve_category(c.slug)
        # backend gap: /api/categories exposes no parent reference.
        raw_categories.append(RawCategory(id=cid, name=c.name, parent_id=None))
        cat_id_by_slug[c.slug] = cid
        cat_id_by_name.setdefault(c.name, cid)

    api_products = client.list_products()
    raw_products: list[RawProduct] = []
    clamped_price = clamped_stock = 0
    for p in api_products:
        price = p.price
        if price is None or price <= 0:
            price = 0.01
            clamped_price += 1
        stock = p.stock_quantity if p.stock_quantity and p.stock_quantity > 0 else 0
        if p.stock_quantity is not None and p.stock_quantity < 0:
            clamped_stock += 1
        raw_products.append(
            RawProduct(
                id=resolver.resolve_product(p.slug),
                # backend gap: category_slug may be a placeholder ("string")
                # not present in /api/categories -> category_id 0, which
                # InMemoryProductCatalogAdapter treats as "no category".
                category_id=cat_id_by_slug.get(p.category_slug or "", 0),
                slug=p.slug,
                name=p.name,
                description=p.description,
                brand=None,               # backend gap: no brand field
                price=price,
                sale_price=None,           # backend gap: no sale price
                discount_percentage=None,  # backend gap: no discount
                stock_quantity=stock,
                ingredients=None,          # backend gap: no ingredients
                is_active=True,            # backend gap: no isActive flag - assume active, stock gates eligibility
                product_image=p.product_image_url,
                alt_text=p.alt_text,
            )
        )
    if clamped_price:
        logger.warning("backend load: %d product(s) had non-positive price, clamped to 0.01", clamped_price)
    if clamped_stock:
        logger.warning("backend load: %d product(s) had negative stock, clamped to 0", clamped_stock)

    with_tags = sum(1 for p in api_products if p.tags)
    logger.info(
        "backend catalog loaded: %d categories, %d products (%d carry list-level tags, not consumed)",
        len(raw_categories), len(raw_products), with_tags,
    )
    # The RawTag / RawProductTag join is left empty, so product `tags` never
    # reach the Sentence Transformer text for this source.
    #
    # Note (2026-09-09): the live dev backend DOES now return `tags` on the
    # list projection, but the backend team has stated the production
    # backend will not. Wiring them in would change every product embedding
    # and therefore invalidate the current trained artifacts, so tags stay
    # unconsumed pending the real-backend retraining decision - see
    # docs/data-mapping.md section 19.12, which also records the
    # dev-vs-production discrepancy for the backend team to reconcile.
    return BackendCatalog(raw_categories, raw_products, [], [], cat_id_by_slug, cat_id_by_name)


def load_backend_events(
    activities: list[ApiActivity], resolver: ExternalIdentityResolver, catalog: BackendCatalog
) -> tuple[list[UserInteraction], dict[int, str]]:
    """Map `/api/user-activities` rows to canonical `UserInteraction`s.

    Dropped (counted + logged, never silently mis-signalled):
    - rows whose `actionType` maps to no canonical signal (favorites,
      cart/favorite removals -> known-ignored; anything else -> unknown);
    - rows with a null/blank product slug (the backend records some
      actions without resolving a product);
    - rows whose product slug is not in the current catalog (deleted /
      unknown product - matches the eligibility contract: an unknown
      external id must never resolve to *a* product).
    """
    interactions: list[UserInteraction] = []
    guid_by_internal: dict[int, str] = {}
    dropped_action = Counter()
    dropped_no_slug = 0
    dropped_unknown_product = 0
    unknown_action_values: set[str] = set()

    for row in activities:
        canonical = map_action_type(row.action_type)
        if canonical is None:
            dropped_action[row.action_type] += 1
            if not is_known(row.action_type):
                unknown_action_values.add(row.action_type)
            continue
        if not row.slug:
            dropped_no_slug += 1
            continue
        product_id = resolver.peek_product(row.slug)
        if product_id is None or row.slug not in catalog.product_slugs:
            dropped_unknown_product += 1
            continue
        user_id = resolver.resolve_user(row.user_id)
        guid_by_internal.setdefault(user_id, row.user_id)
        interactions.append(
            UserInteraction(
                user_id=user_id,
                product_id=product_id,
                action_type=canonical,
                # naive backend timestamp is treated as UTC wall-clock,
                # matching data.sqlite.loader._parse_timestamp / the
                # reference_time convention in serving.pipeline.
                action_time=_as_naive_utc(row.timestamp),
            )
        )

    if unknown_action_values:
        logger.warning("backend load: unknown activity actionType value(s) ignored: %s", sorted(unknown_action_values))
    if dropped_action:
        logger.info("backend load: %d activity row(s) ignored by action-type policy: %s", sum(dropped_action.values()), dict(dropped_action))
    if dropped_no_slug:
        logger.info("backend load: %d activity row(s) dropped (no product slug)", dropped_no_slug)
    if dropped_unknown_product:
        logger.info("backend load: %d activity row(s) dropped (product slug not in catalog)", dropped_unknown_product)
    logger.info("backend load: %d canonical interactions from %d activity rows", len(interactions), len(activities))
    return interactions, guid_by_internal


def load_backend_users(
    client: BackendApiClient,
    guid_by_internal: dict[int, str],
    catalog: BackendCatalog,
) -> list[RawUser]:
    """One `RawUser` per user that appears in the (already resolved)
    interaction stream. Each is enriched via `GET /api/users/{guid}`
    best-effort; while that endpoint is still auth-gated every fetch
    returns `None` and the profile stays bare (id only) - which the
    canonical `UserProfile` already tolerates (preferred_category /
    age_group Optional). Enrichment starts working with no code change
    once the backend opens the endpoint.

    The served user population is therefore "users with >=1 recorded
    activity". Serving zero-activity users via pure cold-start would need
    a full roster endpoint (`GET /api/users`, currently auth-gated) - out
    of scope here.
    """
    raw_users: list[RawUser] = []
    auth_blocked = False
    enriched = 0
    for probe_index, (internal_id, guid) in enumerate(sorted(guid_by_internal.items())):
        api_user = None
        if not auth_blocked:
            try:
                api_user = client.get_user(guid)
            except BackendAuthError:
                api_user = None
            if api_user is None and probe_index + 1 >= _ENRICH_AUTH_PROBE_LIMIT and enriched == 0:
                auth_blocked = True
                logger.warning(
                    "GET /api/users/{guid} returned no profile for the first %d users - "
                    "treating the endpoint as still auth-gated and skipping the rest; "
                    "profiles will be bare until the backend opens it",
                    _ENRICH_AUTH_PROBE_LIMIT,
                )
        if api_user is not None:
            enriched += 1
        raw_users.append(_to_raw_user(internal_id, api_user, catalog))

    logger.info("backend load: %d users (%d enriched via /api/users/{guid})", len(raw_users), enriched)
    return raw_users


def load_backend_reviews(
    client: BackendApiClient,
    catalog: BackendCatalog,
    guid_by_internal: dict[int, str],
) -> list[RawReview]:
    """`GET /api/reviews` -> canonical `RawReview`s.

    Drop policy mirrors `load_backend_events` exactly: a row that cannot be
    resolved to a product and a user *this load already knows* is counted,
    logged, and discarded - never allowed to mint a phantom id or attach to
    the wrong entity. Also dropped: ratings outside the canonical 1-5 range
    (`RawReview.rating` is `ge=1, le=5`, so an out-of-range row would
    otherwise abort the whole load).

    **Identity gap - why this currently yields no reviews.** The rows are
    real and the endpoint works, but `/api/reviews` addresses users and
    products by the backend's **int32 primary keys** (`userId`,
    `productId`), while every other endpoint this integration consumes
    addresses users by GUID and products by slug. No endpoint exposes both
    for the same row, so there is no join key and `_resolve_*` below cannot
    match. That is the same gap the backend team's in-progress "immutable
    product UUID/ID in product responses and /api/user-activities" work
    closes - see docs/data-mapping.md section 19.6. Nothing is guessed in
    the meantime: reviews are an *optional* auxiliary signal
    (`EngagementProfile.reviews` defaults to `[]`, and
    `features.product_features.build_product_features` falls back to
    neutral rating defaults for a review-free catalog), so an empty result
    is semantics-preserving, not fabricated.

    Skipped entirely (one log line, no request) when service credentials
    are unset - the endpoint is Bearer-gated and would otherwise 401 once
    per load.
    """
    if not client.has_service_credentials():
        logger.info(
            "skipping /api/reviews: it is Bearer-gated and no service credentials are configured "
            "(set %s / %s); reviews are an optional signal and the load continues without them",
            ENV_CLIENT_ID, ENV_CLIENT_SECRET,
        )
        return []

    try:
        api_reviews = client.list_reviews()
    except (BackendCredentialsError, BackendAuthError) as exc:
        logger.warning("GET /api/reviews unauthorized (%s) - continuing without review signals", exc)
        return []

    if not api_reviews:
        logger.info("backend load: /api/reviews returned no rows")
        return []

    internal_by_guid = {guid: internal for internal, guid in guid_by_internal.items()}
    reviews: list[RawReview] = []
    dropped_rating = dropped_unknown_product = dropped_unknown_user = dropped_no_id = 0

    for row in api_reviews:
        if row.review_id is None:
            dropped_no_id += 1
            continue
        rating = _valid_rating(row.rating)
        if rating is None:
            dropped_rating += 1
            continue
        product_id = _resolve_review_product(row, catalog)
        if product_id is None:
            dropped_unknown_product += 1
            continue
        user_id = _resolve_review_user(row, internal_by_guid)
        if user_id is None:
            dropped_unknown_user += 1
            continue
        reviews.append(
            RawReview(
                id=row.review_id,
                user_id=user_id,
                product_id=product_id,
                rating=rating,
                comment=row.comment,
                # Same convention as activities: a naive backend timestamp
                # is UTC wall-clock (see `_as_naive_utc`).
                creation_date=_as_naive_utc(row.created_at),
            )
        )

    if dropped_no_id:
        logger.info("backend load: %d review row(s) dropped (no reviewId)", dropped_no_id)
    if dropped_rating:
        logger.info("backend load: %d review row(s) dropped (rating missing or outside 1-5)", dropped_rating)
    if dropped_unknown_user:
        logger.info("backend load: %d review row(s) dropped (user not in this load's activity stream)", dropped_unknown_user)
    if dropped_unknown_product:
        logger.warning(
            "backend load: %d of %d review row(s) dropped - /api/reviews identifies products by the "
            "backend's int32 productId, which /api/products does not expose, so there is no join key. "
            "Reviews stay unavailable until the backend adds its product id to the product projection "
            "(docs/data-mapping.md section 19.6).",
            dropped_unknown_product, len(api_reviews),
        )
    logger.info("backend load: %d canonical reviews from %d /api/reviews row(s)", len(reviews), len(api_reviews))
    return reviews


def _valid_rating(rating: float | None) -> float | None:
    """Canonical `RawReview.rating` is `ge=1, le=5`. The response schema
    declares no bounds (only the write side does), so an out-of-range or
    missing value is a droppable data problem, not an exception.
    """
    if rating is None:
        return None
    return float(rating) if 1.0 <= float(rating) <= 5.0 else None


def _resolve_review_product(review: ApiReview, catalog: BackendCatalog) -> int | None:
    """Backend int32 `productId` -> internal product id.

    Returns `None` for every row today: the catalog is keyed by slug
    because `/api/products` exposes no numeric id, so there is nothing to
    match `productId` against. **This is the single place to change** when
    the backend adds its product id to the product projection - populate a
    `{backend_product_id: internal_id}` map in `load_backend_catalog` and
    look it up here. Deliberately not pre-built against a guessed field
    name (see docs/data-mapping.md section 19.5).
    """
    return catalog.product_id_by_backend_id.get(review.product_id) if review.product_id is not None else None


def _resolve_review_user(review: ApiReview, internal_by_guid: dict[str, int]) -> int | None:
    """Backend int32 `userId` -> internal user id.

    Same gap on the user side: `/api/user-activities` and
    `/api/users/{guid}` both address users by GUID, so an int `userId` has
    no counterpart. Resolution is intentionally restricted to users this
    load already saw (never `resolver.resolve_user`, which would *mint* a
    new internal id for an unknown key and create a phantom user with a
    review but no activity).
    """
    return None if review.user_id is None else internal_by_guid.get(str(review.user_id))


# --- helpers ----------------------------------------------------------


def _as_naive_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is not None:
        from datetime import timezone

        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _to_raw_user(internal_id: int, api_user, catalog: BackendCatalog) -> RawUser:
    if api_user is None:
        return RawUser(id=internal_id, first_name="", last_name="", email="", preferred_category_id=None, age_group=None)
    # `preferredCategories` is a list (verified live 2026-09-04, see
    # dtos.ApiFavoriteCategory) - only the first entry's category is used;
    # RawUser/UserProfile model a single preferred category, and the
    # backend does not rank/order multiple favorites for us.
    pref_slug = api_user.first_preferred_category_slug()
    pref_name = api_user.first_preferred_category_name()
    pref_id = catalog.category_id_by_slug.get(pref_slug) if pref_slug else None
    if pref_id is None and pref_name:
        pref_id = catalog.category_id_by_name.get(pref_name)
    return RawUser(
        id=internal_id,
        first_name=api_user.first_name or "",
        last_name=api_user.last_name or "",
        email=api_user.email or "",
        preferred_category_id=pref_id,
        # `ageGroup` has no equivalent in the live UserResponse schema at
        # all (verified 2026-09-04) - this is always None today; never
        # derived from `birthDate`, only ever the API's own field.
        age_group=api_user.age_group,
    )
