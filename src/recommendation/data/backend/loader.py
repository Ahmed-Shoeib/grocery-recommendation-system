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

from recommendation.data.backend.client import BackendApiClient
from recommendation.data.backend.dtos import ApiActivity
from recommendation.data.backend.errors import BackendAuthError
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
    ) -> None:
        self.categories = categories
        self.products = products
        self.tags = tags
        self.product_tags = product_tags
        self.category_id_by_slug = category_id_by_slug
        self.category_id_by_name = category_id_by_name
        self.product_slugs = {p.slug for p in products}


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

    logger.info(
        "backend catalog loaded: %d categories, %d products (tags not exposed by the list endpoint)",
        len(raw_categories), len(raw_products),
    )
    # backend gap: the product LIST projection carries no tags, so the
    # RawTag / RawProductTag join is empty. Product `tags` therefore never
    # reach the Sentence Transformer text for this source. Hydrating them
    # would mean one /api/products/{slug} call per product (N+1) - deferred.
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


def load_backend_reviews(client: BackendApiClient) -> list[RawReview]:
    """`/api/reviews` is not implemented by the backend yet (verified: the
    route 404s). Reviews are an *optional* auxiliary ranking signal -
    `EngagementProfile.reviews` defaults to `[]` and
    `features.product_features.build_product_features` handles a
    review-free catalog (rating features fall back to neutral defaults) -
    so returning an empty list here is the existing semantics-preserving
    fallback, NOT fabricated data.

    When the endpoint lands, its expected contract (to require no
    downstream change) is: a cursor-paginated list of
    `{userId: GUID, productSlug: str, rating: number (1-5),
    comment: str?, createdAt: datetime}`. Implement the body then:
    resolve userId via `resolver.resolve_user`, productSlug via
    `resolver.peek_product` (drop unknown, same as activities), build
    `RawReview`. No other file needs to change.
    """
    return []


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
