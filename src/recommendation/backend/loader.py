"""Backend DTOs -> canonical `Raw*` / `UserInteraction` models.

The exact counterpart of `recommendation.sqlite.loader`: it produces
the SAME models the synthetic generator and the SQLite loader produce, so
`adapters.backend_factory.build_backend_api_adapters` can hand them
straight to the existing `InMemoryProductCatalogAdapter` /
`InMemoryUserAdapter` / `InMemoryReviewAdapter` / `UserEventsAdapter`
without a single new adapter class.

All product-id/slug/GUID -> int translation goes through
`ExternalIdentityResolver`. Catalog objects (products, categories) are
*assigned* ids; cross references inside the activity/review streams are
*looked up* only - a row that names a product absent from the catalog is
dropped (counted + logged), never allowed to mint a phantom product id or
attach to the wrong one (docs/data-mapping.md section 19, and the
eligibility/data validation contract in section 5).

**Product identity (2026-09-17 refactor - docs/data-mapping.md 19.5/19.16):
the canonical `product_id` IS the backend's `Product.Id`, passed through
verbatim - `RawProduct.id = p.product_id`, never a value minted by
`ExternalIdentityResolver`.** A prior version of this module routed every
product through `resolver.resolve_product(str(p.product_id))`, which -
despite the name - does not return its input back: it mints a fresh,
unrelated, sequentially-assigned internal int per first-seen key (see
`identity.py`'s namespace counter). That silently created a SECOND product
identity space (a dense `1..N`) sitting between the backend's real,
non-contiguous `Product.Id` (e.g. 82..180 with gaps) and every downstream
consumer (features, embeddings, the ANN index, the ranker, the
recommendation API) - the exact bug a live backend-integration trace
confirmed end to end (recommender `product_id=23` for a product whose
real `Product.Id` was 105, reproduced for every sampled recommendation).
`ExternalIdentityResolver` is still exactly right for categories (slug) -
no numeric identity exists for them at all - and is kept as a defensive
fallback for a product row that (still, in principle) carries no
`product_id` at all (see `_SLUG_FALLBACK_ID_BASE` below); it is simply no
longer invoked for the common case, where the row already carries the
one and only id that matters.

**User identity (2026-09-18 refactor - docs/data-mapping.md 19.5/19.16/
19.17, `ai-user-identity-mapping`): the canonical `user_id` IS the
backend's own database `User.Id`, joined by GUID from a dedicated
protected identity endpoint - `ExternalIdentityResolver` is no longer
used for users at all.** The public/client-facing `GET /api/users` and
`GET /api/users/{guid}` remain GUID-only by the backend team's explicit,
permanent design (the database `User.Id` must never cross that boundary)
- so unlike products, there is no single enriched endpoint to read the
canonical id from directly. Instead, `GET /api/ai/users` (protected,
service-to-service, `users:read`) is the sole authoritative source of the
`GUID <-> User.Id` mapping (`load_ai_user_identities` below), and every
other user-identity touchpoint - the `/api/users` roster/profile data,
and `/api/ai/user-activities`' rows - is joined against it BY GUID, never
minted. This mirrors the review join that already existed
(`_resolve_review_user`, unchanged by this refactor) rather than the old,
now-removed, resolver-minting pattern: a GUID with no matching
`/api/ai/users` entry is dropped (counted, logged), never assigned a
generated id.

**As of the 2026-09-15 atomic source switch**, `backend.client
.BackendApiClient.list_products`/`list_activities` read
`GET /api/ai/products`/`GET /api/ai/user-activities`, which always
populate `product_id` - so every live product/activity row passes its
`Product.Id` straight through as the canonical id today; slug remains
modeled and populated purely as metadata (`RawProduct.slug`). See the
per-field notes below and in `dtos.py`, and docs/data-mapping.md 19.5 for
the full migration story (including why the legacy plain
`/api/products`/`/api/user-activities` are no longer called by
`backend_api` at all).

Field-availability vs the ERD-backed paths (verified against the live API,
most recently 2026-09-15): the backend product projection has NO brand,
sale price, discount, `isActive` flag, ingredients, or parent-category
link. The canonical `Raw*` models keep those fields optional / defaulted,
so the mapping below is lossy-but-valid rather than a schema change. See
each `# backend gap:` note.
"""

from __future__ import annotations

from collections import Counter

from recommendation.backend.auth import ENV_CLIENT_ID, ENV_CLIENT_SECRET
from recommendation.backend.client import BackendApiClient
from recommendation.backend.dtos import ApiActivity, ApiReview, ApiUserIdentity
from recommendation.backend.errors import BackendAuthError, BackendCredentialsError
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.backend.mapping import is_known, map_action_type
from recommendation.schemas.events import UserInteraction
from recommendation.synthetic.raw_schemas import (
    RawCategory,
    RawProduct,
    RawProductTag,
    RawReview,
    RawTag,
    RawUser,
)
from recommendation.logging import get_logger

logger = get_logger(__name__)

# If the first N per-user profile fetches all fail auth, stop trying (the
# endpoint is still gated) rather than emit one 401 per active user.
_ENRICH_AUTH_PROBE_LIMIT = 3

# Defensive-only fallback identity for a product row that carries no
# `product_id` at all (no live source has done this since the 2026-09-15
# switch - see module docstring). Offset comfortably above any realistic
# `Product.Id` (an existing int32 column, observed in the low hundreds)
# so a resolver-minted fallback id can never collide with, or be mistaken
# for, a real backend `Product.Id` - the two id spaces are disjoint by
# construction, not by convention.
_SLUG_FALLBACK_ID_BASE = 1_000_000_000


def _slug_fallback_product_id(resolver: ExternalIdentityResolver, slug: str) -> int:
    """Mints (or reuses) a fallback id for a product with no `product_id` -
    catalog-load only, exactly like `resolver.resolve_product` itself
    (never called from activity/review resolution, which must only ever
    look up, never mint - see `_peek_slug_fallback_product_id`).
    """
    return _SLUG_FALLBACK_ID_BASE + resolver.resolve_product(slug)


def _peek_slug_fallback_product_id(resolver: ExternalIdentityResolver, slug: str) -> int | None:
    raw = resolver.peek_product(slug)
    return None if raw is None else _SLUG_FALLBACK_ID_BASE + raw


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
        product_ids: set[int] | None = None,
    ) -> None:
        self.categories = categories
        self.products = products
        self.tags = tags
        self.product_tags = product_tags
        self.category_id_by_slug = category_id_by_slug
        self.category_id_by_name = category_id_by_name
        self.product_slugs = {p.slug for p in products}
        # The set of canonical ids that are genuinely the backend's own
        # `Product.Id` (i.e. NOT a `_SLUG_FALLBACK_ID_BASE`-offset id) -
        # populated by `load_backend_catalog` for every product whose
        # source row carried `product_id`. Since the 2026-09-15 switch to
        # `GET /api/ai/products`, that is every product. Activities/reviews
        # may only resolve against THIS set (never against a slug-fallback
        # id, which a numeric `productId` on an activity/review row could
        # never legitimately mean) - see `load_backend_events`/
        # `_resolve_review_product`.
        self.product_ids = set(product_ids or ())


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
    numeric_product_ids: set[int] = set()
    clamped_price = clamped_stock = 0
    for p in api_products:
        price = p.price
        if price is None or price <= 0:
            price = 0.01
            clamped_price += 1
        stock = p.stock_quantity if p.stock_quantity and p.stock_quantity > 0 else 0
        if p.stock_quantity is not None and p.stock_quantity < 0:
            clamped_stock += 1
        # Canonical id: the backend's own `Product.Id`, passed through
        # verbatim - NEVER minted by `ExternalIdentityResolver` (see module
        # docstring). `GET /api/ai/products` (the authoritative source as
        # of the 2026-09-15 switch) always sends `product_id`, so this is
        # every product's id in live use; the slug-fallback branch only
        # fires for a row with no `product_id` at all (not expected to
        # trigger against this source - kept for the legacy `ApiProduct`
        # shape and tests, and isolated into its own disjoint id space so
        # it can never collide with, or be confused for, a real
        # `Product.Id` - see `_SLUG_FALLBACK_ID_BASE`).
        if p.product_id is not None:
            internal_id = p.product_id
            numeric_product_ids.add(internal_id)
        else:
            internal_id = _slug_fallback_product_id(resolver, p.slug)
        raw_products.append(
            RawProduct(
                id=internal_id,
                # backend gap: category_slug may be a placeholder ("string")
                # not present in /api/categories -> category_id 0, which
                # InMemoryProductCatalogAdapter treats as "no category".
                category_id=cat_id_by_slug.get(p.category_slug or "", 0),
                slug=p.slug,               # metadata only - see 19.5, identity is product_id above
                name=p.name,
                description=p.description,
                brand=None,               # backend gap: no brand field
                price=price,
                sale_price=None,           # backend gap: no sale price
                discount_percentage=None,  # backend gap: no discount
                stock_quantity=stock,
                ingredients=None,          # backend gap: no ingredients
                is_active=True,            # backend gap: no isActive flag - assume active, stock gates eligibility
                # backend gap: AiProductResponse (the 2026-09-15 authoritative
                # source) has no altText/productImageUrl - both always None
                # for backend_api now. UI-display metadata only, not consumed
                # by features/embeddings (verified - see docs/data-mapping.md
                # 19.5); the legacy /api/products fields they mirror are no
                # longer read.
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
    return BackendCatalog(raw_categories, raw_products, [], [], cat_id_by_slug, cat_id_by_name, numeric_product_ids)


def load_ai_user_identities(client: BackendApiClient) -> tuple[dict[str, int], dict[int, str]]:
    """`GET /api/ai/users` -> the authoritative `GUID <-> User.Id` mapping
    (docs/data-mapping.md 19.5/19.16/19.17). The ONLY function in this
    module that establishes canonical user identity - every other
    user-identity touchpoint (`load_backend_users_roster`,
    `load_backend_events`, `_resolve_review_user`) only ever looks up
    against the maps this returns, never mints.

    Returns `(user_id_by_guid, guid_by_user_id)` - both directions,
    since callers need different ones (roster iterates identities
    forward; activities need to peek by guid; the lazy per-user adapter
    needs guid-by-id to know which guid to sync).

    A row missing `userId` or `userGuid` is skipped (counted, logged) -
    never assigned a generated id. A GUID that maps to two DIFFERENT ids,
    or an id that maps to two DIFFERENT GUIDs, is data corruption: every
    conflicting row past the first-seen one is dropped and logged loudly
    (`logger.error`) rather than silently accepted - a corrupt identity
    mapping must never be allowed to attach one user's history to
    another's canonical id.
    """
    identities = client.list_ai_user_identities()
    user_id_by_guid: dict[str, int] = {}
    guid_by_user_id: dict[int, str] = {}
    dropped_incomplete = 0
    conflicts = 0

    for identity in identities:
        if identity.user_id is None or not identity.user_guid:
            dropped_incomplete += 1
            continue
        guid, uid = identity.user_guid, identity.user_id

        existing_uid = user_id_by_guid.get(guid)
        if existing_uid is not None and existing_uid != uid:
            logger.error(
                "backend load: /api/ai/users has a CONFLICTING mapping - guid already mapped to "
                "user_id=%d, also claims user_id=%d - dropping the conflicting entry, keeping the first seen",
                existing_uid, uid,
            )
            conflicts += 1
            continue
        existing_guid = guid_by_user_id.get(uid)
        if existing_guid is not None and existing_guid != guid:
            logger.error(
                "backend load: /api/ai/users has a CONFLICTING mapping - user_id=%d already mapped to "
                "a guid, also claims a different guid - dropping the conflicting entry, keeping the first seen",
                uid,
            )
            conflicts += 1
            continue

        user_id_by_guid[guid] = uid
        guid_by_user_id[uid] = guid

    if dropped_incomplete:
        logger.warning(
            "backend load: %d /api/ai/users row(s) missing userId or userGuid - dropped, no id generated",
            dropped_incomplete,
        )
    if conflicts:
        logger.error(
            "backend load: %d conflicting /api/ai/users identity row(s) dropped - this indicates backend-side "
            "data corruption and should be investigated",
            conflicts,
        )
    logger.info("backend load: %d canonical user identit(y/ies) loaded from GET /api/ai/users", len(user_id_by_guid))
    return user_id_by_guid, guid_by_user_id


def load_backend_events(
    activities: list[ApiActivity],
    resolver: ExternalIdentityResolver,
    catalog: BackendCatalog,
    user_id_by_guid: dict[str, int],
) -> tuple[list[UserInteraction], dict[int, str]]:
    """Map `GET /api/ai/user-activities` rows to canonical `UserInteraction`s
    (docs/data-mapping.md 19.5 - the authoritative `backend_api` activity
    source since the 2026-09-15 switch; the legacy `/api/user-activities`
    is no longer called).

    Dropped (counted + logged, never silently mis-signalled):
    - rows whose `actionType` maps to no canonical signal (favorites,
      cart/favorite removals -> known-ignored; anything else -> unknown);
    - rows with neither a product id nor a product slug (the backend
      records some actions without resolving a product);
    - rows whose product reference (id or slug) is not in the current
      catalog (deleted / unknown product - matches the eligibility
      contract: an unknown external id must never resolve to *a* product);
    - rows whose user reference cannot be resolved to a canonical
      `User.Id` (see below) - an unknown GUID must never resolve to *a*
      user either.

    Product resolution: unchanged by the 2026-09-18 user-identity
    migration - see the 2026-09-17 docstring notes still in force below.
    `resolver` is retained SOLELY for that product-side slug-fallback path
    (`_peek_slug_fallback_product_id`); it is never consulted for user
    identity any more.

    User resolution (2026-09-18, docs/data-mapping.md 19.17): prefers
    `row.canonical_user_id` (the additive field the backend now sends)
    when present - a plain pass-through, no lookup needed, since it's
    already the canonical `User.Id`. Falls back to a GUID lookup against
    `user_id_by_guid` (populated by `load_ai_user_identities` from the
    authoritative `GET /api/ai/users`) only when `canonical_user_id` is
    absent - `row.user_id` (still, intentionally, the GUID string on this
    DTO - see `dtos.ApiActivity`) is used purely as a lookup KEY here,
    never assigned directly to `UserInteraction.user_id`. Either path is a
    lookup, never a mint: a GUID with no matching identity entry drops the
    row (counted, logged) exactly like an unresolvable product reference -
    it can never create a phantom user.
    """
    interactions: list[UserInteraction] = []
    guid_by_internal: dict[int, str] = {}
    dropped_action = Counter()
    dropped_no_product_ref = 0
    dropped_unknown_product = 0
    dropped_unknown_user = 0
    unknown_action_values: set[str] = set()

    for row in activities:
        canonical = map_action_type(row.action_type)
        if canonical is None:
            dropped_action[row.action_type] += 1
            if not is_known(row.action_type):
                unknown_action_values.add(row.action_type)
            continue

        product_id = None
        if row.product_id is not None:
            if row.product_id in catalog.product_ids:
                product_id = row.product_id  # canonical backend Product.Id, passed through directly
        elif row.slug:
            if row.slug in catalog.product_slugs:
                product_id = _peek_slug_fallback_product_id(resolver, row.slug)

        if product_id is None:
            if row.product_id is None and not row.slug:
                dropped_no_product_ref += 1
            else:
                dropped_unknown_product += 1
            continue

        if row.canonical_user_id is not None:
            user_id = row.canonical_user_id  # already canonical - plain pass-through, no lookup
        else:
            user_id = user_id_by_guid.get(row.user_id)  # lookup only - never mint (see docstring)
        if user_id is None:
            dropped_unknown_user += 1
            continue

        guid_by_internal.setdefault(user_id, row.user_id)  # row.user_id is always the GUID string (see dtos.ApiActivity)
        interactions.append(
            UserInteraction(
                user_id=user_id,
                product_id=product_id,
                action_type=canonical,
                # naive backend timestamp is treated as UTC wall-clock,
                # matching sqlite.loader._parse_timestamp / the
                # reference_time convention in serving.pipeline.
                action_time=_as_naive_utc(row.timestamp),
            )
        )

    if unknown_action_values:
        logger.warning("backend load: unknown activity actionType value(s) ignored: %s", sorted(unknown_action_values))
    if dropped_action:
        logger.info("backend load: %d activity row(s) ignored by action-type policy: %s", sum(dropped_action.values()), dict(dropped_action))
    if dropped_no_product_ref:
        logger.info("backend load: %d activity row(s) dropped (no product id or slug)", dropped_no_product_ref)
    if dropped_unknown_product:
        logger.info("backend load: %d activity row(s) dropped (product id/slug not in catalog)", dropped_unknown_product)
    if dropped_unknown_user:
        logger.info(
            "backend load: %d activity row(s) dropped (no canonical User.Id - neither canonicalUserId "
            "nor a resolvable GUID in the current /api/ai/users mapping)", dropped_unknown_user,
        )
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


def load_backend_users_roster(
    client: BackendApiClient, user_id_by_guid: dict[str, int], catalog: BackendCatalog
) -> tuple[list[RawUser], dict[int, str]]:
    """Builds the FULL known-user roster from the authoritative
    `GET /api/ai/users` identity mapping (`user_id_by_guid`, produced by
    `load_ai_user_identities`), enriched - best-effort - with profile/
    preferred-category data from the public `GET /api/users` (still
    GUID-only by permanent backend-team design; see module docstring).

    **Identity-primary, profile-secondary - this is the key design choice
    of the 2026-09-18 migration** (docs/data-mapping.md 19.17): iteration
    drives off `user_id_by_guid`, not off the `/api/users` list. This
    means a user present in `/api/ai/users` but not yet in `/api/users`
    (their identity landed before their full profile propagated - exactly
    the timing race a newly-created user can hit) still becomes a real,
    known `RawUser` with a bare/default profile (`_to_raw_user` already
    handles `api_user=None`) - a real backend identity is never treated as
    "unknown" merely because enrichment hasn't caught up. Conversely, a
    `/api/users` profile whose GUID has no `/api/ai/users` entry yet is
    counted, logged, and left out of the roster entirely - never assigned
    a generated id (docs/data-mapping.md 19.13's cold-start-roster fix
    still holds: a real, existing user must never be misclassified as
    unknown - but "real" now means "present in the authoritative identity
    mapping", not merely "present in the profile list").

    `RawUser.id` is always `user_id_by_guid`'s value directly - never
    `ExternalIdentityResolver.resolve_user` (removed from this function
    entirely; see module docstring).
    """
    identity_guids = set(user_id_by_guid)
    api_users = client.list_users()
    profile_by_guid = {u.guid: u for u in api_users if u.guid}

    raw_users: list[RawUser] = []
    guid_by_internal: dict[int, str] = {}
    for guid, user_id in user_id_by_guid.items():
        api_user = profile_by_guid.get(guid)  # optional enrichment - None is a valid, expected case
        guid_by_internal[user_id] = guid
        raw_users.append(_to_raw_user(user_id, api_user, catalog))

    dropped_no_identity = sum(1 for u in api_users if u.guid and u.guid not in identity_guids)
    if dropped_no_identity:
        logger.warning(
            "backend load: %d profile(s) from GET /api/users have no canonical User.Id yet in "
            "GET /api/ai/users - skipped, not assigned a generated id (likely a propagation-timing gap)",
            dropped_no_identity,
        )
    logger.info(
        "backend load: %d user(s) discovered via GET /api/ai/users identity mapping "
        "(%d enriched with GET /api/users profile data)",
        len(raw_users), len(raw_users) - sum(1 for g in user_id_by_guid if g not in profile_by_guid),
    )
    return raw_users, guid_by_internal


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

    **Identity status (2026-09-15, both sides closed and live-verified).**
    User side: `ApiReview.user_guid` (present on every row) is the bridge
    to the GUID this load already resolved from the activity stream;
    `_resolve_review_user` is a direct dict lookup, no guessing, no
    hashing, no N+1 calls. Product side: `_resolve_review_product` checks
    `review.product_id` directly against `BackendCatalog.product_ids`
    (populated by `load_backend_catalog` from `GET /api/ai/products`, the
    authoritative product source since the 2026-09-15 switch, which always
    carries `product_id`) - the review's `productId` IS the canonical id
    once validated, no translation step - so the product join is populated
    end to end in live use now, not just in unit tests. See
    docs/data-mapping.md section 19.5/19.6 for live join counts. Reviews
    remain an *optional* auxiliary
    signal either way (`EngagementProfile.reviews` defaults to `[]`, and
    `features.product_features.build_product_features` falls back to
    neutral rating defaults for a review-free catalog), so any rows that
    still fail to resolve (e.g. a reviewer with no other recorded
    activity) are dropped, counted, and logged rather than fabricated.

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
        # Both sides are resolved independently (never short-circuited) so
        # `dropped_unknown_product`/`dropped_unknown_user` each honestly
        # count every row unresolvable on that side, even when a row fails
        # both - needed for accurate live join diagnostics (see the smoke
        # test and docs/data-mapping.md 19.6), not just "reason A wins".
        product_id = _resolve_review_product(row, catalog)
        user_id = _resolve_review_user(row, internal_by_guid)
        if product_id is None:
            dropped_unknown_product += 1
        if user_id is None:
            dropped_unknown_user += 1
        if product_id is None or user_id is None:
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
    if api_reviews:
        resolvable = len(api_reviews) - dropped_no_id - dropped_rating
        logger.info(
            "backend load: reviews join diagnostics - %d/%d product-side resolved, %d/%d user-side resolved "
            "(of %d row(s) that passed id/rating checks)",
            resolvable - dropped_unknown_product, resolvable, resolvable - dropped_unknown_user, resolvable, resolvable,
        )
    if dropped_unknown_product:
        logger.warning(
            "backend load: %d of %d review row(s) dropped - productId not found in "
            "BackendCatalog.product_ids (the id names a product outside the "
            "current /api/ai/products catalog - see docs/data-mapping.md section 19.5/19.6).",
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
    """Backend int32 `productId` IS the canonical product id - validated
    (not translated) against `BackendCatalog.product_ids` (populated by
    `load_backend_catalog` from `GET /api/ai/products`, the authoritative
    product source since the 2026-09-15 switch - see 19.5). Returns `None`
    when the id is not in that set - meaning it genuinely names a product
    outside the current catalog (deleted, out-of-stock-and-filtered, or a
    stale review referencing a removed product), never a missing-data-
    source gap anymore.
    """
    if review.product_id is None:
        return None
    return review.product_id if review.product_id in catalog.product_ids else None


def _resolve_review_user(review: ApiReview, internal_by_guid: dict[str, int]) -> int | None:
    """Backend `userGuid` -> internal user id.

    `AiProductReviewResponse.userGuid` (verified live 2026-09-15, present
    and non-null on every row observed) is the same GUID
    `/api/user-activities`/`/api/users/{guid}` use, so this is a direct
    lookup against `guid_by_internal` reversed to `{guid: internal_id}` -
    no hashing, no positional matching, no int-id guessing, and (since the
    2026-09-18 user-identity migration) never a resolver mint either -
    `guid_by_internal` is itself sourced from the authoritative
    `GET /api/ai/users` mapping (via `load_backend_users_roster`/
    `load_backend_events`), never generated. Resolution stays
    intentionally restricted to users this load already knows canonically
    - a review by a GUID with no matching identity is still dropped,
    counted, and logged, exactly as before. `user_id` (the int32 primary
    key) is no longer used for this join; it is kept on the DTO only as
    non-authoritative metadata.
    """
    return None if review.user_guid is None else internal_by_guid.get(review.user_guid)


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
        return RawUser(id=internal_id, first_name="", last_name="", email="", preferred_category_ids=[], age_group=None)
    # `preferredCategories` is a list (verified live 2026-09-04, see
    # dtos.ApiFavoriteCategory) - EVERY entry is resolved and kept, not
    # just the first: RawUser/UserProfile model the real multi-favorite
    # shape directly (docs/production-feature-parity-audit.md), so no
    # favorite is silently dropped by an arbitrary "pick one" reduction.
    # Each entry is resolved independently (slug first, name fallback),
    # not as two separately-filtered flat lists, so one entry lacking a
    # slug (or a name) can never desync which id gets attached to which.
    pref_ids: list[int] = []
    for ref in api_user.preferred_category_refs():
        cid = catalog.category_id_by_slug.get(ref.slug or "") or catalog.category_id_by_name.get(ref.name or "")
        if cid is not None:
            pref_ids.append(cid)
    return RawUser(
        id=internal_id,
        first_name=api_user.first_name or "",
        last_name=api_user.last_name or "",
        email=api_user.email or "",
        preferred_category_ids=pref_ids,
        # `ageGroup` has no equivalent in the live UserResponse schema at
        # all (verified 2026-09-04) - this is always None today; never
        # derived from `birthDate`, only ever the API's own field. Legacy/
        # metadata only - no production model feature consumes it any more
        # (see schemas.user.UserProfile docstring).
        age_group=api_user.age_group,
    )
