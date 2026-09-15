"""Backend DTO -> canonical Raw* / UserInteraction mapping and the
activity-drop policy (docs/data-mapping.md section 19).
"""

import json
from datetime import datetime

from recommendation.backend.dtos import ApiActivity
from recommendation.backend.errors import BackendAuthError
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.backend.loader import (
    load_backend_catalog,
    load_backend_events,
    load_backend_reviews,
    load_backend_users,
)
from tests._backend_fakes import FakeBackendClient

_CATS = [{"slug": "groceries", "name": "Groceries"}, {"slug": "electronics", "name": "Electronics"}]
_PRODS = [
    {"slug": "orange-juice", "name": "Orange Juice", "price": 4.0, "stockQuantity": 50, "categorySlug": "groceries"},
    {"slug": "headphones", "name": "Headphones", "price": 99.0, "stockQuantity": 0, "categorySlug": "electronics"},
    {"slug": "weird", "name": "Weird", "price": -3.0, "stockQuantity": -5, "categorySlug": "string"},
]


def _resolver(tmp_path):
    return ExternalIdentityResolver(tmp_path / "reg.json")


def test_catalog_maps_fields_and_flags_backend_gaps(tmp_path):
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)

    assert len(catalog.categories) == 2
    assert all(c.parent_id is None for c in catalog.categories)  # backend has no parent link

    by_slug = {p.slug: p for p in catalog.products}
    oj = by_slug["orange-juice"]
    assert oj.brand is None and oj.sale_price is None and oj.ingredients is None
    assert oj.is_active is True  # no isActive field -> assume active
    assert oj.category_id == catalog.category_id_by_slug["groceries"]

    weird = by_slug["weird"]
    assert weird.price == 0.01  # non-positive price clamped
    assert weird.stock_quantity == 0  # negative stock clamped
    assert weird.category_id == 0  # placeholder category slug not in /api/categories


def test_product_id_becomes_the_resolver_key_when_the_source_provides_it(tmp_path):
    """Forward-compatible seam: `dtos.ApiProduct.product_id` is always
    `None` on today's live `/api/products` (docs/data-mapping.md 19.5), but
    the moment a product source populates it, `load_backend_catalog` must
    key identity on it instead of slug, and populate
    `product_id_by_backend_id` so `/api/reviews` starts joining.
    """
    r = _resolver(tmp_path)
    prods = [
        {"slug": "orange-juice", "productId": 501, "name": "Orange Juice", "price": 4.0, "stockQuantity": 50, "categorySlug": "groceries"},
        {"slug": "headphones", "productId": 502, "name": "Headphones", "price": 99.0, "stockQuantity": 0, "categorySlug": "electronics"},
    ]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)

    by_slug = {p.slug: p for p in catalog.products}
    assert catalog.product_id_by_backend_id == {501: by_slug["orange-juice"].id, 502: by_slug["headphones"].id}
    assert catalog.product_ids == {501, 502}
    # The registry key is the backend id, not the slug - `peek_product`
    # only finds it via the id-shaped key.
    assert r.peek_product(str(501)) == by_slug["orange-juice"].id
    assert r.peek_product("orange-juice") is None


def test_changed_slug_does_not_change_product_identity_once_keyed_by_product_id(tmp_path):
    """Slug is metadata once `product_id` is the resolver key: renaming a
    product's slug between reloads must not orphan/change its internal id
    - only a `product_id` change would (matching the pre-existing,
    documented slug-mutability behaviour for the old slug-keyed regime).
    """
    r1 = _resolver(tmp_path)
    c1 = load_backend_catalog(
        FakeBackendClient(products=[{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}], categories=_CATS),
        r1,
    )
    first_id = c1.products[0].id
    r1.save()

    r2 = _resolver(tmp_path)
    c2 = load_backend_catalog(
        FakeBackendClient(products=[{"slug": "orange-juice-renamed", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}], categories=_CATS),
        r2,
    )
    assert c2.products[0].id == first_id
    assert c2.products[0].slug == "orange-juice-renamed"  # metadata updates freely


def test_activity_product_id_resolves_against_the_same_catalog_key_as_products(tmp_path):
    """'ProductId from products == ProductId from activities': an activity
    row carrying `product_id` (see `dtos.ApiActivity.product_id`) must
    resolve to the exact same internal id the catalog assigned that
    product - not a slug-keyed lookup, and not a fresh id.
    """
    r = _resolver(tmp_path)
    prods = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    internal_id = catalog.products[0].id

    activities = [ApiActivity.model_validate({
        "userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, _ = load_backend_events(activities, r, catalog)
    assert len(interactions) == 1
    assert interactions[0].product_id == internal_id


def test_activity_with_unknown_product_id_is_dropped_even_with_a_slug_fallback_absent(tmp_path):
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)
    activities = [ApiActivity.model_validate({
        "userId": "g1", "actionType": "AddToCart", "productId": 999999, "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, _ = load_backend_events(activities, r, catalog)
    assert interactions == []


def test_search_product_activity_maps_to_canonical_search(tmp_path):
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)
    activities = [ApiActivity.model_validate({
        "userId": "g1", "actionType": "SearchProduct", "slug": "orange-juice", "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, _ = load_backend_events(activities, r, catalog)
    assert len(interactions) == 1
    assert interactions[0].action_type.value == "SEARCH"


def test_products_activities_and_reviews_all_resolve_to_the_same_canonical_product(tmp_path):
    """The end-to-end cross-source identity guarantee the 2026-09-15
    atomic switch exists to provide: a product's internal id, as assigned
    by the catalog load, is the SAME id an activity row and a review row
    referencing that product's `Product.Id` resolve to - no separate,
    ambiguous, or slug-derived identity for any of the three sources.
    """
    r = _resolver(tmp_path)
    prods = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    internal_id = catalog.products[0].id

    activities = [ApiActivity.model_validate({
        "userId": "g1", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, guid_by_internal = load_backend_events(activities, r, catalog)
    assert interactions[0].product_id == internal_id

    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "g1", "productId": 501, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        guid_by_internal,
    )
    assert len(reviews) == 1
    assert reviews[0].product_id == internal_id
    # And the review's user resolves to the same internal user the activity did.
    assert reviews[0].user_id == interactions[0].user_id


def test_full_catalog_load_never_leaves_a_slug_keyed_entry_when_every_product_has_an_id(tmp_path):
    """No mixed slug/ProductId duplicates: once every product source row
    carries `product_id` (the live shape since the switch - every row
    from `GET /api/ai/products` has one), the resolver's `product`
    namespace must be keyed entirely by stringified ids, never by any
    product's slug.
    """
    r = _resolver(tmp_path)
    prods = [
        {"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"},
        {"slug": "headphones", "productId": 502, "name": "Headphones", "price": 99.0, "categorySlug": "electronics"},
    ]
    load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    r.save()

    doc = json.loads((tmp_path / "reg.json").read_text(encoding="utf-8"))
    keys = set(doc["namespaces"]["product"]["by_key"].keys())
    assert keys == {"501", "502"}
    assert "orange-juice" not in keys and "headphones" not in keys


def test_ids_are_stable_across_a_reload(tmp_path):
    r1 = _resolver(tmp_path)
    c1 = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r1)
    ids1 = {p.slug: p.id for p in c1.products}
    r1.save()

    r2 = _resolver(tmp_path)
    c2 = load_backend_catalog(FakeBackendClient(products=list(reversed(_PRODS)), categories=_CATS), r2)
    ids2 = {p.slug: p.id for p in c2.products}
    assert ids1 == ids2  # order changed, ids didn't


def test_events_mapping_and_drop_policy(tmp_path):
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)
    activities = [
        {"userId": "g1", "actionType": "ViewProduct", "slug": "orange-juice", "timestamp": "2026-08-01T10:00:00"},
        {"userId": "g1", "actionType": "AddToCart", "slug": "orange-juice", "timestamp": "2026-08-02T10:00:00"},
        {"userId": "g2", "actionType": "PlaceOrder", "slug": "headphones", "timestamp": "2026-08-03T10:00:00"},
        {"userId": "g2", "actionType": "AddedToFavorites", "slug": "orange-juice", "timestamp": "2026-08-04T10:00:00"},
        {"userId": "g2", "actionType": "RemoveFromCart", "slug": None, "timestamp": "2026-08-05T10:00:00"},
        {"userId": "g3", "actionType": "AddToCart", "slug": None, "timestamp": "2026-08-06T10:00:00"},
        {"userId": "g3", "actionType": "AddToCart", "slug": "not-in-catalog", "timestamp": "2026-08-07T10:00:00"},
        {"userId": "g3", "actionType": "Teleport", "slug": "orange-juice", "timestamp": "2026-08-08T10:00:00"},
    ]
    interactions, guid_by_id = load_backend_events(
        [ApiActivity.model_validate(a) for a in activities], r, catalog
    )
    kinds = sorted((i.user_id, i.action_type.value) for i in interactions)
    # only the 3 resolvable positive-signal rows survive
    assert kinds == [
        (guid_for(guid_by_id, "g1"), "ADD_TO_CART"),
        (guid_for(guid_by_id, "g1"), "CLICK"),
        (guid_for(guid_by_id, "g2"), "PURCHASE"),
    ]
    assert all(isinstance(i.action_time, datetime) and i.action_time.tzinfo is None for i in interactions)


def guid_for(guid_by_id: dict[int, str], guid: str) -> int:
    return {v: k for k, v in guid_by_id.items()}[guid]


def test_users_are_bare_when_endpoint_is_auth_gated(tmp_path):
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)
    client = FakeBackendClient(users_status=401)
    guid_by_id = {1: "g1", 2: "g2"}
    users = load_backend_users(client, guid_by_id, catalog)
    assert {u.id for u in users} == {1, 2}
    assert all(u.preferred_category_ids == [] and u.age_group is None for u in users)
    # short-circuits: does not call get_user once per user forever
    assert len(client.user_calls) <= 3


def test_user_enrichment_populates_preferred_category_when_available(tmp_path):
    """Real shape verified live 2026-09-04: `preferredCategories` is a
    LIST, each entry nesting a `category` object (`FavoriteCategoryResponse`
    in Swagger) - not the singular `preferredCategory`/`preferredCategorySlug`
    assumed pre-verification.
    """
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)
    client = FakeBackendClient(users={
        "g1": {
            "guid": "g1",
            "firstName": "A",
            "preferredCategories": [
                {"categoryId": 5, "category": {"slug": "groceries", "name": "Groceries"}, "addedAt": "2026-01-01T00:00:00"},
            ],
        },
    })
    users = load_backend_users(client, {1: "g1"}, catalog)
    u = users[0]
    assert u.preferred_category_ids == [catalog.category_id_by_slug["groceries"]]


def test_user_enrichment_populates_all_preferred_categories_not_just_first(tmp_path):
    """Production-safe contract redesign (docs/production-feature-parity-audit.md):
    every favorite category is resolved, not just the first - no arbitrary
    "pick one" reduction.
    """
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)
    client = FakeBackendClient(users={
        "g1": {
            "guid": "g1",
            "firstName": "A",
            "preferredCategories": [
                {"categoryId": 5, "category": {"slug": "groceries", "name": "Groceries"}, "addedAt": "2026-01-01T00:00:00"},
                {"categoryId": 6, "category": {"slug": "electronics", "name": "Electronics"}, "addedAt": "2026-01-02T00:00:00"},
            ],
        },
    })
    users = load_backend_users(client, {1: "g1"}, catalog)
    u = users[0]
    assert u.preferred_category_ids == [
        catalog.category_id_by_slug["groceries"], catalog.category_id_by_slug["electronics"]
    ]


def test_age_group_stays_none_on_the_real_schema_but_is_forward_compatible(tmp_path):
    """`ageGroup` has no equivalent field in the live `UserResponse` schema
    at all (verified 2026-09-04) - a realistic payload never populates it -
    but the DTO stays tolerant (`extra="ignore"`) so a future backend
    addition needs no code change here.
    """
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)

    real_shape_client = FakeBackendClient(users={
        "g1": {
            "guid": "g1", "firstName": "A", "lastName": "B", "email": "a@example.com",
            "phoneNumber": "0100000000", "birthDate": "2000-01-01T00:00:00",
            "preferredCategories": [], "role": 0, "isActive": True, "createdAt": "2026-01-01T00:00:00",
        },
    })
    assert load_backend_users(real_shape_client, {1: "g1"}, catalog)[0].age_group is None

    forward_compat_client = FakeBackendClient(users={"g1": {"guid": "g1", "ageGroup": "25-34"}})
    assert load_backend_users(forward_compat_client, {1: "g1"}, catalog)[0].age_group == "25-34"


# --- /api/reviews ------------------------------------------------------
#
# Rows below use the real verified `AiProductReviewResponse` shape:
# {reviewId, userId, userGuid, productId, rating, comment, createdAt,
# updatedAt} - userGuid was added live 2026-09-15 and is the user-identity
# bridge (`loader._resolve_review_user`); userId/productId remain the
# backend's int32 primary keys, with productId still the join key on the
# product side (`_resolve_review_product`, via
# `BackendCatalog.product_id_by_backend_id`).


def _catalog_with_backend_ids(tmp_path, mapping):
    """A loaded catalog plus the backend-id -> internal-id map that
    `/api/products` will populate once it exposes its numeric id. Lets the
    resolution path be tested today without inventing a DTO field.
    """
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), _resolver(tmp_path))
    catalog.product_id_by_backend_id.update(mapping)
    return catalog


def test_reviews_join_end_to_end_once_the_catalog_itself_exposes_product_id(tmp_path):
    """The full seam, not the manually-populated shortcut: once
    `load_backend_catalog` sees `product_id` on a real product row, its
    `product_id_by_backend_id` map is populated automatically and
    `load_backend_reviews` resolves against it with no other code change -
    'ProductId from reviews resolves to the same canonical product'.
    """
    r = _resolver(tmp_path)
    prods = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    internal_product_id = catalog.products[0].id

    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "g7", "productId": 501, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "g7"},
    )
    assert len(reviews) == 1
    assert reviews[0].product_id == internal_product_id


def test_reviews_are_skipped_without_credentials_and_not_fetched(tmp_path):
    client = FakeBackendClient(
        reviews=[{"reviewId": 1, "userId": 7, "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"}],
        has_credentials=False,
    )
    catalog = _catalog_with_backend_ids(tmp_path, {})
    assert load_backend_reviews(client, catalog, {1: "g1"}) == []
    assert client.review_calls == 0, "a Bearer-gated endpoint must not be called without credentials"


def test_empty_reviews_response_is_not_an_error(tmp_path):
    catalog = _catalog_with_backend_ids(tmp_path, {})
    assert load_backend_reviews(FakeBackendClient(reviews=[]), catalog, {1: "g1"}) == []


def test_reviews_with_unjoinable_int_ids_are_dropped_not_fabricated(tmp_path):
    """Today's live reality: /api/reviews keys on int32 ids that
    /api/products and /api/user-activities do not expose, so every row is
    dropped rather than guessed onto some product.
    """
    catalog = _catalog_with_backend_ids(tmp_path, {})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "productId": 3, "rating": 5, "comment": "great",
             "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {1: "g1"},
    )
    assert reviews == []


def test_resolvable_reviews_become_canonical_raw_reviews(tmp_path):
    """The seam that activates when the backend exposes its product id:
    with a populated backend-id map the same rows flow through to
    canonical `RawReview`s, unchanged downstream.
    """
    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 42, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": 4, "comment": "good",
             "createdAt": "2026-09-01T10:00:00", "updatedAt": None},
        ]),
        catalog,
        {5: "guid-7"},  # internal user 5 <-> the backend user GUID "guid-7"
    )
    assert len(reviews) == 1
    review = reviews[0]
    assert (review.id, review.user_id, review.product_id, review.rating) == (42, 5, 11, 4.0)
    assert review.comment == "good"
    assert review.creation_date == datetime(2026, 9, 1, 10, 0, 0)


def test_user_guid_is_the_join_key_not_the_legacy_int_user_id(tmp_path):
    """Live 2026-09-15: `userGuid` is the real bridge; the int32 `userId`
    is no longer authoritative. A row whose `userGuid` happens to equal
    the *string form* of an internal id's old int `userId` must NOT
    resolve by coincidence - only an actual matching GUID does.
    """
    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            # userId=7 matches nothing useful; no userGuid at all -> dropped.
            {"reviewId": 1, "userId": 7, "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "guid-7"},
    )
    assert reviews == []


def test_product_and_user_resolution_are_counted_independently(tmp_path, caplog):
    """The two join sides must never short-circuit each other for
    diagnostics: a row failing product resolution must still have its
    user side checked (and counted) too, so live counts (e.g. the smoke
    test / docs 19.6) report the true per-side resolution rate rather
    than whichever check happened to run first.
    """
    import logging

    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})  # only productId 3 is joinable
    with caplog.at_level(logging.INFO, logger="recommendation.backend.loader"):
        reviews = load_backend_reviews(
            FakeBackendClient(reviews=[
                # product resolves (3->11), user does not (unknown guid).
                {"reviewId": 1, "userId": 7, "userGuid": "unknown", "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
                # user resolves (guid-7), product does not (999 unmapped).
                {"reviewId": 2, "userId": 7, "userGuid": "guid-7", "productId": 999, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
                # neither resolves.
                {"reviewId": 3, "userId": 7, "userGuid": "unknown", "productId": 999, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
            ]),
            catalog,
            {5: "guid-7"},
        )
    assert reviews == []  # every row fails at least one side
    diag = next(r.message for r in caplog.records if "reviews join diagnostics" in r.message)
    # 3 rows resolvable-in-principle; product resolves for row 1 only (1/3);
    # user resolves for row 2 only (1/3).
    assert "1/3 product-side resolved" in diag
    assert "1/3 user-side resolved" in diag


def test_unknown_user_is_dropped_never_minted(tmp_path):
    """A review by a user with no recorded activity must not create a
    phantom user id (the activity stream defines the served population) -
    even though the row carries a real-looking, well-formed `userGuid`.
    """
    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 999, "userGuid": "guid-never-seen", "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "guid-7"},
    )
    assert reviews == []


def test_malformed_ratings_and_ids_are_dropped_without_raising(tmp_path):
    """`RawReview.rating` is ge=1/le=5 - an out-of-range row must be
    counted and skipped, never abort the whole load.
    """
    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": 0, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 2, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": 9, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 3, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": None, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": None, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 5, "userId": 7, "userGuid": "guid-7", "productId": None, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 6, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "guid-7"},
    )
    assert [r.id for r in reviews] == [6]


def test_missing_and_aware_timestamps_are_normalized_to_naive_utc(tmp_path):
    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": 5, "createdAt": None},
            {"reviewId": 2, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": 5, "createdAt": "2026-09-01T12:00:00+02:00"},
        ]),
        catalog,
        {5: "guid-7"},
    )
    assert reviews[0].creation_date is None
    # +02:00 -> naive UTC, matching the activity-stream convention.
    assert reviews[1].creation_date == datetime(2026, 9, 1, 10, 0, 0)


def test_auth_failure_degrades_instead_of_failing_the_load(tmp_path):
    class Unauthorized(FakeBackendClient):
        def list_reviews(self):
            raise BackendAuthError("GET /api/reviews returned 403", status_code=403)

    catalog = _catalog_with_backend_ids(tmp_path, {})
    assert load_backend_reviews(Unauthorized(), catalog, {1: "g1"}) == []


def test_canonical_reviews_reach_the_shared_review_adapter(tmp_path):
    """The downstream half of the boundary: once `load_backend_reviews`
    resolves rows (backend-id map populated, as it will be when the
    backend exposes its product id), the `RawReview`s it returns are the
    exact type `InMemoryReviewAdapter` consumes for the synthetic and
    SQLite sources - so `EngagementProfile.reviews` and
    `build_product_features` (already covered by test_product_features.py)
    need no backend-specific code. Here we assert the hand-off:
    per-user `ReviewRecord`s with resolved internal ids and the right
    rating/product mapping.
    """
    from recommendation.adapters.review_adapter import InMemoryReviewAdapter
    from recommendation.features.product_features import compute_review_stats

    catalog = _catalog_with_backend_ids(tmp_path, {3: 11, 4: 12})
    raw_reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "guid-7", "productId": 3, "rating": 5, "comment": "a",
             "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 2, "userId": 7, "userGuid": "guid-7", "productId": 4, "rating": 3, "comment": "b",
             "createdAt": "2026-09-02T10:00:00"},
        ]),
        catalog,
        {5: "guid-7"},
    )
    assert {r.product_id for r in raw_reviews} == {11, 12}

    review_adapter = InMemoryReviewAdapter(raw_reviews)
    records = review_adapter.get_reviews(5)                # EngagementProfile.reviews source
    assert sorted((r.product_id, r.rating) for r in records) == [(11, 5.0), (12, 3.0)]

    stats = compute_review_stats(review_adapter.list_all_reviews())
    assert stats[11] == (5.0, 1) and stats[12] == (3.0, 1)
