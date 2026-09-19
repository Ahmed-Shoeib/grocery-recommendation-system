"""Backend DTO -> canonical Raw* / UserInteraction mapping and the
activity-drop policy (docs/data-mapping.md section 19).
"""

import json
from datetime import datetime

from recommendation.backend.dtos import ApiActivity
from recommendation.backend.errors import BackendAuthError
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.backend.loader import (
    load_ai_user_identities,
    load_backend_catalog,
    load_backend_events,
    load_backend_reviews,
    load_backend_users,
    load_backend_users_roster,
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


def test_product_id_passes_through_directly_when_the_source_provides_it(tmp_path):
    """The canonical id IS `dtos.ApiProduct.product_id`, verbatim - no
    resolver lookup, no remapping. `product_id` is `None` on the legacy
    `/api/products` shape (docs/data-mapping.md 19.5), but the moment a
    product source populates it (the live `/api/ai/products` shape, since
    the 2026-09-15 switch), `RawProduct.id` must equal it exactly.
    """
    r = _resolver(tmp_path)
    prods = [
        {"slug": "orange-juice", "productId": 501, "name": "Orange Juice", "price": 4.0, "stockQuantity": 50, "categorySlug": "groceries"},
        {"slug": "headphones", "productId": 502, "name": "Headphones", "price": 99.0, "stockQuantity": 0, "categorySlug": "electronics"},
    ]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)

    by_slug = {p.slug: p for p in catalog.products}
    assert by_slug["orange-juice"].id == 501
    assert by_slug["headphones"].id == 502
    assert catalog.product_ids == {501, 502}
    # The resolver is never consulted for a numeric-id product - neither
    # the raw id nor its stringified form was ever handed to it.
    assert r.peek_product("501") is None
    assert r.peek_product("orange-juice") is None
    assert r.counts()["product"] == 0


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
    interactions, _ = load_backend_events(activities, r, catalog, {"g1": 1})
    assert len(interactions) == 1
    assert interactions[0].product_id == internal_id


def test_activity_with_unknown_product_id_is_dropped_even_with_a_slug_fallback_absent(tmp_path):
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)
    activities = [ApiActivity.model_validate({
        "userId": "g1", "actionType": "AddToCart", "productId": 999999, "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, _ = load_backend_events(activities, r, catalog, {"g1": 1})
    assert interactions == []


def test_search_product_activity_maps_to_canonical_search(tmp_path):
    r = _resolver(tmp_path)
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), r)
    activities = [ApiActivity.model_validate({
        "userId": "g1", "actionType": "SearchProduct", "slug": "orange-juice", "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, _ = load_backend_events(activities, r, catalog, {"g1": 1})
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
    interactions, guid_by_internal = load_backend_events(activities, r, catalog, {"g1": 5})
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


def test_full_catalog_load_never_touches_the_resolver_product_namespace_when_every_product_has_an_id(tmp_path):
    """Once every product source row carries `product_id` (the live shape
    since the switch - every row from `GET /api/ai/products` has one), the
    resolver's `product` namespace must stay completely EMPTY - neither
    the raw id nor the slug of any product is ever handed to
    `ExternalIdentityResolver`. This is the direct fix for the prior
    behaviour (superseded 2026-09-17), where every numeric id was handed
    to the resolver and silently remapped to an unrelated, densely-
    numbered internal id.
    """
    r = _resolver(tmp_path)
    prods = [
        {"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"},
        {"slug": "headphones", "productId": 502, "name": "Headphones", "price": 99.0, "categorySlug": "electronics"},
    ]
    load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    r.save()

    doc = json.loads((tmp_path / "reg.json").read_text(encoding="utf-8"))
    assert doc["namespaces"]["product"]["by_key"] == {}


def test_non_contiguous_real_backend_ids_survive_catalog_activity_and_review_joins_unchanged(tmp_path):
    """End-to-end canonical-identity proof (2026-09-17 refactor):
    real backend `Product.Id` values are NOT contiguous (this project's
    live catalog runs 82..180 with gaps) - a fixture using exactly that
    shape (85, 105, 162, deliberately out of order and with large gaps)
    must come out the other side of catalog load, activity join, and
    review join with those SAME three integers, never renumbered to
    `1, 2, 3` by `ExternalIdentityResolver` or anything else.
    """
    r = _resolver(tmp_path)
    prods = [
        {"slug": "apple-red-delicious", "productId": 85, "name": "Apple Red Delicious", "price": 3.0, "categorySlug": "groceries"},
        {"slug": "pineapple", "productId": 105, "name": "Pineapple", "price": 4.0, "categorySlug": "groceries"},
        {"slug": "zucchini", "productId": 162, "name": "Zucchini", "price": 2.0, "categorySlug": "groceries"},
    ]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    assert {p.id for p in catalog.products} == {85, 105, 162}
    assert catalog.product_ids == {85, 105, 162}

    activities = [ApiActivity.model_validate({
        "userId": "g1", "actionType": "AddToCart", "productId": 105, "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, guid_by_internal = load_backend_events(activities, r, catalog, {"g1": 9})
    assert len(interactions) == 1
    assert interactions[0].product_id == 105  # not 2, not any resolver-minted position

    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 9, "userGuid": "g1", "productId": 162, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        guid_by_internal,
    )
    assert len(reviews) == 1
    assert reviews[0].product_id == 162

    # And the resolver's product namespace was never touched.
    assert r.counts()["product"] == 0


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
        [ApiActivity.model_validate(a) for a in activities], r, catalog, {"g1": 1, "g2": 2, "g3": 3}
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
# product side (`_resolve_review_product`, a direct membership check
# against `BackendCatalog.product_ids` - the review's `productId` IS the
# canonical id once validated, no translation).


def _catalog_with_products(tmp_path, product_ids: list[int]):
    """A loaded catalog whose products carry the given real backend
    `Product.Id` values directly - lets review-resolution tests exercise
    `catalog.product_ids` membership without depending on `_PRODS` (whose
    products are slug-only, carrying no numeric id at all).
    """
    prods = [
        {"slug": f"p{pid}", "productId": pid, "name": f"Product {pid}", "price": 1.0, "categorySlug": "groceries"}
        for pid in product_ids
    ]
    return load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), _resolver(tmp_path))


def test_reviews_join_end_to_end_once_the_catalog_itself_exposes_product_id(tmp_path):
    """The full seam: once `load_backend_catalog` sees `product_id` on a
    real product row, `load_backend_reviews` resolves reviews against
    `catalog.product_ids` directly - no translation, no other code
    change. 'ProductId from reviews resolves to the same canonical
    product' means the SAME integer, not an equivalent one.
    """
    r = _resolver(tmp_path)
    prods = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    canonical_product_id = catalog.products[0].id
    assert canonical_product_id == 501

    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "g7", "productId": 501, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "g7"},
    )
    assert len(reviews) == 1
    assert reviews[0].product_id == canonical_product_id == 501


def test_reviews_are_skipped_without_credentials_and_not_fetched(tmp_path):
    client = FakeBackendClient(
        reviews=[{"reviewId": 1, "userId": 7, "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"}],
        has_credentials=False,
    )
    catalog = _catalog_with_products(tmp_path, [])
    assert load_backend_reviews(client, catalog, {1: "g1"}) == []
    assert client.review_calls == 0, "a Bearer-gated endpoint must not be called without credentials"


def test_empty_reviews_response_is_not_an_error(tmp_path):
    catalog = _catalog_with_products(tmp_path, [])
    assert load_backend_reviews(FakeBackendClient(reviews=[]), catalog, {1: "g1"}) == []


def test_reviews_with_unjoinable_int_ids_are_dropped_not_fabricated(tmp_path):
    """A review `productId` naming a product outside the current catalog
    is dropped rather than guessed onto some other product.
    """
    catalog = _catalog_with_products(tmp_path, [])
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
    """A review whose `productId` names a real catalog product resolves to
    that exact id, unchanged, downstream.
    """
    catalog = _catalog_with_products(tmp_path, [11])
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 42, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": 4, "comment": "good",
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
    catalog = _catalog_with_products(tmp_path, [11])
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            # userId=7 matches nothing useful; no userGuid at all -> dropped.
            {"reviewId": 1, "userId": 7, "productId": 11, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
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

    catalog = _catalog_with_products(tmp_path, [11])  # only productId 11 is joinable
    with caplog.at_level(logging.INFO, logger="recommendation.backend.loader"):
        reviews = load_backend_reviews(
            FakeBackendClient(reviews=[
                # product resolves (11), user does not (unknown guid).
                {"reviewId": 1, "userId": 7, "userGuid": "unknown", "productId": 11, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
                # user resolves (guid-7), product does not (999 unknown).
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
    catalog = _catalog_with_products(tmp_path, [11])
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 999, "userGuid": "guid-never-seen", "productId": 11, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "guid-7"},
    )
    assert reviews == []


def test_malformed_ratings_and_ids_are_dropped_without_raising(tmp_path):
    """`RawReview.rating` is ge=1/le=5 - an out-of-range row must be
    counted and skipped, never abort the whole load.
    """
    catalog = _catalog_with_products(tmp_path, [11])
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": 0, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 2, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": 9, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 3, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": None, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": None, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 5, "userId": 7, "userGuid": "guid-7", "productId": None, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 6, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "guid-7"},
    )
    assert [r.id for r in reviews] == [6]


def test_missing_and_aware_timestamps_are_normalized_to_naive_utc(tmp_path):
    catalog = _catalog_with_products(tmp_path, [11])
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": 5, "createdAt": None},
            {"reviewId": 2, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": 5, "createdAt": "2026-09-01T12:00:00+02:00"},
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

    catalog = _catalog_with_products(tmp_path, [])
    assert load_backend_reviews(Unauthorized(), catalog, {1: "g1"}) == []


def test_canonical_reviews_reach_the_shared_review_adapter(tmp_path):
    """The downstream half of the boundary: once `load_backend_reviews`
    resolves rows, the `RawReview`s it returns are the exact type
    `InMemoryReviewAdapter` consumes for the synthetic and SQLite sources -
    so `EngagementProfile.reviews` and `build_product_features` (already
    covered by test_product_features.py) need no backend-specific code.
    Here we assert the hand-off: per-user `ReviewRecord`s with the right
    canonical-id/rating mapping.
    """
    from recommendation.adapters.review_adapter import InMemoryReviewAdapter
    from recommendation.features.product_features import compute_review_stats

    catalog = _catalog_with_products(tmp_path, [11, 12])
    raw_reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "userGuid": "guid-7", "productId": 11, "rating": 5, "comment": "a",
             "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 2, "userId": 7, "userGuid": "guid-7", "productId": 12, "rating": 3, "comment": "b",
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


# --- 2026-09-18 user-identity migration: backend User.Id (via the
# protected GET /api/ai/users mapping) is now the canonical recommender
# user_id - ExternalIdentityResolver is never invoked for users at all.
# See docs/data-mapping.md 19.5/19.16/19.17 and the `ai-user-identity-mapping`
# backend contract.

_CUSTOMER_USER_ID = 1547
_CUSTOMER_GUID = "81bfc1f1-36eb-4427-b680-119ec489e156"


def test_ai_user_identities_map_guid_to_backend_user_id_directly(tmp_path):
    client = FakeBackendClient(ai_identities=[
        {"userId": _CUSTOMER_USER_ID, "userGuid": _CUSTOMER_GUID},
        {"userId": 82, "userGuid": "05d74037-20a6-4399-82dd-66488575b5a8"},
    ])
    user_id_by_guid, guid_by_user_id = load_ai_user_identities(client)
    assert user_id_by_guid[_CUSTOMER_GUID] == _CUSTOMER_USER_ID
    assert user_id_by_guid["05d74037-20a6-4399-82dd-66488575b5a8"] == 82
    assert guid_by_user_id[_CUSTOMER_USER_ID] == _CUSTOMER_GUID
    assert guid_by_user_id[82] == "05d74037-20a6-4399-82dd-66488575b5a8"


def test_ai_user_identities_drop_incomplete_rows_without_minting(tmp_path, caplog):
    import logging

    client = FakeBackendClient(ai_identities=[
        {"userId": 1547, "userGuid": _CUSTOMER_GUID},
        {"userId": None, "userGuid": "some-guid-with-no-id"},
        {"userId": 99, "userGuid": None},
    ])
    with caplog.at_level(logging.WARNING, logger="recommendation.backend.loader"):
        user_id_by_guid, guid_by_user_id = load_ai_user_identities(client)
    assert user_id_by_guid == {_CUSTOMER_GUID: 1547}
    assert guid_by_user_id == {1547: _CUSTOMER_GUID}
    assert any("missing userId or userGuid" in r.message for r in caplog.records)


def test_ai_user_identities_conflicting_guid_is_dropped_and_logged_loudly(tmp_path, caplog):
    """GUID A -> 1547, then GUID A -> 1602 (same guid, different id): data
    corruption, must never be silently accepted - the first-seen mapping
    wins, the conflicting row is dropped and logged as an error.
    """
    import logging

    client = FakeBackendClient(ai_identities=[
        {"userId": 1547, "userGuid": _CUSTOMER_GUID},
        {"userId": 1602, "userGuid": _CUSTOMER_GUID},
    ])
    with caplog.at_level(logging.ERROR, logger="recommendation.backend.loader"):
        user_id_by_guid, guid_by_user_id = load_ai_user_identities(client)
    assert user_id_by_guid == {_CUSTOMER_GUID: 1547}
    assert guid_by_user_id == {1547: _CUSTOMER_GUID}
    assert any("CONFLICTING mapping" in r.message for r in caplog.records)


def test_ai_user_identities_conflicting_id_is_dropped_and_logged_loudly(tmp_path, caplog):
    """1547 -> GUID A, then 1547 -> GUID B (same id, different guid): also
    data corruption, dropped and logged the same way.
    """
    import logging

    client = FakeBackendClient(ai_identities=[
        {"userId": 1547, "userGuid": "guid-a"},
        {"userId": 1547, "userGuid": "guid-b"},
    ])
    with caplog.at_level(logging.ERROR, logger="recommendation.backend.loader"):
        user_id_by_guid, guid_by_user_id = load_ai_user_identities(client)
    assert user_id_by_guid == {"guid-a": 1547}
    assert guid_by_user_id == {1547: "guid-a"}
    assert any("CONFLICTING mapping" in r.message for r in caplog.records)


def test_ai_user_identities_two_distinct_users_never_cross_map(tmp_path):
    """GUID A -> 1547 and GUID B -> 1602 must remain EXACTLY those two
    pairs - no accidental cross-assignment between distinct users.
    """
    client = FakeBackendClient(ai_identities=[
        {"userId": 1547, "userGuid": "guid-a"},
        {"userId": 1602, "userGuid": "guid-b"},
    ])
    user_id_by_guid, guid_by_user_id = load_ai_user_identities(client)
    assert user_id_by_guid == {"guid-a": 1547, "guid-b": 1602}
    assert guid_by_user_id == {1547: "guid-a", 1602: "guid-b"}


def test_roster_join_produces_raw_user_with_backend_user_id(tmp_path):
    """/api/ai/users + /api/users joined by GUID -> RawUser.id == backend
    User.Id, verbatim - never a resolver-minted value.
    """
    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        ai_identities=[{"userId": _CUSTOMER_USER_ID, "userGuid": _CUSTOMER_GUID}],
        roster=[{"guid": _CUSTOMER_GUID, "firstName": "A", "preferredCategories": []}],
    )
    catalog = load_backend_catalog(client, _resolver(tmp_path))
    user_id_by_guid, _ = load_ai_user_identities(client)
    raw_users, guid_by_internal = load_backend_users_roster(client, user_id_by_guid, catalog)
    assert len(raw_users) == 1
    assert raw_users[0].id == _CUSTOMER_USER_ID
    assert guid_by_internal[_CUSTOMER_USER_ID] == _CUSTOMER_GUID


def test_profile_without_ai_identity_is_dropped_not_minted(tmp_path, caplog):
    """A /api/users profile whose GUID has no /api/ai/users entry must be
    skipped entirely - never assigned a generated id."""
    import logging

    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        ai_identities=[],  # no canonical identities at all
        roster=[{"guid": "orphan-guid", "firstName": "NoIdentity"}],
    )
    catalog = load_backend_catalog(client, _resolver(tmp_path))
    user_id_by_guid, _ = load_ai_user_identities(client)
    with caplog.at_level(logging.WARNING, logger="recommendation.backend.loader"):
        raw_users, guid_by_internal = load_backend_users_roster(client, user_id_by_guid, catalog)
    assert raw_users == []
    assert guid_by_internal == {}
    assert any("no canonical User.Id yet" in r.message for r in caplog.records)


def test_ai_identity_without_profile_still_becomes_a_bare_known_user(tmp_path):
    """The exact new-user propagation-timing scenario: a canonical
    identity exists but /api/users has no profile row for that guid yet -
    the user must still be known, with a bare/default profile, not
    dropped and not treated as unknown.
    """
    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        ai_identities=[{"userId": _CUSTOMER_USER_ID, "userGuid": _CUSTOMER_GUID}],
        roster=[],  # profile has not propagated yet
    )
    catalog = load_backend_catalog(client, _resolver(tmp_path))
    user_id_by_guid, _ = load_ai_user_identities(client)
    raw_users, guid_by_internal = load_backend_users_roster(client, user_id_by_guid, catalog)
    assert len(raw_users) == 1
    assert raw_users[0].id == _CUSTOMER_USER_ID
    assert raw_users[0].first_name == "" and raw_users[0].preferred_category_ids == []
    assert guid_by_internal[_CUSTOMER_USER_ID] == _CUSTOMER_GUID


def test_activity_prefers_canonical_user_id_when_present(tmp_path):
    r = _resolver(tmp_path)
    prods = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    activities = [ApiActivity.model_validate({
        "userId": _CUSTOMER_GUID, "canonicalUserId": _CUSTOMER_USER_ID,
        "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00",
    })]
    # An intentionally WRONG/empty user_id_by_guid map - proves canonicalUserId
    # is used directly and no GUID lookup is even attempted when it's present.
    interactions, guid_by_internal = load_backend_events(activities, r, catalog, {})
    assert len(interactions) == 1
    assert interactions[0].user_id == _CUSTOMER_USER_ID
    assert guid_by_internal[_CUSTOMER_USER_ID] == _CUSTOMER_GUID


def test_activity_falls_back_to_guid_lookup_when_canonical_user_id_absent(tmp_path):
    """Legacy/transitional row shape: only the GUID `userId` field is
    present (`canonicalUserId` absent) - must resolve via a lookup against
    the authoritative /api/ai/users mapping, never mint.
    """
    r = _resolver(tmp_path)
    prods = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    activities = [ApiActivity.model_validate({
        "userId": _CUSTOMER_GUID, "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, guid_by_internal = load_backend_events(
        activities, r, catalog, {_CUSTOMER_GUID: _CUSTOMER_USER_ID}
    )
    assert len(interactions) == 1
    assert interactions[0].user_id == _CUSTOMER_USER_ID
    assert guid_by_internal[_CUSTOMER_USER_ID] == _CUSTOMER_GUID


def test_activity_with_unknown_guid_is_dropped_never_minted(tmp_path):
    """A GUID absent from the authoritative /api/ai/users mapping (and
    with no canonicalUserId either) must be dropped - never assigned a
    generated integer.
    """
    r = _resolver(tmp_path)
    prods = [{"slug": "orange-juice", "productId": 501, "name": "OJ", "price": 4.0, "categorySlug": "groceries"}]
    catalog = load_backend_catalog(FakeBackendClient(products=prods, categories=_CATS), r)
    activities = [ApiActivity.model_validate({
        "userId": "never-seen-guid", "actionType": "AddToCart", "productId": 501, "timestamp": "2026-08-01T10:00:00",
    })]
    interactions, guid_by_internal = load_backend_events(activities, r, catalog, {_CUSTOMER_GUID: _CUSTOMER_USER_ID})
    assert interactions == []
    assert guid_by_internal == {}
    assert r.counts()["user"] == 0, "the resolver must never mint a user id, even for an unresolvable activity guid"


def test_reported_customer_1547_end_to_end_identity_invariants(tmp_path):
    """The literal regression test for the reported production incident:
    backend User.Id 1547 <-> GUID 81bfc1f1-36eb-4427-b680-119ec489e156
    must resolve consistently through the identity mapping, the roster
    join, and activity resolution - with zero history, matching the
    real customer's live state at the time of the incident.
    """
    client = FakeBackendClient(
        products=_PRODS, categories=_CATS,
        ai_identities=[{"userId": _CUSTOMER_USER_ID, "userGuid": _CUSTOMER_GUID}],
        roster=[{"guid": _CUSTOMER_GUID, "firstName": "Customer", "preferredCategories": []}],
        activities=[],  # zero activity, matching the real reported customer
    )
    catalog = load_backend_catalog(client, _resolver(tmp_path))
    user_id_by_guid, guid_by_user_id = load_ai_user_identities(client)
    assert user_id_by_guid[_CUSTOMER_GUID] == _CUSTOMER_USER_ID
    assert guid_by_user_id[_CUSTOMER_USER_ID] == _CUSTOMER_GUID

    raw_users, guid_by_internal = load_backend_users_roster(client, user_id_by_guid, catalog)
    assert len(raw_users) == 1
    assert raw_users[0].id == _CUSTOMER_USER_ID
    assert guid_by_internal[_CUSTOMER_USER_ID] == _CUSTOMER_GUID
