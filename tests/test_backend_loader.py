"""Backend DTO -> canonical Raw* / UserInteraction mapping and the
activity-drop policy (docs/data-mapping.md section 19).
"""

from datetime import datetime

from recommendation.data.backend.dtos import ApiActivity
from recommendation.data.backend.errors import BackendAuthError
from recommendation.data.backend.identity import ExternalIdentityResolver
from recommendation.data.backend.loader import (
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
    assert all(u.preferred_category_id is None and u.age_group is None for u in users)
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
    assert u.preferred_category_id == catalog.category_id_by_slug["groceries"]


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
# {reviewId, userId, productId, rating, comment, createdAt, updatedAt},
# where userId/productId are the backend's int32 primary keys.


def _catalog_with_backend_ids(tmp_path, mapping):
    """A loaded catalog plus the backend-id -> internal-id map that
    `/api/products` will populate once it exposes its numeric id. Lets the
    resolution path be tested today without inventing a DTO field.
    """
    catalog = load_backend_catalog(FakeBackendClient(products=_PRODS, categories=_CATS), _resolver(tmp_path))
    catalog.product_id_by_backend_id.update(mapping)
    return catalog


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
            {"reviewId": 42, "userId": 7, "productId": 3, "rating": 4, "comment": "good",
             "createdAt": "2026-09-01T10:00:00", "updatedAt": None},
        ]),
        catalog,
        {5: "7"},  # internal user 5 <-> the backend user key "7"
    )
    assert len(reviews) == 1
    review = reviews[0]
    assert (review.id, review.user_id, review.product_id, review.rating) == (42, 5, 11, 4.0)
    assert review.comment == "good"
    assert review.creation_date == datetime(2026, 9, 1, 10, 0, 0)


def test_unknown_user_is_dropped_never_minted(tmp_path):
    """A review by a user with no recorded activity must not create a
    phantom user id (the activity stream defines the served population).
    """
    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 999, "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "7"},
    )
    assert reviews == []


def test_malformed_ratings_and_ids_are_dropped_without_raising(tmp_path):
    """`RawReview.rating` is ge=1/le=5 - an out-of-range row must be
    counted and skipped, never abort the whole load.
    """
    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "productId": 3, "rating": 0, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 2, "userId": 7, "productId": 3, "rating": 9, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 3, "userId": 7, "productId": 3, "rating": None, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": None, "userId": 7, "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 5, "userId": 7, "productId": None, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
            {"reviewId": 6, "userId": 7, "productId": 3, "rating": 5, "createdAt": "2026-09-01T10:00:00"},
        ]),
        catalog,
        {5: "7"},
    )
    assert [r.id for r in reviews] == [6]


def test_missing_and_aware_timestamps_are_normalized_to_naive_utc(tmp_path):
    catalog = _catalog_with_backend_ids(tmp_path, {3: 11})
    reviews = load_backend_reviews(
        FakeBackendClient(reviews=[
            {"reviewId": 1, "userId": 7, "productId": 3, "rating": 5, "createdAt": None},
            {"reviewId": 2, "userId": 7, "productId": 3, "rating": 5, "createdAt": "2026-09-01T12:00:00+02:00"},
        ]),
        catalog,
        {5: "7"},
    )
    assert reviews[0].creation_date is None
    # +02:00 -> naive UTC, matching the activity-stream convention.
    assert reviews[1].creation_date == datetime(2026, 9, 1, 10, 0, 0)


def test_auth_failure_degrades_instead_of_failing_the_load(tmp_path):
    class Unauthorized(FakeBackendClient):
        def list_reviews(self):
            raise BackendAuthError("GET /api/reviews returned 401", status_code=401)

    catalog = _catalog_with_backend_ids(tmp_path, {})
    assert load_backend_reviews(Unauthorized(), catalog, {1: "g1"}) == []
