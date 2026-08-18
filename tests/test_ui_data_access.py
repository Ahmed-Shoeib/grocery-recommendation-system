import numpy as np
import pytest

from recommendation.api.dependencies import RecommendationService
from recommendation.data.adapters.base import (
    AdapterBundle,
    CartAdapter,
    ChatbotContextAdapter,
    ClickAdapter,
    ProductCatalogAdapter,
    PurchaseAdapter,
    ReviewAdapter,
    SearchAdapter,
    UserAdapter,
)
from recommendation.data.schemas.engagement import (
    CartAffinityRecord,
    ChatbotContextRecord,
    ClickRecord,
    EngagementProfile,
    PurchaseRecord,
    SearchRecord,
)
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.features.product_features import build_product_features
from recommendation.ranking.features import RANKING_FEATURE_NAMES
from recommendation.ranking.model import build_ranker_model
from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import build_user_tower
from recommendation.serving.pipeline import generate_recommendations
from recommendation.ui.data_access import (
    format_cart_items,
    format_chatbot_context,
    format_clicks,
    format_purchases,
    format_recommendation_table,
    format_searches,
    list_users,
    load_user_detail,
)
from recommendation.utils.config import AppConfig, ColdStartConfig, RankingConfig, RetrievalConfig, TwoTowerConfig

_EMBEDDING_DIM = 8
_OUTPUT_DIM = 8


class _FakeProducts(ProductCatalogAdapter):
    def __init__(self, products):
        self._products = products
        self._by_id = {p.id: p for p in products}

    def list_products(self):
        return self._products

    def get_product(self, product_id):
        return self._by_id.get(product_id)


class _FakeUsers(UserAdapter):
    def __init__(self, profiles):
        self._profiles = profiles

    def get_user_profile(self, user_id):
        return self._profiles.get(user_id)

    def list_user_ids(self):
        return list(self._profiles.keys())


class _FakePurchases(PurchaseAdapter):
    def __init__(self, by_user):
        self._by_user = by_user

    def get_purchases(self, user_id):
        return self._by_user.get(user_id, [])

    def list_all_purchases(self):
        return [p for records in self._by_user.values() for p in records]


class _FakeCart(CartAdapter):
    def __init__(self, by_user=None):
        self._by_user = by_user or {}

    def get_cart_items(self, user_id):
        return self._by_user.get(user_id, [])

    def list_all_cart_items(self):
        return [c for records in self._by_user.values() for c in records]


class _FakeClicks(ClickAdapter):
    def __init__(self, by_user=None):
        self._by_user = by_user or {}

    def get_clicks(self, user_id):
        return self._by_user.get(user_id, [])

    def list_all_clicks(self):
        return [c for records in self._by_user.values() for c in records]


class _FakeReviews(ReviewAdapter):
    def get_reviews(self, user_id):
        return []

    def list_all_reviews(self):
        return []


class _FakeSearch(SearchAdapter):
    def __init__(self, by_user=None):
        self._by_user = by_user or {}

    def get_search_history(self, user_id):
        return self._by_user.get(user_id, [])


class _FakeChatbot(ChatbotContextAdapter):
    def __init__(self, by_user=None):
        self._by_user = by_user or {}

    def get_chatbot_context(self, user_id):
        return self._by_user.get(user_id)


def _product(pid: int, category: str, brand: str | None = None) -> Product:
    return Product(id=pid, category_id=pid, slug=f"p{pid}", name=f"Product {pid}", price=5.0 + pid, brand=brand, category_name=category)


@pytest.fixture
def service() -> RecommendationService:
    products = [_product(100 + i, f"Cat{i % 3}", f"Brand{i % 2}") for i in range(8)]
    all_ids = [p.id for p in products]
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])

    rng = np.random.default_rng(0)
    product_embeddings = {pid: rng.normal(size=_EMBEDDING_DIM).astype(np.float32) for pid in all_ids}

    tt_encoder = TwoTowerFeatureEncoder.fit(
        category_names=[p.category_name for p in products], brand_names=[p.brand for p in products], age_groups=[],
        prices=[p.price for p in products], embedding_dim=_EMBEDDING_DIM,
    )
    tt_config = TwoTowerConfig(projection_dims=[16, _OUTPUT_DIM], output_dim=_OUTPUT_DIM, category_embedding_dim=4, brand_embedding_dim=4, age_group_embedding_dim=2)
    user_tower = build_user_tower(tt_encoder, tt_config)

    item_embeddings = rng.normal(size=(len(all_ids), _OUTPUT_DIM)).astype(np.float32)
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    vector_index = FaissVectorIndex()
    vector_index.build(all_ids, item_embeddings)
    ranker_model = build_ranker_model(input_dim=len(RANKING_FEATURE_NAMES), config=RankingConfig(hidden_units=[8]))

    profiles = {
        1: UserProfile(user_id=1, preferred_category="Cat1", age_group="25-34"),
        2: UserProfile(user_id=2),  # no preferred_category/age_group on record
    }
    purchases = {1: [PurchaseRecord(user_id=1, product_id=100, order_id=0, quantity=2, unit_price=5.0, order_status="DELIVERED")]}
    cart = {1: [CartAffinityRecord(user_id=1, product_id=101, quantity=1)]}
    clicks = {1: [ClickRecord(user_id=1, product_id=104)]}
    searches = {1: [SearchRecord(user_id=1, search_term="milk", matched_product_id=102), SearchRecord(user_id=1, search_term="cheap snacks")]}
    chatbot = {1: ChatbotContextRecord(user_id=1, summary="Looking for breakfast items", preferred_category="Cat0", mentioned_product_ids=[103], keywords=["breakfast", "quick"])}

    bundle = AdapterBundle(
        products=_FakeProducts(products),
        users=_FakeUsers(profiles),
        purchases=_FakePurchases(purchases),
        cart=_FakeCart(cart),
        clicks=_FakeClicks(clicks),
        reviews=_FakeReviews(),
        search=_FakeSearch(searches),
        chatbot=_FakeChatbot(chatbot),
    )

    config = AppConfig(
        cold_start=ColdStartConfig(strong_history_min_signals=3, sparse_history_min_signals=1),
        retrieval=RetrievalConfig(backend="faiss", candidate_pool_multiplier=5, min_candidate_pool=len(all_ids)),
    )

    engagement_profiles = {
        1: EngagementProfile(user_id=1, profile=profiles[1], clicks=clicks[1], purchases=purchases[1], cart_items=cart[1], searches=searches[1], chatbot_context=chatbot[1]),
        2: EngagementProfile(user_id=2, profile=profiles[2]),
    }

    return RecommendationService(
        product_lookup=product_lookup,
        product_features=product_features,
        product_embeddings=product_embeddings,
        text_embeddings={},
        all_item_ids=all_ids,
        tt_encoder=tt_encoder,
        user_tower=user_tower,
        ranker_model=ranker_model,
        vector_index=vector_index,
        bundle=bundle,
        config=config,
        ranker_model_version="test",
        two_tower_model_version="test",
        engagement_profiles=engagement_profiles,
    )


# --- list_users --------------------------------------------------------

def test_list_users_includes_all_known_users_sorted(service):
    rows = list_users(service)
    assert [r.user_id for r in rows] == [1, 2]


def test_list_users_shows_profile_fields_when_present(service):
    rows = list_users(service)
    row1 = next(r for r in rows if r.user_id == 1)
    assert row1.preferred_category == "Cat1"
    assert row1.age_group == "25-34"


def test_list_users_none_when_profile_fields_missing(service):
    rows = list_users(service)
    row2 = next(r for r in rows if r.user_id == 2)
    assert row2.preferred_category is None
    assert row2.age_group is None


# --- load_user_detail ----------------------------------------------------

def test_load_user_detail_returns_none_for_unknown_user(service):
    assert load_user_detail(service, 999) is None


def test_load_user_detail_strong_history_tier(service):
    detail = load_user_detail(service, 1)
    assert detail is not None
    assert detail.tier.value in ("strong", "sparse")  # depends on exact signal count, both are "has history"
    assert detail.features.purchase_count == 1


def test_load_user_detail_no_history_tier_for_empty_profile(service):
    detail = load_user_detail(service, 2)
    assert detail.tier.value == "no_history"


# --- formatting: engagement signals -----------------------------------

def test_format_purchases_includes_product_name(service):
    detail = load_user_detail(service, 1)
    rows = format_purchases(detail.engagement, service.product_lookup)
    assert rows == [{"product_id": 100, "product_name": "Product 100", "quantity": 2, "unit_price": 5.0, "order_id": 0, "order_status": "DELIVERED"}]


def test_format_purchases_empty_for_no_history_user(service):
    detail = load_user_detail(service, 2)
    assert format_purchases(detail.engagement, service.product_lookup) == []


def test_format_clicks(service):
    detail = load_user_detail(service, 1)
    rows = format_clicks(detail.engagement, service.product_lookup)
    assert rows == [{"product_id": 104, "product_name": "Product 104"}]


def test_format_clicks_empty_for_no_history_user(service):
    detail = load_user_detail(service, 2)
    assert format_clicks(detail.engagement, service.product_lookup) == []


def test_format_cart_items(service):
    detail = load_user_detail(service, 1)
    rows = format_cart_items(detail.engagement, service.product_lookup)
    assert rows == [{"product_id": 101, "product_name": "Product 101", "quantity": 1}]


def test_format_searches_includes_matched_product_when_present(service):
    detail = load_user_detail(service, 1)
    rows = format_searches(detail.engagement, service.product_lookup)
    assert rows[0] == {"search_term": "milk", "matched_product": "Product 102"}
    assert rows[1] == {"search_term": "cheap snacks", "matched_product": None}


def test_format_chatbot_context_present(service):
    detail = load_user_detail(service, 1)
    chatbot = format_chatbot_context(detail.engagement, service.product_lookup)
    assert chatbot["summary"] == "Looking for breakfast items"
    assert chatbot["preferred_category"] == "Cat0"
    assert chatbot["mentioned_products"] == ["Product 103"]
    assert chatbot["keywords"] == "breakfast, quick"


def test_format_chatbot_context_none_when_absent(service):
    detail = load_user_detail(service, 2)
    assert format_chatbot_context(detail.engagement, service.product_lookup) is None


# --- recommendations -----------------------------------------------------
# STEP 9 (docs/data-mapping.md section 18): `ui.data_access.run_recommendations`
# was removed - the live recommendation call now goes through
# `api.routes.get_recommendations` -> `RecommendationService.recommend` ->
# `serving.pipeline.recommend` directly. These tests build a
# `RecommendationResult` via `generate_recommendations` (the same function
# the pipeline itself calls) to keep exercising `format_recommendation_table`,
# which IS still a `ui.data_access` responsibility (used server-side by the
# API route).


def _recommend(service, detail, top_n):
    return generate_recommendations(
        detail.features, service.product_features, service.product_embeddings, service.all_item_ids,
        service.tt_encoder, service.user_tower, service.ranker_model, service.vector_index, service.config, top_n,
    )


def test_format_recommendation_table_joins_catalog_info(service):
    detail = load_user_detail(service, 1)
    result = _recommend(service, detail, top_n=3)
    rows = format_recommendation_table(result, service.product_lookup)
    assert len(rows) == len(result.product_ids)
    for row in rows:
        assert row["name"].startswith("Product")
        assert row["category"] is not None
        assert "rank" in row and "score" in row and "source" in row
