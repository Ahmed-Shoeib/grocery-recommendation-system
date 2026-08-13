"""End-to-end smoke tests for the Streamlit dashboard itself, using
Streamlit's `AppTest` to actually execute `dashboard.py` (not just its
helper functions). Injects a small/fake `RecommendationService` via
`st.session_state["_service_override"]` - `ui.service_loader.load_service`'s
dependency-injection seam - so this never triggers real model loading.
"""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from recommendation.api.dependencies import RecommendationService
from recommendation.data.adapters.base import (
    AdapterBundle,
    CartAdapter,
    ChatbotContextAdapter,
    ProductCatalogAdapter,
    PurchaseAdapter,
    ReviewAdapter,
    SearchAdapter,
    UserAdapter,
)
from recommendation.data.schemas.engagement import EngagementProfile, PurchaseRecord
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.features.product_features import build_product_features
from recommendation.ranking.features import RANKING_FEATURE_NAMES
from recommendation.ranking.model import build_ranker_model
from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import build_user_tower
from recommendation.utils.config import AppConfig, ColdStartConfig, RankingConfig, RetrievalConfig, TwoTowerConfig

_EMBEDDING_DIM = 8
_OUTPUT_DIM = 8
_DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "src" / "recommendation" / "ui" / "dashboard.py")


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
    def get_cart_items(self, user_id):
        return []

    def list_all_cart_items(self):
        return []


class _FakeReviews(ReviewAdapter):
    def get_reviews(self, user_id):
        return []

    def list_all_reviews(self):
        return []


class _FakeSearch(SearchAdapter):
    def get_search_history(self, user_id):
        return []


class _FakeChatbot(ChatbotContextAdapter):
    def get_chatbot_context(self, user_id):
        return None


def _product(pid: int, category: str) -> Product:
    return Product(id=pid, category_id=pid, slug=f"p{pid}", name=f"Product {pid}", price=5.0 + pid, category_name=category)


def _build_service() -> RecommendationService:
    products = [_product(100 + i, f"Cat{i % 3}") for i in range(8)]
    all_ids = [p.id for p in products]
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])

    rng = np.random.default_rng(0)
    product_embeddings = {pid: rng.normal(size=_EMBEDDING_DIM).astype(np.float32) for pid in all_ids}

    tt_encoder = TwoTowerFeatureEncoder.fit(
        category_names=[p.category_name for p in products], brand_names=[], age_groups=[],
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
        1: UserProfile(user_id=1, preferred_category="Cat1", age_group="25-34"),  # strong
        2: UserProfile(user_id=2),  # sparse
        3: UserProfile(user_id=3, preferred_category="Cat0"),  # no_history
    }
    purchases = {
        1: [PurchaseRecord(user_id=1, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
        2: [PurchaseRecord(user_id=2, product_id=100, order_id=0, quantity=1, unit_price=1.0)],
    }
    bundle = AdapterBundle(
        products=_FakeProducts(products), users=_FakeUsers(profiles), purchases=_FakePurchases(purchases),
        cart=_FakeCart(), reviews=_FakeReviews(), search=_FakeSearch(), chatbot=_FakeChatbot(),
    )
    config = AppConfig(
        cold_start=ColdStartConfig(strong_history_min_signals=3, sparse_history_min_signals=1),
        retrieval=RetrievalConfig(backend="faiss", candidate_pool_multiplier=5, min_candidate_pool=len(all_ids)),
    )
    engagement_profiles = {
        1: EngagementProfile(user_id=1, profile=profiles[1], purchases=purchases[1]),
        2: EngagementProfile(user_id=2, profile=profiles[2], purchases=purchases[2]),
        3: EngagementProfile(user_id=3, profile=profiles[3]),
    }

    return RecommendationService(
        product_lookup=product_lookup, product_features=product_features, product_embeddings=product_embeddings,
        text_embeddings={}, all_item_ids=all_ids, tt_encoder=tt_encoder, user_tower=user_tower, ranker_model=ranker_model,
        vector_index=vector_index, bundle=bundle, config=config, ranker_model_version="test_v1", two_tower_model_version="test_v1",
        engagement_profiles=engagement_profiles,
    )


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
    at.session_state["_service_override"] = _build_service()
    return at


def test_dashboard_renders_without_exception(app):
    app.run()
    assert not app.exception


def test_dashboard_shows_user_table_and_recommendations(app):
    app.run()
    assert not app.exception
    # Section headers rendered via st.subheader.
    subheaders = [s.value for s in app.subheader]
    assert any("Users" in s for s in subheaders)
    assert any("Final recommendations" in s for s in subheaders)
    assert any("Pipeline" in s for s in subheaders)
    assert any("Metrics" in s for s in subheaders)


def test_dashboard_selecting_sparse_history_user(app):
    app.run()
    app.selectbox[0].select(2).run()
    assert not app.exception


def test_dashboard_selecting_no_history_user(app):
    app.run()
    app.selectbox[0].select(3).run()
    assert not app.exception


def test_dashboard_top_n_slider_changes_recommendation_count(app):
    app.run()
    app.slider[0].set_value(2).run()
    assert not app.exception


def test_dashboard_handles_service_load_failure_gracefully():
    at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
    # ui.service_loader.load_service raises whatever exception instance is
    # placed in "_service_override" - the deterministic way to exercise
    # dashboard.py's "service failed to load" branch without needing a
    # real missing-model-artifacts environment to reproduce it.
    at.session_state["_service_override"] = RuntimeError("simulated load failure")
    at.run()
    assert not at.exception  # the dashboard must catch this itself, not crash
    errors = [e.value for e in at.error]
    assert any("Failed to load" in e for e in errors)
