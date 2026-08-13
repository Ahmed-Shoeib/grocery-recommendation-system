import numpy as np
import pytest

from recommendation.data.schemas.engagement import EngagementProfile, PurchaseRecord
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.features.product_features import build_product_features
from recommendation.features.user_features import build_user_features
from recommendation.ranking.features import RANKING_FEATURE_NAMES
from recommendation.ranking.model import build_ranker_model
from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import build_user_tower
from recommendation.serving.cold_start import HistoryTier
from recommendation.serving.pipeline import generate_recommendations
from recommendation.utils.config import AppConfig, ColdStartConfig, EligibilityConfig, FeatureConfig, RankingConfig, RetrievalConfig, TwoTowerConfig

_EMBEDDING_DIM = 8
_OUTPUT_DIM = 8


def _product(pid: int, category: str, brand: str, is_active: bool = True, stock: int = 10) -> Product:
    return Product(
        id=pid, category_id=pid, slug=f"p{pid}", name=f"Product {pid}", price=5.0 + pid, brand=brand,
        category_name=category, is_active=is_active, stock_quantity=stock,
    )


@pytest.fixture
def scenario():
    # 12 active/in-stock products across 4 categories, plus 2 explicitly
    # ineligible ones (inactive / out of stock) to exercise eligibility.
    products = [_product(100 + i, f"Cat{i % 4}", f"Brand{i % 3}") for i in range(12)]
    products.append(_product(200, "Cat0", "Brand0", is_active=False))
    products.append(_product(201, "Cat1", "Brand1", stock=0))
    all_ids = [p.id for p in products]

    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])

    rng = np.random.default_rng(0)
    product_embeddings = {pid: rng.normal(size=_EMBEDDING_DIM).astype(np.float32) for pid in all_ids}

    tt_encoder = TwoTowerFeatureEncoder.fit(
        category_names=[p.category_name for p in products],
        brand_names=[p.brand for p in products],
        age_groups=[],
        prices=[p.price for p in products],
        embedding_dim=_EMBEDDING_DIM,
    )
    tt_config = TwoTowerConfig(projection_dims=[16, _OUTPUT_DIM], output_dim=_OUTPUT_DIM, category_embedding_dim=4, brand_embedding_dim=4, age_group_embedding_dim=2)
    user_tower = build_user_tower(tt_encoder, tt_config)

    item_embeddings = rng.normal(size=(len(all_ids), _OUTPUT_DIM)).astype(np.float32)
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    vector_index = FaissVectorIndex()
    vector_index.build(all_ids, item_embeddings)

    ranker_model = build_ranker_model(input_dim=len(RANKING_FEATURE_NAMES), config=RankingConfig(hidden_units=[8]))

    config = AppConfig(
        cold_start=ColdStartConfig(strong_history_min_signals=3, sparse_history_min_signals=1),
        eligibility=EligibilityConfig(require_active=True, require_in_stock=True),
        retrieval=RetrievalConfig(backend="faiss", candidate_pool_multiplier=5, min_candidate_pool=len(all_ids)),
    )

    return dict(
        product_lookup=product_lookup,
        product_features=product_features,
        product_embeddings=product_embeddings,
        all_item_ids=all_ids,
        tt_encoder=tt_encoder,
        user_tower=user_tower,
        ranker_model=ranker_model,
        vector_index=vector_index,
        config=config,
    )


def _run(scenario, profile, top_n=5):
    user_features = build_user_features(profile, scenario["product_lookup"], scenario["product_embeddings"], FeatureConfig())
    return generate_recommendations(
        user_features,
        scenario["product_features"],
        scenario["product_embeddings"],
        scenario["all_item_ids"],
        scenario["tt_encoder"],
        scenario["user_tower"],
        scenario["ranker_model"],
        scenario["vector_index"],
        scenario["config"],
        top_n,
    )


def test_strong_history_user_uses_personalized_source(scenario):
    profile = EngagementProfile(
        user_id=1, profile=UserProfile(user_id=1),
        purchases=[PurchaseRecord(user_id=1, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    result = _run(scenario, profile)
    assert result.tier == HistoryTier.STRONG
    assert result.product_ids
    assert all(source == "personalized" for source in result.sources)


def test_sparse_history_user_blends_sources(scenario):
    profile = EngagementProfile(
        user_id=2, profile=UserProfile(user_id=2, preferred_category="Cat2"),
        purchases=[PurchaseRecord(user_id=2, product_id=100, order_id=0, quantity=1, unit_price=1.0)],
    )
    result = _run(scenario, profile)
    assert result.tier == HistoryTier.SPARSE
    assert result.product_ids
    # A blend should be capable of drawing from more than just personalized.
    assert set(result.sources) <= {"personalized", "preferred_category", "popularity"}


def test_no_history_user_uses_fallback_only(scenario):
    profile = EngagementProfile(user_id=3, profile=UserProfile(user_id=3, preferred_category="Cat1"))
    result = _run(scenario, profile)
    assert result.tier == HistoryTier.NO_HISTORY
    assert result.product_ids
    assert set(result.sources) <= {"preferred_category", "category_popularity", "global_popularity"}
    assert "personalized" not in result.sources


def test_no_history_user_without_preferred_category_falls_back_to_global(scenario):
    profile = EngagementProfile(user_id=4, profile=UserProfile(user_id=4))
    result = _run(scenario, profile)
    assert result.tier == HistoryTier.NO_HISTORY
    assert result.product_ids
    assert set(result.sources) <= {"global_popularity"}


def test_eligibility_excludes_inactive_and_out_of_stock(scenario):
    profile = EngagementProfile(user_id=5, profile=UserProfile(user_id=5))
    result = _run(scenario, profile, top_n=20)
    assert 200 not in result.product_ids
    assert 201 not in result.product_ids
    assert result.num_excluded_by_eligibility >= 0  # 200/201 may or may not have been retrieved into the pool


def test_result_has_no_duplicate_product_ids(scenario):
    profile = EngagementProfile(
        user_id=6, profile=UserProfile(user_id=6, preferred_category="Cat0"),
        purchases=[PurchaseRecord(user_id=6, product_id=100, order_id=0, quantity=1, unit_price=1.0)],
    )
    result = _run(scenario, profile)
    assert len(result.product_ids) == len(set(result.product_ids))


def test_fill_rate_is_one_when_pool_has_enough_eligible_items(scenario):
    profile = EngagementProfile(
        user_id=7, profile=UserProfile(user_id=7),
        purchases=[PurchaseRecord(user_id=7, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    result = _run(scenario, profile, top_n=5)
    assert result.fill_rate == pytest.approx(1.0)
    assert len(result.product_ids) == 5


def test_generate_recommendations_logs_observability_fields(scenario, caplog):
    """Phase 10 observability: tier/fill_rate/pool_size/eligibility-
    exclusion counts must be logged from the shared pipeline function
    itself, so both the API and the dashboard get identical, non-
    duplicated visibility without either caller adding its own logging.
    """
    import logging

    profile = EngagementProfile(
        user_id=9, profile=UserProfile(user_id=9),
        purchases=[PurchaseRecord(user_id=9, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    with caplog.at_level(logging.INFO, logger="recommendation.serving.pipeline"):
        _run(scenario, profile, top_n=5)

    messages = [r.message for r in caplog.records]
    assert any("recommendation generated" in m and "user_id=9" in m and "tier=strong" in m for m in messages)


def test_vector_index_candidate_missing_from_catalog_is_skipped_not_crashed(scenario):
    """Phase 10 reliability: if the VectorIndex (built from Two-Tower
    artifacts) contains an item id absent from the live product catalog
    - artifact/catalog drift - a request must degrade gracefully (skip
    that one candidate), not raise and fail the whole request.
    """
    drifted_id = 999
    assert drifted_id not in scenario["product_features"]

    drifted_ids = scenario["all_item_ids"] + [drifted_id]
    rng = np.random.default_rng(1)
    drifted_embeddings = rng.normal(size=(len(drifted_ids), _OUTPUT_DIM)).astype(np.float32)
    drifted_embeddings /= np.linalg.norm(drifted_embeddings, axis=1, keepdims=True)

    from recommendation.retrieval.index.faiss_index import FaissVectorIndex

    drifted_index = FaissVectorIndex()
    drifted_index.build(drifted_ids, drifted_embeddings)
    scenario["vector_index"] = drifted_index
    scenario["all_item_ids"] = drifted_ids

    profile = EngagementProfile(
        user_id=8, profile=UserProfile(user_id=8),
        purchases=[PurchaseRecord(user_id=8, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    result = _run(scenario, profile, top_n=20)
    assert drifted_id not in result.product_ids
    assert result.product_ids  # still produced real recommendations from the valid candidates
