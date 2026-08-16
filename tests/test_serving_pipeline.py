import dataclasses

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
    """Phase 11: 200 (inactive) and 201 (out of stock) are excluded by the
    HARD PRE-RETRIEVAL gate, not the final validation - they never reach
    candidate generation at all, so the final-validation exclusion count
    stays 0 and the pre-retrieval count reflects exactly these two.
    """
    profile = EngagementProfile(user_id=5, profile=UserProfile(user_id=5))
    result = _run(scenario, profile, top_n=20)
    assert 200 not in result.product_ids
    assert 201 not in result.product_ids
    assert result.num_excluded_pre_retrieval == 2
    assert result.num_excluded_by_eligibility == 0


def test_pre_retrieval_gate_excludes_inactive_and_out_of_stock_from_candidate_generation(scenario):
    """Requirement: inactive/out-of-stock products must not PARTICIPATE in
    retrieval candidate generation at all - checked against the pool
    BEFORE re-ranking/final-validation (`pre_rerank_product_ids`), not
    just the final Top-N.
    """
    profile = EngagementProfile(
        user_id=13, profile=UserProfile(user_id=13),
        purchases=[PurchaseRecord(user_id=13, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    result = _run(scenario, profile, top_n=20)
    assert 200 not in result.pre_rerank_product_ids
    assert 201 not in result.pre_rerank_product_ids
    assert 200 not in result.pre_eligibility_product_ids
    assert 201 not in result.pre_eligibility_product_ids


def test_fallback_no_history_never_surfaces_inactive_or_out_of_stock(scenario):
    """Pre-retrieval eligibility applies to the NO_HISTORY fallback
    (category/global popularity) too, not just personalized Two-Tower
    retrieval - deliberately picks preferred categories ("Cat0"/"Cat1")
    that contain the inactive (200) / out-of-stock (201) products, to
    actually exercise the fallback ranking functions with them present.
    """
    profile = EngagementProfile(user_id=14, profile=UserProfile(user_id=14, preferred_category="Cat0"))
    result = _run(scenario, profile, top_n=20)
    assert result.tier.value == "no_history"
    assert 200 not in result.pre_rerank_product_ids
    assert 200 not in result.product_ids

    profile2 = EngagementProfile(user_id=15, profile=UserProfile(user_id=15, preferred_category="Cat1"))
    result2 = _run(scenario, profile2, top_n=20)
    assert 201 not in result2.pre_rerank_product_ids
    assert 201 not in result2.product_ids


def test_eligible_products_retrieve_normally(scenario):
    """Sanity check: pre-retrieval eligibility only excludes the two
    deliberately-ineligible products - every active/in-stock product is
    still a normal, unaffected candidate.
    """
    profile = EngagementProfile(
        user_id=16, profile=UserProfile(user_id=16),
        purchases=[PurchaseRecord(user_id=16, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    result = _run(scenario, profile, top_n=20)
    assert result.product_ids
    assert set(result.product_ids) <= {100 + i for i in range(12)}
    for pid in result.product_ids:
        pf = scenario["product_features"][pid]
        assert pf.is_active and pf.stock_quantity > 0


def test_fill_rate_drops_when_fewer_eligible_products_than_requested(scenario):
    """12 eligible products exist in the catalog; requesting more than
    that must honestly return fewer than requested, not pad/crash.
    """
    profile = EngagementProfile(
        user_id=17, profile=UserProfile(user_id=17),
        purchases=[PurchaseRecord(user_id=17, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    result = _run(scenario, profile, top_n=20)
    assert len(result.product_ids) == 12
    assert result.fill_rate == pytest.approx(12 / 20)
    assert result.num_excluded_pre_retrieval == 2


def test_stock_and_active_changes_do_not_require_index_or_model_rebuild(scenario):
    """Requirement: stock/isActive changes must not require retraining the
    Two-Tower or ranker, or rebuilding the VectorIndex - only refreshing
    `product_features`. Proven by reusing the exact same VectorIndex/
    user_tower/ranker_model object instances (never rebuilt/retrained
    between the two calls) and only swapping the `product_features` dict.
    """
    profile = EngagementProfile(
        user_id=18, profile=UserProfile(user_id=18),
        purchases=[PurchaseRecord(user_id=18, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    vector_index = scenario["vector_index"]
    user_tower = scenario["user_tower"]
    ranker_model = scenario["ranker_model"]
    size_before = vector_index.size

    result_before = _run(scenario, profile, top_n=20)
    assert 100 in result_before.product_ids  # all 12 eligible items fit within top_n=20

    mutated_features = dict(scenario["product_features"])
    mutated_features[100] = dataclasses.replace(mutated_features[100], stock_quantity=0)
    scenario_after = dict(scenario, product_features=mutated_features)
    result_after = _run(scenario_after, profile, top_n=20)

    # The exact same (never rebuilt) index/model objects were reused.
    assert scenario["vector_index"] is vector_index
    assert scenario["user_tower"] is user_tower
    assert scenario["ranker_model"] is ranker_model
    assert vector_index.size == size_before

    assert 100 not in result_after.product_ids
    assert len(result_after.product_ids) == 11
    assert result_after.num_excluded_pre_retrieval == result_before.num_excluded_pre_retrieval + 1


def test_final_validation_catches_product_that_becomes_unavailable_after_retrieval(scenario):
    """Requirement: the final lightweight validation protects against a
    product becoming unavailable between pre-retrieval filtering and the
    final response. Simulated via `final_product_features`: product 100
    is eligible in `product_features` (used for pre-retrieval + ranking,
    so it IS retrieved/ranked normally), but a fresher snapshot passed
    only to the final-validation stage shows it as inactive - proving the
    final check independently re-validates rather than trusting
    pre-retrieval filtering.
    """
    from recommendation.serving.pipeline import generate_recommendations

    profile = EngagementProfile(
        user_id=19, profile=UserProfile(user_id=19),
        purchases=[PurchaseRecord(user_id=19, product_id=pid, order_id=i, quantity=1, unit_price=1.0) for i, pid in enumerate([100, 101, 102])],
    )
    user_features = build_user_features(profile, scenario["product_lookup"], scenario["product_embeddings"], FeatureConfig())

    fresher_features = dict(scenario["product_features"])
    fresher_features[100] = dataclasses.replace(fresher_features[100], is_active=False)

    result = generate_recommendations(
        user_features, scenario["product_features"], scenario["product_embeddings"], scenario["all_item_ids"],
        scenario["tt_encoder"], scenario["user_tower"], scenario["ranker_model"], scenario["vector_index"],
        scenario["config"], 20, final_product_features=fresher_features,
    )

    assert 100 not in result.product_ids
    assert result.num_excluded_by_eligibility >= 1
    assert "is_active" in result.excluded_reasons.get(100, [])
    # Pre-retrieval filtering (against the ORIGINAL, not-yet-stale snapshot)
    # still let it through as a candidate - only the final check caught it.
    assert 100 in result.pre_eligibility_product_ids


def test_no_eligible_products_returns_empty_result_gracefully(scenario):
    """If literally nothing in the catalog is eligible, the pipeline must
    return an honest, empty, zero-fill-rate result - not raise.
    """
    profile = EngagementProfile(user_id=20, profile=UserProfile(user_id=20))
    user_features = build_user_features(profile, scenario["product_lookup"], scenario["product_embeddings"], FeatureConfig())
    all_ineligible = {pid: dataclasses.replace(pf, is_active=False) for pid, pf in scenario["product_features"].items()}

    from recommendation.serving.pipeline import generate_recommendations

    result = generate_recommendations(
        user_features, all_ineligible, scenario["product_embeddings"], scenario["all_item_ids"],
        scenario["tt_encoder"], scenario["user_tower"], scenario["ranker_model"], scenario["vector_index"],
        scenario["config"], 10,
    )
    assert result.product_ids == []
    assert result.fill_rate == 0.0
    assert result.num_excluded_pre_retrieval == len(all_ineligible)
    assert result.num_excluded_by_eligibility == 0


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
