import numpy as np
import pytest

from recommendation.data.schemas.engagement import EngagementProfile, PurchaseRecord
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.features.product_features import build_product_features
from recommendation.ranking.examples import build_ranking_dataset
from recommendation.retrieval.index.faiss_index import FaissVectorIndex
from recommendation.retrieval.two_tower.feature_encoding import TwoTowerFeatureEncoder
from recommendation.retrieval.two_tower.model import build_user_tower
from recommendation.retrieval.two_tower.splitting import UserSplit
from recommendation.utils.config import FeatureConfig, RankingConfig, TwoTowerConfig

_EMBEDDING_DIM = 8
_OUTPUT_DIM = 8


def _product(pid: int, category: str) -> Product:
    return Product(id=pid, category_id=pid, slug=f"p{pid}", name=f"Product {pid}", price=1.0 + pid, category_name=category)


@pytest.fixture
def scenario():
    # User purchased 10..14 (5 distinct); catalog also has 15..19 (never
    # purchased) so there's an actual negative pool to sample from.
    purchased_ids = [10, 11, 12, 13, 14]
    unpurchased_ids = [15, 16, 17, 18, 19]
    all_ids = purchased_ids + unpurchased_ids

    products = [_product(pid, f"Category{pid % 3}") for pid in all_ids]
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])

    rng = np.random.default_rng(0)
    product_embeddings = {pid: rng.normal(size=_EMBEDDING_DIM).astype(np.float32) for pid in all_ids}

    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[
            PurchaseRecord(user_id=1, product_id=pid, order_id=i, quantity=1, unit_price=1.0)
            for i, pid in enumerate(purchased_ids)
        ],
    )
    engagement_profiles = {1: profile}
    split = UserSplit(user_id=1, train_product_ids=[10, 11, 12], val_product_id=13, test_product_id=14)
    splits = {1: split}

    tt_encoder = TwoTowerFeatureEncoder.fit(
        category_names=[p.category_name for p in products],
        brand_names=[],
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

    ranking_config = RankingConfig(negatives_per_positive=2, random_seed=0)

    return dict(
        engagement_profiles=engagement_profiles,
        splits=splits,
        product_lookup=product_lookup,
        product_features=product_features,
        product_embeddings=product_embeddings,
        text_embeddings={},
        feature_config=FeatureConfig(),
        tt_encoder=tt_encoder,
        user_tower=user_tower,
        vector_index=vector_index,
        pool_size=len(all_ids),
        ranking_config=ranking_config,
    )


def _build(scenario):
    return build_ranking_dataset(
        scenario["engagement_profiles"],
        scenario["splits"],
        scenario["product_lookup"],
        scenario["product_features"],
        scenario["product_embeddings"],
        scenario["text_embeddings"],
        scenario["feature_config"],
        scenario["tt_encoder"],
        scenario["user_tower"],
        scenario["vector_index"],
        scenario["pool_size"],
        scenario["ranking_config"],
    )


def test_positives_match_train_product_ids(scenario):
    train_examples, _, _ = _build(scenario)
    positives = sorted(e.product_id for e in train_examples if e.label == 1)
    assert positives == [10, 11, 12]


def test_train_negatives_exclude_positives_and_held_out(scenario):
    train_examples, _, _ = _build(scenario)
    negatives = {e.product_id for e in train_examples if e.label == 0}
    assert negatives.isdisjoint({10, 11, 12})  # never relabel a known positive as negative
    assert negatives.isdisjoint({13, 14})  # val/test must never leak into training as negatives
    assert negatives <= {15, 16, 17, 18, 19}


def test_train_negative_count_respects_ratio_and_pool_limit(scenario):
    train_examples, _, _ = _build(scenario)
    negatives = [e for e in train_examples if e.label == 0]
    # negatives_per_positive=2 * 3 positives = 6 requested, but only 5 non-purchased items exist.
    assert len(negatives) == 5


def test_feature_vectors_have_consistent_shape(scenario):
    train_examples, val_loss_examples, eval_cases = _build(scenario)
    dims = {e.features.shape for e in train_examples + val_loss_examples}
    assert len(dims) == 1
    (dim,) = dims
    assert eval_cases[0].features.shape[1] == dim[0]


def test_eval_cases_built_only_for_evaluable_user(scenario):
    _, _, eval_cases = _build(scenario)
    assert len(eval_cases) == 1
    assert eval_cases[0].user_id == 1
    assert eval_cases[0].val_product_id == 13
    assert eval_cases[0].test_product_id == 14


def test_eval_case_candidates_cover_full_pool(scenario):
    _, _, eval_cases = _build(scenario)
    case = eval_cases[0]
    assert len(case.candidate_ids) == scenario["pool_size"]
    assert len(case.retrieval_scores) == scenario["pool_size"]
    assert set(case.candidate_ids) == {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}


def test_val_loss_examples_use_val_product_never_test_product_as_positive(scenario):
    _, val_loss_examples, _ = _build(scenario)
    positives = {e.product_id for e in val_loss_examples if e.label == 1}
    assert positives == {13}
    assert 14 not in {e.product_id for e in val_loss_examples}


def test_no_train_examples_for_users_without_train_products():
    from recommendation.data.schemas.engagement import EngagementProfile as EP

    profile = EP(user_id=2, profile=UserProfile(user_id=2), purchases=[])
    split = UserSplit(user_id=2, train_product_ids=[], val_product_id=None, test_product_id=None)
    products = [_product(pid, "Cat") for pid in (10, 11)]
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, [], [], [])
    embeddings = {pid: np.zeros(_EMBEDDING_DIM, dtype=np.float32) for pid in (10, 11)}
    tt_encoder = TwoTowerFeatureEncoder.fit([], [], [], [1.0, 2.0], _EMBEDDING_DIM)
    tt_config = TwoTowerConfig(projection_dims=[8, _OUTPUT_DIM], output_dim=_OUTPUT_DIM, category_embedding_dim=2, brand_embedding_dim=2, age_group_embedding_dim=2)
    user_tower = build_user_tower(tt_encoder, tt_config)
    index = FaissVectorIndex()
    index.build([10, 11], np.eye(2, _OUTPUT_DIM, dtype=np.float32))

    train_examples, val_loss_examples, eval_cases = build_ranking_dataset(
        {2: profile}, {2: split}, product_lookup, product_features, embeddings, {}, FeatureConfig(),
        tt_encoder, user_tower, index, 2, RankingConfig(),
    )
    assert train_examples == []
    assert val_loss_examples == []
    assert eval_cases == []
