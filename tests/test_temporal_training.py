"""Tests for `recommendation.evaluation.temporal_training` (STEP 7):
temporal-consistent Two-Tower/ranker training-example construction on top
of the STEP 5 temporal splits, and the leakage-safe negative-sampling
exclusion set.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from recommendation.data.adapters.base import UserAdapter
from recommendation.data.adapters.review_adapter import InMemoryReviewAdapter
from recommendation.data.schemas.events import ActionType, UserInteraction
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.evaluation.temporal_future_purchase import (
    DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT,
    build_temporal_splits,
    group_events_by_user,
)
from recommendation.evaluation.temporal_training import (
    all_purchased_product_ids_by_user,
    build_temporal_ranking_dataset,
    build_temporal_two_tower_examples,
    evaluate_temporal_retrieval,
)
from recommendation.utils.config import FeatureConfig

T0 = datetime(2026, 1, 1)


def t(days: int) -> datetime:
    return T0 + timedelta(days=days)


def ev(user_id: int, product_id: int, action_type: ActionType, when: datetime) -> UserInteraction:
    return UserInteraction(user_id=user_id, product_id=product_id, action_type=action_type, action_time=when)


class _FakeUsers(UserAdapter):
    def __init__(self, profiles: dict[int, UserProfile]) -> None:
        self._profiles = profiles

    def get_user_profile(self, user_id: int):
        return self._profiles.get(user_id)

    def list_user_ids(self) -> list[int]:
        return list(self._profiles.keys())


def _product_lookup() -> dict[int, Product]:
    return {
        i: Product(id=i, category_id=1, slug=f"p{i}", name=f"P{i}", price=float(i), category_name="X")
        for i in range(1, 30)
    }


@pytest.fixture
def users_adapter() -> _FakeUsers:
    return _FakeUsers({uid: UserProfile(user_id=uid) for uid in (1, 2, 3, 4)})


@pytest.fixture
def reviews_adapter() -> InMemoryReviewAdapter:
    return InMemoryReviewAdapter([])


def _build_scenario():
    events = [
        # User 1: FULL tier, 4 purchases -> test_cutoff=t(20), val_cutoff=t(10)
        ev(1, 1, ActionType.PURCHASE, t(1)),
        ev(1, 2, ActionType.PURCHASE, t(5)),
        ev(1, 3, ActionType.PURCHASE, t(10)),  # val target
        ev(1, 4, ActionType.PURCHASE, t(20)),  # test target
        # User 2: VAL_ONLY, 2 purchases -> val_cutoff=t(8)
        ev(2, 5, ActionType.PURCHASE, t(2)),
        ev(2, 6, ActionType.PURCHASE, t(8)),  # val target
        # User 3: INSUFFICIENT_DEPTH, 1 purchase -> no cutoff, train-only
        ev(3, 7, ActionType.PURCHASE, t(3)),
        # User 4: engagement, no purchase
        ev(4, 8, ActionType.CLICK, t(1)),
    ]
    events_by_user = group_events_by_user(events)
    user_ids = [1, 2, 3, 4]
    splits = build_temporal_splits(events_by_user, user_ids, DEFAULT_MIN_PURCHASE_EVENTS_FOR_FULL_SPLIT)
    return events_by_user, splits


def test_train_examples_count_and_membership(users_adapter, reviews_adapter):
    events_by_user, splits = _build_scenario()
    train_examples, val_loss_examples, val_cases, test_cases = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    # user1: purchases before val_cutoff=t(10) -> products 1, 2 (2 examples)
    # user2: purchase before val_cutoff=t(8) -> product 5 (1 example)
    # user3: no val_cutoff -> its single purchase (product 7) is train-eligible (1 example)
    # user4: zero purchases -> zero examples
    assert len(train_examples) == 4
    train_pairs = {(e.user_id, e.product_id) for e in train_examples}
    assert train_pairs == {(1, 1), (1, 2), (2, 5), (3, 7)}


def test_train_examples_never_include_val_or_test_target_products(users_adapter, reviews_adapter):
    events_by_user, splits = _build_scenario()
    train_examples, _, _, _ = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    train_products_user1 = {e.product_id for e in train_examples if e.user_id == 1}
    assert 3 not in train_products_user1  # val target
    assert 4 not in train_products_user1  # test target


def test_each_training_example_is_point_in_time_not_shared_per_user_cutoff(users_adapter, reviews_adapter):
    """User 1's SECOND training purchase (t(5)) must see the FIRST (t(1))
    as history, but the FIRST training purchase (t(1)) must NOT see the
    second - each example gets its OWN cutoff (its own purchase time),
    not one shared val_cutoff for the whole user.
    """
    events_by_user, splits = _build_scenario()
    train_examples, _, _, _ = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    by_product = {e.product_id: e for e in train_examples if e.user_id == 1}
    assert by_product[1].user_features.purchase_count == 0  # nothing before t(1)
    assert by_product[2].user_features.purchase_count == 1  # only the t(1) purchase is before t(5)


def test_val_and_test_cases_match_split_targets(users_adapter, reviews_adapter):
    events_by_user, splits = _build_scenario()
    _, _, val_cases, test_cases = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    val_by_user = {c.user_id: c for c in val_cases}
    assert val_by_user[1].target_ids == frozenset({3})
    assert val_by_user[2].target_ids == frozenset({6})
    assert len(test_cases) == 1
    assert test_cases[0].user_id == 1
    assert test_cases[0].target_ids == frozenset({4})


def test_val_case_user_features_reflect_history_before_val_cutoff_only(users_adapter, reviews_adapter):
    events_by_user, splits = _build_scenario()
    _, _, val_cases, _ = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    user1_val = next(c for c in val_cases if c.user_id == 1)
    # Before val_cutoff=t(10): purchases at t(1) and t(5) only (t(10) itself and t(20) excluded).
    assert user1_val.user_features.purchase_count == 2


def test_val_loss_examples_mirror_val_targets(users_adapter, reviews_adapter):
    events_by_user, splits = _build_scenario()
    _, val_loss_examples, val_cases, _ = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    assert len(val_loss_examples) == sum(len(c.target_ids) for c in val_cases)
    pairs = {(e.user_id, e.product_id) for e in val_loss_examples}
    assert pairs == {(1, 3), (2, 6)}


def test_insufficient_depth_user_contributes_its_single_purchase_to_train(users_adapter, reviews_adapter):
    events_by_user, splits = _build_scenario()
    train_examples, _, val_cases, test_cases = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    assert any(e.user_id == 3 and e.product_id == 7 for e in train_examples)
    assert not any(c.user_id == 3 for c in val_cases + test_cases)  # never evaluable


def test_engagement_no_purchase_user_contributes_nothing(users_adapter, reviews_adapter):
    events_by_user, splits = _build_scenario()
    train_examples, val_loss_examples, val_cases, test_cases = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    assert not any(e.user_id == 4 for e in train_examples + val_loss_examples)
    assert not any(c.user_id == 4 for c in val_cases + test_cases)


# --- leakage-safe negative sampling (section 8/16) -------------------------

def test_all_purchased_product_ids_by_user_includes_full_history():
    events_by_user, _ = _build_scenario()
    result = all_purchased_product_ids_by_user(events_by_user)
    assert result[1] == frozenset({1, 2, 3, 4})  # ALL of user 1's purchases, train AND held-out
    assert result[2] == frozenset({5, 6})
    assert result[3] == frozenset({7})
    assert result.get(4, frozenset()) == frozenset()  # no purchases


class _FakeVectorIndex:
    """Returns every product id 1..29 as a candidate, in a fixed order,
    with descending scores - deterministic stand-in so negative-sampling
    exclusion can be tested without a real Two-Tower/FAISS index.
    """

    def search(self, query_embeddings, k):
        n = 1 if query_embeddings.ndim == 1 else query_embeddings.shape[0]
        from recommendation.retrieval.index.base import SearchResult

        ids = list(range(1, 30))[:k]
        scores = [1.0 - 0.01 * i for i in range(len(ids))]
        return [SearchResult(item_ids=list(ids), scores=list(scores)) for _ in range(n)]


class _FakeEncoder:
    max_price = 10.0

    def encode_user_batch(self, users):
        return {"numeric": np.zeros((len(users), 1), dtype=np.float32)}


class _FakeUserTower:
    def predict(self, batch, verbose=0):
        n = batch["numeric"].shape[0]
        return np.zeros((n, 4), dtype=np.float32)


def test_future_purchase_is_never_sampled_as_a_negative(users_adapter, reviews_adapter):
    """The essential STEP 7 negative-sampling correctness test: user 1's
    LATER purchases (products 3, 4 - the val/test targets, purchased in
    the future relative to their earlier training examples) must NEVER
    appear as a negative for user 1's EARLIER training examples.
    """
    from recommendation.features.product_features import build_product_features

    events_by_user, splits = _build_scenario()
    train_examples, _, val_cases, _ = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    all_purchased = all_purchased_product_ids_by_user(events_by_user)
    product_features = build_product_features(list(_product_lookup().values()), [], [], [])

    from recommendation.utils.config import RankingConfig

    train_out, val_out = build_temporal_ranking_dataset(
        train_examples, val_cases, all_purchased, product_features, {},
        _FakeEncoder(), _FakeUserTower(), _FakeVectorIndex(), pool_size=10,
        ranking_config=RankingConfig(negatives_per_positive=20, random_seed=1),
    )

    user1_negatives = {e.product_id for e in train_out if e.user_id == 1 and e.label == 0}
    assert 3 not in user1_negatives  # user 1's val target - a FUTURE purchase relative to the earlier train examples
    assert 4 not in user1_negatives  # user 1's test target - likewise


def test_ranking_dataset_positives_are_labeled_one_and_negatives_zero(users_adapter, reviews_adapter):
    from recommendation.features.product_features import build_product_features
    from recommendation.utils.config import RankingConfig

    events_by_user, splits = _build_scenario()
    train_examples, _, val_cases, _ = build_temporal_two_tower_examples(
        events_by_user, splits, users_adapter, reviews_adapter, _product_lookup(), {}, FeatureConfig(), None
    )
    all_purchased = all_purchased_product_ids_by_user(events_by_user)
    product_features = build_product_features(list(_product_lookup().values()), [], [], [])

    train_out, val_out = build_temporal_ranking_dataset(
        train_examples, val_cases, all_purchased, product_features, {},
        _FakeEncoder(), _FakeUserTower(), _FakeVectorIndex(), pool_size=10,
        ranking_config=RankingConfig(negatives_per_positive=3, random_seed=1),
    )
    positives = [e for e in train_out if e.label == 1]
    assert {(e.user_id, e.product_id) for e in positives} == {(e.user_id, e.product_id) for e in train_examples}
    assert all(e.label == 0 for e in train_out if (e.user_id, e.product_id) not in {(x.user_id, x.product_id) for x in train_examples})


# --- evaluate_temporal_retrieval: multi-target, empty-case handling -------

def test_evaluate_temporal_retrieval_empty_cases_returns_zero_report():
    report = evaluate_temporal_retrieval([], "val", None, np.zeros((0, 4)), [], _FakeEncoder(), [5, 10])
    assert report.num_cases == 0
    assert report.recall_at_k == {}
