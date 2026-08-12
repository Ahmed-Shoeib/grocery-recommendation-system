from recommendation.data.schemas.engagement import EngagementProfile, PurchaseRecord
from recommendation.data.schemas.user import UserProfile
from recommendation.retrieval.two_tower.splitting import build_user_splits


def _profile(user_id: int, product_ids: list[int]) -> EngagementProfile:
    return EngagementProfile(
        user_id=user_id,
        profile=UserProfile(user_id=user_id),
        purchases=[
            PurchaseRecord(user_id=user_id, product_id=pid, order_id=i, quantity=1, unit_price=1.0)
            for i, pid in enumerate(product_ids)
        ],
    )


def test_users_below_threshold_are_all_train_no_holdout():
    profiles = {1: _profile(1, [10, 11])}  # 2 distinct products, threshold 3
    splits = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=0)
    split = splits[1]
    assert sorted(split.train_product_ids) == [10, 11]
    assert split.val_product_id is None
    assert split.test_product_id is None
    assert split.is_evaluable is False


def test_user_with_zero_purchases_has_empty_split():
    profiles = {1: _profile(1, [])}
    splits = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=0)
    split = splits[1]
    assert split.train_product_ids == []
    assert split.is_evaluable is False


def test_users_at_or_above_threshold_get_val_and_test_holdout():
    profiles = {1: _profile(1, [10, 11, 12, 13, 14])}
    splits = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=0)
    split = splits[1]
    assert split.is_evaluable is True
    assert split.val_product_id is not None
    assert split.test_product_id is not None
    assert split.val_product_id != split.test_product_id


def test_train_val_test_are_disjoint_and_cover_all_distinct_products():
    profiles = {1: _profile(1, [10, 11, 12, 13, 14, 15])}
    splits = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=1)
    split = splits[1]
    all_ids = set(split.train_product_ids) | split.held_out_ids
    assert all_ids == {10, 11, 12, 13, 14, 15}
    assert set(split.train_product_ids) & split.held_out_ids == set()
    assert len(split.held_out_ids) == 2


def test_split_at_exact_threshold_leaves_at_least_one_train_product():
    profiles = {1: _profile(1, [10, 11, 12])}  # exactly threshold=3
    splits = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=0)
    split = splits[1]
    assert split.is_evaluable is True
    assert len(split.train_product_ids) == 1


def test_repeat_purchases_of_same_product_are_treated_as_one_distinct_product():
    profiles = {1: _profile(1, [10, 10, 10, 11, 12])}  # product 10 bought 3x
    splits = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=0)
    split = splits[1]
    all_ids = set(split.train_product_ids) | split.held_out_ids
    assert all_ids == {10, 11, 12}


def test_split_is_reproducible_with_same_seed():
    profiles = {1: _profile(1, [10, 11, 12, 13, 14])}
    splits_a = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=7)
    splits_b = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=7)
    assert splits_a[1].val_product_id == splits_b[1].val_product_id
    assert splits_a[1].test_product_id == splits_b[1].test_product_id
    assert splits_a[1].train_product_ids == splits_b[1].train_product_ids


def test_split_differs_with_different_seed():
    profiles = {1: _profile(1, list(range(10, 20)))}
    splits_a = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=1)
    splits_b = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=2)
    assert (splits_a[1].val_product_id, splits_a[1].test_product_id) != (
        splits_b[1].val_product_id, splits_b[1].test_product_id
    )


def test_split_operates_independently_per_user():
    profiles = {1: _profile(1, [10, 11]), 2: _profile(2, [20, 21, 22, 23])}
    splits = build_user_splits(profiles, min_distinct_products_for_holdout=3, random_seed=0)
    assert splits[1].is_evaluable is False
    assert splits[2].is_evaluable is True
