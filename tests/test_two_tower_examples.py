import numpy as np

from recommendation.data.schemas.engagement import EngagementProfile, PurchaseRecord
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.retrieval.two_tower.examples import build_eval_cases, build_training_examples
from recommendation.retrieval.two_tower.splitting import UserSplit
from recommendation.utils.config import FeatureConfig


def _product(pid: int, category: str) -> Product:
    return Product(id=pid, category_id=pid, slug=f"p{pid}", name=f"Product {pid}", price=1.0, category_name=category)


def _profile(user_id: int, product_ids: list[int]) -> EngagementProfile:
    return EngagementProfile(
        user_id=user_id,
        profile=UserProfile(user_id=user_id),
        purchases=[
            PurchaseRecord(user_id=user_id, product_id=pid, order_id=i, quantity=1, unit_price=1.0)
            for i, pid in enumerate(product_ids)
        ],
    )


def _setup(product_ids=(10, 11, 12, 13, 14)):
    product_lookup = {pid: _product(pid, f"Category{pid}") for pid in product_ids}
    embeddings = {pid: np.array([float(pid)], dtype=np.float32) for pid in product_ids}
    profile = _profile(1, list(product_ids))
    return product_lookup, embeddings, profile


def test_training_examples_one_per_train_product():
    product_lookup, embeddings, profile = _setup()
    split = UserSplit(user_id=1, train_product_ids=[10, 11, 12], val_product_id=13, test_product_id=14)
    examples = build_training_examples({1: profile}, {1: split}, product_lookup, embeddings, FeatureConfig(), {})
    assert sorted(e.product_id for e in examples) == [10, 11, 12]
    assert all(e.user_id == 1 for e in examples)


def test_training_example_excludes_its_own_target_and_held_out_products():
    product_lookup, embeddings, profile = _setup()
    split = UserSplit(user_id=1, train_product_ids=[10, 11, 12], val_product_id=13, test_product_id=14)
    examples = build_training_examples({1: profile}, {1: split}, product_lookup, embeddings, FeatureConfig(), {})

    example_for_10 = next(e for e in examples if e.product_id == 10)
    # Purchase count should reflect 5 total purchases minus {10 (self), 13 (val), 14 (test)} = 2 remaining (11, 12)
    assert example_for_10.user_features.purchase_count == 2


def test_users_with_no_train_products_produce_no_training_examples():
    product_lookup, embeddings, profile = _setup(product_ids=(10, 11))
    split = UserSplit(user_id=1, train_product_ids=[], val_product_id=None, test_product_id=None)
    examples = build_training_examples({1: profile}, {1: split}, product_lookup, embeddings, FeatureConfig(), {})
    assert examples == []


def test_eval_cases_only_include_evaluable_users():
    product_lookup, embeddings, profile = _setup()
    evaluable = UserSplit(user_id=1, train_product_ids=[10, 11, 12], val_product_id=13, test_product_id=14)
    not_evaluable = UserSplit(user_id=2, train_product_ids=[10, 11], val_product_id=None, test_product_id=None)
    splits = {1: evaluable, 2: not_evaluable}
    profiles = {1: profile, 2: _profile(2, [10, 11])}
    cases = build_eval_cases(profiles, splits, product_lookup, embeddings, FeatureConfig(), {})
    assert [c.user_id for c in cases] == [1]


def test_eval_case_excludes_both_val_and_test_but_keeps_train_visible():
    product_lookup, embeddings, profile = _setup()
    split = UserSplit(user_id=1, train_product_ids=[10, 11, 12], val_product_id=13, test_product_id=14)
    cases = build_eval_cases({1: profile}, {1: split}, product_lookup, embeddings, FeatureConfig(), {})
    case = cases[0]
    # 5 total purchases minus {13, 14} held out = 3 remaining (10, 11, 12) visible.
    assert case.user_features.purchase_count == 3
    assert case.val_product_id == 13
    assert case.test_product_id == 14
