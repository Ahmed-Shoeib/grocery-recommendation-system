from recommendation.features.product_features import ProductFeatures
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.utils.config import EligibilityConfig


def _pf(pid: int, is_active: bool = True, stock_quantity: int = 10) -> ProductFeatures:
    return ProductFeatures(
        product_id=pid, category_id=1, category_name="Dairy", parent_category_name=None, brand=None, tags=[],
        price=1.0, effective_price=1.0, discount_percentage=0.0, is_active=is_active, stock_quantity=stock_quantity,
        purchase_count=0, distinct_purchasers=0, cart_add_count=0, review_count=0, average_rating=None,
    )


def test_active_in_stock_product_is_eligible():
    rules = build_eligibility_rules(EligibilityConfig(require_active=True, require_in_stock=True))
    result = apply_eligibility([1], {1: _pf(1)}, rules)
    assert result.eligible_ids == [1]
    assert result.excluded_ids == []


def test_inactive_product_excluded_with_reason():
    rules = build_eligibility_rules(EligibilityConfig(require_active=True, require_in_stock=True))
    result = apply_eligibility([1], {1: _pf(1, is_active=False)}, rules)
    assert result.eligible_ids == []
    assert result.excluded_ids == [1]
    assert result.excluded_reasons[1] == ["is_active"]


def test_out_of_stock_product_excluded_with_reason():
    rules = build_eligibility_rules(EligibilityConfig(require_active=True, require_in_stock=True))
    result = apply_eligibility([1], {1: _pf(1, stock_quantity=0)}, rules)
    assert result.excluded_ids == [1]
    assert result.excluded_reasons[1] == ["in_stock"]


def test_negative_stock_also_excluded():
    rules = build_eligibility_rules(EligibilityConfig(require_active=True, require_in_stock=True))
    result = apply_eligibility([1], {1: _pf(1, stock_quantity=-1)}, rules)
    assert result.excluded_ids == [1]


def test_inactive_and_out_of_stock_reports_both_reasons():
    rules = build_eligibility_rules(EligibilityConfig(require_active=True, require_in_stock=True))
    result = apply_eligibility([1], {1: _pf(1, is_active=False, stock_quantity=0)}, rules)
    assert set(result.excluded_reasons[1]) == {"is_active", "in_stock"}


def test_require_in_stock_false_allows_zero_stock():
    rules = build_eligibility_rules(EligibilityConfig(require_active=True, require_in_stock=False))
    result = apply_eligibility([1], {1: _pf(1, stock_quantity=0)}, rules)
    assert result.eligible_ids == [1]


def test_preserves_candidate_order_within_eligible_and_excluded():
    rules = build_eligibility_rules(EligibilityConfig(require_active=True, require_in_stock=True))
    features = {1: _pf(1), 2: _pf(2, is_active=False), 3: _pf(3)}
    result = apply_eligibility([1, 2, 3], features, rules)
    assert result.eligible_ids == [1, 3]
    assert result.excluded_ids == [2]


def test_no_rules_configured_admits_everything():
    rules = build_eligibility_rules(EligibilityConfig(require_active=False, require_in_stock=False))
    result = apply_eligibility([1], {1: _pf(1, is_active=False, stock_quantity=-5)}, rules)
    assert result.eligible_ids == [1]
