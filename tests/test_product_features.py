from recommendation.data.schemas.engagement import CartAffinityRecord, PurchaseRecord, ReviewRecord
from recommendation.data.schemas.product import Product
from recommendation.features.product_features import (
    build_product_features,
    compute_cart_add_counts,
    compute_product_popularity,
    compute_review_stats,
)


def _products():
    return [
        Product(id=1, category_id=1, slug="a", name="A", price=5.0, sale_price=4.0, discount_percentage=20.0,
                is_active=True, stock_quantity=10, category_name="Dairy & Eggs", brand="X", tags=["healthy"]),
        Product(id=2, category_id=2, slug="b", name="B", price=3.0, is_active=False, stock_quantity=0,
                category_name="Snacks", brand="Y"),
    ]


def test_compute_product_popularity_counts_purchases_and_distinct_users():
    purchases = [
        PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=2, unit_price=5.0),
        PurchaseRecord(user_id=2, product_id=1, order_id=2, quantity=1, unit_price=5.0),
        PurchaseRecord(user_id=1, product_id=1, order_id=3, quantity=1, unit_price=5.0),
    ]
    popularity = compute_product_popularity(purchases)
    assert popularity[1] == (3, 2)  # 3 purchase records, 2 distinct users


def test_compute_cart_add_counts():
    cart_items = [
        CartAffinityRecord(user_id=1, product_id=1, quantity=2),
        CartAffinityRecord(user_id=2, product_id=1, quantity=1),
        CartAffinityRecord(user_id=1, product_id=2, quantity=1),
    ]
    counts = compute_cart_add_counts(cart_items)
    assert counts == {1: 2, 2: 1}


def test_compute_review_stats_averages_rating():
    reviews = [
        ReviewRecord(user_id=1, product_id=1, rating=5.0),
        ReviewRecord(user_id=2, product_id=1, rating=3.0),
    ]
    stats = compute_review_stats(reviews)
    assert stats[1] == (4.0, 2)


def test_build_product_features_uses_sale_price_as_effective_price():
    features = build_product_features(_products(), [], [], [])
    assert features[1].effective_price == 4.0
    assert features[1].discount_percentage == 20.0
    assert features[2].effective_price == 3.0  # no sale price -> falls back to price
    assert features[2].discount_percentage == 0.0


def test_build_product_features_zero_defaults_for_products_with_no_interactions():
    features = build_product_features(_products(), [], [], [])
    assert features[1].purchase_count == 0
    assert features[1].distinct_purchasers == 0
    assert features[1].cart_add_count == 0
    assert features[1].review_count == 0
    assert features[1].average_rating is None


def test_build_product_features_carries_is_active_and_stock():
    features = build_product_features(_products(), [], [], [])
    assert features[1].is_active is True
    assert features[1].stock_quantity == 10
    assert features[2].is_active is False
    assert features[2].stock_quantity == 0


def test_build_product_features_aggregates_across_all_signal_types():
    purchases = [PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=1, unit_price=5.0)]
    cart_items = [CartAffinityRecord(user_id=1, product_id=1, quantity=1)]
    reviews = [ReviewRecord(user_id=1, product_id=1, rating=4.5)]
    features = build_product_features(_products(), purchases, cart_items, reviews)
    assert features[1].purchase_count == 1
    assert features[1].cart_add_count == 1
    assert features[1].review_count == 1
    assert features[1].average_rating == 4.5
