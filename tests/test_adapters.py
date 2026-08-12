from recommendation.data.adapters.cart_adapter import InMemoryCartAdapter
from recommendation.data.adapters.chatbot_adapter import SyntheticChatbotAdapter
from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.adapters.factory import build_synthetic_adapters
from recommendation.data.adapters.product_adapter import InMemoryProductCatalogAdapter
from recommendation.data.adapters.purchase_adapter import InMemoryPurchaseAdapter
from recommendation.data.adapters.review_adapter import InMemoryReviewAdapter
from recommendation.data.adapters.search_adapter import SyntheticSearchAdapter
from recommendation.data.adapters.user_adapter import InMemoryUserAdapter
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.raw_schemas import (
    RawCart,
    RawCartItem,
    RawCategory,
    RawOrder,
    RawOrderItem,
    RawProduct,
    RawProductTag,
    RawReview,
    RawTag,
    RawUser,
)
from recommendation.utils.config import get_config


def _small_dataset():
    config = get_config().model_copy(deep=True)
    config.synthetic_data.num_users = 40
    return generate_synthetic_dataset(config), config


# --- ProductCatalogAdapter -------------------------------------------------

def test_product_adapter_resolves_category_and_tags():
    categories = [RawCategory(id=1, name="Beverages", parent_id=None), RawCategory(id=2, name="Coffee & Tea", parent_id=1)]
    tags = [RawTag(id=1, name="premium")]
    products = [RawProduct(id=1, category_id=2, slug="coffee", name="Coffee", price=5.0)]
    product_tags = [RawProductTag(id=1, product_id=1, tag_id=1)]

    adapter = InMemoryProductCatalogAdapter(categories, tags, products, product_tags)
    product = adapter.get_product(1)

    assert product is not None
    assert product.category_name == "Coffee & Tea"
    assert product.parent_category_name == "Beverages"
    assert product.tags == ["premium"]


def test_product_adapter_handles_top_level_category_with_no_parent():
    categories = [RawCategory(id=1, name="Snacks", parent_id=None)]
    products = [RawProduct(id=1, category_id=1, slug="chips", name="Chips", price=2.0)]
    adapter = InMemoryProductCatalogAdapter(categories, [], products, [])
    product = adapter.get_product(1)
    assert product.category_name == "Snacks"
    assert product.parent_category_name is None


def test_product_adapter_unknown_id_returns_none():
    adapter = InMemoryProductCatalogAdapter([], [], [], [])
    assert adapter.get_product(999) is None


def test_product_adapter_list_products_matches_catalog_size():
    dataset, _ = _small_dataset()
    adapter = InMemoryProductCatalogAdapter(dataset.categories, dataset.tags, dataset.products, dataset.product_tags)
    assert len(adapter.list_products()) == len(dataset.products)


# --- UserAdapter ------------------------------------------------------------

def test_user_adapter_resolves_preferred_category_id_to_name():
    categories = [RawCategory(id=1, name="Dairy & Eggs", parent_id=None)]
    users = [RawUser(id=1, first_name="A", last_name="B", email="a@b.invalid", preferred_category_id=1, age_group="25-34")]
    adapter = InMemoryUserAdapter(users, categories)
    profile = adapter.get_user_profile(1)
    assert profile.preferred_category == "Dairy & Eggs"
    assert profile.age_group == "25-34"


def test_user_adapter_handles_missing_preferred_category_defensively():
    users = [RawUser(id=1, first_name="A", last_name="B", email="a@b.invalid", preferred_category_id=None, age_group=None)]
    adapter = InMemoryUserAdapter(users, [])
    profile = adapter.get_user_profile(1)
    assert profile.preferred_category is None
    assert profile.age_group is None


def test_user_adapter_dangling_preferred_category_id_resolves_to_none():
    users = [RawUser(id=1, first_name="A", last_name="B", email="a@b.invalid", preferred_category_id=999, age_group=None)]
    adapter = InMemoryUserAdapter(users, [])
    profile = adapter.get_user_profile(1)
    assert profile.preferred_category is None


def test_user_adapter_unknown_user_returns_none():
    adapter = InMemoryUserAdapter([], [])
    assert adapter.get_user_profile(1) is None


# --- PurchaseAdapter ---------------------------------------------------------

def test_purchase_adapter_only_counts_configured_statuses():
    orders = [
        RawOrder(id=1, user_id=1, status="DELIVERED", creation_date="2026-01-01T00:00:00"),
        RawOrder(id=2, user_id=1, status="CANCELLED", creation_date="2026-01-01T00:00:00"),
    ]
    order_items = [
        RawOrderItem(id=1, order_id=1, product_id=10, quantity=1, unit_price=2.0),
        RawOrderItem(id=2, order_id=2, product_id=11, quantity=1, unit_price=2.0),
    ]
    adapter = InMemoryPurchaseAdapter(orders, order_items)
    purchases = adapter.get_purchases(1)
    assert len(purchases) == 1
    assert purchases[0].product_id == 10


def test_purchase_adapter_counted_statuses_are_configurable():
    orders = [RawOrder(id=1, user_id=1, status="PENDING", creation_date="2026-01-01T00:00:00")]
    order_items = [RawOrderItem(id=1, order_id=1, product_id=10, quantity=1, unit_price=2.0)]
    adapter = InMemoryPurchaseAdapter(orders, order_items, counted_statuses={"PENDING"})
    assert len(adapter.get_purchases(1)) == 1


def test_purchase_adapter_unknown_user_returns_empty_list():
    adapter = InMemoryPurchaseAdapter([], [])
    assert adapter.get_purchases(1) == []


# --- CartAdapter --------------------------------------------------------------

def test_cart_adapter_joins_cart_to_user():
    carts = [RawCart(id=1, user_id=7)]
    cart_items = [RawCartItem(id=1, cart_id=1, product_id=5, quantity=2)]
    adapter = InMemoryCartAdapter(carts, cart_items)
    items = adapter.get_cart_items(7)
    assert len(items) == 1
    assert items[0].product_id == 5
    assert items[0].quantity == 2


def test_cart_adapter_empty_cart_returns_empty_list():
    adapter = InMemoryCartAdapter([RawCart(id=1, user_id=1)], [])
    assert adapter.get_cart_items(1) == []


# --- ReviewAdapter --------------------------------------------------------------

def test_review_adapter_groups_by_user():
    reviews = [RawReview(id=1, user_id=1, product_id=5, rating=4.5)]
    adapter = InMemoryReviewAdapter(reviews)
    assert len(adapter.get_reviews(1)) == 1
    assert adapter.get_reviews(2) == []


# --- Search / Chatbot adapters -------------------------------------------------

def test_synthetic_search_adapter_groups_by_user():
    from recommendation.data.schemas.engagement import SearchRecord

    records = [SearchRecord(user_id=1, search_term="oat milk")]
    adapter = SyntheticSearchAdapter(records)
    assert len(adapter.get_search_history(1)) == 1
    assert adapter.get_search_history(2) == []


def test_synthetic_chatbot_adapter_returns_none_when_absent():
    adapter = SyntheticChatbotAdapter([])
    assert adapter.get_chatbot_context(1) is None


# --- Factory + engagement profile builder --------------------------------------

def test_build_synthetic_adapters_produces_a_full_bundle():
    dataset, config = _small_dataset()
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)
    assert len(bundle.products.list_products()) == len(dataset.products)
    assert set(bundle.users.list_user_ids()) == {u.id for u in dataset.users}


def test_build_engagement_profile_combines_all_four_v1_signals_plus_reviews():
    dataset, config = _small_dataset()
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)

    # Pick a user known to have purchases (from the full generation run).
    user_id = next(o.user_id for o in dataset.orders)
    profile = build_engagement_profile(
        user_id, bundle.users, bundle.purchases, bundle.cart, bundle.search, bundle.chatbot, bundle.reviews
    )

    assert isinstance(profile, EngagementProfile)
    assert profile.user_id == user_id
    assert len(profile.purchases) > 0


def test_build_engagement_profile_defaults_for_unknown_user():
    dataset, config = _small_dataset()
    bundle = build_synthetic_adapters(dataset, config.synthetic_data)
    profile = build_engagement_profile(
        999_999, bundle.users, bundle.purchases, bundle.cart, bundle.search, bundle.chatbot, bundle.reviews
    )
    assert profile.user_id == 999_999
    assert profile.profile.user_id == 999_999
    assert profile.purchases == []
    assert profile.cart_items == []
    assert profile.searches == []
    assert profile.chatbot_context is None
    assert profile.reviews == []
