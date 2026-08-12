import pytest
from pydantic import ValidationError

from recommendation.data.schemas import (
    CartAffinityRecord,
    Category,
    ChatbotContextRecord,
    EngagementProfile,
    Product,
    PurchaseRecord,
    ReviewRecord,
    SearchRecord,
    UserProfile,
)


def test_category_parent_is_optional():
    root = Category(id=1, name="Beverages")
    child = Category(id=2, name="Coffee", parent_id=1)
    assert root.parent_id is None
    assert child.parent_id == 1


def test_product_requires_core_fields_but_tags_default_empty():
    product = Product(id=1, category_id=1, slug="oat-milk", name="Oat Milk", price=3.5)
    assert product.tags == []
    assert product.is_active is True
    assert product.stock_quantity == 0

    with pytest.raises(ValidationError):
        Product(id=1, category_id=1, slug="oat-milk", name="Oat Milk")  # missing price


def test_user_profile_preferred_category_and_age_group_are_optional():
    user = UserProfile(user_id=42)
    assert user.preferred_category is None
    assert user.age_group is None

    user_with_profile = UserProfile(user_id=42, preferred_category="Healthy Snacks", age_group="25-34")
    assert user_with_profile.preferred_category == "Healthy Snacks"


def test_chatbot_context_record_matches_prompt_example_shape():
    record = ChatbotContextRecord(
        user_id=17,
        mentioned_product_ids=[4, 18],
        preferred_category="Healthy Snacks",
        summary="User asked for high-protein low-sugar breakfast products",
    )
    assert record.mentioned_product_ids == [4, 18]
    assert record.keywords == []
    assert record.source == "synthetic"


def test_engagement_profile_defaults_to_empty_history():
    profile = EngagementProfile(user_id=1, profile=UserProfile(user_id=1))
    assert profile.purchases == []
    assert profile.cart_items == []
    assert profile.searches == []
    assert profile.chatbot_context is None
    assert profile.reviews == []


def test_engagement_profile_aggregates_all_four_v1_signals():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1, preferred_category="Dairy", age_group="35-44"),
        purchases=[PurchaseRecord(user_id=1, product_id=10, order_id=100, quantity=2, unit_price=4.0)],
        cart_items=[CartAffinityRecord(user_id=1, product_id=11, quantity=1)],
        searches=[SearchRecord(user_id=1, search_term="greek yogurt")],
        chatbot_context=ChatbotContextRecord(user_id=1, preferred_category="Dairy"),
        reviews=[ReviewRecord(user_id=1, product_id=10, rating=4.5)],
    )
    assert len(profile.purchases) == 1
    assert len(profile.cart_items) == 1
    assert len(profile.searches) == 1
    assert profile.chatbot_context is not None
    assert len(profile.reviews) == 1
