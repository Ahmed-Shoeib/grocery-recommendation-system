from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.utils.config import load_config


def _small_config(**overrides):
    config = load_config()
    data = config.model_dump()
    data["synthetic_data"]["num_users"] = overrides.pop("num_users", 60)
    data["synthetic_data"].update(overrides)
    return config.__class__(**data)


def test_generated_dataset_passes_referential_integrity_validation():
    dataset = generate_synthetic_dataset(_small_config())
    issues = validate_dataset(dataset)
    assert issues == []


def test_generated_dataset_is_reproducible_given_same_seed():
    config = _small_config()
    dataset_a = generate_synthetic_dataset(config)
    dataset_b = generate_synthetic_dataset(config)
    assert [u.model_dump() for u in dataset_a.users] == [u.model_dump() for u in dataset_b.users]
    assert [o.model_dump() for o in dataset_a.orders] == [o.model_dump() for o in dataset_b.orders]
    assert dataset_a.debug_user_personas == dataset_b.debug_user_personas


def test_generated_dataset_differs_with_different_seed():
    config_a = _small_config(random_seed=1)
    config_b = _small_config(random_seed=2)
    dataset_a = generate_synthetic_dataset(config_a)
    dataset_b = generate_synthetic_dataset(config_b)
    assert dataset_a.debug_user_personas != dataset_b.debug_user_personas


def test_user_count_matches_config():
    dataset = generate_synthetic_dataset(_small_config(num_users=60))
    assert len(dataset.users) == 60


def test_incomplete_profile_fraction_is_approximately_honored():
    dataset = generate_synthetic_dataset(_small_config(num_users=200, incomplete_profile_fraction=0.05))
    incomplete = sum(1 for u in dataset.users if u.preferred_category_id is None or u.age_group is None)
    assert incomplete == 10  # exact: round(200 * 0.05)
    for u in dataset.users:
        # Never partially incomplete - both fields are set together or both are None.
        assert (u.preferred_category_id is None) == (u.age_group is None)


def test_every_order_item_belongs_to_a_generated_order_and_product():
    dataset = generate_synthetic_dataset(_small_config())
    order_ids = {o.id for o in dataset.orders}
    product_ids = {p.id for p in dataset.products}
    assert all(item.order_id in order_ids for item in dataset.order_items)
    assert all(item.product_id in product_ids for item in dataset.order_items)


def test_purchase_counted_statuses_are_a_subset_of_generated_statuses():
    dataset = generate_synthetic_dataset(_small_config())
    statuses = {o.status for o in dataset.orders}
    assert statuses <= {"DELIVERED", "COMPLETED", "CANCELLED", "PENDING"}


def test_reviews_only_exist_for_purchased_products():
    dataset = generate_synthetic_dataset(_small_config(num_users=150))
    order_by_id = {o.id: o for o in dataset.orders}
    purchased_pairs = {(order_by_id[item.order_id].user_id, item.product_id) for item in dataset.order_items}
    review_pairs = {(r.user_id, r.product_id) for r in dataset.reviews}
    assert review_pairs <= purchased_pairs


def test_click_records_reference_real_products_and_are_synthetic_sourced():
    dataset = generate_synthetic_dataset(_small_config())
    product_ids = {p.id for p in dataset.products}
    assert dataset.click_records  # sanity: the small dataset still produces some clicks
    for record in dataset.click_records:
        assert record.product_id in product_ids
        assert record.source == "synthetic"


def test_search_records_have_either_a_matched_product_or_a_term_only():
    dataset = generate_synthetic_dataset(_small_config())
    product_ids = {p.id for p in dataset.products}
    for record in dataset.search_records:
        assert record.search_term
        assert record.matched_product_id is None or record.matched_product_id in product_ids
        assert record.source == "synthetic"


def test_chatbot_records_are_at_most_one_per_user_and_synthetic_sourced():
    dataset = generate_synthetic_dataset(_small_config())
    user_ids_seen = [r.user_id for r in dataset.chatbot_records]
    assert len(user_ids_seen) == len(set(user_ids_seen))
    assert all(r.source == "synthetic" for r in dataset.chatbot_records)
    assert all(r.summary for r in dataset.chatbot_records)


def test_no_view_click_or_impression_fields_are_present_anywhere():
    dataset = generate_synthetic_dataset(_small_config())
    forbidden = {"view", "click", "impression", "session"}
    for table in (dataset.users, dataset.products, dataset.orders, dataset.order_items, dataset.reviews):
        for row in table:
            assert not (forbidden & set(row.model_dump().keys()))
