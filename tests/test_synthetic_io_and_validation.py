from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.io import load_dataset, save_dataset
from recommendation.data.synthetic.raw_schemas import RawOrder, RawOrderItem
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.utils.config import get_config


def _small_dataset():
    config = get_config().model_copy(deep=True)
    config.synthetic_data.num_users = 30
    return generate_synthetic_dataset(config)


def test_save_and_load_dataset_round_trips(tmp_path):
    dataset = _small_dataset()
    save_dataset(dataset, tmp_path)
    reloaded = load_dataset(tmp_path)

    assert len(reloaded.products) == len(dataset.products)
    assert len(reloaded.users) == len(dataset.users)
    assert len(reloaded.orders) == len(dataset.orders)
    assert [p.model_dump() for p in reloaded.products] == [p.model_dump() for p in dataset.products]
    assert reloaded.debug_user_personas == dataset.debug_user_personas


def test_validate_dataset_detects_dangling_order_item_product_id():
    dataset = _small_dataset()
    dataset.order_items.append(
        RawOrderItem(id=999_999, order_id=dataset.orders[0].id, product_id=999_999, quantity=1, unit_price=1.0)
    )
    issues = validate_dataset(dataset)
    assert any("dangling product_id" in issue for issue in issues)


def test_validate_dataset_detects_dangling_order_user_id():
    dataset = _small_dataset()
    dataset.orders.append(RawOrder(id=999_999, user_id=999_999, status="DELIVERED", creation_date="2026-01-01T00:00:00"))
    issues = validate_dataset(dataset)
    assert any("dangling user_id" in issue for issue in issues)


def test_validate_dataset_clean_dataset_has_no_issues():
    dataset = _small_dataset()
    assert validate_dataset(dataset) == []
