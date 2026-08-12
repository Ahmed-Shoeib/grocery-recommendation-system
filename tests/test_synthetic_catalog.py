import numpy as np

from recommendation.data.synthetic.catalog import (
    CATEGORY_ID_BY_NAME,
    TAG_VOCABULARY,
    build_catalog,
    build_categories,
    build_tags,
)


def test_catalog_has_approximately_fifty_products():
    products, _ = build_catalog(np.random.default_rng(0))
    assert len(products) == 50


def test_categories_include_a_parent_child_pair():
    categories = build_categories()
    beverages = next(c for c in categories if c.name == "Beverages")
    coffee = next(c for c in categories if c.name == "Coffee & Tea")
    assert beverages.parent_id is None
    assert coffee.parent_id == beverages.id


def test_every_product_category_id_is_valid():
    categories = build_categories()
    category_ids = {c.id for c in categories}
    products, _ = build_catalog(np.random.default_rng(1))
    assert all(p.category_id in category_ids for p in products)


def test_every_product_has_at_least_one_tag():
    products, product_tags = build_catalog(np.random.default_rng(2))
    tags_by_product = {}
    for pt in product_tags:
        tags_by_product.setdefault(pt.product_id, []).append(pt.tag_id)
    assert all(len(tags_by_product.get(p.id, [])) > 0 for p in products)


def test_all_product_tags_reference_known_tags():
    tags = build_tags()
    tag_ids = {t.id for t in tags}
    _, product_tags = build_catalog(np.random.default_rng(3))
    assert all(pt.tag_id in tag_ids for pt in product_tags)
    assert {t.name for t in tags} == set(TAG_VOCABULARY)


def test_build_catalog_is_reproducible_given_same_seed():
    products_a, _ = build_catalog(np.random.default_rng(42))
    products_b, _ = build_catalog(np.random.default_rng(42))
    assert [p.model_dump() for p in products_a] == [p.model_dump() for p in products_b]


def test_build_catalog_stock_and_active_vary_with_seed():
    products_a, _ = build_catalog(np.random.default_rng(1))
    products_b, _ = build_catalog(np.random.default_rng(2))
    stock_a = [p.stock_quantity for p in products_a]
    stock_b = [p.stock_quantity for p in products_b]
    assert stock_a != stock_b


def test_category_id_by_name_matches_built_categories():
    categories = build_categories()
    assert CATEGORY_ID_BY_NAME == {c.name: c.id for c in categories}
