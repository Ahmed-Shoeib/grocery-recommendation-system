"""Statistical tests that generated interactions are correlated with latent
personas, not uniformly random - the core Phase 2 requirement ("users who
prefer healthy products should statistically interact more with Greek
yogurt, oats, whole-grain, low-sugar, fruit, protein products").

Uses the full default-size dataset (300 users) for stable statistics.
"""

from collections import defaultdict

from recommendation.data.synthetic.catalog import CATEGORY_NAME_BY_ID
from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.personas import PERSONA_BY_KEY
from recommendation.utils.config import get_config

_DATASET = generate_synthetic_dataset(get_config())


def _tags_by_product():
    tag_name_by_id = {t.id: t.name for t in _DATASET.tags}
    tags_by_product: dict[int, set[str]] = defaultdict(set)
    for pt in _DATASET.product_tags:
        tags_by_product[pt.product_id].add(tag_name_by_id[pt.tag_id])
    return tags_by_product


def _tag_purchase_rate(persona_key: str, tag: str) -> float:
    order_by_id = {o.id: o for o in _DATASET.orders}
    tags_by_product = _tags_by_product()
    hits = total = 0
    for item in _DATASET.order_items:
        order = order_by_id[item.order_id]
        if _DATASET.debug_user_personas[order.user_id] != persona_key:
            continue
        total += 1
        if tag in tags_by_product.get(item.product_id, set()):
            hits += 1
    assert total > 0, f"no purchases found for persona {persona_key}"
    return hits / total


def test_health_conscious_users_buy_healthy_tagged_products_more_than_snack_heavy_users():
    health_rate = _tag_purchase_rate("health_conscious", "healthy")
    snack_rate = _tag_purchase_rate("snack_heavy", "healthy")
    assert health_rate > snack_rate + 0.15


def test_snack_heavy_users_buy_indulgent_tagged_products_more_than_health_conscious_users():
    snack_rate = _tag_purchase_rate("snack_heavy", "indulgent")
    health_rate = _tag_purchase_rate("health_conscious", "indulgent")
    assert snack_rate > health_rate + 0.10


def test_vegetarian_oriented_users_buy_meat_seafood_less_than_average():
    order_by_id = {o.id: o for o in _DATASET.orders}
    product_by_id = {p.id: p for p in _DATASET.products}

    def meat_rate(persona_key: str | None) -> float:
        hits = total = 0
        for item in _DATASET.order_items:
            order = order_by_id[item.order_id]
            if persona_key is not None and _DATASET.debug_user_personas[order.user_id] != persona_key:
                continue
            total += 1
            category_name = CATEGORY_NAME_BY_ID[product_by_id[item.product_id].category_id]
            if category_name == "Meat & Seafood":
                hits += 1
        return hits / total if total else 0.0

    vegetarian_rate = meat_rate("vegetarian_oriented")
    overall_rate = meat_rate(None)
    assert vegetarian_rate < overall_rate


def test_persona_purchases_skew_toward_their_own_dominant_categories():
    order_by_id = {o.id: o for o in _DATASET.orders}
    product_by_id = {p.id: p for p in _DATASET.products}

    category_hits: dict[str, int] = defaultdict(int)
    category_total: dict[str, int] = defaultdict(int)
    for item in _DATASET.order_items:
        order = order_by_id[item.order_id]
        persona_key = _DATASET.debug_user_personas[order.user_id]
        persona = PERSONA_BY_KEY[persona_key]
        category_name = CATEGORY_NAME_BY_ID[product_by_id[item.product_id].category_id]
        category_total[persona_key] += 1
        if category_name in persona.dominant_categories:
            category_hits[persona_key] += 1

    # 13 categories total; personas have 1-3 dominant categories, so a
    # non-correlated baseline would land around 2.5/13 ~ 0.19.
    baseline_rate = 2.5 / 13
    checked_any = False
    for persona_key, total in category_total.items():
        if total < 15:
            continue  # not enough purchases for a stable estimate
        checked_any = True
        own_rate = category_hits[persona_key] / total
        assert own_rate > baseline_rate * 1.5, (
            f"{persona_key}: dominant-category purchase rate {own_rate:.2f} not above expected baseline"
        )
    assert checked_any


def test_family_household_has_higher_average_purchase_quantity_than_health_conscious():
    order_by_id = {o.id: o for o in _DATASET.orders}

    def avg_quantity(persona_key: str) -> float:
        quantities = [
            item.quantity
            for item in _DATASET.order_items
            if _DATASET.debug_user_personas[order_by_id[item.order_id].user_id] == persona_key
        ]
        assert quantities
        return sum(quantities) / len(quantities)

    assert avg_quantity("family_household") > avg_quantity("health_conscious")


def test_interactions_are_not_deterministic_across_seeds():
    """Same persona, different seed, should not reproduce identical purchase
    sets - confirms noise is actually applied, not just category weighting.
    """
    config_a = get_config().model_copy(deep=True)
    config_b = get_config().model_copy(deep=True)
    config_a.synthetic_data.random_seed = 101
    config_b.synthetic_data.random_seed = 202

    dataset_a = generate_synthetic_dataset(config_a)
    dataset_b = generate_synthetic_dataset(config_b)

    user_1_products_a = {i.product_id for i in dataset_a.order_items if order_user(dataset_a, i) == 1}
    user_1_products_b = {i.product_id for i in dataset_b.order_items if order_user(dataset_b, i) == 1}
    assert user_1_products_a != user_1_products_b


def order_user(dataset, item) -> int:
    order_by_id = {o.id: o for o in dataset.orders}
    return order_by_id[item.order_id].user_id
