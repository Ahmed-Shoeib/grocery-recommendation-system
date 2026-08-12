"""Synthetic Order/OrderItem, Cart/CartItem, and Review generation.

All three are driven by `affinity.product_affinity_scores` using each
user's `UserLatentProfile`, so purchases, cart adds, and review ratings are
correlated with the same latent persona - not independently random. No
recency features are derived anywhere downstream from the timestamps
generated here (see docs/data-mapping.md section 1/7); they exist purely
for ERD fidelity.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from recommendation.data.synthetic.affinity import product_affinity_scores, sample_product_ids
from recommendation.data.synthetic.catalog import CATEGORY_NAME_BY_ID
from recommendation.data.synthetic.personas import PERSONA_BY_KEY
from recommendation.data.synthetic.raw_schemas import (
    RawCart,
    RawCartItem,
    RawOrder,
    RawOrderItem,
    RawProduct,
    RawProductTag,
    RawReview,
    RawTag,
    RawUser,
)
from recommendation.data.synthetic.users import UserLatentProfile
from recommendation.utils.config import SyntheticDataConfig

REFERENCE_NOW = datetime(2026, 8, 12)


def _tags_by_product_id(product_tags: list[RawProductTag], tags: list[RawTag]) -> dict[int, list[str]]:
    tag_name_by_id = {t.id: t.name for t in tags}
    out: dict[int, list[str]] = {}
    for pt in product_tags:
        out.setdefault(pt.product_id, []).append(tag_name_by_id[pt.tag_id])
    return out


def _weighted_status(rng: np.random.Generator, weights: dict[str, float]) -> str:
    statuses = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs = probs / probs.sum()
    return statuses[int(rng.choice(len(statuses), p=probs))]


def generate_orders(
    users: list[RawUser],
    latent_profiles: dict[int, UserLatentProfile],
    products: list[RawProduct],
    product_tags: list[RawProductTag],
    tags: list[RawTag],
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> tuple[list[RawOrder], list[RawOrderItem]]:
    tags_by_product = _tags_by_product_id(product_tags, tags)
    product_ids = [p.id for p in products]
    product_by_id = {p.id: p for p in products}

    orders: list[RawOrder] = []
    order_items: list[RawOrderItem] = []
    order_id = 1
    order_item_id = 1

    for user in users:
        latent = latent_profiles[user.id]
        persona = PERSONA_BY_KEY[latent.persona_key]
        scores = product_affinity_scores(
            products, CATEGORY_NAME_BY_ID, tags_by_product, persona,
            latent.preferred_brand, config.brand_affinity_bonus, rng, config.affinity_noise_scale,
        )

        num_orders = min(int(rng.poisson(config.order_count_lambda * latent.activity_level)), config.order_max_count)
        for _ in range(num_orders):
            status = _weighted_status(rng, config.order_status_weights)
            days_ago = int(rng.integers(1, config.order_date_lookback_days + 1))
            creation_date = REFERENCE_NOW - timedelta(days=days_ago)
            delivery_date = (
                creation_date + timedelta(days=int(rng.integers(1, 8)))
                if status in ("DELIVERED", "COMPLETED")
                else None
            )

            num_items = int(rng.integers(config.order_items_min, config.order_items_max + 1))
            sampled_ids = sample_product_ids(rng, product_ids, scores, k=num_items, replace=False)
            if not sampled_ids:
                continue

            orders.append(RawOrder(id=order_id, user_id=user.id, status=status,
                                    creation_date=creation_date, delivery_date=delivery_date))
            for product_id in sampled_ids:
                quantity = max(1, int(round(rng.integers(1, config.order_quantity_max + 1) * persona.quantity_bias)))
                order_items.append(
                    RawOrderItem(
                        id=order_item_id, order_id=order_id, product_id=product_id,
                        quantity=quantity, unit_price=product_by_id[product_id].price,
                    )
                )
                order_item_id += 1
            order_id += 1

    return orders, order_items


def generate_carts(
    users: list[RawUser],
    latent_profiles: dict[int, UserLatentProfile],
    products: list[RawProduct],
    product_tags: list[RawProductTag],
    tags: list[RawTag],
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> tuple[list[RawCart], list[RawCartItem]]:
    tags_by_product = _tags_by_product_id(product_tags, tags)
    product_ids = [p.id for p in products]

    carts: list[RawCart] = []
    cart_items: list[RawCartItem] = []
    cart_item_id = 1

    for cart_id, user in enumerate(users, start=1):
        latent = latent_profiles[user.id]
        persona = PERSONA_BY_KEY[latent.persona_key]
        carts.append(RawCart(id=cart_id, user_id=user.id))

        if rng.random() >= config.cart_nonempty_prob:
            continue

        scores = product_affinity_scores(
            products, CATEGORY_NAME_BY_ID, tags_by_product, persona,
            latent.preferred_brand, config.brand_affinity_bonus, rng, config.affinity_noise_scale,
        )
        num_items = int(rng.integers(1, config.cart_items_max + 1))
        sampled_ids = sample_product_ids(rng, product_ids, scores, k=num_items, replace=False)
        for product_id in sampled_ids:
            quantity = max(1, int(round(rng.integers(1, config.cart_quantity_max + 1) * persona.quantity_bias)))
            cart_items.append(RawCartItem(id=cart_item_id, cart_id=cart_id, product_id=product_id, quantity=quantity))
            cart_item_id += 1

    return carts, cart_items


def generate_reviews(
    users: list[RawUser],
    latent_profiles: dict[int, UserLatentProfile],
    orders: list[RawOrder],
    order_items: list[RawOrderItem],
    products: list[RawProduct],
    product_tags: list[RawProductTag],
    tags: list[RawTag],
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> list[RawReview]:
    """One review per (user, product) purchase pair with probability
    `review_prob`. Rating is biased by the same affinity score that drove
    the purchase (higher affinity -> higher average rating), clipped to
    [1, 5] and rounded to the nearest half star.
    """
    tags_by_product = _tags_by_product_id(product_tags, tags)
    product_by_id = {p.id: p for p in products}
    order_by_id = {o.id: o for o in orders}

    purchased_pairs: dict[int, set[int]] = {}
    for item in order_items:
        order = order_by_id[item.order_id]
        purchased_pairs.setdefault(order.user_id, set()).add(item.product_id)

    reviews: list[RawReview] = []
    review_id = 1
    templates = {
        "high": ["Great product, will buy again.", "Exactly what I was looking for.", "Really good quality."],
        "mid": ["Decent, does the job.", "Good but nothing special.", "Solid everyday choice."],
        "low": ["Not what I expected.", "Wouldn't repurchase.", "Below average quality."],
    }

    for user in users:
        latent = latent_profiles.get(user.id)
        product_ids_bought = purchased_pairs.get(user.id)
        if not latent or not product_ids_bought:
            continue
        persona = PERSONA_BY_KEY[latent.persona_key]

        for product_id in product_ids_bought:
            if rng.random() >= config.review_prob:
                continue
            product = product_by_id[product_id]
            category_name = CATEGORY_NAME_BY_ID.get(product.category_id, "")
            product_tags_list = tags_by_product.get(product_id, [])

            affinity = 1.0 + persona.category_weights.get(category_name, 0.0)
            affinity += sum(persona.tag_weights.get(t, 0.0) for t in product_tags_list)
            normalized = 1.0 / (1.0 + np.exp(-affinity / 2.5))  # squash to (0, 1)
            rating = 3.0 + normalized * 2.0 + float(rng.normal(0, 0.4))
            rating = float(np.clip(round(rating * 2) / 2, 1.0, 5.0))

            bucket = "high" if rating >= 4.5 else "mid" if rating >= 3.0 else "low"
            comment = templates[bucket][int(rng.integers(0, len(templates[bucket])))]

            reviews.append(
                RawReview(
                    id=review_id, user_id=user.id, product_id=product_id, rating=rating,
                    comment=comment, creation_date=REFERENCE_NOW - timedelta(days=int(rng.integers(1, 170))),
                )
            )
            review_id += 1

    return reviews
