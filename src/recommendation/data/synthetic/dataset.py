"""Top-level synthetic dataset orchestrator.

`generate_synthetic_dataset()` is the single entrypoint that produces a
complete, internally-consistent, reproducible V1 dataset: the ERD-shaped
raw tables (categories, tags, products, product-tags, users, orders,
order items, carts, cart items, reviews) plus the two synthetic-only
canonical signals (search, chatbot context).

Reproducibility: a single `numpy.random.SeedSequence` derived from
`config.synthetic_data.random_seed` is spawned into independent child
generators per generation stage, so re-running with the same seed produces
byte-identical output, and changing one stage's logic doesn't perturb the
random draws of another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from recommendation.data.schemas.engagement import ChatbotContextRecord, SearchRecord
from recommendation.data.synthetic.catalog import build_catalog, build_categories, build_tags
from recommendation.data.synthetic.chatbot import generate_chatbot_records
from recommendation.data.synthetic.interactions import generate_carts, generate_orders, generate_reviews
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
from recommendation.data.synthetic.search import generate_search_records
from recommendation.data.synthetic.users import generate_users
from recommendation.utils.config import AppConfig, SyntheticDataConfig, get_config


@dataclass
class SyntheticDataset:
    categories: list[RawCategory]
    tags: list[RawTag]
    products: list[RawProduct]
    product_tags: list[RawProductTag]
    users: list[RawUser]
    orders: list[RawOrder]
    order_items: list[RawOrderItem]
    carts: list[RawCart]
    cart_items: list[RawCartItem]
    reviews: list[RawReview]
    search_records: list[SearchRecord]
    chatbot_records: list[ChatbotContextRecord]
    # Generation-only debug metadata (latent persona per user). Not a
    # backend/canonical concept - used for stats reporting and tests only.
    debug_user_personas: dict[int, str] = field(default_factory=dict)


def generate_synthetic_dataset(config: AppConfig | None = None) -> SyntheticDataset:
    config = config or get_config()
    sd: SyntheticDataConfig = config.synthetic_data

    seed_sequence = np.random.SeedSequence(sd.random_seed)
    catalog_rng, user_rng, order_rng, cart_rng, review_rng, search_rng, chatbot_rng = (
        np.random.default_rng(s) for s in seed_sequence.spawn(7)
    )

    categories = build_categories()
    tags = build_tags()
    products, product_tags = build_catalog(catalog_rng)

    users, latent_profiles = generate_users(sd.num_users, user_rng, sd)

    orders, order_items = generate_orders(users, latent_profiles, products, product_tags, tags, order_rng, sd)
    carts, cart_items = generate_carts(users, latent_profiles, products, product_tags, tags, cart_rng, sd)
    reviews = generate_reviews(
        users, latent_profiles, orders, order_items, products, product_tags, tags, review_rng, sd
    )
    search_records = generate_search_records(users, latent_profiles, products, product_tags, tags, search_rng, sd)
    chatbot_records = generate_chatbot_records(users, latent_profiles, products, product_tags, tags, chatbot_rng, sd)

    return SyntheticDataset(
        categories=categories,
        tags=tags,
        products=products,
        product_tags=product_tags,
        users=users,
        orders=orders,
        order_items=order_items,
        carts=carts,
        cart_items=cart_items,
        reviews=reviews,
        search_records=search_records,
        chatbot_records=chatbot_records,
        debug_user_personas={uid: p.persona_key for uid, p in latent_profiles.items()},
    )
