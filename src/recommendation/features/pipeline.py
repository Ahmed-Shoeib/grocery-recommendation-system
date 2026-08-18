"""Top-level feature pipeline: AdapterBundle -> product/user features + embeddings.

Depends only on `recommendation.data.adapters.base.AdapterBundle` (an
interface type) and `recommendation.utils.config.AppConfig` - nothing here
imports `recommendation.data.synthetic`, so pointing a real-backend
`AdapterBundle` at this function requires no changes here (see
docs/data-mapping.md).

Only the Sentence Transformer product embeddings are persisted to disk
(`config.embedding.cache_path`) - that's the expensive-to-recompute
artifact. User features are cheap arithmetic over the cached embeddings
and adapter reads, so they're recomputed on demand rather than cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from recommendation.data.adapters.base import AdapterBundle
from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.schemas.engagement import EngagementProfile
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.embeddings.product_embeddings import ProductEmbeddingCache, get_or_compute_product_embeddings
from recommendation.features.price import build_price_catalog_context
from recommendation.features.product_features import ProductFeatures, build_product_features
from recommendation.features.user_features import UserFeatures, build_user_features, build_user_text_embeddings
from recommendation.utils.config import AppConfig, resolve_path


@dataclass
class FeaturePipelineResult:
    product_embeddings: ProductEmbeddingCache
    product_features: dict[int, ProductFeatures]
    engagement_profiles: dict[int, EngagementProfile]
    user_features: dict[int, UserFeatures]
    embeddings_recomputed: bool


def run_feature_pipeline(
    bundle: AdapterBundle,
    config: AppConfig,
    encoder: SentenceTransformerEncoder | None = None,
    reference_time: datetime | None = None,
) -> FeaturePipelineResult:
    """`reference_time` is forwarded to every `build_user_features` call as
    the recency reference point (see that function's docstring) - `None`
    (default) leaves recency inactive (neutral weights), matching every
    existing caller of this function (`scripts/build_features.py`,
    `train_two_tower.py`/`train_ranker.py`/`run_pipeline.py` via their own
    feature-building, the dashboard's batch load). Pass an explicit
    `datetime` (e.g. `datetime.now()`) only from a call site that actually
    wants recency-aware features. Offline temporal evaluation does NOT go
    through this function (it calls `build_user_features` directly per
    point-in-time profile with `reference_time=cutoff`; see
    `evaluation.temporal_future_purchase`).
    """
    products = bundle.products.list_products()
    encoder = encoder or SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model,
        device=config.embedding.device,
        batch_size=config.embedding.encode_batch_size,
    )

    cache, recomputed = get_or_compute_product_embeddings(
        products, encoder, resolve_path(config.embedding.cache_path)
    )
    product_embeddings_by_id = cache.as_dict()

    all_purchases = bundle.purchases.list_all_purchases()
    all_cart_items = bundle.cart.list_all_cart_items()
    all_reviews = bundle.reviews.list_all_reviews()
    product_features = build_product_features(products, all_purchases, all_cart_items, all_reviews)

    user_ids = bundle.users.list_user_ids()
    engagement_profiles = {
        uid: build_engagement_profile(
            uid, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
        )
        for uid in user_ids
    }

    text_embeddings = build_user_text_embeddings(list(engagement_profiles.values()), encoder)
    product_lookup = {p.id: p for p in products}
    # STEP 6 (docs/data-mapping.md section 15): catalog-only, built ONCE
    # and reused for every user - never a leakage surface (see
    # `features.price` module docstring).
    price_context = build_price_catalog_context(products)
    user_features = {
        uid: build_user_features(
            profile,
            product_lookup,
            product_embeddings_by_id,
            config.features,
            text_embeddings=text_embeddings,
            reference_time=reference_time,
            price_context=price_context,
        )
        for uid, profile in engagement_profiles.items()
    }

    return FeaturePipelineResult(
        product_embeddings=cache,
        product_features=product_features,
        engagement_profiles=engagement_profiles,
        user_features=user_features,
        embeddings_recomputed=recomputed,
    )
