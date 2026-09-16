"""End-to-end proof that the behavior-derived model features
(`ranking.features.RANKING_FEATURE_NAMES_USER`,
`retrieval.two_tower.feature_encoding.USER_NUMERIC_FEATURE_NAMES`) built
from a `backend_api` adapter bundle reflect a user's COMPLETE history,
not just whatever fell inside the bounded global activity window - the
train-serve parity gap this phase exists to close (docs/data-mapping.md
19.14).
"""

from __future__ import annotations

import numpy as np

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.adapters.engagement import build_engagement_profile
from recommendation.backend.dtos import ApiActivity
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.config import AppConfig
from recommendation.features.price import build_price_catalog_context
from recommendation.features.user_features import build_user_features
from tests._backend_fakes import FakeBackendClient

_CATS = [{"slug": "groceries", "name": "Groceries"}]
_PRODS = [{"slug": "oj", "productId": 501, "name": "OJ", "price": 4.0, "stockQuantity": 5, "categorySlug": "groceries"}]


def _build(tmp_path, activities):
    client = FakeBackendClient(products=_PRODS, categories=_CATS, roster=[{"guid": "g1"}], activities=activities)
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle = build_backend_api_adapters(
        client=client, resolver=resolver,
        activity_cache_path=tmp_path / "activity_cache.json",
        user_activity_cache_path=tmp_path / "user_activity_cache.json",
    )
    return bundle, client


def _purchase(ts):
    return {"userId": "g1", "actionType": "PlaceOrder", "productId": 501, "timestamp": ts}


def test_purchase_count_reflects_complete_history_not_the_bounded_window(tmp_path):
    """The bounded global window (fed to the adapter's constructor) sees
    only ONE purchase; the user's real (userId-filtered) history has
    THREE. `UserFeatures.purchase_count` - which directly becomes
    `user_log_purchase_count` (ranker) and the Two-Tower user tower's
    `log_purchase_count` input - must reflect the complete count.
    """
    in_window = [_purchase("2026-08-03T00:00:00")]
    bundle, client = _build(tmp_path, in_window)
    complete_history = [_purchase("2026-08-03T00:00:00"), _purchase("2026-08-02T00:00:00"), _purchase("2026-01-01T00:00:00")]
    client._activities = [ApiActivity.model_validate(a) for a in complete_history]

    user_id = bundle.users.list_user_ids()[0]
    profile = build_engagement_profile(
        user_id, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    products = bundle.products.list_products()
    price_context = build_price_catalog_context(products)
    features = build_user_features(profile, {p.id: p for p in products}, {}, AppConfig().features, price_context=price_context)

    assert features.purchase_count == 3, "purchase_count must reflect the complete per-user history, not the bounded window's partial view of 1"
    assert np.log1p(features.purchase_count) != np.log1p(1), "the ranker/Two-Tower log-count input must differ from the truncated value"
