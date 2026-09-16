"""Genuine STRONG / SPARSE / NO_HISTORY classification through the real
`backend_api` adapter pipeline (docs/data-mapping.md 19.14, section 6 of
the train-serve parity audit).

Deterministic, fixture-based (not the live backend - see
`scripts/verify_activity_loading_fix.py` for the live strong-user proof;
the live backend does not reliably expose known SPARSE/NO_HISTORY real
accounts on demand, so those two tiers are proven here instead, end to
end through `build_backend_api_adapters` -> `build_engagement_profile`
-> `build_user_features` -> `serving.cold_start.determine_history_tier`
- the exact same functions live serving calls, just with a controlled
input instead of whatever the real backend happens to contain right
now).

Default thresholds (`configs/base.yaml` / `ColdStartConfig` defaults):
STRONG >= 5 engagement events, SPARSE = 1-4, NO_HISTORY = 0.
"""

from __future__ import annotations

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.adapters.engagement import build_engagement_profile
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.config import AppConfig, ColdStartConfig
from recommendation.features.price import build_price_catalog_context
from recommendation.features.user_features import build_user_features
from recommendation.serving.cold_start import HistoryTier, determine_history_tier
from tests._backend_fakes import FakeBackendClient

_CATS = [{"slug": "groceries", "name": "Groceries"}]
_PRODS = [
    {"slug": f"p{i}", "productId": i, "name": f"Product {i}", "price": 5.0, "stockQuantity": 10, "categorySlug": "groceries"}
    for i in range(1, 8)
]


def _build(tmp_path, roster, activities):
    client = FakeBackendClient(products=_PRODS, categories=_CATS, roster=roster, activities=activities)
    resolver = ExternalIdentityResolver(tmp_path / "reg.json")
    bundle = build_backend_api_adapters(
        client=client, resolver=resolver,
        activity_cache_path=tmp_path / "activity_cache.json",
        user_activity_cache_path=tmp_path / "user_activity_cache.json",
    )
    return bundle


def _tier_for(bundle, user_id: int) -> HistoryTier:
    profile = build_engagement_profile(
        user_id, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    price_context = build_price_catalog_context(products)
    features = build_user_features(profile, product_lookup, {}, AppConfig().features, price_context=price_context)
    return determine_history_tier(features.total_engagement_events, ColdStartConfig())


def _acts(guid: str, n: int) -> list[dict]:
    return [
        {"userId": guid, "actionType": "AddToCart", "productId": (i % 7) + 1, "timestamp": f"2026-08-01T00:{i:02d}:00"}
        for i in range(n)
    ]


def test_genuine_strong_classification(tmp_path):
    roster = [{"guid": "strong-user"}]
    bundle = _build(tmp_path, roster, _acts("strong-user", 6))
    user_id = bundle.users.list_user_ids()[0]
    assert _tier_for(bundle, user_id) == HistoryTier.STRONG


def test_genuine_sparse_classification(tmp_path):
    roster = [{"guid": "sparse-user"}]
    bundle = _build(tmp_path, roster, _acts("sparse-user", 2))
    user_id = bundle.users.list_user_ids()[0]
    assert _tier_for(bundle, user_id) == HistoryTier.SPARSE


def test_genuine_no_history_classification(tmp_path):
    """A valid, known (roster-listed) user with genuinely zero activity -
    proves the completeness fetch correctly confirms "nothing more exists"
    (`hasNext=false` on an empty first page) rather than defaulting to
    NO_HISTORY simply because a fetch never happened.
    """
    roster = [{"guid": "cold-user"}]
    bundle = _build(tmp_path, roster, activities=[])
    user_id = bundle.users.list_user_ids()[0]
    assert _tier_for(bundle, user_id) == HistoryTier.NO_HISTORY


def test_sparse_and_no_history_users_are_distinguishable_in_the_same_load(tmp_path):
    """Not a coincidence of one-user fixtures: three real users loaded
    together classify independently and correctly.
    """
    roster = [{"guid": "u-strong"}, {"guid": "u-sparse"}, {"guid": "u-cold"}]
    activities = _acts("u-strong", 6) + _acts("u-sparse", 2)
    bundle = _build(tmp_path, roster, activities)
    by_guid = {}
    for uid in bundle.users.list_user_ids():
        # Resolve which fixture guid this internal id maps to via the
        # adapter's own guid map (test-internal, not a production API).
        guid = bundle.purchases._guid_by_internal[uid]
        by_guid[guid] = _tier_for(bundle, uid)
    assert by_guid["u-strong"] == HistoryTier.STRONG
    assert by_guid["u-sparse"] == HistoryTier.SPARSE
    assert by_guid["u-cold"] == HistoryTier.NO_HISTORY
