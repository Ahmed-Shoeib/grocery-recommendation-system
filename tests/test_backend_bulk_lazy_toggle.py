"""`api.service._load_data_snapshot` must not let the eager "build an
engagement profile for every known user" bulk pass
(`features.pipeline.run_feature_pipeline`, used only for the dashboard's
user list) turn into one bounded per-user complete-history backend call
PER USER not yet known-complete.

Without the `LazyBackendUserEventsAdapter.lazy_enabled` toggle this file
covers, a real roster (547+ live users) would make every snapshot
build/refresh cost one extra HTTP request per such user - exactly the
per-user-request fan-out the activity-loading architecture fix
(docs/data-mapping.md 19.13) and the train-serve parity fix (19.14, which
made EVERY user - not just zero-activity ones - subject to the
complete-history check) both exist to avoid at startup. The actual
per-request serving path (`RecommendationService.recommend`) must still
get the complete-history fetch - covered separately in
`test_backend_cold_start_lazy.py`.
"""

from __future__ import annotations

import numpy as np

import recommendation.api.service as service_module
from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.config import AppConfig, PathsConfig
from tests._backend_fakes import FakeBackendClient

_CATS = [{"slug": "groceries", "name": "Groceries"}]
_PRODS = [{"slug": "oj", "productId": 501, "name": "OJ", "price": 4.0, "stockQuantity": 5, "categorySlug": "groceries"}]
_FAKE_EMBEDDING_DIM = 384


def _fake_encoder():
    from types import SimpleNamespace

    def _encode(texts, normalize=False):
        if not texts:
            return np.empty((0, _FAKE_EMBEDDING_DIM), dtype=np.float32)
        return np.zeros((len(texts), _FAKE_EMBEDDING_DIM), dtype=np.float32)

    return SimpleNamespace(model_name="fake", embedding_dim=_FAKE_EMBEDDING_DIM, encode=_encode)


def test_bulk_engagement_profile_pass_never_triggers_a_per_user_lazy_backend_call(tmp_path, monkeypatch):
    # A roster of users with NO activity in the (bounded) window at all -
    # exactly the shape that would fan out into one lazy call per user if
    # the bulk-pass toggle were missing.
    roster = [{"guid": f"g{i}", "firstName": f"U{i}", "email": f"u{i}@x.invalid"} for i in range(20)]

    def fake_build(config):
        client = FakeBackendClient(products=_PRODS, categories=_CATS, activities=[], roster=roster)
        resolver = ExternalIdentityResolver(tmp_path / "reg.json")
        bundle = build_backend_api_adapters(
            client=client, resolver=resolver, activity_cache_path=tmp_path / "activity_cache.json", user_activity_cache_path=tmp_path / "user_activity_cache.json"
        )
        bundle._test_client = client  # smuggle the fake out for assertions below
        return bundle

    monkeypatch.setattr(service_module, "build_backend_api_adapters", fake_build)

    config = AppConfig(paths=PathsConfig(data_source="backend_api"))
    snapshot = service_module._load_data_snapshot(config, _fake_encoder())

    assert len(snapshot.engagement_profiles) == 20
    per_user_lazy_calls = [c for c in snapshot.bundle._test_client.activity_page_calls if c is not None]
    assert per_user_lazy_calls == [], (
        "the bulk engagement-profile pass must not make one backend call per zero-activity user"
    )
    # The toggle must be restored afterward so real per-request serving
    # still gets the cold-start safety net.
    assert snapshot.bundle.purchases.lazy_enabled is True


def test_lazy_disabled_during_bulk_pass_is_re_enabled_even_if_the_pipeline_raises(tmp_path, monkeypatch):
    built_bundles = []

    def fake_build(config):
        client = FakeBackendClient(products=_PRODS, categories=_CATS, activities=[], roster=[{"guid": "g1"}])
        resolver = ExternalIdentityResolver(tmp_path / "reg.json")
        bundle = build_backend_api_adapters(
            client=client, resolver=resolver, activity_cache_path=tmp_path / "activity_cache.json", user_activity_cache_path=tmp_path / "user_activity_cache.json"
        )
        built_bundles.append(bundle)
        return bundle

    monkeypatch.setattr(service_module, "build_backend_api_adapters", fake_build)

    def boom(*a, **k):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(service_module, "run_feature_pipeline", boom)

    config = AppConfig(paths=PathsConfig(data_source="backend_api"))
    try:
        service_module._load_data_snapshot(config, _fake_encoder())
        raised = False
    except RuntimeError:
        raised = True

    assert raised
    assert built_bundles[0].purchases.lazy_enabled is True
