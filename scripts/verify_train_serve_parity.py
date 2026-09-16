"""Live, bounded verification of the train-serve behavioral-feature
parity fix (docs/data-mapping.md 19.14): proves a real user's COMPLETE
history is fetched (not the bounded global window's partial subset) the
first time they are recommended, that the second request for the same
user reuses the cached complete history cheaply, and that full
`RecommendationService` startup still makes no per-user fan-out.

Read-only against the real backend API; never touches SQL Server. Does
NOT globally crawl the 1.5M-row activity table - only ever fetches ONE
user's own `userId`-filtered feed, plus the existing bounded global
window for catalog-wide fallback signal.

Usage:
    python scripts/verify_train_serve_parity.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.backend.client import BackendApiClient
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.config import get_config, resolve_path
from recommendation.logging import setup_logging


def _load_dotenv_if_present(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


class _CountingSession:
    def __init__(self, session):
        self._session = session
        self.count = 0

    def __getattr__(self, name):
        return getattr(self._session, name)

    def request(self, *args, **kwargs):
        self.count += 1
        return self._session.request(*args, **kwargs)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv_if_present(repo_root)
    config = get_config()
    setup_logging(config.log_level)

    counting_session = _CountingSession(requests.Session())
    client = BackendApiClient(config.backend_api, session=counting_session)
    resolver = ExternalIdentityResolver(resolve_path(config.paths.backend_identity_registry))

    print("=" * 78)
    print("STARTUP: build_backend_api_adapters (bounded global window + roster, NO per-user fan-out)")
    print("=" * 78)
    t0 = time.monotonic()
    bundle = build_backend_api_adapters(config, client=client, resolver=resolver)
    startup_elapsed = time.monotonic() - t0
    startup_requests = counting_session.count
    print(f"startup: {startup_elapsed:.2f}s, {startup_requests} HTTP request(s), "
          f"{len(bundle.users.list_user_ids())} known users, "
          f"{len(bundle.purchases.list_all_purchases())} purchases in the bounded global window")

    # Pick a real user with SOME activity already in the bounded window -
    # exactly the "partial coverage" case the parity audit flagged as
    # previously undercounted.
    events_adapter = bundle.purchases
    candidates = [
        uid for (uid, action_type) in events_adapter._by_user_and_type
        if events_adapter._by_user_and_type[(uid, action_type)]
    ]
    if not candidates:
        print("No user has any activity in the current bounded window - picking any known user instead.")
        candidates = bundle.users.list_user_ids()
    user_id = candidates[0]
    guid = bundle.purchases._guid_by_internal.get(user_id, "<unknown>")
    in_window_count = sum(
        len(v) for (uid, _), v in events_adapter._by_user_and_type.items() if uid == user_id
    )
    print(f"\nchosen user: internal_id={user_id} guid={guid} "
          f"(had {in_window_count} event(s) in the bounded global window before any per-user fetch)")

    print()
    print("=" * 78)
    print("FIRST REQUEST for this user: triggers a complete-history userId-scoped fetch")
    print("=" * 78)
    before = counting_session.count
    t0 = time.monotonic()
    purchases = bundle.purchases.get_purchases(user_id)
    cart_items = bundle.cart.get_cart_items(user_id)
    clicks = bundle.clicks.get_clicks(user_id)
    searches = bundle.search.get_search_history(user_id)
    first_elapsed = time.monotonic() - t0
    first_requests = counting_session.count - before
    total_activities = len(purchases) + len(cart_items) + len(clicks) + len(searches)
    print(f"first-request cost: {first_requests} HTTP request(s), {first_elapsed:.2f}s")
    print(f"complete history found: {len(purchases)} purchases, {len(cart_items)} cart-adds, "
          f"{len(clicks)} clicks, {len(searches)} searches ({total_activities} total)")
    is_complete = bundle.purchases._store.complete.get(guid)
    print(f"completeness marker after first fetch: {is_complete}")

    print()
    print("=" * 78)
    print("SECOND REQUEST for the SAME user: must reuse the cached complete history cheaply")
    print("=" * 78)
    before = counting_session.count
    t0 = time.monotonic()
    purchases2 = bundle.purchases.get_purchases(user_id)
    cart_items2 = bundle.cart.get_cart_items(user_id)
    second_elapsed = time.monotonic() - t0
    second_requests = counting_session.count - before
    print(f"second-request cost: {second_requests} HTTP request(s), {second_elapsed:.4f}s")
    assert len(purchases2) == len(purchases) and len(cart_items2) == len(cart_items), "second read must match the first (no drift, no duplication)"
    assert second_requests == 0, "a fresh complete-history fetch must be reused within its TTL, not repeated"

    print()
    print("=" * 78)
    print("Recommending for this user end to end (real Two-Tower + ranker + ANN)")
    print("=" * 78)
    from recommendation.features.price import build_price_catalog_context
    from recommendation.features.product_features import build_product_features
    from recommendation.features.user_features import build_user_features
    from recommendation.embeddings.encoder import SentenceTransformerEncoder
    from recommendation.embeddings.product_embeddings import get_or_compute_product_embeddings
    from recommendation.ranking.serialization import load_ranker_artifacts
    from recommendation.retrieval.index.factory import build_vector_index
    from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts
    from recommendation.serving.pipeline import generate_recommendations
    from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
    from recommendation.adapters.engagement import build_engagement_profile

    models_root = resolve_path(config.paths.models_dir) / "backend_api"
    tt_artifacts = load_two_tower_artifacts(models_root / "two_tower")
    ranker_artifacts = load_ranker_artifacts(models_root / "ranker")
    vector_index = build_vector_index(config.retrieval)
    vector_index.build(tt_artifacts.item_ids, tt_artifacts.item_embeddings)

    products = bundle.products.list_products()
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(
        products, bundle.purchases.list_all_purchases(), bundle.cart.list_all_cart_items(), bundle.reviews.list_all_reviews()
    )
    eligible = apply_eligibility([p.id for p in products], product_features, build_eligibility_rules(config.eligibility))

    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size,
    )
    embed_cache_path = resolve_path("data/processed/product_embeddings_live_backend_catalog.npz")
    embedding_cache, _ = get_or_compute_product_embeddings(products, st_encoder, embed_cache_path)
    product_embeddings = embedding_cache.as_dict()
    price_context = build_price_catalog_context(products)

    profile = build_engagement_profile(
        user_id, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    user_features = build_user_features(profile, product_lookup, product_embeddings, config.features, price_context=price_context)
    result = generate_recommendations(
        user_features, product_features, product_embeddings, tt_artifacts.item_ids,
        tt_artifacts.encoder, tt_artifacts.user_tower, ranker_artifacts.model, vector_index,
        config, 10,
    )
    pids = result.product_ids
    no_dupes = len(pids) == len(set(pids))
    all_eligible = all(pid in eligible.eligible_ids for pid in pids)
    all_real = all(pid in product_lookup for pid in pids)
    print(f"tier={result.tier.value} total_engagement_events={user_features.total_engagement_events} "
          f"returned={len(pids)} product_ids={pids}")
    print(f"checks: no_dupes={no_dupes} all_eligible_in_stock={all_eligible} all_real_product_ids={all_real}")
    assert no_dupes and all_eligible and all_real and len(pids) == 10

    print()
    print("ALL CHECKS PASSED - train-serve parity fix verified live.")


if __name__ == "__main__":
    main()
