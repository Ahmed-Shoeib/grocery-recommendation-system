"""LIVE smoke test for the real backend REST integration - NOT part of the
deterministic pytest suite (it needs network access to a running backend).

Proves end to end:

    backend REST API (GET /api/ai/products, GET /api/ai/user-activities,
                      GET /api/categories, + Bearer-gated
                      /api/users/{guid}, /api/reviews)
        -> recommendation.backend.client / auth / loader / identity
        -> recommendation.adapters.backend_factory.build_backend_api_adapters
        -> canonical AdapterBundle / EngagementProfile
        -> existing feature engineering + cold-start tiering + eligibility gate

Does NOT train, retrain, rebuild an index, or write model artifacts. It
DOES write the identity registry (that is the point - run it twice and the
ProductId/GUID -> int mapping must be identical the second time).

Usage:
    RECS_BACKEND_API_BASE_URL=https://<host>:<port> \
    RECS_BACKEND_TLS_VERIFY=false \
    RECS_DATA_SOURCE=backend_api \
    RECS_BACKEND_SERVICE_CLIENT_ID=... \
    RECS_BACKEND_SERVICE_CLIENT_SECRET=... \
    python scripts/backend_api_smoke_test.py

**Both credential vars are now REQUIRED** (docs/data-mapping.md 19.5): the
2026-09-15 atomic switch made `GET /api/ai/products`/
`GET /api/ai/user-activities` (Bearer-gated) the authoritative
catalog/activity sources, so without credentials the whole load fails -
this script detects that up front and fails clearly rather than letting an
unhandled exception fly. This script prints token *metadata* only - never
the token itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.adapters.engagement import build_engagement_profile
from recommendation.backend.auth import ENV_CLIENT_ID, ENV_CLIENT_SECRET
from recommendation.backend.client import BackendApiClient
from recommendation.backend.errors import BackendApiError, BackendAuthError
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.serving.cold_start import determine_history_tier
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.config import get_config


def _check_service_auth(config) -> bool:
    """Exercise the client-credentials exchange and the Bearer-gated
    endpoints directly, so a failure is attributed to auth rather than
    surfacing later as a confusing catalog-load crash. Prints token
    metadata only.

    Since the 2026-09-15 atomic source switch, `GET /api/ai/products`/
    `GET /api/ai/user-activities` are Bearer-gated and mandatory - missing
    credentials now fail the whole run (returns `False`), not just the
    optional reviews/profile-enrichment signals.
    """
    client = BackendApiClient(config.backend_api)
    if not client.has_service_credentials():
        print(f"\nFAIL: service credentials NOT configured ({ENV_CLIENT_ID}/{ENV_CLIENT_SECRET} unset) - "
              "GET /api/ai/products and GET /api/ai/user-activities are Bearer-gated and mandatory "
              "since the 2026-09-15 source switch (docs/data-mapping.md 19.5); the catalog/activity "
              "load cannot proceed without them.")
        return False
    try:
        raw_reviews = client.list_reviews()
    except BackendAuthError as exc:
        if exc.status_code == 403:
            # Token is valid (it works for /api/users/{guid}); this client
            # just lacks the /api/reviews scope. Degrade like the loader
            # does - not a smoke-test failure. See docs/data-mapping.md 19.6.
            print("\nservice auth: token OK, but GET /api/reviews -> 403 "
                  "(service client lacks the reviews scope) - degrading to 0 reviews")
            return True
        print(f"\nFAIL: service auth configured but /api/reviews returned {exc.status_code}: {exc}")
        return False
    except BackendApiError as exc:
        print(f"\nFAIL: service auth configured but /api/reviews failed: {type(exc).__name__}: {exc}")
        return False
    print(f"\nservice auth: OK (token acquired and reused) - GET /api/reviews returned {len(raw_reviews)} row(s)")
    if raw_reviews:
        sample = raw_reviews[0]
        print(f"  sample row: reviewId={sample.review_id} userId={sample.user_id} userGuid={sample.user_guid} "
              f"productId={sample.product_id} rating={sample.rating} createdAt={sample.created_at}")
        with_product_id = sum(1 for r in raw_reviews if r.product_id is not None)
        with_user_guid = sum(1 for r in raw_reviews if r.user_guid is not None)
        print(f"  {with_product_id}/{len(raw_reviews)} row(s) carry a productId, "
              f"{with_user_guid}/{len(raw_reviews)} carry a userGuid - actual join success/failure "
              "against the live catalog and activity-stream users is reported below, after the "
              "full catalog load (see 'reviews join diagnostics' in the log output).")
    return True


def main() -> int:
    config = get_config()
    if config.paths.data_source != "backend_api":
        print("NOTE: paths.data_source is not 'backend_api'; set RECS_DATA_SOURCE=backend_api to match live wiring.")
    print(f"backend base_url = {config.backend_api.base_url!r}  tls_verify = {config.backend_api.tls_verify}")

    if not _check_service_auth(config):
        return 1

    bundle = build_backend_api_adapters(config)
    products = bundle.products.list_products()
    user_ids = bundle.users.list_user_ids()
    reviews = bundle.reviews.list_all_reviews()
    print(f"\ncatalog: {len(products)} products, {len(user_ids)} active users, {len(reviews)} canonical reviews")
    if not products:
        print("FAIL: no products returned from the backend")
        return 1

    # Eligibility gate sees real stock values from this source. Use the
    # same dedicated embedding cache path `api.service._load_data_snapshot`
    # uses for this source, so this script never clobbers the synthetic /
    # SQLite caches.
    feature_config = config.model_copy(
        update={"embedding": config.embedding.model_copy(
            update={"cache_path": "data/processed/product_embeddings_backend_api.npz"})}
    )
    product_features_result = run_feature_pipeline(bundle, feature_config)
    product_features = product_features_result.product_features
    rules = build_eligibility_rules(config.eligibility)
    result = apply_eligibility(list(product_features), product_features, rules)
    print(f"eligibility: {len(result.eligible_ids)}/{len(product_features)} products pass isActive/stockQuantity")

    product_lookup = {p.id: p for p in products}
    for uid in sorted(user_ids)[:10]:
        engagement = build_engagement_profile(
            uid, bundle.users, bundle.purchases, bundle.cart, bundle.clicks,
            bundle.search, bundle.chatbot, bundle.reviews,
        )
        total = (
            len(engagement.clicks) + len(engagement.purchases) + len(engagement.cart_items)
            + len(engagement.searches) + (1 if engagement.chatbot_context is not None else 0)
        )
        tier = determine_history_tier(total, config.cold_start)
        sample = [product_lookup[c.product_id].name for c in engagement.clicks[:2] if c.product_id in product_lookup]
        print(
            f"  user {uid}: signals={total} tier={tier.value} "
            f"click={len(engagement.clicks)} cart={len(engagement.cart_items)} purchase={len(engagement.purchases)} "
            f"pref_cats={engagement.profile.preferred_categories!r} sample_clicks={sample}"
        )

    print("\nOK - backend REST data flowed through the canonical pipeline with no schema changes.")
    print(f"identity registry written to: {get_config().paths.backend_identity_registry} "
          f"(re-run this script; the mapping must be identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
