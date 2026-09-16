"""Live `backend_api` serving smoke test (task section 22) - exercises the
trained `models/backend_api/` artifacts + the just-built real-catalog ANN
(`scripts/build_live_backend_ann.py`) through `serving.pipeline
.generate_recommendations` (the same function `api.service
.RecommendationService.recommend` calls) for several REAL backend users.

**Known blocker this script works around, not silently**: a full
`GET /api/ai/user-activities` pull (what `api.service
.build_recommendation_service`'s normal `data_source=backend_api` startup
path uses via `adapters.backend_factory.build_backend_api_adapters`) is
currently infeasible against the live backend - the endpoint hard-caps
`pageSize` at 20 server-side regardless of what is requested, and the
first page's pagination cursor alone decodes to an offset of 1,506,718,
meaning the real `UserActivities` table has grown past ~1.5 MILLION rows.
A full traversal would need 75,000+ HTTP round trips - the existing client
correctly refuses this (`BackendPaginationError` at its 10,000-page safety
cap) rather than hanging indefinitely. See the accompanying report's
"blocking production readiness" section - this is a real, currently-open
gap in the live `backend_api` path, not something this script fixes.

What this script does instead: fetches a BOUNDED, small number of recent
activity pages directly (default 100 pages x 20 rows = up to 2,000 rows -
seconds, not hours) to find a few real users with enough recent activity
to exercise the STRONG/SPARSE personalized-retrieval code paths with real
data, plus a synthetic zero-signal profile (using the real catalog) to
exercise NO_HISTORY/cold-start. This is an honest, explicitly-labeled
PARTIAL smoke test - it proves the trained artifacts + real ANN + serving
pipeline work correctly end-to-end against real product/category data, not
that the full-history `RecommendationService` startup path works (it
currently does not, for the reason above).

Does NOT train, retrain, or modify any artifact. Read-only against the
real backend API; never touches SQL Server.

Usage:
    python scripts/live_serving_smoke_test.py [--activity-pages 100]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.adapters.product_adapter import InMemoryProductCatalogAdapter
from recommendation.adapters.user_events_adapter import UserEventsAdapter
from recommendation.backend.client import BackendApiClient
from recommendation.backend.dtos import ApiActivity
from recommendation.backend.identity import ExternalIdentityResolver
from recommendation.backend.loader import load_backend_catalog, load_backend_events
from recommendation.embeddings.encoder import SentenceTransformerEncoder
from recommendation.embeddings.product_embeddings import get_or_compute_product_embeddings
from recommendation.features.price import build_price_catalog_context
from recommendation.features.product_features import build_product_features
from recommendation.features.user_features import build_user_features
from recommendation.ranking.serialization import load_ranker_artifacts
from recommendation.retrieval.index.factory import build_vector_index
from recommendation.retrieval.two_tower.serialization import load_two_tower_artifacts
from recommendation.schemas.engagement import EngagementProfile
from recommendation.schemas.user import UserProfile
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.serving.pipeline import generate_recommendations
from recommendation.serving.startup_validation import (
    validate_ranker_artifacts,
    validate_two_tower_artifacts,
    validate_vector_index_compatibility,
)
from recommendation.config import get_config, resolve_path
from recommendation.logging import get_logger, setup_logging

logger = get_logger(__name__)


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


def _fetch_bounded_activities(client: BackendApiClient, max_pages: int) -> list[ApiActivity]:
    """Fetches up to `max_pages` pages of `GET /api/ai/user-activities`
    directly (NOT via `client.list_activities()`, which loops until
    exhaustion or its 10,000-page safety cap - infeasible here, see module
    docstring). The backend hard-caps page size at 20 regardless of the
    requested `pageSize`.
    """
    rows: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        params = {"pageSize": 100}
        if cursor is not None:
            params["cursor"] = cursor
        resp = client._request("/api/ai/user-activities", params, auth=True)
        page = resp.get("data", [])
        rows.extend(page)
        pagination = resp.get("pagination", {})
        if not pagination.get("hasNext"):
            break
        cursor = pagination.get("nextCursor")
        if cursor is None:
            break
    return [ApiActivity.model_validate(r) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-pages", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv_if_present(repo_root)
    config = get_config()
    setup_logging(config.log_level)

    models_root = resolve_path(config.paths.models_dir) / "backend_api"

    print("=" * 78)
    print("Loading trained artifacts + validating (same checks api.service.build_recommendation_service runs)")
    print("=" * 78)
    tt_artifacts = load_two_tower_artifacts(models_root / "two_tower")
    ranker_artifacts = load_ranker_artifacts(models_root / "ranker")
    validate_two_tower_artifacts(tt_artifacts, config)
    validate_ranker_artifacts(ranker_artifacts)
    print(f"Two-Tower: PASS (contract_version={tt_artifacts.encoder.contract_version}, "
          f"ann_catalog_source={tt_artifacts.metadata.get('ann_catalog_source')})")
    print(f"Ranker: PASS ({len(ranker_artifacts.feature_names)} features)")
    print(f"Two-Tower item catalog (the just-built live ANN): {len(tt_artifacts.item_ids)} real ProductIds")
    assert all(isinstance(pid, int) for pid in tt_artifacts.item_ids), "non-integer id in live ANN"

    vector_index = build_vector_index(config.retrieval)
    vector_index.build(tt_artifacts.item_ids, tt_artifacts.item_embeddings)
    validate_vector_index_compatibility(vector_index.size, len(tt_artifacts.item_ids))
    print(f"VectorIndex: PASS ({vector_index.size} products indexed)")

    print()
    print("=" * 78)
    print("Fetching real catalog + product features (same as build_live_backend_ann.py)")
    print("=" * 78)
    client = BackendApiClient(config.backend_api)
    resolver = ExternalIdentityResolver(resolve_path(config.paths.backend_identity_registry))
    catalog = load_backend_catalog(client, resolver)
    products_adapter = InMemoryProductCatalogAdapter(catalog.categories, catalog.tags, catalog.products, catalog.product_tags)
    products = products_adapter.list_products()
    product_lookup = {p.id: p for p in products}
    product_features = build_product_features(products, all_purchases=[], all_cart_items=[], all_reviews=[])
    eligibility_rules = build_eligibility_rules(config.eligibility)
    eligible = apply_eligibility([p.id for p in products], product_features, eligibility_rules)
    print(f"real products: {len(products)}  eligible (stock>0): {len(eligible.eligible_ids)}")

    st_encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    embed_cache_path = resolve_path("data/processed/product_embeddings_live_backend_catalog.npz")
    embedding_cache, recomputed = get_or_compute_product_embeddings(products, st_encoder, embed_cache_path)
    product_embeddings = embedding_cache.as_dict()
    print(f"semantic embeddings: {len(product_embeddings)} products (recomputed={recomputed}, should be False - reused build_live_backend_ann.py's cache)")

    price_context = build_price_catalog_context(products)

    print()
    print("=" * 78)
    print(f"Fetching a BOUNDED activity sample ({args.activity_pages} pages, backend caps at 20 rows/page - "
          "see module docstring re: the full-history pagination blocker)")
    print("=" * 78)
    activities = _fetch_bounded_activities(client, args.activity_pages)
    print(f"fetched {len(activities)} activity rows (partial/recent sample, NOT full history)")

    resolver_for_events = ExternalIdentityResolver(resolve_path(config.paths.backend_identity_registry))
    # Re-load catalog against this resolver instance so product ids resolve consistently.
    catalog_for_events = load_backend_catalog(client, resolver_for_events)
    interactions, guid_by_internal = load_backend_events(activities, resolver_for_events, catalog_for_events)
    print(f"canonical interactions resolved: {len(interactions)}  distinct real users touched: {len(guid_by_internal)}")

    action_counts = Counter(e.action_type.value for e in interactions)
    print(f"action distribution in this bounded sample: {dict(action_counts)}")

    events_adapter = UserEventsAdapter(interactions)
    events_by_user: dict[int, int] = Counter(e.user_id for e in interactions)
    ranked_users = [uid for uid, _ in events_by_user.most_common()]

    print()
    print("=" * 78)
    print("Running generate_recommendations for real users (strong/sparse-in-sample) + a cold-start profile")
    print("=" * 78)

    def _profile_for(user_id: int) -> EngagementProfile:
        return EngagementProfile(
            user_id=user_id,
            profile=UserProfile(user_id=user_id),
            clicks=events_adapter.get_clicks(user_id),
            purchases=events_adapter.get_purchases(user_id),
            cart_items=events_adapter.get_cart_items(user_id),
            searches=events_adapter.get_search_history(user_id),
            chatbot_context=events_adapter.get_chatbot_context(user_id),
        )

    test_cases: list[tuple[str, EngagementProfile]] = []
    if ranked_users:
        test_cases.append(("most-active-in-sample", _profile_for(ranked_users[0])))
    if len(ranked_users) > 1:
        mid = ranked_users[len(ranked_users) // 2]
        test_cases.append(("median-activity-in-sample", _profile_for(mid)))
    test_cases.append(("cold-start (no history)", EngagementProfile(user_id=-1, profile=UserProfile(user_id=-1))))

    all_ok = True
    for label, profile in test_cases:
        user_features = build_user_features(
            profile, product_lookup, product_embeddings, config.features, price_context=price_context,
        )
        result = generate_recommendations(
            user_features, product_features, product_embeddings, tt_artifacts.item_ids,
            tt_artifacts.encoder, tt_artifacts.user_tower, ranker_artifacts.model, vector_index,
            config, args.top_n,
        )
        pids = result.product_ids
        real_ids_ok = all(pid in product_lookup for pid in pids)
        no_synthetic_ids = all(isinstance(pid, int) and pid in {p.id for p in products} for pid in pids)
        no_dupes = len(pids) == len(set(pids))
        no_out_of_stock = all(product_features[pid].stock_quantity > 0 for pid in pids)
        within_limit = len(pids) <= args.top_n
        n_categories = len({product_lookup[pid].category_name for pid in pids if pid in product_lookup})
        ok = real_ids_ok and no_synthetic_ids and no_dupes and no_out_of_stock and within_limit
        all_ok = all_ok and ok
        print(
            f"\n[{label}] user_id={profile.user_id} tier={result.tier.value} "
            f"total_engagement_events={user_features.total_engagement_events}"
        )
        print(f"  returned {len(pids)}/{args.top_n} product_ids: {pids}")
        print(f"  sources: {result.sources}")
        print(f"  distinct categories in result: {n_categories}")
        print(
            f"  checks: real_ids={real_ids_ok} no_synthetic_ids={no_synthetic_ids} no_dupes={no_dupes} "
            f"no_out_of_stock={no_out_of_stock} within_limit={within_limit} fill_rate={result.fill_rate:.2f} "
            f"-> {'PASS' if ok else 'FAIL'}"
        )

    print()
    print("=" * 78)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print("=" * 78)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
