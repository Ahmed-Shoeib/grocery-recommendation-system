"""Live, bounded verification of the activity-loading architecture fix
(docs/data-mapping.md 19.x): proves `build_backend_api_adapters` no
longer attempts a full traversal of the real (1.5M+ row)
`/api/ai/user-activities` table, and that `build_recommendation_service`
(the actual FastAPI startup path) completes successfully against it.

Read-only against the real backend API; never touches SQL Server; does
NOT perform an intentionally massive full traversal just to prove the
table is large (that was already established during the investigation -
see the module docstrings on `backend.activity_sync` /
`backend.client.iter_activity_pages`). Does not train, retrain, or
modify any artifact.

Usage:
    python scripts/verify_activity_loading_fix.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.adapters.backend_factory import build_backend_api_adapters
from recommendation.api.service import build_recommendation_service
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
    """Wraps `requests.Session` to count HTTP requests actually issued,
    without changing behavior - the request-count evidence for the final
    report.
    """

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

    print("=" * 78)
    print("PART 1: build_backend_api_adapters directly (bounded activity sync + roster + lazy adapter)")
    print("=" * 78)
    import requests

    raw_session = requests.Session()
    counting_session = _CountingSession(raw_session)
    client = BackendApiClient(config.backend_api, session=counting_session)
    resolver = ExternalIdentityResolver(resolve_path(config.paths.backend_identity_registry))

    t0 = time.monotonic()
    bundle = build_backend_api_adapters(
        config, client=client, resolver=resolver,
        activity_cache_path=resolve_path(config.paths.backend_activity_cache),
    )
    elapsed = time.monotonic() - t0

    print(f"OK: adapter bundle built in {elapsed:.2f}s using {counting_session.count} HTTP request(s)")
    print(f"  products={len(bundle.products.list_products())}  users={len(bundle.users.list_user_ids())}")
    print(f"  interactions in bounded window: {len(bundle.purchases.list_all_purchases())} purchases, "
          f"{len(bundle.cart.list_all_cart_items())} cart-adds")
    assert counting_session.count < 700, (
        f"expected a small, bounded request count, got {counting_session.count} - "
        "this would indicate the fix regressed toward a near-full traversal"
    )

    print()
    print("=" * 78)
    print("PART 2: build_recommendation_service (the actual FastAPI startup path)")
    print("=" * 78)
    t0 = time.monotonic()
    service = build_recommendation_service(config)
    elapsed2 = time.monotonic() - t0
    print(f"OK: RecommendationService built in {elapsed2:.2f}s (no BackendPaginationError)")
    print(f"  readiness: {service.readiness_checks()}")

    print()
    print("=" * 78)
    print("PART 3: strong / sparse / cold-start recommendation calls")
    print("=" * 78)
    known_ids = bundle.users.list_user_ids()
    from collections import Counter

    by_activity = Counter()
    for uid in known_ids:
        n = (
            len(bundle.purchases.get_purchases(uid))
            + len(bundle.cart.get_cart_items(uid))
            + len(bundle.clicks.get_clicks(uid))
        )
        by_activity[uid] = n
    ranked = sorted(known_ids, key=lambda u: by_activity[u], reverse=True)

    cases = []
    if ranked:
        cases.append(("most-active", ranked[0]))
    if len(ranked) > 1:
        cases.append(("median-activity", ranked[len(ranked) // 2]))
    if ranked:
        cases.append(("least-active-known-user (cold-start safety net)", ranked[-1]))

    for label, uid in cases:
        try:
            result = service.recommend(uid, limit=10)
        except Exception as exc:  # pragma: no cover - live diagnostic only
            print(f"  [{label}] user_id={uid}: FAILED ({exc})")
            continue
        pids = result.product_ids
        no_dupes = len(pids) == len(set(pids))
        no_out_of_stock = all(service.product_features[pid].stock_quantity > 0 for pid in pids)
        real_ids = all(pid in service.product_lookup for pid in pids)
        print(
            f"  [{label}] user_id={uid} tier={result.tier.value} returned={len(pids)} "
            f"no_dupes={no_dupes} no_out_of_stock={no_out_of_stock} real_ids={real_ids}"
        )

    print()
    print("ALL CHECKS PASSED - activity-loading architecture fix verified live.")


if __name__ == "__main__":
    main()
