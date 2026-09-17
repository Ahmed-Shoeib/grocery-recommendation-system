# Grocery Recommendation System

A production personalized recommendation engine for a grocery
e-commerce backend: Two-Tower retrieval, an approximate nearest-neighbor
(ANN) index, a neural ranker, and cold-start-aware, diversity-respecting,
eligibility-filtered re-ranking - served live against the backend's real
REST API, using the backend's own database `Product.Id` as the single
product identifier end to end.

## Overview

- **Serving**: the real backend REST API (`RECS_DATA_SOURCE=backend_api`)
  - products, categories, reviews, and a scalable, bounded/complete
  activity-loading architecture. See [Production Serving](#production-serving).
- **Training / offline evaluation**: a controlled, production-aligned
  SQLite database whose category taxonomy, price/stock ranges, and
  action semantics mirror the real backend, so a model trained here
  transfers to live serving without a domain shift. See
  [Training vs Serving](#training-vs-serving).
- **Product identity**: the backend's own `Product.Id` is canonical
  everywhere - catalog, activities, features, retrieval, ranking, and the
  API response. See [Product Identity Contract](#product-identity-contract).
- **Retrieval**: a Two-Tower neural model (user tower / item tower, both
  L2-normalized into a shared 128-D space) + an ANN index (FAISS locally,
  ScaNN in Docker/Linux) for sub-linear candidate retrieval.
- **Ranking**: a 22-feature neural MLP re-scores retrieved candidates
  with richer, more explicit signal than the retrieval embedding alone
  exposes.
- **Re-ranking**: category-diversity penalty (continuous, not a hard
  quota) + a final eligibility check.
- **Cold start**: three-level personalization (STRONG / SPARSE /
  NO_HISTORY) sized against a user's total engagement signal, with a
  fallback to category or global popularity.

## Architecture

Production request path:

```
Live Backend API
        |
Recommendation Service
        |
Two-Tower retrieval
        |
ANN (FAISS / ScaNN)
        |
Neural ranker (22 features)
        |
Diversity re-ranking + final eligibility check
        |
Top-K recommendations (real backend Product.Id)
```

Concretely, for `RECS_DATA_SOURCE=backend_api`:

```
Backend REST API
   (GET /api/ai/products, /api/categories, /api/users, /api/reviews,
    /api/ai/user-activities)
        |
Bounded global activity sync + complete lazy per-user history
        |
Complete EngagementProfile per requested user
        |
Trained Two-Tower + ranker (production_safe_v2)
        |
Real-catalog ANN (built from live GET /api/ai/products)
        |
Recommendations (real backend Product.Id, no internal remapping)
```

## Key Features

- Real-time personalized retrieval + ranking against live backend data
- Canonical, non-contiguous backend `Product.Id` identity throughout the
  pipeline - no generated or internal product identifier anywhere
- Cold-start tiering with popularity-based fallback for new/sparse users
- Bounded, self-healing activity sync with complete per-user lazy history
- Category-diversity re-ranking and a final in-stock eligibility check
- Startup artifact validation (contract version, feature schema,
  dimensions, and - for `backend_api` - a canary against a regressed,
  contiguous `1..N` product-id scheme)

## Product Identity Contract

The backend database `Product.Id` is the **single canonical product
identifier**, end to end. No intermediate or generated identity space
exists anywhere in the pipeline:

```
Backend Product.Id = 105
        |
Recommender product_id = 105
        |
Features / retrieval / ANN / ranker = 105
        |
Public API product_id = 105
```

The same id is used consistently across every source and every stage:

```
backend product ProductId
  ==
activity ProductId
  ==
recommender product_id
  ==
ANN item id mapping
  ==
recommendation response product_id
```

Real backend `ProductId`s are **non-contiguous** (e.g. `82, 83, 84, 85,
86, ..., 105, ..., 162`, with genuine gaps). The recommender never
assumes or requires a contiguous `1..N` range, and never maps a product
to a separate internal integer. `ExternalIdentityResolver`
(`backend.identity`) is still used where the backend itself has no
numeric identity of its own - category slugs and user GUIDs - but is
never invoked for products once the backend supplies a `productId`.

Examples confirmed live against the backend catalog and the production
recommendation API:

| Product | Backend `Product.Id` |
|---|---|
| Pineapple | `105` |
| Zucchini | `162` |
| Apple Red Delicious | `85` |
| Solid Potato | `156` |
| Plum | `106` |

`serving.startup_validation.validate_backend_api_product_identity`
enforces this at process startup: a `backend_api` artifact whose item
ids form a dense `1..N` run (the signature of a regressed, generated
identity scheme) fails startup rather than serving silently-wrong ids.

## Training vs Serving

Training and serving intentionally use different data sources - the
code never blurs the line between them:

```
TRAINING (offline)
data/sqlite/production_aligned_training.db
        |
scripts/train_backend_api_pipeline.py
        |
models/backend_api/

SERVING (production, RECS_DATA_SOURCE=backend_api)
Live backend REST API
        |
Recommendation API (FastAPI)
        |
models/backend_api/  (already-trained artifacts, loaded read-only)
```

- `production_aligned_training.db` is a synthetic, production-aligned
  SQLite dataset - the real backend's exact category names, calibrated
  price/stock distributions, and production-safe action semantics
  (PURCHASE/ADD_TO_CART/SEARCH/CLICK/CHATBOT). It is never the source of
  a live production request.
- Production requests are always served from the live backend API,
  never from SQLite.
- `models/backend_api/` is the one artifact set both paths agree on:
  `train_backend_api_pipeline.py` trains the Two-Tower/ranker weights
  against the SQLite dataset, then `build_live_backend_ann.py` re-embeds
  the *live* backend catalog through those already-trained weights (pure
  inference, no retraining) so the serving-time ANN/item embeddings are
  keyed by real backend `Product.Id`s.

## Feature Contract

- **Contract version**: `production_safe_v2`
  (`retrieval.two_tower.feature_encoding.CURRENT_CONTRACT_VERSION`)
- **Two-Tower item numeric dim**: 5 - `normalized_price`, `log_review_count`,
  `average_rating`, `has_rating`, `category_relative_price`
- **Two-Tower user numeric dim**: 8 - `log_purchase_count`,
  `log_cart_item_count`, `log_search_count`, `log_total_engagement_events`,
  `has_chatbot_context`, `has_preferred_category`, `has_semantic_embedding`,
  `normalized_typical_price`
- **Ranker**: 22 features (`ranking.features.RANKING_FEATURE_NAMES`)

| # | Feature | # | Feature |
|---|---|---|---|
| 1 | `user_log_purchase_count` | 12 | `item_category_relative_price` |
| 2 | `user_log_cart_item_count` | 13 | `category_affinity_match` |
| 3 | `user_log_search_count` | 14 | `preferred_category_match` |
| 4 | `user_log_total_engagement_events` | 15 | `semantic_cosine_similarity` |
| 5 | `user_has_chatbot_context` | 16 | `has_semantic_similarity` |
| 6 | `user_has_preferred_category` | 17 | `user_normalized_typical_price` |
| 7 | `item_normalized_price` | 18 | `user_has_price_profile` |
| 8 | `item_log_review_count` | 19 | `price_relative_distance` |
| 9 | `item_average_rating` | 20 | `price_tier_match` |
| 10 | `item_has_rating` | 21 | `retrieval_score` |
| 11 | `item_log_stock_quantity` | 22 | `retrieval_rank_normalized` |

Catalog-wide product purchase/cart counts were removed entirely from
both the ranker and the Two-Tower item tower: the real backend has no
aggregate/popularity endpoint capable of reproducing a true lifetime
count efficiently, so training-vs-serving values could never be
guaranteed to match. They remain available as a **serving-only fallback
heuristic** (`serving.fallback.global_popularity_ranking` /
`category_popularity_ranking`, used for SPARSE/NO_HISTORY candidate
fallback only). See `docs/production-feature-parity-audit.md` and
`docs/data-mapping.md` for the full rationale.

`serving.startup_validation` rejects any artifact whose
`contract_version`/`feature_names` don't match the running code's
expectations, so a stale or mismatched artifact fails loudly at startup
rather than silently serving wrong predictions.

## Eligibility

A product must be **in stock** (`stock_quantity > 0`) to be recommended
- checked once, cheaply, before retrieval (pre-retrieval eligibility),
and again as a final lightweight safety net after ranking/re-ranking. An
`isActive`-based rule was removed entirely from the config: the real
backend `Products` table has no such column, so that check was already a
structural no-op against live data.

## Cold Start / User History

- **STRONG** history -> personalized retrieval + ranking.
- **SPARSE** / **NO_HISTORY** -> category- or global-popularity fallback,
  blended with whatever personalization signal is available.
- A user's live backend activity history is lazily synchronized: the
  first time a user is actually recommended, their full activity feed is
  walked to genuine completion (not just a bounded recent window) and
  cached; later requests reuse that cache until it goes stale, then
  refresh with a cheap incremental delta rather than a full re-fetch.
- This complete-history cache persists correctly across service refresh
  cycles - a periodic data reload never truncates an already-completed
  user back to a partial view.

## Project Structure

```
src/recommendation/
  adapters/       Canonical AdapterBundle interfaces + backend_api / sqlite / synthetic implementations
  api/            FastAPI app, routes, RecommendationService (startup/orchestration)
  backend/        Backend REST client, auth, identity resolution, activity sync/cache, DTO -> canonical mapping
  config.py       Typed configuration (configs/base.yaml / configs/docker.yaml)
  embeddings/     Sentence Transformer product-text encoding + content-hash-validated cache
  evaluation/     Temporal future-purchase protocol, offline report persistence, latency measurement
  features/       User/product feature engineering, price, recency
  ranking/        Neural ranker: 22-feature vector, model, training/serialization
  reranking/      Category-diversity re-ranking
  retrieval/      Two-Tower model + feature encoding; FAISS/ScaNN ANN index
  schemas/        Canonical Product/User/Engagement/Event schemas
  serving/        Request-time pipeline: eligibility, cold-start, fallback, orchestration
  sqlite/         SQLite connection + loader for the training/offline-evaluation source
  synthetic/      Original synthetic dataset generator (data_source: "synthetic", kept for backward compatibility)
  ui/             Streamlit dashboard + its HTTP client of the FastAPI service

scripts/
  train_backend_api_pipeline.py          Current, sole training entrypoint (Two-Tower + ANN + ranker, one run)
  build_live_backend_ann.py              Rebuild the live-serving ANN from the real backend catalog (no retraining)
  generate_production_aligned_sqlite.py  Regenerate data/sqlite/production_aligned_training.db
  verify_activity_loading_fix.py         Live verification: bounded startup, no per-user fan-out
  verify_train_serve_parity.py           Live verification: complete-history fetch cost + a real recommendation
  live_serving_smoke_test.py             Live serving smoke test against a bounded activity sample
  run_api.py / run_dashboard.py          Launch the FastAPI service / Streamlit dashboard
  generate_backend_shaped_sqlite.py      Generate the backend-shaped SQLite integration-test fixture
  generate_offline_report.py             Persist an offline evaluation report for GET /v1/metrics/offline

tests/            pytest suite (production_safe_v2 contract, backend_api, activity sync/cache, canonical
                  product identity, cold start, ANN, ranking, reranking, eligibility, artifact validation, API/service)
configs/          base.yaml (native dev, FAISS) / docker.yaml (Docker/Linux, ScaNN)
data/sqlite/      Tracked training databases (production_aligned_training.db is current)
docs/             Architecture/data-mapping reference and the production feature-parity audit history
```

## Setup

Requires Python 3.11-3.13.

```bash
git clone <repo-url>
cd grocery-recommendation-system
python -m venv .venv
.venv/Scripts/activate      # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -e ".[full]"
```

## Configuration

`configs/base.yaml` (native dev, FAISS) / `configs/docker.yaml`
(Docker/Linux, ScaNN) hold the typed defaults (`src/recommendation/config.py`).
Override individual values via environment variables or a `.env` file
(gitignored) - see `.env.example` for the full list. The variables that
matter most:

| Variable | Purpose |
|---|---|
| `RECS_DATA_SOURCE` | `synthetic` \| `sqlite` \| `backend_api` |
| `RECS_BACKEND_API_BASE_URL` | Live backend base URL (`backend_api` only) |
| `RECS_BACKEND_TLS_VERIFY` | Set `false` only for a self-signed dev backend |
| `RECS_BACKEND_SERVICE_CLIENT_ID` / `RECS_BACKEND_SERVICE_CLIENT_SECRET` | Service-to-service credentials (secret, never committed) |
| `RECS_API_PORT` | API listen port inside the container (default `8000`) |

Never commit `.env`, service credentials, or `data/processed/*` (runtime
caches) - all are gitignored.

## Training

```bash
python scripts/train_backend_api_pipeline.py
```

- **Source**: `data/sqlite/production_aligned_training.db` - a
  controlled, versioned SQLite database using the real backend's exact
  six category names, production-like price/stock ranges, multi-favorite
  users, and production-safe action semantics. The real backend is used
  only for schema/domain verification, never queried directly by this
  pipeline.
- **Split protocol**: temporal future-purchase evaluation - per-user
  cutoffs from real `action_time` values, history truncated strictly
  before the cutoff, held-out future PURCHASE events as ground truth.
  Every evaluation point is leakage-audited; the pipeline aborts if any
  violation is found.
- **What gets trained**: the Two-Tower retrieval model, then a
  temporally-evaluated ANN, then the 22-feature ranker (negatives sampled
  from the same ANN's retrieved candidates) - end to end, in one script.
- **Output**: `models/backend_api/` (`two_tower/`, `ranker/`,
  `vector_index/`, `offline_report.json`) - gitignored, never committed
  (see [Artifacts](#artifacts)).

`--db <path>` overrides the training database; `--embed-cache <path>`
overrides the product-text embedding cache path.

After training, rebuild the live-serving ANN against the real backend
catalog (no retraining - pure inference through the already-trained item
tower):

```bash
python scripts/build_live_backend_ann.py
```

## Running the API

```bash
python scripts/run_api.py
# or, with the real backend as the serving source:
RECS_DATA_SOURCE=backend_api RECS_BACKEND_API_BASE_URL=https://<host>:<port> python scripts/run_api.py

# Equivalent, in Docker:
docker compose up api
```

## Docker

```bash
# Production-serving api container (RECS_DATA_SOURCE=backend_api,
# models/backend_api/ mounted read-only from the host)
docker compose up api
# host 8001 -> container 8000 (the app always listens on 8000 inside the
# container; 8001 is the host-side port a reverse proxy forwards to)

# Streamlit dashboard - a pure HTTP client of the api service
docker compose up dashboard
# host 8501 -> container 8501

# Training (profile-gated, SQLite only, explicit --db, never inherits backend_api)
docker compose --profile train run --rm train
```

`models/` and `data/` are bind-mounted, never baked into the image -
`models/backend_api/` must already exist on the host (trained locally,
then transferred out of band - never via git) before `docker compose up
api` will start successfully. `configs/docker.yaml`'s own file-level
default stays `data_source: "sqlite"`; the `api` service overrides it to
`backend_api` on its own, so an ordinary test build never silently
depends on live backend credentials.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/v1/health` | Liveness check |
| GET | `/v1/ready` | Readiness: catalog / Two-Tower / ranker / VectorIndex all loaded |
| GET | `/v1/users/{user_id}/recommendations` | Top-K personalized recommendations |
| GET | `/v1/users` | List known users (dashboard) |
| GET | `/v1/users/{user_id}/profile` | Engagement/feature snapshot for one user |
| GET | `/v1/metrics/offline` | Persisted offline evaluation report |

Example:

```
GET /v1/users/1/recommendations?limit=10
```

Every `product_id` in the response is the backend's real `Product.Id` -
see [Product Identity Contract](#product-identity-contract). Response
items also carry `product_name`, `category`, `price`, `is_active`, and
`stock_quantity` for display without a separate catalog lookup.

## Testing

```bash
python -m pytest -q
```

Current: **835 passed, 3 skipped, 0 failed**.

Coverage includes: canonical backend `Product.Id` preservation end to
end (catalog/activity/review joins, non-contiguous ids, no
resolver-generated identity), the live backend adapter and its
bounded/lazy activity sync, cold-start tiering, startup/artifact
validation, the full API surface, and eligibility/re-ranking.

## Metrics

Current `production_safe_v2` artifacts, trained against
`data/sqlite/production_aligned_training.db` (temporal future-purchase
protocol):

| Stage | Metric | Validation | Test |
|---|---|---|---|
| Two-Tower retrieval | Recall@10 | 0.676 | 0.593 |
| Ranker | AUC | 0.850 (val) | - |
| Full pipeline | Precision@10 | 0.076 | 0.073 |
| Full pipeline | Recall@10 | 0.755 | 0.734 |
| Full pipeline | NDCG@10 | 0.566 | 0.546 |
| Full pipeline | MRR | 0.510 | 0.491 |

Full architecture rationale and live-verification evidence live in
`docs/data-mapping.md` and `docs/production-feature-parity-audit.md`.

## Production Deployment / Status

- **Live URL**: https://recommender.54-93-65-9.sslip.io
- **Swagger**: https://recommender.54-93-65-9.sslip.io/docs
- **Serving config**: `RECS_DATA_SOURCE=backend_api`, artifacts from
  `models/backend_api/` (`production_safe_v2` contract)
- **Host mapping**: `8001:8000`, proxied by Nginx

Live-validated:

- returned `product_id` values equal the real backend `Product.Id` for
  every sampled recommendation, with no internal remapping
- recommendations are active, in stock, and unique
- `/v1/ready` reports all four checks healthy
- live backend connectivity (categories, products, user activities)
  confirmed end to end

Code ships via GitHub (this repository); trained model artifacts
(`models/backend_api/`) ship separately, out of band from git, directly
to the deployment target - they are gitignored and never committed.
Expected host layout, matching `docker-compose.yml`'s bind mounts:

```
repo/
├── src/, scripts/, configs/, ...   (from git)
├── models/
│   └── backend_api/                (transferred separately)
│       ├── two_tower/
│       ├── ranker/
│       └── vector_index/
├── data/                           (runtime caches - gitignored, created on first run)
└── .env                            (gitignored - service credentials, never in git)
```

## Artifacts

`models/backend_api/` (Two-Tower, ranker, vector index, offline report)
is:

- **generated locally** by `scripts/train_backend_api_pipeline.py` +
  `scripts/build_live_backend_ann.py`
- **gitignored** - never committed to this repository
- **required for production serving** (`RECS_DATA_SOURCE=backend_api`)
- **transferred separately** to the deployment target - not via git

`models/sqlite_baseline/` (a legacy artifact set from an earlier,
pre-`production_safe_v2` architecture) is likewise gitignored and is not
part of the current production system.

## Tech Stack

Python, FastAPI, TensorFlow / Keras 3, FAISS (native dev) / ScaNN
(Docker/Linux), Sentence-Transformers (`all-MiniLM-L6-v2`), Pydantic,
Streamlit (dashboard), Docker Compose.
