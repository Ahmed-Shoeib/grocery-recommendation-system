# Grocery Recommendation System

A production-safe personalized recommendation engine for a grocery
e-commerce backend: SQLite-controlled training, a real backend REST API
as the live serving data source, Two-Tower retrieval, an approximate
nearest-neighbor (ANN) index, a neural ranker, and cold-start-aware,
diversity-respecting, eligibility-filtered re-ranking.

Every learned model input is chosen so it can be reproduced **exactly
and efficiently** from the live backend API - see
[Production Data Contract](#production-data-contract) below.

## Overview

- **Training / offline evaluation source**: a controlled SQLite database
  (`data/sqlite/production_aligned_training.db`) whose category taxonomy,
  price/stock ranges, and action semantics mirror the real backend
  exactly - so a model trained here transfers to live serving without a
  domain shift.
- **Live serving source**: the real backend REST API
  (`data_source: "backend_api"`) - products, categories, reviews, and a
  scalable, bounded/complete activity-loading architecture (see
  [Production Serving](#production-serving)).
- **Retrieval**: a Two-Tower neural model (user tower / item tower, both
  L2-normalized into a shared 128-D space) + an ANN index (FAISS locally,
  ScaNN in Docker/Linux) for sub-linear candidate retrieval.
- **Ranking**: a 22-feature neural MLP re-scores retrieved candidates
  with richer, more explicit signal than the retrieval embedding alone
  exposes.
- **Re-ranking**: category-diversity penalty (continuous, not a hard
  quota) + a final in-stock eligibility check.
- **Cold start**: three-level personalization (STRONG / SPARSE /
  NO_HISTORY) sized against a user's total engagement signal, with a
  waterfall/blend fallback to category or global popularity.

## Current Architecture

```
SQLite production-aligned training data
        |
Feature engineering (user/product/text)
        |
Two-Tower retrieval model
        |
ANN retrieval (FAISS / ScaNN)
        |
22-feature neural ranker
        |
Category diversity re-ranking
        |
In-stock eligibility check
        |
FastAPI (GET /v1/recommendations/{user_id})
```

For live production serving, the data feeding that pipeline comes from
the real backend, not SQLite:

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
Real-product ANN (built from live GET /api/ai/products)
        |
Recommendations (real ProductIds only)
```

## Production Data Contract

Every model input is a field the real backend REST API actually exposes:

| Concept | Source |
|---|---|
| `ProductId` | `GET /api/ai/products` - the authoritative external product identity |
| name / description / category | product text embedding input (`name + category + description`) |
| price / stock | `effective_price`, price tiers, category-relative price, stock eligibility |
| category | `CategoryId`-backed, real production taxonomy |
| reviews | rating / review count |
| user behavior | purchases, cart adds, searches, chatbot mentions, clicks - via `GET /api/ai/user-activities` |
| user GUID | `GET /api/users` - the authoritative external user identity |
| favorite categories | `UserProfile.preferred_categories: list[str]`, the real `FavoriteCategory[]` join shape |

**The model does NOT depend on**: `brand`, `discountPercentage`/`salePrice`,
`isActive`, `ageGroup`, `tags`, parent-category hierarchy, or catalog-wide
lifetime purchase/cart counts - none of these exist on the real backend
(or, for lifetime popularity counts, cannot be reproduced exactly and
efficiently from it - see [Feature Contract](#feature-contract)). A
product's own current stock/price/category and a user's own complete
behavioral history remain the only signals the model is trained to see.

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

Catalog-wide product purchase/cart counts (`item_log_purchase_count`/
`item_log_cart_add_count` in the prior 24-feature contract) were removed
entirely from both the ranker and the Two-Tower item tower: the real
backend has no aggregate/popularity endpoint and no delta filter capable
of reproducing a true lifetime count efficiently, so training-vs-serving
values could never be guaranteed to match. They remain available as a
**serving-only fallback heuristic** (`serving.fallback.
global_popularity_ranking`/`category_popularity_ranking`, used for
SPARSE/NO_HISTORY candidate fallback only) - a fallback ranking has no
training semantics to be unfaithful to, so the same bounded-window
approximation that would be wrong as a learned input is perfectly fine
there. See `docs/production-feature-parity-audit.md` §20 and
`docs/data-mapping.md` §19.15 for the full rationale.

`serving.startup_validation` rejects any artifact whose
`contract_version`/`feature_names` don't match the running code's
expectations, so a stale or mismatched artifact fails loudly at startup
rather than silently serving wrong predictions.

## Training

```
python scripts/train_backend_api_pipeline.py
```

- **Source**: `data/sqlite/production_aligned_training.db` - a
  controlled, versioned SQLite database using the real backend's exact
  six category names (`Fruits`, `Packages`, `vegetables`,
  `"test category"`, `davidfr3f`, `david`), production-like price/stock
  ranges, multi-favorite users, and production-safe action semantics
  (PURCHASE/ADD_TO_CART/SEARCH/CLICK/CHATBOT). SQLite is the training and
  offline-evaluation source - the real SQL Server database is used only
  for schema/domain verification, never queried directly by this
  pipeline.
- **Split protocol**: temporal future-purchase evaluation - per-user
  cutoffs from real `action_time` values, history truncated strictly
  before the cutoff, held-out future PURCHASE events as ground truth.
  Every evaluation point is leakage-audited (`audit_no_leakage`); the
  pipeline aborts if any violation is found.
- **What gets trained**: the Two-Tower retrieval model, then a
  temporally-evaluated ANN, then the 22-feature ranker (negatives sampled
  from the same ANN's retrieved candidates) - end to end, in one script.
- **Output**: `models/backend_api/` (`two_tower/`, `ranker/`,
  `vector_index/`, `offline_report.json`) - gitignored, never committed
  (see [Artifacts](#artifacts)).

`--db <path>` overrides the training database; `--embed-cache <path>`
overrides the product-text embedding cache path.

## Production Serving

`data_source: "backend_api"` is the live serving configuration:

- **Real `ProductId`s only** - `GET /api/ai/products` is the sole
  catalog source; `GET /api/categories`, `GET /api/reviews`, and
  `GET /api/users` (the full user roster, independent of any recorded
  activity) round out the catalog/user layer.
- **Bounded global activity sync** (`backend.activity_sync`,
  `backend.activity_cache`) - the real `UserActivities` table has grown
  past 1.5 million rows with no server-side delta filter, so startup
  loads a bounded, persisted recent window (bootstrap once, incremental
  delta thereafter, self-healing fallback to a fresh bootstrap on a
  stale checkpoint) instead of ever attempting a full traversal. This
  window backs catalog-wide fallback popularity only.
- **Complete lazy per-user history** (`adapters.backend_lazy_events_adapter
  .LazyBackendUserEventsAdapter`, `backend.user_activity_cache`) - the
  first time a specific user is actually recommended, their
  `userId`-filtered activity feed is walked to genuine completion
  (`hasNext=false`, not just a page cap), replacing any partial rows the
  bounded global window already had for them. This is what keeps
  behavioral features (`user_log_purchase_count`, `category_affinity`,
  the semantic embedding, the price profile) computed from a user's
  COMPLETE history, matching how the model was trained - not silently
  truncated to whatever fell inside the bounded global window. An
  explicit completeness marker (never inferred from "has some rows")
  distinguishes a user with a few cached events from one whose history is
  genuinely, confirmedly complete.
- **Checkpoint/cache strategy**: both caches are atomic-JSON-write,
  version-checked, corruption-safe (a bad file degrades to "nothing
  cached yet," never a crash), and TTL-gated so a completed user's
  history is reused with zero network calls until it goes stale, then
  refreshed with a cheap incremental delta rather than a full re-fetch.
  Both live under `data/processed/` - gitignored, never committed.
- **No per-user fan-out at startup**: the eager "build an engagement
  profile for every known user" bulk pass (used only for the dashboard's
  user list) explicitly disables the per-user complete-history fetch
  around itself, so a large user roster never turns into one request per
  user at every startup/refresh.
- **Live-verified** (`scripts/verify_train_serve_parity.py`,
  `scripts/verify_activity_loading_fix.py`): ~3-5s / ~11 HTTP requests at
  startup; ~1 request for a user's first complete-history fetch; 0
  requests to reuse it afterward.

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
  train_backend_api_pipeline.py        Current training entrypoint (Two-Tower + ANN + ranker, one run)
  build_live_backend_ann.py            Rebuild the live-serving ANN from the real backend catalog
  generate_production_aligned_sqlite.py  Regenerate data/sqlite/production_aligned_training.db
  verify_activity_loading_fix.py       Live verification: bounded startup, no per-user fan-out
  verify_train_serve_parity.py         Live verification: complete-history fetch cost + a real recommendation
  live_serving_smoke_test.py           Live serving smoke test against a bounded activity sample
  run_api.py / run_dashboard.py        Launch the FastAPI service / Streamlit dashboard
  generate_backend_shaped_sqlite.py    Generate the backend-shaped SQLite integration-test fixture
  generate_offline_report.py           Persist an offline evaluation report for GET /v1/metrics/offline

tests/            pytest suite (production_safe_v2 contract, backend_api, activity sync/cache, identity,
                  cold start, ANN, ranking, reranking, eligibility, artifact validation, API/service)
configs/          base.yaml (native dev, FAISS) / docker.yaml (Docker/Linux, ScaNN)
data/sqlite/      Tracked training databases (production_aligned_training.db is current)
docs/             Architecture/data-mapping reference and the production feature-parity audit history
```

## Training Commands

```bash
# Current production training pipeline (Two-Tower -> ANN -> ranker, temporal evaluation)
python scripts/train_backend_api_pipeline.py

# Rebuild the live-serving ANN from the real backend catalog (no retraining)
python scripts/build_live_backend_ann.py

# Regenerate the production-aligned SQLite training database (only if the domain needs to change)
python scripts/generate_production_aligned_sqlite.py

# Equivalent, in Docker (writes to the host's ./models/backend_api/ via bind mount)
docker compose --profile train run --rm train
```

## Running API

```bash
python scripts/run_api.py
# or, with the real backend as the serving source:
RECS_DATA_SOURCE=backend_api RECS_BACKEND_API_BASE_URL=https://<host>:<port> python scripts/run_api.py

# Equivalent, in Docker (already configured for backend_api serving - see Deployment)
docker compose up api
```

`GET /v1/ready` reports readiness (catalog/Two-Tower/ranker/VectorIndex
all loaded); `GET /v1/recommendations/{user_id}` serves recommendations.

## Testing

```bash
python -m pytest -q
```

## Artifacts

`models/backend_api/` (Two-Tower, ranker, vector index, offline report)
is:

- **generated locally** by `scripts/train_backend_api_pipeline.py` +
  `scripts/build_live_backend_ann.py`
- **gitignored** - never committed to this repository
- **required for production serving** (`data_source: "backend_api"`)
- **transferred separately** to the deployment target (AWS) - not via
  git

`models/sqlite_baseline/` (a legacy artifact set from an earlier,
pre-`production_safe_v2` architecture) is likewise gitignored and is not
part of the current production system.

## Deployment

**Training and serving use two deliberately different data sources - the
Docker setup keeps them separate rather than one shared default:**

- **Training** (`docker compose --profile train run --rm train`) always
  uses SQLite (`data/sqlite/production_aligned_training.db`, passed
  explicitly via `--db`) - it never depends on live backend user-activity
  data, and never inherits a `backend_api` setting from anywhere else.
- **Production serving** (`docker compose up api`) sets
  `RECS_DATA_SOURCE=backend_api` on the `api` service only, which is also
  what `api.service.resolve_models_root` uses to select
  `models/backend_api/` as the artifact directory - automatically, with
  no separate "artifact root" setting to keep in sync. If those artifacts
  are missing or contract-incompatible, startup fails loudly rather than
  silently falling back to `models/sqlite_baseline/` or any other path.

`configs/docker.yaml`'s own file-level default stays `data_source:
"sqlite"` deliberately - it is loaded by every stage built from the
`base` image, including `docker build --target test`, so a global
`backend_api` default there would make an ordinary test run silently
depend on live backend credentials/network.

Code ships via GitHub (this repository); trained model artifacts
(`models/backend_api/`) ship separately, out of band from git, directly
to the deployment target - they are gitignored and never committed.
Expected host layout, matching `docker-compose.yml`'s bind mounts:

```
repo/
├── src/, scripts/, configs/, ...   (from git)
├── models/
│   └── backend_api/                (transferred separately - rsync/scp/etc.)
│       ├── two_tower/
│       ├── ranker/
│       └── vector_index/
├── data/                           (runtime caches - gitignored, created on first run)
└── .env                            (gitignored - service credentials, never in git)
```

`docker-compose.yml`'s `api`/`train` services load `.env` via `env_file:`
(the file path only - never a literal secret value in the compose file
itself). **Never commit `.env`, service credentials, or any runtime
cache** (`data/processed/*`) - all are gitignored; see `.env.example` for
the required variable names only.

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

Live-verified real-catalog ANN: 85 real products, 0 unknown categories,
0 duplicate/synthetic `ProductId`s (`models/backend_api/
live_ann_build_report.json`).

## Status

- Feature contract: **`production_safe_v2`**
- Ranker: **22 features**; Two-Tower item/user numeric dims: **5/8**
- Training source: `data/sqlite/production_aligned_training.db`
  (real backend's exact six categories)
- Serving source: `data_source: "backend_api"`, real `ProductId`s,
  scalable bounded/complete activity loading
- `models/sqlite_baseline/` and the original synthetic-only pipeline are
  legacy, kept only where a current test still exercises them - never
  the production serving path, and no longer what `docker compose
  --profile train` runs (that now runs the current production_safe_v2
  entrypoint - see [Deployment](#deployment))
- Current test suite: **818 passed, 3 skipped, 0 failed**

Full architecture rationale, live-verification evidence, and the
train-serve parity audit history live in `docs/data-mapping.md` and
`docs/production-feature-parity-audit.md`.
