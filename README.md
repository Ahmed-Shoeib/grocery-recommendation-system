# Grocery Recommendation System

A production-oriented, modular personalized recommendation service for a
grocery e-commerce backend: Two-Tower retrieval, an approximate-nearest-
neighbor `VectorIndex` (ScaNN in production/Docker, FAISS on native
Windows dev), a neural ranker, cold-start-aware re-ranking, and
business-rules/eligibility filtering, served through a versioned FastAPI
service and inspectable through an internal Streamlit dashboard.

Trained and served, by default, against a **synthetic**, backend-ERD-shaped
SQLite dataset (`data/sqlite/backend_shaped_synthetic.db` - 1,200 products,
1,000 users, a genuine per-event `User_events` activity log) - architected
from the start so the real grocery backend can be substituted with no model
redesign once it's available (see
[How the real backend will replace synthetic adapters](#how-the-real-backend-will-replace-synthetic-adapters)).
The original, smaller in-package synthetic generator (~50 products, ~300
users, no timestamps) is still present and selectable
(`paths.data_source: "synthetic"`), kept only for backward compatibility.

See `docs/data-mapping.md` for the full ERD reconciliation, scope
boundaries, and the rationale behind every design decision, and
`docs/production-readiness.md` for a critical, classified review of what
is and isn't ready for real production traffic.

## Current status

- **Data source**: `data/sqlite/backend_shaped_synthetic.db` - a
  backend-ERD-shaped SQLite database, entirely synthetic - is the
  default (`paths.data_source: "sqlite"`).
- **`User_events` engagement contract**: one append-style activity-log
  table (`id, user_id, product_id, action_time, action_type`) is the
  sole engagement-truth source for all five personalization signals -
  CLICK, SEARCH, ADD_TO_CART, PURCHASE, CHATBOT. `Cart`/`Cart_Item`/
  `Order`/`Order_Item` exist in the same database (kept relationally
  consistent by the generator) but are deliberately never read by this
  adapter path, so the same real-world purchase/cart action can never be
  double-counted through two independent code paths.
- **Recency weighting**: exponential half-life decay
  (`recency_weight = 0.5 ** (age_days / half_life_days)`, default
  `half_life_days = 21`) applied to category/brand affinity, the
  semantic-embedding blend, and the user price profile - opt-in per
  call via an explicit `reference_time`, never an implicit
  `datetime.now()` inside reusable feature functions.
- **Price-aware derived features**: effective price, discount status,
  catalog price tiers, category-relative price, a user price profile
  (purchase-history -> preferred-category-prior -> catalog-prior
  fallback), and price-distance/tier-match cross features - a
  **learned compatibility signal**, never a hard "cheaper is better"
  business rule. Entirely derived in the feature layer; no backend
  ERD/schema field was added or changed for any of this.
- **Temporal future-purchase evaluation**: per-user cutoffs built from
  real `action_time` values, history truncated strictly before the
  cutoff, held-out future PURCHASE events as ground truth - a separate
  protocol from the original non-temporal leave-one-out one still used
  to train/evaluate the legacy synthetic-only artifacts.
- **29-feature ranker** - 9 item-numeric + 9 user-numeric encoder dims
  feeding the Two-Tower model, 29 explicit features feeding the ranker -
  the recency+price configuration served by default, validated by a
  controlled ablation experiment (`docs/data-mapping.md` §17) against a
  23-feature/no-price baseline condition. The reduced-feature code path
  has since been removed from the codebase entirely - the current
  architecture always builds the full 29-feature/9-9-dim price-aware
  shape; `include_price_features` remains only as a reported metadata
  field (always `True`), not a branching flag.
- **Pre-retrieval eligibility + final safety check**: `isActive`/
  `stockQuantity` gate candidate generation itself - before Two-Tower/
  VectorIndex retrieval and every fallback source ever runs - plus a
  final lightweight re-validation immediately before the response is
  built, as defense-in-depth, not the primary mechanism.
- **FAISS (native Windows dev) / ScaNN (Docker/Linux, primary)** - both
  do genuine approximate nearest-neighbor (ANN) search over the same
  L2-normalized 128-D embeddings (FAISS: HNSW; ScaNN: tree partitioning +
  asymmetric-hashing quantization + exact-score reordering -
  `retrieval.index.faiss_index`/`scann_index`). `EligibilityRestrictedIndex`
  restricts which retrieved ids may enter the candidate pool at query
  time via bounded oversampling + progressive widening (not a full-index
  scan) - it does not rebuild either backend's index structure, so a
  stock/active change never triggers a retrain or an index rebuild.
- **`models/sqlite_baseline/`** is the current SQLite-serving artifact
  root (Two-Tower + ranker + FAISS index + the persisted offline
  report). Legacy pre-price-feature-era artifacts and one-off ablation
  artifacts have been removed as stale, regenerable, gitignored build
  output - neither was read by the current runtime.
- **FastAPI is the single serving path**: it is the only process that
  ever constructs a `RecommendationService`, loads Two-Tower/ranker
  artifacts, builds the VectorIndex, or touches an adapter/SQLite
  connection.
- **Streamlit is a pure HTTP client**: every recommendation and every
  piece of user/catalog data it shows comes from an HTTP call to the
  running FastAPI service (`ui.api_client.RecommendationApiClient`); it
  never loads a model artifact or builds a `RecommendationService`
  itself, and requires FastAPI to already be running.
- **Persisted offline report**: `scripts/generate_offline_report.py`
  runs the temporal evaluation protocol separately (batch, on demand)
  and writes `models/sqlite_baseline/offline_report.json`;
  `GET /v1/metrics/offline` only reads and provenance-validates that
  persisted file - it never recomputes recommendations or runs an
  evaluation pass inside the HTTP request.

## Architecture

**Online serving** (every request, milliseconds; default `paths.data_source: "sqlite"`):

```
SQLite backend-shaped synthetic DB (data/sqlite/backend_shaped_synthetic.db)
   Product, Category, Tag, User, Review, User_events (CLICK/SEARCH/ADD_TO_CART/PURCHASE/CHATBOT)
                              |
                     Data Adapter Layer  (AdapterBundle, 8 ABCs - same shape for every data source)
                              |
                 Canonical Engagement Model  (EngagementProfile)
                              |
                      Feature Engineering
                 +------------+------------+
    User Features (recency-weighted        Product Features (effective_price,
    category/brand affinity, price          price_tier, category_relative_price,
    profile, semantic embedding)            popularity/rating - never recency-weighted)
                                             |
                                  Sentence Transformer (384-D, frozen)
                                             |
                              +--------------+--------------+
                          User Tower                   Item Tower
                        (128-D, L2-norm)             (128-D, L2-norm)
                              +--------------+--------------+
                                     cosine compatibility
                                             |
                     Hard PRE-RETRIEVAL eligibility (isActive, stock - catalog state,
                     evaluated BEFORE any candidate generation, personalized or fallback)
                                             |
              VectorIndex (ScaNN primary/Docker, FAISS Windows dev fallback)
              - full catalog embeddings; EligibilityRestrictedIndex restricts which
                retrieved ids may enter the candidate pool, at query time only
                                             |
                                   Oversized candidate pool
                                    (capped by eligible count)
                                             |
                     Cold-start candidate assembly where applicable
                (STRONG: personalized only · SPARSE: blend w/ category+global
                 popularity · NO_HISTORY: waterfall fallback, no personalized part)
                                             |
                                    Neural Ranker (29-feature MLP)
                                             |
                                   Diversity re-ranking (dedup + category/brand,
                                    continuous score penalty, not a hard quota)
                                             |
                   Final lightweight eligibility safety re-check (defense in depth)
                                             |
                                        Final Top-N
                                             |
                                   RecommendationService
                                             |
                                        FastAPI (/v1)
                                             |
                                            HTTP
                                             |
                                 Streamlit (pure HTTP client)
```

**Temporal offline evaluation** (separate, batch, on demand - NOT part of
the request path above):

```
data/sqlite/backend_shaped_synthetic.db
        |
  Per-user temporal cutoffs from real action_time values
  (FULL / VAL_ONLY / INSUFFICIENT_DEPTH / ENGAGEMENT_NO_PURCHASE / NO_HISTORY)
        |
  History strictly before cutoff -> the SAME feature engineering + serving
  pipeline above -> Top-N recommendations at that point in time
        |
  Compared against held-out future PURCHASE events (the ground truth)
        |
  scripts/generate_offline_report.py -> models/sqlite_baseline/offline_report.json
        |
  GET /v1/metrics/offline  (reads + provenance-validates the persisted file only -
                             never recomputes recommendations inside the request)
```

The serving-tier names above (`STRONG`/`SPARSE`/`NO_HISTORY`,
`serving.cold_start.HistoryTier` - how much engagement HISTORY exists)
are a distinct classification from the temporal-evaluation tiers
(`FULL`/`VAL_ONLY`/`INSUFFICIENT_DEPTH`/`ENGAGEMENT_NO_PURCHASE`/
`NO_HISTORY`, `evaluation.temporal_future_purchase.TemporalEligibilityTier`
- how much PURCHASE-holdout depth a user has for evaluation purposes) -
a user can be `HistoryTier.STRONG` while temporally
`INSUFFICIENT_DEPTH`, and vice versa.

**Hard pre-retrieval eligibility, applied first**: `isActive`/
`stockQuantity` are global catalog-eligibility facts, not model
knowledge, so they gate candidate generation itself - inactive/out-of-
stock products never enter Two-Tower/VectorIndex retrieval, the neural
ranker, or re-ranking. This never touches the Two-Tower, the ranker, or
the VectorIndex's built structure/embeddings - retrieval restriction
happens at query time (see `retrieval.index
.eligibility_filter.EligibilityRestrictedIndex`), so changing stock or
`isActive` never triggers a retrain or an index rebuild, only a
refresh of `product_features` (the plain per-product-state dict, cheap
to recompute from current catalog state). A **final, lightweight
validation** re-checks the same two rules again immediately before the
response is built - defense in depth against a product becoming
unavailable between pre-retrieval filtering and the final response, not
the primary filtering mechanism. Other, user-specific/list-specific
business rules stay at that final stage, not pre-retrieval - pre-
retrieval is reserved for hard, global catalog eligibility only. The
pipeline still ranks/re-ranks an **oversized candidate pool**
(config-driven, `retrieval.candidate_pool_multiplier`/
`min_candidate_pool`, capped by the *eligible* catalog size), not
just the requested Top-N, so a rare final-stage exclusion still leaves
enough eligible candidates to fill the request - `fill_rate` reports how
close it came. See `docs/data-mapping.md` §5 for the full rationale.

**Four personalization signals**: previous purchases, add-to-cart
habit, searched items, and chatbot context - combined with
`preferredCategory` and `ageGroup` into a canonical `EngagementProfile`.
See `docs/data-mapping.md` for exactly which signals come from real ERD
entities today versus synthetic adapters.

**Three-level cold-start strategy**, based on a user's total engagement
signal count (`configs/base.yaml: cold_start.*`):

| Tier | Condition | Strategy |
|---|---|---|
| STRONG | signals ≥ `strong_history_min_signals` | Two-Tower → VectorIndex → Ranker, used directly. |
| SPARSE | signals ≥ `sparse_history_min_signals`, below strong | Personalized candidates weight-blended with preferred-category and global-popularity fallbacks (`cold_start.sparse_blend`). |
| NO_HISTORY | 0 signals | Deterministic ordered fallback: preferredCategory → category popularity → global popularity (`cold_start.no_history_fallback_order`). |

**ScaNN vs. FAISS**: ScaNN is the primary, production-intended
`VectorIndex` backend - confirmed Linux-only (no Windows wheel exists or
is planned upstream), so it runs inside Docker. FAISS (`faiss-cpu`, real
Windows wheels) is the native-Windows **development fallback**, so local
dev/tests/training run without Docker. Both do genuine APPROXIMATE
nearest-neighbor search over the same L2-normalized 128-D embeddings -
FAISS via `IndexHNSWFlat` (a navigable small-world graph, `METRIC_INNER_
PRODUCT`), ScaNN via tree partitioning + asymmetric-hashing quantization
with exact-score reordering of the top candidates - both still
mathematically cosine similarity, since embeddings are unit-norm. Their
top-k results are expected to overlap heavily but are not guaranteed
bit-identical (see `scripts/evaluate_ann_recall.py` for a measured
recall-vs-exact comparison). HNSW/ScaNN parameters (`M`/
`efConstruction`/`efSearch`; leaf counts/AH quantization/reorder depth)
are config-driven (`RetrievalConfig` in `utils/config.py`) and derived
from catalog size where it matters, not hard-coded for one catalog size -
switching FAISS to IVF/IVF-PQ or retuning ScaNN's tree/AH parameters
later is still a change inside one class, not an interface change. Full
rationale: `docs/data-mapping.md` §10.

**Pre-retrieval eligibility restriction is backend-agnostic**: neither
backend's index structure is filtered/rebuilt when stock or `isActive`
changes - `retrieval.index.eligibility_filter.EligibilityRestrictedIndex`
wraps either backend and restricts `search()` results to a caller-
supplied eligible-id set at query time via bounded oversampling +
progressive widening (ask for a multiple of `k`, filter, widen and retry
up to a capped number of attempts if still short) - never a full-index
scan on the normal path, since that would defeat the point of an ANN
backend as the catalog grows. ScaNN's pybind searcher has no native
per-query id-filtering hook, so rather than give FAISS and ScaNN two
different filtering code paths (native `IDSelector` for one, something
else for the other), both go through this one backend-agnostic widening
loop - the simplest abstraction that treats them identically, and it
still guarantees an inactive/out-of-stock item is never returned. See
`docs/data-mapping.md` §5 for the full rationale.

## Repository layout

```
configs/                    YAML configuration (base.yaml = Windows/FAISS dev, docker.yaml = Docker/ScaNN)
data/{raw,processed,synthetic}/   Gitignored, regenerable - never committed
data/sqlite/backend_shaped_synthetic.db   Backend-ERD-shaped SQLite dataset - COMMITTED (not gitignored), the current default data source
docs/
  erd.jpeg                  Source-of-truth backend ERD
  data-mapping.md            ERD reconciliation, scope boundaries, every design decision's rationale
  production-readiness.md    Critical review (Ready now / Acceptable limitation / Must address / Future)
models/                     Serialized model artifacts (gitignored - regenerable). sqlite_baseline/ = current SQLite-serving artifacts + the persisted offline report.
src/recommendation/
  data/
    adapters/                Backend + synthetic data adapters -> canonical schemas
    schemas/                 Canonical pydantic schemas (Category, Product, UserProfile, EngagementProfile, ...)
    synthetic/                Synthetic dataset generator
  features/                  EngagementProfile -> feature vectors
  embeddings/                 Sentence Transformer product encoding + cache
  retrieval/
    two_tower/                User Tower / Item Tower model
    index/                     VectorIndex (ScaNN primary/production - Docker, FAISS Windows dev fallback) + EligibilityRestrictedIndex (query-time pre-retrieval eligibility wrapper)
  ranking/                    Neural ranker over VectorIndex candidates (features, model, train, evaluation, serialization)
  reranking/                  Duplicate removal + category/brand diversity re-ranking
  evaluation/                  Offline metrics + latency measurement + temporal future-purchase protocol + persisted offline-report (de)serialization/provenance
  serving/                    Cold-start tiering, fallback candidates, two-stage eligibility (hard pre-retrieval gate + final lightweight validation), startup artifact validation, the full pipeline orchestrator
  api/                         FastAPI app (v1) - app/routes/schemas/dependencies, thin wrapper over serving.pipeline
  ui/                           Streamlit dashboard (dashboard.py rendering-only, api_client.py typed HTTP client) - a pure HTTP client of the FastAPI service, never loads a model artifact or RecommendationService itself
  utils/                       Config loading (incl. env var overrides), logging
scripts/                     One entrypoint per workflow step - see Training/Inference workflows below
tests/                       pytest suite (see Testing below)
Dockerfile                   Multi-stage: base / test / api / dashboard
docker-compose.yml           train (profile-gated) / api / dashboard orchestration
```

## Setup (native Windows dev)

Requires **Python 3.11–3.13** (3.14 is not yet supported by the pinned ML
libraries — verified against current TensorFlow/faiss-cpu/torch PyPI wheel
availability).

```bash
# from the repo root
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"       # lightweight: config/schema/test deps only
pip install -e ".[full]"      # everything: TF/FAISS/FastAPI/Streamlit (ScaNN excluded - no Windows wheel)
```

FAISS (not ScaNN) is the retrieval backend here - `configs/base.yaml`
selects it by default, and `retrieval-scann`'s `sys_platform=='linux'`
marker makes it a harmless no-op on Windows rather than an install
failure.

## Docker / Linux / ScaNN

ScaNN has no Windows wheel, so it only runs in the Docker/Linux image.
The same multi-stage `Dockerfile` also runs the full test suite and
serves the API/dashboard - `models/`/`data/` are never baked into the
image or git; mount them at runtime.

```bash
# 1. Build (or `docker compose build`)
docker build --target test -t grocery-recs-test .        # runs the full suite when you `docker run` it
docker build --target api -t grocery-recs-api .
docker build --target dashboard -t grocery-recs-dashboard .

# 2. Full test suite in Linux (real ScaNN, zero skips)
docker run --rm grocery-recs-test

# 3. Train (writes to the bind-mounted ./models, ./data)
docker compose --profile train run --rm train-two-tower
docker compose --profile train run --rm train-ranker

# 4. Serve
docker compose up api dashboard
# API:       http://localhost:8000/v1/health, /v1/ready, /v1/users/{id}/recommendations
# Dashboard: http://localhost:8501
```

`configs/docker.yaml` (loaded via `RECS_CONFIG_PATH=/app/configs/docker.yaml`,
set in the image) selects `retrieval.backend: scann`; `configs/base.yaml`
(native Windows) selects `faiss`. The `tensorflow~=2.20.0` pin in the
Dockerfile's `pip install` (narrower than the Windows-facing
`pyproject.toml` range) exists specifically because `scann`'s compiled
ops are ABI-incompatible with newer TensorFlow releases - see
`docs/data-mapping.md` §10 for the full story.

## Training workflow

Run in order (each script loads what the previous one produced; none
retrain what already exists unless you re-run them):

```bash
python scripts/generate_synthetic_dataset.py   # optional standalone preview - training scripts generate it themselves too
python scripts/build_features.py               # Sentence Transformer product embeddings (cached) + user/product features
python scripts/train_two_tower.py              # -> models/two_tower/
python scripts/build_vector_index.py           # VectorIndex build/correctness/latency report (FAISS locally, ScaNN in Docker)
python scripts/train_ranker.py                 # -> models/ranker/ (loads Two-Tower artifacts, does not retrain them)
```

All seeds (`synthetic_data.random_seed`, `two_tower.random_seed`,
`ranking.random_seed`) are fixed in config, so re-running this sequence
against an unchanged config reproduces the same dataset and equivalent
metrics every time.

## Inference workflow

```bash
python scripts/run_pipeline.py         # legacy synthetic pipeline eval report + qualitative examples + latency
python scripts/generate_offline_report.py  # temporal offline evaluation -> models/sqlite_baseline/offline_report.json
python scripts/run_api.py              # FastAPI service (loads artifacts once at startup, never trains) - start this FIRST
python scripts/run_dashboard.py        # Streamlit dashboard - pure HTTP client of the running API, start run_api.py first
```

### API usage

```
GET /v1/health                                    liveness
GET /v1/ready                                      readiness (catalog/Two-Tower/ranker/VectorIndex all loaded)
GET /v1/users/{user_id}/recommendations?limit=10   Top-N recommendations
GET /v1/users                                      read-only user list (for the dashboard's picker)
GET /v1/users/{user_id}/profile                    read-only engagement/feature snapshot
GET /v1/metrics/offline                            persisted temporal offline-evaluation report - reads only, never recomputes
```

Response: `product_id`, `rank`, `score`, `source` per item, PLUS
server-joined display fields - `product_name`, `category`,
`brand`, `price`, `is_active`, `stock_quantity` - so a client never needs
its own separate catalog access; still never an internal model
tensor/embedding/feature vector. Also `meta` (tier, requested/returned
counts, fill_rate, pool_size, eligibility exclusions, api/model version,
latency_ms - pipeline latency only, not full HTTP round-trip). Unknown
users get a structured `404`; invalid Top-N gets a structured `422`;
unexpected failures get a structured `500` - one consistent
`{error, message}` body shape across every failure path.

### Dashboard usage

Select a user from the full user table (shows `preferredCategory`/
`ageGroup` when present) to see: the personalization engagement signals
with explicit empty states; cold-start tier and category/brand affinity;
final recommendations with catalog info joined in for display only;
pipeline diagnostics (candidate pool size, eligibility exclusions,
source breakdown, category distribution, this request's server-side
latency); and a Metrics/Debug section reading the persisted temporal
offline-evaluation report (`GET /v1/metrics/offline` - a cheap read of
`models/sqlite_baseline/offline_report.json`, produced separately by
`scripts/generate_offline_report.py`; the dashboard triggers no
evaluation pass of its own), explicitly labeled as offline/synthetic,
never a production metric. Every value shown comes from an HTTP call to
the running FastAPI service - the dashboard requires `scripts/run_api.py`
to already be running (see Inference workflow above).

## Testing

```bash
pytest                                    # native Windows - full suite passes
docker run --rm grocery-recs-test         # Docker/Linux - full suite passes, ScaNN-specific tests run for real too
```

The skips on native Windows are exactly the ScaNN-specific tests that
need a Linux wheel: the `test_scann_index.py` and
`test_step7_scann_sqlite_integration.py` modules (each skipped as a
single collection unit via `pytest.importorskip`) plus two
individually-skipped ScaNN tests in `test_eligibility_restricted_index.py`
- all executed for real, including the FAISS-vs-ScaNN cross-backend
agreement tests, in Docker.

## Configuration

All tunables (paths, hyperparameters, candidate-pool sizing, cold-start
thresholds and blend weights, model version, random seeds) live in
`configs/base.yaml` (Windows/FAISS) or `configs/docker.yaml`
(Docker/ScaNN - a full standalone copy, not a partial override), loaded
and validated by `src/recommendation/utils/config.py`.

- Which file loads: `RECS_CONFIG_PATH` env var (defaults to `configs/base.yaml`).
- A small, explicit set of individual settings can be overridden on top
  via env vars, without editing any YAML file - useful for containers/
  deployment:

  | Env var | Overrides |
  |---|---|
  | `RECS_MODELS_DIR` | `paths.models_dir` |
  | `RECS_LOG_LEVEL` | `log_level` |
  | `RECS_RETRIEVAL_BACKEND` | `retrieval.backend` (`faiss` \| `scann`) |
  | `RECS_API_HOST` / `RECS_API_PORT` | `api.host` / `api.port` |
  | `RECS_API_DEFAULT_TOP_N` | `api.default_recommendation_count` |
  | `RECS_API_MAX_TOP_N` | `api.max_recommendation_count` |

No secrets are hardcoded anywhere - this system has none to hold (no
auth, no external API keys; the synthetic dataset and local model
artifacts are the only "data" the system touches).

## Metrics (offline, synthetic data - see caveat below)

**Scope note**: the table and latency figures below are from the
original, smaller synthetic-only pipeline (`scripts/run_pipeline.py`) -
kept as historical evidence under the non-temporal leave-one-out
protocol - not re-verified against the current default SQLite-backed
dataset. For the current `data/sqlite/backend_shaped_synthetic.db`
dataset, temporal future-purchase metrics for the current recency+price
configuration (`models/sqlite_baseline/`) exist in two different,
non-interchangeable evaluation configurations - see
`docs/data-mapping.md` §17 for the full provenance trace of why they
differ:

- **The controlled base-vs-recency+price ablation experiment**
  (evaluated at `TOP_N=20`, not the live-serving default): test Recall@20 0.088 →
  0.402, test NDCG@20 0.048 → 0.299, test MRR 0.036 → 0.268 vs. the
  ablation's base condition - evidence recency+price helped, under a
  fair, controlled comparison.
- **The current persisted, live-served baseline**
  (`models/sqlite_baseline/offline_report.json`, what
  `GET /v1/metrics/offline` actually returns, evaluated at the real
  live-serving default `top_n=10`) - full figures below. Recall/NDCG/MRR
  are lower than the ablation's `TOP_N=20` figures only because a
  10-item served list structurally can't exceed what Recall@10 already
  captures (Recall@20 equals Recall@10 here by construction), not
  because of a different model, dataset, or configuration - it is the
  identical trained model as the ablation's improved condition.

**Current SQLite offline report** (`models/sqlite_baseline/offline_report.json`,
generated `2026-08-22T13:59:24Z`, recency+price config, `top_n=10`):

| Split | Cases | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 | MRR | Mean distinct categories | Catalog coverage | Fill rate |
|---|---|---|---|---|---|---|---|---|---|
| Val | 378 | 0.0479 | 0.4788 | 0.4788 | 0.3990 | 0.3725 | 6.47 | 0.838 | 1.00 |
| Test | 204 | 0.0377 | 0.3775 | 0.3775 | 0.3093 | 0.2864 | 6.84 | 0.588 | 1.00 |

Recall@k/HitRate@k/NDCG@k at `k=20` equal the `k=10` figures above by
construction (the served list only has 10 items to begin with); at
`k=5` (val / test): Precision 0.0921 / 0.0735, Recall 0.4603 / 0.3676,
NDCG 0.3931 / 0.3060. Re-run `python scripts/generate_offline_report.py`
to regenerate these figures, or query `GET /v1/metrics/offline` directly
(see "Offline metrics architecture" below) - they will drift from the
table above as the dataset/model artifacts change.

From the full pipeline evaluation (`scripts/run_pipeline.py`, 162
held-out leave-one-out eval users, real trained artifacts, same
Two-Tower/ranker artifacts before and after the eligibility-gate
architecture change):

| | Test NDCG@10 | Test Recall@10 | Test MRR | Mean distinct categories | Catalog coverage | Fill rate |
|---|---|---|---|---|---|---|
| **Before** (eligibility applied last) - ranker only | 0.3498 | 0.7037 | 0.2597 | 4.59 | 0.92 | 1.00 |
| **Before** - full pipeline | 0.3356 | 0.6605 | 0.2361 | 6.20 | 0.88 | 1.00 |
| **After** (hard pre-retrieval gate) - ranker only | 0.3502 | 0.6975 | 0.2613 | 4.72 | 0.88 | 1.00 |
| **After** - full pipeline | 0.3333 | 0.6481 | 0.2365 | 6.19 | 0.88 | 1.00 |

**Do not over-interpret these deltas** - the catalog has only ~50
synthetic products (2 of them the deliberately inactive/out-of-stock
ones exercised by the eligibility tests), so a handful of eval users'
recommendations shifting by one rank position moves these metrics by
hundredths. The one delta that IS a direct, expected consequence of the
architecture change, not noise: **"ranker only" catalog coverage drops
from 0.92 to 0.88**, becoming identical to the full-pipeline figure -
before the change, the "ranker only" (pre-re-rank/eligibility) slice could
still include the 2 inactive/out-of-stock products (only excluded at the
very end), so they could count toward coverage; after the change they're
excluded before the ranker ever sees them, so "ranker only" and "full
pipeline" coverage are now the same by construction - proof the hard
pre-retrieval gate actually gates retrieval, not just the final list.
Fill rate stays exactly 1.00 before and after (enough eligible products
exist at this catalog scale to fill every request); diversity re-ranking
still delivers roughly the same +~35% mean-distinct-categories lift over
the ranker-only baseline it did before (unaffected by the eligibility
change - re-ranking itself wasn't touched).

**Latency** (Windows/FAISS, single machine, no load - see
`docs/production-readiness.md` for what this does and doesn't prove):

| | Before | After |
|---|---|---|
| FAISS retrieval (single query) | ~0.9ms | ~0.77ms |
| End-to-end pipeline (mean / p95) | ~295-300ms / ~330ms | ~230ms / ~238ms |

Raw FAISS retrieval latency is unaffected by design - `VectorIndex.search`
itself is unchanged; `EligibilityRestrictedIndex` only wraps it inside
the serving pipeline, and the small pool-size reduction (since pool
sizing is now capped by the *eligible* catalog count) is not enough to
explain a measurable difference on its own. The end-to-end figure
looking faster after the change is most plausibly ordinary
single-machine run-to-run variance rather than a real effect of this
change - take both numbers as sanity checks ("still fast, still
dominated by Keras `.predict()` overhead at this tiny batch scale, not by
the extra eligibility bookkeeping"), not a precise A/B benchmark.

**This is not evidence of real-world recommendation quality.** All
numbers above come from a synthetic, persona-correlated dataset with no
real user behavior - they demonstrate that the pipeline is implemented
correctly and that each stage (ranker, re-ranking, eligibility) measurably
does what it's supposed to relative to the stage before it, nothing more.
See `docs/data-mapping.md` §8.

### Offline metrics architecture

`GET /v1/metrics/offline` does **not** recompute recommendations or run
an evaluation pass inside the HTTP request. Instead:

```
scripts/generate_offline_report.py   (batch, on demand - not part of any request)
    -> loads the already-trained models/sqlite_baseline/{two_tower,ranker} artifacts
    -> re-derives the temporal splits/eval cases (docs/data-mapping.md §8.1)
    -> runs the full temporal future-purchase evaluation for val + test
    -> writes models/sqlite_baseline/offline_report.json

GET /v1/metrics/offline   (online, every request - milliseconds)
    -> reads + schema-validates the persisted JSON above
    -> provenance-validates it against the currently-loaded model/dataset identity
       (mismatch -> HTTP 409, never silently served)
    -> returns it
```

This replaced an earlier version of the endpoint that ran a full
(non-temporal) evaluation pass synchronously per request (~88s,
tripping the dashboard's client timeout) - see `docs/data-mapping.md`
§18.1 for the full history.

## Known limitations & scope

Summarized here; full rationale for each in `docs/data-mapping.md` and
`docs/production-readiness.md`:

- **Synthetic data only** - see the Metrics caveat above.
- **No timestamps on most engagement signals - legacy synthetic path
  only** ⇒ leave-one-out with a content-based leakage heuristic instead
  of a genuine temporal split (`docs/data-mapping.md` §12). The current
  default `User_events`/SQLite-sourced path DOES carry real timestamps
  and has a genuine per-user temporal future-purchase evaluation
  protocol built and applied against it.
- **No event-tracking pipeline** ⇒ no CTR/impression/conversion metrics
  anywhere - only offline proxy metrics (`docs/data-mapping.md` §7-8).
- **Search and chatbot context are synthetic-only adapters** - the
  `SearchAdapter`/`ChatbotContextAdapter` interfaces are real and ready;
  no backend table for either exists yet (`docs/data-mapping.md` §4).
- **`preferredCategory`/`ageGroup`** are confirmed-but-not-yet-live
  backend `User` fields, modeled `Optional[str]` throughout so the
  system degrades gracefully without them (`docs/data-mapping.md` §2).
- See `docs/production-readiness.md` for the full classified list,
  including what must change before real production traffic (no auth,
  no rate limiting, no TLS, an async route that blocks on synchronous
  model calls, and more).

### How the real backend will replace synthetic adapters

Every model/feature/serving component depends on the `AdapterBundle`
interface (`src/recommendation/data/adapters/base.py`) - eight ABCs
(`ProductCatalogAdapter`, `UserAdapter`, `PurchaseAdapter`,
`CartAdapter`, `ClickAdapter`, `ReviewAdapter`, `SearchAdapter`,
`ChatbotContextAdapter`) - never on the fact that
`data.adapters.factory.build_synthetic_adapters` currently populates
them from in-memory synthetic data. Pointing the system at the real
backend is: implement one `build_backend_adapters(...)` factory
returning the same `AdapterBundle` from real SQL/API calls, then swap
the one call site (`scripts/*.py`, `api.dependencies
.build_recommendation_service`) - no change to features, models,
ranking, re-ranking, eligibility, the API, or the dashboard.

This pattern is no longer just theoretical: `data.adapters.sqlite_factory
.build_sqlite_adapters` is a second, working `AdapterBundle` factory,
reading the backend-ERD-shaped, entirely-synthetic
`data/sqlite/backend_shaped_synthetic.db` (`scripts
.generate_backend_shaped_sqlite.py`) instead of the in-memory generators -
see `docs/data-mapping.md` §4's "SQLite integration" subsection for the
full mapping. It is an integration/experimentation path, not (yet) the
live API/dashboard data source, and it is read-only by construction. A
real backend factory would follow the exact same shape.

## Development phases

Each phase below was implemented and reviewed as a separate commit. The
description reflects each phase's current implementation, not just its
original scope - later work extended several phases after the initial
set was complete, and is folded into the phase it improved rather than
presented as additional stages. See `docs/data-mapping.md` §13 for the
full section-to-phase map.

1. **Foundation & architecture.** Project scaffolding, configuration
   loading, logging.
2. **Canonical data layer & dataset integration.** The `AdapterBundle`
   interface (eight ABCs) and canonical schemas; the original small
   synthetic generator (~50 products/300 users, kept for backward
   compatibility, `paths.data_source: "synthetic"`); and, as the current
   default, the backend-ERD-shaped SQLite dataset
   (`data/sqlite/backend_shaped_synthetic.db` - 1,200 products, 1,000
   users) with the confirmed `User_events` activity-log contract and
   five engagement signals (CLICK/SEARCH/ADD_TO_CART/PURCHASE/CHATBOT).
3. **Feature engineering & semantic product embeddings.** Sentence
   Transformer product embeddings; category/brand affinity; and, folded
   in by later work, recency (time-decay) weighting of behavioral
   signals and price-aware user/product features (effective price,
   price tiers, category-relative price, a user price profile) -
   `docs/data-mapping.md` §§14-15.
4. **Neural Two-Tower retrieval model.** 128-D, L2-normalized user/item
   embeddings. Current numeric encoder dimensions are **9 item-numeric /
   9 user-numeric** (extended from 7/8 by the price-aware feature work
   above); the 7/8-dimensional encoder is retained only as the base
   condition of the controlled ablation experiment below.
5. **ANN retrieval:** ScaNN (primary/production backend, Linux/Docker) +
   FAISS (native-Windows dev fallback), both genuine approximate search
   (FAISS HNSW, ScaNN tree+AH+reorder) over the same embeddings.
6. **Neural ranking** of VectorIndex candidates, richer than retrieval
   features, evaluated (NDCG/Precision/Recall/HitRate/MRR) against a
   raw-retrieval-score baseline. Current ranker uses **29 explicit
   features** (extended from 23 by the price-aware feature work); the
   23-feature vector is retained only as the base condition of the same
   ablation experiment - `docs/data-mapping.md` §17.
7. **Full serving pipeline:** three-level cold-start blending
   (strong/sparse/no-history, sized against five engagement signals),
   dedup + category/brand diversity re-ranking, business rules/
   eligibility (originally applied last, now a hard pre-retrieval
   gate - see phase 11). This is also the pipeline the temporal
   future-purchase evaluation protocol runs point-in-time, per-user
   cutoff, for its primary offline metrics - `docs/data-mapping.md`
   §§8.1, 16.
8. **Versioned (`/v1`) FastAPI recommendation API** - dependency-injected,
   model artifacts loaded once at startup, thin wrapper over the phase 7
   pipeline (no duplicated logic). Also serves persisted temporal offline
   metrics (`GET /v1/metrics/offline`) by reading a batch-generated,
   provenance-validated report rather than evaluating inside the
   request - `docs/data-mapping.md` §18.1.
9. **Internal Streamlit dashboard** for demonstrating/debugging the
   recommendation engine - user signals, cold-start tier, candidate-pool/
   eligibility diagnostics, offline metrics. Current architecture is a
   **pure HTTP client** of the FastAPI service
   (`ui.api_client.RecommendationApiClient`); it originally reused the
   API's `RecommendationService` in-process - `docs/data-mapping.md`
   §18.
10. **Production hardening:** multi-stage Docker (test/api/dashboard),
    startup artifact validation, env-var config overrides, structured
    observability, a full reliability pass, and Windows + Docker/Linux
    integration verification - see `docs/production-readiness.md` for
    the critical review.
11. **Eligibility architecture change:** hard PRE-retrieval eligibility
    (isActive/stockQuantity gate candidate generation itself, via
    `retrieval.index.eligibility_filter.EligibilityRestrictedIndex` for
    the VectorIndex path) plus a final lightweight eligibility
    re-validation as a defense-in-depth safety net - no Two-Tower/ranker
    retraining or VectorIndex rebuild required when catalog/stock state
    changes.
