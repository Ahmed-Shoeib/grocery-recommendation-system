# Grocery Recommendation System

A production-oriented, modular personalized recommendation service for a
grocery e-commerce backend: Two-Tower retrieval, an approximate-nearest-
neighbor `VectorIndex` (ScaNN in production/Docker, FAISS on native
Windows dev), a neural ranker, cold-start-aware re-ranking, and
business-rules/eligibility filtering, served through a versioned FastAPI
service and inspectable through an internal Streamlit dashboard.

Currently trained/served, by default, against a **synthetic**,
backend-ERD-shaped SQLite dataset (`data/sqlite/backend_shaped_synthetic.db`
- 1,200 products, 1,000 users, a genuine per-event `User_events` activity
log) - architected from the start so the real grocery backend can be
substituted with no model redesign once it's available (see
[How the real backend will replace synthetic adapters](#how-the-real-backend-will-replace-synthetic-adapters)).
The original, smaller in-package synthetic generator (~50 products, ~300
users, no timestamps) is still present and selectable
(`paths.data_source: "synthetic"`), kept only for backward compatibility.
See [Current POC status](#current-poc-status) below for a concise summary
of what's implemented today, `docs/data-mapping.md` for the full ERD
reconciliation, V1/V2 scope boundary, and every design decision's
rationale, and `docs/production-readiness.md` for a critical, classified
review of what is and isn't ready for real production use (a Phase 10
snapshot - see that document's own notice for how later mentor-driven
work folded into Phases 2-9 since).

## Current POC status

The items below reflect the CURRENT implementation of Phases 2-9,
verified directly against the running code - see `docs/data-mapping.md`
sections 14-18.1 for the full rationale behind each:

- **Data source**: `data/sqlite/backend_shaped_synthetic.db` - a
  backend-ERD-shaped SQLite database, entirely synthetic - is the
  default (`paths.data_source: "sqlite"`).
- **`User_events` engagement contract**: one append-style activity-log
  table (`id, user_id, product_id, action_time, action_type`) is the
  sole engagement-truth source for all five V1 signals - CLICK, SEARCH,
  ADD_TO_CART, PURCHASE, CHATBOT. `Cart`/`Cart_Item`/`Order`/`Order_Item`
  exist in the same database (kept relationally consistent by the
  generator) but are deliberately never read by this adapter path, so
  the same real-world purchase/cart action can never be double-counted
  through two independent code paths.
- **Recency weighting**: exponential half-life decay
  (`recency_weight = 0.5 ** (age_days / half_life_days)`, default
  `half_life_days = 21`) applied to category/brand affinity, the
  semantic-embedding blend, and the user price profile - opt-in per
  call via an explicit `reference_time`, never an implicit
  `datetime.now()` inside reusable feature functions.
- **Price-aware derived features**: effective price, discount status,
  catalog price tiers, category-relative price, a user price profile
  (purchase-history → preferred-category-prior → catalog-prior
  fallback), and price-distance/tier-match cross features - a
  **learned compatibility signal**, never a hard "cheaper is better"
  business rule. Entirely derived in the feature layer; **no backend
  ERD/schema field was added or changed for any of this.**
- **Temporal future-PURCHASE evaluation**: per-user cutoffs built from
  real `action_time` values, history truncated strictly before the
  cutoff, held-out future PURCHASE events as ground truth - a separate
  protocol from the original non-temporal leave-one-out one still used
  to train/evaluate the legacy synthetic-V1 artifacts.
- **29-feature current ranker** (Phase 6, extended by later price-aware
  feature work) - 9 item-numeric + 9 user-numeric encoder dims feeding
  the Two-Tower (Phase 4), 29 explicit features feeding the ranker
  (Phase 6) - the RECENCY+PRICE configuration served by default,
  validated by a controlled ablation experiment (`docs/data-mapping.md`
  §17). A 23-feature variant (`RANKING_FEATURE_NAMES_BASE`, 7
  item-numeric/8 user-numeric encoder dims) exists ONLY as the BASE
  condition of that ablation experiment - it is not what's currently
  served.
- **Pre-retrieval eligibility + final safety check**: `isActive`/
  `stockQuantity` gate candidate generation itself - before Two-Tower/
  VectorIndex retrieval and every fallback source ever runs - plus a
  final lightweight re-validation immediately before the response is
  built, as defense-in-depth, not the primary mechanism.
- **FAISS (native Windows dev) / ScaNN (Docker/Linux, primary)** - both
  exact brute-force search over the same L2-normalized 128-D embeddings;
  `EligibilityRestrictedIndex` restricts which retrieved ids may enter
  the candidate pool at query time - it does not rebuild either
  backend's index structure, so a stock/active change never triggers a
  retrain or an index rebuild.
- **`models/sqlite_baseline/`** is the current SQLite-serving artifact
  root (Two-Tower + ranker + FAISS index + the persisted offline
  report), distinct from the legacy top-level `models/two_tower`/
  `models/ranker` (pre-price-feature-era, synthetic-V1 only) and
  `models/ablation/base/` (the ablation experiment's BASE-condition
  artifacts only).
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

## Status

All 11 phases are complete. See "Development phases" below for exactly
what each phase implements TODAY - later mentor-driven work (recency
weighting, price-aware features, the backend-shaped SQLite dataset and
`User_events` contract, temporal future-purchase evaluation, and the
Streamlit HTTP-client rewiring) was folded into the phase it improved,
not tracked as additional stages after Phase 11. See "Current POC
status" above for a concise current-state summary, and
`docs/data-mapping.md` §13 for the full section-to-phase map.

**What's next** (deliberately out of scope for this project's own phase
plan): further optimization once a real dataset is available - see
`docs/production-readiness.md`'s "Future optimization" section.

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

**Hard pre-retrieval eligibility, applied first** (mentor-reviewed
Phase 11 architecture change, superseding the original "filter last"
design): `isActive`/`stockQuantity` are global catalog-eligibility facts,
not model knowledge, so they gate candidate generation itself -
inactive/out-of-stock products never enter Two-Tower/VectorIndex
retrieval, the neural ranker, or re-ranking. This never touches the
Two-Tower, the ranker, or the VectorIndex's built structure/embeddings -
retrieval restriction happens at query time (see `retrieval.index
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
`min_candidate_pool`, now capped by the *eligible* catalog size), not
just the requested Top-N, so a rare final-stage exclusion still leaves
enough eligible candidates to fill the request - `fill_rate` reports how
close it came. See `docs/data-mapping.md` §5 for the full rationale.

**Four V1 personalization signals**: previous purchases, add-to-cart
habit, searched items, and chatbot context - combined with
`preferredCategory` and `ageGroup` into a canonical `EngagementProfile`.
See `docs/data-mapping.md` for exactly which signals come from real ERD
entities today versus synthetic V1 adapters.

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
dev/tests/training run without Docker. Both do exact
(brute-force) normalized-inner-product search over the same
L2-normalized 128-D embeddings - mathematically identical to cosine
similarity - and are verified to produce numerically identical results
for the same request (see `docs/production-readiness.md`). At the
current ~50-item catalog, exact search is both correct and effectively
free; both backends are structured so switching to an approximate
variant (FAISS IVF/HNSW, ScaNN tree+AH) later is a change inside one
class, not an interface change. Full rationale: `docs/data-mapping.md` §10.

**Pre-retrieval eligibility restriction is backend-agnostic**: neither
backend's index structure is filtered/rebuilt when stock or `isActive`
changes - `retrieval.index.eligibility_filter.EligibilityRestrictedIndex`
wraps either backend and restricts `search()` results to a caller-
supplied eligible-id set at query time. ScaNN's brute-force pybind
searcher has no native per-query id-filtering hook, so rather than give
FAISS and ScaNN two different filtering code paths (native `IDSelector`
for one, something else for the other), both go through this one
backend-agnostic wrapper - the simplest abstraction that treats them
identically. See `docs/data-mapping.md` §5 for the full rationale.

## Repository layout

```
configs/                    YAML configuration (base.yaml = Windows/FAISS dev, docker.yaml = Docker/ScaNN)
data/{raw,processed,synthetic}/   Gitignored, regenerable - never committed
data/sqlite/backend_shaped_synthetic.db   Backend-ERD-shaped SQLite dataset - COMMITTED (not gitignored), the current default data source
docs/
  erd.jpeg                  Source-of-truth backend ERD
  data-mapping.md            ERD reconciliation, V1/V2 scope, every design decision's rationale
  production-readiness.md    Phase 10 critical review (Ready now / Acceptable limitation / Must address / Future)
models/                     Serialized model artifacts (gitignored - regenerable). sqlite_baseline/ = CURRENT SQLite-serving artifacts + the persisted offline report; top-level two_tower/ranker/ = LEGACY pre-price-feature-era synthetic-V1 artifacts; ablation/base/ = BASE-condition artifacts for the controlled ablation experiment only
src/recommendation/
  data/
    adapters/                Backend + synthetic data adapters -> canonical schemas
    schemas/                 Canonical pydantic schemas (Category, Product, UserProfile, EngagementProfile, ...)
    synthetic/                Synthetic V1 dataset generator
  features/                  EngagementProfile -> feature vectors
  embeddings/                 Sentence Transformer product encoding + cache
  retrieval/
    two_tower/                User Tower / Item Tower model
    index/                     VectorIndex (ScaNN primary/production - Docker, FAISS Windows dev fallback) + EligibilityRestrictedIndex (query-time pre-retrieval eligibility wrapper)
  ranking/                    Neural ranker over VectorIndex candidates (features, model, train, evaluation, serialization)
  reranking/                  Duplicate removal + category/brand diversity re-ranking
  evaluation/                  Offline metrics + latency measurement + temporal future-purchase protocol (temporal_future_purchase.py) + persisted offline-report (de)serialization/provenance (offline_report.py)
  serving/                    Cold-start tiering, fallback candidates, two-stage eligibility (hard pre-retrieval gate + final lightweight validation), startup artifact validation, the full pipeline orchestrator
  api/                         FastAPI app (v1) - app/routes/schemas/dependencies, thin wrapper over serving.pipeline
  ui/                           Streamlit dashboard (dashboard.py rendering-only, api_client.py typed HTTP client) - a PURE HTTP client of the FastAPI service (Phase 9's current architecture), never loads a model artifact or RecommendationService itself; data_access.py/metrics.py now run server-side only (data_access.py) or are legacy/unused by the live route (metrics.py)
  utils/                       Config loading (incl. env var overrides), logging
scripts/                     One entrypoint per workflow step - see Training/Inference workflows below
tests/                       pytest suite (see Testing below)
Dockerfile                   Multi-stage: base / test / api / dashboard
docker-compose.yml           train (profile-gated) / api / dashboard orchestration
```

## Setup (native Windows dev)

Requires **Python 3.11–3.13** (3.14 is not yet supported by the pinned ML
libraries — verified against current TensorFlow/faiss-cpu/torch PyPI wheel
availability). This repo was developed against the 3.13 interpreter at
`C:\Users\ahmed.shoeib\AppData\Local\Programs\Python\Python313\python.exe`.

```bash
# from the repo root
"C:\Users\ahmed.shoeib\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
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
`docs/data-mapping.md` §10 for the full story (a real bug found and
fixed during Phase 10 integration verification).

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
metrics every time (verified repeatedly during Phase 10 - e.g. offline
NDCG/Recall/MRR figures below matched to 4 decimal places across
independent reruns in the same session).

## Inference workflow

```bash
python scripts/run_pipeline.py         # legacy synthetic-V1 pipeline eval report + qualitative examples + latency
python scripts/generate_offline_report.py  # Phase 8: temporal offline evaluation -> models/sqlite_baseline/offline_report.json
python scripts/run_api.py              # FastAPI service (loads artifacts once at startup, never trains) - start this FIRST
python scripts/run_dashboard.py        # Streamlit dashboard - pure HTTP client of the running API (Phase 9), start run_api.py first
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
`ageGroup` when present) to see: the five V1 engagement signals with
explicit empty states; cold-start tier and category/brand affinity;
final recommendations with catalog info joined in for display only;
pipeline diagnostics (candidate pool size, eligibility exclusions,
source breakdown, category distribution, this request's server-side
latency); and a Metrics/Debug section reading the PERSISTED temporal
offline-evaluation report (`GET /v1/metrics/offline` - a cheap read of
`models/sqlite_baseline/offline_report.json`, produced separately by
`scripts/generate_offline_report.py`; the dashboard triggers no
evaluation pass of its own), explicitly labeled as offline/synthetic,
never a production metric. Every value shown comes from an HTTP call to
the running FastAPI service - the dashboard requires `scripts/run_api.py`
to already be running (see Inference workflow above).

## Testing

```bash
pytest                                    # native Windows - 387 passed, 3 skipped (ScaNN: no Windows wheel)
docker run --rm grocery-recs-test         # Docker/Linux - 404 passed, 0 skipped (ScaNN runs for real)
```

The difference is exactly the ScaNN-specific tests - the `test_scann_index.py` module (skipped as a single collection unit via `pytest.importorskip`) plus two individually-skipped ScaNN tests in `test_eligibility_restricted_index.py` (Phase 11) - all executed for real, including the FAISS-vs-ScaNN cross-backend agreement tests, in Docker.

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

No secrets are hardcoded anywhere - V1 has none to hold (no auth, no
external API keys; the synthetic dataset and local model artifacts are
the only "data" the system touches).

## Metrics (offline, synthetic data - see caveat below)

**Scope note**: the table and latency figures below are from the
ORIGINAL, smaller synthetic-V1 pipeline (`scripts/run_pipeline.py`,
`models/two_tower`/`models/ranker`, non-temporal leave-one-out
protocol) - kept as historical Phase 10/11 evidence, not re-verified
against the current default SQLite-backed dataset. For the CURRENT
`data/sqlite/backend_shaped_synthetic.db` dataset, temporal
future-purchase metrics for the current RECENCY+PRICE configuration
(`models/sqlite_baseline/`, Phases 3/4/6) are in
`docs/data-mapping.md` §17 (e.g. test Recall@20: 0.088 → 0.402, test
NDCG@20: 0.048 → 0.299 vs. the ablation experiment's BASE condition) and
are also available live via `GET /v1/metrics/offline` (see "Offline
metrics architecture" below) or by re-running
`python scripts/generate_offline_report.py`.

From the most recent full pipeline evaluation (`scripts/run_pipeline.py`,
162 held-out leave-one-out eval users, real trained artifacts, SAME
Two-Tower/ranker artifacts before and after Phase 11 - neither was
retrained for the architecture change):

| | Test NDCG@10 | Test Recall@10 | Test MRR | Mean distinct categories | Catalog coverage | Fill rate |
|---|---|---|---|---|---|---|
| **Before** (Phase 10, eligibility applied last) - ranker only | 0.3498 | 0.7037 | 0.2597 | 4.59 | 0.92 | 1.00 |
| **Before** (Phase 10) - full pipeline | 0.3356 | 0.6605 | 0.2361 | 6.20 | 0.88 | 1.00 |
| **After** (Phase 11, hard pre-retrieval gate) - ranker only | 0.3502 | 0.6975 | 0.2613 | 4.72 | 0.88 | 1.00 |
| **After** (Phase 11) - full pipeline | 0.3333 | 0.6481 | 0.2365 | 6.19 | 0.88 | 1.00 |

**Do not over-interpret these deltas** - the catalog has only ~50
synthetic products (2 of them the deliberately inactive/out-of-stock
ones exercised by the eligibility tests), so a handful of eval users'
recommendations shifting by one rank position moves these metrics by
hundredths. The one delta that IS a direct, expected consequence of the
architecture change, not noise: **"ranker only" catalog coverage drops
from 0.92 to 0.88**, becoming identical to the full-pipeline figure -
before Phase 11, the "ranker only" (pre-re-rank/eligibility) slice could
still include the 2 inactive/out-of-stock products (only excluded at the
very end), so they could count toward coverage; after Phase 11 they're
excluded before the ranker ever sees them, so "ranker only" and "full
pipeline" coverage are now the same by construction - proof the hard
pre-retrieval gate actually gates retrieval, not just the final list.
Fill rate stays exactly 1.00 before and after (enough eligible products
exist at this catalog scale to fill every request); diversity re-ranking
still delivers the same +~35% mean-distinct-categories lift over the
ranker-only baseline it did before (Phase 7's original finding,
unaffected by Phase 11 - re-ranking itself wasn't touched).

(Two-Tower-retrieval-only Recall@K/HitRate@K figures from Phase 4's own
leave-one-out evaluation are in that phase's training report output,
not reproduced here to avoid restating a number not re-verified in this
session - re-run `scripts/train_two_tower.py` for a fresh one.)

**Latency** (Windows/FAISS, single machine, no load - see
`docs/production-readiness.md` for what this does and doesn't prove):

| | Before (Phase 10) | After (Phase 11) |
|---|---|---|
| FAISS retrieval (single query) | ~0.9ms | ~0.77ms |
| End-to-end pipeline (mean / p95) | ~295-300ms / ~330ms | ~230ms / ~238ms |

Raw FAISS retrieval latency is unaffected by design - `VectorIndex.search`
itself is unchanged; `EligibilityRestrictedIndex` only wraps it inside
the serving pipeline, and the small pool-size reduction (50 -> 48
candidates, since pool sizing is now capped by the *eligible* catalog
count) is not enough to explain a measurable difference on its own. The
end-to-end figure looking faster after Phase 11 is most plausibly ordinary
single-machine run-to-run variance (different session, different
background load) rather than a real effect of this change - re-running
either figure independently on this same machine has historically shown
some spread; take both numbers as sanity checks ("still fast, still
dominated by Keras `.predict()` overhead at this tiny batch scale, not by
the extra eligibility bookkeeping"), not a precise A/B benchmark.

**This is not evidence of real-world recommendation quality.** All
numbers above come from a synthetic, persona-correlated dataset with no
real user behavior - they demonstrate that the pipeline is implemented
correctly and that each stage (ranker, re-ranking, eligibility) measurably
does what it's supposed to relative to the stage before it, nothing more.
See `docs/data-mapping.md` §8.

### Offline metrics architecture (current — Phase 8)

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

## Known limitations & V1 scope

Summarized here; full rationale for each in `docs/data-mapping.md` and
`docs/production-readiness.md`:

- **Synthetic data only** - see the Metrics caveat above.
- **No timestamps on most engagement signals - legacy synthetic-V1 path
  only** ⇒ leave-one-out with a content-based leakage heuristic instead
  of a genuine temporal split (`docs/data-mapping.md` §12). The current
  default `User_events`/SQLite-sourced path DOES carry real timestamps
  and has a genuine per-user temporal future-PURCHASE evaluation
  protocol built and applied against it - see "Current POC status"
  above and `docs/data-mapping.md` §§8.1/14/16.
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

Each phase below is a separate commit, reviewed and approved before the
next began. The description is each phase's CURRENT implementation, not
just its original scope - later mentor-driven work extended several of
these phases after the initial 10 were complete (historically tracked
with its own "STEP" numbering, STEP 5-9); that work is folded into the
phase it improved below rather than presented as additional stages. See
`docs/data-mapping.md` §13 for the full section-to-phase map, including
exactly which STEP absorbed into which phase.

1. **Foundation & architecture.** Project scaffolding, configuration
   loading, logging.
2. **Canonical data layer & dataset integration.** The `AdapterBundle`
   interface (eight ABCs) and canonical schemas; the original small
   synthetic generator (~50 products/300 users, kept for backward
   compatibility, `paths.data_source: "synthetic"`); and, as the current
   default, the backend-ERD-shaped SQLite dataset
   (`data/sqlite/backend_shaped_synthetic.db` - 1,200 products, 1,000
   users) with the confirmed `User_events` activity-log contract and
   five engagement signals (CLICK/SEARCH/ADD_TO_CART/PURCHASE/CHATBOT,
   up from the original four).
3. **Feature engineering & semantic product embeddings.** Sentence
   Transformer product embeddings; category/brand affinity; and, folded
   in by later work, recency (time-decay) weighting of behavioral
   signals and price-aware user/product features (effective price,
   price tiers, category-relative price, a user price profile) -
   `docs/data-mapping.md` §§14-15.
4. **Neural Two-Tower retrieval model.** 128-D, L2-normalized user/item
   embeddings. Current numeric encoder dimensions are **9 item-numeric /
   9 user-numeric** (extended from 7/8 by the price-aware feature work
   above); the 7/8-dimensional encoder is retained only as the BASE
   condition of the controlled ablation experiment below.
5. ANN retrieval: ScaNN (primary/production backend, Linux/Docker) +
   FAISS (native-Windows dev fallback), both exact search over the same
   embeddings.
6. **Neural ranking** of VectorIndex candidates, richer than retrieval
   features, evaluated (NDCG/Precision/Recall/HitRate/MRR) against a
   raw-retrieval-score baseline. Current ranker uses **29 explicit
   features** (extended from 23 by the price-aware feature work); the
   23-feature vector is retained only as the BASE condition of the same
   ablation experiment - `docs/data-mapping.md` §17.
7. **Full serving pipeline:** three-level cold-start blending
   (strong/sparse/no-history, now sized against five engagement
   signals), dedup + category/brand diversity re-ranking, business
   rules/eligibility (originally applied last, now a hard pre-retrieval
   gate - see Phase 11). This is also the pipeline the temporal
   future-purchase evaluation protocol runs point-in-time, per-user
   cutoff, for its primary offline metrics - `docs/data-mapping.md`
   §§8.1, 16.
8. **Versioned (`/v1`) FastAPI recommendation API** - dependency-injected,
   model artifacts loaded once at startup, thin wrapper over the Phase 7
   pipeline (no duplicated logic). Current implementation also serves
   persisted temporal offline metrics (`GET /v1/metrics/offline`) by
   reading a batch-generated, provenance-validated report rather than
   evaluating inside the request - `docs/data-mapping.md` §18.1.
9. **Internal Streamlit dashboard** for demonstrating/debugging the
   recommendation engine - user signals, cold-start tier, candidate-pool/
   eligibility diagnostics, offline metrics. Current architecture is a
   **pure HTTP client** of the FastAPI service
   (`ui.api_client.RecommendationApiClient`); it originally reused the
   API's `RecommendationService` in-process - `docs/data-mapping.md`
   §18.
10. Production hardening: multi-stage Docker (test/api/dashboard),
    startup artifact validation, env-var config overrides, structured
    observability, a full reliability pass, Windows + Docker/Linux
    integration verification, and this documentation - see
    `docs/production-readiness.md` for the critical review (a Phase 10
    snapshot, annotated where later work superseded it).
11. Mentor-reviewed architecture change: hard PRE-retrieval eligibility
    (isActive/stockQuantity gate candidate generation itself, via
    `retrieval.index.eligibility_filter.EligibilityRestrictedIndex` for
    the VectorIndex path) plus a final lightweight eligibility
    re-validation as a defense-in-depth safety net - no Two-Tower/ranker
    retraining or VectorIndex rebuild required when catalog/stock state
    changes.
