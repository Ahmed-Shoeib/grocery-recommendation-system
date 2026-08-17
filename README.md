# Grocery Recommendation System

A production-oriented, modular personalized recommendation service for a
grocery e-commerce backend: Two-Tower retrieval, an approximate-nearest-
neighbor `VectorIndex` (ScaNN in production/Docker, FAISS on native
Windows dev), a neural ranker, cold-start-aware re-ranking, and
business-rules/eligibility filtering, served through a versioned FastAPI
service and inspectable through an internal Streamlit dashboard.

Currently trained/served against a small **synthetic** dataset (~50
products, ~300 users) - architected from the start so the real grocery
backend can be substituted with no model redesign once it's available
(see [How the real backend will replace synthetic adapters](#how-the-real-backend-will-replace-synthetic-adapters)).
See `docs/data-mapping.md` for the full ERD reconciliation, V1/V2 scope
boundary, and every design decision's rationale, and
`docs/production-readiness.md` for a critical, classified review of
what is and isn't ready for real production use.

## Status

All 10 planned phases are complete: project scaffolding; canonical data
adapters + synthetic dataset; the Sentence Transformer + feature-
engineering pipeline; a trained Two-Tower retrieval model (128-D,
L2-normalized, leave-one-out evaluated); a ScaNN/FAISS `VectorIndex`
over its embeddings; a neural ranker that re-scores VectorIndex
candidates with richer cross features (beats the raw-retrieval-score
baseline on every ranking metric); the full V1 serving pipeline
(three-level cold-start blending, dedup/diversity re-ranking, hard
pre-retrieval eligibility + a final lightweight eligibility validation -
see Phase 11 below); a versioned FastAPI recommendation API (`/v1`); an
internal Streamlit dashboard; and Phase 10 production hardening -
containerization, startup artifact validation, environment-variable
config overrides, structured observability, and a full
integration/reliability pass, documented in `docs/production-readiness.md`.

**Phase 11** (mentor-reviewed architecture change, post-Phase-10):
`isActive`/`stockQuantity` moved from a final-only filter to a hard
PRE-retrieval eligibility gate - inactive/out-of-stock products never
become Two-Tower/VectorIndex retrieval candidates in the first place,
with the final eligibility check kept as a lightweight defense-in-depth
safety net, not the primary mechanism. Zero Two-Tower/ranker retraining
and zero VectorIndex rebuild required - see the architecture diagram and
`docs/data-mapping.md` §5.

**What's next** (deliberately out of scope for this project's own phase
plan): a separate, controlled benchmarking/experimentation pass, and
further optimization once a real dataset is available - see
`docs/production-readiness.md`'s "Future optimization" section.

## Architecture

```
Backend ERD (Order/OrderItem, Cart/CartItem, User, Review) + Synthetic (Search, Chatbot)
                              |
                     Data Adapter Layer
                              |
                 Canonical Engagement Model
                              |
                      Feature Engineering
                 +------------+------------+
            User Features              Product Features
                                             |
                                  Sentence Transformer (384-D, frozen)
                                             |
                              +--------------+--------------+
                          User Tower                   Item Tower
                        (128-D, L2-norm)             (128-D, L2-norm)
                              +--------------+--------------+
                                     cosine compatibility
                                             |
                     Hard PRE-RETRIEVAL eligibility (isActive, stock - catalog state)
                                             |
              VectorIndex (ScaNN primary/Docker, FAISS Windows dev fallback)
                       - restricted to eligible products only
                                             |
                                   Oversized candidate pool
                                    (capped by eligible count)
                                             |
                                    Neural Ranker (MLP)
                                             |
                                   Re-ranking + cold-start blend
                                             |
                              Remaining business rules
                                             |
                   Final lightweight isActive/stock validation (safety net)
                                             |
                                        Final Top-N
                                    +--------+--------+
                              Recommendation API   Streamlit Dashboard
```

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
docs/
  erd.jpeg                  Source-of-truth backend ERD
  data-mapping.md            ERD reconciliation, V1/V2 scope, every design decision's rationale
  production-readiness.md    Phase 10 critical review (Ready now / Acceptable limitation / Must address / Future)
models/                     Serialized model artifacts (gitignored - regenerable via scripts/train_*.py)
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
  evaluation/                  Offline metrics + latency measurement
  serving/                    Cold-start tiering, fallback candidates, two-stage eligibility (hard pre-retrieval gate + final lightweight validation), startup artifact validation, the full pipeline orchestrator
  api/                         FastAPI app (v1) - app/routes/schemas/dependencies, thin wrapper over serving.pipeline
  ui/                           Streamlit dashboard (dashboard.py rendering-only; data_access.py/metrics.py pure + unit-tested) - reuses api.dependencies.RecommendationService in-process
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
python scripts/run_pipeline.py   # full pipeline eval report + qualitative examples + latency, using existing artifacts
python scripts/run_api.py        # FastAPI service (loads artifacts once at startup, never trains)
python scripts/run_dashboard.py  # Streamlit dashboard (same artifacts, in-process, no API dependency)
```

### API usage

```
GET /v1/health                                    liveness
GET /v1/ready                                      readiness (catalog/Two-Tower/ranker/VectorIndex all loaded)
GET /v1/users/{user_id}/recommendations?limit=10   Top-N recommendations
```

Response: `product_id`, `rank`, `score`, `source` per item (never
catalog data like name/price - resolve that from the product catalog
separately) plus `meta` (tier, requested/returned counts, fill_rate,
pool_size, eligibility exclusions, api/model version, latency_ms -
pipeline latency only, not full HTTP round-trip). Unknown users get a
structured `404`; invalid Top-N gets a structured `422`; unexpected
failures get a structured `500` - one consistent `{error, message}`
body shape across every failure path.

### Dashboard usage

Select a user from the full user table (shows `preferredCategory`/
`ageGroup` when present) to see: the five V1 engagement signals with
explicit empty states; cold-start tier and category/brand affinity;
final recommendations with catalog info joined in for display only;
pipeline diagnostics (candidate pool size, eligibility exclusions with
reasons, source breakdown, category distribution, this request's
latency); and a button-triggered offline-metrics section (a full
leave-one-out evaluation pass isn't free, so it's on-demand, not
automatic) explicitly labeled as offline/synthetic, never a production
metric.

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

## Known limitations & V1 scope

Summarized here; full rationale for each in `docs/data-mapping.md` and
`docs/production-readiness.md`:

- **Synthetic data only** - see the Metrics caveat above.
- **No timestamps on most engagement signals** ⇒ leave-one-out with a
  content-based leakage heuristic instead of a genuine temporal split
  (`docs/data-mapping.md` §12).
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

1. Foundation & architecture.
2. Canonical data layer & synthetic grocery dataset.
3. Feature engineering & semantic product embeddings.
4. Neural Two-Tower retrieval model.
5. ANN retrieval: ScaNN (primary/production backend, Linux/Docker) + FAISS (native-Windows dev fallback).
6. Neural ranking of VectorIndex candidates, richer than retrieval features, evaluated (NDCG/Precision/Recall/HitRate/MRR) against a raw-retrieval-score baseline.
7. Full serving pipeline: three-level cold-start blending (strong/sparse/no-history), dedup + category/brand diversity re-ranking, business rules/eligibility (applied last).
8. Versioned (`/v1`) FastAPI recommendation API - dependency-injected, model artifacts loaded once at startup, thin wrapper over the Phase 7 pipeline (no duplicated logic).
9. Internal Streamlit dashboard for demonstrating/debugging the recommendation engine - user signals, cold-start tier, candidate-pool/eligibility diagnostics, offline metrics - reusing the API's RecommendationService in-process.
10. Production hardening: multi-stage Docker (test/api/dashboard), startup artifact validation, env-var config overrides, structured observability, a full reliability pass, Windows + Docker/Linux integration verification, and this documentation - see `docs/production-readiness.md` for the critical review.
11. Mentor-reviewed architecture change: hard PRE-retrieval eligibility (isActive/stockQuantity gate candidate generation itself, via `retrieval.index.eligibility_filter.EligibilityRestrictedIndex` for the VectorIndex path) plus a final lightweight eligibility re-validation as a defense-in-depth safety net - no Two-Tower/ranker retraining or VectorIndex rebuild required when catalog/stock state changes.

Each phase is a separate commit, reviewed and approved before the next
begins. See `docs/data-mapping.md` §13 for the full phase-to-decision map.
