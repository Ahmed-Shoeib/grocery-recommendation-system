# Production-Readiness Review (Phase 10)

A critical self-review of the current system, written at the end of the
10-phase V1 build. Findings are reported, not silently fixed - genuine
bugs found *during* Phase 10's own reliability/integration work were
fixed as part of that work (see `docs/data-mapping.md` and commit
history for specifics, e.g. the ScaNN/TensorFlow ABI pin below); this
document is the place larger architectural trade-offs are surfaced for
a human decision, not quietly resolved.

> **Snapshot notice**: this document is a point-in-time review written
> at the end of Phase 10 and is NOT re-verified against the later
> mentor-driven work that extended Phases 2, 3, 4, 6, 7, 8, and 9 since
> (recency weighting and price-aware features in Phase 3, the current
> 9/9-dimensional Two-Tower in Phase 4, the 29-feature ranker in Phase 6,
> the backend-shaped SQLite data source and five-signal `User_events`
> contract in Phase 2, the temporal future-purchase evaluation protocol
> running through Phase 7, the persisted offline-metrics endpoint in
> Phase 8, and the pure-HTTP-client Streamlit rewiring in Phase 9 - see
> `docs/data-mapping.md` sections 14-18.1; that later work was
> historically tracked as STEP 5-9, a labeling since superseded by the
> phase-numbered roadmap). Several items below were resolved by that
> later work; each such item is annotated inline. Where this document
> and the current code disagree, the code is authoritative - see
> `README.md`'s "Current POC status" section for an up-to-date summary.

Every finding is classified as one of:

- **Ready now** - solid as-is for what V1 is (a synthetic-data
  demonstration/evaluation system).
- **Acceptable V1 limitation** - a real constraint, deliberately not
  solved yet, that does not block using the system for its actual V1
  purpose.
- **Must address before real production deployment** - would need
  resolving before this serves real users/real data/real traffic.
- **Future optimization** - not a defect, a scaling/efficiency lever
  available later, at the point it starts to matter.

## Ready now

- **Core ML pipeline** (hard pre-retrieval eligibility → Two-Tower →
  VectorIndex → Neural Ranker → Re-ranking → remaining business rules →
  final lightweight eligibility validation → Final Top-N) is functionally
  correct, leakage-safe (docs/data-mapping.md §12), and extensively
  tested - see the README's Testing section for exact current pass/skip
  counts (updated for Phase 11's added eligibility/retrieval coverage).
- **ScaNN as the production retrieval backend** is verified working end
  to end in Docker (real container, real mounted Phase 4/6 artifacts,
  `retrieval_backend=scann` confirmed in the service-startup log) and
  produces results numerically identical to the FAISS dev fallback for
  the same request (cross-backend agreement, both being exact search).
- **FastAPI service**: versioned (`/v1`), dependency-injected
  (`RecommendationService` built once, not per-request), validates
  input (Top-N bounds, path param types), returns structured errors
  with one consistent contract across every failure path (including
  FastAPI's own automatic validation errors), distinguishes "unknown
  user" (404) from "known user, no history" (200 + fallback
  recommendations), and never leaks a raw traceback.
- **Streamlit dashboard** (Phase 9 - its architecture evolved since this
  review, `docs/data-mapping.md` §18): at the time of this review, shared
  the exact same `RecommendationService` and pipeline call as the API
  in-process (no duplicated recommendation logic), handled service-load
  and pipeline-call failures gracefully (both paths tested via
  `AppTest`), verified against real trained artifacts for genuine
  STRONG/SPARSE/NO_HISTORY users. In Phase 9's current implementation,
  the dashboard no longer constructs a `RecommendationService` at all -
  it is a pure `ui.api_client.RecommendationApiClient` HTTP client of the
  API, tested and live-verified end-to-end for the same three cold-start
  tiers (see §18's "Live integration verification").
- **Startup artifact validation**: missing, corrupt, or
  dimensionally/schema-incompatible Two-Tower or ranker artifacts fail
  loudly and fast (checked before any dataset/embedding work, not
  after) - the system never silently serves recommendations built from
  a mismatched model combination.
- **No hardcoded secrets**: there are none to hardcode in V1 (no
  auth, no external credentials, no third-party API keys) - the
  synthetic dataset and local model artifacts are the only "data" the
  system touches.
- **Reliability**: verified graceful handling of missing/corrupt
  artifacts, empty candidate pools structurally impossible (validated
  at startup), unknown users, insufficient recommendations (honest
  `fill_rate`), a VectorIndex/ranker/catalog-drift edge case (a
  candidate id absent from the live catalog is skipped and logged, not
  a crash), unavailable products (Phase 11: never enter retrieval/ranking
  in the first place, via a hard pre-retrieval eligibility gate, PLUS a
  final lightweight re-validation as a defense-in-depth safety net - not
  a single filter-after-the-fact step), duplicate ids (deduped
  unconditionally in re-ranking), and malformed API requests (structured
  422s).
- **Observability**: every recommendation call logs cold-start tier,
  requested/returned counts, fill rate, candidate pool size, pre-
  retrieval exclusion count, and final-validation exclusion count from
  one place (`serving.pipeline.generate_recommendations`, shared by API
  and dashboard - not duplicated per caller); the API additionally logs
  per-request latency and structured request/error middleware logs. No
  CTR/conversion metrics are invented - V1 has no event-tracking
  pipeline to compute those from (see "Acceptable V1 limitation" below).

## Acceptable V1 limitation

- **All data is synthetic.** Offline metrics (NDCG, Recall, MRR,
  diversity, coverage, fill rate) demonstrate pipeline correctness and
  relative comparisons between components, not real-world
  recommendation quality - stated explicitly everywhere these numbers
  are reported (docs/data-mapping.md §8, every phase's evaluation
  script output).
- **No timestamps on search/chatbot/most engagement signals** ⇒ no
  genuine temporal train/test split; leave-one-out with a content-based
  leakage heuristic is used instead (docs/data-mapping.md §12) - a
  known, documented approximation, not a bug. **Scope note**: this
  applies to the original synthetic V1 path only. The `User_events`/
  SQLite-sourced path (`paths.data_source: "sqlite"`, the current
  default) DOES carry real timestamps and has a genuine temporal
  future-purchase evaluation protocol built and applied against it
  (docs/data-mapping.md §§8.1, 14, 16) - see this document's snapshot
  notice at the top.
- **No event-tracking pipeline** ⇒ no CTR, impressions, or conversion
  metrics exist or are computed anywhere in this system.
- **Search and chatbot context are synthetic-only adapters** - no real
  backend table exists for either yet (docs/data-mapping.md §4); the
  adapter interface is real and ready for a real implementation.
- **`preferredCategory`/`ageGroup`** are confirmed backend additions
  that don't exist in production yet - modeled as `Optional[str]`
  throughout so the system degrades to popularity-based signals for
  users without them, rather than assuming they're always populated.
- **~5.35GB Docker image** for the api/dashboard services - large
  because the full ML stack (PyTorch CPU, TensorFlow, Sentence
  Transformers, ScaNN, Streamlit) is bundled in one image. Reasonable
  for an internal V1 system; not a lean microservice image.
- ~~**API and dashboard each load their own full copy of the
  models**~~ - **resolved in Phase 9's current implementation**
  (`docs/data-mapping.md` §18): the dashboard is now a pure HTTP client
  of the API (`ui.api_client.RecommendationApiClient`); it no longer
  loads any model artifact, and `dashboard.api_base_url` is now actively
  used, not reserved/unused. Kept here, struck through, as a record of
  the Phase-10-era limitation this later work fixed.
- **The Sentence Transformer model is downloaded fresh from the
  Hugging Face Hub on every cold container start** (not baked into the
  image or cached on a persistent volume) - adds startup latency and a
  network dependency each time a fresh container starts.
- **`build_recommendation_service` regenerates the full dataset and
  re-runs the Phase 3 feature pipeline at every process startup**
  (product embeddings are cached to disk and reused; everything else is
  recomputed) - fine at V1's original 50-product/300-user synthetic
  scale (~50s cold start including the Hub download, mostly the
  Sentence Transformer). In Phase 2's current implementation,
  `paths.data_source: "sqlite"` is the default, regenerating features
  for the larger backend-shaped SQLite catalog (1,200 products/1,000
  users, `data/sqlite/backend_shaped_synthetic.db`) instead of the
  original synthetic generator at every startup - not re-measured in
  this review; would not scale as-is to a large real catalog either way.

## Must address before real production deployment

- **The FastAPI recommendation route is `async def` but calls fully
  synchronous, blocking code directly** (Keras `.predict()`, Sentence
  Transformer encoding) - this blocks the event loop for the duration
  of each request (currently ~250-300ms), so concurrent requests are
  effectively serialized rather than handled in parallel. The fix is
  well-understood (a plain `def` route, which FastAPI runs in a
  threadpool automatically, or explicit `run_in_threadpool`/a worker
  process pool) but changes the request-handling execution model, so
  it's reported here rather than changed silently during this review.
- **No authentication or authorization** on the API - anyone who can
  reach the port can request any user's recommendations. Acceptable for
  an internal synthetic-data tool; not for real user data.
- **No rate limiting.**
- **No TLS** - plain HTTP; a real deployment needs a reverse proxy/
  ingress terminating TLS in front of this service.
- **Real backend adapters don't exist yet** - only the synthetic
  in-memory adapters (`data.adapters.factory.build_synthetic_adapters`)
  are implemented. The `AdapterBundle` interface
  (`data.adapters.base`) is designed for this swap (see "How the real
  backend will replace synthetic adapters" in the README), but writing
  and validating the real SQL/API-backed adapters is real, scoped work
  that hasn't happened yet.
- **No artifact versioning/rollback story in deployment** - `models/`
  is a single directory snapshot; there's no registry, no canary/A-B
  serving path, no automated rollback if a newly trained model is
  worse.
- **No metrics exporter wired up** - structured logs exist and are
  designed to translate directly into Prometheus counters/histograms
  (tier, fill_rate, pool_size, eligibility exclusions, latency are all
  already discrete fields on every log line), but nothing currently
  scrapes/exports them.
- **No persistent Hugging Face model cache volume** in the Docker
  setup - combined with the "must address" async-blocking issue above,
  repeated cold starts under real deployment churn (rolling
  deploys, autoscaling) would be slower and more network-dependent than
  necessary.

## Future optimization

- Split the single image into leaner per-purpose images (e.g. an API
  image without Streamlit). ~~Consider moving the dashboard to call the
  API over HTTP~~ - **done, in Phase 9's current implementation**
  (`docs/data-mapping.md` §18): the dashboard is now a pure HTTP client,
  so it no longer double-loads the model stack; a leaner Streamlit-only
  image (without the ML stack) is still a live opportunity now that the
  dashboard process needs none of it.
- Precompute/cache more of the startup path (not just product
  embeddings) so serving doesn't regenerate the dataset/feature
  pipeline on every process start.
- Approximate ANN variants (FAISS IVF/HNSW, ScaNN tree+AH) once the
  catalog is large enough that exact brute-force search stops being
  effectively free - both backends are already structured so this is a
  change inside one class, not an interface change.
- Add real Prometheus/OpenTelemetry exporters on top of the existing
  structured logging.
- Load testing at realistic catalog size and request concurrency, after
  the async-blocking fix above.
- ~~A genuinely temporal evaluation protocol once real backend
  timestamps exist (V2)~~ - **implemented, folded into Phases 3, 6, and
  7** (docs/data-mapping.md §§8.1, 14, 16) for the `User_events`/SQLite-
  sourced path: `evaluation.temporal_future_purchase` builds real
  per-user cutoffs and future-PURCHASE targets from genuine
  `action_time` values, and `models/sqlite_baseline/` is trained and
  evaluated against exactly this protocol. The ORIGINAL synthetic V1
  path (`models/two_tower`/`models/ranker`, kept for backward
  compatibility) still has no timestamps and still uses the older
  content-based leakage heuristic (§12) - this item is resolved only
  for the current default (`paths.data_source: "sqlite"`) path, not for
  the legacy synthetic one.

## What was actually fixed during Phase 10 (not just reported)

These were concrete bugs/gaps found while doing Phase 10's reliability
and integration-verification work, fixed as part of that work (in scope
per the phase's own "verify graceful handling of..." requirement,
distinct from the architectural trade-offs above):

- **ScaNN/TensorFlow ABI incompatibility**: installing the general `ml`
  extra (unpinned TensorFlow, resolves to 2.21.0) together with
  `retrieval-scann` in the same environment broke `import scann`
  (`undefined symbol: ...absl...internal_log_function...`) - never
  caught before because Phase 5's ScaNN-only verification image never
  installed TensorFlow at all. Fixed by pinning `tensorflow~=2.20.0`
  (matching scann's own declared PyPI compatibility) specifically in
  the Docker build, without touching the wider Windows-facing
  `pyproject.toml` range. See docs/data-mapping.md §10.
- **VectorIndex/catalog drift crash**: a candidate id returned by
  `VectorIndex.search` but absent from the current `product_features`
  dict raised an unhandled `KeyError`, failing the whole request.
  Fixed to skip and log the mismatched candidate instead.
- **Dashboard pipeline-call crash**: `run_recommendations` inside
  `dashboard.py` wasn't wrapped in error handling (only service
  *loading* was) - any pipeline-stage failure crashed the whole page.
  Fixed with the same graceful-error pattern used for service loading.
- **Inconsistent API error-response shape**: FastAPI's automatic query
  validation (e.g. `limit=0`) returned a different JSON body shape than
  the manually-raised `HTTPException` path (e.g. `limit` too large).
  Fixed with a `RequestValidationError` handler using the same
  `ErrorResponse` contract.
- **`config.log_level` was declared but never applied** - every
  script/service called `setup_logging()` with no arguments. Fixed by
  loading config before calling `setup_logging(config.log_level)`
  everywhere, and adding the missing call to the dashboard entirely.
- **Artifact-missing failures were expensive to discover** -
  `build_recommendation_service` used to generate the full synthetic
  dataset and load the Sentence Transformer *before* checking whether
  trained artifacts existed at all. Reordered so a missing-artifacts
  failure is near-instant (verified: well under the 5-second bound of a
  real Sentence Transformer load).
