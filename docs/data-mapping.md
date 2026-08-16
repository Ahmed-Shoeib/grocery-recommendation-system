# ERD → Canonical Schema Mapping & V1 Scope Decisions

Source of truth for the backend schema: `docs/erd.jpeg`. This document
reconciles that ERD with the recommendation system's requirements, records
the assumptions V1 makes, and defines the scope boundary between V1 and
deferred V2 work. It is updated as clarifications arrive (last updated
after the Phase 1 clarification round).

## 1. Entities as drawn in the ERD

| Entity | Fields | Recommendation relevance |
|---|---|---|
| User | Id, FirstName, LastName, Email, PhoneNumber, HashedPassword, RefreshToken, Role, CreatedAt, UpdatedAt | Identity + (confirmed, pending) `preferredCategory`, `ageGroup` - see §2. |
| Category | Id, ParentId (self-FK), Name, CreatedAt | Category + parent-category features. |
| Product | Id, CategoryId, Slug, Name, Description, Brand, Price, SalePrice, DiscountPercentage, StockQuantity, Ingredients, isActive, ProductImage, AltText | Core item metadata; text fields feed the Sentence Transformer; isActive/StockQuantity feed the hard pre-retrieval eligibility gate AND the final lightweight eligibility validation (§5). |
| ProductTags / Tag | join table + Name | Tag text feeds the Sentence Transformer input. |
| Cart / Cart_Item | Cart(Id, CartItemId FK, UserId); CartItem(Id, CartId, ProductId, Quantity) | Add-to-cart habit signal (D). |
| Order / Order_Item | Order(Id, UserId, VoucherId, AddressId, IdempotenceKey, TotalAmount, Status, PaymentMethod, CreationDate, DeliveryDate); OrderItem(Id, ProductId, OrderId, Quantity, UnitPrice) | Previous purchases signal (A) - the strongest V1 signal. |
| Review | Id, UserId, ProductId, Rating, Comment, CreationDate | Auxiliary ranking signal (rating/review-count affinity). |
| UserAddress, Voucher | address/voucher fields | Not used by the recommender. |

### Known ERD discrepancy

`Cart.CartItemId` is drawn as an FK on `Cart` pointing at `CartItem`, which
would imply one cart item per cart. The sane relational direction -
`CartItem.CartId -> Cart.Id`, one cart with many items - is what the
`CartAdapter` implements. Flagged for the backend team to confirm; not a
blocker for V1 synthetic data.

### Order.Status

No enum is specified in the ERD. The `PurchaseAdapter` treats "counts as
a purchase" as a **configurable set of statuses** rather than hard-coding
one, so pointing it at the real backend later is a config change.

## 2. `preferredCategory` and `ageGroup` - CONFIRMED backend additions

Per explicit confirmation from the product owner (2026-08-12): these two
fields **are** being added to the real `User` entity by the backend team.
`docs/erd.jpeg` simply predates that change - it is not evidence they
won't exist.

Treatment in this codebase:

- Present in synthetic V1 data for every user.
- Modeled as `Optional[str]` on the canonical `UserProfile`
  (`src/recommendation/data/schemas/user.py`) so any code path that
  consumes them degrades gracefully (falls back to popularity-based
  signals) for users created before the backend migration lands, rather
  than assuming they are always populated.
- `preferredCategory` is used as a first-class personalization and
  cold-start signal (see §3).
- `ageGroup` is treated as an **opaque categorical label** - no hard-coded
  age-based assumptions (e.g. "18-24 likes X") are encoded anywhere. It is
  offered to the ranking model as a feature and its actual predictive
  value is left to be validated experimentally, not assumed.

## 3. Three-level personalization strategy

Cold start is not binary. History strength is computed from event counts
across the four V1 engagement signals (purchases, cart adds, searches,
chatbot-context presence) and thresholded via `configs/base.yaml:
cold_start.*` (implemented in Phase 7):

| Tier | Condition | Strategy |
|---|---|---|
| STRONG_HISTORY | signal count ≥ `strong_history_min_signals` | Two-Tower personalized retrieval, full ranking pipeline. |
| SPARSE_HISTORY | signal count ≥ `sparse_history_min_signals`, below strong threshold | Two-Tower personalized candidates blended with preferred-category and popularity candidates, per `cold_start.sparse_blend` weights. |
| NO_HISTORY | signal count 0 | Deterministic fallback: `preferredCategory` → category popularity → global popularity (`cold_start.no_history_fallback_order`). |

All thresholds and blend weights are config-driven, not hard-coded, and
tiering is deterministic and testable.

**Implemented (Phase 7)**: `serving.cold_start.determine_history_tier`
(tiering), `serving.fallback` (candidate sources + merge strategies -
`blend_candidate_lists` for the SPARSE weighted blend,
`waterfall_candidates` for the NO_HISTORY ordered chain), orchestrated
by `serving.pipeline.generate_recommendations`. `"category_popularity"`
and `"preferred_category"` are DISTINCT fallback sources:
`"preferred_category"` uses the confirmed `UserProfile.preferred_category`
attribute directly; `"category_popularity"` uses the user's single
highest-affinity category from `UserFeatures.category_affinity` (which
factors in `preferred_category` plus whatever sparse purchase/cart/
search/chatbot signal exists). For a true zero-signal user these
necessarily coincide (affinity has nothing else to draw on); for a
SPARSE_HISTORY user with some real signal they can genuinely differ -
this is why both are listed as separate fallback tiers rather than one.

Duplicate removal and category/brand diversity re-ranking
(`reranking.diversity.rerank`) run uniformly across all three tiers'
candidate lists, before eligibility filtering. Diversity is a continuous
score penalty (`reranking.diversity_strength`), not a hard per-category/
brand quota - a candidate can only be reordered by it, never dropped, so
it cannot "over-diversify" a list into irrelevance; at
`diversity_strength=0` it reproduces the input ranking exactly. Measured
on the real V1 synthetic catalog (`scripts/run_pipeline.py`, default
`diversity_strength=0.5`): mean distinct categories per 10-item list rose
from 4.59 (ranker-only order) to 6.20 (post-re-ranking) - a ~35% increase
- while NDCG@10 moved from 0.3498 to 0.3356 on the test split (0.3608 to
0.3523 on validation), i.e. a small, deliberate relevance cost for a
real diversity gain, not a catastrophic one.

**Phase 8 addition - unknown user vs. NO_HISTORY user**: the API draws a
deliberate distinction the pipeline itself doesn't need to. A NO_HISTORY
tier user (`serving.cold_start`) is a real, known user (a `UserProfile`
exists) who simply has zero engagement signals - a normal `200` response
using the fallback chain above. An **unknown user** (no `UserProfile` at
all - `api.errors.UnknownUserError`, checked in
`api.dependencies.RecommendationService.recommend` before the pipeline
ever runs) is a different failure mode entirely and returns `404` with a
structured error body. Conflating the two would hide a real "this user
doesn't exist" condition (e.g. a stale/mistyped id) behind what looks
like a normal cold-start recommendation.

## 4. Search and chatbot context - adapters, not backend tables

Neither `SearchHistory` nor any chatbot entity exists in the ERD. V1 uses:

- `SearchAdapter` (interface) / `SyntheticSearchAdapter` (V1 implementation)
- `ChatbotContextAdapter` (interface) / `SyntheticChatbotAdapter` (V1 implementation)

Both produce the canonical `SearchRecord` / `ChatbotContextRecord` schemas
(`src/recommendation/data/schemas/engagement.py`). The chatbot record
supports summary text, mentioned product IDs, a preferred category, and
keywords/intent - whichever subset a future real backend (API, DB entity,
event summary) provides, the adapter maps it into this same shape. If
summary text is present, it may be encoded with the same Sentence
Transformer used for product text (Phase 3+).

No permanent backend table is assumed or invented for either signal.

## 5. Eligibility / business rules - hard PRE-retrieval gate + final validation

**Revised 2026-08-16** (mentor-reviewed architecture change, supersedes
the 2026-08-12 "filter last" revision below): `isActive`/`stockQuantity`
are HARD, global catalog-eligibility facts, not serving-time noise to
defer - they now gate candidate generation itself, before retrieval:

```
Catalog -> HARD PRE-RETRIEVAL ELIGIBILITY (isActive, stockQuantity)
        -> Two-Tower / VectorIndex retrieval (eligible products only)
        -> Neural Ranker -> Re-ranking -> remaining business rules
        -> FINAL LIGHTWEIGHT isActive/stockQuantity VALIDATION
        -> Final Top-N
```

V1 still uses exactly the fields the ERD actually has: `Product.isActive`
and `Product.stockQuantity`. No `isDeleted` or similar field is invented.
Both are carried through as plain structured `ProductFeatures` fields
(Phase 3), which is what makes this a config/query-time change, not a
data-model change - no candidate source (personalized Two-Tower/
VectorIndex retrieval OR the SPARSE/NO_HISTORY category-/global-
popularity fallbacks) is allowed to surface an ineligible product any
more, at any point.

**Why the reordering (2026-08-16)**: the original 2026-08-12 rationale
(below) was explicitly scoped to "at the current catalog scale, the
wasted-ranking-compute argument for an early filter is negligible" -
correct as far as it went, but it missed that letting the ranker/
re-ranker operate on a pool that can include ineligible items lets those
items occupy pool/ranking slots that could otherwise go to eligible
products, an architectural correctness concern independent of catalog
scale or ranking-compute cost. Moving `isActive`/`stockQuantity` to a
hard pre-retrieval gate removes that risk entirely: the ranker and
re-ranker only ever see products a customer could actually buy right
now. This does NOT reintroduce coupling between serving-time state and
the ML stages the 2026-08-12 revision was protecting: retrieval/ranking
still never store or reason about stock/active state themselves - they
simply receive a smaller, pre-restricted candidate universe from the
caller. Concretely:

- The Two-Tower model, its 128-D item embeddings, the ranker, and each
  backend's built ANN index structure are completely untouched by
  eligibility changes - none of them are retrained or rebuilt when stock
  or `isActive` changes. `retrieval.index.eligibility_filter
  .EligibilityRestrictedIndex` wraps the (already-built, unmodified)
  `VectorIndex` and restricts *query results* to a caller-supplied
  eligible-id set - see §10 for why this is the right abstraction for
  both ScaNN and FAISS specifically.
- The ONLY thing that needs to reflect current inventory state is
  `product_features` (a plain `dict[int, ProductFeatures]`, cheaply
  re-derived from current catalog rows via `features.product_features
  .build_product_features`) - not a model artifact, not an index file.
  In a real backend deployment, this is exactly the kind of per-request
  (or short-TTL-cached) catalog read a serving layer already needs to do
  for other reasons; V1's synthetic service builds it once at process
  startup (see `api.dependencies.build_recommendation_service`), a known
  V1 characteristic of the synthetic demo, not a Phase 11 limitation.
- Retrieval/ranking/re-ranking still keep operating on a `pool_size`
  candidate pool derived from `retrieval.candidate_pool_size` -
  unchanged Phase 5 sizing logic - except the pool is now capped by the
  **eligible** catalog size rather than the full catalog size, since
  requesting more eligible candidates than exist is meaningless.

Other, user-specific/list-specific business rules (a future regional
restriction, a purchase-eligibility rule tied to the requesting user,
anything not a hard *global* catalog fact) stay at the final stage only
- pre-retrieval is reserved specifically for `isActive`/`stockQuantity`-
style hard, global eligibility, not a general early-filter mechanism.

**A final, lightweight eligibility validation still runs**, immediately
before the Top-N is returned, re-checking the identical rules
(`serving.eligibility.build_eligibility_rules`/`apply_eligibility`, the
SAME policy object used pre-retrieval) against the full re-ranked pool.
This is deliberately a safety net, not the primary filtering mechanism:
defense-in-depth against a product becoming unavailable between
pre-retrieval filtering and the final response (a real possibility in a
live system where retrieval and final-response assembly can observe
catalog state at different points in time -
`serving.pipeline.generate_recommendations`'s `final_product_features`
parameter lets a caller pass a fresher read for exactly this check). In
the common case, with pre-retrieval filtering already having excluded
every ineligible product, this stage excludes nothing -
`RecommendationResult.num_excluded_by_eligibility` is expected to be 0
in steady state; `num_excluded_pre_retrieval` is the new field reporting
how many catalog products the hard gate excluded before candidate
generation ran.

Filtering (both stages) still runs over the FULL candidate/re-ranked
pool (config-driven size, Phase 5's `candidate_pool_size`), not just the
requested Top-N, specifically so that a rare final-stage exclusion still
leaves enough eligible candidates to fill the request -
`serving.pipeline.RecommendationResult.fill_rate` reports how close the
final list came to the requested count (1.0 in practice at the current
V1 catalog scale). If fewer eligible products exist catalog-wide than
requested, returning fewer than N is the correct, honest behavior, not a
bug - `fill_rate` reports that accurately.

**Implemented**: `serving.eligibility.build_eligibility_rules` (a list of
named `EligibilityRule(name, predicate)` pairs, appending a rule requires
no change elsewhere) + `apply_eligibility`, called twice from
`serving.pipeline.generate_recommendations` - once pre-retrieval (over
the full catalog, to derive the eligible-id set threaded through every
candidate source), once at the end (final validation, over the re-ranked
pool). One policy object, two call sites - they can never silently drift
apart into different rules.

<details>
<summary>Original 2026-08-12 "filter last" rationale (superseded above)</summary>

For V1's first pass, eligibility/business-rules filtering ran at the very
end of the pipeline only, not as a pre-ranking filter. The rationale at
the time: at the ~50-product catalog scale, the "avoid wasting ranking
compute on ineligible items" argument for an early filter was negligible,
while placing the check last kept retrieval and ranking entirely free of
serving-time state (stock, active flag) and gave one single, simple place
business rules could evolve independently of the ML stages. This is
superseded by the 2026-08-16 revision above, which found an architectural
correctness reason (not a performance one) to filter earlier: a candidate
pool that can include ineligible items can let them occupy pool/ranking
slots that should go to eligible products. Kept here for historical
context, not as current behavior.

</details>

## 6. Popularity terminology

V1 has no timestamps, views, clicks, or event tracking feeding the
recommender (see §7), so nothing is called "trending" - that word implies
a time window V1 cannot compute. The two fallback signals are:

- **global popularity** - derived from order quantities (and, where
  useful, cart/review volume) aggregated across all time, no decay.
- **category popularity** - the same aggregation, scoped to a category.

## 7. V1 scope vs. the original user story - explicitly deferred to V2

The following are **documented requirements, not implemented in V1**.
Nothing in this codebase should claim these acceptance criteria are met:

- Event tracking pipeline (product views, clicks, impressions)
- Real-time ingestion / event bus
- Session infrastructure, anonymous-user sessions
- Recency / time-decay features and freshness scoring
- Recommendation impression logging
- CTR, add-to-cart conversion, purchase conversion (online metrics)
- Sequential / recent-intent models over event streams

Timestamp fields that already exist in the ERD (`Order.CreationDate`,
`Order.DeliveryDate`, `Review.CreationDate`) are **retained in the
canonical schemas** (`PurchaseRecord.order_created_at`,
`ReviewRecord.review_created_at`) precisely so V2 can add recency features
without a schema migration - but V1 feature engineering must not derive
any recency/decay signal from them. This is enforced by convention now and
should be enforced by a feature-engineering-layer test once Phase 3 lands.

Extensibility is achieved through interfaces (adapters, the `context`
parameter on the serving entrypoint, the eligibility policy interface),
not by building unused event infrastructure now.

## 8. Offline vs. online evaluation (V1 vs. V2)

**Implemented in V1**: Recall@K/HitRate@K (Phase 4, `evaluation
.retrieval_metrics`), retrieval latency (Phase 5, `evaluation.latency`),
Precision@K/NDCG@K/reciprocal rank (MRR) (Phase 6, added to the same
`retrieval_metrics` module and used by `ranking.evaluation` to compare
the neural ranker against the raw-retrieval-score baseline), and catalog
coverage, intra-list category diversity, duplicate rate, requested-slot
fill rate, cold-start tier distribution, and pipeline latency up through
eligibility (Phase 7, `serving.evaluation.evaluate_pipeline` +
`scripts/run_pipeline.py`). True end-to-end (HTTP API-level) latency -
network + request/response serialization overhead on top of the
pipeline latency Phase 7 measures - is implemented in Phase 8 by timing
real HTTP requests against a running `scripts/run_api.py` instance; this
is a synthetic/local measurement (single process, loopback network, one
concurrent client), not production SLA evidence. Phase 9's dashboard
Metrics/Debug section (`ui.metrics.compute_offline_metrics`) is a UI
over this exact same Phase 7 evaluation call, computed on demand (not
automatically, since a full leave-one-out pass is not free) and
explicitly labeled in the UI as offline/synthetic - it does not add or
substitute for any new metric.

**Deferred to V2** (requires the event-tracking pipeline in §7): CTR,
impression volume, recommendation-click conversion, add-to-cart
conversion, purchase conversion.

Because all V1 data is synthetic, offline metrics demonstrate **pipeline
correctness and relative comparisons between V1 model variants**, not
real-world recommendation quality. This must not be overstated in any
report generated by this project.

## 9. Recommendation surfaces

V1 ships one model and one retrieval pipeline, but the serving layer
(`recommendation.serving`) accepts an optional `context` parameter from
the start (currently unused by any model) so that future surface-specific
behavior - home, product detail "you may also like", cart, search/discovery
- can be added by conditioning the existing pipeline, without a breaking
API change or separate per-surface models.

**Phase 8 status**: the API (`/v1/users/{user_id}/recommendations`)
does not yet accept or forward a `context` value - it calls
`RecommendationService.recommend(user_id, limit)`, which always passes
`context=None` to `serving.pipeline.recommend`. The hook exists and the
pipeline already accepts it; wiring an actual `context` query/body
parameter through the versioned wire contract is deferred until a real
surface-specific use case exists, consistent with "not by building
unused event infrastructure now" below.

**Phase 9 status**: the internal Streamlit dashboard
(`recommendation.ui`) is a second consumer of the same pipeline, but -
deliberately, for this V1 internal debug tool - it does NOT go through
the Phase 8 HTTP API. `ui.service_loader.load_service` reuses
`api.dependencies.RecommendationService`/`build_recommendation_service`
directly, in-process, so a single `streamlit run` works with no API
server to start first. This is still "one pipeline, reused, not
duplicated" - the SAME `RecommendationService` class and the SAME
`serving.pipeline.generate_recommendations` call the API makes, just
without an HTTP hop. `configs/base.yaml: dashboard.api_base_url` is
consequently unused for now, kept for a possible future microservice
split rather than removed.

## 10. ANN retrieval backend

**Revised 2026-08-12** (supersedes the original "FAISS default, ScaNN
optional in Phase 10" design): ScaNN is the **primary, production-
intended** `VectorIndex` backend, implemented in Phase 5, not deferred.
Two backends, both exact (brute-force) normalized-inner-product search
over the L2-normalized 128-D Two-Tower embeddings - i.e. exactly cosine
similarity - sufficient for the ~50-item synthetic catalog, with
approximate variants (FAISS IVF/HNSW, ScaNN tree+AH) available without an
interface change when the catalog grows:

- **ScaNN** - the intended production backend. Confirmed Linux-only, no
  Windows wheel exists or is planned upstream (verified against
  PyPI/GitHub; `scann==1.4.2` ships `cp313-manylinux_2_27_x86_64`
  wheels). Runs inside the Docker/Linux image (repo-root `Dockerfile`;
  a minimal ScaNN-only variant in Phase 5, superseded by the full
  multi-stage production image in Phase 10 - see section 10.1). Uses
  `score_brute_force(quantize=False)` - exact, unquantized dot-product
  search - the ScaNN equivalent of FAISS's `IndexFlatIP`.

  **ScaNN/TensorFlow ABI compatibility (Phase 10 finding)**: `scann`'s
  compiled ops extension is NOT ABI-compatible with an arbitrary
  TensorFlow version - it must match what scann was built against.
  `scann==1.4.2`'s own PyPI metadata declares `tensorflow~=2.20.0`; the
  project's general `ml` extra (`tensorflow>=2.16,<2.22`, used
  everywhere ELSE, including native-Windows dev where scann is never
  installed) lets pip resolve 2.21.0, which fails at `import scann` with
  `undefined symbol: ...absl...internal_log_function...`. This only
  surfaces when `ml` and `retrieval-scann` are installed in the SAME
  environment - Phase 5's ScaNN-only image never installed `tensorflow`
  at all (`retrieval.index.embeddings_io` deliberately avoids it), so
  the conflict was latent until Phase 10 unified everything into one
  image. Fixed by pinning `tensorflow~=2.20.0` explicitly in the
  Dockerfile's `pip install` (narrower than pyproject.toml's `ml` extra,
  which stays as-is for Windows).
- **FAISS** (`faiss-cpu`) - the native-Windows **development fallback**.
  Has genuine Windows wheels (verified: PyPI ships `cp313-win_amd64`
  builds), so local dev/tests/training run natively on this Windows
  machine without Docker. `IndexFlatIP` over the same embeddings.

Selected via `retrieval.backend` in `configs/base.yaml` (`faiss` - native
Windows dev default) or `configs/docker.yaml` (`scann` - what the Docker
image loads via `RECS_CONFIG_PATH`).

**Pre-retrieval eligibility restriction (Phase 11)**: neither backend's
`build`/index structure changes for this - `isActive`/`stockQuantity` are
serving-time catalog state, not something either backend's index needs to
know about internally. `retrieval.index.eligibility_filter
.EligibilityRestrictedIndex` wraps an already-built `VectorIndex` (either
backend) and restricts `search()` results to a caller-supplied eligible-
id set at query time - it asks the wrapped index for everything it has
(`index.size`, cheap at exact-brute-force V1 scale) and filters/truncates
in Python. This was chosen over backend-specific filtering (e.g. FAISS's
native `IDSelector`) specifically because ScaNN's brute-force pybind
searcher used here has no equivalent per-query id-filtering hook - one
backend-agnostic post-filter keeps both backends behaving identically
(both exact, both filtered the same way) rather than diverging into two
different filtering code paths with potentially different edge-case
behavior. If either backend switches to an approximate/partitioned
structure later (see above), this wrapper's "ask for everything" strategy
would need to become an adaptive oversampling strategy - noted, not
solved now, since it isn't a concern at V1's catalog scale.

## 11. Neural ranking (Phase 6)

`ranking.model.build_ranker_model` is a plain feedforward MLP over a
concatenated dense feature vector (`ranking.features
.RANKING_FEATURE_NAMES`), deliberately simpler-structured than the
Two-Tower's dual-embedding-tower architecture - no learned category/
brand embedding lookups; retrieval learns a shared embedding space for
ANN search, ranking uses explicit, interpretable cross features for
precision on a small candidate set. Features: user history aggregates,
item metadata/popularity (including `stockQuantity`/`isActive` -
available as features, never used to filter here, see §5 - since §5's
2026-08-16 revision, every candidate reaching the ranker is already
pre-retrieval-eligible, so `isActive` is a constant `True` and
`stockQuantity` is always `> 0` for every row the ranker scores; this
does not change the ranker itself, only the value distribution of two
of its many input features - not a regression protection concern per
Phase 11's scope, since the ranker's architecture/training is untouched),
explicit
user-item cross features (category/brand affinity match, semantic
cosine similarity), and the retrieval stage's own score/rank for each
candidate (`ranking.examples`).

Training is pointwise binary classification: a candidate is labeled 1 if
the user purchased it (train-split), 0 if sampled from what
`VectorIndex.search` actually retrieves minus known positives and
held-out val/test ids (retrieved-negatives, not raw-catalog-uniform
negatives, to match serving-time candidate distribution). Splits reuse
Two-Tower's own `UserSplit`s (same seed/threshold) so held-out val/test
targets are identical between the two models - required for the
ranker-vs-raw-retrieval-score baseline comparison (`ranking.evaluation`)
to be apples-to-apples. Leave-one-out leakage discipline extends §12's
guard: a positive candidate's own retrieval score/rank is computed from
a user embedding with that candidate excluded from history, same as
Two-Tower training.

Evaluated with Precision@K/NDCG@K/Recall@K/HitRate@K/MRR
(`evaluation.retrieval_metrics`) against both the ranker's own scores
and the unchanged retrieval-score ordering, over the identical retrieved
candidate pool for both - isolating what re-ranking alone contributes.
At the ~50-item V1 catalog the candidate pool equals the full catalog
(see §10), so this validates the ranking pipeline's correctness and
leakage safety, not ranking-under-genuine-ANN-truncation at production
scale.

## 12. V1 leakage limitation - no timestamps

Because V1 has no timestamps on search/chatbot interactions (§7), feature
engineering (Phase 3) cannot tell whether a search or chatbot conversation
happened *before* or *after* a given purchase. This matters specifically
when Phase 4 builds a (user, target_product) training example: the
user-side features must not encode "the user already told us about this
exact product," or the model would trivially memorize instead of
generalizing.

Mitigations, implemented in `recommendation.features.user_features`:

- **Purchases/cart**: excluding a product drops the entire record (its
  content *is* the product reference) - see `exclude_product_ids` on
  `build_user_features`.
- **Search**: a search matched to the excluded product (`matched_product_id`)
  has that match scrubbed; an *unmatched* search whose raw `search_term`
  text names the excluded product (substring match against the product's
  name) is also scrubbed from the semantic-embedding signal, even without
  a resolved product id.
- **Chatbot**: if a chatbot record mentions the excluded product by id
  *or* by name in its free-text summary, its entire content contribution
  (all mentioned-product embeddings and the summary embedding) is dropped
  - not just the flagged reference - since a single conversation's "safe"
  and "risky" parts can't be reliably separated without a timestamp.

This is a **heuristic, not a guarantee**: it catches exact-name mentions,
not paraphrases, synonyms, or a search/chatbot event that references the
target only indirectly. Event *counts* (`search_count`,
`has_chatbot_context`) are intentionally left unaffected by exclusion -
only the product-content signal is scrubbed - so the leakage guard doesn't
silently distort the history-strength signal Phase 7's cold-start tiering
depends on. Fully resolving this requires real timestamps (V2) to build a
genuinely temporal held-out split instead of this content-based heuristic.

## 13. Phase map (which phase implements what in this document)

| Section here | Implementing phase |
|---|---|
| §2 UserProfile fields | Phase 2 |
| §3 Cold-start tiers | Phase 7 |
| §4 Search/Chatbot adapters | Phase 2 |
| §5 Eligibility/business rules policy (hard pre-retrieval gate + final lightweight validation) | Phase 7 (policy interface, originally applied last) / Phase 11 (moved to a hard pre-retrieval gate, mentor-reviewed) |
| §6 Popularity | Phase 2 (data) / Phase 7 (fallback ranking) |
| §8 Offline evaluation | Phase 4 (Recall/HitRate) / Phase 5 (latency) / Phase 6 (Precision/NDCG/MRR) / Phase 7 (coverage/diversity/duplicate/fill-rate/cold-start/pipeline latency) / Phase 8 (HTTP end-to-end latency) |
| §9 Surface context hook / dashboard architecture | Phase 7 (pipeline parameter, not yet exposed via API) / Phase 9 (dashboard reuses RecommendationService in-process, not via HTTP) |
| §10 VectorIndex backends | Phase 5 (ScaNN primary/Docker + FAISS Windows dev fallback) |
| §11 Neural ranking (VectorIndex candidates, richer cross features, baseline comparison) | Phase 6 |
| §12 Leakage-limitation mitigation | Phase 3 (guard) / Phase 4 (consumer) / Phase 6 (extended to ranking negatives) |
