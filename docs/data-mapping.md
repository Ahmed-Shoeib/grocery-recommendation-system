# ERD → Canonical Schema Mapping & V1 Scope Decisions

Source of truth for the backend schema: `docs/erd.jpeg`. This document
reconciles that ERD with the recommendation system's requirements, records
the assumptions V1 makes, and defines the scope boundary between V1 and
deferred V2 work. It is updated as clarifications arrive (last updated
after the backend team's `User_events` activity-log contract was
confirmed - see §4).

## 1. Entities as drawn in the ERD

| Entity | Fields | Recommendation relevance |
|---|---|---|
| User | Id, FirstName, LastName, Email, PhoneNumber, HashedPassword, RefreshToken, Role, CreatedAt, UpdatedAt | Identity + (confirmed, pending) `preferredCategory`, `ageGroup` - see §2. |
| Category | Id, ParentId (self-FK), Name, CreatedAt | Category + parent-category features. |
| Product | Id, CategoryId, Slug, Name, Description, Brand, Price, SalePrice, DiscountPercentage, StockQuantity, Ingredients, isActive, ProductImage, AltText | Core item metadata; text fields feed the Sentence Transformer; isActive/StockQuantity feed the hard pre-retrieval eligibility gate AND the final lightweight eligibility validation (§5). |
| ProductTags / Tag | join table + Name | Tag text feeds the Sentence Transformer input. |
| Cart / Cart_Item | Cart(Id, CartItemId FK, UserId); CartItem(Id, CartId, ProductId, Quantity) | Add-to-cart habit signal (D) - today's ERD-backed path only; see §4 for the future `User_events`-sourced path. |
| Order / Order_Item | Order(Id, UserId, VoucherId, AddressId, IdempotenceKey, TotalAmount, Status, PaymentMethod, CreationDate, DeliveryDate); OrderItem(Id, ProductId, OrderId, Quantity, UnitPrice) | Previous purchases signal (A) - the strongest V1 signal. Today's ERD-backed path only; see §4. |
| Review | Id, UserId, ProductId, Rating, Comment, CreationDate | Auxiliary ranking signal (rating/review-count affinity). Unaffected by the `User_events` contract (§4) - stays ERD-backed. |
| UserAddress, Voucher | address/voucher fields | Not used by the recommender. |
| **User_events** (confirmed, not yet built) | Id, UserId, ProductId, ActionTime, ActionType (`CLICK` / `ADD_TO_CART` / `PURCHASE` / `SEARCH` / `CHATBOT`) | The future single append-style activity log for ALL FIVE engagement signals - see §4 for the full contract. Not in `docs/erd.jpeg` (which predates this confirmation), analogous to the §2 `preferredCategory`/`ageGroup` addition. |

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
across the five V1 engagement signals (clicks, purchases, cart adds,
searches, chatbot-context presence) and thresholded via `configs/base.yaml:
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
factors in `preferred_category` plus whatever sparse click/purchase/cart/
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

## 4. `User_events` - the confirmed future engagement source, click/search/chatbot adapters today

**Confirmed by the backend team (2026-08-17)**: engagement tracking will
NOT be five separate backend tables. It will be one append-style
activity/audit-log table:

```
User_events
    id
    user_id
    product_id
    action_time
    action_type
```

`action_type` is one of five values, covering ALL FIVE V1 engagement
signals: `CLICK`, `ADD_TO_CART`, `PURCHASE`, `SEARCH`, `CHATBOT`. This is
an append-only activity log, not a deduplicated relation: multiple rows
for the same `(user_id, product_id)` pair are valid and expected - a user
clicking, then adding to cart, then purchasing the same product each
produce their own row, each with its own `action_time`.

**`product_id` is required for every event this recommender consumes -
including SEARCH and CHATBOT.** The backend only writes a `User_events`
row for those two signals once the interaction has been resolved to a
specific product:

- **SEARCH**: a row is written only once a search has been resolved to a
  specific product. The raw search query text is NOT part of this
  contract and is NOT needed by this recommender - an unresolved search
  simply never produces a row here.
- **CHATBOT**: a row is written only once a chatbot conversation has been
  resolved to a specific product. Chatbot summary text, keywords, intent
  JSON, or any other chatbot-specific metadata are NOT part of this
  contract and NOT needed by this recommender - a conversation that never
  names a product simply never produces a row here.

This recommender intentionally operates on resolved, product-level
interactions only, never on raw free text for these two signals. Category/
brand/etc. are derived via `User_events.product_id -> Product`; age_group/
preferredCategory are derived via `User_events.user_id -> User` (§2) - no
additional columns (`search_query`, `chatbot_text`, `chatbot_summary`,
`chatbot_keywords`, `chatbot_preferred_category`, a metadata JSON blob,
etc.) are needed on `User_events` itself.

### Canonical representation and the adapter boundary

`data.schemas.events.UserInteraction` (`user_id`, `product_id`,
`action_type`, `action_time`) is the canonical, source-agnostic shape of
one `User_events` row. `data.adapters.user_events_adapter
.UserEventsAdapter` is the seam that will translate real `User_events`
rows into `UserInteraction`s and, from there, into the SAME per-signal
canonical records (`ClickRecord`, `PurchaseRecord`, `CartAffinityRecord`,
`SearchRecord`, `ChatbotContextRecord`) that `EngagementProfile`
(`data.schemas.engagement`) already uses today - by implementing the
existing `ClickAdapter`/`PurchaseAdapter`/`CartAdapter`/`SearchAdapter`/
`ChatbotContextAdapter` interfaces from that one unified event list
(action-type fan-out), instead of five separate ERD-backed adapters. Feature
engineering, the Two-Tower model, the ranker, and the serving pipeline
depend only on `EngagementProfile`/`AdapterBundle` and never see
`UserInteraction` or know which adapter produced their input - the
architectural goal stated at the top of this document:

```
Today (synthetic V1):
  synthetic generators -> existing per-signal adapters -> EngagementProfile -> features -> models

Future (real backend):
  User_events table -> UserEventsAdapter -> EngagementProfile -> features -> models
                        (same interfaces, same downstream, only construction differs)
```

`build_user_events_adapters(events, products_adapter, users_adapter,
reviews_adapter)` (`data.adapters.user_events_adapter`) is the real-backend
counterpart of today's `build_synthetic_adapters` - it returns the same
`AdapterBundle` type, so no call site depending on it needs to change once
a real `User_events` query replaces the synthetic generators.

**What actually changes when the real database arrives**: only whatever
runs the SQL query against `User_events` and maps each row into a
`UserInteraction` - i.e. the construction of the `events` list passed to
`UserEventsAdapter`. `UserEventsAdapter` itself, `AdapterBundle`,
`EngagementProfile`, all feature engineering, the Two-Tower model, the
ranker, and the serving pipeline require no changes.

### Field-availability gap vs. the ERD-backed path

`User_events` never carries `Order`/`OrderItem`-specific fields
(`order_id`, `unit_price`) or `CartItem.quantity`, or any search/chatbot
free text. `PurchaseRecord.order_id`/`unit_price`/`order_status` and
`CartAffinityRecord`/`PurchaseRecord.quantity` are therefore optional on
the canonical schemas (default `None`/`1`) specifically so
`UserEventsAdapter` can construct them with only what `User_events`
actually provides - see `data.schemas.engagement` for the full field-by-
field rationale. `ChatbotContextRecord` stays a single aggregated record
per user (one entry in `mentioned_product_ids` per resolved CHATBOT row);
see that class's docstring for the known V1 simplification this implies
(no per-mention timestamp within that aggregate).

### CLICK - the fifth V1 engagement signal

CLICK (the user viewed/clicked a specific product) is now a first-class
V1 engagement signal, not deferred to V2 (contrast with §7, which still
defers the *real-time event-tracking pipeline* - impressions, CTR,
session-level click-stream - not the click signal itself as consumed by
this recommender). Today, with no `User_events` table yet, V1 sources it
the same way SEARCH/CHATBOT are sourced: `ClickAdapter` (interface) /
`SyntheticClickAdapter` (V1 implementation), producing canonical
`ClickRecord`s. It contributes to `UserFeatures.category_affinity`/
`brand_affinity`/`semantic_embedding` (weighted by
`config.features.click_weight`) and to `total_engagement_events` (cold-
start tiering, §3), exactly like the other signals.

`click_weight` is a small, deliberately unbenchmarked placeholder (see
`FeatureConfig` docstring, `src/recommendation/utils/config.py`) - CLICK
has no real usage data to calibrate against yet. It does NOT add a new
numeric input slot to the Two-Tower or ranker feature vectors (those
dimensions are baked into already-trained model artifacts); it only flows
through the existing vocabulary-/embedding-dimensioned affinity/embedding
features and the existing `total_engagement_events` numeric slot. All
signal weights - not just this one - should be re-benchmarked once the
real `User_events` dataset is available; nothing in this codebase should
claim the current weights are tuned.

### Timestamps

`action_time` is preserved end-to-end from the adapter layer
(`UserEventsAdapter` populates `PurchaseRecord.order_created_at`/
`CartAffinityRecord.action_time`/`ClickRecord.action_time`/
`SearchRecord.action_time` from `UserInteraction.action_time`, and
`ChatbotContextRecord.action_time` from the most recent resolved mention -
see §14) so the temporal future-purchase evaluation split (§8.1) and
recency weighting (§14) don't require another data-contract change.
Sequential/event-order features beyond point-in-time truncation and
recency remain out of scope (§7).

### SQLite integration: `data/sqlite/backend_shaped_synthetic.db`

**This database is entirely SYNTHETIC - not real user or purchase data,
and no documentation should ever describe it as such.** It is a
deterministic, backend-ERD-shaped SQLite database (`scripts
.generate_backend_shaped_sqlite.py`, `scripts.validate_backend_shaped_sqlite.py`)
built to mirror the confirmed backend contract above as closely as
possible: 1,000 users, 1,200 products, and a genuine row-per-action
`User_events` table (all five action types, all product-resolved,
individually timestamped) - replacing an earlier-inspected POC
(`ecommerce.db`) that used a pre-aggregated, non-conforming activity table
and is kept only as an untouched historical reference, not integrated.

`recommendation.data.adapters.sqlite_factory.build_sqlite_adapters` is the
adapter path for this source - the SQLite counterpart of
`adapters.factory.build_synthetic_adapters`, returning the exact same
`AdapterBundle` type:

```
data/sqlite/backend_shaped_synthetic.db
    -> recommendation.data.sqlite.loader (SQL -> RawCategory/RawProduct/
       RawUser/RawReview/UserInteraction - the SAME raw/canonical models
       the synthetic generator path already produces)
    -> InMemoryProductCatalogAdapter / InMemoryUserAdapter /
       InMemoryReviewAdapter (reused as-is, just fed SQLite rows) +
       UserEventsAdapter (reused as-is, fed the loaded UserInteraction list)
    -> AdapterBundle  (identical interface to the synthetic path)
    -> EngagementProfile -> feature engineering -> Two-Tower -> ranker -> serving
```

No new adapter *classes* were needed - only the SQL-to-Raw-object mapping
in `data.sqlite.loader`. Access is read-only (`data.sqlite.connection
.open_readonly_connection`, SQLite `mode=ro` URI - verified to actually
reject writes, not just a naming convention); this dataset must never be
mutated by the recommender.

**Purchase/cart authoritative source (avoiding double-counting):**
`User_events` (`action_type = PURCHASE` / `ADD_TO_CART`) is the sole
engagement-truth source consumed by this adapter path. `data.sqlite.loader`
never queries `Cart`/`Cart_Item` or `"Order"`/`Order_Item` at all -
those tables exist in the database (kept relationally consistent with
`User_events` by the generator, see `scripts
.generate_backend_shaped_sqlite.py`'s module docstring) but are simply
never read here, so double-counting is structurally impossible, not just
avoided by convention.

**Configuration:** `config.paths.data_sqlite` (default
`data/sqlite/backend_shaped_synthetic.db`) - `build_sqlite_adapters(db_path=None)`
falls back to this when no explicit path is passed, following the
existing `paths.*` config pattern rather than introducing a new
"data source" abstraction. This is an integration/experimentation path
only in this phase - the live API/dashboard service
(`api.dependencies.build_recommendation_service`) still uses the
synthetic path exclusively; nothing about which source serves live
requests changed here.

**Status of price/timestamp readiness:** `Product.Price`/`SalePrice`/
`DiscountPercentage` are correctly exposed through the canonical `Product`
schema, but no price-aware feature consumes them yet - still deferred.
`action_time` is parsed into real `datetime` values and survives the
adapter boundary exactly as it does for the synthetic path; it is now
consumed for BOTH the temporal future-purchase evaluation split (§8.1)
AND recency weighting (§14, STEP 5) - time-of-day/session-sequence
features remain the only still-deferred timestamp use, matching §7's
scope boundary.

**Future real backend integration:** per this section's flow above,
swapping this database for a real backend DB/API means writing a new
`build_<real>_adapters` factory (or updating `data.sqlite.loader`'s SQL if
the real backend also happens to be SQLite/SQL-compatible) - it does not
mean touching `EngagementProfile`, feature engineering, Two-Tower, the
ranker, or serving. The adapter boundary is exactly where this integration
was designed to absorb that change.

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

V1 does not derive recency/time-decay from any timestamp, and no
real-time event-tracking/impression pipeline feeds the recommender (see
§7), so nothing is called "trending" - that word implies a time window V1
cannot compute. This is true even though CLICK is now a first-class
engagement *signal* (§4): clicks feed user-side category/brand affinity
and the semantic embedding, not the popularity fallback signals below,
which remain purchase/cart/review-derived only. The two fallback signals
are:

- **global popularity** - derived from order quantities (and, where
  useful, cart/review volume) aggregated across all time, no decay. Not
  click-derived in V1 - see §7 for why a click-based popularity/CTR
  signal stays deferred.
- **category popularity** - the same aggregation, scoped to a category.

## 7. V1 scope vs. the original user story - explicitly deferred to V2

The following are **documented requirements, not implemented in V1**.
Nothing in this codebase should claim these acceptance criteria are met:

- Real-time event-tracking pipeline / event bus / impression logging
  (CLICK as a per-user, per-product *engagement signal* is implemented in
  V1 as of §4 - what's still deferred is a live ingestion pipeline,
  impression-level tracking, and CTR/conversion computed from it)
- Session infrastructure, anonymous-user sessions
- ~~Recency / time-decay features and freshness scoring~~ - **implemented
  as of the STEP 5 recency phase, see §14**, for every signal that
  actually carries a real `action_time` (the `User_events`/SQLite-sourced
  path); the old, timestamp-less ERD synthetic signals still get a
  neutral (unweighted) fallback, so this bullet's ORIGINAL scope (a
  freshness signal derived from real timestamps) is what's now done -
  "freshness scoring" as a separate product-side signal (e.g. a listing's
  own age) remains out of scope.
- CTR, add-to-cart conversion, purchase conversion (online metrics)
- Sequential / recent-intent models over event streams
- Product-level click-popularity/"trending" fallback signal (§6) - clicks
  are consumed as a user-side affinity signal only, not yet aggregated
  into a popularity ranking signal

Timestamp fields that already exist in the ERD (`Order.CreationDate`,
`Order.DeliveryDate`, `Review.CreationDate`), plus `action_time` on the
confirmed future `User_events` table (§4, threaded through to
`PurchaseRecord.order_created_at`/`CartAffinityRecord.action_time`/
`ClickRecord.action_time`/`SearchRecord.action_time`), are **retained in
the canonical schemas** precisely so recency features could be added
without another schema migration - as of the STEP 5 recency phase (§14),
feature engineering DOES derive a decay signal from them, superseding
this section's original "must not" for those fields specifically.

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

### 8.1 Temporal future-purchase evaluation protocol (`evaluation.temporal_future_purchase`)

**What the metrics are based on.** The primary offline relevance target
is a held-out **future PURCHASE**: given only the engagement history that
existed before a cutoff, could the recommender have retrieved/ranked the
product(s) the user went on to purchase after that cutoff? CLICK, SEARCH,
CHATBOT, and ADD_TO_CART are **input signals** describing intent/history,
never automatically treated as evaluation targets themselves:

```
INPUT (point-in-time history, action_time < cutoff):
  CLICK, SEARCH, CHATBOT, ADD_TO_CART, and historical PURCHASE events
TARGET (what "relevant" means for every metric in this protocol):
  PURCHASE events with action_time >= cutoff, up to the next split boundary
```

This is an ADDITIONAL protocol, implemented in `recommendation.evaluation
.temporal_future_purchase`, alongside - not replacing - the original V1
leave-one-out-by-product-id protocol (`retrieval.two_tower.splitting`,
§12 below). That protocol has no timestamps to work with and is what the
currently-trained Two-Tower/ranker artifacts under `models/` were fit
against; it remains what `scripts/train_two_tower.py`/`train_ranker.py`/
`run_pipeline.py` use. The temporal protocol targets `data
/sqlite/backend_shaped_synthetic.db` (via `data.adapters.sqlite_factory
.build_sqlite_adapters`), which has real per-event `action_time` values -
**this database is entirely SYNTHETIC, mirroring the expected backend
structure; it is NOT real production data**, and must never be described
as such in any report this project produces.

**Per-user temporal split** (`build_temporal_splits`, deterministic - no
RNG, since chronological order is a total order given by the data
itself). Let a user's PURCHASE-event timestamps, sorted ascending, be
`t_1 <= t_2 <= ... <= t_n`:

| Purchase-event count `n` | Tier | val_cutoff | test_cutoff |
|---|---|---|---|
| `n >= 3` | `FULL` | `t_{n-1}` | `t_n` |
| `n == 2` | `VAL_ONLY` | `t_n` (the later one) | none |
| `n == 1` | `INSUFFICIENT_DEPTH` | none | none |
| `n == 0`, other events present | `ENGAGEMENT_NO_PURCHASE` | none | none |
| `n == 0`, no events at all | `NO_HISTORY` | none | none |

This is a distinct classification from `serving.cold_start.HistoryTier`
(STRONG/SPARSE/NO_HISTORY, which measures overall point-in-time
engagement *volume*) - a user can be `HistoryTier.STRONG` (lots of
clicks/searches) while still being temporally `INSUFFICIENT_DEPTH` (too
few purchase events to hold anything out), and the dry-run script reports
both dimensions.

**Cutoff / same-timestamp semantics** (`events_before_cutoff`,
`purchase_targets_in_window`): history uses **strict `<`**; the
target/relevant-set window uses **`>=`** (lower bound inclusive) and,
where there is a next split boundary, `<` that boundary (upper bound
exclusive) so val and test target sets are always disjoint. An event with
`action_time == cutoff` is therefore never part of history and IS
eligible to be part of the target window - deterministic, independent of
database row order.

**Point-in-time feature construction** (`build_point_in_time_engagement_profile`)
truncates the RAW `UserInteraction` event list to `action_time < cutoff`
**before** building any canonical per-signal record, via the existing,
unmodified `UserEventsAdapter` + `adapters.engagement.build_engagement_profile`
- not by truncating an already-built `EngagementProfile`. This is what
makes CHATBOT truncation correct despite `ChatbotContextRecord` having no
per-mention timestamp (a documented aggregate-record simplification, see
`data.schemas.engagement.ChatbotContextRecord`'s docstring): the aggregate
is built fresh from only the pre-cutoff CHATBOT events, so it never needs
a per-mention timestamp to be truncated correctly. No product-id-based
exclusion (`exclude_product_ids`) is used for this protocol at all - time-
based truncation alone is both necessary and sufficient, and it is
strictly more precise than the older product-id-based guard (which exists
only because the original V1 protocol has no timestamps to truncate by).

**Repeat purchases** (docs' grocery-replenishment requirement): a
product is never excluded from candidacy just because the user bought it
before. An earlier (< cutoff) purchase of the same product that later
becomes the held-out target is legitimate, preserved history - this falls
out automatically from time-based truncation, with no special-case code.
The dry-run script reports how many val/test targets are repeat purchases
(on the real SQLite dataset: 14/378 val targets, 11/204 test targets are
repeats - see the phase's dry-run report for the full numbers).

**Multiple relevant targets**: `purchase_targets_in_window` returns a
`frozenset[int]`, not a single id - ties at a cutoff timestamp (two
products purchased at the identical moment) naturally produce a multi-
item relevant set. `evaluation.retrieval_metrics`'s functions
(`recall_at_k`/`precision_at_k`/`ndcg_at_k`/`hit_rate_at_k`/
`reciprocal_rank`) already accept `relevant_ids: set[int]` generically -
no metric-function changes were needed for this protocol, only proof (see
`tests/test_retrieval_metrics.py`'s multi-relevant-item worked example)
that they behave correctly for sets larger than one.

**Eligibility / unavailable-target policy**: this SQLite dataset has no
historical stock/active snapshot - only the CURRENT `Product.isActive`/
`StockQuantity`. A future-purchase target that is currently inactive or
out of stock was structurally impossible for the hard pre-retrieval
eligibility gate (§5) to ever surface, regardless of model quality.
`split_targets_by_eligibility` partitions targets into eligible/
ineligible using the SAME `serving.eligibility` policy (never a separate
one) so this can be reported honestly rather than silently counted as an
ordinary miss (on the real dataset: 222/270 distinct val targets and
144/170 distinct test targets are currently eligible).

**Leakage audit** (`audit_no_leakage`): for a real evaluation point,
audits the ACTUAL historical event list that was used (not a redundant
re-derivation of the same filter), flagging any event with a missing or
`>= cutoff` timestamp. A legitimate earlier purchase of a product that
later becomes the target is explicitly NOT flagged (see above). The
SQLite dry run audited every val/test evaluation point in the dataset (582
points) with zero violations.

**Accuracy vs. list-quality vs. latency** (unchanged, restated for
clarity given this new protocol): Precision/Recall/HitRate/NDCG/MRR - all
computed against the future-purchase relevant set above - measure
*prediction accuracy*. Catalog coverage, intra-list category diversity,
duplicate rate, and fill rate (§7's `serving.evaluation`) measure *list
quality/behavior*, not prediction accuracy - `fill_rate = 1.0` means the
system filled the requested slots, not that it predicted correctly.
Latency (`evaluation.latency`) measures *serving performance* only. None
of these three categories substitutes for another; this phase did not
change any of them except adding the new accuracy-category protocol
above.

**What this phase does NOT do**: compute actual Top-N recommendation
metrics (real Recall/NDCG/etc. numbers from a real ranked list) against
this SQLite dataset. Doing so meaningfully requires the Two-Tower item
tower to encode this database's 1,200-product catalog and a VectorIndex
built over those embeddings - the currently-trained artifacts were fit
against the OLD 50-product synthetic catalog (different product ids,
different brand vocabulary) and rebuilding the index is out of scope for
this phase (reserved for the retraining phase). The metric *functions*
are proven correct against hand-computed worked examples including a
genuine multi-relevant-item case (`tests/test_retrieval_metrics.py`) and
are ready to be plugged into real Top-N output once retraining happens.
Recency weighting, temporal train/validation/test splitting FOR MODEL
TRAINING (as opposed to this evaluation-only split), price-aware
features, and time-of-day features are explicitly deferred to later
phases - `action_time` is preserved and now consumed for point-in-time
truncation, but nothing in this phase changes interaction *strength*
based on time.

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

- **Clicks/purchases/cart**: excluding a product drops the entire record
  (its content *is* the product reference, same as purchases/cart) - see
  `exclude_product_ids` on `build_user_features`. Clicks carry no free
  text, so no separate text-leakage heuristic is needed for them.
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
target only indirectly. Event *counts* (`click_count`, `search_count`,
`has_chatbot_context`) are intentionally left unaffected by exclusion -
only the product-content signal is scrubbed - so the leakage guard doesn't
silently distort the history-strength signal Phase 7's cold-start tiering
depends on. Fully resolving this requires real timestamps (V2) to build a
genuinely temporal held-out split instead of this content-based heuristic.

## 13. Phase map (which phase implements what in this document)

| Section here | Implementing phase |
|---|---|
| §2 UserProfile fields | Phase 2 |
| §3 Cold-start tiers | Phase 7 (four signals) / User_events contract change (five signals, click added) |
| §4 Click/Search/Chatbot adapters; `User_events` contract, `UserInteraction`, `UserEventsAdapter` | Phase 2 (search/chatbot synthetic adapters) / User_events contract change (click signal + synthetic adapter, `UserInteraction` canonical event, `UserEventsAdapter` future real-backend adapter, action_time preservation) / SQLite integration phase (`backend_shaped_synthetic.db`, `adapters.sqlite_factory.build_sqlite_adapters`, `data.sqlite.*`) |
| §5 Eligibility/business rules policy (hard pre-retrieval gate + final lightweight validation) | Phase 7 (policy interface, originally applied last) / Phase 11 (moved to a hard pre-retrieval gate, mentor-reviewed) |
| §6 Popularity | Phase 2 (data) / Phase 7 (fallback ranking) |
| §8 Offline evaluation | Phase 4 (Recall/HitRate) / Phase 5 (latency) / Phase 6 (Precision/NDCG/MRR) / Phase 7 (coverage/diversity/duplicate/fill-rate/cold-start/pipeline latency) / Phase 8 (HTTP end-to-end latency) |
| §8.1 Temporal future-purchase evaluation protocol (`evaluation.temporal_future_purchase`, SQLite temporal splits, point-in-time truncation, leakage audit) | Offline evaluation redesign phase (evaluation/splitting infrastructure only - no retraining, no index rebuild) |
| §9 Surface context hook / dashboard architecture | Phase 7 (pipeline parameter, not yet exposed via API) / Phase 9 (dashboard reuses RecommendationService in-process, not via HTTP) |
| §10 VectorIndex backends | Phase 5 (ScaNN primary/Docker + FAISS Windows dev fallback) |
| §11 Neural ranking (VectorIndex candidates, richer cross features, baseline comparison) | Phase 6 |
| §12 Leakage-limitation mitigation | Phase 3 (guard) / Phase 4 (consumer) / Phase 6 (extended to ranking negatives) / User_events contract change (extended to clicks) |
| §14 Recency weighting (`features.recency`, `effective_weight`, `_signal_embedding_component`) | STEP 5 recency phase |
| §15 Price-aware derived features (`features.price`, `price_tier_id`/`PriceCatalogContext`/`UserPriceProfile`) | STEP 6 price phase |

## 14. Recency weighting (STEP 5)

**Formula.** `features.recency.effective_weight(base_signal_weight,
event_time, reference_time, config.recency)` =
`base_signal_weight * 0.5 ** (age_days / half_life_days)`, where
`age_days = (reference_time - event_time).total_seconds() / 86400`.
`age=0 -> 1.0`, `age=half_life -> 0.5`, `age=2*half_life -> 0.25`,
asymptotically approaching but never reaching 0 for very old events.

**Config** (`configs/base.yaml: features.recency`): `enabled` (default
`true`) and `half_life_days` (default `21.0`, three weeks) - a single
global half-life, not per-signal, kept deliberately simple/interpretable
for this phase. `half_life_days=21` is an **initial, explicitly
unbenchmarked baseline** for grocery's weekly/biweekly replenishment
cadence, chosen by inspecting this SQLite dataset's ~7-month event span
(2026-01-15 .. 2026-08-17) - not tuned against real usage data, and
should be re-evaluated once real usage data exists (same caveat as
`click_weight`, §4).

**Recency is strictly opt-in per call**, never an implicit wall-clock
lookup: `build_user_features(..., reference_time=None)` (the default)
leaves every recency-weighted contribution neutral
(`effective_weight == base_weight`), REGARDLESS of `config.recency
.enabled`. A caller must explicitly pass a `reference_time` for recency
to take effect:
  - Offline temporal evaluation (§8.1): the caller passes the evaluation
    cutoff (`TemporalUserSplit.val_cutoff`/`test_cutoff`), matching the
    already-point-in-time-truncated `EngagementProfile`.
  - Live serving (`serving.pipeline.recommend`): passes `datetime.now()`
    at the request boundary.
  - The pre-existing, non-temporal leave-one-out training path
    (`retrieval.two_tower.examples`, `ranking.examples`,
    `retrieval.two_tower.train`) deliberately never passes
    `reference_time` - those per-training-example calls have no natural
    per-example "now." This was NOT a hypothetical concern: an earlier
    version of this phase defaulted `reference_time` to
    `datetime.now()` whenever `config.recency.enabled` was true and no
    explicit value was supplied, and that silently decayed the OLD
    ERD-based synthetic dataset's purchase dates relative to whatever day
    the code happened to run on - caught by
    `test_two_tower_train_pipeline.py::test_training_beats_random_baseline_on_test_set`
    dropping below its quality floor. The opt-in design fixes this: that
    training path's features are now provably byte-for-byte unchanged by
    this phase (see `tests/test_user_features.py::test_reference_time_omitted_leaves_recency_neutral_even_when_events_are_timestamped`).

**Missing-timestamp fallback**: an event with `event_time is None` (the
old, timestamp-less ERD synthetic CLICK/ADD_TO_CART/SEARCH generators)
also gets a neutral weight - never dropped, never down-weighted.

**Future-event guard**: `features.recency.recency_weight`/
`effective_weight` raise `RecencyLeakageError` if `event_time >=
reference_time` - a future-relative event is never silently accepted
(only ever hit by a caller bug, since both the temporal harness and live
serving guarantee `event_time < reference_time` by construction).

**Where applied** (`features.user_features.build_user_features`):
  - `category_affinity`/`brand_affinity`: each per-event contribution is
    scaled by `effective_weight` before accumulating into the `Counter`
    that `_normalize_and_truncate` turns into a distribution - unchanged
    structurally, just each addend is now recency-scaled.
  - `semantic_embedding`: reworked from "one unweighted mean vector per
    signal" to `_signal_embedding_component`, which recency-weights BOTH
    which items dominate the mean WITHIN a signal (content) and that
    signal's overall blend weight via the AVERAGE (not sum) of its
    per-event recency weights (magnitude) - the average keeps this
    invariant to event count, exactly matching the pre-recency behavior
    when recency is disabled or every event lacks a timestamp (reduces to
    the original code byte-for-byte).
  - `Product.preferred_category`'s static profile-attribute contribution
    is NEVER recency-weighted (it isn't a timestamped interaction).
  - Chatbot: `ChatbotContextRecord` gained one field, `action_time` (the
    most recent resolved-mention timestamp for that user -
    `UserEventsAdapter.get_chatbot_context`); since per-mention timestamps
    aren't preserved (§12), this ONE timestamp governs the whole chatbot
    record's contribution (preferred category, mentions, summary
    embedding alike) as a single group. This is a recommendation-layer
    canonical-schema field, not a backend `User_events` column - no
    backend contract change.
  - Raw event *counts* (`click_count`, `purchase_count`,
    `total_engagement_events`, ...) are NEVER recency-weighted - cold-
    start tiering (§3) keeps meaning "how much history," not "how much
    RECENT history."

**Evaluation - does recency improve future-purchase quality?**
`scripts/sqlite_recency_evaluation.py` compares BASELINE
(`recency.enabled=False`) vs RECENCY (`half_life_days=21`) on IDENTICAL
temporal evaluation points (same users/cutoffs/targets/eligibility/
candidate pool - the only intended difference is recency), scored by
cosine similarity between each user's `semantic_embedding` and the SAME
frozen Sentence Transformer product embeddings both arms use (a
retraining-free way to isolate this phase's effect - see that script's
module docstring for why Two-Tower/ranker retraining on this 1,200-
product catalog is out of scope here, matching §8.1's prior scope note).
Results (this synthetic dataset, `min_purchase_events_for_full_split=3`):

| split | arm | Recall@5 | Recall@10 | Recall@20 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|
| val (n=378) | baseline | 0.087 | 0.151 | 0.246 | 0.106 | 0.068 |
| val (n=378) | recency | 0.601 | 0.640 | 0.701 | 0.567 | 0.527 |
| test (n=204) | baseline | 0.078 | 0.118 | 0.181 | 0.090 | 0.066 |
| test (n=204) | recency | 0.564 | 0.632 | 0.691 | 0.512 | 0.459 |

(HitRate@K equals Recall@K here per the single-arm-consistent-metric note
in `evaluation.retrieval_metrics`'s module docstring - both arms hold
multiple relevant items per query, so this is not a coincidence of
leave-one-out.) Recency is a large, consistent win on both splits. List-
quality stayed comparable (mean distinct top-20 categories ~4.8-5.0 for
both arms; catalog coverage 0.81-0.97; fill rate 1.0 for both), and
feature-build latency stayed sub-millisecond for both arms (~0.26-0.36ms
mean) - no serving-time regression. **Interpretation caveat** (§8's
existing caveat applies with extra force here): this dataset's session
generator correlates SEARCH -> CLICK -> ADD_TO_CART -> PURCHASE onto the
same product within a session (`scripts
.generate_backend_shaped_sqlite.py`), so a user's most recent interaction
is often mechanically predictive of their very next purchase in a way
real-world browsing may or may not replicate as strongly - this result
demonstrates the mechanism works and is directionally sound, not a
real-world-calibrated lift estimate.

`scripts/recency_diagnostics.py` prints the decay table for a fixed set
of ages plus 3 real SQLite users' category/brand affinity with recency on
vs. off, so the "recent dominates, old still contributes" property is
directly inspectable.

**Artifacts**: no Two-Tower/ranker artifacts were retrained for this
phase (see the evaluation scope note above) - `models/` is unchanged.
Product embeddings for this SQLite catalog are cached separately at
`data/processed/product_embeddings_sqlite.npz` (gitignored, regenerable),
never overwriting the synthetic V1 cache (`embedding.cache_path`).

## 15. Price-aware derived features (STEP 6)

**Non-negotiable ERD constraint**: every concept below (`effective_price`,
price tiers, category-relative price, the user price profile,
compatibility features) is derived ENTIRELY in the recommendation feature
layer (`features.price`) from `Product.Price`/`SalePrice`/
`DiscountPercentage` (plain backend data inputs) and purchase history.
NONE of it is written back to `Product`, `User`, `User_events`, or any
other backend/SQLite entity - no `price_tier`, `preferred_price`,
`price_sensitivity`, etc. column was added anywhere in the ERD.

**Price support that existed BEFORE this phase**: `ProductFeatures
.effective_price` (`sale_price if set else price`, no validity check) and
two Two-Tower/ranker numeric features (`normalized_price`,
`discount_fraction`/`item_discount_fraction`) - product-side only. There
was NO user-side price signal anywhere in the pipeline.

**`effective_price`** (`features.price.effective_price`): `sale_price`
when it's a VALID discount (`0 < sale_price < price`), else `price` -
stricter than the pre-existing formula (which only checked `sale_price is
not None`); a malformed `sale_price >= price` now safely falls back to
`price` instead of being treated as a discount. Not observed in
`data/sqlite/backend_shaped_synthetic.db` (0 malformed rows out of 466
discounted products, verified by inspection), but `Product`'s schema
doesn't forbid it.

**Product-side features added** (`ProductFeatures`, computed once over
the full catalog in `build_product_features`): `is_discounted` (bool),
`price_tier` (`"budget"`/`"mid"`/`"premium"` - catalog-wide TERTILES of
`effective_price`, `compute_catalog_tier_boundaries` - data-relative, not
arbitrary absolute cutoffs), `category_relative_price` (percentile rank
of `effective_price` WITHIN the product's own category, `0.5` for a
category of size 1 - "no meaningful comparison" possible - so the same
$20 item can rank very differently depending on category, e.g. $20
cereal vs. $20 meat).
All three default to neutral values (`False`/`"mid"`/`0.5`) on the
dataclass so every PRE-EXISTING call site/test that constructs a
`ProductFeatures` directly kept working unchanged.

**User price profile** (`features.price.UserPriceProfile`/
`build_user_price_profile`, one new `UserFeatures.price_profile` field,
`None` unless a caller supplies `price_context`): `typical_price`,
`price_spread` (population std - `0.0` for a single data point, never
NaN), `price_tier`, `supporting_purchase_count`, `fallback_source`.
**PURCHASE is the sole driver** - clicks/cart/search/chatbot are
deliberately NOT used as price evidence (a click on a $500 item doesn't
prove willingness to pay $500; only a purchase does - keeping this phase
simple per its own scope, not an oversight). Three-level, fully
deterministic fallback hierarchy:

  1. `"purchase_history"` - >=1 purchase with a resolvable price.
     `typical_price` is the RECENCY-WEIGHTED mean of those prices, reusing
     STEP 5's `features.recency.effective_weight`/`reference_time`
     UNCHANGED (no second recency implementation) - so a documented
     "recent spending shift" (e.g. purchases at ~20-30 six months ago,
     ~70-80 recently) pulls `typical_price` toward the recent cluster
     rather than averaging them away (proven in
     `tests/test_price.py::test_recent_purchases_outweigh_old_ones_when_recency_enabled`
     and shown live for a real SQLite user by `scripts
     /price_diagnostics.py`).
  2. `"preferred_category_prior"` - zero usable purchases, but the user
     has a `preferredCategory`: falls back to that category's catalog
     median price (catalog-only, never user behavior).
  3. `"catalog_prior"` - zero usable purchases AND no preferred category
     (the true NO_HISTORY case): falls back to the whole-catalog median.

  AgeGroup is deliberately NEVER used to infer price preference (no
  older-is-premium/younger-is-budget assumption) - it plays no role
  anywhere in this section.

**Historical-price limitation (important, honest gap)**:
`PurchaseRecord.unit_price` (the real transaction price) is populated
ONLY for the ERD-based synthetic path (`Order`/`OrderItem.UnitPrice`) -
the confirmed `User_events` contract (§4) carries no price at all, so a
`User_events`/SQLite-sourced purchase falls back to the product's CURRENT
`effective_price` as the best available proxy for "what did this cost."
This is not necessarily the price at the actual moment of purchase - only
matters if a product's price has since changed, which this static
catalog snapshot never does, but is documented rather than hidden (same
spirit as §12's "no timestamps" limitation).

**Leakage safety (temporal evaluation)**: `build_user_price_profile`
takes the SAME `reference_time`/`config.recency` STEP 5 already threads
through `build_user_features` - no separate wall-clock lookup, no second
leakage surface. Catalog-wide statistics (`PriceCatalogContext` - median/
std/tertile boundaries/per-category median/std) are built ONCE from the
product catalog ALONE (`build_price_catalog_context` takes only a
`products` list - verified by a signature-introspection test, not just
documented by convention) and are therefore never a leakage risk
regardless of evaluation cutoff - this project has exactly one static
catalog snapshot throughout, so "the snapshot available at evaluation
time" is trivially always the same snapshot.
`tests/test_temporal_future_purchase.py::test_price_profile_over_point_in_time_profile_never_sees_the_future_target`
proves the essential scenario end-to-end: an old purchase, a held-out
FUTURE purchase at a very different price, and a cutoff between them -
the future purchase's price has zero influence before its cutoff, and
correctly becomes usable history once the cutoff moves past it (§23's
repeat-purchase policy, unchanged).

**User x product compatibility features** (`features.price
.price_relative_distance` + a tier-match check, consumed by the ranker):
`price_relative_distance = |candidate_price - user_typical_price| /
max(user_typical_price, epsilon)` (scale-invariant) and `price_tier_match`
(coarse categorical agreement) - a deliberately SMALL, non-redundant set,
not every possible price-distance formulation. **No hand-coded "expensive
is better" rule exists anywhere** - price answers "is this price
compatible with this user's behavior," never "how expensive is this
candidate."

**Product/Item Tower** (`retrieval.two_tower`): `ITEM_NUMERIC_FEATURE_NAMES`
gained `category_relative_price`/`is_discounted` (+2). A NEW categorical
input, `price_tier_id`, was added with its OWN learned `Embedding`
(`config.two_tower.price_tier_embedding_dim`, default 8) over a FIXED
4-value vocabulary (`features.price.PRICE_TIERS` = budget/mid/premium +
an "unknown" bucket at index 0) - deliberately NOT an ordinal 0/1/2
number (a categorical tier has no numeric distance the model should be
forced to assume).

**User Tower**: `USER_NUMERIC_FEATURE_NAMES` gained
`normalized_typical_price` (+1, normalized by the SAME catalog
`max_price` the item tower already uses - consistent scale, no new
normalization stat). The user's derived `price_tier` shares the item
tower's exact `price_tier_id`/`Embedding` mechanism (same fixed
vocabulary). No separate "has price profile" numeric flag is needed on
this side - `price_tier_id`'s dedicated "unknown" bucket already
disambiguates "no `price_profile` at all" (an old/unwired call site) from
every REAL profile (which always gets a real budget/mid/premium tier,
even the catalog-prior fallback).

**Ranker** (`ranking.features`): gained `item_category_relative_price`,
`item_is_discounted` (item block), and `user_normalized_typical_price`,
`user_has_price_profile`, `price_relative_distance`, `price_tier_match`
(cross-feature block) - 6 new features. `user_has_price_profile` IS
needed here (unlike the towers) because the ranker has no embedding-based
"unknown" bucket for its plain numeric cross features - it disambiguates
"no data" (distance=0.0, flag=0.0) from "genuinely close" (distance=0.0,
flag=1.0).

**Exact dimensions - BEFORE -> AFTER**:

| Encoder | Before | After | Delta |
|---|---|---|---|
| Two-Tower item numeric (`item_numeric_dim`) | 7 | 9 | +2 |
| Two-Tower user numeric (`user_numeric_dim`) | 8 | 9 | +1 |
| Two-Tower item tower categorical inputs | category_id, brand_id | + price_tier_id | +1 input |
| Two-Tower user tower categorical inputs | preferred_category_id, age_group_id | + price_tier_id | +1 input |
| Ranker feature vector (`RANKING_FEATURE_NAMES`) | 23 | 29 | +6 |

**Model artifact compatibility**: because `item_numeric_dim`/
`user_numeric_dim`/the new `price_tier_id` input/`RANKING_FEATURE_NAMES`'
length all changed, the Two-Tower/ranker artifacts currently under
`models/` (trained against the OLD 7/8/23-dimensional shapes) are
DIMENSION-INCOMPATIBLE with the code as of this phase - loading them and
calling `.predict()` would raise a Keras shape-mismatch error (verified:
this is exactly what broke `RecommendationService`/live API/dashboard
would hit, NOT the automated test suite - every test that exercises a
real Keras Two-Tower/ranker model BUILDS one fresh via `build_user_tower`/
`build_item_tower`/`build_ranker_model` in the same test, so it always
matches the CURRENT encoder's dimensions; no test loads a stale
cross-run artifact). `models/` was intentionally left untouched (not
retrained, not mutated) - the NEXT phase is specifically scoped to
retrain Two-Tower/ranker/rebuild the ScaNN/FAISS index against this
SQLite catalog with the new price-aware feature set.
`TwoTowerFeatureEncoder.from_dict` degrades a pre-STEP-6 serialized
encoder dict missing the `price_tier_vocab` key to the same fixed
`PRICE_TIERS` vocabulary `fit()` always produces, rather than raising.

**Evaluation in this phase**: feature-level diagnostics only
(`scripts/price_diagnostics.py`: product examples, 5 real-user price-
profile examples spanning strong history / recent spending shift / one
purchase / engagement-no-purchase / NO_HISTORY, and 3 user x candidate
compatibility examples) - no retraining, no Top-N recommendation-quality
comparison (that requires the SAME retrained-Two-Tower/ranker work the
NEXT phase is scoped to do, per §8.1's precedent of deferring Top-N
evaluation until an appropriately-dimensioned model exists for this
catalog).

**Revenue-aware metrics**: NOT implemented this phase (deliberately -
see §35 of this phase's own spec: "prepare, don't optimize yet"). The
primary offline ground truth is UNCHANGED and remains held-out FUTURE
PURCHASE products (§8.1) - Precision/Recall/HitRate/NDCG/MRR@K continue
to treat every future purchase as equally "relevant" regardless of price;
a future Revenue@K or purchase-value-weighted metric would be a SEPARATE,
secondary business metric, never a redefinition of relevance.

**Limitations of synthetic price behavior**: this dataset's purchase
prices are drawn from a synthetic price-affinity model, not real
willingness-to-pay data - the price tiers/typical-price estimates this
phase produces are internally consistent and mechanically correct, but
(like every other V1 offline result, §8) they demonstrate PIPELINE
CORRECTNESS, not a real-world-calibrated price-sensitivity signal.

