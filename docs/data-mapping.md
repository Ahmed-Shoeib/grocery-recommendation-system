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
`SearchRecord.action_time` from `UserInteraction.action_time`) so a V2
temporal train/validation/test split, recency weighting, or event-
sequence feature doesn't require another data-contract change. Per §7/§12,
V1 feature engineering does NOT derive any recency/decay signal from these
timestamps, and this architecture change does not add one - only
preservation, not consumption, changed.

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

**Status of price/timestamp readiness (see the SQLite inspection report
from this phase, not reproduced here):** `Product.Price`/`SalePrice`/
`DiscountPercentage` are correctly exposed through the canonical `Product`
schema, but no price-aware feature consumes them yet. `action_time` is
parsed into real `datetime` values and survives the adapter boundary
exactly as it does for the synthetic path, but is likewise not consumed
for recency/temporal-splitting/time-of-day purposes yet - both remain
explicitly deferred, matching §7/§12's existing scope boundary.

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
- Recency / time-decay features and freshness scoring
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
the canonical schemas** precisely so V2 can add recency features without
another schema migration - but V1 feature engineering must not derive any
recency/decay signal from them. This is enforced by convention now and
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
| §9 Surface context hook / dashboard architecture | Phase 7 (pipeline parameter, not yet exposed via API) / Phase 9 (dashboard reuses RecommendationService in-process, not via HTTP) |
| §10 VectorIndex backends | Phase 5 (ScaNN primary/Docker + FAISS Windows dev fallback) |
| §11 Neural ranking (VectorIndex candidates, richer cross features, baseline comparison) | Phase 6 |
| §12 Leakage-limitation mitigation | Phase 3 (guard) / Phase 4 (consumer) / Phase 6 (extended to ranking negatives) / User_events contract change (extended to clicks) |
