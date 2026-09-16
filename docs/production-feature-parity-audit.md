# Production Feature-Parity Audit — SQLite Training vs. Real SQL Server vs. REST API Serving

**Date**: 2026-09-15. **Scope**: audit only, per explicit instruction. No retraining, no
model-artifact changes, no architecture changes were made. `main` was confirmed clean
(`git status`) before this audit and remains clean after it — the only change is this
file.

**Architecture under audit**:

```
SQLite   -> training + evaluation baseline   (data/sqlite/backend_shaped_synthetic.db)
SQL Server -> real production schema/data contract (verified against live tables below)
REST API -> live serving                     (GET /api/ai/products, /api/ai/user-activities,
                                                /api/categories, /api/reviews, /api/users/{guid})
```

**Headline finding**: this repo already contains a rigorous, code-verified prior audit of
almost this exact question — `docs/data-mapping.md` §19 (added/updated through
2026-09-15), specifically §19.12 ("Permanently unavailable: isActive, brand, list-level
tags"). This document independently re-derives and **confirms** that finding against the
verified real SQL Server schema supplied for this audit, and extends it: the prior audit
flagged 2 of the ranker's 29 features as unsafe (`brand_affinity_match`, `item_is_active`);
this audit's field-by-field pass finds **9 of 29** are unsafe or blocked, plus 5 more that
need reconciliation work before they can be trusted — see §5.

---

## 1. Baseline verification

- `git fetch` + `git status`: `main` up to date with `origin/main`, clean, at `5c531f7`.
- Full test suite run before any inspection changed anything:

  ```
  738 passed, 4 skipped, 0 failed, 68 warnings in 287.28s (0:04:47)
  ```

  The 4 skips are the documented ScaNN-only tests (`test_scann_index.py`,
  `test_step7_scann_sqlite_integration.py`, 2 cases in
  `test_eligibility_restricted_index.py`) — expected on native Windows, since ScaNN ships
  no Windows wheel (`configs/docker.yaml` is what actually exercises ScaNN, in CI/Docker).
- Current model artifacts on disk: `models/sqlite_baseline/{two_tower,ranker,vector_index}/`
  (the live-serving artifacts, `data_source: sqlite`) plus a legacy top-level
  `models/vector_index/scann_index`. **`models/backend_api/` does not exist** — confirmed
  by directory listing, matching `config.py`'s comment that no retrain against the real
  catalog has happened yet. `ranker/feature_names.json` confirms the ranker vector is
  exactly today's **29** named features (verbatim list used throughout this audit).

---

## 2. The real SQL Server contract, as verified

Given directly by you and cross-checked against this repo's own most recent live scan
(`docs/data-mapping.md` §19, dated 2026-09-15):

| Table | Columns | Notable *absences* vs. this repo's ERD assumptions |
|---|---|---|
| `Products` | Id, Slug, Name, Description, Price, StockQuantity, ProductImage, AltText, CreationDate, CategoryId, UpdatedAt | **No Brand. No isActive. No SalePrice. No DiscountPercentage. No Ingredients.** |
| `Categories` | Id, Name, ImageUrl, Slug, CreatedAt, UpdatedAt | **No ParentId — flat, no category hierarchy at all.** |
| `Users` | Id, Guid, FirstName, LastName, Email, PhoneNumber, BirthDate, HashedPassword, Role, IsActive, CreatedAt, UpdatedAt | **No PreferredCategory column. No AgeGroup column.** (A "favorite categories" join *does* exist and is exposed via `GET /api/users/{guid}`, see §7.) |
| `UserActivities` | Id, UserId, ProductId (nullable), ActionType, Timestamp | 8 raw `ActionType` values, not 5 — see §7. |
| `ProductReviews` | Id, UserId, ProductId, Rating, Comment (nullable), CreatedAt, UpdatedAt (nullable) | close match to canonical `Review`. |
| `ProductTags`/`Tags` | ProductId/TagId/Id; Id/Name/Slug | exist in the DB; wire/production exposure is disputed, see §9. |
| `Order`/`OrderItem` | full order + line-item snapshot incl. real historical `UnitPrice` | **structurally unused by the recommender on either side** — both the SQLite loader and the `backend_api` loader source purchases exclusively from the activity log to avoid double-counting (see §8). |

Live row counts you supplied (Products ~85, Categories ~6, Users ~547, UserActivities
~813,340, ProductReviews ~11, Order ~9,262, OrderItem ~9,387, ProductTags ~30, Tags ~8)
are a slightly later snapshot than the repo's own last live scan (85/6/513/318,965/11) —
both are internally consistent and simply reflect that `UserActivities` is a
high-write, growing table (SEARCH alone is documented as ~87% of all rows). **Treat any
row-count figure in this audit as a snapshot, not a fixed constant** — re-scan before
using volume as evidence of anything.

---

## 3. Complete feature inventory

### 3a. Two-Tower — Item Tower inputs (`retrieval/two_tower/feature_encoding.py`, `model.py`)

| Input | Kind | Computed from |
|---|---|---|
| `semantic_embedding` (384-d) | dense | Sentence-Transformer over `embeddings/text_builder.py` product text |
| `category_id` | learned embedding, vocab fit on catalog | `Product.category_name` |
| `brand_id` | learned embedding, vocab fit on catalog | `Product.brand` |
| `price_tier_id` | learned embedding, fixed 3-tier vocab | `Product.price`/`sale_price` tertile |
| `numeric[9]` | dense | `normalized_price, discount_fraction, log_purchase_count, log_cart_add_count, log_review_count, average_rating, has_rating, category_relative_price, is_discounted` |

### 3b. Two-Tower — User Tower inputs

| Input | Kind | Computed from |
|---|---|---|
| `semantic_embedding` (384-d) | dense | recency-weighted mean over interacted-product embeddings + search/chatbot free text |
| `preferred_category_id` | learned embedding, shares category vocab | `UserProfile.preferred_category` |
| `age_group_id` | learned embedding, dedicated vocab | `UserProfile.age_group` |
| `category_affinity` | dense vector, width = \|category_vocab\| | weighted event counts by category across all 5 signals |
| `brand_affinity` | dense vector, width = \|brand_vocab\| | weighted event counts by brand across all 5 signals |
| `price_tier_id` | learned embedding, shared tier vocab | `UserPriceProfile.price_tier` |
| `numeric[9]` | dense | `log_purchase_count, log_cart_item_count, log_search_count, log_total_engagement_events, has_chatbot_context, has_preferred_category, has_age_group, has_semantic_embedding, normalized_typical_price` |

Vocabularies (`category_vocab`, `brand_vocab`, `price_tier` buckets) are fit once from the
training catalog and serialized with the model (`two_tower/feature_encoder.json`); an
unseen category/brand at inference maps to a reserved "unknown" index 0.

### 3c. The exact 29 ranker features (`ranking/features.py::RANKING_FEATURE_NAMES`, verified against `models/sqlite_baseline/ranker/feature_names.json`)

| # | Feature | Raw source | Code |
|---:|---|---|---|
| 1 | `user_log_purchase_count` | PURCHASE events | `UserFeatures.purchase_count` |
| 2 | `user_log_cart_item_count` | ADD_TO_CART events | `UserFeatures.cart_item_count` |
| 3 | `user_log_search_count` | SEARCH events | `UserFeatures.search_count` |
| 4 | `user_log_total_engagement_events` | all 5 signals | `UserFeatures.total_engagement_events` |
| 5 | `user_has_chatbot_context` | CHATBOT events | `UserFeatures.has_chatbot_context` |
| 6 | `user_has_preferred_category` | `User.preferredCategory` | `UserFeatures.has_preferred_category` |
| 7 | `user_has_age_group` | `User.ageGroup` | `UserFeatures.has_age_group` |
| 8 | `item_normalized_price` | `Product.Price`/`SalePrice` | `ProductFeatures.effective_price` |
| 9 | `item_discount_fraction` | `Product.DiscountPercentage` | `ProductFeatures.discount_percentage` |
| 10 | `item_log_purchase_count` | PURCHASE events | `ProductFeatures.purchase_count` |
| 11 | `item_log_cart_add_count` | ADD_TO_CART events | `ProductFeatures.cart_add_count` |
| 12 | `item_log_review_count` | ProductReviews | `ProductFeatures.review_count` |
| 13 | `item_average_rating` | ProductReviews.Rating | `ProductFeatures.average_rating` |
| 14 | `item_has_rating` | derived | — |
| 15 | `item_log_stock_quantity` | `Product.StockQuantity` | `ProductFeatures.stock_quantity` |
| 16 | `item_is_active` | `Product.isActive` | `ranking/features.py:140` — `1.0 if product_features.is_active else 0.0` |
| 17 | `item_category_relative_price` | `Product.Price` within category | `ProductFeatures.category_relative_price` |
| 18 | `item_is_discounted` | `Product.SalePrice`/`DiscountPercentage` | `ProductFeatures.is_discounted` |
| 19 | `category_affinity_match` | user category_affinity × item category | cross |
| 20 | `brand_affinity_match` | `Product.brand` | `ranking/features.py:115` — `user_features.brand_affinity.get(product_features.brand or "", 0.0)` |
| 21 | `preferred_category_match` | `User.preferredCategory` × item category | cross |
| 22 | `semantic_cosine_similarity` | user/item embeddings | cross |
| 23 | `has_semantic_similarity` | derived | — |
| 24 | `user_normalized_typical_price` | `UserPriceProfile.typical_price` | cross |
| 25 | `user_has_price_profile` | derived | — |
| 26 | `price_relative_distance` | price profile × item price | cross |
| 27 | `price_tier_match` | price profile tier × item tier | cross |
| 28 | `retrieval_score` | VectorIndex search output | pipeline-internal |
| 29 | `retrieval_rank_normalized` | VectorIndex search output | pipeline-internal |

### 3d. Product embedding text (`embeddings/text_builder.py::build_product_text`)

Fields concatenated, in order, each conditionally skipped if empty:
`name` → `"Brand: {brand}."` → `"Category: {parent} > {category}."` → `"Tags: {tags}."` →
`description` → `"Ingredients: {ingredients}."`

### 3e. Eligibility (`serving/eligibility.py`)

Two rules, same policy object, applied as both a hard pre-retrieval gate and a final
lightweight validation: `is_active` (`pf.is_active`), `in_stock` (`pf.stock_quantity > 0`).

### 3f. Diversity / reranking (`reranking/diversity.py::apply_diversity`)

Penalizes repetition of `category_name` (`category_repetition_penalty=0.15`) and `brand`
(`brand_repetition_penalty=0.08`), scaled by `diversity_strength=0.5`.

### 3g. Cold-start / fallback (`serving/cold_start.py`, `serving/fallback.py`)

- `cold_start.py`: single input, `total_engagement_events` vs. config thresholds. **No
  brand dependency.**
- `fallback.py`: `global_popularity_ranking` (purchase/cart counts),
  `category_popularity_ranking` (category filter), `top_affinity_category`
  (`category_affinity` dict). **No brand dependency anywhere**, despite
  `UserFeatures.brand_affinity` existing — fallback candidate generation only ever uses
  category.

### 3h. Offline evaluation (`evaluation/*`)

Recall/Precision/NDCG/HitRate/MRR (`retrieval_metrics.py`), latency (`latency.py`),
catalog coverage / intra-list diversity / duplicate rate / fill rate / cold-start tier
distribution (`serving/evaluation.py`), and the temporal future-purchase protocol
(`evaluation/temporal_future_purchase.py`, `temporal_training.py`) — none of these compute
new raw features; they consume the same `ProductFeatures`/`UserFeatures`/ranker/Two-Tower
outputs audited above, so their parity status is inherited from §3a–3g, not independent.

---

## 4. `schemas/product.py` — is brand/is_active "synthetic-only" in the code itself?

No — nothing in the schema layer flags them as such. Both are plain, non-optional-in-spirit
fields:

```python
class Product(BaseModel):
    ...
    brand: str | None = None       # no synthetic-only marking anywhere in this file
    is_active: bool = True         # no synthetic-only marking anywhere in this file
    tags: list[str] = Field(default_factory=list)
    category_name: str | None = None
    parent_category_name: str | None = None   # <- also has no real-schema equivalent, see §5
```

The module docstring says this "mirrors the ERD's Product entity" — i.e. the code was
written against the *old* ERD (`docs/erd.jpeg`), which did list `Brand`/`isActive` on
`Product`. That ERD predates the now-verified real SQL Server contract, which has neither
column, nor `SalePrice`/`DiscountPercentage`/`Ingredients`, nor `Category.ParentId`. The
risk is entirely external to this file — nothing here would tell you, from reading it in
isolation, that these fields don't exist in production.

---

## 5. Ranker feature-by-feature parity verdict (all 29)

Legend: ✅ full parity · ⚠️ reconcilable (shape mismatch or proxy quality, not a hard block)
· ❌ unsafe (confirmed absent from the real schema, or contaminated by a field that is)

| # | Feature | Verdict | Why |
|---:|---|:-:|---|
| 1 | user_log_purchase_count | ✅ | PURCHASE is a real, live, high-volume signal |
| 2 | user_log_cart_item_count | ✅ | ADD_TO_CART real, live |
| 3 | user_log_search_count | ✅ | SEARCH real, live (86%+ of real activity volume) |
| 4 | user_log_total_engagement_events | ✅ | sum of real signals |
| 5 | user_has_chatbot_context | ⚠️ | field real, but real volume is ~8-16 events/1 user vs. 1,991/684 users in SQLite — see §8 |
| 6 | user_has_preferred_category | ⚠️ | real backend models this as a `FavoriteCategory[]` join via `GET /api/users/{guid}`, not a scalar column — needs a reduction rule (e.g. most-recent/first favorite) applied identically at train and serve time before this is truly reproducible |
| 7 | user_has_age_group | ❌ | real `Users`/`UserResponse` has **no age_group field at all** — permanently `None`/`False` in production, not merely unpopulated |
| 8 | item_normalized_price | ✅ | `Price` is real everywhere; `effective_price` collapses to plain `Price` in production since `SalePrice` doesn't exist, which is fine — feature stays valid, just loses its sale-aware branch |
| 9 | item_discount_fraction | ❌ | needs `SalePrice`/`DiscountPercentage` — no such columns in the real schema, always 0 in production |
| 10 | item_log_purchase_count | ✅ | real |
| 11 | item_log_cart_add_count | ✅ | real |
| 12 | item_log_review_count | ✅ | real (ProductReviews exists), but only 11 rows live — sparse data, not a code gap |
| 13 | item_average_rating | ✅ | same, sparse |
| 14 | item_has_rating | ✅ | derived |
| 15 | item_log_stock_quantity | ✅ | `StockQuantity` real |
| 16 | item_is_active | ❌ | no `isActive` column in real `Products` — always constant `True` in production; the real availability signal is `stock_quantity > 0`, already feature #15 |
| 17 | item_category_relative_price | ✅ | purely a function of `Price` + `CategoryId`, both real — no brand/discount dependency |
| 18 | item_is_discounted | ❌ | same root cause as #9 |
| 19 | category_affinity_match | ✅ | category is real; identity nuance: the API exposes category by `category_slug`, not numeric id — see §9 |
| 20 | brand_affinity_match | ❌ | no `Brand` column in real `Products`, permanently `None` |
| 21 | preferred_category_match | ⚠️ | same shape-mismatch as #6 |
| 22 | semantic_cosine_similarity | ❌ | mechanism is fine, but the embedding *space* was fit on text containing brand/tags/ingredients/parent-category segments that collapse to empty in production — systematic train/serve distribution shift, not a missing scalar (see §6) |
| 23 | has_semantic_similarity | ❌ | same root cause as #22 |
| 24 | user_normalized_typical_price | ⚠️ | computable (Price is real), but both SQLite and the real activity log lack a true historical unit price, so this always falls back to the product's *current* price as a proxy — a real historical source (`OrderItem.UnitPrice`) exists in SQL Server but is deliberately unused by the recommender on either side (§8) |
| 25 | user_has_price_profile | ⚠️ | same as #24 |
| 26 | price_relative_distance | ⚠️ | same as #24 |
| 27 | price_tier_match | ✅ | pure function of `Price`, no brand/discount dependency |
| 28 | retrieval_score | ❌ | inherits the Two-Tower embedding contamination described at #22 — the ranker feature itself is just "whatever the retrieval stage returned," but that stage's output is currently untrustworthy for the same reason |
| 29 | retrieval_rank_normalized | ❌ | same as #28 |

**Tally: 15 of 29 (52%) full parity · 5 of 29 (17%) reconcilable · 9 of 29 (31%) unsafe or
blocked.** This is a materially larger blast radius than the 2-of-29 the repo's existing
docs had previously enumerated (`brand_affinity_match`, `item_is_active`) — this audit adds:
the 3 discount/sale-price features (#9, #18, and #8's sale-aware branch), `user_has_age_group`
(#7), the 4 embedding-contaminated features (#22, #23, #28, #29), and flags 5 more (#6, #21,
#24, #25, #26) as needing reconciliation work rather than being drop-in safe.

---

## 6. Embedding / product-text parity

| Field in current template | Real `Products`/`Categories` equivalent | Verdict |
|---|---|---|
| `name` | `Products.Name` | ✅ |
| `brand` | none | ❌ permanently absent |
| `category` (+ **parent** category) | `Categories.Name` exists; **parent hierarchy does not** — real `Categories` has no `ParentId` | ✅ category name / ❌ parent segment |
| `tags` | `Tags`/`ProductTags` exist in the DB and are present on the live dev API wire response, but deliberately unconsumed by this codebase and disputed for production — see §9 | ⚠️ unresolved |
| `description` | `Products.Description`, confirmed present on the list-projection endpoint | ✅ |
| `ingredients` | none | ❌ permanently absent, no column at all |

**Safest production product-text template, using only fields verified available at both
training and serving**: **`name + description + category name`** (no parent hierarchy, no
brand, no tags, no ingredients). This matches the recommendation already reached
independently in `docs/data-mapping.md` §19.12 — this audit corroborates it and additionally
flags that `parent_category_name` and `ingredients` must also be dropped, since neither has
a real-schema equivalent at all (the prior doc's recommendation only explicitly named
brand/tags).

---

## 7. Eligibility parity

`is_active` — **not production-safe**: no such column exists on the real `Products` table.
The `backend_api` loader already hard-codes `is_active=True` for every product (there is no
"deleted"/"inactive" concept in the real schema at all). Real availability is expressed
purely through `StockQuantity`.

`in_stock` (`stock_quantity > 0`) — **production-safe**: `StockQuantity` is a real column,
fetched live via `GET /api/ai/products`.

**Practical consequence**: because `is_active` is hard-coded `True` for `backend_api`, the
`is_active` eligibility rule is already a functional no-op in that path — it never excludes
anything. It is not *causing* incorrect behavior today, but it is dead weight that implies a
distinction the production data cannot support, and the ranker's corresponding
`item_is_active` feature has zero variance for `backend_api` traffic (see §5, #16).
**Recommendation**: drop the `is_active` rule and rely on `stock_quantity > 0` alone; no
invented replacement field is needed since stock is already a genuine, sufficient signal.

---

## 8. Diversity / reranking parity — including a live operational risk

Category-based diversity is fully production-safe (`category_name` is real).

Brand-based diversity is **not production-safe**, and worse than merely "unused": because
`backend_api` sets every product's `brand` to `None`, `reranking/diversity.py`'s
`brand_counts.get(brand, 0)` buckets **every single candidate under the same `None` key**.
If `data_source: backend_api` serving were ever turned on without first zeroing
`brand_repetition_penalty`, the brand-diversity penalty would fire on every candidate pair
after the first — not "inert," but a silent, uniform diversity distortion across the whole
list. **Recommendation**: for any `backend_api`-sourced serving path, set
`reranking.brand_repetition_penalty: 0.0` (or remove the brand branch) rather than leaving
it at its current SQLite-tuned default. Category-only diversity is the safe production
alternative; no replacement attribute is needed since category already carries the
diversity signal.

---

## 9. User feature parity

| User feature | Reconstructable from real data? |
|---|---|
| category affinity | ✅ — activity log × `Product.category` |
| brand affinity | ❌ — no `Product.brand` |
| price preference (typical price / tier) | ⚠️ — computable, but only as a current-price proxy (see §5 #24) |
| recency | ✅ — `UserActivities.Timestamp` is real and populated |
| purchase frequency | ✅ — `PlaceOrder`→PURCHASE events, live (11,707+ rows observed) |
| interaction-type counts | ✅ — all 5 canonical types map from real `ActionType` values |
| review behavior | not currently a modeled user-side feature (only product-side `review_count`/`average_rating` exist); real `ProductReviews.UserId` would support one if added later, but this is out of the current inventory, not a parity gap |
| preferred category | ⚠️ — real equivalent is a join (`FavoriteCategory[]`), not a scalar; see §5 #6 |
| age group | ❌ — no real equivalent column at all |

---

## 10. SQLite vs. real SQL Server — raw schema diff

**Products** — common: `Id, CategoryId, Slug, Name, Description, Price, StockQuantity,
ProductImage, AltText`. SQLite-only (no real column): `Brand, SalePrice,
DiscountPercentage, Ingredients, isActive`. Real-only, unused by the recommender:
`CreationDate, UpdatedAt`.

**Categories** — common: `Id, Name`. SQLite-only: `ParentId` (real has **no** hierarchy at
all). Real-only, unused: `ImageUrl, Slug, CreatedAt, UpdatedAt`.

**Users** — common: `Id, FirstName, LastName, Email` (+ several unused fields on both
sides). SQLite-only: `PreferredCategoryId, AgeGroup` (as plain columns — the real
equivalent for preferred category is a join, and age group has no equivalent at all).
Real-only, unused: `Guid` (matters only for `backend_api` identity bridging, not features),
`PhoneNumber, BirthDate, HashedPassword, Role, IsActive, CreatedAt, UpdatedAt`.

**`User_events` (SQLite) vs. `UserActivities` (real)** — semantically aligned
(`id/user_id/product_id/action_time/action_type` ↔ `Id/UserId/ProductId/Timestamp/ActionType`).
Real `ProductId` is nullable at the schema level (an unresolved event *could* have no
product); SQLite's is generated `NOT NULL`, consistent with the documented policy that
only product-resolved events are ever logged — but this is a generator *assumption*, not a
DB-level guarantee, so the real loader path should defensively skip/ignore any row with a
null `ProductId` rather than assume it can't happen. Categorical values differ in
cardinality: SQLite generates exactly the 5 canonical action types directly; the real table
has 8 raw values, 3 of which (`AddedToFavorites`, `RemovedFromFavorites`, `RemoveFromCart`)
are currently mapped to `IGNORE` and never reach the model at all — a real signal type
(favoriting) is available and currently unused, worth a future V2 look, not a parity gap
in what's already used.

**Reviews** — close match: `Id/UserId/ProductId/Rating/Comment/CreatedAt` common; real adds
a nullable `UpdatedAt`, unused.

**Tags/ProductTags** — exist in both. SQLite wires them into canonical `Product.tags`; the
real backend's exposure is disputed (§ below) and, either way, deliberately left unconsumed
by `backend/loader.py` to avoid invalidating every trained product embedding.

**Order/OrderItem** — schemas diverge substantially and this is moot for the recommender:
neither the SQLite path nor the `backend_api` path ever reads `Order`/`OrderItem` (or
`Cart`/`Cart_Item`); `User_events`/`UserActivities` is the sole purchase-truth source on
both sides, by design, specifically to make double-counting structurally impossible.

**Net finding**: every SQLite-only column that actually feeds a feature (`Brand`,
`isActive`, `PreferredCategoryId`, `AgeGroup`, `Category.ParentId`, `SalePrice`,
`DiscountPercentage`) is a column the real SQL Server schema you supplied **does not have**
— confirmed absent at the database level, not merely missing from today's API projection.

---

## 11. SQLite vs. REST API serving parity

The `backend_api` adapter path (`backend/loader.py`) does not add or hide anything relative
to the raw DB — it faithfully reflects the same absences. Concretely, per
`backend/dtos.py`:

- `ApiProduct` (`GET /api/ai/products`): `slug, name, price, stock_quantity, category_slug,
  description, alt_text (always None on this route), product_image_url (always None on this
  route), creation_date, tags (present on the wire, never consumed), product_id`. **No
  brand. No isActive.**
- `ApiCategory` (`GET /api/categories`): `slug, name, image_url, created_at` — **no numeric
  id, no parent**; category identity on the wire is slug-based, unlike `Product`'s explicit
  `product_id`. This is worth flagging as its own reconciliation point: training fits a
  numeric `category_vocab` from `CategoryId`, serving resolves category by `slug` — the
  mapping layer needs to keep these consistent (this appears to already be handled via
  `category_slug` joins in `backend/mapping.py`, but is a coupling point to watch, not
  something this audit found broken).
- `ApiActivity` (`GET /api/ai/user-activities`): `user_id (GUID), action_type, slug (always
  None on this route), timestamp, product_id`.
- `ApiReview` (`GET /api/reviews`): `review_id, user_id, user_guid, product_id, rating,
  comment, created_at, updated_at`.
- `ApiUser` (`GET /api/users/{guid}`): `guid, first_name, last_name, email,
  preferred_categories: list[...], age_group`. `age_group` is always `None` — **no live
  schema equivalent exists at all**. `preferred_categories` is a real, live join array; the
  repo's own recent live probes found it empty on the samples checked so far — a
  data-population state to re-verify periodically, not a code gap.

Action-type mapping (`backend/mapping.py`), confirmed against all 8 real values from your
brief: `ViewProduct→CLICK, SearchProduct→SEARCH, AddToCart→ADD_TO_CART, PlaceOrder→PURCHASE,
Chatbot→CHATBOT, AddedToFavorites/RemovedFromFavorites/RemoveFromCart→IGNORE` (logged, not
silently dropped).

---

## 12. Full parity matrix

| Feature / Input | Pipeline stage | SQLite | SQL Server | REST API | Prod-safe? | Action |
|---|---|:-:|:-:|:-:|:-:|---|
| price | product/ranker/TT | yes | yes | yes | ✅ | keep |
| stock_quantity | eligibility/ranker | yes | yes | yes | ✅ | keep |
| category_name | TT/ranker/embed/diversity | yes | yes | yes (via slug) | ✅ | keep; watch slug/id vocab coupling |
| purchase_count / cart_add_count | product/user/ranker/TT | yes | yes | yes | ✅ | keep |
| search_count | user/ranker/TT | yes | yes | yes (majority of real volume) | ✅ | keep |
| review_count / average_rating | product/ranker | yes | yes | yes (sparse: ~11 rows) | ✅ (data sparse, not a code gap) | keep |
| click_count (ViewProduct→CLICK) | user/ranker/TT | yes | yes (schema) | mapped; historically near-zero live rows | ⚠️ | keep code path; re-validate once real CLICK volume accrues |
| chatbot_context | user/ranker/TT | yes (1,991 events/684 users — see §13) | yes (schema) | mapped; ~8-16 live rows, 1 user | ⚠️ | keep mechanism; treat weight as unvalidated |
| sale_price / discount_percentage / is_discounted / discount_fraction | product/TT/ranker | yes | **no such columns** | absent → constant 0/False | ❌ | drop, or accept as structurally-constant |
| is_active (eligibility + ranker #16) | eligibility/ranker | yes | **no such column** | absent → constant True | ❌ | replace with stock-only eligibility; drop `item_is_active` |
| brand / brand_id / brand_affinity / brand_affinity_match / brand diversity | TT/ranker/embed/diversity/user-features | yes | **no such column** | absent → None | ❌ | remove or replace with a real attribute (category, price tier) |
| age_group / age_group_id / user_has_age_group | TT/ranker | yes | **no such column** | confirmed absent from live schema | ❌ | drop entirely |
| parent_category_name | embeddings | yes (`ParentId`) | **no hierarchy at all** | absent | ❌ | drop from text template |
| ingredients | embeddings | not modeled the same way (varies) | **no such column** | absent | ❌ | drop from text template |
| tags (embed text) | embeddings | yes | yes (DB) | present on wire, unconsumed; production status disputed | ⚠️ | do not depend on |
| preferred_category / preferred_category_match | TT/ranker/cold-start | yes (scalar) | join shape only | present (join), empty on samples checked | ⚠️ | keep mechanism; reconcile scalar-vs-list reduction rule |
| unit_price (historical purchase price) | price profile | proxy only (both sides) | real column exists (`OrderItem.UnitPrice`) but unused by design | same proxy | ✅ (parity is honest — same proxy both sides) | keep; optionally exploit `OrderItem` later |
| semantic_cosine_similarity / retrieval_score / retrieval_rank_normalized | ranker | mechanism real | mechanism real | mechanism real | ❌ | mechanism fine, but embedding space is contaminated by brand/tag/ingredient/parent-category text — untrustworthy until re-embedded on a safe template |
| category_relative_price / price_tier_match | ranker | yes | yes | yes | ✅ | keep — pure functions of real `Price`+`CategoryId` |

---

## 13. Production-safe vs. unsafe vs. requiring-replacement — rollups

**✅ Safe for production as-is**: price, stock_quantity, category (flat, no hierarchy),
purchase/cart/search aggregate counts, review count/rating (sparse but real),
category_relative_price, price_tier_match, retrieval_score's *mechanism* (not its current
trained output — see below).

**⚠️ Sparse-but-structurally-safe (data-volume gap, not a code gap)**: click_count,
chatbot_context, review-derived features, preferred_category_match (mechanism real,
observed-empty so far).

**❌ Unsafe / requires replacement**: `brand`-anything (5 features + embedding text +
diversity), `is_active`/`item_is_active`, `sale_price`/`discount_percentage`-anything (3
features), `age_group`/`user_has_age_group`, `parent_category_name`, `ingredients`,
and — as a *consequence* of the above — `semantic_cosine_similarity`,
`has_semantic_similarity`, `retrieval_score`, `retrieval_rank_normalized` (untrustworthy
until the encoder is retrained on a clean text/vocab template, even though the mechanism
itself is fine).

**Synthetic-only fields** (SQLite has them; the real DB structurally does not):
`Product.Brand`, `Product.isActive`, `Product.SalePrice`, `Product.DiscountPercentage`,
`Product.Ingredients`, `Category.ParentId`, `User.PreferredCategoryId` (as a plain column),
`User.AgeGroup`.

**Real-DB-only fields** (not modeled in SQLite; irrelevant to the recommender):
`Users.Guid/BirthDate/IsActive`, `Categories.ImageUrl/Slug/UpdatedAt`,
`Products.CreationDate/UpdatedAt`, `ProductReviews.UpdatedAt`, `Tags.Slug`, the full
`Order`/`OrderItem` snapshot shape (including real historical `UnitPrice`, currently
unexploited by design on both sides).

**API-only nuance**: `tags` *are* present on the live dev backend's product wire response
(the repo's own probe found 23/83 products carrying non-empty tags), even though the
backend team's stated production position is "will never provide tags." This is an
explicitly unresolved discrepancy in the repo's own prior documentation, not a settled fact
either way — do not build a production decision on it until it's reconciled with the
backend team.

---

## 14. Exact SQLite changes needed, if SQLite remains the training source

1. Stop generating `Brand` (or generate it for narrative flavor only, never expose it to
   feature code) — no real equivalent, ever.
2. Stop generating variable `isActive` — hard-code `True` in the generator (matching what
   `backend_api` already returns) and make eligibility rely on `stock_quantity > 0` alone.
3. Stop generating `SalePrice`/`DiscountPercentage` variability, or accept that
   `item_discount_fraction`/`item_is_discounted` become structurally constant in a retrain,
   matching production.
4. Drop `Category.ParentId` / stop deriving `parent_category_name` — no hierarchy exists in
   production.
5. Drop `User.AgeGroup` — real schema has no such field at all, not just unpopulated; an
   `age_group_id` Two-Tower input trained on synthetic variety would map every real user to
   the same "unknown" bucket.
6. Reshape `User.PreferredCategoryId` from a single scalar into a small list
   (`favorite_category_ids: list[int]`) mirroring the real `FavoriteCategory[]` join, and
   apply the *same* single-value reduction rule (e.g., most-recently-added) at both train
   and serve time.
7. Rebuild the product-text embedding template to `name + description + category` only
   (drop brand/tags/ingredients/parent-category) and re-embed the full catalog.
8. Refit `category_vocab`/`brand_vocab` (or retire `brand_vocab` entirely) against a
   catalog sized closer to the real one (currently ~85 products/6 categories vs. SQLite's
   1,200/≈20+) — a straight reuse of the 1,200-item fitting risks a vocabulary the real,
   much smaller catalog never fully exercises.
9. Keep all 5 canonical action types (already present) but also generate the 3 currently-
   `IGNORE`d real action types (`AddedToFavorites`, `RemovedFromFavorites`, `RemoveFromCart`)
   so the SQLite dataset exercises `backend/mapping.py`'s full real vocabulary end-to-end,
   even while they remain unconsumed as signals.
10. Rebalance CHATBOT (and to a lesser extent CLICK) event density — SQLite's current
    684-user/769-product CHATBOT footprint is orders of magnitude denser than anything
    observed in real production traffic so far (see §13/§15) — not because volume itself is
    bad, but because training a real weight on a distribution this different from live
    reality risks overfitting to synthetic chatbot behavior patterns that don't exist yet.

**Missing real fields worth adding to SQLite for future-proofing** (not required for any
current feature): `Categories.ImageUrl/Slug`, `Products.CreationDate/UpdatedAt` (would
enable a future product-freshness feature), `Users.BirthDate` (a real field that *could*
seed a computed age-bucket if the team wants to keep an age dimension without relying on
a nonexistent `AgeGroup` column).

---

## 15. CHATBOT data coverage

| Source | Events | Distinct users | Distinct products |
|---|---|---|---|
| **Real SQL Server** (your brief / repo's latest live scan) | 16 (repo's own scan: 8) | 1 | 8 (repo's own scan: 5+) |
| **`data/sqlite/backend_shaped_synthetic.db`** (queried directly this audit) | **1,991** | **684** (of 1,000) | **769** (of 1,200) |
| Legacy ERD-based `data/synthetic/chatbot_records.json` | 126 records | ≤126 | ≤45 |

The SQLite baseline's CHATBOT signal is roughly two orders of magnitude denser in event
count, and covers ~680x more distinct users, than what real production data currently
shows — which is currently a single session from a single user. This is **not** a coverage
shortfall to fix by generating more synthetic data (volume is already generous); it's a
**representativeness** problem — the synthetic distribution (which users chat, which
products get mentioned, how often) has no real behavioral pattern to calibrate against yet,
because real usage is one anecdote. **Recommendation**: do not generate more CHATBOT
synthetic data right now; instead (a) keep `chatbot`-derived weights explicitly flagged as
unvalidated (matching the existing `click_weight`-style disclaimer already in
`configs/base.yaml`), and (b) re-audit this section once real `Chatbot` activity volume
grows past a handful of sessions from a handful of users — a "target" synthetic coverage
number isn't meaningful until there's real data to match it against.

---

## 16. Decision: can SQLite remain the training + evaluation baseline?

**Yes, conditionally.** The event-sourcing *architecture* already matches production
exactly: `sqlite.loader` reads only the activity-log table (`User_events`), never
`Cart`/`Order`/`Order_Item`, for the identical reason `backend/loader.py` does the same
against the real `UserActivities`/`Order`/`OrderItem` — structurally impossible
double-counting on both sides. What's misaligned is the **product/user schema underneath
that architecture**: SQLite currently carries 8 fields (`Brand`, `isActive`, `SalePrice`,
`DiscountPercentage`, `Ingredients`, `Category.ParentId`, `PreferredCategoryId` (shape),
`AgeGroup`) that the real SQL Server schema you supplied structurally does not have. None
of this is a new discovery this audit invented — `docs/data-mapping.md` §19.12 already
reached the same conclusion for brand/isActive/tags before this audit began; this audit's
contribution is (a) independently confirming it against your freshly-verified real schema,
field for field, and (b) quantifying the actual downstream blast radius as 9 (not 2) of the
ranker's 29 features, plus the embedding template and Two-Tower encoder.

If no: n/a — the answer is yes, with the §14 change list as the precondition.

---

## 17. Exact next step before retraining

1. Confirm with the backend team whether the schema is final — no plans to add
   Brand/isActive/AgeGroup/category-parent later — since that determines whether §14's
   changes are permanent or provisional, and resolve the tags discrepancy (§13, last
   bullet) one way or the other.
2. Apply the §14 SQLite generator/schema changes so a retrain against SQLite produces a
   feature contract `backend_api` can actually supply at serving time.
3. Rebuild the product-text template (`name + description + category` only) and re-embed
   the full catalog.
4. Refit Two-Tower/ranker vocabularies against a catalog sized closer to the real one (or
   at minimum stress-test robustness at ~85-product scale, not only 1,200).
5. Retrain Two-Tower + ranker + rebuild the ANN index under `models/backend_api/`. Ranker
   feature count would drop from 29 to as low as ~20-22 once the 9 unsafe features (§5) are
   removed outright, or fewer if the 5 reconcilable ones (§5) are addressed instead of cut.
6. Re-run the offline temporal report against real `backend_api`-sourced interactions
   (hundreds of thousands of PURCHASE/ADD_TO_CART/SEARCH rows already exist to evaluate
   against; CLICK and CHATBOT remain near-zero live and should be evaluated for graceful
   degradation, not for accuracy, until real volume accrues).

**None of the above was executed as part of this audit** — this is the audit's
recommendation only, exactly as instructed.

---

## 18. Current full test result

```
738 passed, 4 skipped, 0 failed, 68 warnings in 287.28s (0:04:47)
```

4 skips are the documented ScaNN-only tests, expected on native Windows. No code was
changed during this audit; `git status` remains clean on `main` at `5c531f7` apart from
this new file.

---

## 19. Implemented production-safe contract (2026-09-15 follow-up)

**Status: DONE — code + data contract only. No retraining, no artifact regeneration, no
deployment.** This section records what changed after §1–18's audit was reviewed and
acted on. The full before/after report (old/new feature lists, exact removed/reworked
fields, files changed, line counts, test results) lives in the PR/session notes for this
change; this section is the durable summary for future readers of this document.

**LEGACY SQLITE BASELINE (what `models/sqlite_baseline/` was trained against, and what
`data/sqlite/backend_shaped_synthetic.db` still physically contains)**: `Product.Brand`,
`Product.isActive` (as an independently-varying attribute), `Product.SalePrice`/
`DiscountPercentage`, `Product.Ingredients`, `Category.ParentId`, `User.PreferredCategoryId`
(a single FK), `User.AgeGroup`. These columns are **left physically in place** in the
SQLite database and in the canonical `Product`/`RawProduct`/`ProductFeatures` schemas —
removing them outright would break the still-supported legacy `data_source: "synthetic"`
path and cause unnecessary migration churn for no parity benefit (per §3's own
"leave-legacy-fields-present-when-safe" guidance). They are now explicitly documented as
**legacy/metadata-only** in each schema's docstring.

**NEW PRODUCTION-SAFE TRAINING CONTRACT (what every Two-Tower input, ranker feature,
product-text segment, eligibility rule, and diversity penalty now actually consumes)**:

- **Product text** (`embeddings.text_builder.build_product_text`): `name + description +
  category` only. Brand, parent category, tags, and ingredients removed.
- **Two-Tower item tower**: `semantic_embedding`, `category_id`, `price_tier_id`,
  7-D numeric (`normalized_price, log_purchase_count, log_cart_add_count,
  log_review_count, average_rating, has_rating, category_relative_price`). No `brand_id`,
  no `discount_fraction`/`is_discounted`.
- **Two-Tower user tower**: `semantic_embedding`, `category_affinity`, `price_tier_id`,
  8-D numeric (adds `has_preferred_category`, drops `has_age_group` vs. before). No
  `preferred_category_id` (folded into `category_affinity` instead — every favorite
  category contributes, not just one), no `age_group_id`, no `brand_affinity`.
- **Ranker**: 24 features (down from 29). Removed: `user_has_age_group`,
  `item_discount_fraction`, `item_is_active`, `item_is_discounted`,
  `brand_affinity_match`. Reassessed, not removed: `user_has_preferred_category` /
  `preferred_category_match` now mean "has ≥1 preferred category" / "candidate's category
  is one of the user's preferred categories" (list-based, real backend shape). Retained
  unchanged: `semantic_cosine_similarity`, `has_semantic_similarity`, `retrieval_score`,
  `retrieval_rank_normalized` — their mechanisms are still valid; only the inputs feeding
  the embeddings they consume changed, so they become trustworthy again once the
  Two-Tower is retrained on the new text/vocab (per this doc's own §5/§13 finding that
  deleting them outright would have been over-correction, not the audit's conclusion).
- **Eligibility** (`serving.eligibility`): `stock_quantity > 0` only. The `isActive`-based
  rule was removed, not just made a no-op.
- **Diversity** (`reranking.diversity`): category-only. The brand-repetition penalty was
  removed, not defaulted to zero — with `backend_api`, every candidate's `brand` was the
  same constant `None`, so leaving the old penalty in place would have actively distorted
  re-ranking (penalizing every pair as "same brand"), not merely done nothing.
- **`UserProfile.preferred_category: str | None` → `preferred_categories: list[str]`**:
  matches the real backend's `FavoriteCategory[]` join exactly. No arbitrary "pick the
  first favorite" reduction anywhere in the codebase — every favorite contributes to
  `category_affinity` (weight split evenly across favorites) and to the ranker's
  `preferred_category_match` (any-match, not first-match).
- **Artifact-version safety**: `TwoTowerFeatureEncoder.contract_version` is stamped on
  every newly-fit encoder and checked at startup
  (`serving.startup_validation.validate_two_tower_artifacts`); a pre-redesign artifact
  (still carrying `brand_vocab`/`age_group_vocab` keys) is rejected loudly, the same way
  `validate_ranker_artifacts` already rejected a `RANKING_FEATURE_NAMES` mismatch.

**Consequence**: `models/sqlite_baseline/` is now legacy-only and will fail
`ArtifactValidationError` at startup under the current code. A retrain (regenerate the
SQLite dataset if desired for a cleaner contract match, then `scripts/train_two_tower.py`
→ `scripts/train_ranker.py` → rebuild the ANN index) is required before `data_source:
"sqlite"` or `"backend_api"` live serving works again. This redesign did not perform that
retrain, per its own scope instructions.

## 20. `production_safe_v2`: zero learned-feature train-serve mismatch (2026-09-15 follow-up)

**Status: DONE — feature removal, retrain, live real-catalog ANN rebuild, live
verification, full test suite.** This follows §19's contract and the separate
activity-loading-architecture work (docs/data-mapping.md §19.13/19.14): once
`backend_api` serving gained a working per-user complete-history mechanism (19.14), an
audit of every remaining behavior-derived learned feature found ONE genuine, undisclosed
train-serve mismatch left: `item_log_purchase_count`/`item_log_cart_add_count` (both the
Two-Tower item tower and the ranker) were trained from the COMPLETE SQLite dataset's
purchase/cart history but could only be served from a BOUNDED recent-window
approximation of the real backend's 1.5M+-row activity table — the real backend has no
product-popularity/aggregate endpoint and no delta filter capable of reproducing a true
lifetime count efficiently (confirmed via live Swagger, 2026-09-15), and a one-time
15,000+-request full crawl was explicitly ruled out.

**Final production rule established**: every learned model input must be reproducible
exactly and efficiently from the live API. A feature that cannot satisfy this is removed
from the model, not silently approximated. Bounded/approximate values remain acceptable
only as a **serving heuristic** (a fallback ranking has no training semantics to be
unfaithful to) — never as a **model feature**.

**Two-Tower item tower**: numeric vector 7 → 5 (`log_purchase_count`/`log_cart_add_count`
removed; `normalized_price`, `log_review_count`, `average_rating`, `has_rating`,
`category_relative_price` remain). **Ranker**: 24 → 22 features
(`item_log_purchase_count`/`item_log_cart_add_count` removed; `item_log_stock_quantity` —
a real, catalog-native field, not a behavior-derived aggregate — is unaffected). No dummy
zero placeholder was kept for either removed slot — the vectors are genuinely
shorter, not zero-padded.

`ProductFeatures.purchase_count`/`cart_add_count` are NOT removed from the schema — they
remain, computed the same way, but their only remaining consumer is
`serving.fallback.global_popularity_ranking`/`category_popularity_ranking` (NO_HISTORY/
SPARSE_HISTORY fallback ordering). See `serving.fallback` and
`features.product_features.ProductFeatures`'s docstrings for the explicit MODEL FEATURE
vs SERVING HEURISTIC distinction this established — the same rule applies to any future
feature proposal: if live `backend_api` cannot reproduce it exactly and efficiently, it
may only ever be a serving heuristic, never a Two-Tower/ranker input.

**Contract version bumped**: `production_safe_v1` → `production_safe_v2`
(`retrieval.two_tower.feature_encoding.CURRENT_CONTRACT_VERSION`). `serving
.startup_validation.validate_two_tower_artifacts`/`validate_ranker_artifacts` already
compare `contract_version`/`feature_names` generically (no hard-coded dimension
numbers), so the prior `production_safe_v1`/24-feature artifacts are rejected
automatically with no validator code changes needed — verified by
`tests/test_production_safe_v2_feature_removal.py`.

**Retrained** (structural dimension change, so required — not optional): `models
/backend_api/` from scratch against the SAME `data/sqlite/production_aligned_training.db`
(unchanged; only the two global product-count inputs were removed, not the training
catalog/domain), same hyperparameters/protocol as before for a fair comparison. Temporal
leakage re-audited: 0 violations. Live real-catalog ANN rebuilt from `GET
/api/ai/products` (85 real `ProductId`s, `production_safe_v2`, no duplicates, no unknown
categories). `models/sqlite_baseline/` left untouched. Full before/after metrics, live
verification, and file/test lists are in the session report for this change.
