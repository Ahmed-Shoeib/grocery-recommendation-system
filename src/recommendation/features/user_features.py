"""User feature engineering from the canonical EngagementProfile.

Combines the five V1 signals (clicks, previous purchases, add-to-cart
behavior, searched items, chatbot context) plus the confirmed
`preferredCategory` attribute into:

  - `category_affinity` / `brand_affinity`: normalized distributions,
    blended across signals using `config.features.*_weight` (never
    hard-coded - docs/data-mapping.md section 2/3).
  - `semantic_embedding`: a single vector in the same space as product
    Sentence Transformer embeddings, built by weighted-averaging the
    embeddings of products the user purchased/carted/searched-and-matched,
    plus encoding free-text search terms and the chatbot summary directly.
    This gives the (Phase 4) User Tower direct access to "what does this
    user's history mean semantically" in the same space the Item Tower
    already operates in.

Recency weighting (docs/data-mapping.md section 14): every per-event
contribution to `category_affinity`/`brand_affinity`/`semantic_embedding`
is scaled by `features.recency.effective_weight(base_signal_weight,
event_time, reference_time, config.recency)` -
`effective_weight = signal_weight * 0.5 ** (age_days / half_life_days)` -
so a more recent interaction contributes more strongly than an equally-
weighted older one, without deleting old history. `reference_time` is the
point in time recency is measured against - the caller-supplied
evaluation cutoff for offline (temporal future-purchase) evaluation, or
"now" for live serving - see `build_user_features`'s docstring. An event
with no timestamp (the old ERD-based synthetic generators for click/cart/
search/chatbot) falls back to a neutral (unweighted) contribution rather
than being dropped - see `features.recency.effective_weight`. Raw event
*counts* (`click_count`, `purchase_count`, `total_engagement_events`, ...)
are NEVER recency-weighted - they stay exact historical counts so cold-
start tiering (`serving.cold_start`) keeps meaning "how much history does
this user have," not "how much RECENT history." `age_group` is carried
through as an opaque label, never given hard-coded semantics (section 2).

Target-leakage guard: `exclude_product_ids` lets a caller building a
(user, candidate_product) training example (Phase 4) strip that
candidate's own influence out of the user's own history-derived features
first (leave-one-out) - so "the user bought exactly this product" can't
leak into a feature meant to *predict* that same purchase.

Known limitation (see docs/data-mapping.md, "V1 leakage limitation - no
timestamps"): without timestamps we cannot tell whether a search or
chatbot interaction happened before or after a given purchase, so
`matched_product_id`/`mentioned_product_ids` exclusion alone isn't
airtight - free text can reference an excluded product without a resolved
product id. This module handles that conservatively: an unmatched search
whose `search_term` textually contains an excluded product's name is
dropped from the semantic-embedding signal, and a chatbot record is
dropped *entirely* (summary embedding included, not just the flagged
mention) if it mentions an excluded product by id or by name in its
summary text - rather than trying to salvage the "safe" remainder of that
conversation. Clicks carry no free text at all (`ClickRecord`'s entire
content IS the product reference, like `PurchaseRecord`/
`CartAffinityRecord`), so excluding a product simply drops the matching
click record(s) - no separate text-leakage heuristic is needed for it.

CLICK is a newly-added signal (`config.features.click_weight`, currently
a small, unbenchmarked placeholder - see `FeatureConfig` docstring). It
deliberately does NOT introduce a new numeric input slot on the Two-Tower
or ranker feature vectors (`retrieval.two_tower.feature_encoding`,
`ranking.features`): those vectors have fixed dimensions baked into
already-trained model artifacts, and adding a slot there would silently
break loading them. Clicks instead flow only through the same
vocabulary-/embedding-dimensioned features every other signal already
uses (`category_affinity`, `brand_affinity`, `semantic_embedding`) and
through `total_engagement_events` (an existing numeric slot whose *value*,
not shape, now also reflects clicks) - see docs/data-mapping.md section 4.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from recommendation.data.schemas.engagement import EngagementProfile, SearchRecord
from recommendation.data.schemas.product import Product
from recommendation.features.recency import effective_weight
from recommendation.utils.config import FeatureConfig, RecencyConfig


@dataclass
class UserFeatures:
    user_id: int

    # Both confirmed backend User attributes (docs/data-mapping.md section
    # 2), both Optional with a matching has_* presence flag for defensive
    # handling. They're treated differently downstream on purpose:
    # preferred_category is also folded into category_affinity below (it's
    # a direct category signal); age_group is carried through untouched as
    # an opaque categorical label - it has no category/brand mapping, so
    # Phase 4's User Tower consumes it as its own learned embedding lookup
    # rather than blending it into an affinity distribution.
    preferred_category: str | None
    age_group: str | None
    has_preferred_category: bool
    has_age_group: bool

    click_count: int
    purchase_count: int
    distinct_products_purchased: int
    cart_item_count: int
    search_count: int
    has_chatbot_context: bool
    total_engagement_events: int  # sum of the five V1 signal event counts; Phase 7 thresholds this for cold-start tiering

    category_affinity: dict[str, float] = field(default_factory=dict)
    brand_affinity: dict[str, float] = field(default_factory=dict)
    semantic_embedding: np.ndarray | None = None


def _normalize_and_truncate(counts: Counter, top_k: int) -> dict[str, float]:
    if not counts:
        return {}
    top = counts.most_common(top_k)
    total = sum(v for _, v in top)
    if total <= 0:
        return {}
    return {k: v / total for k, v in top}


def _weighted_average(components: list[tuple[np.ndarray, float]]) -> np.ndarray | None:
    """Weighted average over only the (vector, effective_weight) components
    a user actually has data for (missing signals contribute zero
    components, not a zero-padded placeholder), so a user with only a
    chatbot summary isn't diluted toward a zero vector for the purchase/
    cart/search components they lack. Generic over what one "component"
    represents - see `_signal_embedding_component` below, which is what
    turns a whole signal's (product-embedding, event_time) pairs into the
    single recency-weighted `(vector, effective_signal_weight)` component
    this function actually consumes.
    """
    present = [(vec, w) for vec, w in components if w > 0]
    if not present:
        return None
    vectors = np.stack([v for v, _ in present])
    weights = np.array([w for _, w in present], dtype=np.float64)
    weights = weights / weights.sum()
    return (vectors * weights[:, None]).sum(axis=0).astype(np.float32)


def _signal_embedding_component(
    vectors_with_times: list[tuple[np.ndarray, datetime | None]],
    base_weight: float,
    reference_time: datetime | None,
    recency_config: RecencyConfig,
) -> tuple[np.ndarray, float] | None:
    """Turns one signal's per-event `(product/text vector, event_time)`
    pairs into a single `_weighted_average` component:
    `(recency-weighted mean vector, effective_signal_weight)`.

    Two separate uses of recency, deliberately kept distinct:
      - CONTENT: the mean vector is itself a per-event recency-weighted
        average, so which particular items dominate the signal's blended
        vector shifts toward recently-interacted ones (docs/data-mapping.md
        section 14's "recent Coffee CLICK should contribute more" example).
      - MAGNITUDE: `effective_signal_weight = base_weight * mean(per-event
        recency weights)` - the AVERAGE (not sum) of this signal's
        per-event recency weights, so this signal's overall blend weight
        shifts toward `base_weight` when its history is fresh and shrinks
        when it's stale, WITHOUT becoming sensitive to how many events the
        signal has (an average is invariant to event count, matching the
        pre-recency behavior where one component == one fixed
        `base_weight` regardless of event volume - see `_weighted_average`
        docstring). When recency is disabled, or every event in this
        signal has no timestamp, every per-event weight is exactly 1.0, so
        this reduces to the original plain-mean-of-vectors /
        `component_weight=base_weight` behavior byte-for-byte.
    """
    if not vectors_with_times:
        return None
    weights = np.array(
        [effective_weight(1.0, t, reference_time, recency_config) for _, t in vectors_with_times],
        dtype=np.float64,
    )
    vectors = np.stack([v for v, _ in vectors_with_times])
    weight_sum = weights.sum()
    if weight_sum <= 0:
        mean_vector = vectors.mean(axis=0)
    else:
        mean_vector = (vectors * (weights / weight_sum)[:, None]).sum(axis=0)
    effective_signal_weight = base_weight * float(weights.mean())
    return mean_vector.astype(np.float32), effective_signal_weight


def _normalize_text(text: str) -> str:
    return text.strip().lower()


def _text_mentions_excluded_product(
    text: str | None, exclude_product_ids: frozenset[int], product_lookup: dict[int, Product]
) -> bool:
    """Conservative (no-timestamp) leakage check: True if `text` contains
    an excluded product's name as a substring, even though it wasn't
    linked via `matched_product_id`. See module docstring.
    """
    if not exclude_product_ids or not text:
        return False
    normalized = _normalize_text(text)
    return any(
        (product := product_lookup.get(pid)) is not None and _normalize_text(product.name) in normalized
        for pid in exclude_product_ids
    )


def build_user_text_embeddings(profiles: list[EngagementProfile], encoder) -> dict[str, np.ndarray]:
    """Batch-encode every unique free-text string needed for semantic user
    features (search terms with no matched product + chatbot summaries) in
    one pass. Keyed by raw text so `build_user_features` looks values up
    instead of re-encoding per user/per call.
    """
    texts: set[str] = set()
    for profile in profiles:
        for s in profile.searches:
            if s.matched_product_id is None:
                texts.add(s.search_term)
        if profile.chatbot_context and profile.chatbot_context.summary:
            texts.add(profile.chatbot_context.summary)
    if not texts:
        return {}
    ordered = sorted(texts)
    vectors = encoder.encode(ordered)
    return dict(zip(ordered, vectors))


def build_user_features(
    profile: EngagementProfile,
    product_lookup: dict[int, Product],
    product_embeddings: dict[int, np.ndarray],
    config: FeatureConfig,
    text_embeddings: dict[str, np.ndarray] | None = None,
    exclude_product_ids: frozenset[int] = frozenset(),
    reference_time: datetime | None = None,
) -> UserFeatures:
    """`reference_time` is the point in time recency (`config.recency`) is
    measured against - see `features.recency` and docs/data-mapping.md
    section 14. Recency is strictly OPT-IN per call: omitting
    `reference_time` (the default) makes every recency-weighted
    contribution neutral (`effective_weight == base_weight`), regardless
    of `config.recency.enabled` - see `features.recency.effective_weight`.
    There is no implicit wall-clock fallback here.

      - Offline (temporal future-purchase) evaluation: the caller MUST
        pass the evaluation cutoff (e.g.
        `evaluation.temporal_future_purchase.TemporalUserSplit.val_cutoff`/
        `test_cutoff`) - the profile passed in is already point-in-time
        truncated to that same cutoff
        (`build_point_in_time_engagement_profile`), and passing anything
        else here would let recency be computed relative to a time the
        evaluation example isn't supposed to know about.
      - Live serving: the request boundary (`serving.pipeline`) passes
        `datetime.now()` explicitly.
      - The pre-existing, non-temporal leave-one-out training path
        (`retrieval.two_tower.examples`, `ranking.examples`) deliberately
        never passes `reference_time` - those per-training-example calls
        have no natural per-example "now," and recency must not silently
        introduce wall-clock-relative drift into a protocol with no
        temporal semantics of its own (this was caught by a training
        pipeline regression test - see git history for this phase).
    """
    text_embeddings = text_embeddings or {}

    # Leave-one-out exclusion rule: a PurchaseRecord/CartAffinityRecord's
    # entire content IS the product reference, so excluding the product
    # drops the whole record (event count included). A SearchRecord and
    # the chatbot context carry information independent of any one
    # product (the search term text, the chatbot summary, other mentioned
    # products), so those records/events are kept - only the specific
    # excluded product's contribution to category/brand/embedding signals
    # is scrubbed. This is what makes leave-one-out correct without also
    # fabricating a different search/chatbot interaction history.
    clicks = [c for c in profile.clicks if c.product_id not in exclude_product_ids]
    purchases = [p for p in profile.purchases if p.product_id not in exclude_product_ids]
    cart_items = [c for c in profile.cart_items if c.product_id not in exclude_product_ids]
    searches = profile.searches
    chatbot = profile.chatbot_context

    def _search_signal_product_id(search: SearchRecord) -> int | None:
        if search.matched_product_id is None or search.matched_product_id in exclude_product_ids:
            return None
        return search.matched_product_id

    def _search_text_is_leakage_risky(search: SearchRecord) -> bool:
        return search.matched_product_id is None and _text_mentions_excluded_product(
            search.search_term, exclude_product_ids, product_lookup
        )

    # Conservative chatbot handling (see module docstring): if this
    # chatbot record mentions an excluded product by id OR by name in its
    # summary text, drop its ENTIRE content contribution (all mentions +
    # the summary embedding), not just the flagged product - we can't
    # separate "safe" from "risky" parts of one conversation without a
    # timestamp telling us it happened before the target interaction.
    chatbot_content_excluded = chatbot is not None and (
        any(pid in exclude_product_ids for pid in chatbot.mentioned_product_ids)
        or _text_mentions_excluded_product(chatbot.summary or "", exclude_product_ids, product_lookup)
    )
    chatbot_mentions = [] if chatbot_content_excluded or chatbot is None else list(chatbot.mentioned_product_ids)

    # --- category / brand affinity -----------------------------------
    category_counts: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()

    for cl in clicks:
        product = product_lookup.get(cl.product_id)
        if product is None:
            continue
        w = effective_weight(config.click_weight, cl.action_time, reference_time, config.recency)
        if product.category_name:
            category_counts[product.category_name] += w
        if product.brand:
            brand_counts[product.brand] += w

    for p in purchases:
        product = product_lookup.get(p.product_id)
        if product is None:
            continue
        w = effective_weight(config.purchase_weight, p.order_created_at, reference_time, config.recency) * p.quantity
        if product.category_name:
            category_counts[product.category_name] += w
        if product.brand:
            brand_counts[product.brand] += w

    for c in cart_items:
        product = product_lookup.get(c.product_id)
        if product is None:
            continue
        w = effective_weight(config.cart_weight, c.action_time, reference_time, config.recency) * c.quantity
        if product.category_name:
            category_counts[product.category_name] += w
        if product.brand:
            brand_counts[product.brand] += w

    for s in searches:
        signal_product_id = _search_signal_product_id(s)
        if signal_product_id is None:
            continue
        product = product_lookup.get(signal_product_id)
        if product is None:
            continue
        w = effective_weight(config.search_weight, s.action_time, reference_time, config.recency)
        if product.category_name:
            category_counts[product.category_name] += w
        if product.brand:
            brand_counts[product.brand] += w

    if chatbot is not None and not chatbot_content_excluded:
        # Whole-chatbot-record recency: see `ChatbotContextRecord.action_time`
        # docstring - one timestamp (most recent resolved mention) governs
        # every part of this chatbot record's contribution (preferred
        # category, mentions, summary embedding alike), since per-mention
        # timestamps aren't preserved.
        chatbot_w = effective_weight(config.chatbot_weight, chatbot.action_time, reference_time, config.recency)
        if chatbot.preferred_category:
            category_counts[chatbot.preferred_category] += chatbot_w
        for pid in chatbot_mentions:
            product = product_lookup.get(pid)
            if product is None:
                continue
            if product.category_name:
                category_counts[product.category_name] += chatbot_w
            if product.brand:
                brand_counts[product.brand] += chatbot_w

    if profile.profile.preferred_category:
        # A standing profile attribute, not a timestamped interaction -
        # never recency-weighted (docs/data-mapping.md section 14).
        category_counts[profile.profile.preferred_category] += config.preferred_category_weight

    category_affinity = _normalize_and_truncate(category_counts, config.max_top_categories)
    brand_affinity = _normalize_and_truncate(brand_counts, config.max_top_brands)

    # --- semantic embedding --------------------------------------------
    # Each signal's vectors are paired with their own event_time so
    # `_signal_embedding_component` can recency-weight both WHICH items
    # dominate within the signal and how strongly the signal's whole
    # component counts in the cross-signal blend - see that function's
    # docstring.
    click_vectors_t = [
        (product_embeddings[cl.product_id], cl.action_time) for cl in clicks if cl.product_id in product_embeddings
    ]
    purchase_vectors_t = [
        (product_embeddings[p.product_id], p.order_created_at) for p in purchases if p.product_id in product_embeddings
    ]
    cart_vectors_t = [
        (product_embeddings[c.product_id], c.action_time) for c in cart_items if c.product_id in product_embeddings
    ]
    search_vectors_t = [
        (product_embeddings[pid], s.action_time)
        for s in searches
        if (pid := _search_signal_product_id(s)) is not None and pid in product_embeddings
    ] + [
        (text_embeddings[s.search_term], s.action_time)
        for s in searches
        if s.matched_product_id is None and s.search_term in text_embeddings and not _search_text_is_leakage_risky(s)
    ]
    chatbot_vectors_t: list[tuple[np.ndarray, datetime | None]] = []
    if chatbot is not None and not chatbot_content_excluded:
        chatbot_vectors_t = [
            (product_embeddings[pid], chatbot.action_time) for pid in chatbot_mentions if pid in product_embeddings
        ]
        if chatbot.summary and chatbot.summary in text_embeddings:
            chatbot_vectors_t.append((text_embeddings[chatbot.summary], chatbot.action_time))

    components: list[tuple[np.ndarray, float]] = []
    for vectors_t, base_weight in (
        (click_vectors_t, config.click_weight),
        (purchase_vectors_t, config.purchase_weight),
        (cart_vectors_t, config.cart_weight),
        (search_vectors_t, config.search_weight),
        (chatbot_vectors_t, config.chatbot_weight),
    ):
        component = _signal_embedding_component(vectors_t, base_weight, reference_time, config.recency)
        if component is not None:
            components.append(component)

    semantic_embedding = _weighted_average(components)

    total_events = len(clicks) + len(purchases) + len(cart_items) + len(searches) + (1 if chatbot is not None else 0)

    return UserFeatures(
        user_id=profile.user_id,
        preferred_category=profile.profile.preferred_category,
        age_group=profile.profile.age_group,
        has_preferred_category=profile.profile.preferred_category is not None,
        has_age_group=profile.profile.age_group is not None,
        click_count=len(clicks),
        purchase_count=len(purchases),
        distinct_products_purchased=len({p.product_id for p in purchases}),
        cart_item_count=len(cart_items),
        search_count=len(searches),
        has_chatbot_context=chatbot is not None,
        total_engagement_events=total_events,
        category_affinity=category_affinity,
        brand_affinity=brand_affinity,
        semantic_embedding=semantic_embedding,
    )
