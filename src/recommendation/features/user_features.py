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

No recency: purchases/cart/search/chatbot are aggregated as unweighted-by-
time counts/means (docs/data-mapping.md section 1/7) - `age_group` is
carried through as an opaque label, never given hard-coded semantics
(section 2).

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

import numpy as np

from recommendation.data.schemas.engagement import EngagementProfile, SearchRecord
from recommendation.data.schemas.product import Product
from recommendation.utils.config import FeatureConfig


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
    """Weighted average over only the signal components a user actually
    has data for (missing signals are omitted, not zero-padded), so a
    user with only a chatbot summary isn't diluted toward a zero vector
    for the purchase/cart/search components they lack.
    """
    present = [(vec, w) for vec, w in components if w > 0]
    if not present:
        return None
    vectors = np.stack([v for v, _ in present])
    weights = np.array([w for _, w in present], dtype=np.float64)
    weights = weights / weights.sum()
    return (vectors * weights[:, None]).sum(axis=0).astype(np.float32)


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
) -> UserFeatures:
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
        if product.category_name:
            category_counts[product.category_name] += config.click_weight
        if product.brand:
            brand_counts[product.brand] += config.click_weight

    for p in purchases:
        product = product_lookup.get(p.product_id)
        if product is None:
            continue
        if product.category_name:
            category_counts[product.category_name] += config.purchase_weight * p.quantity
        if product.brand:
            brand_counts[product.brand] += config.purchase_weight * p.quantity

    for c in cart_items:
        product = product_lookup.get(c.product_id)
        if product is None:
            continue
        if product.category_name:
            category_counts[product.category_name] += config.cart_weight * c.quantity
        if product.brand:
            brand_counts[product.brand] += config.cart_weight * c.quantity

    for s in searches:
        signal_product_id = _search_signal_product_id(s)
        if signal_product_id is None:
            continue
        product = product_lookup.get(signal_product_id)
        if product is None:
            continue
        if product.category_name:
            category_counts[product.category_name] += config.search_weight
        if product.brand:
            brand_counts[product.brand] += config.search_weight

    if chatbot is not None and not chatbot_content_excluded:
        if chatbot.preferred_category:
            category_counts[chatbot.preferred_category] += config.chatbot_weight
        for pid in chatbot_mentions:
            product = product_lookup.get(pid)
            if product is None:
                continue
            if product.category_name:
                category_counts[product.category_name] += config.chatbot_weight
            if product.brand:
                brand_counts[product.brand] += config.chatbot_weight

    if profile.profile.preferred_category:
        category_counts[profile.profile.preferred_category] += config.preferred_category_weight

    category_affinity = _normalize_and_truncate(category_counts, config.max_top_categories)
    brand_affinity = _normalize_and_truncate(brand_counts, config.max_top_brands)

    # --- semantic embedding --------------------------------------------
    click_vectors = [product_embeddings[cl.product_id] for cl in clicks if cl.product_id in product_embeddings]
    purchase_vectors = [product_embeddings[p.product_id] for p in purchases if p.product_id in product_embeddings]
    cart_vectors = [product_embeddings[c.product_id] for c in cart_items if c.product_id in product_embeddings]
    search_vectors = [
        product_embeddings[pid]
        for s in searches
        if (pid := _search_signal_product_id(s)) is not None and pid in product_embeddings
    ] + [
        text_embeddings[s.search_term]
        for s in searches
        if s.matched_product_id is None and s.search_term in text_embeddings and not _search_text_is_leakage_risky(s)
    ]
    chatbot_vectors = [product_embeddings[pid] for pid in chatbot_mentions if pid in product_embeddings]
    if chatbot is not None and not chatbot_content_excluded and chatbot.summary and chatbot.summary in text_embeddings:
        chatbot_vectors.append(text_embeddings[chatbot.summary])

    components: list[tuple[np.ndarray, float]] = []
    if click_vectors:
        components.append((np.mean(click_vectors, axis=0), config.click_weight))
    if purchase_vectors:
        components.append((np.mean(purchase_vectors, axis=0), config.purchase_weight))
    if cart_vectors:
        components.append((np.mean(cart_vectors, axis=0), config.cart_weight))
    if search_vectors:
        components.append((np.mean(search_vectors, axis=0), config.search_weight))
    if chatbot_vectors:
        components.append((np.mean(chatbot_vectors, axis=0), config.chatbot_weight))

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
