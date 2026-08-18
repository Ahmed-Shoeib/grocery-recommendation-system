from datetime import datetime, timedelta

import numpy as np
import pytest

from recommendation.data.schemas.engagement import (
    CartAffinityRecord,
    ChatbotContextRecord,
    ClickRecord,
    EngagementProfile,
    PurchaseRecord,
    SearchRecord,
)
from recommendation.data.schemas.product import Product
from recommendation.data.schemas.user import UserProfile
from recommendation.features.user_features import build_user_features, build_user_text_embeddings
from recommendation.utils.config import FeatureConfig, RecencyConfig


def _products() -> dict[int, Product]:
    return {
        1: Product(id=1, category_id=1, slug="a", name="Greek Yogurt", price=4.0, category_name="Dairy & Eggs", brand="GreenValley"),
        2: Product(id=2, category_id=2, slug="b", name="Potato Chips", price=2.0, category_name="Snacks", brand="SnackWorks"),
        3: Product(id=3, category_id=1, slug="c", name="Whole Milk", price=2.5, category_name="Dairy & Eggs", brand="DairyBest"),
    }


def _embeddings() -> dict[int, np.ndarray]:
    return {
        1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        3: np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }


def _config(**overrides) -> FeatureConfig:
    return FeatureConfig(**overrides)


class _FakeEncoder:
    """Deterministic stand-in for SentenceTransformerEncoder - avoids
    pulling the real model into fast unit tests that only exercise blending
    logic, not encoding quality (covered separately in test_embeddings.py).
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([[float(len(t)), 0.0, 0.0, 1.0] for t in texts], dtype=np.float32)


def test_no_history_user_has_empty_features_and_no_embedding():
    profile = EngagementProfile(user_id=1, profile=UserProfile(user_id=1))
    features = build_user_features(profile, _products(), _embeddings(), _config())

    assert features.click_count == 0
    assert features.purchase_count == 0
    assert features.cart_item_count == 0
    assert features.search_count == 0
    assert features.has_chatbot_context is False
    assert features.total_engagement_events == 0
    assert features.category_affinity == {}
    assert features.brand_affinity == {}
    assert features.semantic_embedding is None
    assert features.has_preferred_category is False
    assert features.has_age_group is False


def test_preferred_category_and_age_group_are_carried_through():
    profile = EngagementProfile(user_id=1, profile=UserProfile(user_id=1, preferred_category="Snacks", age_group="25-34"))
    features = build_user_features(profile, _products(), _embeddings(), _config())
    assert features.preferred_category == "Snacks"
    assert features.age_group == "25-34"
    assert features.has_preferred_category is True
    assert features.has_age_group is True
    # preferred_category alone (no behavioral history) still produces an affinity signal.
    assert features.category_affinity == {"Snacks": 1.0}


def test_purchases_dominate_category_affinity_when_weighted_higher():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=1, unit_price=4.0)],
        cart_items=[CartAffinityRecord(user_id=1, product_id=2, quantity=1)],
    )
    config = _config(purchase_weight=0.8, cart_weight=0.2)
    features = build_user_features(profile, _products(), _embeddings(), config)
    assert features.category_affinity["Dairy & Eggs"] == pytest.approx(0.8)
    assert features.category_affinity["Snacks"] == pytest.approx(0.2)


def test_purchase_quantity_scales_category_weight():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[
            PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=3, unit_price=4.0),
            PurchaseRecord(user_id=1, product_id=2, order_id=2, quantity=1, unit_price=2.0),
        ],
    )
    config = _config(purchase_weight=1.0)
    features = build_user_features(profile, _products(), _embeddings(), config)
    # product 1 bought at qty 3 vs product 2 at qty 1 -> 3:1 ratio in affinity.
    assert features.category_affinity["Dairy & Eggs"] == pytest.approx(0.75)
    assert features.category_affinity["Snacks"] == pytest.approx(0.25)


def test_semantic_embedding_is_pure_purchase_vector_when_only_purchases_present():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=1, unit_price=4.0)],
    )
    features = build_user_features(profile, _products(), _embeddings(), _config())
    assert features.semantic_embedding is not None
    assert np.allclose(features.semantic_embedding, _embeddings()[1])


def test_semantic_embedding_blends_purchase_and_cart_by_configured_weight():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=1, unit_price=4.0)],
        cart_items=[CartAffinityRecord(user_id=1, product_id=2, quantity=1)],
    )
    config = _config(purchase_weight=0.75, cart_weight=0.25)
    features = build_user_features(profile, _products(), _embeddings(), config)
    expected = 0.75 * _embeddings()[1] + 0.25 * _embeddings()[2]
    assert np.allclose(features.semantic_embedding, expected)


def test_missing_signal_does_not_dilute_embedding_toward_zero():
    """A user with ONLY a chatbot summary (no purchase/cart/search) should
    get an embedding equal to the chatbot vector, not that vector scaled
    down by chatbot_weight against phantom zero vectors from absent
    signals.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        chatbot_context=ChatbotContextRecord(user_id=1, summary="asking about healthy snacks"),
    )
    text_embeddings = build_user_text_embeddings([profile], _FakeEncoder())
    features = build_user_features(profile, _products(), _embeddings(), _config(chatbot_weight=0.15), text_embeddings=text_embeddings)
    assert features.semantic_embedding is not None
    assert np.allclose(features.semantic_embedding, text_embeddings["asking about healthy snacks"])


def test_search_without_matched_product_uses_text_embedding():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        searches=[SearchRecord(user_id=1, search_term="low sugar breakfast")],
    )
    text_embeddings = build_user_text_embeddings([profile], _FakeEncoder())
    features = build_user_features(profile, _products(), _embeddings(), _config(), text_embeddings=text_embeddings)
    assert features.semantic_embedding is not None
    assert np.allclose(features.semantic_embedding, text_embeddings["low sugar breakfast"])


def test_search_with_matched_product_uses_product_embedding_not_text():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        searches=[SearchRecord(user_id=1, search_term="greek yogurt", matched_product_id=1)],
    )
    features = build_user_features(profile, _products(), _embeddings(), _config())
    assert np.allclose(features.semantic_embedding, _embeddings()[1])


def test_exclude_product_ids_removes_target_leakage_from_purchases():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[
            PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=1, unit_price=4.0),
            PurchaseRecord(user_id=1, product_id=3, order_id=2, quantity=1, unit_price=2.5),
        ],
    )
    # Without exclusion: category affinity is 100% Dairy & Eggs (both products are that category)
    # and semantic embedding averages products 1 and 3.
    full = build_user_features(profile, _products(), _embeddings(), _config())
    assert full.purchase_count == 2

    # Excluding product 1 (imagine it's the training target) should leave
    # only product 3's influence.
    leave_one_out = build_user_features(
        profile, _products(), _embeddings(), _config(), exclude_product_ids=frozenset({1})
    )
    assert leave_one_out.purchase_count == 1
    assert np.allclose(leave_one_out.semantic_embedding, _embeddings()[3])


def test_exclude_product_ids_also_scrubs_chatbot_mentions_and_matched_searches():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        searches=[SearchRecord(user_id=1, search_term="milk", matched_product_id=3)],
        chatbot_context=ChatbotContextRecord(user_id=1, mentioned_product_ids=[3]),
    )
    features = build_user_features(profile, _products(), _embeddings(), _config(), exclude_product_ids=frozenset({3}))
    assert features.search_count == 1  # the search record itself remains, just without product 3's signal
    assert features.category_affinity == {}
    assert features.semantic_embedding is None


def test_total_engagement_events_sums_all_five_signals():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[ClickRecord(user_id=1, product_id=3)],
        purchases=[PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=1, unit_price=4.0)],
        cart_items=[CartAffinityRecord(user_id=1, product_id=2, quantity=1)],
        searches=[SearchRecord(user_id=1, search_term="milk")],
        chatbot_context=ChatbotContextRecord(user_id=1, summary="hi"),
    )
    features = build_user_features(profile, _products(), _embeddings(), _config())
    assert features.total_engagement_events == 5  # 1 click + 1 purchase + 1 cart + 1 search + 1 (chatbot presence)


# --- CLICK (fifth V1 signal) --------------------------------------------

def test_click_contributes_to_category_and_brand_affinity():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[ClickRecord(user_id=1, product_id=1)],
    )
    config = _config(click_weight=1.0)
    features = build_user_features(profile, _products(), _embeddings(), config)
    assert features.click_count == 1
    assert features.category_affinity == {"Dairy & Eggs": pytest.approx(1.0)}
    assert features.brand_affinity == {"GreenValley": pytest.approx(1.0)}


def test_click_contributes_to_semantic_embedding_when_only_signal():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[ClickRecord(user_id=1, product_id=2)],
    )
    features = build_user_features(profile, _products(), _embeddings(), _config(click_weight=0.1))
    assert features.semantic_embedding is not None
    assert np.allclose(features.semantic_embedding, _embeddings()[2])


def test_exclude_product_ids_removes_target_leakage_from_clicks():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[
            ClickRecord(user_id=1, product_id=1),
            ClickRecord(user_id=1, product_id=3),
        ],
    )
    full = build_user_features(profile, _products(), _embeddings(), _config())
    assert full.click_count == 2

    leave_one_out = build_user_features(
        profile, _products(), _embeddings(), _config(), exclude_product_ids=frozenset({1})
    )
    assert leave_one_out.click_count == 1


def test_max_top_categories_truncates_and_renormalizes():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[
            PurchaseRecord(user_id=1, product_id=1, order_id=1, quantity=5, unit_price=4.0),  # Dairy & Eggs
            PurchaseRecord(user_id=1, product_id=2, order_id=2, quantity=1, unit_price=2.0),  # Snacks
        ],
    )
    config = _config(max_top_categories=1)
    features = build_user_features(profile, _products(), _embeddings(), config)
    assert list(features.category_affinity.keys()) == ["Dairy & Eggs"]
    assert features.category_affinity["Dairy & Eggs"] == pytest.approx(1.0)


def test_unmatched_search_text_mentioning_excluded_product_name_is_dropped():
    """Conservative no-timestamp leakage guard: an unmatched search whose
    raw text names the excluded product should not leak into the
    semantic embedding even without a matched_product_id link.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        searches=[SearchRecord(user_id=1, search_term="greek yogurt")],  # textually names product 1, unmatched
    )
    text_embeddings = build_user_text_embeddings([profile], _FakeEncoder())
    features = build_user_features(
        profile, _products(), _embeddings(), _config(), text_embeddings=text_embeddings, exclude_product_ids=frozenset({1})
    )
    assert features.search_count == 1  # event still counted
    assert features.semantic_embedding is None  # but its content is suppressed


def test_unmatched_search_text_not_mentioning_excluded_product_is_kept():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        searches=[SearchRecord(user_id=1, search_term="low sugar breakfast")],
    )
    text_embeddings = build_user_text_embeddings([profile], _FakeEncoder())
    features = build_user_features(
        profile, _products(), _embeddings(), _config(), text_embeddings=text_embeddings, exclude_product_ids=frozenset({1})
    )
    assert features.semantic_embedding is not None
    assert np.allclose(features.semantic_embedding, text_embeddings["low sugar breakfast"])


def test_chatbot_mentioning_excluded_product_by_id_drops_entire_chatbot_content():
    """A chatbot record that mentions the excluded product should have its
    ENTIRE content contribution dropped (summary embedding included), not
    just the flagged product id - we can't safely salvage the rest of that
    conversation without a timestamp.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        chatbot_context=ChatbotContextRecord(
            user_id=1, mentioned_product_ids=[1], preferred_category="Snacks", summary="asking about snacks"
        ),
    )
    text_embeddings = build_user_text_embeddings([profile], _FakeEncoder())
    features = build_user_features(
        profile, _products(), _embeddings(), _config(chatbot_weight=0.5),
        text_embeddings=text_embeddings, exclude_product_ids=frozenset({1}),
    )
    assert features.has_chatbot_context is True  # event still counted
    assert features.category_affinity == {}  # preferred_category contribution also dropped
    assert features.semantic_embedding is None  # summary embedding also dropped


def test_chatbot_mentioning_excluded_product_by_name_in_summary_drops_content():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        chatbot_context=ChatbotContextRecord(user_id=1, summary="User asked about Greek Yogurt options"),
    )
    text_embeddings = build_user_text_embeddings([profile], _FakeEncoder())
    features = build_user_features(
        profile, _products(), _embeddings(), _config(), text_embeddings=text_embeddings, exclude_product_ids=frozenset({1})
    )
    assert features.semantic_embedding is None


def test_chatbot_not_mentioning_excluded_product_is_unaffected():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        chatbot_context=ChatbotContextRecord(user_id=1, summary="asking about snacks"),
    )
    text_embeddings = build_user_text_embeddings([profile], _FakeEncoder())
    features = build_user_features(
        profile, _products(), _embeddings(), _config(), text_embeddings=text_embeddings, exclude_product_ids=frozenset({1})
    )
    assert features.semantic_embedding is not None


def test_build_user_text_embeddings_deduplicates_and_only_covers_needed_text():
    profiles = [
        EngagementProfile(
            user_id=1, profile=UserProfile(user_id=1),
            searches=[SearchRecord(user_id=1, search_term="milk", matched_product_id=1),  # matched -> not encoded
                      SearchRecord(user_id=1, search_term="healthy snacks")],
            chatbot_context=ChatbotContextRecord(user_id=1, summary="healthy snacks"),  # duplicate text
        ),
    ]
    result = build_user_text_embeddings(profiles, _FakeEncoder())
    assert set(result.keys()) == {"healthy snacks"}


# --- Recency weighting (docs/data-mapping.md section 14) ------------------

T0 = datetime(2026, 6, 1, 0, 0, 0)


def _t(days_ago: float) -> datetime:
    return T0 - timedelta(days=days_ago)


def test_reference_time_omitted_leaves_recency_neutral_even_when_events_are_timestamped():
    """Recency is strictly opt-in per call - a caller that never passes
    `reference_time` (the pre-existing, non-temporal leave-one-out
    training path) must get byte-for-byte the same features whether or
    not the underlying records happen to carry an `action_time`.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[ClickRecord(user_id=1, product_id=1, action_time=_t(365))],
    )
    config = _config(click_weight=1.0)
    with_time = build_user_features(profile, _products(), _embeddings(), config)  # no reference_time
    assert with_time.category_affinity == {"Dairy & Eggs": pytest.approx(1.0)}
    assert np.allclose(with_time.semantic_embedding, _embeddings()[1])


def test_recent_click_contributes_more_than_older_click_to_category_affinity():
    """Two clicks of equal configured weight in DIFFERENT categories - the
    recent one (Dairy & Eggs) should end up with a larger affinity share
    than the older one (Snacks) once recency is enabled.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[
            ClickRecord(user_id=1, product_id=1, action_time=_t(0)),  # Dairy & Eggs, recent
            ClickRecord(user_id=1, product_id=2, action_time=_t(120)),  # Snacks, old
        ],
    )
    config = _config(click_weight=1.0, recency=RecencyConfig(enabled=True, half_life_days=21.0))
    features = build_user_features(profile, _products(), _embeddings(), config, reference_time=T0)
    assert features.category_affinity["Dairy & Eggs"] > features.category_affinity["Snacks"]
    assert sum(features.category_affinity.values()) == pytest.approx(1.0)


def test_recent_purchase_dominates_category_affinity_over_older_purchase_same_weight():
    """Two purchases of equal configured weight in DIFFERENT categories -
    the recent one should end up with a larger affinity share than the
    older one, once recency is enabled with an explicit reference_time.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[
            PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=_t(0)),  # Dairy & Eggs, recent
            PurchaseRecord(user_id=1, product_id=2, quantity=1, order_created_at=_t(120)),  # Snacks, old
        ],
    )
    config = _config(purchase_weight=1.0, recency=RecencyConfig(enabled=True, half_life_days=21.0))
    features = build_user_features(profile, _products(), _embeddings(), config, reference_time=T0)
    assert features.category_affinity["Dairy & Eggs"] > features.category_affinity["Snacks"]

    # A recency-disabled run over the identical data should split evenly instead.
    disabled_config = _config(purchase_weight=1.0, recency=RecencyConfig(enabled=False))
    disabled_features = build_user_features(profile, _products(), _embeddings(), disabled_config, reference_time=T0)
    assert disabled_features.category_affinity["Dairy & Eggs"] == pytest.approx(0.5)
    assert disabled_features.category_affinity["Snacks"] == pytest.approx(0.5)


def test_recent_purchase_pulls_semantic_embedding_toward_itself():
    """P10 (product 1) very recent, P20 (product 2) old - the recency-
    weighted embedding should sit closer to product 1's vector than an
    unweighted (recency-disabled) blend would.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[
            PurchaseRecord(user_id=1, product_id=1, quantity=1, order_created_at=_t(0)),
            PurchaseRecord(user_id=1, product_id=2, quantity=1, order_created_at=_t(150)),
        ],
    )
    config = _config(purchase_weight=1.0, recency=RecencyConfig(enabled=True, half_life_days=21.0))
    recency_features = build_user_features(profile, _products(), _embeddings(), config, reference_time=T0)

    disabled_config = _config(purchase_weight=1.0, recency=RecencyConfig(enabled=False))
    baseline_features = build_user_features(profile, _products(), _embeddings(), disabled_config, reference_time=T0)

    p1, p2 = _embeddings()[1], _embeddings()[2]
    # Cosine similarity to the recent product's own vector should be higher
    # under recency weighting than under the unweighted 50/50 baseline.
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cos(recency_features.semantic_embedding, p1) > cos(baseline_features.semantic_embedding, p1)


def test_raw_event_counts_are_unaffected_by_recency():
    """Cold-start tiering depends on raw historical counts, not recency-
    weighted strength (docs/data-mapping.md section 14) - counts must be
    identical whether recency is enabled or disabled, and regardless of
    reference_time.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[ClickRecord(user_id=1, product_id=1, action_time=_t(200))],
        purchases=[PurchaseRecord(user_id=1, product_id=2, quantity=1, order_created_at=_t(5))],
        cart_items=[CartAffinityRecord(user_id=1, product_id=3, quantity=1, action_time=_t(1))],
        searches=[SearchRecord(user_id=1, search_term="milk", matched_product_id=3, action_time=_t(50))],
        chatbot_context=ChatbotContextRecord(user_id=1, summary="hi", action_time=_t(10)),
    )
    enabled = _config(recency=RecencyConfig(enabled=True, half_life_days=21.0))
    disabled = _config(recency=RecencyConfig(enabled=False))

    f_enabled = build_user_features(profile, _products(), _embeddings(), enabled, reference_time=T0)
    f_disabled = build_user_features(profile, _products(), _embeddings(), disabled, reference_time=T0)
    f_no_ref = build_user_features(profile, _products(), _embeddings(), enabled)  # no reference_time at all

    for f in (f_enabled, f_disabled, f_no_ref):
        assert f.click_count == 1
        assert f.purchase_count == 1
        assert f.cart_item_count == 1
        assert f.search_count == 1
        assert f.has_chatbot_context is True
        assert f.total_engagement_events == 5


def test_category_affinity_still_normalizes_to_one_with_recency_enabled():
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        purchases=[
            PurchaseRecord(user_id=1, product_id=1, quantity=2, order_created_at=_t(3)),
            PurchaseRecord(user_id=1, product_id=2, quantity=1, order_created_at=_t(90)),
        ],
    )
    config = _config(purchase_weight=0.6, recency=RecencyConfig(enabled=True, half_life_days=21.0))
    features = build_user_features(profile, _products(), _embeddings(), config, reference_time=T0)
    assert sum(features.category_affinity.values()) == pytest.approx(1.0)


def test_missing_action_time_falls_back_to_neutral_weight_even_when_recency_enabled():
    """Old ERD-based synthetic click/cart/search records with no
    `action_time` must not be dropped or down-weighted - see
    `features.recency.effective_weight`'s missing-timestamp policy.
    """
    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[ClickRecord(user_id=1, product_id=1, action_time=None)],
    )
    config = _config(click_weight=1.0, recency=RecencyConfig(enabled=True, half_life_days=21.0))
    features = build_user_features(profile, _products(), _embeddings(), config, reference_time=T0)
    assert features.category_affinity == {"Dairy & Eggs": pytest.approx(1.0)}
    assert np.allclose(features.semantic_embedding, _embeddings()[1])


def test_future_event_relative_to_reference_time_raises_not_silently_accepted():
    """A defensive leakage guard: if a caller (bug) hands `build_user_features`
    an event dated at/after its own `reference_time`, this must fail loudly
    rather than silently computing a nonsensical (or clamped) weight - see
    `features.recency.RecencyLeakageError` and docs/data-mapping.md section 14.
    """
    from recommendation.features.recency import RecencyLeakageError

    profile = EngagementProfile(
        user_id=1,
        profile=UserProfile(user_id=1),
        clicks=[ClickRecord(user_id=1, product_id=1, action_time=T0 + timedelta(days=1))],
    )
    config = _config(click_weight=1.0, recency=RecencyConfig(enabled=True, half_life_days=21.0))
    with pytest.raises(RecencyLeakageError):
        build_user_features(profile, _products(), _embeddings(), config, reference_time=T0)
