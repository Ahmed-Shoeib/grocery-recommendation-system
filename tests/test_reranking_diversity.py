from recommendation.features.product_features import ProductFeatures
from recommendation.reranking.candidates import RankedCandidate
from recommendation.reranking.diversity import apply_diversity, deduplicate, rerank
from recommendation.utils.config import ReRankingConfig


def _pf(pid: int, category: str, brand: str | None = None) -> ProductFeatures:
    return ProductFeatures(
        product_id=pid, category_id=1, category_name=category, parent_category_name=None, brand=brand, tags=[],
        price=1.0, effective_price=1.0, discount_percentage=0.0, is_active=True, stock_quantity=10,
        purchase_count=0, distinct_purchasers=0, cart_add_count=0, review_count=0, average_rating=None,
    )


def test_deduplicate_keeps_first_occurrence():
    candidates = [RankedCandidate(1, 0.9, "a"), RankedCandidate(1, 0.5, "b"), RankedCandidate(2, 0.8, "a")]
    deduped = deduplicate(candidates)
    assert [c.product_id for c in deduped] == [1, 2]
    assert deduped[0].score == 0.9


def test_deduplicate_preserves_order_of_uniques():
    candidates = [RankedCandidate(3, 0.9, "a"), RankedCandidate(1, 0.8, "a"), RankedCandidate(3, 0.1, "a")]
    assert [c.product_id for c in deduplicate(candidates)] == [3, 1]


def test_diversity_strength_zero_preserves_input_order():
    candidates = [RankedCandidate(1, 0.9, "a"), RankedCandidate(2, 0.8, "a"), RankedCandidate(3, 0.7, "a")]
    features = {1: _pf(1, "Dairy"), 2: _pf(2, "Dairy"), 3: _pf(3, "Dairy")}
    config = ReRankingConfig(diversity_strength=0.0)
    result = apply_diversity(candidates, features, config)
    assert [c.product_id for c in result] == [1, 2, 3]


def test_diversity_promotes_different_category_when_strength_high():
    # 1 and 2 same category (Dairy), 3 different (Snacks); high penalty should push 3 above 2.
    candidates = [RankedCandidate(1, 0.90, "a"), RankedCandidate(2, 0.89, "a"), RankedCandidate(3, 0.85, "a")]
    features = {1: _pf(1, "Dairy"), 2: _pf(2, "Dairy"), 3: _pf(3, "Snacks")}
    config = ReRankingConfig(diversity_strength=1.0, category_repetition_penalty=0.5, brand_repetition_penalty=0.0)
    result = apply_diversity(candidates, features, config)
    assert [c.product_id for c in result] == [1, 3, 2]


def test_diversity_never_drops_a_candidate():
    candidates = [RankedCandidate(i, 1.0 - i * 0.01, "a") for i in range(10)]
    features = {i: _pf(i, "SameCategory") for i in range(10)}
    config = ReRankingConfig(diversity_strength=1.0, category_repetition_penalty=1.0)
    result = apply_diversity(candidates, features, config)
    assert {c.product_id for c in result} == set(range(10))
    assert len(result) == 10


def test_diversity_handles_unknown_product_gracefully():
    candidates = [RankedCandidate(1, 0.9, "a"), RankedCandidate(2, 0.8, "a")]
    config = ReRankingConfig(diversity_strength=1.0)
    result = apply_diversity(candidates, {}, config)  # no product_features entries at all
    assert len(result) == 2


def test_rerank_dedupes_then_diversifies():
    candidates = [
        RankedCandidate(1, 0.95, "a"),
        RankedCandidate(1, 0.10, "a"),  # duplicate, should be dropped before diversity runs
        RankedCandidate(2, 0.90, "a"),
    ]
    features = {1: _pf(1, "Dairy"), 2: _pf(2, "Snacks")}
    config = ReRankingConfig(diversity_strength=0.0)
    result = rerank(candidates, features, config)
    assert [c.product_id for c in result] == [1, 2]
