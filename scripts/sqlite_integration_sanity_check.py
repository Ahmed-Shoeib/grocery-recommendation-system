"""Sanity-check-only script (no training) proving:

    data/sqlite/backend_shaped_synthetic.db
        -> SQLite adapters (recommendation.data.adapters.sqlite_factory)
        -> canonical EngagementProfile / UserFeatures / ProductFeatures
        -> existing feature engineering

works end to end, and that the pre-retrieval eligibility gate sees correct
isActive/stockQuantity values from this source. Does not train, retrain,
rebuild any index, or write model artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.data.adapters.engagement import build_engagement_profile
from recommendation.data.adapters.sqlite_factory import build_sqlite_adapters
from recommendation.features.pipeline import run_feature_pipeline
from recommendation.serving.cold_start import determine_history_tier
from recommendation.serving.eligibility import apply_eligibility, build_eligibility_rules
from recommendation.utils.config import get_config


def find_user_by_event_count(bundle, predicate) -> int | None:
    for uid in bundle.users.list_user_ids():
        n = len(bundle.clicks.get_clicks(uid)) + len(bundle.purchases.get_purchases(uid)) \
            + len(bundle.cart.get_cart_items(uid)) + len(bundle.search.get_search_history(uid)) \
            + (1 if bundle.chatbot.get_chatbot_context(uid) is not None else 0)
        if predicate(n):
            return uid
    return None


def describe_user(bundle, config, user_id: int, product_lookup) -> None:
    profile = bundle.users.get_user_profile(user_id)
    engagement = build_engagement_profile(
        user_id, bundle.users, bundle.purchases, bundle.cart, bundle.clicks, bundle.search, bundle.chatbot, bundle.reviews
    )
    total = (
        len(engagement.clicks) + len(engagement.purchases) + len(engagement.cart_items)
        + len(engagement.searches) + (1 if engagement.chatbot_context is not None else 0)
    )
    tier = determine_history_tier(total, config.cold_start)

    print(f"\n--- User {user_id} ---")
    print(f"  age_group={profile.age_group!r}  preferred_category={profile.preferred_category!r}")
    print(f"  tier(from total={total}) = {tier.value}")
    print(f"  click={len(engagement.clicks)} search={len(engagement.searches)} "
          f"chatbot={1 if engagement.chatbot_context else 0} cart={len(engagement.cart_items)} "
          f"purchase={len(engagement.purchases)}")

    examples = []
    for c in engagement.clicks[:2]:
        examples.append(("CLICK", c.product_id, c.action_time))
    for s in engagement.searches[:2]:
        examples.append(("SEARCH", s.matched_product_id, s.action_time))
    for p in engagement.purchases[:2]:
        examples.append(("PURCHASE", p.product_id, p.order_created_at))
    for action_type, pid, t in examples:
        name = product_lookup[pid].name if pid in product_lookup else "?"
        print(f"    {action_type:10s} product_id={pid} name={name!r} action_time={t}")


def main() -> None:
    config = get_config()
    bundle = build_sqlite_adapters()
    product_lookup = {p.id: p for p in bundle.products.list_products()}

    print("=== SECTION 19: representative users (from SQLite, not synthetic) ===")
    no_history_uid = find_user_by_event_count(bundle, lambda n: n == 0)
    sparse_uid = find_user_by_event_count(bundle, lambda n: 1 <= n <= 4)
    strong_uid = find_user_by_event_count(bundle, lambda n: n >= 15)

    describe_user(bundle, config, no_history_uid, product_lookup)
    describe_user(bundle, config, sparse_uid, product_lookup)
    describe_user(bundle, config, strong_uid, product_lookup)

    print("\n=== SECTION 21: eligibility sanity check ===")
    rules = build_eligibility_rules(config.eligibility)
    products = bundle.products.list_products()
    from recommendation.features.product_features import build_product_features
    product_features = build_product_features(products, [], [], [])

    def show_example(is_active: bool, in_stock: bool) -> None:
        match = next(
            (p for p in products if p.is_active == is_active and (p.stock_quantity > 0) == in_stock), None
        )
        if match is None:
            print(f"  isActive={is_active} in_stock={in_stock}: no example product found")
            return
        result = apply_eligibility([match.id], product_features, rules)
        eligible = match.id in result.eligible_ids
        print(f"  product_id={match.id} isActive={is_active} in_stock={in_stock} "
              f"stock_quantity={match.stock_quantity} -> eligible={eligible}")

    show_example(True, True)
    show_example(True, False)
    show_example(False, True)
    show_example(False, False)

    print("\n=== SECTION 20: feature-pipeline sanity check (no training) ===")
    from recommendation.embeddings.encoder import SentenceTransformerEncoder

    encoder = SentenceTransformerEncoder(
        config.embedding.sentence_transformer_model, device=config.embedding.device, batch_size=config.embedding.encode_batch_size
    )
    result = run_feature_pipeline(bundle, config, encoder=encoder)
    print(f"  product_features: {len(result.product_features)} products")
    print(f"  user_features: {len(result.user_features)} users")

    for uid in [no_history_uid, sparse_uid, strong_uid]:
        uf = result.user_features[uid]
        print(f"\n  UserFeatures[{uid}]: click_count={uf.click_count} purchase_count={uf.purchase_count} "
              f"cart_item_count={uf.cart_item_count} search_count={uf.search_count} "
              f"has_chatbot_context={uf.has_chatbot_context} total_engagement_events={uf.total_engagement_events}")
        print(f"    category_affinity(top3)={dict(list(uf.category_affinity.items())[:3])}")
        print(f"    semantic_embedding is None: {uf.semantic_embedding is None}")

    sample_pid = next(iter(result.product_features))
    pf = result.product_features[sample_pid]
    print(f"\n  ProductFeatures[{sample_pid}]: category={pf.category_name} brand={pf.brand} "
          f"price={pf.price} effective_price={pf.effective_price} is_active={pf.is_active} "
          f"stock_quantity={pf.stock_quantity}")


if __name__ == "__main__":
    main()
