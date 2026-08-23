"""Generate, validate, save, and report stats for the synthetic dataset.

Usage:
    python scripts/generate_synthetic_dataset.py [--out data/synthetic]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recommendation.data.synthetic.dataset import generate_synthetic_dataset
from recommendation.data.synthetic.io import save_dataset
from recommendation.data.synthetic.validation import validate_dataset
from recommendation.utils.config import get_config
from recommendation.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def print_stats(dataset) -> None:
    n_users = len(dataset.users)
    n_products = len(dataset.products)
    n_orders = len(dataset.orders)
    n_order_items = len(dataset.order_items)
    n_cart_items = len(dataset.cart_items)
    n_reviews = len(dataset.reviews)
    n_clicks = len(dataset.click_records)
    n_searches = len(dataset.search_records)
    n_chatbot = len(dataset.chatbot_records)

    users_with_purchases = len({o.user_id for o in dataset.orders})
    cart_user_by_cart_id = {c.id: c.user_id for c in dataset.carts}
    users_with_cart_items = len({cart_user_by_cart_id[ci.cart_id] for ci in dataset.cart_items})
    users_incomplete_profile = sum(1 for u in dataset.users if u.preferred_category_id is None)

    persona_counts = Counter(dataset.debug_user_personas.values())

    print("=== Synthetic Dataset Stats ===")
    print(f"Users:                 {n_users}")
    print(f"  incomplete profile:  {users_incomplete_profile} ({users_incomplete_profile / n_users:.1%})")
    print(f"  with >=1 purchase:   {users_with_purchases} ({users_with_purchases / n_users:.1%})")
    print(f"  with cart items:     {users_with_cart_items} ({users_with_cart_items / n_users:.1%})")
    print(f"Products:              {n_products}")
    print(f"Categories:            {len(dataset.categories)}")
    print(f"Tags:                  {len(dataset.tags)}")
    print(f"Orders:                {n_orders} ({n_order_items} order items)")
    print(f"Cart items:            {n_cart_items}")
    print(f"Reviews:               {n_reviews}")
    print(f"Click records:         {n_clicks}")
    print(f"Search records:        {n_searches}")
    print(f"Chatbot context recs:  {n_chatbot} ({n_chatbot / n_users:.1%} of users)")
    print()
    print("Persona distribution:")
    for key, count in persona_counts.most_common():
        print(f"  {key:22s} {count:4d} ({count / n_users:.1%})")


def main() -> None:
    config = get_config()
    setup_logging(config.log_level)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Output directory (default: config paths.data_synthetic)")
    args = parser.parse_args()

    out_dir = args.out or config.paths.data_synthetic

    logger.info("Generating synthetic dataset (seed=%s)", config.synthetic_data.random_seed)
    dataset = generate_synthetic_dataset(config)

    issues = validate_dataset(dataset)
    if issues:
        logger.error("Validation found %d issue(s):", len(issues))
        for issue in issues[:20]:
            logger.error("  %s", issue)
        raise SystemExit(1)
    logger.info("Validation passed: no referential-integrity issues found.")

    save_dataset(dataset, out_dir)
    logger.info("Saved dataset to %s", out_dir)

    print_stats(dataset)


if __name__ == "__main__":
    main()
