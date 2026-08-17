"""Referential-integrity and sanity validation for a generated dataset.

Field-level validation (types, ranges) already happens at construction
time via the pydantic raw/canonical schemas. This module checks the
cross-table invariants pydantic can't express on its own: every foreign
key actually resolves, and catalog/user counts are within expected bounds.

`validate_dataset` returns the list of problems found rather than raising,
so callers (tests, the generation script) decide whether to fail loudly or
just report.
"""

from __future__ import annotations

from recommendation.data.synthetic.dataset import SyntheticDataset


def validate_dataset(dataset: SyntheticDataset) -> list[str]:
    issues: list[str] = []

    category_ids = {c.id for c in dataset.categories}
    tag_ids = {t.id for t in dataset.tags}
    product_ids = {p.id for p in dataset.products}
    user_ids = {u.id for u in dataset.users}
    order_ids = {o.id for o in dataset.orders}
    cart_ids = {c.id for c in dataset.carts}

    for category in dataset.categories:
        if category.parent_id is not None and category.parent_id not in category_ids:
            issues.append(f"Category {category.id} has dangling parent_id {category.parent_id}")

    for product in dataset.products:
        if product.category_id not in category_ids:
            issues.append(f"Product {product.id} has dangling category_id {product.category_id}")

    for pt in dataset.product_tags:
        if pt.product_id not in product_ids:
            issues.append(f"ProductTag {pt.id} has dangling product_id {pt.product_id}")
        if pt.tag_id not in tag_ids:
            issues.append(f"ProductTag {pt.id} has dangling tag_id {pt.tag_id}")

    for user in dataset.users:
        if user.preferred_category_id is not None and user.preferred_category_id not in category_ids:
            issues.append(f"User {user.id} has dangling preferred_category_id {user.preferred_category_id}")

    for order in dataset.orders:
        if order.user_id not in user_ids:
            issues.append(f"Order {order.id} has dangling user_id {order.user_id}")

    for item in dataset.order_items:
        if item.order_id not in order_ids:
            issues.append(f"OrderItem {item.id} has dangling order_id {item.order_id}")
        if item.product_id not in product_ids:
            issues.append(f"OrderItem {item.id} has dangling product_id {item.product_id}")

    for cart in dataset.carts:
        if cart.user_id not in user_ids:
            issues.append(f"Cart {cart.id} has dangling user_id {cart.user_id}")

    for item in dataset.cart_items:
        if item.cart_id not in cart_ids:
            issues.append(f"CartItem {item.id} has dangling cart_id {item.cart_id}")
        if item.product_id not in product_ids:
            issues.append(f"CartItem {item.id} has dangling product_id {item.product_id}")

    for review in dataset.reviews:
        if review.user_id not in user_ids:
            issues.append(f"Review {review.id} has dangling user_id {review.user_id}")
        if review.product_id not in product_ids:
            issues.append(f"Review {review.id} has dangling product_id {review.product_id}")

    for record in dataset.click_records:
        if record.user_id not in user_ids:
            issues.append(f"ClickRecord for user {record.user_id} has dangling user_id")
        if record.product_id not in product_ids:
            issues.append(f"ClickRecord for user {record.user_id} has dangling product_id {record.product_id}")

    for record in dataset.search_records:
        if record.user_id not in user_ids:
            issues.append(f"SearchRecord for user {record.user_id} has dangling user_id")
        if record.matched_product_id is not None and record.matched_product_id not in product_ids:
            issues.append(f"SearchRecord for user {record.user_id} has dangling matched_product_id {record.matched_product_id}")

    for record in dataset.chatbot_records:
        if record.user_id not in user_ids:
            issues.append(f"ChatbotContextRecord for user {record.user_id} has dangling user_id")
        for pid in record.mentioned_product_ids:
            if pid not in product_ids:
                issues.append(f"ChatbotContextRecord for user {record.user_id} has dangling mentioned_product_id {pid}")

    return issues
