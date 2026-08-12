"""CartAdapter backed by ERD entities Cart, CartItem, Product.

Implements the sane relational direction (CartItem.cart_id -> Cart.id,
one cart with many items) per the documented ERD discrepancy in
docs/data-mapping.md section 1.
"""

from __future__ import annotations

from collections import defaultdict

from recommendation.data.adapters.base import CartAdapter
from recommendation.data.schemas.engagement import CartAffinityRecord
from recommendation.data.synthetic.raw_schemas import RawCart, RawCartItem


class InMemoryCartAdapter(CartAdapter):
    def __init__(self, carts: list[RawCart], cart_items: list[RawCartItem]) -> None:
        cart_by_id = {c.id: c for c in carts}
        self._items_by_user: dict[int, list[CartAffinityRecord]] = defaultdict(list)
        for item in cart_items:
            cart = cart_by_id.get(item.cart_id)
            if cart is None:
                continue
            self._items_by_user[cart.user_id].append(
                CartAffinityRecord(user_id=cart.user_id, product_id=item.product_id, quantity=item.quantity)
            )

    def get_cart_items(self, user_id: int) -> list[CartAffinityRecord]:
        return list(self._items_by_user.get(user_id, []))
