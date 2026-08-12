"""PurchaseAdapter backed by ERD entities Order, OrderItem, Product.

The strongest V1 engagement signal. `counted_statuses` is a constructor
parameter, not a hard-coded check, because the real Order.Status enum is
unknown (docs/data-mapping.md section 1) - pointing this at the real
backend later means passing the real "this counts as a completed
purchase" status set, not touching adapter logic. No recency is derived
here; `order_created_at` is passed through untouched for schema fidelity
only.
"""

from __future__ import annotations

from collections import defaultdict

from recommendation.data.adapters.base import PurchaseAdapter
from recommendation.data.schemas.engagement import PurchaseRecord
from recommendation.data.synthetic.raw_schemas import RawOrder, RawOrderItem

DEFAULT_COUNTED_STATUSES = frozenset({"DELIVERED", "COMPLETED"})


class InMemoryPurchaseAdapter(PurchaseAdapter):
    def __init__(
        self,
        orders: list[RawOrder],
        order_items: list[RawOrderItem],
        counted_statuses: frozenset[str] | set[str] | None = None,
    ) -> None:
        counted = frozenset(counted_statuses) if counted_statuses is not None else DEFAULT_COUNTED_STATUSES
        order_by_id = {o.id: o for o in orders}

        self._purchases_by_user: dict[int, list[PurchaseRecord]] = defaultdict(list)
        for item in order_items:
            order = order_by_id.get(item.order_id)
            if order is None or order.status not in counted:
                continue
            self._purchases_by_user[order.user_id].append(
                PurchaseRecord(
                    user_id=order.user_id,
                    product_id=item.product_id,
                    order_id=order.id,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    order_status=order.status,
                    order_created_at=order.creation_date,
                )
            )

    def get_purchases(self, user_id: int) -> list[PurchaseRecord]:
        return list(self._purchases_by_user.get(user_id, []))

    def list_all_purchases(self) -> list[PurchaseRecord]:
        return [record for records in self._purchases_by_user.values() for record in records]
