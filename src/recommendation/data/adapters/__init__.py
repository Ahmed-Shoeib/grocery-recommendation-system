"""Data adapter interfaces and implementations.

Adapters translate a data source (real backend ERD tables, or a synthetic
V1 provider) into the canonical schemas defined in
`recommendation.data.schemas`. Recommendation models depend only on the
canonical schemas, never on adapter internals, so swapping a synthetic
adapter for a real backend adapter later requires no model changes.

Implemented starting Phase 2:
  PurchaseAdapter, CartAdapter, UserAdapter, ReviewAdapter (backed by the
  real ERD entities: Order/OrderItem, Cart/CartItem, User, Review), and
  ClickAdapter / SearchAdapter / ChatbotContextAdapter (synthetic-only in
  V1, since no backend table for these exists yet - see
  docs/data-mapping.md section 4).

`user_events_adapter.UserEventsAdapter` is the confirmed future real
implementation of ClickAdapter/PurchaseAdapter/CartAdapter/SearchAdapter/
ChatbotContextAdapter, all backed by ONE unified `User_events`
activity-log table instead of five separate ones - see that module's
docstring for the full mapping.
"""
