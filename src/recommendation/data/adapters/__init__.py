"""Data adapter interfaces and implementations.

Adapters translate a data source (real backend ERD tables, or a synthetic
V1 provider) into the canonical schemas defined in
`recommendation.data.schemas`. Recommendation models depend only on the
canonical schemas, never on adapter internals, so swapping a synthetic
adapter for a real backend adapter later requires no model changes.

Implemented starting Phase 2:
  PurchaseAdapter, CartAdapter, UserAdapter, ReviewAdapter (backed by the
  real ERD entities: Order/OrderItem, Cart/CartItem, User, Review), and
  SearchAdapter / ChatbotContextAdapter (synthetic-only in V1, since the
  ERD has no corresponding tables yet).
"""
