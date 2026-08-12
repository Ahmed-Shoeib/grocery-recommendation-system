"""Shared inference/service layer used by both the API and the dashboard.

Loads trained model artifacts (Two-Tower towers, VectorIndex, ranker) and
exposes a single `recommend(user_id, limit, context=None)` entrypoint so
recommendation logic lives in exactly one place. `context` is an
extensibility hook for future recommendation surfaces (home, product
detail, cart, search) - unused by any V1 model, but present so the API
doesn't need a breaking change to add surface-aware behavior later.
"""
