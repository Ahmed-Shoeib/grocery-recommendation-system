"""Feature engineering: EngagementProfile -> user/product feature vectors.

`product_features.py` and `user_features.py` consume only the canonical
schemas (`recommendation.data.schemas`); `pipeline.py` is the one place
that also depends on `recommendation.data.adapters.base.AdapterBundle` (an
interface type, not a concrete synthetic/backend implementation) to wire
everything together end to end.
"""
