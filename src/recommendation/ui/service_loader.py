"""Loads the `RecommendationService` once per Streamlit session and
reuses it across reruns (`st.cache_resource`) - the dashboard shares the
exact same Two-Tower/ranker/VectorIndex artifacts and the exact same
`serving.pipeline.recommend` code path the Phase 8 API serves, in-process
(no HTTP hop to a separately-running API server) so the dashboard is a
single `streamlit run` away, with no other service to start first.

`load_service()` checks `st.session_state["_service_override"]` first -
the dependency-injection seam `tests/test_ui_dashboard.py` uses (via
Streamlit's `AppTest`) to inject a small/fake service and skip the real
model-loading startup path entirely, the same pattern `api.app
.create_app(service=...)` uses for the API's own tests. Setting the
override to an `Exception` instance instead of a service makes
`load_service()` raise it - how tests exercise `dashboard.py`'s "service
failed to load" error state deterministically, without needing an actual
missing-model-artifacts environment to reproduce the failure.
"""

from __future__ import annotations

import streamlit as st

from recommendation.api.dependencies import RecommendationService, build_recommendation_service
from recommendation.utils.config import get_config


@st.cache_resource(show_spinner="Loading recommendation service (Two-Tower, ranker, VectorIndex)...")
def _load_service_cached() -> RecommendationService:
    return build_recommendation_service(get_config())


def load_service() -> RecommendationService:
    override = st.session_state.get("_service_override")
    if isinstance(override, BaseException):
        raise override
    if override is not None:
        return override
    return _load_service_cached()
