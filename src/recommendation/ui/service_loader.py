"""Constructs the single `RecommendationApiClient` the dashboard uses,
cached once per Streamlit session (`st.cache_resource`) - STEP 9
(docs/data-mapping.md section 18) replaced the in-process
`RecommendationService` this module used to build with a thin HTTP
client; FastAPI is now the only process that loads model artifacts or
touches SQLite/synthetic adapters.

`load_api_client()` checks `st.session_state["_api_client_override"]`
first - the dependency-injection seam `tests/test_ui_dashboard.py` uses
(via Streamlit's `AppTest`) to inject a fake client, so dashboard tests
never make a real HTTP call or require a running FastAPI process. This is
the same override pattern `api.app.create_app(service=...)` uses for the
API's own tests, and the one this module itself used pre-STEP-9 for
`RecommendationService` injection.
"""

from __future__ import annotations

import streamlit as st

from recommendation.ui.api_client import RecommendationApiClient
from recommendation.utils.config import get_config


@st.cache_resource(show_spinner=False)
def _load_api_client_cached() -> RecommendationApiClient:
    config = get_config()
    return RecommendationApiClient(config.dashboard.api_base_url, timeout_seconds=config.dashboard.request_timeout_seconds)


def load_api_client() -> RecommendationApiClient:
    override = st.session_state.get("_api_client_override")
    if override is not None:
        return override
    return _load_api_client_cached()
