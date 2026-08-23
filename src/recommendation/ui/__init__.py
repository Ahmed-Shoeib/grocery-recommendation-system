"""Internal Streamlit recommendation dashboard. A demonstration/debug
tool for inspecting the recommendation engine, not the storefront.

`dashboard.py` is a thin rendering layer only. All data access/formatting
lives in `data_access.py`/`metrics.py` (plain functions, no Streamlit
import, unit-testable in isolation); recommendations themselves always
come from `serving.pipeline.generate_recommendations` via the shared
`RecommendationService` (`service_loader.py`, reusing `api.dependencies`
in-process) - no recommendation logic is duplicated or reimplemented
here.
"""
