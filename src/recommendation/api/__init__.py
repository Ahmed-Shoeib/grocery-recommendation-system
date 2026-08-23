"""FastAPI recommendation service. Thin layer over `recommendation
.serving` - no recommendation logic lives here: `app.create_app` builds
the FastAPI application; `dependencies.RecommendationService`/
`build_recommendation_service` load the Two-Tower/ranker/VectorIndex
artifacts ONCE and inject them into every request, never rebuilding
per-request. `schemas.py` is the versioned wire contract, kept separate
from internal domain/model schemas. `routes.py` implements
`/v1/health`, `/v1/ready`, `/v1/users/{user_id}/recommendations`, and
the read-only `/v1/users`, `/v1/users/{user_id}/profile`,
`/v1/metrics/offline` endpoints - all deferring to `serving.pipeline
.recommend`/`ui.data_access`/`evaluation.offline_report` for the actual
work rather than duplicating it here. `/v1/metrics/offline` is a cheap
read of a report PERSISTED by `scripts/generate_offline_report.py`; it
does not call `ui.metrics` (legacy, no longer wired into any live route
- see that module's docstring).

Deliberately does NOT re-export `create_app`
(`from recommendation.api.app import create_app`) here: since Python
always executes a parent package's `__init__.py` before any of its
submodules, that eager import would pull `api.dependencies` (TensorFlow,
the Two-Tower/ranker/adapters stack) into every import under
`recommendation.api.*`, including `api.schemas` - the pure-Pydantic wire
contract `ui.api_client` (the Streamlit process) needs to import without
dragging in model/training code (verified by
`tests/test_ui_api_client.py::test_api_client_module_does_not_import_recommendation_service_or_model_code`).
Callers that need `create_app` (`api.app` itself, `tests/test_api.py`,
ASGI entry points) import it directly from `recommendation.api.app`.
"""

from __future__ import annotations
