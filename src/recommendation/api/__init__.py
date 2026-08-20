"""FastAPI recommendation service (Phase 8). Thin layer over
`recommendation.serving` - no recommendation logic lives here.

`app.create_app` builds the FastAPI application; `dependencies
.RecommendationService`/`build_recommendation_service` load the Two-
Tower/ranker/VectorIndex artifacts ONCE and inject them into every
request, never rebuilding per-request. `schemas.py` is the versioned
wire contract, kept separate from internal domain/model schemas.
`routes.py` implements `/v1/health`, `/v1/ready`,
`/v1/users/{user_id}/recommendations`, and the STEP 9 read-only
`/v1/users`, `/v1/users/{user_id}/profile`, `/v1/metrics/offline`
endpoints - all deferring to `serving.pipeline.recommend`/`ui.data_access`/
`evaluation.offline_report` for the actual work, never duplicating it
here. `/v1/metrics/offline` is a cheap read of a report PERSISTED by
`scripts/generate_offline_report.py` (STEP 9 follow-up fix -
docs/data-mapping.md section 18.1); it does not call `ui.metrics`
(legacy, no longer wired into any live route - see that module's
docstring).

STEP 9 (docs/data-mapping.md section 18): deliberately does NOT
re-export `create_app` (`from recommendation.api.app import create_app`)
the way earlier phases did - that eager import pulls in `api.dependencies`
(TensorFlow, the Two-Tower/ranker/adapters stack) as a side effect of
importing this package's `__init__.py`, which would run on ANY import
under `recommendation.api.*` including `api.schemas` - the pure-Pydantic
wire contract `ui.api_client` (the Streamlit process) needs. Since
Python always executes a parent package's `__init__.py` before any of
its submodules, that side effect could not be scoped any other way;
removing the re-export is what actually lets `ui.api_client` import
`api.schemas` without dragging model/training code into the Streamlit
process (verified by
`tests/test_ui_api_client.py::test_api_client_module_does_not_import_recommendation_service_or_model_code`,
which failed before this fix). Callers that need `create_app` (`api.app`
itself, `tests/test_api.py`, ASGI entry points) already import it
directly from `recommendation.api.app` - grepped with zero call sites
relying on the old `recommendation.api.create_app` re-export.
"""

from __future__ import annotations
