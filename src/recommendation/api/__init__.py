"""FastAPI recommendation service (Phase 8). Thin layer over
`recommendation.serving` - no recommendation logic lives here.

`app.create_app` builds the FastAPI application; `dependencies
.RecommendationService`/`build_recommendation_service` load the Two-
Tower/ranker/VectorIndex artifacts ONCE and inject them into every
request, never rebuilding per-request. `schemas.py` is the versioned
wire contract, kept separate from internal domain/model schemas.
`routes.py` implements `/v1/health`, `/v1/ready`, and
`/v1/users/{user_id}/recommendations`, always deferring to `serving
.pipeline.recommend` (Two-Tower -> VectorIndex -> Ranker -> Re-ranking
-> Business Rules/Eligibility -> Final Top-N) - business-rule filtering
stays exactly where Phase 7 put it, at the end.
"""

from __future__ import annotations

from recommendation.api.app import create_app

__all__ = ["create_app"]
