"""FastAPI app factory (Phase 8).

`create_app(service=None)` is the dependency-injection seam: pass a
pre-built `RecommendationService` (a real one with small/fake artifacts,
or a stub that raises to simulate failure) to get a fully wired app
without the expensive real-model-loading startup path - what
`tests/test_api.py` does. Passing nothing (production/`scripts
/run_api.py`) makes the `lifespan` context build the real service once
at startup via `build_recommendation_service`, stored on `app.state
.service` and never rebuilt per-request.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from recommendation.api.dependencies import RecommendationService, build_recommendation_service
from recommendation.api.routes import router
from recommendation.api.schemas import ErrorResponse
from recommendation.utils.config import get_config
from recommendation.utils.logging import get_logger

logger = get_logger(__name__)

_ERROR_CODES = {
    404: "not_found",
    409: "offline_report_unavailable",
    422: "invalid_request",
    503: "not_ready",
}


def create_app(service: RecommendationService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service if service is not None else build_recommendation_service(get_config())
        yield
        app.state.service = None

    app = FastAPI(title="Grocery Recommendation API", version="1.0.0", lifespan=lifespan)
    app.include_router(router)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request method=%s path=%s status=%d duration_ms=%.2f",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        error_code = _ERROR_CODES.get(exc.status_code, "error")
        return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=error_code, message=str(exc.detail)).model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Same ErrorResponse contract as manually-raised HTTPExceptions (e.g.
        # the max_recommendation_count check) - a client shouldn't see two
        # different error body shapes for "limit is invalid" depending on
        # which validation path caught it.
        first = exc.errors()[0] if exc.errors() else {}
        message = first.get("msg", "invalid request parameters")
        field = ".".join(str(p) for p in first.get("loc", []) if p != "query")
        return JSONResponse(status_code=422, content=ErrorResponse(error="invalid_request", message=f"{field}: {message}" if field else message).model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error while handling path=%s", request.url.path)
        return JSONResponse(status_code=500, content=ErrorResponse(error="internal_error", message="An unexpected error occurred").model_dump())

    return app
