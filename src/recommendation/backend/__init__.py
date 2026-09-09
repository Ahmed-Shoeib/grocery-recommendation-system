"""Real-backend REST integration - the third `AdapterBundle` producer.

There is NO direct database access to the production backend. The
recommendation service obtains catalog, user, and engagement data through
the backend's HTTP API (Swagger: ``/swagger/index.html``). This package is
the ONLY place in the codebase that knows HTTP / the backend's JSON wire
shapes:

    backend REST API
        -> recommendation.backend.auth      (service token: POST /api/auth/service/token)
        -> recommendation.backend.client   (HTTP, pagination, retries, TLS, Bearer)
        -> recommendation.backend.dtos      (external response models)
        -> recommendation.backend.loader    (DTO -> Raw* / UserInteraction,
                                                   via ExternalIdentityResolver)
        -> recommendation.adapters.backend_factory.build_backend_api_adapters
        -> AdapterBundle   (identical interface to the synthetic / SQLite paths)
        -> EngagementProfile -> feature engineering -> Two-Tower -> ranker -> serving

Everything downstream of `build_backend_api_adapters` is unchanged and
never sees a slug, a GUID, an HTTP status code, or a backend field name -
the canonical schemas remain the stability boundary (docs/data-mapping.md
section 19).

Identity: the backend exposes products/categories by SLUG and users by
GUID, with no numeric ids. `recommendation.backend.identity
.ExternalIdentityResolver` maps each external key to a stable, persistent
internal `int` so the recommender core and the trained model artifacts
keep operating on the canonical integer-id contract they were built
against. The one exception is `/api/reviews`, which addresses rows by the
backend's own int32 primary keys - a key space nothing else exposes, so
those rows cannot be joined yet (docs/data-mapping.md section 19.6).

Auth: `/api/products`, `/api/categories` and `/api/user-activities` are
public. `/api/users/{guid}` and `/api/reviews` are Bearer-gated and use
`recommendation.backend.auth.ServiceTokenProvider`, whose
credentials come from the environment and whose token is cached in memory
only - never persisted, never logged, never attached to a public request.
"""
